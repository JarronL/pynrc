# Makes print and division act like Python 3
from __future__ import print_function, division

# Import the usual libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from copy import deepcopy

from .nrc_utils import S, stellar_spectrum, jupiter_spec, bp_2mass
from .nrc_utils import dist_image, pad_or_cut_to_size
from .nrc_utils import read_filter, channel_select, coron_ap_locs
from .obs_nircam import model_to_hdulist, obs_hci
from .obs_nircam import plot_contrasts, plot_contrasts_mjup, planet_mags, plot_planet_patches

import logging
_log = logging.getLogger('nb_funcs')

import pynrc
pynrc.setup_logging('WARNING', verbose=False)

"""
Common functions for notebook simulations and plotting.
This is my attempt to standardize these routines over
all the various GTO programs.
"""


# Observation Definitions
# Functions to create and optimize a series of observation objects stored as a dictionary.
bp_k = bp_2mass('k')

def make_key(filter, pupil=None, mask=None):
    """Create identification key (string) based on filter, pupil, and mask"""
    mask_key = 'none' if mask is None else mask
    pupil_key = 'none' if pupil is None else pupil
    key = '{}_{}_{}'.format(filter,mask_key,pupil_key)
    return key


def model_info(source, filt, dist):
    
    base_dir  = '/Volumes/NIRData/Andras_models_v2/'
    model_dir = base_dir + source + '/'
    
    # Match filters with model
    filt_switch = {'F182M':'F210M', 'F210M':'F210M', 'F250M':'F250M',
                   'F300M':'F300M', 'F335M':'F335M', 'F444W':'F444W'}
    filt_model = filt_switch.get(filt, filt)
    fname = source + '_' + filt_model +'sc.fits'

    bp = read_filter(filt_model)
    w0 = bp.avgwave() / 1e4

    # Model pixels are 4x oversampled
    detscale = (channel_select(bp))[0]
    model_scale = detscale / 4.
    
    # File name, arcsec/pix, dist (pc), wavelength (um), flux units
    args_model = (model_dir+fname, model_scale, dist, w0, 'Jy/pixel')

    return args_model


def obs_wfe(wfe_drift, filt_list, sp_sci, dist, sp_ref=None, args_disk=None, 
            wind_mode='WINDOW', subsize=None, fov_pix=None, verbose=False, narrow=False):
    """
    For a given WFE drift and series of filters, create a list of 
    NIRCam observations.
    """
    
    if sp_ref is None: sp_ref = sp_sci

    obs_dict = {}
    for filt, mask, pupil in filt_list:
        # Create identification key
        key = make_key(filt, mask=mask, pupil=pupil)
        print(key)

        # Disk Model
        if args_disk is None:
            hdu_disk = None
        elif 'auto' in args_disk:
            # Convert to photons/sec in specified filter
            args_disk = model_info(sp_sci.name, filt, dist)
            hdu_disk = model_to_hdulist(args_disk, sp_sci, filt, pupil=pupil, mask=mask)
        else:
            hdu_disk = model_to_hdulist(args_disk, sp_sci, filt, pupil=pupil, mask=mask)
            
        fov_pix_orig = fov_pix
        # Define the subarray readout size
        if 'FULL' in wind_mode: # Full frame
            subuse = 2048
            
            # Define PSF pixel size defaults
            if mask is None: 
                fov_pix = 400 if fov_pix is None else fov_pix
            elif ('210R' in mask) or ('SWB' in mask): 
                fov_pix = 640 if fov_pix is None else fov_pix
            else:
                fov_pix = 320 if fov_pix is None else fov_pix
                
        elif subsize is None: # Window Mode defaults
            if mask is None: # Direct Imaging
                subuse = 400
            elif ('210R' in mask) or ('SWB' in mask): # SW Coronagraphy
                subuse = 640
            else: # LW Coronagraphy
                subuse = 320
        else: # No effect if full frame
            subuse = subsize
                
        # Define PSF pixel size
        fov_pix = subuse if fov_pix is None else fov_pix

        # Make sure fov_pix is odd for direct imaging
        if (mask is None) and (np.mod(fov_pix,2)==0):
            fov_pix += 1
        # Other coronagraph vs direct imaging settings
        module, oversample = ('B', 4) if mask is None else ('A', 2)
        
        if narrow and ('SWB' in mask):
            bar_offset=-8
        elif narrow and ('LWB' in mask):
            bar_offset=8
        else:
            bar_offset=None
        
        # Initialize and store the observation
        # A reference observation is stored inside each parent obs_hci class.
        obs_dict[key] = obs_hci(sp_sci, sp_ref, dist, filter=filt, mask=mask, pupil=pupil, 
                                wfe_ref_drift=wfe_drift, fov_pix=fov_pix, oversample=oversample, 
                                wind_mode=wind_mode, xpix=subuse, ypix=subuse,
                                disk_hdu=hdu_disk, verbose=verbose, bar_offset=bar_offset)
        fov_pix = fov_pix_orig
        
    return obs_dict


def obs_optimize(obs_dict, sp_opt=None, well_levels=None, tacq_max=1800, **kwargs):
    """
    Perform ramp optimization on each science and reference observation
    in a list of filter observations. Updates the detector MULTIACCUM
    settings for each observation in the dictionary.
    
    snr_goal = 5
    snr_frac = 0.02
    tacq_max = 1400
    tacq_frac = 0.01
    nint_min = 15
    ng_max = 10
    """

    # A very faint bg object on which to maximize S/N
    # If sp_opt is not set, then default to a 20th magnitude flat source
    if sp_opt is None:
        sp_opt = stellar_spectrum('flat', 20, 'vegamag', bp_k)

    
    # Some observations may saturate, so define a list of  maximum well level
    # values that we will incrementally check until a ramp setting is found
    # that meets the contraints.
    if well_levels is None:
        well_levels = [0.8, 1.5, 3.0, 5.0, 10.0, 20.0, 100.0, 150.0, 300.0, 500.0]
   
    filt_keys = list(obs_dict.keys())
    filt_keys.sort()
    print(['Pattern', 'NGRP', 'NINT', 't_int', 't_exp', 't_acq', 'SNR', 'Well', 'eff'])
    for j, key in enumerate(filt_keys):
        print('')
        print(key)

        obs = obs_dict[key]
        sp_sci, sp_ref = (obs.sp_sci, obs.sp_ref)
        
        # SW filter piggy-back on two LW filters, so 2 x tacq
        is_SW = obs.bandpass.avgwave()/1e4 < 2.5

        sci = obs
        ref = sci.nrc_ref

        # Ramp optimization for both science and reference targets
        for obs2, sp in zip([sci, ref], [sp_sci, sp_ref]):
            i = nrow = 0
            while nrow==0:
                well_max = well_levels[i]
                tbl = obs2.ramp_optimize(sp_opt, sp, well_frac_max=well_max, tacq_max=tacq_max, **kwargs)
                nrow = len(tbl)
                i+=1
                
            # Grab the highest ranked MULTIACCUM settings and update the detector readout
            v1, v2, v3 = tbl['Pattern', 'NGRP', 'NINT'][0]
            
            vals = list(tbl[0])#.as_void()
            strout = '{:8} {} {}'.format(vals[0], vals[1], vals[2])
            for v in vals[3:]:
                strout = strout + ', {:.4f}'.format(v)
            print(strout)

            # SW filter piggy-back on two LW filters, so 2 x tacq
            is_SW = obs.bandpass.avgwave()/1e4 < 2.5
            if is_SW: v3 *= 2
            
            # Coronagraphic observations have two roll positions, so cut NINT by 2
            if obs.mask is not None: v3 = int(v3/2) 
            obs2.update_detectors(read_mode=v1, ngroup=v2, nint=v3)
        


###########################################
# Functions to run a series of operations
###########################################

# Optimize observations
def do_opt(obs_dict, tacq_max=1800, **kwargs):
    sp_opt = stellar_spectrum('flat', 20, 'vegamag', bp_k)
    obs_optimize(obs_dict, sp_opt=sp_opt, tacq_max=tacq_max, **kwargs)


# For each filter setting, generate a series of contrast curves at different WFE values
def do_contrast(obs_dict, wfe_list, filt_keys, nsig=5, roll_angle=10, verbose=True, **kwargs):
    """
    kwargs to pass to calc_contrast() and their defaults:

    no_ref = False
    func_std = robust.medabsdev
    exclude_disk = True
    exclude_planets = True
    exclude_noise = False
    opt_diff = True
    fix_sat = False
    ref_scale_all = False
    """
    contrast_all = {}
    for i, key in enumerate(filt_keys):
        if verbose: print(key)
        obs = obs_dict[key]

        wfe_roll_temp = obs.wfe_roll_drift
        wfe_ref_temp  = obs.wfe_ref_drift

        # Stores tuple of (Radial Distances, Contrast, and Sensitivity) for each WFE drift
        curves = []
        for wfe_drift in wfe_list:
            
            if ('no_ref' in list(kwargs.keys())) and (kwargs['no_ref']==True):
                obs.wfe_roll_drift = wfe_drift
            else:
                obs.wfe_ref_drift = wfe_drift
            result = obs.calc_contrast(roll_angle=roll_angle, nsig=nsig, **kwargs)
            curves.append(result)
            
        obs.wfe_roll_drift = wfe_roll_temp
        obs.wfe_ref_drift = wfe_ref_temp
            
        contrast_all[key] = curves
    return contrast_all


def do_gen_hdus(obs_dict, filt_keys, wfe_ref_drift, verbose=True, **kwargs):
    
    """
    kwargs to pass to gen_roll_image() and their defaults:
    
    PA1 = 0
    PA2 = 10
    zfact         = None
    oversample    = None
    exclude_disk  = False
    exclude_noise = False
    opt_diff      = True
    use_cmask     = False
    """
    
    hdulist_dict = {}
    for key in filt_keys:
        if verbose: print(key)
        obs = obs_dict[key]
        obs.wfe_ref_drift = wfe_ref_drift
        hdulist = obs.gen_roll_image(**kwargs)
        
        hdulist_dict[key] = hdulist
        
    return hdulist_dict

def do_sat_levels(obs, satval=0.95, ng_min=2, ng_max=None, verbose=True, 
                  plot=True, xylim=2.5, return_fig_axes=False):
    
    ng_max = obs.det_info['ngroup'] if ng_max is None else ng_max
    
    # Well level of each pixel for science source
    image = obs.gen_slope_image(exclude_noise=True, use_cmask=True, quick_PSF=True)
    sci_levels1 = obs.saturation_levels(ngroup=ng_min, image=image)
    sci_levels2 = obs.saturation_levels(ngroup=ng_max, image=image)

    # Well level of each pixel for reference source
    image = obs.gen_slope_image(exclude_noise=True, use_cmask=True, quick_PSF=True, do_ref=True)
    ref_levels1 = obs.saturation_levels(ngroup=ng_min, image=image, do_ref=True)
    ref_levels2 = obs.saturation_levels(ngroup=ng_max, image=image, do_ref=True)
    
    # Which pixels are saturated?
    sci_mask1 = sci_levels1 > satval
    sci_mask2 = sci_levels2 > satval

    # Which pixels are saturated?
    ref_mask1 = ref_levels1 > satval
    ref_mask2 = ref_levels2 > satval

    # How many saturated pixels?
    nsat1_sci = len(sci_levels1[sci_mask1])
    nsat2_sci = len(sci_levels2[sci_mask2])

    # How many saturated pixels?
    nsat1_ref = len(ref_levels1[ref_mask1])
    nsat2_ref = len(ref_levels2[ref_mask2])

    if verbose:
        print(obs.sp_sci.name)
        print('{} saturated pixel at NGROUP=2'.format(nsat1_sci))
        print('{} saturated pixel at NGROUP={}'.format(nsat2_sci,ng_max))
        print('')
        print(obs.sp_ref.name)
        print('{} saturated pixel at NGROUP=2'.format(nsat1_ref))
        print('{} saturated pixel at NGROUP={}'.format(nsat2_ref,ng_max))
        
    if nsat2_sci==nsat2_ref==0:
        plot=False
        if verbose:
            print('')
            print('No saturation detected.')

    if plot:
        fig, axes_all = plt.subplots(2,2, figsize=(8,8))

        xlim = ylim = np.array([-1,1])*xylim

        
        # Plot science source
        nsat1, nsat2 = (nsat1_sci, nsat2_sci)
        sat_mask1, sat_mask2 = (sci_mask1, sci_mask2)
        sp = obs.sp_sci
        nrc = obs

        xpix, ypix = (nrc.det_info['xpix'], nrc.det_info['ypix'])
        bar_offpix = nrc.bar_offset / nrc.pixelscale
        if ('FULL' in nrc.det_info['wind_mode']) and (nrc.mask is not None):
            cdict = coron_ap_locs(nrc.module, nrc.channel, nrc.mask, full=True)
            xcen, ycen = cdict['cen_V23']
            xcen += bar_offpix
        else:
            xcen, ycen = (xpix/2 + bar_offpix, ypix/2)
        delx, dely = (xcen - xpix/2, ycen - ypix/2)
        
        extent_pix = np.array([-xpix/2-delx,xpix/2-delx,-ypix/2-dely,ypix/2-dely])
        extent = extent_pix * nrc.pix_scale

        axes = axes_all[0]
        axes[0].imshow(sat_mask1, extent=extent)
        axes[1].imshow(sat_mask2, extent=extent)

        axes[0].set_title('{} Saturation (NGROUP=2)'.format(sp.name))
        axes[1].set_title('{} Saturation (NGROUP={})'.format(sp.name,ng_max))

        for ax in axes:
            ax.set_xlabel('Arcsec')
            ax.set_ylabel('Arcsec')

            ax.tick_params(axis='both', color='white', which='both')
            for k in ax.spines.keys():
                ax.spines[k].set_color('white')

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        # Plot ref source sat mask
        nsat1, nsat2 = (nsat1_ref, nsat2_ref)
        sat_mask1, sat_mask2 = (ref_mask1, ref_mask2)
        sp = obs.sp_ref
        nrc = obs.nrc_ref

        axes = axes_all[1]
        axes[0].imshow(sat_mask1, extent=extent)
        axes[1].imshow(sat_mask2, extent=extent)

        axes[0].set_title('{} Saturation (NGROUP=2)'.format(sp.name))
        axes[1].set_title('{} Saturation (NGROUP={})'.format(sp.name,ng_max))

        for ax in axes:
            ax.set_xlabel('Arcsec')
            ax.set_ylabel('Arcsec')

            ax.tick_params(axis='both', color='white', which='both')
            for k in ax.spines.keys():
                ax.spines[k].set_color('white')

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        fig.tight_layout()
        
    #masks = [[sci_mask1,sci_mask2], [ref_mask1,ref_mask2]]
        
    # Return saturation radius
    if nsat1_sci == nsat1_ref == 0:
        sat_rad = 0
    else:
        sat_mask = sci_mask1 if nsat1_sci > nsat1_ref else ref_mask1
        rho = dist_image(sat_mask, center=(xcen,ycen))
        sat_rad = rho[sat_mask].max() * obs.pixelscale
        
    if return_fig_axes and plot:
        return (fig, axes), sat_rad
    else:
        return sat_rad
    
###########################################
# Simulated Data
###########################################

def average_slopes(hdulist):
    """
    For a series of ramps, calculate the slope images then average together.
    """
    
    ramps = hdulist[1].data
    header = hdulist[0].header

    slopes_fin = []
    for i in range(len(ramps)):
        data = ramps[i]

        # Create time array
        ng, ypix, xpix = data.shape
        tvals = (np.arange(ng)+1) * header['TGROUP']

        # Flatten image space to 1D
        data = data.reshape([ng,-1])

        # Make saturation mask
        sat_val = 0.95*data.max()
        sat_mask = data > sat_val

        # Create slope images
        # Cycle through groups using only unsaturated pixels 
        im_slope = np.zeros_like(data[0]) - 10
        for i in np.arange(1,ng)[::-1]:
            ind = (im_slope==-10) & (~sat_mask[i])
            if np.any(ind): # Check if any pixels are still True
                im_slope[ind] = pynrc.fast_poly.jl_poly_fit(tvals, data[:,ind])[1]
            #print(im_slope[ind].shape)

        # Special case of only first frame unsaturated
        ind = (im_slope==-10) & (~sat_mask[0])
        im_slope[ind] = data[:,ind] / tvals[0]
        #print(im_slope[ind].shape)

        # If saturated on first frame, set to NaN
        ind = sat_mask[0]
        im_slope[ind] = np.nan
        #print(im_slope[ind].shape)

        data = data.reshape([ng,ypix,xpix])
        im_slope = im_slope.reshape([ypix,xpix])
        slopes_fin.append(im_slope)

    # Average slopes together
    # us nanmean() to ignore those with NaNs
    slopes_fin = np.array(slopes_fin)
    slope_final = np.nanmean(slopes_fin, axis=0)
    
    return slope_final
    
###########################################
# Plotting images and contrast curves
###########################################

def do_plot_contrasts(curves_ref, curves_roll, nsig, wfe_list, obs, age, age2=None, 
    sat_rad=0, jup_mag=True, xr=[0,10], xr2=[0,10], yscale2='symlog',
    save_fig=False, outdir='', return_fig_axes=False):
    """
    Plot series of contrast curves.
    """
    
    if (curves_ref is None) and (curves_roll is None):
        _log.warning('Both curves set no none. Returning...')
        return
    
    lin_vals = np.linspace(0.2,0.8,len(wfe_list))
    c1 = plt.cm.Blues_r(lin_vals)
    c2 = plt.cm.Reds_r(lin_vals)
    c3 = plt.cm.Purples_r(lin_vals)
    c4 = plt.cm.Greens_r(lin_vals)

    fig, axes = plt.subplots(1,2, figsize=(14,4.5))

    ax = axes[0]
    if curves_ref is not None:
        plot_contrasts(curves_ref, nsig, wfe_list, obs=obs, ax=ax, colors=c1, xr=xr)
    if curves_roll is not None:
        obs_kw = None if curves_ref is not None else obs
        plot_contrasts(curves_roll, nsig, wfe_list, obs=obs_kw, ax=ax, colors=c2, xr=xr)
    #plot_planet_patches(ax, obs, age=age, av_vals=None, cond=True)

    ax.set_ylim([22,8])

    # Legend organization
    nwfe = len(wfe_list)
    if curves_ref is None:
        ax.legend(loc='upper right', title='Roll Sub')
    elif curves_roll is None:
        ax.legend(loc='upper right', title='Ref Sub')
    else:
        handles, labels = ax.get_legend_handles_labels()
        h1 = handles[0:nwfe][::-1]
        h2 = handles[nwfe:][::-1]
        h1_t = [mpatches.Patch(color='none', label='Ref Sub')]
        h2_t = [mpatches.Patch(color='none', label='Roll Sub')]
        handles_new = h1_t + h1 + h2_t + h2
        ax.legend(ncol=2, handles=handles_new, loc='upper right')
    
    # Magnitude of Jupiter at object's distance
    if jup_mag:
        jspec = jupiter_spec(dist=obs.distance)
        jobs = S.Observation(jspec, obs.bandpass, binset=obs.bandpass.wave)
        jmag = jobs.effstim('vegamag')
        if jmag<np.max(ax.get_ylim()):
            ax.plot(xr, [jmag,jmag], color='C2', ls='--')
            txt = 'Jupiter at {:.1f} pc'.format(obs.distance)
            ax.text(xr[0]+0.02*(xr[1]-xr[0]), jmag, txt, horizontalalignment='left', verticalalignment='bottom')

    # Plot in terms of Jupiter Masses
    ax = axes[1]
    age1 = age
    if curves_ref is not None:
        plot_contrasts_mjup(curves_ref, nsig, wfe_list, obs=obs, age=age1, ax=ax, colors=c1, xr=xr2, twin_ax=True, yr=None)
    if curves_roll is not None:
        twin_kw = False if curves_ref is not None else True
        plot_contrasts_mjup(curves_roll, nsig, wfe_list, obs=obs, age=age1, ax=ax, colors=c2, xr=xr2, twin_ax=twin_kw, yr=None)
    if age2 is not None:
        if curves_ref is not None:
            plot_contrasts_mjup(curves_ref, nsig, wfe_list, obs=obs, age=age2, ax=ax, colors=c3, xr=xr2, yr=None)
        if curves_roll is not None:
            plot_contrasts_mjup(curves_roll, nsig, wfe_list, obs=obs, age=age2, ax=ax, colors=c4, xr=xr2, yr=None)

        # Legend organization
        handles, labels = ax.get_legend_handles_labels()
        if curves_ref is None:
            handles_new = [handles[i*nwfe] for i in range(2)]
            labels_new = ['Roll Sub ({:.0f} Myr)'.format(age1),
                          'Roll Sub ({:.0f} Myr)'.format(age2)
                          ]
        elif curves_roll is None:
            handles_new = [handles[i*nwfe] for i in range(2)]
            labels_new = ['Ref Sub ({:.0f} Myr)'.format(age1),
                          'Ref Sub ({:.0f} Myr)'.format(age2)
                          ]
        else:
            handles_new = [handles[i*nwfe] for i in range(4)]
            labels_new = ['Ref Sub ({:.0f} Myr)'.format(age1),
                          'Roll Sub ({:.0f} Myr)'.format(age1),
                          'Ref Sub ({:.0f} Myr)'.format(age2),
                          'Roll Sub ({:.0f} Myr)'.format(age2),
                         ]
    else:
        handles, labels = ax.get_legend_handles_labels()
        if curves_ref is None:
            handles_new = [handles[0]]
            labels_new = ['Roll Sub ({:.0f} Myr)'.format(age1)]
        elif curves_roll is None:
            handles_new = [handles[0]]
            labels_new = ['Ref Sub ({:.0f} Myr)'.format(age1)]
       
        else:
            handles_new = [handles[i*nwfe] for i in range(2)]
            labels_new = ['Ref Sub ({:.0f} Myr)'.format(age1),
                          'Roll Sub ({:.0f} Myr)'.format(age1),
                         ]

    ax.legend(handles=handles_new, labels=labels_new, loc='upper right', title='COND Models')
        
    # Some fancy log+linear plotting
    from matplotlib.ticker import FixedLocator, ScalarFormatter
    if yscale2=='symlog':
        ax.set_ylim([0,100])
        yr = ax.get_ylim()
        ax.set_yscale('symlog', linthreshy=10, linscaley=2)
        ax.set_yticks(list(range(0,10)) + [10,100,1000])
        #ax.get_yaxis().set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())

        minor_log = list(np.arange(20,100,10)) + list(np.arange(200,1000,100))
        minorLocator = FixedLocator(minor_log)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.set_ylim([0,yr[1]])
    elif yscale2=='log':
        ax.set_yscale('log')
        ax.set_ylim([0.1,100])
        ax.yaxis.set_major_formatter(ScalarFormatter())

    # Saturation regions
    if sat_rad > 0:
        sat_rad_asec = sat_rad
        for ax in axes:
            ylim = ax.get_ylim()
            rect = mpatches.Rectangle((0, ylim[0]), sat_rad, ylim[1]-ylim[0], alpha=0.2, color='k', zorder=2)
            ax.add_patch(rect)

    name_sci = obs.sp_sci.name
    name_ref = obs.sp_ref.name
    title_str = '{} (dist = {:.1f} pc; PSF Ref: {}) -- {} Contrast Curves'\
        .format(name_sci, obs.distance, name_ref, obs.filter)
    fig.suptitle(title_str, fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85, bottom=0.1 , left=0.05, right=0.97)

    fname = "{}_contrast_{}.pdf".format(name_sci.replace(" ", ""), obs.mask)
    if save_fig: 
        fig.savefig(outdir+fname)
        
    if return_fig_axes:
        return fig, axes
        
        
def plot_images(obs_dict, hdu_dict, filt_keys, wfe_drift, fov=10, 
                save_fig=False, outdir='', return_fig_axes=False):
                
                
    nfilt = len(filt_keys)
    ext_name = ['Model', 'Sim Image (linear scale)', 'Sim Image ($r^2$ scale)']
    nim = len(ext_name)

    fig, axes = plt.subplots(nfilt, nim, figsize=(8.5,6.5))
    #axes = axes.transpose()
    for j, k in enumerate(filt_keys):
        obs = obs_dict[k]
        hdu_mod = obs.disk_hdulist
        if hdu_mod is None:
            raise ValueError('Disk model image is None. Did you forget to add the disk image?')
        hdu_sim = hdu_dict[k]
        data  = hdu_sim[0].data
        data -= np.nanmedian(data)

        # Make r^2 scaled version of data
        hdu_sim_r2 = deepcopy(hdu_sim)
        data  = hdu_sim_r2[0].data
        data -= np.nanmedian(data)
        header = hdu_sim_r2[0].header
        rho = dist_image(data, pixscale=header['PIXELSCL'])
        data *= rho**2

        # Max value for model
        data_mod   = hdu_mod[0].data
        header_mod = hdu_mod[0].header
        rho_mod    = dist_image(data_mod, pixscale=header_mod['PIXELSCL'])
        data_mod_r2 = data_mod*rho_mod**2
        vmax  = np.max(data_mod)
        vmax2 = np.max(data_mod_r2)

        # Scale value for data
        im_temp = pad_or_cut_to_size(data_mod, hdu_sim[0].data.shape)
        mask_good = im_temp>(0.1*vmax)
        scl1 = np.nanmedian(hdu_sim[0].data[mask_good] / im_temp[mask_good])
        scl1 = np.abs(scl1)

        # Scale value for r^2 version
        im_temp = pad_or_cut_to_size(data_mod_r2, hdu_sim_r2[0].data.shape)
        mask_good = im_temp>(0.1*vmax2)
        scl2 = np.nanmedian(hdu_sim_r2[0].data[mask_good] / im_temp[mask_good])
        scl2 = np.abs(scl2)

        vmax_vals = [vmax,vmax*scl1,vmax2*scl2]
        hdus = [hdu_mod, hdu_sim, hdu_sim_r2]
        for i, ax in enumerate(axes[j]):
            hdulist = hdus[i]
            data = hdulist[0].data
            header = hdulist[0].header

            pixscale = header['PIXELSCL']
            rho = dist_image(data, pixscale=pixscale)

            rad = data.shape[0] * pixscale / 2
            extent = [-rad, rad, -rad, rad]

            ax.imshow(data, vmin=0, vmax=0.9*vmax_vals[i], extent=extent)

            ax.set_aspect('equal')
            if i > 0: ax.set_yticklabels([])
            if j < nfilt-1: ax.set_xticklabels([])
            if j==nfilt-1: ax.set_xlabel('Arcsec')
            if j==0: ax.set_title(ext_name[i])

            if i==0: 
                texp = obs.multiaccum_times['t_exp']
                texp = round(2*texp/100)*100
                exp_text = "{:.0f} sec".format(texp)
                ax.set_title('{} ({})'.format(obs.filter, exp_text))

            xlim = [-fov/2,fov/2]
            ylim = [-fov/2,fov/2]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            ax.xaxis.get_major_locator().set_params(nbins=10, steps=[1, 2, 5, 10])
            ax.yaxis.get_major_locator().set_params(nbins=10, steps=[1, 2, 5, 10])

            ax.tick_params(axis='both', color='white', which='both')
            for k in ax.spines.keys():
                ax.spines[k].set_color('white')

    name_sci = obs.sp_sci.name
    wfe_text = "WFE Drift = {} nm".format(wfe_drift)
    fig.suptitle('{} ({})'.format(name_sci, wfe_text), fontsize=16);
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.9, bottom=0.1)
    #fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9, bottom=0.07 , left=0.05, right=0.97)
    
    fname = "{}_images_{}.pdf".format(name_sci.replace(" ", ""), obs.mask)
    if save_fig: 
        fig.savefig(outdir+fname)
        
    if return_fig_axes:
        return fig, axes

        
def plot_images_swlw(obs_dict, hdu_dict, filt_keys, wfe_drift, fov=10, 
                    save_fig=False, outdir='', return_fig_axes=False):


    nfilt = len(filt_keys)
    ext_name = ['Model', 'Sim Image (linear scale)', 'Sim Image ($r^2$ scale)']
    nim = len(ext_name)

    fig, axes = plt.subplots(nim, nfilt, figsize=(14,7.5))
    axes = axes.transpose()
    for j, k in enumerate(filt_keys):
        obs = obs_dict[k]
        hdu_mod = obs.disk_hdulist
        if hdu_mod is None:
            raise ValueError('Disk model image is None. Did you forget to add the disk image?')
        hdu_sim = hdu_dict[k]
        data  = hdu_sim[0].data
        data -= np.nanmedian(data)

        # Make r^2 scaled version of data
        hdu_sim_r2 = deepcopy(hdu_sim)
        data  = hdu_sim_r2[0].data
        data -= np.nanmedian(data)
        header = hdu_sim_r2[0].header
        rho = dist_image(data, pixscale=header['PIXELSCL'])
        data *= rho**2

        # Max value for model
        data_mod   = hdu_mod[0].data
        header_mod = hdu_mod[0].header
        rho_mod    = dist_image(data_mod, pixscale=header_mod['PIXELSCL'])
        data_mod_r2 = data_mod*rho_mod**2
        vmax  = np.max(data_mod)
        vmax2 = np.max(data_mod_r2)
        
        # Scale value for data
        im_temp = pad_or_cut_to_size(data_mod, hdu_sim[0].data.shape)
        mask_good = im_temp>(0.1*vmax)
        scl1 = np.nanmedian(hdu_sim[0].data[mask_good] / im_temp[mask_good])
        scl1 = np.abs(scl1)
        
        # Scale value for r^2 version
        im_temp = pad_or_cut_to_size(data_mod_r2, hdu_sim_r2[0].data.shape)
        mask_good = im_temp>(0.1*vmax2)
        scl2 = np.nanmedian(hdu_sim_r2[0].data[mask_good] / im_temp[mask_good])
        scl2 = np.abs(scl2)

        vmax_vals = [vmax,vmax*scl1,vmax2*scl2]
        hdus = [hdu_mod, hdu_sim, hdu_sim_r2]
        for i, ax in enumerate(axes[j]):
            hdulist = hdus[i]
            data = hdulist[0].data
            header = hdulist[0].header

            pixscale = header['PIXELSCL']
            rho = dist_image(data, pixscale=pixscale)

            rad = data.shape[0] * pixscale / 2
            extent = [-rad, rad, -rad, rad]

            ax.imshow(data, vmin=0, vmax=0.9*vmax_vals[i], extent=extent)

            ax.set_aspect('equal')
            if j > 0: ax.set_yticklabels([])
            if i < nim-1: ax.set_xticklabels([])
            if i==nim-1: ax.set_xlabel('Arcsec')
            if j==0: ax.set_ylabel(ext_name[i])

            if i==0: 
                texp = obs.multiaccum_times['t_exp']
                texp = round(2*texp/100)*100
                exp_text = "{:.0f} sec".format(texp)
                ax.set_title('{} ({})'.format(obs.filter, exp_text))

            xlim = [-fov/2,fov/2]
            ylim = [-fov/2,fov/2]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            ax.xaxis.get_major_locator().set_params(nbins=9, steps=[1, 2, 5, 10])
            ax.yaxis.get_major_locator().set_params(nbins=9, steps=[1, 2, 5, 10])

            if fov<=2*rad:
                ax.tick_params(axis='both', color='white', which='both')
                for k in ax.spines.keys():
                    ax.spines[k].set_color('white')

    name_sci = obs.sp_sci.name
    wfe_text = "WFE Drift = {} nm".format(wfe_drift)
    fig.suptitle('{} ({})'.format(name_sci, wfe_text), fontsize=16);
    fig.tight_layout()

    fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9, bottom=0.07 , left=0.05, right=0.97)
    
    fname = "{}_images_{}.pdf".format(name_sci.replace(" ", ""), obs.mask)
    if save_fig: 
        fig.savefig(outdir+fname)
        
    if return_fig_axes:
        return fig, axes
