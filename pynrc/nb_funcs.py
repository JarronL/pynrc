# Makes print and division act like Python 3
from __future__ import print_function, division

# Import the usual libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from copy import deepcopy

#from .nrc_utils import S, stellar_spectrum, jupiter_spec, cond_table, cond_filter
#from .nrc_utils import read_filter, bp_2mass, channel_select, coron_ap_locs
#from .nrc_utils import dist_image, pad_or_cut_to_size
from .nrc_utils import *
from .obs_nircam import obs_hci
#from .obs_nircam import plot_contrasts, plot_contrasts_mjup, planet_mags, plot_planet_patches

from webbpsf_ext.synphot_ext import Observation

from tqdm.auto import tqdm, trange

import logging
_log = logging.getLogger('nb_funcs')

import pynrc
pynrc.setup_logging('WARN', verbose=False)

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


# Disk Models
def model_info(source, filt, dist, model_dir=''):
    
    # base_dir  = '/Volumes/NIRData/Andras_models_v2/'
    # model_dir = base_dir + source + '/'
    
    # Match filters with model
    filt_switch = {'F182M':'F210M', 'F210M':'F210M', 'F250M':'F250M',
                   'F300M':'F300M', 'F335M':'F335M', 'F444W':'F444W'}
    filt_model = filt_switch.get(filt, filt)
    fname = source + '_' + filt_model +'sc.fits'

    bp = read_filter(filt_model)
    w0 = bp.avgwave().to_value('um')

    # Model pixels are 4x oversampled
    detscale = (channel_select(bp))[0]
    model_scale = detscale / 4.
    
    # File name, arcsec/pix, dist (pc), wavelength (um), flux units, cen_star?
    model_dict = {
        'file'       : os.path.join(model_dir, fname), 
        'pixscale'   : model_scale, 
        'dist'       : dist, 
        'wavelength' : w0, 
        'units'      : 'Jy/pixel', 
        'cen_star'   : True
    }
    # args_model = (model_dir+fname, model_scale, dist, w0, 'Jy/pixel', True)

    return model_dict

def disk_rim_model(a_asec, b_asec, pa=0, sig_asec=0.1, flux_frac=0.5,
                   flux_tot=1.0, flux_units='mJy', wave_um=None, dist_pc=None,
                   pixsize=0.007, fov_pix=401):
    """
    Simple geometric model of an inner disk rim that simply creates an
    ellipsoidal ring with a brightness gradient along the major axis.
    
    Parameters
    ----------
    a_asec : float
        Semi-major axis of ellipse
    ba_asec : float
        Semi-minor axis of ellipse
        
    Keyword Args
    ------------
    pa : float
        Position angle of major axis
    sig_asec : float
        Sigma width of ring model
    flux_frac : float
        A brightness gradient can be applied along the semi-major axis. 
        This parameter dictates the relative brightness of the minimum flux
        (at the center of the axis) compared to the flux at the out edge
        of the geometric ring.
    flux_tot : float
        The total integrated flux of disk model.
    flux_units : str
        Units corresponding to `flux_tot`.
    wave_um : float or None
        Wavelength (in um) corresponding to `flux_tot`. Saved in output
        FITS header unless the value is None.
    dist_pc : float or None
        Assumed distance of model (in pc). Saved in output FITS header 
        unless the value is None.
    pixsize : float
        Desired model pixel size in arcsec.
    fov_pix : int
        Number of pixels for x/y dimensions of output model data.
    """
    
    
    from astropy.modeling.models import Ellipse2D
    from astropy.convolution import Gaussian2DKernel, convolve_fft
    from astropy.io import fits

    
    # Get polar and cartesian pixel coordinate grid
    sh = (fov_pix, fov_pix)
    r_pix, th_ang = dist_image(np.ones(sh), return_theta=True)
    x_pix, y_pix = rtheta_to_xy(r_pix, th_ang)
    
    # In terms of arcsec
    x_asec = pixsize * x_pix
    y_asec = pixsize * y_pix
    r_asec = pixsize * r_pix
    
    # Semi major/minor axes (pix)
    a_pix = a_asec / pixsize
    b_pix = b_asec / pixsize

    # Create ellipse functions
    e1 = Ellipse2D(theta=0, a=a_pix+1, b=b_pix+1)
    e2 = Ellipse2D(theta=0, a=a_pix-1, b=b_pix-1)
    
    # Make the two ellipse images and subtract
    e1_im = e1(x_pix,y_pix)
    e2_im = e2(x_pix,y_pix)
    e_im = e1_im - e2_im

    # Produce a brightness gradient along major axis
    grad_im = (1-flux_frac) * np.abs(x_pix) / a_pix + flux_frac
    e_im = e_im * grad_im
    
    # Convolve image with Gaussian to simulate scattering
    sig_pix = sig_asec / pixsize
    kernel = Gaussian2DKernel(sig_pix)
    e_im = convolve_fft(e_im, kernel)

    # Rotate
    th_deg = pa - 90.
    e_im = rotate_offset(e_im, angle=-th_deg, order=3, reshape=False)
    
    e_im = flux_tot * e_im / np.sum(e_im)
    
    hdu = fits.PrimaryHDU(e_im)
    hdu.header['PIXELSCL'] = (pixsize, "Pixel Scale (asec/pix)")
    hdu.header['UNITS'] = "{}/pixel".format(flux_units)
    if wave_um is not None:
        hdu.header['WAVE'] = (wave_um, "Wavelength (microns)")
    if dist_pc is not None:
        hdu.header['DISTANCE'] = (dist_pc, "Distance (pc)")
        
    return fits.HDUList([hdu])


def obs_wfe(wfe_ref_drift, filt_list, sp_sci, dist, sp_ref=None, args_disk=None, 
            wind_mode='WINDOW', subsize=None, fov_pix=None, verbose=False, narrow=False,
            model_dir=None, large_grid=True, sgd_type=None, slew_std=0, fsm_std=0, 
            quiet=False, **kwargs):
    """
    For a given WFE drift and series of filters, create a list of 
    NIRCam observations.
    """
    
    if sp_ref is None: sp_ref = sp_sci

    obs_dict = {}
    for filt, mask, pupil in filt_list:
        # Create identification key
        key = make_key(filt, mask=mask, pupil=pupil)
        if not quiet:
            print(key)

        # Disk Model
        if args_disk is None:
            args_disk_temp = None
        elif 'auto' in args_disk:
            # Convert to photons/sec in specified filter
            name = sp_sci.name.replace(' ', '')
            args_disk_temp = model_info(name, filt, dist, model_dir=model_dir)
        else:
            args_disk_temp = args_disk
                        
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
        # if (mask is None) and (np.mod(fov_pix,2)==0):
        #     fov_pix += 1
        if np.mod(fov_pix,2)==0:
            fov_pix += 1
        # Other coronagraph vs direct imaging settings
        module, oversample = ('B', 4) if mask is None else ('A', 2)
        
        if mask is None:
            bar_offset = None
        elif narrow and ('SWB' in mask):
            bar_offset = -8
        elif narrow and ('LWB' in mask):
            bar_offset = 8
        else:
            bar_offset = None

        # Select detector for imaging mode
        if module=='B':
            bp = read_filter(filt)
            detector = 'NRCB1' if bp.avgwave().to_value('um') < 2.5 else 'NRCB5'
        else:
            detector = None
        
        # Initialize and store the observation
        # A reference observation is stored inside each parent obs_hci class.
        obs = obs_hci(sp_sci, dist, sp_ref=sp_ref, filter=filt, image_mask=mask, pupil_mask=pupil, 
                      detector=detector, wind_mode=wind_mode, xpix=subuse, ypix=subuse,
                      wfe_ref_drift=wfe_ref_drift, fov_pix=fov_pix, oversample=oversample, 
                      disk_params=args_disk_temp, verbose=verbose, bar_offset=bar_offset,
                      autogen_coeffs=False, sgd_type=sgd_type, slew_std=slew_std, fsm_std=fsm_std, **kwargs)

        obs.gen_psf_coeff()
        # Enable WFE drift
        obs.gen_wfedrift_coeff()
        # Enable mask-dependent
        obs.gen_wfemask_coeff(large_grid=large_grid)
        # Calculate PSF offset to center
        obs.calc_psf_offset_from_center()

        obs_dict[key] = obs
        fov_pix = fov_pix_orig

        # if there's a disk input, then we want to remove disk 
        # contributions from stellar flux and recompute to make 
        # sure total flux counts matches what we computed for 
        # sp_sci in previous section to match real photometry
        if args_disk is not None:
            obs = obs_dict[key]
    
            star_flux = obs.star_flux(sp=sp_sci) # Pass original input spectrum
            disk_flux = obs.disk_hdulist[0].data.sum()
            obs.sp_sci = sp_sci * (1 - disk_flux / star_flux)
            obs.sp_sci.name = sp_sci.name

            if verbose:
                filt = key.split('_')[0]
                print(f'{filt:6} | {disk_flux:9.0f} | {star_flux:9.0f} | {obs.star_flux():9.0f}')


            if sp_ref is sp_sci:
                obs.sp_ref = obs.sp_sci
                
    # Generation mask position dependent PSFs
    if quiet or (args_disk_temp is None):
        iter_vals = obs_dict.keys()
    else:
        iter_vals = tqdm(obs_dict.keys(), desc='Obs', leave=False)
    for key in iter_vals:
        obs_dict[key].gen_disk_psfs()

    return obs_dict


def obs_optimize(obs_dict, sp_opt=None, well_levels=None, tacq_max=2000, **kwargs):
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

    verbose = kwargs.pop('verbose', True)

    # A very faint bg object on which to maximize S/N
    # If sp_opt is not set, then default to a 20th magnitude flat source
    if sp_opt is None:
        sp_opt = stellar_spectrum('flat', 20, 'vegamag', bp_k)

    
    # Some observations may saturate, so define a list of  maximum well level
    # values that we will incrementally check until a ramp setting is found
    # that meets the constraints.
    if well_levels is None:
        well_levels = [0.8, 1.5, 3.0, 5.0, 10.0, 20.0, 100.0, 150.0, 300.0, 500.0]

    if verbose:
        print(['Pattern', 'NGRP', 'NINT', 't_int', 't_exp', 't_acq', 'SNR', 'Well', 'eff'])

    filt_keys = list(obs_dict.keys())
    filt_keys.sort()
    for j, key in enumerate(filt_keys):
        if verbose:
            print('')
            print(key)

        obs = obs_dict[key]

        sp_sci, sp_ref = (obs.sp_sci, obs.sp_ref)
        
        # Ramp optimization for both science and reference targets
        for j, sp in enumerate([sp_sci, sp_ref]):
            i = nrow = 0
            while nrow==0:
                well_max = well_levels[i]
                tbl = obs.ramp_optimize(sp_opt, sp, well_frac_max=well_max, tacq_max=tacq_max, **kwargs)
                nrow = len(tbl)
                i+=1
                
            # Grab the highest ranked MULTIACCUM settings and update the detector readout
            v1, v2, v3 = tbl['Pattern', 'NGRP', 'NINT'][0]
            
            vals = list(tbl[0])#.as_void()
            strout = '{:10} {:4.0f} {:4.0f}'.format(vals[0], vals[1], vals[2])
            for v in vals[3:]:
                strout = strout + ', {:.4f}'.format(v)
            if verbose:
                print(strout)

            # SW filter piggy-back on two LW filters, so 2 x tacq
            # is_SW = obs.bandpass.avgwave().to_value('um') < 2.5
            # if is_SW: 
            #     v3 *= 2
            
            # Coronagraphic observations have two roll positions, so cut NINT by 2
            # if obs.image_mask is not None: 
            #     v3 = int(v3/2) 
            if j==0:
                obs.update_detectors(read_mode=v1, ngroup=v2, nint=v3)
            else:
                obs.update_detectors(read_mode=v1, ngroup=v2, nint=v3, do_ref=True)
        


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
    if verbose:
        iter_vals = trange(len(filt_keys), leave=False)
    else:
        iter_vals = range(len(filt_keys))

    for i in iter_vals:
        key = filt_keys[i]
        obs = obs_dict[key]
        if verbose: 
            iter_vals.set_description(key, refresh=True)

        # Stores tuple of (Radial Distances, Contrast, and Sensitivity) for each WFE drift
        curves = []
        if verbose:
            jter_vals = tqdm(wfe_list, leave=False, desc='WFE Drift')
        else:
            jter_vals = wfe_list
        for wfe_drift in jter_vals:
            
            no_ref = kwargs.get('no_ref', False)
            if no_ref:
                wfe_ref_drift = 0
                wfe_roll_drift = wfe_drift
            else:
                # Assume drift between Roll1 and Roll2 is 2 nm WFE or less
                wfe_ref_drift = wfe_drift
                wfe_roll_drift = wfe_ref_drift/2 if wfe_ref_drift<=2 else 2

            kwargs['wfe_ref_drift'] = wfe_ref_drift
            kwargs['wfe_roll_drift'] = wfe_roll_drift

            result = obs.calc_contrast(roll_angle=roll_angle, nsig=nsig, **kwargs)
            curves.append(result)
            
        contrast_all[key] = curves
    return contrast_all


def do_gen_hdus(obs_dict, filt_keys, wfe_ref_drift, wfe_roll_drift, 
                return_oversample=True, **kwargs):
    
    """
    kwargs to pass to gen_roll_image() and their defaults:
    
    PA1 = 0
    PA2 = 10
    zfact = None
    return_oversample = True
    exclude_disk  = False
    exclude_noise = False
    no_ref        = False
    opt_diff      = False
    use_cmask     = False
    ref_scale_all = False
    xyoff_roll1   = None
    xyoff_roll2   = None
    xyoff_ref     = None
    """
    
    hdulist_dict = {}
    for key in tqdm(filt_keys):
        # if verbose: print(key)
        obs = obs_dict[key]
        use_cmask = kwargs.pop('use_cmask', False)
        hdulist = obs.gen_roll_image(return_oversample=return_oversample, use_cmask=use_cmask,
            wfe_ref_drift=wfe_ref_drift, wfe_roll_drift=wfe_roll_drift, **kwargs)
        
        hdulist_dict[key] = hdulist
        
    return hdulist_dict

def do_sat_levels(obs, satval=0.95, ng_min=2, ng_max=None, verbose=True, 
                  charge_migration=True, niter=5, satmax=1, corners=True,
                  plot=True, xylim=2.5, return_fig_axes=False, return_more=False,
                  **kwargs):

    """Only for obs.hci classes

    Parameters
    ------------
    charge_migration : bool
        Include charge migration effects?
    satmax : float
        Saturation value to limit charge migration. Default is 1.5.
    niter : int
        Number of iterations for charge migration. Default is 5.
    corners : bool
        Include corner pixels in charge migration? Default is True.
    return_more : bool
        Return additional information (sat_rad, sci_levels2_max, ref_levels2_max)? 
        Default is False.
    """

    # Charge migration keywords
    kwargs['charge_migration'] = charge_migration
    kwargs['niter'] = niter
    kwargs['satmax'] = satmax
    kwargs['corners'] = corners
    
    ng_max_sci = obs.Detector.multiaccum.ngroup if ng_max is None else ng_max
    ng_max_ref = obs.Detector_ref.multiaccum.ngroup if ng_max is None else ng_max
    kw_gen_psf = {'return_oversample': False,'return_hdul': False}
    
    # Well level of each pixel for science source
    image = obs.calc_psf_from_coeff(sp=obs.sp_sci, **kw_gen_psf)
    sci_levels1 = obs.saturation_levels(ngroup=ng_min, image=image, **kwargs)
    sci_levels2 = obs.saturation_levels(ngroup=ng_max_sci, image=image, **kwargs)
    if charge_migration:
        # Get max well fill without charge migration
        kwargs_temp = kwargs.copy()
        kwargs_temp['charge_migration'] = False
        sci_levels1_temp = obs.saturation_levels(ngroup=ng_min, image=image, **kwargs_temp)
        sci_levels2_temp = obs.saturation_levels(ngroup=ng_max_sci, image=image, **kwargs_temp)
        sci_levels1_max = sci_levels1_temp.max()
        sci_levels2_max = sci_levels2_temp.max()
    else:
        sci_levels1_max = sci_levels1.max()
        sci_levels2_max = sci_levels2.max()

    # Well level of each pixel for reference source
    image = obs.calc_psf_from_coeff(sp=obs.sp_ref, **kw_gen_psf)
    ref_levels1 = obs.saturation_levels(ngroup=ng_min, image=image, do_ref=True, **kwargs)
    ref_levels2 = obs.saturation_levels(ngroup=ng_max_ref, image=image, do_ref=True, **kwargs)
    if charge_migration:
        # Get max well fill without charge migration
        kwargs_temp = kwargs.copy()
        kwargs_temp['charge_migration'] = False
        ref_levels1_temp = obs.saturation_levels(ngroup=ng_min, image=image, do_ref=True, **kwargs_temp)
        ref_levels2_temp = obs.saturation_levels(ngroup=ng_max_ref, image=image, do_ref=True, **kwargs_temp)
        ref_levels1_max = ref_levels1_temp.max()
        ref_levels2_max = ref_levels2_temp.max()
    else:
        ref_levels1_max = ref_levels1.max()
        ref_levels2_max = ref_levels2.max()
        
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

    # Get saturation radius
    pixscale = obs.pixelscale
    if nsat1_sci == nsat1_ref == 0:
        sat_rad_max = sat_rad = 0
    else:
        mask_temp = sci_mask1 if nsat1_sci>nsat1_ref else ref_mask1
        rho_asec = dist_image(mask_temp, pixscale=pixscale)
        sat_rad_max = rho_asec[mask_temp].max() + pixscale/2
        sat_rad = np.sqrt(nsat1_sci / np.pi) * pixscale
    
    if verbose:
        print('Sci: {}'.format(obs.sp_sci.name))
        print('  {} saturated pixel at NGROUP={}; Max Well: {:.2f}'\
            .format(nsat1_sci, ng_min, sci_levels1_max))
        print('  {} saturated pixel at NGROUP={}; Max Well: {:.2f}'\
            .format(nsat2_sci, ng_max_sci, sci_levels2_max))

        print('Ref: {}'.format(obs.sp_ref.name))
        print('  {} saturated pixel at NGROUP={}; Max Well: {:.2f}'.\
            format(nsat1_ref, ng_min, ref_levels1_max))
        print('  {} saturated pixel at NGROUP={}; Max Well: {:.2f}'.\
            format(nsat2_ref, ng_max_ref, ref_levels2_max))

        print(f'Sat Max Dist NG={ng_min}: {sat_rad_max/pixscale:.2f} pix ({sat_rad_max:.2f} arcsec)')
        print(f'Sat Avg Dist NG={ng_min}: {sat_rad/pixscale:.2f} pix ({sat_rad:.2f} arcsec)')


    if (nsat2_sci==nsat2_ref==0) and (plot==True):
        plot=False
        print('Plotting turned off; no saturation detected.')

    if plot:
        fig, axes_all = plt.subplots(2,2, figsize=(8,8))

        xlim = ylim = np.array([-1,1])*xylim

        # Plot science source
        nsat1, nsat2 = (nsat1_sci, nsat2_sci)
        sat_mask1, sat_mask2 = (sci_mask1, sci_mask2)
        sp = obs.sp_sci

        xpix, ypix = (obs.det_info['xpix'], obs.det_info['ypix'])
        bar_offpix = obs.bar_offset / obs.pixelscale
        # Determine final shift amounts to location along bar
        # Shift to position relative to center of image
        if (('FULL' in obs.det_info['wind_mode']) and (obs.image_mask is not None)) or obs._use_ap_info:
            xcen, ycen = (obs.siaf_ap.XSciRef - 1, obs.siaf_ap.YSciRef - 1)
            # Offset relative to center of image
            delx_pix = (xcen - (xpix/2 - 0.5))  # 'sci' pixel shifts
            dely_pix = (ycen - (ypix/2 - 0.5))  # 'sci' pixel shifts
        else:
            # Otherwise assumed mask is in center of subarray for simplicity
            # For odd dimensions, this is in a pixel center.
            # For even dimensions, this is at the pixel boundary.
            xcen, ycen = (xpix/2. - 0.5, ypix/2. - 0.5)
            # Add bar offset
            xcen += bar_offpix  # Add bar offset
            delx_pix, dely_pix = (bar_offpix, 0)

        delx, dely = (xcen - xpix/2, ycen - ypix/2)
        extent_pix = np.array([-xpix/2-delx,xpix/2-delx,-ypix/2-dely,ypix/2-dely])
        extent = extent_pix * obs.pixelscale

        axes = axes_all[0]
        axes[0].imshow(sat_mask1, extent=extent)
        axes[1].imshow(sat_mask2, extent=extent)

        axes[0].set_title('{} Saturation (NGROUP=2)'.format(sp.name))
        axes[1].set_title('{} Saturation (NGROUP={})'.format(sp.name, ng_max_sci))

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

        axes = axes_all[1]
        axes[0].imshow(sat_mask1, extent=extent)
        axes[1].imshow(sat_mask2, extent=extent)

        axes[0].set_title('{} Saturation (NGROUP=2)'.format(sp.name))
        axes[1].set_title('{} Saturation (NGROUP={})'.format(sp.name, ng_max_ref))

        for ax in axes:
            ax.set_xlabel('Arcsec')
            ax.set_ylabel('Arcsec')

            ax.tick_params(axis='both', color='white', which='both')
            for k in ax.spines.keys():
                ax.spines[k].set_color('white')

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        fig.tight_layout()
        
    if return_more:
        res = (sat_rad, sci_levels2_max, ref_levels2_max)
    else:
        res = sat_rad
        
    if return_fig_axes and plot:
        return (fig, axes), res
    else:
        return res
    
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
                im_slope[ind] = jl_poly_fit(tvals, data[:,ind])[1]
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
    
def plot_contrasts_mjup(curves, nsig, wfe_list, obs=None, sat_rad=None, age=100,
    ax=None, colors=None, xr=[0,10], yr=None, file=None, linder_models=True,
    twin_ax=False, return_axes=False, **kwargs):
    """Plot mass contrast curves

    Plot a series of mass contrast curves for corresponding WFE drifts.

    Parameters
    ----------
    curves : list
        A list with length corresponding to `wfe_list`. Each list element
        has three arrays in a tuple: the radius in arcsec, n-sigma contrast,
        and n-sigma sensitivity limit (vega mag).
    nsig : float
        N-sigma limit corresponding to sensitivities/contrasts.
    wfe_list : array-like
        List of WFE drift values corresponding to each set of sensitivities
        in `curves` argument.

    Keyword Args
    ------------
    obs : :class:`obs_hci`
        Corresponding observation class that created the contrast curves.
        Uses distances and stellar magnitude to plot contrast and AU
        distances on opposing axes. Also necessary for mjup=True.
    sat_rad : float
        Saturation radius in arcsec. If >0, then that part of the contrast
        curve is excluded from the plot
    age : float
        Required for plotting limiting planet masses.
    file : string
        Location and name of COND or Linder isochrone file.
    ax : matplotlib.axes
        Axes on which to plot curves.
    colors : None, array-like
        List of colors for contrast curves. Default is gradient of blues.
    twin_ax : bool
        Plot opposing axes in alternate units.
    return_axes : bool
        Return the matplotlib axes to continue plotting. If `obs` is set,
        then this returns three sets of axes.

    """
    if sat_rad is None:
        sat_rad = 0

    if ax is None:
        fig, ax = plt.subplots()
    if colors is None:
        lin_vals = np.linspace(0.2,0.8,len(wfe_list))
        colors = plt.cm.Blues_r(lin_vals)

    filt = obs.filter
    mod = obs.module
    dist = obs.distance
    if linder_models:
        # Grab Linder model data
        tbl = linder_table(file=file)
        mass_data, mag_data = linder_filter(tbl, filt, age, dist=dist)
    else:
        # Grab COND model data
        tbl = cond_table(age=age, file=file)
        mass_data, mag_data = cond_filter(tbl, filt, module=mod, dist=dist)

    # Plot the data
    isort = np.argsort(mag_data)
    for j, wfe_ref_drift in enumerate(wfe_list):
        rr, contrast, mag_sens = curves[j]
        label='$\Delta$' + "WFE = {} nm".format(wfe_list[j])

        # Interpolate in log space
        xv, yv = mag_data[isort], np.log10(mass_data[isort])
        xint = mag_sens
        yint = np.interp(xint, xv, yv)
        # Choose the lowest mass value brighter than the given mag limits
        yvals = np.array([np.min(yint[xint<=xv]) for xv in xint])
        yvals = 10**yvals

        xvals = rr[rr>sat_rad]
        yvals = yvals[rr>sat_rad]
        ax.plot(xvals, yvals, label=label, color=colors[j], zorder=1, lw=2)

    if xr is not None: ax.set_xlim(xr)
    if yr is not None: ax.set_ylim(yr)

    ax.xaxis.get_major_locator().set_params(nbins=10, steps=[1, 2, 5, 10])
    ax.yaxis.get_major_locator().set_params(nbins=10, steps=[1, 2, 5, 10])

    ylabel = 'Mass Limits ($M_{\mathrm{Jup}}$)'
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Separation (arcsec)')

    if twin_ax:
        # Plot opposing axes in alternate units
        yr2 = np.array(ax.get_ylim()) * 318.0 # Convert to Earth masses
        ax2 = ax.twinx()
        ax2.set_ylim(yr2)
        ax2.set_ylabel('Earth Masses')

        ax3 = ax.twiny()
        xr3 = np.array(ax.get_xlim()) * obs.distance
        ax3.set_xlim(xr3)
        ax3.set_xlabel('Separation (AU)')

        ax3.xaxis.get_major_locator().set_params(nbins=9, steps=[1, 2, 5, 10])

        if return_axes:
            return (ax, ax2, ax3)
    else:
        if return_axes:
            return ax


def plot_contrasts(curves, nsig, wfe_list, obs=None, sat_rad=None, ax=None,
    colors=None, xr=[0,10], yr=[25,5], return_axes=False):
    """Plot contrast curves

    Plot a series of contrast curves for corresponding WFE drifts.

    Parameters
    ----------
    curves : list
        A list with length corresponding to `wfe_list`. Each list element
        has three arrays in a tuple: the radius in arcsec, n-sigma contrast,
        and n-sigma sensitivity limit (vega mag).
    nsig : float
        N-sigma limit corresponding to sensitivities/contrasts.
    wfe_list : array-like
        List of WFE drift values corresponding to each set of sensitivities
        in `curves` argument.

    Keyword Args
    ------------
    obs : :class:`obs_hci`
        Corresponding observation class that created the contrast curves.
        Uses distances and stellar magnitude to plot contrast and AU
        distances on opposing axes.
    sat_rad : float
        Saturation radius in arcsec. If >0, then that part of the contrast
        curve is excluded from the plot
    ax : matplotlib.axes
        Axes on which to plot curves.
    colors : None, array-like
        List of colors for contrast curves. Default is gradient of blues.
    return_axes : bool
        Return the matplotlib axes to continue plotting. If `obs` is set,
        then this returns three sets of axes.

    """
    if sat_rad is None:
        sat_rad = 0
    
    if ax is None:
        fig, ax = plt.subplots()
    if colors is None:
        lin_vals = np.linspace(0.3,0.8,len(wfe_list))
        colors = plt.cm.Blues_r(lin_vals)
        
    delta_str = '$\Delta$'
    for j in range(len(wfe_list)): #for j, wfe_ref_drift in enumerate(wfe_list):
        rr, contrast, mag_sens = curves[j]
        xvals = rr[rr>sat_rad]
        yvals = mag_sens[rr>sat_rad]
        label= f"{delta_str}WFE = {wfe_list[j]} nm"
        ax.plot(xvals, yvals, label=label, color=colors[j], zorder=1, lw=2)

    if xr is not None: ax.set_xlim(xr)
    if yr is not None: ax.set_ylim(yr)


    ax.xaxis.get_major_locator().set_params(nbins=10, steps=[1, 2, 5, 10])
    ax.yaxis.get_major_locator().set_params(nbins=10, steps=[1, 2, 5, 10])

    ax.set_ylabel('{:.0f}-$\sigma$ Sensitivities (mag)'.format(nsig))
    ax.set_xlabel('Separation (arcsec)')

    # Plot opposing axes in alternate units
    if obs is not None:
        yr1 = np.array(ax.get_ylim())
        yr2 = 10**((obs.star_flux('vegamag') - yr1) / 2.5)
        ax2 = ax.twinx()
        ax2.set_yscale('log')
        ax2.set_ylim(yr2)
        ax2.set_ylabel(f'{nsig:.0f}-$\sigma$ Contrast ({obs.filter})')

        ax3 = ax.twiny()
        xr3 = np.array(ax.get_xlim()) * obs.distance
        ax3.set_xlim(xr3)
        ax3.set_xlabel('Separation (AU)')

        ax3.xaxis.get_major_locator().set_params(nbins=9, steps=[1, 2, 5, 10])

        if return_axes:
            return (ax, ax2, ax3)
    else:
        if return_axes:
            return ax


def planet_mags(obs, age=10, entropy=13, mass_list=[10,5,2,1], av_vals=[0,25], atmo='hy3s',
    cond=False, linder=False, **kwargs):
    """Exoplanet Magnitudes

    Determine a series of exoplanet magnitudes for given observation.
    By default, use Spiegel & Burrows 2012 models, but has the option
    to use the COND models from https://phoenix.ens-lyon.fr/Grids.
    These are useful because SB12 model grids only ranges from 1-1000 Myr
    with masses 1-15 MJup.

    cond : bool
        Instead of plotting sensitivities, use COND models to plot the
        limiting planet masses.
    linder : bool
        Instead of plotting sensitivities, use Linder models to plot the
        limiting planet masses.
    file : string
        Location and name of COND or Linder file.
    """

    if av_vals is None:
        av_vals = [0,0]

    pmag = {}
    for i,m in enumerate(mass_list):
        flux_list = []
        for j,av in enumerate(av_vals):
            sp = obs.planet_spec(mass=m, age=age, Av=av, entropy=entropy, atmo=atmo, **kwargs)
            sp_obs = Observation(sp, obs.bandpass, binset=obs.bandpass.waveset)
            flux = sp_obs.effstim('vegamag')
            flux_list.append(flux)
        pmag[m] = tuple(flux_list)

    # Do COND models instead
    # But still want SB12 models to get A_V information
    if cond or linder:
        # All mass and mag data for specified filter
        filt = obs.filter
        mod = obs.module
        dist = obs.distance
        if linder:
            tbl = linder_table(**kwargs)
            mass_data, mag_data = linder_filter(tbl, filt, age, dist=dist, **kwargs)
        else:
            # Grab COND model data
            tbl = cond_table(age=age, **kwargs)
            mass_data, mag_data = cond_filter(tbl, filt, module=mod, dist=dist, **kwargs)

        # Mag information for the requested masses
        isort = np.argsort(mass_data)
        xv, yv = np.log10(mass_data[isort]), mag_data[isort]
        mags0 = np.interp(np.log10(mass_list), np.log10(mass_data[isort]), mag_data[isort])
    
        # Apply extinction
        for i, m in enumerate(mass_list):
            if np.allclose(av_vals, 0):
                dm = np.array([0,0])
            else:
                #SB12 at A_V=0
                sp = obs.planet_spec(mass=m, age=age, Av=0, entropy=entropy, atmo=atmo, **kwargs)
                sp_obs = Observation(sp, obs.bandpass, binset=obs.bandpass.waveset)
                sb12_mag = sp_obs.effstim('vegamag')

                # Get magnitude offset due to extinction
                dm = np.array(pmag[m]) - sb12_mag
                dm2 = pmag[m][1] - sb12_mag

            # Apply extinction to COND models
            pmag[m] = tuple(mags0[i] + dm)

    return pmag


def plot_planet_patches(ax, obs, age=10, entropy=13, mass_list=[10,5,2,1], av_vals=[0,25],
    cols=None, update_title=False, linder=False, **kwargs):
    """Plot exoplanet magnitudes in region corresponding to extinction values."""

    import matplotlib.patches as mpatches

    # Don't plot anything if 
    if mass_list is None:
        _log.info("mass_list=None; Not plotting planet patch locations.")
        return

    xlim = ax.get_xlim()


    #lin_vals = np.linspace(0,0.5,4)
    #cols = plt.cm.Purples_r(lin_vals)[::-1]
    if cols is None:
        cols = plt.cm.tab10(np.linspace(0,1,10))

    dist = obs.distance

    if entropy<8: entropy=8
    if entropy>13: entropy=13

    pmag = planet_mags(obs, age, entropy, mass_list, av_vals, linder=linder, **kwargs)
    for i,m in enumerate(mass_list):
        label = 'Mass = {} '.format(m) + '$M_{\mathrm{Jup}}$'
        if av_vals is None:
            ax.plot(xlim, pmag[m], color=cols[i], lw=1, ls='--', label=label)
        else:
            pm_min, pm_max = pmag[m]
            rect = mpatches.Rectangle((xlim[0], pm_min), xlim[1], pm_max-pm_min,
                                     alpha=0.2, color=cols[i], label=label, zorder=2)
            ax.add_patch(rect)
            ax.plot(xlim, [pm_min]*2, color=cols[i], lw=1, alpha=0.3)
            ax.plot(xlim, [pm_max]*2, color=cols[i], lw=1, alpha=0.3)

    entropy_switch = {13:'Hot', 8:'Cold'}
    entropy_string = entropy_switch.get(entropy, "Warm")
    ent_str = 'BEX Models' if linder else '{} Start'.format(entropy_string)

    if av_vals is None:
        av_str = ''
    else:
        av_str = ' ($A_V = [{:.0f},{:.0f}]$)'.format(av_vals[0],av_vals[1])
    #age_str = 'Age = {:.0f} Myr; '.format(age)
    #dist_str = 'Dist = {:.1f} pc; '.format(dist) if dist is not None else ''
    #dist_str=""

    #ax.set_title('{} -- {} ({}{}{})'.format(obs.filter,ent_str,age_str,dist_str,av_str))

    if update_title:
        ax.set_title('{} -- {}{}'.format(obs.filter,ent_str,av_str))


def plot_hdulist(hdulist, ext=0, xr=None, yr=None, ax=None, return_ax=False,
    cmap=None, scale='linear', vmin=None, vmax=None, axes_color='white',
    half_pix_shift=False, cb_label='Counts/sec', **kwargs):

    from webbpsf import display_psf

    if ax is None:
        fig, ax = plt.subplots()
    if cmap is None:
        cmap = matplotlib.rcParams['image.cmap']

    # This has to do with even/odd number of pixels in array.
    # Usually everything is centered in the middle of a pixel
    # and for odd array sizes that is where (0,0) will be plotted.
    # However, even array sizes will have (0,0) at the pixel border,
    # so this just shifts the entire image accordingly.
    if half_pix_shift:
        oversamp = hdulist[ext].header['OSAMP']
        shft = 0.5*oversamp
        hdul = deepcopy(hdulist)
        hdul[0].data = fourier_imshift(hdul[0].data, shft, shft)
    else:
        hdul = hdulist

    data = hdul[ext].data
    if vmax is None:
        vmax = 0.75 * np.nanmax(data) if scale=='linear' else np.nanmax(data)
    if vmin is None:
        vmin = 0 if scale=='linear' else vmax/1e6

    
    out = display_psf(hdul, ext=ext, ax=ax, title='', cmap=cmap,
                      scale=scale, vmin=vmin, vmax=vmax, return_ax=True, **kwargs)
    try:
        ax, cb = out
        cb.set_label(cb_label)
    except:
        ax = out

    ax.set_xlim(xr)
    ax.set_ylim(yr)
    ax.set_xlabel('Arcsec')
    ax.set_ylabel('Arcsec')


    ax.tick_params(axis='both', color=axes_color, which='both')
    for k in ax.spines.keys():
        ax.spines[k].set_color(axes_color)

    ax.xaxis.get_major_locator().set_params(nbins=9, steps=[1, 2, 5, 10])
    ax.yaxis.get_major_locator().set_params(nbins=9, steps=[1, 2, 5, 10])
    
    if return_ax:
        return ax


    
    
###########################################
# Plotting images and contrast curves
###########################################

def update_yscale(ax, scale_type, ylim=None):
    
    # Some fancy log+linear plotting
    from matplotlib.ticker import FixedLocator, ScalarFormatter, LogFormatterSciNotation
    if scale_type=='symlog':
        ylim = [0,100] if ylim is None else ylim
        ax.set_ylim(ylim)
        yr = ax.get_ylim()
        ax.set_yscale('symlog', linthreshy=10, linscaley=2)
        ax.set_yticks(list(range(0,10)) + [10,100,1000])
        #ax.get_yaxis().set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())

        minor_log = list(np.arange(20,100,10)) + list(np.arange(200,1000,100))
        minorLocator = FixedLocator(minor_log)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.set_ylim([0,yr[1]])
    elif scale_type=='log':
        ax.set_yscale('log')
        ylim = [0.1,100] if ylim is None else ylim
        ax.set_ylim(ylim)
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    elif 'lin' in scale_type:
        ax.set_yscale('linear')
        ylim = [0,100] if ylim is None else ylim
        ax.set_ylim(ylim)
        

def do_plot_contrasts(curves_ref, curves_roll, nsig, wfe_list, obs, age, age2=None, 
    sat_rad=0, jup_mag=True, xr=[0,10], yr=[22,8], xr2=[0,10], yscale2='log', yr2=None,
    save_fig=False, outdir='', return_fig_axes=False, **kwargs):
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
        ax1, ax2, ax3 = plot_contrasts(curves_ref, nsig, wfe_list, 
            obs=obs, ax=ax, colors=c1, xr=xr, yr=yr, return_axes=True)
    if curves_roll is not None:
        obs_kw = None if curves_ref is not None else obs
        axes2 = plot_contrasts(curves_roll, nsig, wfe_list, 
            obs=obs_kw, ax=ax, colors=c2, xr=xr, yr=yr, return_axes=True)
        if curves_ref is None:
            ax1, ax2, ax3 = axes2
    axes1_all = [ax1, ax2, ax3]
    #plot_planet_patches(ax, obs, age=age, av_vals=None, cond=True)

    #ax.set_ylim([22,8])

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
        jobs = Observation(jspec, obs.bandpass, binset=obs.bandpass.waveset)
        jmag = jobs.effstim('vegamag')
        if jmag<np.max(ax.get_ylim()):
            ax.plot(xr, [jmag,jmag], color='C2', ls='--')
            txt = 'Jupiter at {:.1f} pc'.format(obs.distance)
            ax.text(xr[0]+0.02*(xr[1]-xr[0]), jmag, txt, horizontalalignment='left', verticalalignment='bottom')

    # Plot in terms of Jupiter Masses
    ax = axes[1]
    age1 = age
    if curves_ref is not None:
        ax1, ax2, ax3 = plot_contrasts_mjup(curves_ref, nsig, wfe_list, obs=obs, 
            age=age1, ax=ax, colors=c1, xr=xr2, twin_ax=True, yr=None, return_axes=True)
    if curves_roll is not None:
        twin_kw = False if curves_ref is not None else True
        axes2 = plot_contrasts_mjup(curves_roll, nsig, wfe_list, obs=obs, 
            age=age1, ax=ax, colors=c2, xr=xr2, twin_ax=twin_kw, yr=None, return_axes=True)
        if curves_ref is None:
            ax1, ax2, ax3 = axes2
    axes2_all = [ax1, ax2, ax3]

            
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
        
    # Update fancing y-axis scaling on right plot
    update_yscale(ax, yscale2, ylim=yr2)
    yr_temp = np.array(ax.get_ylim()) * 318.0
    update_yscale(axes2_all[1], yscale2, ylim=yr_temp)


    # Saturation regions
    if sat_rad > 0:
        sat_rad_asec = sat_rad
        for ax in axes:
            ylim = ax.get_ylim()
            rect = mpatches.Rectangle((0, ylim[0]), sat_rad, ylim[1]-ylim[0], alpha=0.2, color='k', zorder=2)
            ax.add_patch(rect)

    name_sci = obs.sp_sci.name
    name_ref = obs.sp_ref.name
    if curves_ref is None:
        title_str = '{} (dist = {:.1f} pc) -- {} Contrast Curves'\
            .format(name_sci, obs.distance, obs.filter)
    else:
        title_str = '{} (dist = {:.1f} pc; PSF Ref: {}) -- {} Contrast Curves'\
            .format(name_sci, obs.distance, name_ref, obs.filter)
    fig.suptitle(title_str, fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85, bottom=0.1 , left=0.05, right=0.95)

    fname = "{}_contrast_{}.pdf".format(name_sci.replace(" ", ""), obs.image_mask)
    if save_fig: 
        fig.savefig(outdir+fname)
        
    if return_fig_axes:
        return fig, (axes1_all, axes2_all)
        
        
        
def do_plot_contrasts2(key1, key2, curves_all, nsig, obs_dict, wfe_list, age, sat_dict=None,
                       label1='Curves1', label2='Curves2', xr=[0,10], yr=[24,8], 
                       yscale2='log', yr2=None, av_vals=[0,10], curves_all2=None, 
                       c1=None, c2=None, linder_models=True, planet_patches=True, **kwargs):

    fig, axes = plt.subplots(1,2, figsize=(14,4.5))

    lin_vals = np.linspace(0.2,0.8,len(wfe_list))
    if c1 is None: c1 = plt.cm.Blues_r(lin_vals)
    if c2 is None: c2 = plt.cm.Reds_r(lin_vals)
    c3 = plt.cm.Purples_r(lin_vals)
    c4 = plt.cm.Greens_r(lin_vals)

    # Left plot (5-sigma sensitivities)
    ax = axes[0]

    k = key1
    curves = curves_all[k]
    obs = obs_dict[k]
    sat_rad = None if sat_dict is None else sat_dict[k]
    ax, ax2, ax3 = plot_contrasts(curves, nsig, wfe_list, obs=obs, sat_rad=sat_rad,
                                  ax=ax, colors=c1, xr=xr, yr=yr, return_axes=True)
    axes1_all = [ax, ax2, ax3]

    if key2 is not None:
        k = key2
        curves = curves_all[k] if curves_all2 is None else curves_all2[k]
        obs = None
        sat_rad = None if sat_dict is None else sat_dict[k]
        plot_contrasts(curves, nsig, wfe_list, obs=obs, sat_rad=sat_rad, 
                       ax=ax, xr=xr, yr=yr, colors=c2)

    # Planet mass locations
    if planet_patches:
        plot_planet_patches(ax, obs_dict[key1], age=age, update_title=True, av_vals=av_vals, 
                            linder=linder_models, **kwargs)
    ax.set_title('Flux Sensitivities')

    # Right plot (Converted to MJup/MEarth)
    ax = axes[1]
    k = key1
    curves = curves_all[k]
    obs = obs_dict[k]
    sat_rad = None if sat_dict is None else sat_dict[k]
    ax, ax2, ax3 = plot_contrasts_mjup(curves, nsig, wfe_list, obs=obs, age=age, sat_rad=sat_rad, 
                             ax=ax, colors=c1, xr=xr, twin_ax=True, return_axes=True,
                             linder_models=linder_models)
    axes2_all = [ax, ax2, ax3]
    
    if key2 is not None:
        k = key2
        curves = curves_all[k] if curves_all2 is None else curves_all2[k]
        obs = obs_dict[k]
        sat_rad = None if sat_dict is None else sat_dict[k]
        plot_contrasts_mjup(curves, nsig, wfe_list, obs=obs, age=age, sat_rad=sat_rad, 
                            ax=ax, colors=c2, xr=xr, linder_models=linder_models)

    mod_str = 'BEX' if linder_models else 'COND'
    ax.set_title(f'Mass Sensitivities -- {mod_str} Models')

    # Update fancy y-axis scaling on right plot
    ax = axes2_all[0]
    update_yscale(ax, yscale2, ylim=yr2)
    yr_temp = np.array(ax.get_ylim()) * 318.0
    update_yscale(axes2_all[1], yscale2, ylim=yr_temp)

    # Left legend
    nwfe = len(wfe_list)
    ax=axes[0]
    handles, labels = ax.get_legend_handles_labels()
    h1 = handles[0:nwfe][::-1]
    h2 = handles[nwfe:2*nwfe][::-1]
    h3 = handles[2*nwfe:]
    h1_t = [mpatches.Patch(color='none', label=label1)]
    h2_t = [mpatches.Patch(color='none', label=label2)]
    lfilt = obs_dict[key1].filter
    h3_t = [mpatches.Patch(color='none', label=f'{mod_str} ({lfilt})')]
    if planet_patches:
        if key2 is not None:
            handles_new = h1_t + h1 + h2_t + h2 + h3_t + h3
            ncol = 3
        else:
            h3 = handles[nwfe:]
            handles_new = h1_t + h1 + h3_t + h3
            ncol = 2
    else:
        if key2 is not None:
            handles_new = h1_t + h1 + h2_t + h2
            ncol = 2
        else:
            handles_new = h1_t + h1
            ncol = 1        
    ax.legend(ncol=ncol, handles=handles_new, loc=1, fontsize=9)

    # Right legend
    ax=axes[1]
    handles, labels = ax.get_legend_handles_labels()
    h1 = handles[0:nwfe][::-1]
    h2 = handles[nwfe:2*nwfe][::-1]
    h1_t = [mpatches.Patch(color='none', label=label1)]
    h2_t = [mpatches.Patch(color='none', label=label2)]
    if key2 is not None:
        handles_new = h1_t + h1 + h2_t + h2
        ncol = 2
    else:
        handles_new = h1_t + h1
        ncol = 1
    ax.legend(ncol=ncol, handles=handles_new, loc=1, fontsize=9)

    # Title
    name_sci = obs.sp_sci.name
    dist = obs.distance
    age_str = f'Age = {age:.0f} Myr'
    dist_str = f'Distance = {dist:.1f} pc' if dist is not None else ''
    title_str = f'{name_sci} ({age_str}, {dist_str})'

    fig.suptitle(title_str, fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(top=0.8, bottom=0.1 , left=0.05, right=0.95)

    return (fig, (axes1_all, axes2_all))
    
        
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

        # Scale to data pixelscale
        data_mod = frebin(data_mod, scale=header_mod['PIXELSCL']/header['PIXELSCL'])
        rho_mod    = dist_image(data_mod, pixscale=header['PIXELSCL'])
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

        vmax_vals = [vmax, vmax*scl1, vmax2*scl2]
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
                ax.set_ylabel('{} ({})'.format(obs.filter, exp_text))

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
    
    fname = "{}_images_{}.pdf".format(name_sci.replace(" ", ""), obs.image_mask)
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
        # Scale to data pixelscale
        data_mod = frebin(data_mod, scale=header_mod['PIXELSCL']/header['PIXELSCL'])
        rho_mod    = dist_image(data_mod, pixscale=header['PIXELSCL'])
        data_mod_r2 = data_mod*rho_mod**2
        # Ignore inner pixels
        mask_good = rho_mod > 0.15
        vmax  = np.max(data_mod[mask_good])
        vmax2 = np.max(data_mod_r2[mask_good])
        
        # Scale value for data
        im_temp = pad_or_cut_to_size(data_mod, hdu_sim[0].data.shape)
        rho_temp = dist_image(im_temp, pixscale=header['PIXELSCL'])
        mask_good = (im_temp>(0.1*vmax)) & (rho_temp>0.15)
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
    fig.suptitle('{} ({})'.format(name_sci, wfe_text), fontsize=16)
    fig.tight_layout()

    fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9, bottom=0.07 , left=0.05, right=0.97)
    
    fname = "{}_images_{}.pdf".format(name_sci.replace(" ", ""), obs.image_mask)
    if save_fig: 
        fig.savefig(outdir+fname)
        
    if return_fig_axes:
        return fig, axes

def plot_spectrum(src, bp_list, sptype=None, src_ref=None,
                  return_fig_axes=False, save_fig=False, outdir='', 
                  xr=[2.5, 5.5], **kwargs):

    name = src.name
    sp = src.sp_model

    # Plot spectra 
    fig, axes = plt.subplots(1,2, figsize=(12,4))

    ax = axes[0]
    src.plot_SED(ax=axes[0], xr=[0.5,30])

    spt_label = '' if sptype is None else f' ({sptype})'
    ax.set_title(f'{name} SED{spt_label}')
    ax.set_xlabel(r'Wavelength ($\mathdefault{\mu m}$)')
    # ax.set_ylim([0.5,20])
    # ax.set_xscale('linear')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax = axes[1]

    bp = bp_list[-1]
    w = sp.wave / 1e4
    o = Observation(sp, bp, binset=bp.waveset)
    sp.convert('photlam')
    f = sp.flux / sp.flux[(w>xr[0]) & (w<xr[1])].max()

    ind = (w>=xr[0]) & (w<=xr[1])
    ax.plot(w[ind], f[ind], lw=1, label=sp.name)
    ax.set_ylabel('Normalized Flux (photons/s/wave)')
    sp.convert('flam')

    if src_ref is not None:
        sp_ref = src_ref.sp_model
        sp_ref.convert('photlam')
        w_ref = sp_ref.wave / 1e4
        f_ref = sp_ref.flux / sp_ref.flux[(w_ref>xr[0]) & (w_ref<xr[1])].max()
        ind = (w_ref>=xr[0]) & (w_ref<=xr[1])
        label = f"{sp_ref.name} (Ref)"
        ax.plot(w_ref[ind], f_ref[ind], lw=1, label=label, color='C3', alpha=0.75)
        sp_ref.convert('flam')

    ax.set_xlim(xr)
    ax.set_xlabel(r'Wavelength ($\mathdefault{\mu m}$)')
    ax.set_title(f'{sp.name} Spectrum and Bandpasses')

    # Overplot Filter Bandpass
    ax2 = ax.twinx()
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, bp in enumerate(bp_list):
        ax2.plot(bp.wave/1e4, bp.throughput, color=cols[i+1], label=bp.name+' Bandpass')
    ax2.set_ylim([0,1.1*ax2.get_ylim()[1]])
    ax2.set_xlim(xr)
    ax2.set_ylabel('Bandpass Throughput')

    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    fig.tight_layout()

    if save_fig: 
        name_str = name.replace(' ','')
        fig_path = os.path.join(outdir, f'{name_str}_SED.pdf')
        fig.savefig(fig_path, bbox_inches='tight')

    if return_fig_axes:
        return fig, axes