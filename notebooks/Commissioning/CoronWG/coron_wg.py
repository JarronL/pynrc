# Import the usual libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from webbpsf_ext.bandpasses import read_filter

import pynrc
from pynrc import nrc_utils

from pynrc.maths.image_manip import fourier_imshift, fshift, frebin
from pynrc.maths.coords import rtheta_to_xy, xy_to_rtheta, dist_image
from pynrc.nb_funcs import plot_hdulist
from pynrc.nrc_utils import radial_std, pad_or_cut_to_size, align_LSQ

from webbpsf_ext.opds import OPDFile_to_HDUList
from webbpsf_ext.webbpsf_ext_core import nrc_mask_trans

from astropy.io import fits

from copy import deepcopy

from scipy.interpolate import interp1d
from scipy.ndimage import zoom

# Progress bar
from tqdm.auto import tqdm, trange

# Disable informational messages and only include warnings and higher
pynrc.setup_logging(level='WARN')

opd_dir = 'NIRCAM_OPDS/'

# Scenario to consider [best, nominal, requirements]
scenarios = ['Best Case', 'Nominal', 'Requirements']
# LOS Jitter [2.5, 3.8, 5.8] per axis
# Samples from random distribution, 1 sample per time step. 
# hdul_jitter = fits.open(opd_dir + 'LOS_JITTER.fits')
jitter_modes = [2.5, 3.8, 5.8, 0]
# Target Acqs
# Three values for best, nominal, and requirements: 6.2568703,  8.759604 , 12.513741
#   - these are per axis
tacq_ref = np.array([6.2568703,  8.759604 , 12.513741, 0])
tacq_ref = tacq_ref.repeat(2).reshape([-1,2])

tacq_sci = 0

# Define 2MASS Ks bandpass and source information
bp_k = pynrc.bp_2mass('k')

# Solar analog at 10pc
# https://iopscience.iop.org/article/10.3847/1538-4365/aabfdf#apjsaabfdft3
# Science      source,  dist, age, sptype, Teff, [Fe/H], log_g, mag, band
args_sources = ('G2V', 10.0, 100,  'G2V', 5777, 0, 4.43, 3.27, bp_k)
# References     source,  sptype, Teff, [Fe/H], log_g, mag, band
ref_sources = ('G2V Ref', 'G2V', 5777, 0, 4.43, 3.27, bp_k)

# Get science spectrum
name_sci, dist_sci, age, spt_sci, Teff_sci, feh_sci, logg_sci, mag_sci, bp_sci = args_sources
args = (spt_sci, mag_sci, 'vegamag', bp_sci)
kwargs = {'Teff':Teff_sci, 'metallicity':feh_sci, 'log_g':logg_sci}
sp_sci = pynrc.stellar_spectrum(*args, **kwargs)
sp_sci.name = name_sci

# Do the same for the reference source
name_ref, spt_ref, Teff_ref, feh_ref, logg_ref, mag_ref, bp_ref = ref_sources
args = (spt_ref, mag_ref, 'vegamag', bp_ref)
kwargs = {'Teff':Teff_ref, 'metallicity':feh_ref, 'log_g':logg_ref}
sp_ref = pynrc.stellar_spectrum(*args, **kwargs)
sp_ref.name = name_ref

# Load OPD and other information
tvals_sec = fits.getdata(opd_dir + 'time_vector.fits') * 60

# Static OPDs
hdul_opds_static = OPDFile_to_HDUList(opd_dir + 'STATIC_NIRCAM-A_INPUT.fits')

# Thermal, frill, and IEC
hdul_opds_thermal = fits.open(opd_dir + 'TD_NIRCAM.fits')
hdul_opds_frill   = fits.open(opd_dir + 'FRILLCO_NIRCAM.fits')
hdul_opds_iec     = fits.open(opd_dir + 'IEC_NIRCAM.fits')

# Convert OPDS to units of microns
for hdul in [hdul_opds_static, hdul_opds_thermal, hdul_opds_frill, hdul_opds_iec]:
    for hdu in hdul:
        hdu.data *= 1e6
        hdu.header['BUNIT'] = 'micron'

def run_obs(filt, mask, pupil, imode=0, imode_tacq=None, imode_jitt=None, 
            jitt_groups=False, save=True, verbose=False):

    if imode_jitt is None:
        imode_jitt = imode

    # Define detector configuration and PSF simulation
    bp = pynrc.read_filter(filt)

    channel = 'SW' if bp.avgwave()/1e4 < 2.4 else 'LW'
    if 'LW' in channel:
        wind_mode, subsize = ('WINDOW', 320)
        fov_pix, oversample = (160, 2)
    else:
        wind_mode, subsize = ('WINDOW', 640)
        fov_pix, oversample = (320, 2)

    # Science configuration
    nrc_sci = pynrc.NIRCam(filter=filt, image_mask=mask, pupil_mask=pupil,
                           wind_mode=wind_mode, xpix=subsize, ypix=subsize, 
                           fov_pix=fov_pix, oversample=oversample)
    # Reference configuration
    nrc_ref = pynrc.NIRCam(filter=filt, image_mask=mask, pupil_mask=pupil,
                        wind_mode=wind_mode, xpix=subsize, ypix=subsize, 
                        fov_pix=fov_pix, oversample=oversample)

    # Shooting for ~3600 sec of acquisition time
    if 'LW' in channel:
        nrc_sci.update_detectors(read_mode='MEDIUM8', ngroup=7, nint=50, verbose=verbose)
        nrc_ref.update_detectors(read_mode='MEDIUM8', ngroup=7, nint=50, verbose=False)
    else:
        nrc_sci.update_detectors(read_mode='BRIGHT1', ngroup=9, nint=50, verbose=verbose)
        nrc_ref.update_detectors(read_mode='BRIGHT1', ngroup=9, nint=50, verbose=False)

    # Turn off jitter estimation
    for nrc in [nrc_sci, nrc_ref]:
        nrc.options['jitter'] = None
        nrc.options['jitter_sigma'] = 0
    nrc_sci.options['source_offset_r'] = 0
    nrc_sci.options['source_offset_theta'] = 0

    # Create a time series of frames
    det = nrc_sci.Detector
    nint = det.multiaccum.nint
    ngroup = det.multiaccum.ngroup

    # Group times within a given integration
    tg_arr = det.times_group_avg

    # Repeat nint and add total integration time
    tg_all = tg_arr.reshape([-1,1]).repeat(nint, axis=1)
    tg_all = tg_all.transpose()
    tg_all = tg_all + np.arange(nint).reshape([-1,1]) * det.time_total_int2

    # Time step of integration (average)
    tint_all = np.mean(tg_all, axis=1)

    # Print configuration(s) info
    print('WFE Drift: ', scenarios[imode])
    print('Jitter:    ', jitter_modes[imode_jitt])
    if imode_tacq is None:
        ta_iter = np.arange(3)
    else:
        ta_iter = imode_tacq
    print('Target Aq: ', tacq_ref[ta_iter,0])

    # Create drifted OPDs for each integration
    # Interpolate OPDS for each integration
    dopds = np.zeros([nint,1024,1024])
    dopds_ref = np.zeros([nint,1024,1024])
    for hdul in tqdm([hdul_opds_thermal, hdul_opds_frill, hdul_opds_iec], leave=False):
        # Flip along y-axis for correct orientation
        opds = hdul[imode].data[:,::-1,:]

        # Interpolate science dOPDs
        func = interp1d(tvals_sec, opds, axis=0, kind='linear', bounds_error=True)
        opds_interp = func(tint_all)
        dopds += frebin(opds_interp, dimensions=1024, total=False)

        # Interpolate reference dOPDs
        if hdul is hdul_opds_iec:
            # For IEC, start 3600 sec later
            opds_interp = func(tint_all + nrc_sci.Detector.time_total)
        else:
            # Flip sign for thermal and frill for reference obs
            opds_interp = func(tint_all)
            opds_interp *= -1
        dopds_ref += frebin(opds_interp, dimensions=1024, total=False)
    
    test1 = np.mean(dopds_ref + dopds[-1], axis=0) - np.mean(dopds, axis=0)
    print(calc_rms(test1) * 1000)

    # Create random jitter realizations for each group timestep [nint, ngroup, 2]
    jitter_sig = jitter_modes[imode_jitt]
    jitter_rand = np.random.normal(scale=jitter_sig, size=(nint,ngroup,2)) / 1000
    jitter_rand_ref = np.random.normal(scale=jitter_sig, size=(nint,ngroup,2)) / 1000

    # Create slopes for all integrations
    im_slope_sci = []
    for i in trange(nint, desc='Sci INT'):
        # Create copy of OPD and add delta
        opd_int = deepcopy(hdul_opds_static)
        opd_int[0].data += dopds[i]

        # Initial TA Offsets
        if tacq_sci==0:
            nrc_ref.options['source_offset_r']     = 0
            nrc_ref.options['source_offset_theta'] = 0
        else:
            ta_x, ta_y  = -1 * tacq_sci / 1000  # arcsec
            ta_r, ta_th = xy_to_rtheta(ta_x, ta_y)
            nrc_ref.options['source_offset_r']     = ta_r
            nrc_ref.options['source_offset_theta'] = ta_th

        im_slope = gen_slope_image(nrc_sci, opd_int, jitter_rand[i], 
                                   sp=sp_sci, jitt_groups=jitt_groups)
        im_slope_sci.append(im_slope)
        
    im_slope_sci = np.array(im_slope_sci)

    # Save final OPD for input into reference
    opd_sci_last = deepcopy(opd_int)

    if imode_tacq is None:
        ta_iter = trange(4, desc='Ref TA')
    else:
        ta_iter = [imode_tacq]

    for i_ta in ta_iter:
        # Initial TA Offsets
        ta_x, ta_y  = tacq_ref[i_ta] / 1000  # arcsec
        ta_r, ta_th = xy_to_rtheta(ta_x, ta_y)
        nrc_ref.options['source_offset_r']     = ta_r
        nrc_ref.options['source_offset_theta'] = ta_th
        im_slope_ref = get_ref_slopes(nrc_ref, opd_sci_last, dopds_ref, 
                                      jitter_rand_ref, jitt_groups=jitt_groups)

        # Realign reference images
        tax_pix = ta_x / nrc_ref.pixelscale
        tay_pix = ta_y / nrc_ref.pixelscale
        im_slope_ref_sh = fourier_imshift(im_slope_ref, -tax_pix, -tay_pix, pad=False)

        im_sci = np.mean(im_slope_sci, axis=0)
        im_ref = np.mean(im_slope_ref, axis=0)
        im_ref_sh = np.mean(im_slope_ref_sh, axis=0)

        diff1 = im_sci - im_ref
        diff2 = im_sci - im_ref_sh

        plot_images(nrc_sci, im_sci, diff1, diff2,
                    imode, jitter_sig, i_ta, save=save)

        calc_contrast(diff2, im_sci, nrc_sci, imode, jitter_sig, i_ta, 
                  nsig=5, save=save, plot=False)


def get_ref_slopes(nrc_ref, opd_sci_last, dopds_ref, jitter_rand_ref, jitt_groups=False):

    det = nrc_ref.Detector
    nint = det.multiaccum.nint

    # Create slopes for all integrations
    im_slope_ref = []
    for i in trange(nint, leave=False, desc='Ref INT'):
        # Create copy of OPD and add delta
        opd_int = deepcopy(opd_sci_last)
        opd_int[0].data += dopds_ref[i]

        im_slope = gen_slope_image(nrc_ref, opd_int, jitter_rand_ref[i], 
                                   sp=sp_ref, jitt_groups=jitt_groups)
        im_slope_ref.append(im_slope)
        
    return np.array(im_slope_ref)


def plot_spec(filt, pupil=None, mask=None, save=False):

    # Plot the two spectra
    fig, ax = plt.subplots(1,1, figsize=(8,5))

    xr = [2.5,5.5]

    for sp in [sp_sci, sp_ref]:
        w = sp.wave / 1e4
        ind = (w>=xr[0]) & (w<=xr[1])
        sp.convert('Jy')
        f = sp.flux / np.interp(4.0, w, sp.flux)
        ax.semilogy(w[ind], f[ind], lw=1.5, label=sp.name)
        ax.set_ylabel(r'$F_{\nu}~/~F_{\nu}~(\lambda=4\mathdefault{\mu m})$ [Jy]')
        sp.convert('flam')

    ax.set_xlim(xr)
    ax.set_xlabel(r'Wavelength [$\mathdefault{\mu m}$]')
    ax.set_title('Spectral Sources')

    # Overplot Filter Bandpass
    bp = pynrc.read_filter(filt, pupil=pupil, mask=mask)
    ax2 = ax.twinx()
    ax2.plot(bp.wave/1e4, bp.throughput, color='C2', label=bp.name+' Bandpass')
    ax2.set_ylim([0,0.8])
    ax2.set_xlim(xr)
    ax2.set_ylabel('Bandpass Throughput')

    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    fig.tight_layout()

    if save:
        fig.savefig('output/' + f'spectra_{filt}.pdf')


def plot_images(nrc_sci, im_sci, diff1, diff2,
                imode, jitter_sig, imode_tacq, save=True):
    
    hdul_sci   = fits.HDUList(fits.PrimaryHDU(im_sci))
    hdul_diff1 = fits.HDUList(fits.PrimaryHDU(diff1))
    hdul_diff2 = fits.HDUList(fits.PrimaryHDU(diff2))

    # Get min/max for plotting difference images
    im = hdul_diff2[0].data
    im_rho = dist_image(im, pixscale=nrc_sci.pixelscale)
    ind = (im_rho>1) & (im_rho<3)
    vmax_diff = np.max(np.abs(im[ind]))
    vmin_diff = -1 * vmax_diff

    fig, axes = plt.subplots(1,3,figsize=(14,4.25))

    titles = ['Raw PSF', 'Differenced', 'Aligned Diff']
    for i, hdul in enumerate([hdul_sci, hdul_diff1, hdul_diff2]):
        hdul[0].header['PIXELSCL'] = nrc_sci.pixelscale
        ax = axes[i]

        if i==0:
            vmin = vmax = None
            axes_color='white'
            cmap = None
        else:
            vmin = vmin_diff
            vmax = vmax_diff
            axes_color='k'
            cmap = 'RdBu_r'

        plot_hdulist(hdul, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, axes_color=axes_color)

        hdu = hdul[0]
        im_rho = dist_image(hdu.data, pixscale=hdu.header['PIXELSCL'])

        rms = np.std(hdu.data[im_rho<2])
        rms_str = f' (RMS={rms:.2f})'
        ax.set_title(titles[i] + rms_str)


    ta_ref = tacq_ref[imode_tacq][0]
    ta_sci = tacq_sci
    ta_val = np.sqrt(ta_ref**2 + ta_sci**2)
    title = f'{scenarios[imode]} $\Delta$WFE; Jitter = {jitter_sig:.1f} mas; TA = {ta_val:.1f} mas'
    fig.suptitle(title, fontsize=16)

    fig.tight_layout()

    if save:
        fname0 = f'{nrc_sci.filter}_{nrc_sci.image_mask}'
        fname1 = f'{fname0}_{scenarios[imode]}_jit{jitter_sig:.1f}mas_tacq{ta_val:.1f}mas'
        fname1 = fname1.replace(' ','')

        fname = f'{fname1}_images.pdf'
        fig.savefig('output/'+fname)

        # Save files
        f1 = f'{fname1}_psf.fits'
        f2 = f'{fname1}_diff1.fits'
        f3 = f'{fname1}_diff2.fits'

        hdul_sci.writeto('output/' + f1, overwrite=True)
        hdul_diff1.writeto('output/' + f2, overwrite=True)
        hdul_diff2.writeto('output/' + f3, overwrite=True)

def calc_contrast(diff2, im_sci, nrc_sci, imode, jitter_sig, imode_tacq, 
                  nsig=5, save=True, plot=False):
    
    pixscale = nrc_sci.pixelscale

    # Get standard deviation at each radial bin
    rr, stds = radial_std(diff2, pixscale=pixscale)

    # Bin to detector-sampled data
    xpix, ypix = (nrc_sci.det_info['xpix'], nrc_sci.det_info['ypix'])

    ny, nx = (ypix, xpix)
    yv = (np.arange(ny) - ny/2) * pixscale
    xv = np.zeros_like(yv)

    # Get mask transmission
    trans = nrc_mask_trans(nrc_sci.image_mask, xv, yv)
    # Linear combination of min/max to determine PSF max value at given distance
    # Get a and b values for each position
    avals = trans**2
    bvals = 1 - avals

    # Get stellar source
    star_flux = spec_flux(sp_sci, nrc_sci.filter, units='counts')
    star_mag = spec_flux(sp_sci, nrc_sci.filter, units='vegamag')
        
    psf_cen = pad_or_cut_to_size(im_sci / star_flux, 20)
    psf_off = nrc_sci.calc_psf_from_coeff(use_bg_psf=True, 
                                          return_oversample=False, 
                                          return_hdul=False)

    psf_cen_max = psf_cen.max()
    psf_off_max = psf_off.max()

    # Linear combination
    psf_max = avals * psf_off_max + bvals * psf_cen_max
    # Interpolate values at rr locations
    psf_max = 10**np.interp(rr, yv, np.log10(psf_max))
    # Fix anything outside of bounds
    if rr.max() > 10:
        psf_max[rr>10] = psf_max[(rr>5) & (rr<10)].max()

    # Count rate necessary to obtain some nsig
    texp  = nrc_sci.multiaccum_times['t_exp']
    p     = 1 / texp
    crate = (p*nsig**2 + nsig*np.sqrt((p*nsig)**2 + 4*stds**2)) / 2
    # Get total count rate
    crate /= psf_max

    # Get contrast
    contrast = crate / star_flux
    sen_mag = star_mag - 2.5*np.log10(contrast)

    # Background sensitivity
    sen, _ = nrc_sci.sensitivity(sp=sp_sci, nsig=nsig, units='vegamag')
    bg_sen = 10**((star_mag - sen['sensitivity']) / 2.5)
    bg_sen_arr = np.ones_like(rr)*bg_sen
    
    # Save arrays to disk
    ta_ref = tacq_ref[imode_tacq][0]
    ta_sci = tacq_sci
    ta_val = np.sqrt(ta_ref**2 + ta_sci**2)

    if save:
        fname0 = f'{nrc_sci.filter}_{nrc_sci.image_mask}'
        fname = f'{fname0}_{scenarios[imode]}_jit{jitter_sig:.1f}mas_tacq{ta_val:.1f}mas_contrast.npz'
        fname = fname.replace(' ','')
        np.savez('output/' + fname, rr=rr, contrast=contrast, sen_mag=sen_mag, bg_sen_arr=bg_sen_arr)

    if plot:
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.semilogy(rr, contrast)
        ax.plot(rr, bg_sen_arr, ls='--')
        fig.tight_layout()

    # Load by:
    #   res = np.load('output/' + fout)
    # Then access:
    #   rr = res['rr']
    
    # return rr, contrast, sen_mag, bg_sen_arr

def gen_slope_image(nrc, opd, jitter_vals, sp=None, jitt_groups=False):
    """
    Wrapper to creae a slope image given some OPD and
    jitter information. 
    """

    # Initial values
    opd_init = nrc.pupilopd
    r_init  = nrc.options.get('source_offset_r', 0)
    th_init = nrc.options.get('source_offset_theta', 0)
    xoff_init, yoff_init = rtheta_to_xy(r_init, th_init)
    
    # Update OPD info
    nrc.pupilopd = opd

    det = nrc.Detector
    ma = det.multiaccum
    ngroup = ma.ngroup
    
    # Generate offset images for each group
    if jitt_groups:
        psf_arr = []
        for j in trange(ngroup, leave=False, desc='Groups'):
            
            # Add jitter to initial values
            xoff, yoff = jitter_vals[j,:]
            r_off, th_off = xy_to_rtheta(xoff+xoff_init, yoff+yoff_init)
            
            # Update offset values
            nrc.options['source_offset_r'] = r_off
            nrc.options['source_offset_theta'] = th_off
            hdul_out = nrc.calc_psf(sp=sp)
            psf_arr.append(hdul_out[1].data) # Non-distorted version
            
        # Get average slope for this integration
        psf_int = np.mean(np.array(psf_arr), axis=0)
    else:
        # Add jitter to initial values
        xoff, yoff = jitter_vals[0,:]
        r_off, th_off = xy_to_rtheta(xoff+xoff_init, yoff+yoff_init)
        # Update offset values
        nrc.options['source_offset_r'] = r_off
        nrc.options['source_offset_theta'] = th_off
        # Calculate PSFs
        hdul_out = nrc.calc_psf(sp=sp)
        psf_int = hdul_out[1].data # Non-distorted version
    
    # Return to initial conditions
    nrc.pupilopd = opd_init
    nrc.options['source_offset_r'] = r_init
    nrc.options['source_offset_theta'] = th_init
    
    return psf_int


def calc_rms(im):
    ind = (im != 0) & (np.isfinite(im))
    rms = np.sqrt((im[ind] ** 2).mean())
    return rms


def spec_flux(sp, filter, units='counts'):

    bp = pynrc.read_filter(filter)
    obs = nrc_utils.S.Observation(sp, bp, binset=bp.wave)
    return obs.effstim(units)