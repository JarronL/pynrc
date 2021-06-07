"""pyNRC utility functions"""

from __future__ import absolute_import, division, print_function, unicode_literals

# The six library is useful for string compatibility
import six
import os, re

# Import libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

on_rtd = os.environ.get('READTHEDOCS') == 'True'
# Update matplotlib settings
rcvals = {'xtick.minor.visible': True, 'ytick.minor.visible': True,
          'xtick.direction': 'in', 'ytick.direction': 'in',
          'xtick.top': True, 'ytick.right': True, 'font.family': ['serif'],
          'xtick.major.size': 6, 'ytick.major.size': 6,
          'xtick.minor.size': 3, 'ytick.minor.size': 3,
          'image.interpolation': 'nearest', 'image.origin': 'lower',
          'figure.figsize': [8,6], 'mathtext.fontset':'cm'}#,
          #'text.usetex': True, 'text.latex.preamble': ['\usepackage{gensymb}']}
if not on_rtd:
    matplotlib.rcParams.update(rcvals)
    cmap_pri, cmap_alt = ('viridis', 'gist_heat')
    matplotlib.rcParams['image.cmap'] = cmap_pri if cmap_pri in plt.colormaps() else cmap_alt


import datetime, time
import sys, platform
import multiprocessing as mp
import traceback

from astropy.io import fits, ascii
from astropy.table import Table
from astropy.time import Time
# from astropy import units

#from scipy.optimize import least_squares#, leastsq
#from scipy.ndimage import fourier_shift
from scipy.interpolate import griddata, RegularGridInterpolator, interp1d
from numpy.polynomial import legendre

from . import conf
from .logging_utils import setup_logging

from .maths import robust
from .maths.fast_poly import *
from .maths.image_manip import *
from .maths.coords import *
# from .maths.image_manip import frebin, fshift, pad_or_cut_to_size
# from .maths.image_manip import hist_indices, binned_statistic
# from .maths.coords import dist_image, xy_to_rtheta, rtheta_to_xy, xy_rot
# from .maths.coords import det_to_sci, sci_to_det, plotAxes

###########################################################################
#    Logging info
###########################################################################

import logging
_log = logging.getLogger('pynrc')

###########################################################################
#    WebbPSF
###########################################################################

try:
    import webbpsf_ext
except ImportError:
    raise ImportError('webbpsf_ext is not installed. pyNRC depends on its inclusion.')

# Some useful functions for displaying and measuring PSFs
import webbpsf, poppy
from poppy import (radial_profile, measure_radial, measure_fwhm, measure_ee)
from poppy import (measure_sharpness, measure_centroid, measure_strehl)

import pysynphot as S
# Extend default wavelength range to 5.6 um
S.refs.set_default_waveset(minwave=500, maxwave=56000, num=10000.0, delta=None, log=False)
# JWST 25m^2 collecting area
# Flux loss from masks and occulters are taken into account in WebbPSF
# S.refs.setref(area = 25.4e4) # cm^2
S.refs.setref(area = 25.78e4) # cm^2 according to jwst_pupil_RevW_npix1024.fits.gz

# The following won't work on readthedocs compilation
if not on_rtd:
    # Grab WebbPSF assumed pixel scales
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)
    nc_temp = webbpsf_ext.NIRCam_ext()
    setup_logging(log_prev, verbose=False)

    pixscale_SW = nc_temp._pixelscale_short
    pixscale_LW = nc_temp._pixelscale_long
    del nc_temp

    _jbt_exists = True
    try:
        from jwst_backgrounds import jbt
    except ImportError:
        _log.info("  jwst_backgrounds is not installed and will not be used for bg estimates.")
        _jbt_exists = False



#__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
#__location__ += '/'

__epsilon = np.finfo(float).eps


###########################################################################
#
#    Pysynphot Bandpasses
#
###########################################################################

from webbpsf_ext.bandpasses import bp_igood, bp_wise, bp_2mass
from webbpsf_ext.bandpasses import nircam_filter as read_filter


###########################################################################
#
#    Sensitivities and Saturation Limits
#
###########################################################################

from webbpsf_ext.bandpasses import nircam_grism_res as grism_res
from webbpsf_ext.bandpasses import nircam_grism_wref as grism_wref
from webbpsf_ext.maths import radial_std

def channel_select(bp):
    """Select wavelength channel

    Based on input bandpass, return the pixel scale, dark current, and
    excess read noise parameters. These values are typical for either
    a SW or LW NIRCam detector.

    Parameters
    ----------
    bp : :mod:`pysynphot.obsbandpass`
        NIRCam filter bandpass.
    """

    if bp.avgwave()/1e4 < 2.3:
        pix_scale = pixscale_SW # pixel scale (arcsec/pixel)
        idark = 0.003      # dark current (e/sec)
        pex = (1.0,5.0)
    else:
        pix_scale = pixscale_LW
        idark = 0.03
        pex = (1.5,10.0)

    return (pix_scale, idark, pex)

def place_grismr_tso(waves, imarr, siaf_ap, wref=None, im_coords='sci'):
    """
    Shift image such that undeviated wavelength sits at the
    SIAF aperture reference location.
    """
    
    from .maths.coords import det_to_sci

    if len(imarr.shape) > 2:
        nz, ny_in, nx_in = imarr.shape
    else:
        nz = 1
        ny_in, nx_in = imarr.shape
        imarr = imarr.reshape([nz,ny_in,nx_in])
    
    # Convert to sci coordinates
    if im_coords=='det':
        det_name = siaf_ap.AperName[3:5]
        imarr = det_to_sci(imarr, det_name)

    # Determine reference wavelength
    if wref is None:
        if 'GRISMC' in siaf_ap.AperName:
            pupil = 'GRISMC'
        elif 'GRISM' in siaf_ap.AperName:
            pupil = 'GRISMR'
        else: # generic grism
            pupil = 'GRISM'
        module = 'A' if 'NRCA' in siaf_ap.AperName else 'B'
        wref = grism_wref(pupil, module)

    # Get reference coordinates
    yref, xref = (siaf_ap.YSciRef, siaf_ap.XSciRef)
    
    # Final image size
    ny_out, nx_out = (siaf_ap.YSciSize, siaf_ap.XSciSize)
    
    # Empirically determine shift value in dispersion direction
    wnew_temp = pad_or_cut_to_size(waves, nx_out)
    
    # Index of reference wavelength associated with ref pixel
    ind = (wnew_temp>wref-0.01) & (wnew_temp<wref+0.01)
    xnew_temp = np.interp(wref, wnew_temp[ind], np.arange(nx_out)[ind])
    xoff = xref - xnew_temp
    
    # Move to correct position in y
    yoff = yref - (int(ny_out/2) - 1)
    # if np.mod(ny_in,2)==0: # If even, shift by half a pixel?
    #     yoff = yoff + 0.5
    
    imarr = pad_or_cut_to_size(imarr, (ny_out,nx_out), offset_vals=(yoff,xoff), fill_val=np.nan)
    waves = pad_or_cut_to_size(waves, nx_out, offset_vals=xoff, fill_val=np.nan)
    
    # Remove NaNs
    ind_nan = np.isnan(imarr)
    imarr[ind_nan] = np.min(imarr[~ind_nan])
    # Remove NaNs
    # Fill in with wavelength solution (linear extrapolation)
    ind_nan = np.isnan(waves)
    # waves[ind_nan] = 0
    arr = np.arange(nx_out)
    cf = jl_poly_fit(arr[~ind_nan], waves[~ind_nan])
    waves[ind_nan] = jl_poly(arr[ind_nan], cf)

    return waves, imarr



def get_SNR(filter_or_bp, pupil=None, mask=None, module='A', pix_scale=None,
    sp=None, tf=10.737, ngroup=2, nf=1, nd2=0, nint=1,
    coeff=None, coeff_hdr=None, fov_pix=11, oversample=4, quiet=True, **kwargs):
    """SNR per pixel

    Obtain the SNR of an input source spectrum with specified instrument setup.
    This is simply a wrapper for bg_sensitivity(forwardSNR=True).
    """

    return bg_sensitivity(filter_or_bp, \
        pupil=pupil, mask=mask, module=module, pix_scale=pix_scale, \
        sp=sp, tf=tf, ngroup=ngroup, nf=nf, nd2=ngroup, nint=nint, \
        coeff=coeff, coeff_hdr=None, fov_pix=fov_pix, oversample=oversample, \
        quiet=quiet, forwardSNR=True, **kwargs)

def _mlim_helper(sub_im, mag_norm=10, mag_arr=np.arange(5,35,1),
    nsig=5, nint=1, snr_fact=1, forwardSNR=False, **kwargs):
    """Helper function for determining grism sensitivities"""

    sub_im_sum = sub_im.sum()

    # Just return the SNR for the input sub image
    if forwardSNR:
        im_var = pix_noise(fsrc=sub_im, **kwargs)**2
        ns_sum = np.sqrt(np.sum(im_var) / nint)
        return snr_fact * sub_im_sum / ns_sum

    fact_arr = 10**((mag_arr-mag_norm)/2.5)
    snr_arr = []

    for f in fact_arr:
        im = sub_im / f
        im_var = pix_noise(fsrc=im, **kwargs)**2
        im_sum = sub_im_sum / f
        ns_sum = np.sqrt(np.sum(im_var) / nint)

        snr_arr.append(im_sum / ns_sum)
    snr_arr = snr_fact*np.asarray(snr_arr)
    return np.interp(nsig, snr_arr[::-1], mag_arr[::-1])

def bg_sensitivity(filter_or_bp, pupil=None, mask=None, module='A', pix_scale=None,
    sp=None, units=None, nsig=10, tf=10.737, ngroup=2, nf=1, nd2=0, nint=1,
    coeff=None, coeff_hdr=None, fov_pix=11, oversample=4, quiet=True, forwardSNR=False,
    offset_r=0, offset_theta=0, return_image=False, image=None,
    cr_noise=True, dw_bin=None, ap_spec=None, rad_EE=None, **kwargs):
    """Sensitivity Estimates

    Estimates the sensitivity for a set of instrument parameters.
    By default, a flat spectrum is convolved with the specified bandpass.
    For imaging, this function also returns the surface brightness sensitivity.

    The number of photo-electrons are computed for a source at some magnitude
    as well as the noise from the detector readout and some average zodiacal
    background flux. Detector readout noise follows an analytical form that
    matches extensive long dark observations during cryo-vac testing.

    This function returns the n-sigma background limit in units of uJy (unless
    otherwise specified; valid units can be found on the Pysynphot webpage at
    https://pysynphot.readthedocs.io/).

    For imaging, a single value is given assuming aperture photometry with a
    radius of ~1 FWHM rounded to the next highest integer pixel (or 2.5 pixels,
    whichever is larger). For spectral observations, this function returns an
    array of sensitivities at 0.1um intervals with apertures corresponding to
    2 spectral pixels and a number of spatial pixels equivalent to 1 FWHM rounded
    to the next highest integer (minimum of 5 spatial pixels).

    Parameters
    ==========

    Instrument Settings
    -------------------
    filter_or_bp : Either the name of the filter or pre-computed Pysynphot bandpass.
    pupil  : NIRCam pupil elements such as grisms or lyot stops
    mask   : Specify the coronagraphic occulter (spots or bar)
    module : 'A' or 'B'
    pix_scale : Pixel scale in arcsec/pixel

    Spectrum Settings
    -------------------
    sp         : A pysynphot spectral object to calculate sensitivity
                 (default: Flat spectrum in photlam)
    nsig       : Desired nsigma sensitivity
    units      : Output units (defaults to uJy for grisms, nJy for imaging)
    forwardSNR : Find the SNR of the input spectrum instead of determining sensitivity.

    Ramp Settings
    -------------------
    tf     : Time per frame
    ngroup : Number of groups per integration
    nf     : Number of averaged frames per group
    nd2    : Number of dropped frames per group
    nint   : Number of integrations/ramps to consider

    PSF Information
    -------------------
    coeff : A cube of polynomial coefficients for generating PSFs. This is
            generally oversampled with a shape (fov_pix*oversamp, fov_pix*oversamp, deg).
            If not set, this will be calculated using :func:`gen_psf_coeff`.
    coeff_hdr    : Header associated with coeff cube.
    fov_pix      : Number of detector pixels in the image coefficient and PSF.
    oversample   : Factor of oversampling of detector pixels.
    offset_r     : Radial offset of the target from center.
    offset_theta : Position angle for that offset, in degrees CCW (+Y).

    Misc.
    -------------------
    image        : Explicitly pass image data rather than calculating from coeff.
    return_image : Instead of calculating sensitivity, return the image calced from coeff.
                   Useful if needing to calculate sensitivities for many different settings.
    rad_EE       : Extraction aperture radius (in pixels) for imaging mode.
    dw_bin       : Delta wavelength to calculate spectral sensitivities (grisms & DHS).
    ap_spec      : Instead of dw_bin, specify the spectral extraction aperture in pixels.
                   Takes priority over dw_bin. Value will get rounded up to nearest int.
    cr_noise     : Include noise from cosmic ray hits?

    Keyword Args
    -------------------
    zodi_spec     - zfact, ra, dec, thisday, [locstr, year, day]
    pix_noise     - rn, ktc, idark, and p_excess
    gen_psf_coeff - npsf and ndeg
    read_filter   - ND_acq
    """

    # PSF coefficients
    from pynrc.psfs import gen_image_coeff

    grism_obs = (pupil is not None) and ('GRISM' in pupil)
    dhs_obs   = (pupil is not None) and ('DHS'   in pupil)
    coron_obs = (pupil is not None) and ('LYOT'   in pupil)

    # Get filter throughput and create bandpass
    if isinstance(filter_or_bp, six.string_types):
        filter = filter_or_bp
        bp = read_filter(filter, pupil=pupil, mask=mask, module=module, **kwargs)
    else:
        bp = filter_or_bp
        filter = bp.name
    waveset = np.copy(bp.wave)

    # If not set, select some settings based on filter (SW or LW)
    args = channel_select(bp)
    if pix_scale is None: pix_scale = args[0] # Pixel scale (arcsec/pixel)

    # Spectrum and bandpass to report magnitude that saturates NIRCam band
    if sp is None:
        sp = S.ArraySpectrum(waveset, 0*waveset + 10.)
        sp.name = 'Flat spectrum in photlam'

    if forwardSNR:
        sp_norm = sp
    else:
        # Renormalize to 10th magnitude star
        mag_norm = 10
        sp_norm = sp.renorm(mag_norm, 'vegamag', bp)
        sp_norm.name = sp.name

    # Zodiacal Light Stuff
    sp_zodi = zodi_spec(**kwargs)
    obs_zodi = S.Observation(sp_zodi, bp, binset=waveset)
    fzodi_pix = obs_zodi.countrate() * (pix_scale/206265.0)**2  # e-/sec/pixel
    # Collecting area gets reduced for coronagraphic observations
    # This isn't accounted for later, because zodiacal light doesn't use PSF information
    if coron_obs: fzodi_pix *= 0.19

    # The number of pixels to span spatially for WebbPSF calculations
    fov_pix = int(fov_pix)
    oversample = int(oversample)

    # Generate the PSF image for analysis.
    # This process can take a while if being done over and over again.
    # Let's provide the option to skip this with a pre-generated image.
    # Skip image generation if `image` keyword is not None.
    # Remember, this is for a very specific NORMALIZED spectrum
    t0 = time.time()
    if image is None:
        image = gen_image_coeff(bp, pupil=pupil, mask=mask, module=module, 
            sp_norm=sp_norm, coeff=coeff, coeff_hdr=coeff_hdr, 
            fov_pix=fov_pix, oversample=oversample,
            offset_r=offset_r, offset_theta=offset_theta, **kwargs)
    t1 = time.time()
    _log.debug('fov_pix={0}, oversample={1}'.format(fov_pix,oversample))
    _log.debug('Took %.2f seconds to generate images' % (t1-t0))
    if return_image:
        return image

    # Cosmic Ray Loss (JWST-STScI-001721)
    # SNR with cosmic ray events depends directly on ramp integration time
    if cr_noise:
        tint = (ngroup*nf + (ngroup-1)*nd2) * tf
        snr_fact = 1.0 - tint*6.7781e-5
    else:
        snr_fact = 1.0

    # Central position (in pixel coords) of PSF
    if offset_r==0:
        center = None
    else:
        xp, yp = rtheta_to_xy(offset_r/pix_scale, offset_theta)
        xp += image.shape[1] / 2.0 # x value in pixel position
        yp += image.shape[0] / 2.0 # y value in pixel position
        center = (xp, yp)

    # If grism spectroscopy
    if grism_obs:

        if units is None: units = 'uJy'
        wspec, spec = image

        # Wavelengths to grab sensitivity values
        #igood2 = bp.throughput > (bp.throughput.max()/4)
        igood2 = bp_igood(bp, min_trans=bp.throughput.max()/3, fext=0)
        wgood2 = waveset[igood2] / 1e4
        wsen_arr = np.unique((wgood2*10 + 0.5).astype('int')) / 10

        # Add an addition 0.1 on either side
        dw = 0.1
        wsen_arr = np.concatenate(([wsen_arr.min()-dw],wsen_arr,[wsen_arr.max()+dw]))

        #wdel = wsen_arr[1] - wsen_arr[0]

        # FWHM at each pixel position
        #fwhm_pix_arr = np.ceil(wsen_arr * 0.206265 / 6.5 / pix_scale)
        # Make sure there's at least 5 total pixels in spatial dimension
        #temp = fwhm_pix_arr.repeat(2).reshape([fwhm_pix_arr.size,2])
        #temp[:,0] = 2
        #rad_arr = temp.max(axis=1)
        # Ignore the above, let's always do a 5pix spatial aperture
        rad_arr = np.zeros(wsen_arr.size) + 2 # (2*2+1)

        # Spatial aperture size at each wavelength
        ap_spat = (2*rad_arr+1).astype('int')
        # Indices with spectral image
        ispat1 = (fov_pix - ap_spat) // 2
        ispat2 = ispat1 + ap_spat

        # Get spectral indices on the spectral image
        if (dw_bin is None) and (ap_spec is None):
            ap_spec = 2
        elif (dw_bin is not None) and (ap_spec is None):
            ap_spec = wspec.size * dw_bin / (wspec.max() - wspec.min())
            ap_spec = int(ap_spec+0.5)
        else:
            ap_spec = int(ap_spec+0.5)
        diff = abs(wspec.reshape(wspec.size,1) - wsen_arr)
        ind_wave = []
        for i in np.arange(wsen_arr.size):
            ind = (np.where(diff[:,i]==min(diff[:,i])))[0]
            ind_wave.append(ind[0])
        ispec1 = np.asarray(ind_wave) - ap_spec // 2
        ispec2 = ispec1 + ap_spec

        # At each wavelength, grab a sub image and find the limiting magnitude
        bglim_arr = []
        for i in np.arange(wsen_arr.size):
            sub_im = spec[ispat1[i]:ispat2[i],ispec1[i]:ispec2[i]]

            if forwardSNR:
                snr = _mlim_helper(sub_im, nint=nint, forwardSNR=forwardSNR,
                    ngroup=ngroup, nf=nf, nd2=nd2, tf=tf, fzodi=fzodi_pix,
                    snr_fact=snr_fact, **kwargs)
                bglim_arr.append(snr)

            else:
                # Interpolate over a coarse magnitude grid
                mag_arr=np.arange(5,35,1)
                mag_lim = _mlim_helper(sub_im, mag_norm, mag_arr, nsig=nsig, nint=nint,
                    ngroup=ngroup, nf=nf, nd2=nd2, tf=tf, fzodi=fzodi_pix,
                    snr_fact=snr_fact, **kwargs)

                # Zoom in and interoplate over finer grid
                mag_arr = np.arange(mag_lim-1,mag_lim+1,0.05)
                mag_lim = _mlim_helper(sub_im, mag_norm, mag_arr, nsig=nsig, nint=nint,
                    ngroup=ngroup, nf=nf, nd2=nd2, tf=tf, fzodi=fzodi_pix,
                    snr_fact=snr_fact, **kwargs)

                # Renormalize spectrum to magnitude limit and convert to desired units
                sp_norm2 = sp.renorm(mag_lim, 'vegamag', bp)
                sp_norm2.convert(units)
                bglim = np.interp(wsen_arr[i],sp_norm2.wave/1e4, sp_norm2.flux)
                bglim_arr.append(bglim)

        bglim_arr = np.asarray(bglim_arr)

        # Return sensitivity list along with corresponding wavelengths to dictionary
        if forwardSNR:
            sp_norm.convert(units)
            fvals = np.interp(wsen_arr, sp_norm.wave/1e4, sp_norm.flux)
            out = {'wave':wsen_arr.tolist(), 'snr':bglim_arr.tolist(),
                   'flux_units':units, 'flux':fvals.tolist(), 'Spectrum':sp.name}

            if quiet == False:
                print('{0} SNR for {1} source'.format(bp.name,sp.name))
                names = ('Wave','SNR','Flux ({})'.format(units))
                tbl = Table([wsen_arr,bglim_arr, fvals], names=names)
                for k in tbl.keys():
                    tbl[k].format = '9.2f'
                print(tbl)

        else:
            out = {'wave':wsen_arr.tolist(), 'sensitivity':bglim_arr.tolist(),
                   'units':units, 'nsig':nsig, 'Spectrum':sp.name}

            if quiet == False:
                print('{} Background Sensitivity ({}-sigma) for {} source'.\
                    format(bp.name,nsig,sp.name))

                names = ('Wave','Limit ({})'.format(units))
                tbl = Table([wsen_arr,bglim_arr], names=names)
                for k in tbl.keys():
                    tbl[k].format = '9.2f'
                print(tbl)

        return out

    # DHS spectroscopy
    elif dhs_obs:
        raise NotImplementedError('DHS has yet to be fully included')

    # Imaging (includes coronagraphy)
    else:
        if units is None: units = 'nJy'

        # Wavelength to grab sensitivity values
        obs = S.Observation(sp_norm, bp, binset=waveset)
        efflam = obs.efflam()*1e-4 # microns

        # Encircled energy
        rho_pix = dist_image(image)
        bins = np.arange(rho_pix.min(), rho_pix.max() + 1, 1)
        # Groups indices for each radial bin
        igroups, _, rad_pix = hist_indices(rho_pix, bins, True)
        # Sum of each radial annulus
        sums = binned_statistic(igroups, image, func=np.sum)
        # Encircled energy within each radius
        EE_flux = np.cumsum(sums)

        # How many pixels do we want?
        fwhm_pix = 1.2 * efflam * 0.206265 / 6.5 / pix_scale
        if rad_EE is None:
            rad_EE = np.max([fwhm_pix,2.5])
        npix_EE = np.pi * rad_EE**2

        # For surface brightness sensitivity (extended object)
        # Assume the fiducial (sp_norm) to be in terms of mag/arcsec^2
        # Multiply countrate() by pix_scale^2 to get in terms of per pixel (area)
        # This is the count rate per pixel for the fiducial starting point
        image_ext = obs.countrate() * pix_scale**2 # e-/sec/pixel
        #print(image_ext)

        if forwardSNR:
            im_var = pix_noise(ngroup=ngroup, nf=nf, nd2=nd2, tf=tf,
                fzodi=fzodi_pix, fsrc=image, **kwargs)**2

            # root squared sum of noise within each radius
            sums = binned_statistic(igroups, im_var, func=np.sum)
            EE_var = np.cumsum(sums)
            EE_sig = np.sqrt(EE_var / nint)

            EE_snr = snr_fact * EE_flux / EE_sig
            snr_rad = np.interp(rad_EE, rad_pix, EE_snr)
            flux_val = obs.effstim(units)
            out1 = {'type':'Point Source', 'snr':snr_rad, 'Spectrum':sp.name,
                'flux':flux_val, 'flux_units':units}

            # Extended object surfrace brightness
            im_var = pix_noise(ngroup=ngroup, nf=nf, nd2=nd2, tf=tf,
                fzodi=fzodi_pix, fsrc=image_ext, **kwargs)**2
            im_sig = np.sqrt(im_var*npix_EE / nint)
            # Total number of pixels within r=fwhm or 2.5 pixels
            fsum2 = image_ext * npix_EE
            snr2 = snr_fact * fsum2 / im_sig # SNR per "resolution element"ish
            out2 = {'type':'Surface Brightness', 'snr':snr2, 'Spectrum':sp.name,
                'flux':flux_val, 'flux_units':units+'/arcsec^2'}

            if quiet == False:
                for out in [out1,out2]:
                    print('{} SNR ({:.2f} {}): {:.2f} sigma'.\
                        format(out['type'], out['flux'], out['flux_units'], out['snr']))

        else:
            # Interpolate over a coarse magnitude grid to get SNR
            # Then again over a finer grid
            for ii in np.arange(2):
                if ii==0: mag_arr = np.arange(5,35,1)
                else: mag_arr = np.arange(mag_lim-1,mag_lim+1,0.05)

                fact_arr = 10**((mag_arr-mag_norm)/2.5)
                snr_arr = []
                for f in fact_arr:
                    #im_var = image/f/tint + var_const

                    im_var = pix_noise(ngroup=ngroup, nf=nf, nd2=nd2, tf=tf,
                        fzodi=fzodi_pix, fsrc=image/f, **kwargs)**2

                    # root squared sum of noise within each radius
                    sums = binned_statistic(igroups, im_var, func=np.sum)
                    EE_var = np.cumsum(sums)
                    EE_sig = np.sqrt(EE_var / nint)

                    EE_snr = snr_fact * (EE_flux/f) / EE_sig
                    snr_rad = np.interp(rad_EE, rad_pix, EE_snr)
                    snr_arr.append(snr_rad)

                snr_arr = np.asarray(snr_arr)
                mag_lim = np.interp(nsig, snr_arr[::-1], mag_arr[::-1])

                _log.debug('Mag Limits [{0:.2f},{1:.2f}]; {2:.0f}-sig: {3:.2f}'.\
                    format(mag_arr.min(),mag_arr.max(),nsig,mag_lim))

            # Renormalize spectrum at given magnitude limit
            sp_norm2 = sp.renorm(mag_lim, 'vegamag', bp)
            # Determine effective stimulus
            obs2 = S.Observation(sp_norm2, bp, binset=waveset)
            bglim = obs2.effstim(units)

            out1 = {'sensitivity':bglim, 'units':units, 'nsig':nsig, 'Spectrum':sp.name}

            # Same thing as above, but for surface brightness
            for ii in np.arange(2):
                if ii==0: mag_arr = np.arange(5,35,1)
                else: mag_arr = np.arange(mag_lim-1,mag_lim+1,0.05)

                fact_arr = 10**((mag_arr-mag_norm)/2.5)
                snr_arr = []
                for f in fact_arr:
                    im_var = pix_noise(ngroup=ngroup, nf=nf, nd2=nd2, tf=tf,
                        fzodi=fzodi_pix, fsrc=image_ext/f, **kwargs)**2

                    im_sig = np.sqrt(im_var*npix_EE / nint)
                    fsum2 = image_ext * npix_EE / f
                    snr2 = snr_fact * fsum2 / im_sig
                    #print('{:.5f} {:.5f} {:.2f}'.format(fsum2,im_sig,snr2))

                    snr_arr.append(snr2)

                snr_arr = np.asarray(snr_arr)
                mag_lim = np.interp(nsig, snr_arr[::-1], mag_arr[::-1])

                _log.debug('Mag Limits (mag/asec^2) [{0:.2f},{1:.2f}]; {2:.0f}-sig: {3:.2f}'.\
                    format(mag_arr.min(),mag_arr.max(),nsig,mag_lim))

            # mag_lim is in terms of mag/arcsec^2 (same as mag_norm)
            sp_norm2 = sp.renorm(mag_lim, 'vegamag', bp)
            obs2 = S.Observation(sp_norm2, bp, binset=waveset)
            bglim2 = obs2.effstim(units) # units/arcsec**2

            out2 = out1.copy()
            out2['sensitivity'] = bglim2
            out2['units'] = units+'/arcsec^2'

            if quiet == False:
                print('{} Sensitivity ({}-sigma): {:.2f} {}'.\
                       format('Point Source', nsig, bglim, out1['units']))
                print('{} Sensitivity ({}-sigma): {:.2f} {}'.\
                       format('Surface Brightness', nsig, bglim2, out2['units']))

        return out1, out2


def sat_limit_webbpsf(filter_or_bp, pupil=None, mask=None, module='A', pix_scale=None,
    sp=None, bp_lim=None, int_time=21.47354, full_well=81e3, well_frac=0.8,
    coeff=None, coeff_hdr=None, fov_pix=11, oversample=4, quiet=True, units='vegamag',
    offset_r=0, offset_theta=0, **kwargs):
    """Saturation limits

    Estimate the saturation limit of a point source for some bandpass.
    By default, it outputs the max K-Band magnitude assuming a G2V star,
    following the convention on the UA NIRCam webpage. This can be useful if
    one doesn't know how bright a source is in the selected NIRCam filter
    bandpass. However any user-defined bandpass (or user-defined spectrum)
    can be specifed. These must follow the Pysynphot conventions found here:
    http://pysynphot.readthedocs.org/en/latest/using_pysynphot.html

    This function returns the saturation limit in Vega magnitudes by default,
    however, any flux unit supported by Pysynphot is possible via the 'units'
    keyword.

    Parameters
    ==========

    Instrument Settings
    -------------------
    filter_or_bp : Either the name of the filter or pre-computed Pysynphot bandpass.
    pupil : NIRCam pupil elements such as grisms or lyot stops
    mask : Specify the coronagraphic occulter (spots or bar)
    module : 'A' or 'B'

    Spectrum Settings
    -------------------
    sp : A Pysynphot spectrum to calculate saturation (default: G2V star)
    bp_lim : A Pysynphot bandpass at which we report the magnitude that will
        saturate the NIRCam band assuming some spectrum sp
    units : Output units for saturation limit

    Detector Settings
    -------------------
    int_time : Integration time in seconds (default corresponds to 2 full frames)
    full_well : Detector full well level in electrons.
    well_frac : Fraction of full well to consider "saturated." 0.8 by default.

    PSF Information
    -------------------
    coeff : A cube of polynomial coefficients for generating PSFs. This is
        generally oversampled and has the shape:

            [fov_pix*oversample, fov_pix*oversample, deg]

        If not set, this this will be calculated from fov_pix, oversample,
        and npsf by generating a number of webbPSF images within the bandpass
        and fitting a high-order polynomial.
    fov_pix      : Number of detector pixels in the image coefficient and PSF.
    oversample   : Factor of oversampling of detector pixels.
    offset_r     : Radial offset of the target from center.
    offset_theta : Position angle for that offset, in degrees CCW (+Y).

    Keyword Args
    -------------------
    gen_psf_coeff - npsf and ndeg
    read_filter   - ND_acq
    """

    # PSF coefficients
    from pynrc.psfs import gen_image_coeff

    # Get filter throughput and create bandpass
    if isinstance(filter_or_bp, six.string_types):
        filter = filter_or_bp
        bp = read_filter(filter, pupil=pupil, mask=mask, module=module, **kwargs)
    else:
        bp = filter_or_bp
        filter = bp.name

    if bp_lim is None:
        bp_lim = S.ObsBandpass('johnson,k')
        bp_lim.name = 'K-Band'

    # If not set, select some settings based on filter (SW or LW)
    args = channel_select(bp)
    if pix_scale is None: 
        pix_scale = args[0] # Pixel scale (arcsec/pixel)

    # Spectrum and bandpass to report magnitude that saturates NIRCam band
    if sp is None: 
        sp = stellar_spectrum('G2V')

    # Just for good measure, make sure we're all in the same wave units
    bp_lim.convert(bp.waveunits)
    sp.convert(bp.waveunits)

    # Renormalize to 10th magnitude star (Vega mags)
    mag_norm = 10.0
    sp_norm = sp.renorm(mag_norm, 'vegamag', bp_lim)
    sp_norm.name = sp.name

    # Set up an observation of the spectrum using the specified bandpass
    # Use the bandpass wavelengths to bin the fluxes
    obs = S.Observation(sp_norm, bp, binset=bp.wave)
    # Convert observation to counts (e/sec)
    obs.convert('counts')

    # The number of pixels to span spatially
    fov_pix = int(fov_pix)
    oversample = int(oversample)
    # Generate the PSF image for analysis
    t0 = time.time()
    result = gen_image_coeff(bp, pupil=pupil, mask=mask, module=module, 
        sp_norm=sp_norm, coeff=coeff, coeff_hdr=coeff_hdr,
        fov_pix=fov_pix, oversample=oversample,
        offset_r=offset_r, offset_theta=offset_theta, **kwargs)
    t1 = time.time()
    _log.debug('Took %.2f seconds to generate images' % (t1-t0))

    # Total stellar flux and associated magnitude
    star_flux = obs.countrate() # e/sec
    mag_nrc = obs.effstim('vegamag')
    _log.debug('Total Source Count Rate for {0} = {1:0.1f} mags: {2:.0f} e-/sec'.\
        format(bp_lim.name, mag_norm, star_flux))
    _log.debug('Magnitude in {0} band: {1:.2f}'.format(bp.name, mag_nrc))

    # Saturation level (some fraction of full well) in electrons
    sat_level = well_frac * full_well

    # If grism spectroscopy
    if (pupil is not None) and ('GRISM' in pupil):
        wspec, spec = result

         # Time to saturation for 10-mag source
        sat_time = sat_level / spec
        _log.debug('Approximate Time to {1:.2f} of Saturation: {0:.1f} sec'.\
            format(sat_time.min(),well_frac))

        # Magnitude necessary to saturate a given pixel
        ratio = int_time/sat_time
        ratio[ratio < __epsilon] = __epsilon
        sat_mag = mag_norm + 2.5*np.log10(ratio)

        # Wavelengths to grab saturation values
        igood2 = bp.throughput > (bp.throughput.max()/4)
        wgood2 = bp.wave[igood2] / 1e4
        wsat_arr = np.unique((wgood2*10 + 0.5).astype('int')) / 10
        wdel = wsat_arr[1] - wsat_arr[0]
        msat_arr = []
        for w in wsat_arr:
            l1 = w-wdel/4
            l2 = w+wdel/4
            ind = ((wspec > l1) & (wspec <= l2))
            msat = sat_mag[fov_pix//2-1:fov_pix//2+2,ind].max()
            sp_temp = sp.renorm(msat, 'vegamag', bp_lim)
            obs_temp = S.Observation(sp_temp, bp_lim, binset=bp_lim.wave)
            msat_arr.append(obs_temp.effstim(units))

        msat_arr = np.array(msat_arr)

        # Print verbose information
        if not quiet:
            if bp_lim.name == bp.name:
                print('{0} Saturation Limit assuming {1} source:'.\
                    format(bp_lim.name,sp.name))
            else:
                print('{0} Saturation Limit for {1} assuming {2} source:'.\
                    format(bp_lim.name,bp.name,sp.name))

            names = ('Wave','Sat Limit ({})'.format(units))
            tbl = Table([wsat_arr,msat_arr], names=names)
            for k in tbl.keys():
                tbl[k].format = '9.2f'
            print(tbl)


        # Return saturation list along with corresponding wavelengths to dictionary
        return {'wave':wsat_arr.tolist(), 'satmag':msat_arr.tolist(),
            'units':units, 'Spectrum':sp_norm.name, 'bp_lim':bp_lim.name}

    # DHS spectroscopy
    elif (pupil is not None) and ('DHS' in pupil):
        raise NotImplementedError

    # Imaging
    else:
        psf = result

         # Time to saturation for 10-mag source
         # Only need the maximum pixel value
        sat_time = sat_level / psf.max()
        _log.debug('Point source approximate Time to {1:.2f} of Saturation: {0:.2f} sec'.\
            format(sat_time,well_frac))

        # Magnitude necessary to saturate a given pixel
        ratio = int_time/sat_time
        sat_mag = mag_norm + 2.5*np.log10(ratio)

        # Convert to desired unit
        sp_temp = sp.renorm(sat_mag, 'vegamag', bp_lim)
        obs_temp = S.Observation(sp_temp, bp_lim, binset=bp_lim.wave)
        res1 = obs_temp.effstim(units)
        
        out1 = {'satlim':res1, 'units':units, 'bp_lim':bp_lim.name, 'Spectrum':sp_norm.name}

        # For surface brightness saturation (extended object)
        # Assume the fiducial (sp_norm) to be in terms of mag/arcsec^2
        # Multiply countrate() by pix_scale^2 to get in terms of per pixel (area)
        # This is the count rate per pixel for the fiducial starting point
        image_ext = obs.countrate() * pix_scale**2 # e-/sec/pixel
        
        sat_time = sat_level / image_ext
        _log.debug('Extended object approximate Time to {1:.2f} of Saturation: {0:.2f} sec'.\
            format(sat_time,well_frac))
        
        # Magnitude necessary to saturate a given pixel
        ratio = int_time / sat_time
        sat_mag_ext = mag_norm + 2.5*np.log10(ratio)

        # Convert to desired unit
        sp_temp = sp.renorm(sat_mag_ext, 'vegamag', bp_lim)
        obs_temp = S.Observation(sp_temp, bp_lim, binset=bp_lim.wave)
        res2 = obs_temp.effstim(units)

        out2 = out1.copy()
        out2['satlim'] = res2
        out2['units'] = units+'/arcsec^2'

        # Print verbose information
        if not quiet:
            if bp_lim.name == bp.name:
                print('{} Saturation Limit assuming {} source (point source): {:.2f} {}'.\
                    format(bp_lim.name, sp_norm.name, out1['satlim'], out1['units']) )
                print('{} Saturation Limit assuming {} source (extended): {:.2f} {}'.\
                    format(bp_lim.name, sp_norm.name, out2['satlim'], out2['units']) )
            else:
                print('{} Saturation Limit for {} assuming {} source (point source): {:.2f} {}'.\
                    format(bp_lim.name, bp.name, sp_norm.name, out1['satlim'], out1['units']) )
                print('{} Saturation Limit for {} assuming {} source (extended): {:.2f} {}'.\
                    format(bp_lim.name, bp.name, sp_norm.name, out2['satlim'], out2['units']) )

        return out1, out2


def var_ex_model(ng, nf, params):
    """ Variance Excess Model

    Measured pixel variance shows a slight excess above the measured values.
    The input `params` describes this excess variance. This funciton can be 
    used to fit the excess variance for a variety of different readout patterns.
    """
    return 12. * (ng - 1.)/(ng + 1.) * params[0]**2 - params[1] / nf**0.5

def pix_noise(ngroup=2, nf=1, nd2=0, tf=10.73677, rn=15.0, ktc=29.0, p_excess=(0,0),
    fsrc=0.0, idark=0.003, fzodi=0, fbg=0, ideal_Poisson=False,
    ff_noise=False, **kwargs):
    """Noise per pixel

    Theoretical noise calculation of a generalized MULTIACCUM ramp in terms of e-/sec.
    Includes flat field errors from JWST-CALC-003894.

    Parameters
    ----------
    n : int
        Number of groups in integration rampl
    m : int
        Number of frames in each groupl
    s : int
        Number of dropped frames in each groupl
    tf : float
        Frame time
    rn : float
        Read Noise per pixel (e-).
    ktc : float
        kTC noise (in e-). Only valid for single frame (n=1)l
    p_excess : array-like
        An array or list of two elements that holds the parameters
        describing the excess variance observed in effective noise plots.
        By default these are both 0. For NIRCam detectors, recommended
        values are [1.0,5.0] for SW and [1.5,10.0] for LW.
    fsrc : float
        Flux of source in e-/sec/pix.
    idark : float
        Dark current in e-/sec/pix.
    fzodi : float
        Zodiacal light emission in e-/sec/pix.
    fbg : float
        Any additional background (telescope emission or scattered light?)
    ideal_Poisson : bool
        If set to True, use total signal for noise estimate,
        otherwise MULTIACCUM equation is used.
    ff_noise : bool
        Include flat field errors in calculation? From JWST-CALC-003894.
        Default=False.

    Notes
    -----
    Various parameters can either be single values or numpy arrays.
    If multiple inputs are arrays, make sure their array sizes match.
    Variables that need to have the same array shapes (or a single value):

        - n, m, s, & tf
        - rn, idark, ktc, fsrc, fzodi, & fbg

    Array broadcasting also works.

    Example
    -------

    >>> n = np.arange(50)+1  # An array of different ngroups to test out

    >>> # Create 2D Gaussian PSF with FWHM = 3 pix
    >>> npix = 20  # Number of pixels in x and y direction
    >>> fwhm = 3.0
    >>> x = np.arange(0, npix, 1, dtype=float)
    >>> y = x[:,np.newaxis]
    >>> x0 = y0 = npix // 2  # Center position
    >>> fsrc = np.exp(-4*np.log(2.) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    >>> fsrc /= fsrc.max()
    >>> fsrc *= 10  # 10 counts/sec in peak pixel
    >>> fsrc = fsrc.reshape(npix,npix,1)  # Necessary for broadcasting

    >>> # Represents pixel array w/ slightly different RN/pix
    >>> rn = 15 + np.random.normal(loc=0, scale=0.5, size=(1,npix,npix))
    >>> # Results is a 50x(20x20) showing the noise in e-/sec/pix at each group
    >>> noise = pix_noise(ngroup=n, rn=rn, fsrc=fsrc)
    """

    # Convert everything to arrays
    n = np.array(ngroup)
    m = np.array(nf)
    s = np.array(nd2)
    tf = np.array(tf)

    # Total flux (e-/sec/pix)
    ftot = fsrc + idark + fzodi + fbg

    # Special case if n=1
    # To be inserted at the end
    if (n==1).any():
        # Variance after averaging m frames
        var = ktc**2 + (rn**2 + ftot*tf) / m
        noise = np.sqrt(var)
        noise /= tf # In terms of e-/sec

        if (n==1).all(): return noise
        noise_n1 = noise

    ind_n1 = (n==1)
    temp = np.array(rn+ktc+ftot)
    temp_bool = np.zeros(temp.shape, dtype=bool)
    ind_n1_all = (temp_bool | ind_n1)

    # Group time
    tg = tf * (m + s)
    # Effective integration time
    tint = tg * (n - 1)

    # Read noise, group time, and frame time variances
    # This is the MULTIACCUM eq from Rauscher et al. (2007).
    # This equation assumes that the slope-fitting routine uses
    # incorrect covariance matrix that doesn't take into account
    # the correlated Poisson noise up the ramp.
    var_rn = rn**2       * 12.               * (n - 1.) / (m * n * (n + 1.))
    var_gp = ftot * tint * 6. * (n**2. + 1.) / (5 * n * (n + 1.))
    var_fm = ftot   * tf * 2. * (m**2. - 1.) * (n - 1.) / (m * n * (n + 1.))

    # Functional form for excess variance above theoretical
    # Empirically measured formulation
    # var_ex = 12. * (n - 1.)/(n + 1.) * p_excess[0]**2 - p_excess[1] / m**0.5
    var_ex = var_ex_model(n, m, p_excess)

    # Variance of total signal
    var_poisson = (ftot * tint) if ideal_Poisson else (var_gp - var_fm)

    # Total variance
    var = var_rn + var_poisson + var_ex
    sig = np.sqrt(var)

    # Noise in e-/sec
    noise = sig / tint
    # Make sure to copy over ngroup=1 cases
    if (n==1).any():
        noise[ind_n1_all] = noise_n1[ind_n1_all]
    #print(ind_n1_all.shape,noise.shape,noise_n1.shape)

    # Include flat field noise
    # JWST-CALC-003894
    if ff_noise:
        noise_ff = 1E-4 # Uncertainty in the flat field
        factor = 1 + noise_ff*np.sqrt(ftot)
        noise *= factor

    return noise

###########################################################################
#
#    Pysynphot Spectrum Wrappers
#
###########################################################################

from webbpsf_ext.spectra import BOSZ_spectrum, stellar_spectrum, source_spectrum
from webbpsf_ext.spectra import planets_sb12, sp_accr, jupiter_spec, companion_spec
from webbpsf_ext.spectra import linder_table, linder_filter, cond_table, cond_filter
from webbpsf_ext.spectra import bin_spectrum

def bin_spectrum(sp, wave, waveunits='um'):
    """Rebin spectrum

    Rebin a :mod:`pysynphot.spectrum` to a different wavelength grid.
    This function first converts the input spectrum to units
    of counts then combines the photon flux onto the
    specified wavelength grid.

    Output spectrum units are the same as the input spectrum.

    Parameters
    -----------
    sp : :mod:`pysynphot.spectrum`
        Spectrum to rebin.
    wave : array_like
        Wavelength grid to rebin onto.
    waveunits : str
        Units of wave input. Must be recognizeable by Pysynphot.

    Returns
    -------
    :mod:`pysynphot.spectrum`
        Rebinned spectrum in same units as input spectrum.
    """

    waveunits0 = sp.waveunits
    fluxunits0 = sp.fluxunits

    # Convert wavelength of input spectrum to desired output units
    sp.convert(waveunits)
    # We also want input to be in terms of counts to conserve flux
    sp.convert('flam')

    edges = S.binning.calculate_bin_edges(wave)
    ind = (sp.wave >= edges[0]) & (sp.wave <= edges[-1])
    binflux = binned_statistic(sp.wave[ind], sp.flux[ind], np.mean, bins=edges)

    # Interpolate over NaNs
    ind_nan = np.isnan(binflux)
    finterp = interp1d(wave[~ind_nan], binflux[~ind_nan], kind='cubic')
    binflux[ind_nan] = finterp(wave[ind_nan])

    sp2 = S.ArraySpectrum(wave, binflux, waveunits=waveunits, fluxunits='flam')
    sp2.convert(waveunits0)
    sp2.convert(fluxunits0)

    # Put back units of original input spectrum
    sp.convert(waveunits0)
    sp.convert(fluxunits0)

    return sp2


def zodi_spec(zfact=None, ra=None, dec=None, thisday=None, **kwargs):
    """Zodiacal light spectrum.

    New: Use `ra`, `dec`, and `thisday` keywords to call `jwst_backgrounds`
    to obtain more accurate predictions of the background.

    Creates a spectrum of the zodiacal light emission in order to estimate the
    in-band sky background flux. This is primarily the addition of two blackbodies
    at T=5300K (solar scattered light) and T=282K (thermal dust emission)
    that have been scaled to match literature flux values. 

    In reality, the intensity of the zodiacal dust emission varies as a
    function of viewing position. In this case, we have added the option
    to scale the zodiacal level (or each component individually) by some
    user-defined factor 'zfact'. The user can set zfact as a scalar in order
    to scale the entire spectrum. If defined as a list, tuple, or np array,
    then the each component gets scaled where T=5300K corresponds to the first
    elements and T=282K is the second element of the array. 

    The `zfact` parameter has no effect if `jwst_backgrounds` is called.
    Representative values for zfact:

        * 0.0 - No zodiacal emission
        * 1.0 - Minimum zodiacal emission from JWST-CALC-003894
        * 1.2 - Required NIRCam performance
        * 2.5 - Average (default)
        * 5.0 - High
        * 10.0 - Maximum


    Parameters
    ----------
    zfact : float
        Factor to scale Zodiacal spectrum (default 2.5).
    ra : float
        Right ascension in decimal degrees
    dec : float
        Declination in decimal degrees
    thisday: int
        Calendar day to use for background calculation.  If not given, will 
        use the average of visible calendar days.

    Returns
    -------
    :mod:`pysynphot.spectrum`
        Output is a Pysynphot spectrum with default units of flam (erg/s/cm^2/A/sr).
        Note: Pysynphot doesn't recognize that it's per steradian, but we must keep
        that in mind when integrating the flux per pixel.

    Notes
    -----
    Added the ability to query the Euclid background model using
    :func:`zodi_euclid` for a specific location and observing time.
    The two blackbodies will be scaled to the 1.0 and 5.5 um emission.
    This functionality is deprecated in favor of jwst_backgrounds.

    Keyword Args
    ------------
    locstr :
        Object name or RA/DEC (decimal degrees or sexigesimal).
        Queries the `IPAC Euclid Background Model
        <http://irsa.ipac.caltech.edu/applications/BackgroundModel/>`_
    year : int
        Year of observation.
    day : float
        Day of observation.

    """

    
    if (ra is not None) and (dec is not None):
        if _jbt_exists == False:
            _log.warning("`jwst_backgrounds` not installed. `ra`, `dec`, and `thisday` parameters will not work.")
        else:
            # Wavelength for "bathtub plot" (not used here)
            wave_bath = 2.5
            bkg = jbt.background(ra, dec, wave_bath)
            # Get wavelength and flux values 
            wvals = bkg.bkg_data['wave_array'] # Wavelength (um)
            farr = bkg.bkg_data['total_bg'] # Total background (MJy/sr)

            if thisday is None:
                # Use average of visible calendar days
                ftot = farr.mean(axis=0)
            else:
                calendar = bkg.bkg_data['calendar']
                if thisday in calendar:
                    ind = np.where(calendar==thisday)[0][0]
                    ftot = farr[ind]
                else:
                    _log.warning("The input calendar day {}".format(thisday)+" is not available. \
                                 Choosing closest visible day.")
                    diff = np.abs(calendar-thisday)
                    ind = np.argmin(diff)
                    ftot = farr[ind]

            sp = S.ArraySpectrum(wave=wvals*1e4, flux=ftot*1e6, fluxunits='Jy')
            sp.convert('flam')
            sp.name = 'Total Background'

            return sp


    if zfact is None: 
        zfact = 2.5
    #_log.debug('zfact:{0:.1f}'.format(zfact))

    if isinstance(zfact, (list, tuple, np.ndarray)):
        f1, f2 = zfact
    else:
        f1 = f2 = zfact
    # These values have been scaled to match JWST-CALC-003894 values
    # in order to work with Pysynphot's blackbody function.
    # Pysynphot's BB function is normalized to 1Rsun at 1kpc by default.
    f1 *= 4.0e7
    f2 *= 2.0e13

    bb1 = f1 * S.BlackBody(5300.0)
    bb2 = f2 * S.BlackBody(282.0)


    # Query Euclid Background Model
    locstr = kwargs.get('locstr')
    year  = kwargs.get('year')
    day   = kwargs.get('day')
    if (locstr is not None) and (year is not None) and (day is not None):

        # Wavelengths in um and values in MJy
        waves = np.array([1.0,5.5])
        vals = zodi_euclid(locstr, year, day, waves, **kwargs)

        bb1.convert('Jy')
        bb2.convert('Jy')

        # MJy at wavelength locations
        f_bb1 = bb1.sample(waves*1e4) / 1e6
        f_bb2 = bb2.sample(waves*1e4) / 1e6

        bb1 *= (vals[0]-f_bb2[0])/f_bb1[0]
        bb2 *= (vals[1]-f_bb1[1])/f_bb2[1]

    sp_zodi = bb1 + bb2
    sp_zodi.convert('flam')
    sp_zodi.name = 'Zodiacal Light'


    return sp_zodi


def zodi_euclid(locstr, year, day, wavelengths=[1,5.5], ido_viewin=0, **kwargs):
    """IPAC Euclid Background Model

    Queries the `IPAC Euclid Background Model
    <http://irsa.ipac.caltech.edu/applications/BackgroundModel/>`_
    in order to get date and position-specific zodiacal dust emission.

    The program relies on ``urllib3`` to download the page in XML format.
    However, the website only allows single wavelength queries, so
    this program implements a multithreaded procedure to query
    multiple wavelengths simultaneously. However, due to the nature
    of the library, only so many requests are allowed to go out at a time, 
    so this process can take some time to complete.
    Testing shows about 500 wavelengths in 10 seconds as a rough ballpark.

    Recommended to grab only a few wavelengths for normalization purposes.

    Parameters
    ----------
    locstr : str
        This input field must contain either coordinates (as string), 
        or an object name resolveable via NED or SIMBAD.
    year: string
        Year. Limited to 2018 to 2029 for L2 position.
    day : string
        Day of year (1-366). Limited to 2018 Day 274 to 2029 Day 120 
        for L2 position and ido_viewin=0.
    wavelength : array-like
        Wavelength in microns (0.5-1000).
    ido_viewin : 0 or 1 
        If set to 0, returns zodiacal emission at specific location for input time.
        If set to 1, then gives the median value for times of the year that the object 
        is in a typical spacecraft viewing zone. Currently this is set to solar 
        elongations between 85 and 120 degrees.

    References
    ----------
    See the `Euclid Help Website
    <http://irsa.ipac.caltech.edu/applications/BackgroundModel/docs/dustProgramInterface.html>`_
    for more details.

    """

    # from urllib2 import urlopen
    import urllib3
    import xmltodict
    from multiprocessing.pool import ThreadPool

    def fetch_url(url):
        """
        TODO: Add error handling.
        """
        # response = urlopen(url)
        # response = response.read()

        http = urllib3.PoolManager()
        response = http.request('GET', url)

        d = xmltodict.parse(response.data, xml_attribs=True)
        fl_str = d['results']['result']['statistics']['zody']
        return float(fl_str.split(' ')[0])


    #locstr="17:26:44 -73:19:56"
    #locstr = locstr.replace(' ', '+')
    #year=2019
    #day=1
    #obslocin=0
    #ido_viewin=1
    #wavelengths=None

    req_list = []
    for w in wavelengths:
        url = 'http://irsa.ipac.caltech.edu/cgi-bin/BackgroundModel/nph-bgmodel?'
        req = "{}&locstr={}&wavelength={:.2f}&year={}&day={}&obslocin=0&ido_viewin={}"\
            .format(url, locstr, w, year, day, ido_viewin)
        req_list.append(req)

    nthread = np.min([50,len(wavelengths)])
    pool = ThreadPool(nthread)
    results = pool.imap(fetch_url, req_list)

    res = []
    for r in results: res.append(r)
    pool.close()

    return np.array(res)


# def _zodi_spec_old(level=2):
# 	"""
# 	Create a spectrum of the zodiacal light emission in order to estimate the
# 	in-band sky background flux. This is simply the addition of two blackbodies
# 	at T=5800K (solar scattered light) and T=300K (thermal dust emission)
# 	that have been scaled to match the literature flux values.
#
# 	In reality, the intensity of the zodiacal dust emission varies as a
# 	function of viewing position. In this case, we have added different levels
# 	intensity similiar to the results given by old NIRCam ETC. These have not
# 	been validated in any way and should be used with caution, but at least
# 	give an order of magnitude of the zodiacal light background flux.
#
# 	There are four different levels that can be passed through the level
# 	parameter: 0=None, 1=Low, 2=Avg, 3=High
#
# 	For instance set sp_zodi = zodi_spec(3) for a highish sky flux.
# 	Default is 2
# 	"""
#
# 	bb1 = S.BlackBody(5800.); bb2 = S.BlackBody(300.)
# 	sp_zodi = (1.7e7*bb1 + 2.3e13*bb2) * 3.73
# 	sp_zodi.convert('flam')
#
# 	# This is how some case statements are done in Python
# 	# Select the level of zodiacal light emission
# 	# 0=None, 1=Low, 2=Avg, 3=High
# 	switcher = {0:0.0, 1:0.5, 2:1.0, 3:1.8}
# 	factor = switcher.get(level, None)
#
# 	if factor is None:
# 		_log.warning('The input parameter level=%s is not valid. Setting zodiacal light to 0.' % level)
# 		_log.warning('Valid values inlclude: %s' % switcher.keys())
# 		factor = 0
#
# 	sp_zodi *= factor
# 	sp_zodi.name = 'Zodiacal Light'
#
# 	return sp_zodi

def grism_background_image(filter, pupil='GRISM0', module='A', sp_bg=None, 
                           include_com=True, **kwargs):
    """Create full grism background image"""

    # Option for GRISMR/GRISMC
    if 'GRISMR' in pupil:
        pupil = 'GRISM0'
    elif 'GRISMC' in pupil: 
        pupil = 'GRISM90'

    upper = 9.6 if include_com else 31.2
    g_bg = grism_background(filter, pupil, module, sp_bg, upper=upper, **kwargs)

    final_image = np.zeros([2048,2048])
    if 'GRISM0' in pupil:
        final_image = final_image + g_bg.reshape([1,-1])
    else:
        final_image = final_image + g_bg.reshape([-1,1])
        # Add COM background
        if include_com:
            final_image += grism_background_com(filter, pupil, module, sp_bg, **kwargs)

    return final_image
    

def grism_background(filter, pupil='GRISM0', module='A', sp_bg=None, 
                     orders=[1,2], wref=None, upper=9.6, **kwargs):
    """
    
    Returns a 1D array of grism Zodiacal/thermal background
    emission model, including roll-off from pick-off mirror (POM)
    edges. By default, this includes light dispersed by the
    1st and 2nd grism orders (m=1 and m=2). 
    
    For column dipsersion, we ignore the upper region occupied by
    the coronagraphic mask region by default. The preferred way to
    include this region is to add the dispersed COM image from the
    `grism_background_com` function to create the full 2048x2048
    image. Or, more simply (but less accurate) is to set an `upper`
    value of 31.2, which is the approximately distance (in arcsec)
    from the top of the detector to the top of the coronagraphic
    field of view.
    
    Parameters
    ==========
    filter : str
        Name of filter (Long Wave only).
    pupil : str
        Either 'GRISM0' ('GRISMR') or 'GRISM90' ('GRISMC').
    module : str
        NIRCam 'A' or 'B' module.
    sp_bg : :mod:`pysynphot.spectrum`
        Spectrum of Zodiacal background emission, which gets
        multiplied by bandpass throughput to determine final
        wavelength-dependent flux that is then dispersed.
    orders : array-like
        What spectral orders to include? Valid orders are 1 and 2.
    wref : float or None
        Option to set the undeviated wavelength, otherwise this will
        search a lookup table depending on the grism.
    upper : float
        Set the maximum bounds for out-of-field flux to be dispersed
        onto the detector. By default, this value is 9.6", corresponding
        to the bottom of the coronagraphic mask. Use `grism_background_com`
        to then include image of dispersed COM mask. 
        If you want something simpler, increase this value to 31.2" to 
        assume the coronagraphic FoV is free of any holder blockages or 
        substrate and occulting masks.
        
    Keyword Args
    ============
    zfact : float
        Factor to scale Zodiacal spectrum (default 2.5).
    ra : float
        Right ascension in decimal degrees
    dec : float
        Declination in decimal degrees
    thisday: int
        Calendar day to use for background calculation.  If not given, will 
        use the average of visible calendar days.
    """
    
    # Option for GRISMR/GRISMC
    if 'GRISMR' in pupil:
        pupil = 'GRISM0'
    elif 'GRISMC' in pupil:
        pupil = 'GRISM90'
        
    # Pixel scale
    pix_scale, _, _ = channel_select(read_filter(filter))

    # Undeviated wavelength
    if wref is None: 
        wref = grism_wref(pupil, module) 
    
    # Background spectrum
    if sp_bg is None:
        sp_bg = zodi_spec(**kwargs)

    # Total number of "virtual" pixels spanned by pick-off mirror
    border = np.array([8.4, 8.0]) if ('GRISM0' in pupil) else np.array([12.6, upper])
    extra_pix = (border / pix_scale + 0.5).astype('int')
    extra_pix[extra_pix<=0] = 1 # Ensure there's at least 1 extra pixel
    npix_tot = 2048 + extra_pix.sum()

    flux_all = np.zeros(npix_tot)
    for grism_order in orders:
        # Get filter throughput and create bandpass
        bp = read_filter(filter, pupil=pupil, module=module, 
                         grism_order=grism_order, **kwargs)
        
        # Get wavelength dispersion solution
        res, dw = grism_res(pupil, module, grism_order) # Resolution and dispersion
        
        # Observation spectrum converted to count rate
        obs_bg = S.Observation(sp_bg, bp, bp.wave)
        obs_bg.convert('counts')

        # Total background flux per pixel (not dispersed)
        area_scale = (pix_scale/206265.0)**2
        fbg_tot = obs_bg.countrate() * area_scale
        # Total counts/sec within each wavelength bin
        binwave = obs_bg.binwave/1e4
        binflux = obs_bg.binflux*area_scale
            
        # Interpolation function
        fint = interp1d(binwave, binflux, kind='cubic')
        # Wavelengths at each pixel to interpolate
        wave_vals = np.arange(binwave.min(), binwave.max(), dw)
        # Get flux values and preserve total flux
        flux_vals = fint(wave_vals)
        flux_vals = fbg_tot * flux_vals / flux_vals.sum()
        
        # # Wavelengths at each pixel to interpolate
        # wave_vals = np.arange(bp.wave.min()/1e4, bp.wave.max()/1e4, dw)
    
        # # Rebin onto desired wavelength grid
        # sp_new = bin_spectrum(sp_bg, wave_vals, waveunits='um')
        # obs_bg = S.Observation(sp_new, bp, binset=sp_new.wave)
        # # Get flux values per pixel
        # obs_bg.convert('counts')
        # flux_vals = obs_bg.binflux * (pix_scale/206265.0)**2
    
        # Index of reference wavelength
        iref = int((wref - wave_vals[0]) / (wave_vals[1] - wave_vals[0]))

        # Determine the array indices that contribute for each pixel
        # Use indexing rather than array shifting for speed
        # This depends on the size of the POM relative to detector
        offset = -1*int(wref*res/2 + 0.5) if grism_order==2 else 0
        i1_arr = np.arange(iref,iref-npix_tot,-1)[::-1] + offset
        i2_arr = np.arange(iref,iref+npix_tot,+1) + offset
        i1_arr[i1_arr<0] = 0
        i1_arr[i1_arr>len(wave_vals)] = len(wave_vals)
        i2_arr[i2_arr<0] = 0
        i2_arr[i2_arr>len(wave_vals)] = len(wave_vals)

        flux_all += np.array([flux_vals[i1:i2].sum() for i1,i2 in zip(i1_arr,i2_arr)])
                
    # Crop only detector pixels
    flux_all = flux_all[extra_pix[0]:-extra_pix[1]]
    
    # Module B GRISM0/R disperses in opposite direction ('sci' coords)
    if ('GRISM0' in pupil) and (module=='B'):
        flux_all = flux_all[::-1]
        
    # Return single 
    return flux_all

def grism_background_com(filter, pupil='GRISM90', module='A', sp_bg=None, 
                         wref=None, **kwargs):
    
    
    # Option for GRISMR/GRISMC
    if 'GRISMR' in pupil:
        pupil = 'GRISM0'
    elif 'GRISMC' in pupil:
        pupil = 'GRISM90'
        
    if 'GRISM0' in pupil:
        _log.info('COM feature not present for row grisms.')
        return 0

    # Only see COM for 1st order
    # Minimum wavelength is 2.4um, which means 2nd order is 2400 pixels away.
    grism_order = 1
    # Get filter throughput and create bandpass
    bp = read_filter(filter, pupil=pupil, module=module, grism_order=grism_order, 
                     coron_substrate=True, **kwargs)

    # Pixel scale
    pix_scale, _, _ = channel_select(read_filter(filter))

    # Get wavelength dispersion solution
    res, dw = grism_res(pupil, module, grism_order)

    # Undeviated wavelength
    wref = grism_wref(pupil, module) if wref is None else wref
         
    # Background spectrum
    sp_bg = zodi_spec(**kwargs) if sp_bg is None else sp_bg
        
    # Coronagraphic mask image
    im_com = build_mask_detid(module+'5')
    # Crop to mask holder
    # Remove anything that is 0 or max
    im_collapse = im_com.sum(axis=1)
    ind_cut = (im_collapse == im_collapse.max()) | (im_collapse == 0)
    im_com = im_com[~ind_cut]
    ny_com, nx_com = im_com.shape

    # Observation spectrum converted to count rate
    obs_bg = S.Observation(sp_bg, bp, bp.wave)
    obs_bg.convert('counts')

    # Total background flux per pixel (not dispersed)
    area_scale = (pix_scale/206265.0)**2
    fbg_tot = obs_bg.countrate() * area_scale
    # Total counts/sec within each wavelength bin
    binwave = obs_bg.binwave/1e4
    binflux = obs_bg.binflux*area_scale

    # Interpolation function
    fint = interp1d(binwave, binflux, kind='cubic')
    # Wavelengths at each pixel to interpolate
    wave_vals = np.arange(binwave.min(), binwave.max(), dw)
    # Get flux values and preserve total flux
    flux_vals = fint(wave_vals)
    flux_vals = fbg_tot * flux_vals / flux_vals.sum()

    # Index of reference wavelength in spectrum
    iref = int((wref - wave_vals[0]) / (wave_vals[1] - wave_vals[0]))
        
    # Pixel position of COM image lower and upper bounds
    upper = 9.6
    ipix_ref = 2048 + int(upper/pix_scale + 0.5)
    ipix_lower = ipix_ref - iref 
    ipix_upper = ipix_lower + ny_com + len(flux_vals)
    # print('COM', ipix_lower, ipix_upper)
        
    # Only include if pixel positions overlap detector frame
    if (ipix_upper>0) and (ipix_lower<2048):
        # Shift and add images
        im_shift = np.zeros([ny_com+len(flux_vals), nx_com])
        # print(len(flux_vals))
        for i, f in enumerate(flux_vals):
            im_shift[i:i+ny_com,:] += im_com*f
            
        # Position at appropriate location within detector frame
        # First, either pad the lower, or crop to set bottom of detector
        if ipix_lower>=0 and ipix_lower<2048:
            im_shift = np.pad(im_shift, ((ipix_lower,0),(0,0)))
        elif ipix_lower<0:
            im_shift = im_shift[-ipix_lower:,:]
            
        # Expand or contract to final full detector size
        if im_shift.shape[0]<2048:
            im_shift = np.pad(im_shift, ((0,2048-im_shift.shape[0]),(0,0)))
        else:
            im_shift = im_shift[0:2048,:]

        res = im_shift 
    else:
        res = 0

    return res

###########################################################################
#
#    Coronagraphic Disk Imaging Routines
#
###########################################################################

def nproc_use_convolve(fov_pix, oversample, npsf=None):
    """
    Attempt to estimate a reasonable number of processes to use for multiple
    simultaneous convolve_fft calculations.

    Here we attempt to estimate how many such calculations can happen in
    parallel without swapping to disk, with a mixture of empiricism and conservatism.
    One really does not want to end up swapping to disk with huge arrays.

    NOTE: Requires psutil package. Otherwise defaults to mp.cpu_count() / 2

    Parameters
    -----------
    fov_pix : int
        Square size in detector-sampled pixels of final PSF image.
    oversample : int
        The optical system that we will be calculating for.
    npsf : int
        Number of PSFs. Sets maximum # of processes.
    """

    try:
        import psutil
    except ImportError:
        nproc = int(mp.cpu_count() // 2)
        if nproc < 1: nproc = 1

        _log.info("No psutil package available, cannot estimate optimal nprocesses.")
        _log.info("Returning nproc=ncpu/2={}.".format(nproc))
        return nproc

    mem = psutil.virtual_memory()
    avail_GB = mem.available / (1024**3) - 1.0 # Leave 1 GB

    fov_pix_over = fov_pix * oversample

    # Memory formulas are based on fits to memory usage stats for:
    #   fov_arr = np.array([16,32,128,160,256,320,512,640,1024,2048])
    #   os_arr = np.array([1,2,4,8])
    # In MBytes
    mem_total = 300*(fov_pix_over)**2 * 8 / (1024**2)

    # Convert to GB
    mem_total /= 1024

    # How many processors to split into?
    nproc = avail_GB // mem_total
    nproc = np.min([nproc, mp.cpu_count(), poppy.conf.n_processes])
    if npsf is not None:
        nproc = np.min([nproc, npsf])
        # Resource optimization:
        # Split iterations evenly over processors to free up minimally used processors.
        # For example, if there are 5 processes only doing 1 iteration, but a single
        #   processor doing 2 iterations, those 5 processors (and their memory) will not
        #   get freed until the final processor is finished. So, to minimize the number
        #   of idle resources, take the total iterations and divide by two (round up),
        #   and that should be the final number of processors to use.
        np_max = np.ceil(npsf / nproc)
        nproc = int(np.ceil(npsf / np_max))

    if nproc < 1: nproc = 1

    return int(nproc)


###########################################################################
#
#    Coronagraphic Mask Transmission
#
###########################################################################

def offset_bar(filt, mask):
    """Bar mask offset locations

    Get the appropriate offset in the x-position to place a source on a bar mask.
    Each bar is 20" long with edges and centers corresponding to::

        SWB: [1.03, 2.10, 3.10] (um) => [-10, 0, +10] (asec)
        LWB: [2.30, 4.60, 6.90] (um) => [+10, 0, -10] (asec)
    """

    if (mask is not None) and ('WB' in mask):
        # What is the effective wavelength of the filter?
        #bp = pynrc.read_filter(filter)
        #w0 = bp.avgwave() / 1e4
        w0 = np.float(filt[1:-1])/100

        # Choose wavelength from dictionary
        wdict = {'F182M': 1.84, 'F187N': 1.88, 'F210M': 2.09, 'F212N': 2.12,
                 'F250M': 2.50, 'F300M': 2.99, 'F335M': 3.35, 'F360M': 3.62,
                 'F410M': 4.09, 'F430M': 4.28, 'F460M': 4.63, 'F480M': 4.79,
                 'F200W': 2.23, 'F277W': 3.14, 'F356W': 3.97, 'F444W': 4.99}
        w = wdict.get(filt, w0)

        # Get appropriate x-offset
        #xoff_asec = np.interp(w,wpos,xpos)

        if 'SWB' in mask:
            if filt[-1]=="W": xoff_asec = 6.83 * (w - 2.196)
            else:             xoff_asec = 7.14 * (w - 2.100)
        elif 'LWB' in mask:
            if filt[-1]=="W": xoff_asec = -3.16 * (w - 4.747)
            else:             xoff_asec = -3.26 * (w - 4.600)

        #print(w, xoff_asec)

        yoff_asec = 0.0

        r, theta = xy_to_rtheta(xoff_asec, yoff_asec)
    else:
        r, theta = (0.0, 0.0)

    # Want th_bar to be -90 so that r matches webbpsf
    if theta>0: 
        r  = -1 * r
        theta = -1 * theta

    #print(r, theta)
    return r, theta


def coron_trans(name, module='A', pixelscale=None, npix=None, oversample=1, 
    nd_squares=True, shift_x=None, shift_y=None):
    """
    Build a transmission image of a coronagraphic mask spanning
    the 20" coronagraphic FoV.

    oversample is used only if pixelscale is set to None.
    """

    from webbpsf.optics import NIRCam_BandLimitedCoron

    shifts = {'shift_x': shift_x, 'shift_y': shift_y}

    bar_offset = None
    if name=='MASK210R':
        pixscale = pixscale_SW
        channel = 'short'
        filter = 'F210M'
    elif name=='MASK335R':
        pixscale = pixscale_LW
        channel = 'long'
        filter = 'F335M'
    elif name=='MASK430R':
        pixscale = pixscale_LW
        channel = 'long'
        filter = 'F430M'
    elif name=='MASKSWB':
        pixscale = pixscale_SW
        channel = 'short'
        filter = 'F210M'
        bar_offset = 0
    elif name=='MASKLWB':
        pixscale = pixscale_LW
        channel = 'long'
        filter = 'F430M'
        bar_offset = 0

    if pixelscale is None:
        pixelscale = pixscale / oversample
        if npix is None:
            npix = 320 if channel=='long' else 640
            npix = int(npix * oversample + 0.5)
    elif npix is None:
        # default to 20" if pixelscale is set but no npix
        npix = int(20 / pixelscale + 0.5)

    mask = NIRCam_BandLimitedCoron(name=name, module=module, bar_offset=bar_offset, auto_offset=None, 
                                   nd_squares=nd_squares, **shifts)

    # Create wavefront to pass through mask and obtain transmission image
    bandpass = read_filter(filter)
    wavelength = bandpass.avgwave() / 1e10
    wave = poppy.Wavefront(wavelength=wavelength, npix=npix, pixelscale=pixelscale)
    im = mask.get_transmission(wave)

    return im


def build_mask(module='A', pixscale=pixscale_LW):
    """Create coronagraphic mask image

    Return a truncated image of the full coronagraphic mask layout
    for a given module. Assumes each mask is exactly 20" across.

    +V3 is up, and +V2 is to the left.
    """
    if module=='A':
        names = ['MASK210R', 'MASK335R', 'MASK430R', 'MASKSWB', 'MASKLWB']
    elif module=='B':
        names = ['MASKSWB', 'MASKLWB', 'MASK430R', 'MASK335R', 'MASK210R']

    npix = int(20 / pixscale + 0.5)
    allims = [coron_trans(name, module=module, pixelscale=pixscale, npix=npix) for name in names]
    return np.concatenate(allims, axis=1)


def build_mask_detid(detid, oversample=1, ref_mask=None, pupil=None):
    """Create mask image for a given detector

    Return a full coronagraphic mask image as seen by a given SCA.
    +V3 is up, and +V2 is to the left.

    Parameters
    ----------
    detid : str
        Name of detector, 'A1', A2', ... 'A5' (or 'ALONG'), etc.
    oversample : float
        How much to oversample output mask relative to detector sampling.
    ref_mask : str or None
        Reference mask for placement of coronagraphic mask elements.
        If None, then defaults are chosen for each detector.
    pupil : str or None
        Which Lyot pupil stop is being used? This affects holder placement.
        If None, then defaults based on ref_mask.
    """

    names = ['A1', 'A2', 'A3', 'A4', 'A5',
             'B1', 'B2', 'B3', 'B4', 'B5']

    # In case input is 'NRC??'
    if 'NRC' in detid:
        detid = detid[3:]

    # Convert ALONG to A5 name
    module = detid[0]
    detid = '{}5'.format(module) if 'LONG' in detid else detid

    # Make sure we have a valid name
    if detid not in names:
        raise ValueError("Invalid detid: {0} \n  Valid names are: {1}" \
              .format(detid, ', '.join(names)))

    pixscale = pixscale_LW if '5' in detid else pixscale_SW
    pixscale_over = pixscale / oversample

    # Build the full mask
    xpix = ypix = 2048
    xpix_over = int(xpix * oversample)
    ypix_over = int(ypix * oversample)

    cmask = np.ones([ypix_over, xpix_over], dtype='float64')

    # These detectors don't see any of the mask structure
    if detid in ['A1', 'A3', 'B2', 'B4']:
        return cmask

    if detid=='A2':
        cnames = ['MASK210R', 'MASK335R', 'MASK430R']
        ref_mask = 'MASK210R' if ref_mask is None else ref_mask
    elif detid=='A4':
        cnames = ['MASK430R', 'MASKSWB', 'MASKLWB']
        ref_mask = 'MASKSWB' if ref_mask is None else ref_mask
    elif detid=='A5':
        cnames = ['MASK210R', 'MASK335R', 'MASK430R', 'MASKSWB', 'MASKLWB']
        ref_mask = 'MASK430R' if ref_mask is None else ref_mask
    elif detid=='B1':
        cnames = ['MASK430R', 'MASK335R', 'MASK210R']
        ref_mask = 'MASK210R' if ref_mask is None else ref_mask
    elif detid=='B3':
        cnames = ['MASKSWB', 'MASKLWB', 'MASK430R']
        ref_mask = 'MASKSWB' if ref_mask is None else ref_mask
    elif detid=='B5':
        cnames = ['MASKSWB', 'MASKLWB', 'MASK430R', 'MASK335R', 'MASK210R']
        ref_mask = 'MASK430R' if ref_mask is None else ref_mask

    # Generate sub-images for each aperture
    # npix = int(ypix / len(cnames))
    npix = int(20.5 / pixscale_over + 0.5)
    npix_large = int(26 / pixscale_over + 0.5)
    allims = [coron_trans(cname, module=module, pixelscale=pixscale_over, npix=npix_large) for cname in cnames]
    
    if pupil is None:
        pupil = 'WEDGELYOT' if ('WB' in ref_mask) else 'CIRCLYOT'

    # For each sub-image, expand and move to correct location
    channel = 'LW' if '5' in detid else 'SW'
    for i, name in enumerate(cnames):
        cdict = coron_ap_locs(module, channel, name, pupil=pupil, full=False)
        # Crop off large size
        im_crop = pad_or_cut_to_size(allims[i], (npix, npix_large))
        # Expand and offset
        xsci, ysci = cdict['cen_sci']
        xoff = xsci*oversample - ypix_over/2
        yoff = ysci*oversample - xpix_over/2
        im_expand = pad_or_cut_to_size(im_crop+1000, (ypix_over, xpix_over), offset_vals=(yoff,xoff))
        ind_good = ((cmask<100) & (im_expand>100)) | ((cmask==1001) & (im_expand>100))
        cmask[ind_good] = im_expand[ind_good]

    # Remove offsets
    cmask[cmask>100] = cmask[cmask>100] - 1000

    # Place cmask in detector coords
    cmask = sci_to_det(cmask, detid)

    # Place blocked region from coronagraph holder
    if detid=='A2':
        if 'CIRCLYOT' in pupil:
            i1, i2 = [int(920*oversample), int(390*oversample)]
            cmask[0:i1,0:i2]=0
            i1 = int(220*oversample)
            cmask[0:i1,:] = 0
        else:
            i1, i2 = [int(935*oversample), int(393*oversample)]
            cmask[0:i1,0:i2]=0
            i1 = int(235*oversample)
            cmask[0:i1,:] = 0
            
    elif detid=='A4':
        if 'CIRCLYOT' in pupil:
            i1, i2 = [int(920*oversample), int(1463*oversample)]
            cmask[0:i1,i2:]=0
            i1 = int(220*oversample)
            cmask[0:i1,:] = 0
        else:
            i1, i2 = [int(935*oversample), int(1465*oversample)]
            cmask[0:i1,i2:]=0
            i1 = int(235*oversample)
            cmask[0:i1,:] = 0
            
    elif detid=='A5':
        if 'CIRCLYOT' in pupil:
            i1, i2 = [int(1480*oversample), int(270*oversample)]
            cmask[i1:,0:i2]=0
            i1, i2 = [int(1480*oversample), int(1880*oversample)]
            cmask[i1:,i2:]=0
            i1 = int(1825*oversample)
            cmask[i1:,:] = 0
        else:
            i1, i2 = [int(1485*oversample), int(275*oversample)]
            cmask[i1:,0:i2]=0
            i1, i2 = [int(1485*oversample), int(1883*oversample)]
            cmask[i1:,i2:]=0
            i1 = int(1830*oversample)
            cmask[i1:,:] = 0
            
    elif detid=='B1':
        if 'CIRCLYOT' in pupil:
            i1, i2 = [int(910*oversample), int(1615*oversample)]
            cmask[0:i1,i2:]=0
            i1 = int(210*oversample)
            cmask[0:i1,:] = 0
        else:
            i1, i2 = [int(905*oversample), int(1609*oversample)]
            cmask[0:i1,i2:]=0
            i1 = int(205*oversample)
            cmask[0:i1,:] = 0

    elif detid=='B3':
        if 'CIRCLYOT' in pupil:
            i1, i2 = [int(920*oversample), int(551*oversample)]
            cmask[0:i1,0:i2]=0
            i1 = int(210*oversample)
            cmask[0:i1,:] = 0
        else:
            i1, i2 = [int(920*oversample), int(548*oversample)]
            cmask[0:i1,0:i2]=0
            i1 = int(210*oversample)
            cmask[0:i1,:] = 0
    elif detid=='B5':
        if 'CIRCLYOT' in pupil:
            i1, i2 = [int(560*oversample), int(207*oversample)]
            cmask[0:i1,0:i2]=0
            i1, i2 = [int(550*oversample), int(1815*oversample)]
            cmask[0:i1,i2:]=0
            i1 = int(215*oversample)
            cmask[0:i1,:] = 0
        else:
            i1, i2 = [int(560*oversample), int(211*oversample)]
            cmask[0:i1,0:i2]=0
            i1, i2 = [int(550*oversample), int(1819*oversample)]
            cmask[0:i1,i2:]=0
            i1 = int(215*oversample)
            cmask[0:i1,:] = 0

    # Fix SW/LW wedge abuttment
    if detid=='A4':
        if 'CIRCLYOT' in pupil:
            x0 = 819
            x1 = 809
            x2 = x1 + 10
        else:
            x0 = 821
            x1 = 812
            x2 = x1 + 9
        y1, y2 = (400, 650)
        ix0 = int(x0*oversample)
        iy1, iy2 = int(y1*oversample), int(y2*oversample)
        ix1, ix2 = int(x1*oversample), int(x2*oversample)
        cmask[iy1:iy2,ix1:ix2] = cmask[iy1:iy2,ix0].reshape([-1,1])
    elif detid=='A5':
        if 'CIRCLYOT' in pupil:
            x0 = 587
            x1 = x0 + 1
            x2 = x1 + 5
        else:
            x0 = 592
            x1 = x0 + 1
            x2 = x1 + 5
        y1, y2 = (1600, 1750)
        ix0 = int(x0*oversample)
        iy1, iy2 = int(y1*oversample), int(y2*oversample)
        ix1, ix2 = int(x1*oversample), int(x2*oversample)
        cmask[iy1:iy2,ix1:ix2] = cmask[iy1:iy2,ix0].reshape([-1,1])
            
    elif detid=='B3':
        if 'CIRCLYOT' in pupil:
            x0 = 1210
            x1 = 1196
            x2 = x1 + 14
        else:
            x0 = 1204
            x1 = 1192
            x2 = x1 + 12
        y1, y2 = (350, 650)
        ix0 = int(x0*oversample)
        iy1, iy2 = int(y1*oversample), int(y2*oversample)
        ix1, ix2 = int(x1*oversample), int(x2*oversample)
        cmask[iy1:iy2,ix1:ix2] = cmask[iy1:iy2,ix0].reshape([-1,1])
    elif detid=='B5':
        if 'CIRCLYOT' in pupil:
            x0 = 531
            x1 = 525
            x2 = x1 + 6
        else:
            x0 = 535
            x1 = 529
            x2 = x1 + 6
        y1, y2 = (300, 420)
        ix0 = int(x0*oversample)
        iy1, iy2 = int(y1*oversample), int(y2*oversample)
        ix1, ix2 = int(x1*oversample), int(x2*oversample)
        cmask[iy1:iy2,ix1:ix2] = cmask[iy1:iy2,ix0].reshape([-1,1])

    # Convert back to 'sci' orientation
    cmask = det_to_sci(cmask, detid)

    return cmask


def coron_ap_locs(module, channel, mask, pupil=None, full=False):
    """Coronagraph mask aperture locations and sizes

    Returns a dictionary of the detector aperture sizes
    and locations. Attributes 'cen' and 'loc' are in terms
    of (x,y) detector pixels. 'cen_sci' is sci coords location.
    """

    if channel=='long':
        channel = 'LW'
    elif channel=='short':
        channel = 'SW'
    
    if pupil is None:
        pupil = 'WEDGELYOT' if 'WB' in mask else 'CIRCLYOT'

    if module=='A':
        if channel=='SW':
            if '210R' in mask:
                cdict_rnd = {'det':'A2', 'cen':(712,525), 'size':640}
                cdict_bar = {'det':'A2', 'cen':(716,536), 'size':640}
            elif '335R' in mask:
                cdict_rnd = {'det':'A2', 'cen':(1368,525), 'size':640}
                cdict_bar = {'det':'A2', 'cen':(1372,536), 'size':640}
            elif '430R' in mask:
                cdict_rnd = {'det':'A2', 'cen':(2025,525), 'size':640}
                cdict_bar = {'det':'A2', 'cen':(2029,536), 'size':640}
            elif 'SWB' in mask:
                cdict_rnd = {'det':'A4', 'cen':(487,523), 'size':640}
                cdict_bar = {'det':'A4', 'cen':(490,536), 'size':640}
            elif 'LWB' in mask:
                cdict_rnd = {'det':'A4', 'cen':(1141,523), 'size':640}
                cdict_bar = {'det':'A4', 'cen':(1143,536), 'size':640}
            else:
                raise ValueError('Mask {} not recognized for {} channel'\
                                 .format(mask, channel))
        elif channel=='LW':
            if '210R' in mask:
                cdict_rnd = {'det':'A5', 'cen':(1720, 1672), 'size':320}
                cdict_bar = {'det':'A5', 'cen':(1725, 1682), 'size':320}
            elif '335R' in mask:
                cdict_rnd = {'det':'A5', 'cen':(1397,1672), 'size':320}
                cdict_bar = {'det':'A5', 'cen':(1402,1682), 'size':320}
            elif '430R' in mask:
                cdict_rnd = {'det':'A5', 'cen':(1074,1672), 'size':320}
                cdict_bar = {'det':'A5', 'cen':(1078,1682), 'size':320}
            elif 'SWB' in mask:
                cdict_rnd = {'det':'A5', 'cen':(752,1672), 'size':320}
                cdict_bar = {'det':'A5', 'cen':(757,1682), 'size':320}
            elif 'LWB' in mask:
                cdict_rnd = {'det':'A5', 'cen':(430,1672), 'size':320}
                cdict_bar = {'det':'A5', 'cen':(435,1682), 'size':320}
            else:
                raise ValueError('Mask {} not recognized for {} channel'\
                                 .format(mask, channel))
        else:
            raise ValueError('Channel {} not recognized'.format(channel))


    elif module=='B':
        if channel=='SW':
            if '210R' in mask:
                cdict_rnd = {'det':'B1', 'cen':(1293,513), 'size':640}
                cdict_bar = {'det':'B1', 'cen':(1287,508), 'size':640}
            elif '335R' in mask:
                cdict_rnd = {'det':'B1', 'cen':(637,513), 'size':640}
                cdict_bar = {'det':'B1', 'cen':(632,508), 'size':640}
            elif '430R' in mask:
                cdict_rnd = {'det':'B1', 'cen':(-20,513), 'size':640}
                cdict_bar = {'det':'B1', 'cen':(-25,508), 'size':640}
            elif 'SWB' in mask:
                cdict_rnd = {'det':'B3', 'cen':(874,519), 'size':640}
                cdict_bar = {'det':'B3', 'cen':(870,516), 'size':640}
            elif 'LWB' in mask:
                cdict_rnd = {'det':'B3', 'cen':(1532,519), 'size':640}
                cdict_bar = {'det':'B3', 'cen':(1526,516), 'size':640}
            else:
                raise ValueError('Mask {} not recognized for {} channel'\
                                 .format(mask, channel))
        elif channel=='LW':
            if '210R' in mask:
                cdict_rnd = {'det':'B5', 'cen':(1656,360), 'size':320}
                cdict_bar = {'det':'B5', 'cen':(1660,360), 'size':320}
            elif '335R' in mask:
                cdict_rnd = {'det':'B5', 'cen':(1334,360), 'size':320}
                cdict_bar = {'det':'B5', 'cen':(1338,360), 'size':320}
            elif '430R' in mask:
                cdict_rnd = {'det':'B5', 'cen':(1012,360), 'size':320}
                cdict_bar = {'det':'B5', 'cen':(1015,360), 'size':320}
            elif 'SWB' in mask:
                cdict_rnd = {'det':'B5', 'cen':(366,360), 'size':320}
                cdict_bar = {'det':'B5', 'cen':(370,360), 'size':320}
            elif 'LWB' in mask:
                cdict_rnd = {'det':'B5', 'cen':(689,360), 'size':320}
                cdict_bar = {'det':'B5', 'cen':(693,360), 'size':320}
            else:
                raise ValueError('Mask {} not recognized for {} channel'\
                                 .format(mask, channel))
        else:
            raise ValueError('Channel {} not recognized'.format(channel))

    else:
        raise ValueError('Module {} not recognized'.format(module))

    # Choose whether to use round or bar Lyot mask
    cdict = cdict_rnd if 'CIRC' in pupil else cdict_bar

    x0, y0 = np.array(cdict['cen']) - cdict['size']/2
    cdict['loc'] = (int(x0), int(y0))


    # Add in 'sci' coordinates (V2/V3 orientation)
    # X is flipped for A5, Y is flipped for all others
    cen = cdict['cen']
    if cdict['det'] == 'A5':
        cdict['cen_sci'] = (2048-cen[0], cen[1])
    else:
        cdict['cen_sci'] = (cen[0], 2048-cen[1])

    if full:
        cdict['size'] = 2048
        cdict['loc'] = (0,0)

    return cdict

def coron_detector(mask, module, channel=None):
    """
    Return detector name for a given coronagraphic mask, module,
    and channel.
    """
    
    # Grab default channel
    if channel is None:
        if ('210R' in mask) or ('SW' in mask):
            channel = 'SW'
        else:
            channel = 'LW'
    
    # If LW, always A5 or B5
    # If SW, bar masks are A4/B3, round masks A2/B1; M430R is invalid
    if channel=='LW':
        detname = module + '5'
    elif (channel=='SW') and ('430R' in mask):
        raise AttributeError("MASK430R not valid for SW channel")
    else:
        if module=='A':
            detname = 'A2' if mask[-1]=='R' else 'A4'
        else:
            detname = 'B1' if mask[-1]=='R' else 'B3'
            
    return detname
