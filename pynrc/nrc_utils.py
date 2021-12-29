"""pyNRC utility functions"""
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

from webbpsf_ext.bandpasses import nircam_com_th, nircam_com_nd

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
from poppy import (measure_sharpness, measure_centroid) #, measure_strehl)

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


###########################################################################
#    pysiaf
###########################################################################

import pysiaf
from pysiaf import JWST_PRD_VERSION, rotations, Siaf

# Create this once since it takes time to call multiple times
siaf_nrc = Siaf('NIRCam')
siaf_nrc.generate_toc()


#__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
#__location__ += '/'

__epsilon = np.finfo(float).eps


###########################################################################
#
#    Pysynphot Bandpasses
#
###########################################################################

from webbpsf_ext.bandpasses import bp_igood, bp_wise, bp_2mass, bp_gaia
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


def get_detname(det_id):
    """Return NRC[A-B][1-5] for valid detector/SCA IDs"""

    det_dict = {481:'A1', 482:'A2', 483:'A3', 484:'A4', 485:'A5',
                486:'B1', 487:'B2', 488:'B3', 489:'B4', 490:'B5'}
    scaids = det_dict.keys()
    detids = det_dict.values()
    detnames = ['NRC' + idval for idval in detids]

    # If already valid, then return
    if det_id in detnames:
        return det_id
    elif det_id in scaids:
        detname = 'NRC' + det_dict[det_id]
    elif det_id in detids:
        detname = 'NRC' + det_id
    else:
        detname = det_id

    # If NRCALONG or or NRCBLONG, change 'LONG' to '5' 
    if 'LONG' in detname:
        detname = detname[0:4] + '5'

    if detname not in detnames:
        raise ValueError("Invalid detector: {} \n\tValid names are: {}" \
                  .format(detname, ', '.join(detnames)))
        
    return detname

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
from webbpsf_ext.spectra import bin_spectrum, mag_to_counts

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
            try:
                bkg = jbt.background(ra, dec, wave_bath)
            except:
                _log.error('Cannot reach JWST Background servers. Reverting to `zfact` input.')
            else:
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
    im_com = build_mask_detid(module+'5', filter=filter)
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


def make_grism_slope(nrc, src_tbl, tel_pointing, expnum, add_offset=None, **kwargs):
    """ Create slope image 
    """
    
    # Set SIAF aperture info
    siaf_ap_obs = tel_pointing.siaf_ap_obs
    ap_siaf = siaf_ap_obs
    ap_obs = ap_siaf.AperName
    
    # Convert expnum to int if input as string
    expnum = int(expnum) if isinstance(expnum, str) else expnum
    # Grab poining info for specific exposure number
    ind = np.where(tel_pointing.exp_nums == expnum)[0][0]
    
    # Include a offset shift to better reposition spectrum?
    add_offset = np.array([0,0]) if add_offset is None else np.asarray(add_offset)
    idl_off = np.array([(tel_pointing.position_offsets_act[ind])]) + add_offset
    
    ra_deg  = src_tbl['ra'].to('deg').value
    dec_deg = src_tbl['dec'].to('deg').value
    mags    = src_tbl[nrc.filter].data
    try:
        teff = src_tbl['Teff']
        sptype = None
    except:
        teff = None
        sptype = src_tbl['SpType']
    
    # Convert all RA/Dec values to V3/V3
    v2_obj, v3_obj = tel_pointing.radec_to_frame((ra_deg, dec_deg), frame_out='tel', idl_offsets=idl_off)
    
    # Get pixel locations
    xpix, ypix = ap_siaf.tel_to_sci(v2_obj, v3_obj)
    # Pickoff mirror information
    x1, x2, y1, y2 = pickoff_xy(ap_obs)
    
    # Mask out all sources that are outside pick-off mirror
    mask_pom = ((xpix>x1) & (xpix<x2-1)) & ((ypix>y1) & (ypix<y2-1))

    # Mask out all sources that will not contribute to final slope image
    wspec, imspec_temp = nrc.calc_psf_from_coeff(return_oversample=False, return_hdul=False)
    ypsf, xpsf = imspec_temp.shape
    
    xmin, ymin = -1*np.array([xpsf,ypsf]).astype('int') / 2 - 1
    xmax = int(ap_siaf.XSciSize + xpsf / 2 + 1)
    ymax = int(ap_siaf.YSciSize + ypsf / 2 + 1)

    xmask = (xpix>=xmin) & (xpix<=xmax)
    ymask = (ypix>=ymin) & (ypix<=ymax)

    # Final mask
    mask = mask_pom & xmask & ymask

    # Select final positions and spectral type information
    xpix = xpix[mask]
    ypix = ypix[mask]
    mags_field = mags[mask]
    teff   = teff[mask]   if teff   is not None else None
    sptype = sptype[mask] if sptype is not None else None
    # src_flux = mag_to_counts(mags_field, nrc.bandpass, **kwargs)

    _log.info(np.array([xpix, ypix, mags_field]).T)

    # Get undeviated wavelength
    wref = grism_wref(nrc.pupil_mask, nrc.module)

    nx, ny = (nrc.Detector.xpix, nrc.Detector.ypix)
    im_slope = np.zeros([ny,nx])

    # Build final image
    wspec_all = []
    for i in range(xpix.size):
        # Get stellar spectrum
        teff_i = teff[i]   if teff   is not None else None
        sptp_i = sptype[i] if sptype is not None else 'G2V'
        sp = stellar_spectrum(sptp_i, mags_field[i], 'vegamag', nrc.bandpass,
                              Teff=teff_i, metallicity=0, log_g=4.5)

        # Create spectral image
        xr, yr = xpix[i], ypix[i]
        wspec, imspec = place_grism_spec(nrc, sp, xr, yr, wref=wref, return_oversample=False)

        im_slope += imspec
        wspec_all.append(wspec)
        del imspec

    wspec_all = np.asarray(wspec_all)

    return wspec_all, im_slope

def place_grism_spec(nrc, sp, xpix, ypix, wref=None, return_oversample=False):
    """ Create spectral image and place ref wavelenght at (x,y) location 
    
    Given a NIRCam instrument object and input spectrum, create a dispersed
    PSF and place the undeviated reference wavelength at the specified
    (xpix,ypix) coordinates (assuming 'sci' coords). 

    Returned values will be a tuple of (wspec, imspec) 

    """

    nx, ny = (nrc.Detector.xpix, nrc.Detector.ypix)
    oversample = nrc.oversample
    nx_over = nx * oversample
    ny_over = ny * oversample

    pupil_mask = nrc.pupil_mask
    if pupil_mask is None:
        _log.warn('place_grism_spec: NIRCam pupil mask set to None. Should be GRISMR or GRISMC.')
        pupil_mask = 'NONE'

    # Determine reference wavelength
    if wref is None:
        if 'GRISMC' in pupil_mask:
            pupil = 'GRISMC'
        elif 'GRISM' in pupil_mask:
            pupil = 'GRISMR'
        else: # generic grism
            pupil = 'GRISM'
        wref = grism_wref(pupil, nrc.module)

    # Create image
    wspec, imspec = nrc.calc_psf_from_coeff(sp=sp, return_hdul=False, return_oversample=True)

    # Place undeviated wavelength at (xpix,ypix) location
    xr, yr = (np.array([xpix, ypix]) - 0.5) * oversample
    yshift = yr - ny_over/2

    # Empirically determine shift value in dispersion direction
    wnew_temp = pad_or_cut_to_size(wspec, nx_over)
    # Index of reference wavelength associated with ref pixel
    imask = (wnew_temp>wref-0.01) & (wnew_temp<wref+0.01)
    ix_ref = np.interp(wref, wnew_temp[imask], np.arange(nx_over)[imask])
    xshift = xr - ix_ref

    # Shift and crop to output size
    imspec = pad_or_cut_to_size(imspec, (ny_over,nx_over), offset_vals=(yshift,xshift), fill_val=np.nan)
    wspec  = pad_or_cut_to_size(wspec, nx_over, offset_vals=xshift, fill_val=np.nan)

    # Remove NaNs in image
    ind_nan = np.isnan(imspec)
    imspec[ind_nan] = np.min(imspec[~ind_nan])

    # Fill NaNs in wavelength solution with linear extrapolation
    ind_nan = np.isnan(wspec)
    arr = np.arange(nx_over)
    cf = jl_poly_fit(arr[~ind_nan], wspec[~ind_nan])
    wspec[ind_nan] = jl_poly(arr[ind_nan], cf)

    if return_oversample:
        return wspec, imspec
    else:
        wspec = frebin(wspec, scale=1/oversample, total=False)
        imspec = frebin(imspec, scale=1/oversample)
        return wspec, imspec


def place_grismr_tso(waves, imarr, siaf_ap, wref=None, im_coords='sci'):
    """
    Shift image such that undeviated wavelength sits at the
    SIAF aperture reference location.

    Return image in sience coords.

    Parameters
    ==========
    waves : ndarray
        Wavelength solution of input image
    imarr : ndarray
        Input dispersed grism PSF (can be multiple PSFs)
    siaf_ap : pysiaf aperture
        Grism-specific SIAF aperture class to determine final
        subarray size and reference point
    
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



###########################################################################
#
#    Pick-off images for a given module
#
###########################################################################

def pickoff_xy(ap_obs_name):
    """
    Return pickoff mirror FoV x/y limits in terms of science pixel coordinates
    
    ap_obs : Aperture to create observation (e.g., 'NRCA5_FULL')
    """

    ap_siaf = siaf_nrc[ap_obs_name]
    module = ap_obs_name[3:4]

    # Determine pick-off mirror FoV from 
    ap1 = siaf_nrc['NRC{}5_GRISMC_WFSS'.format(module)]
    ap2 = siaf_nrc['NRC{}5_GRISMR_WFSS'.format(module)]
    ap3 = siaf_nrc['NRCA5_FULL_MASK335R']

    # V2/V3 coordinates of pick-off FoV
    v2_1, v3_1 = ap1.corners('tel', False)
    v2_2, v3_2 = ap2.corners('tel', False)
    v2_3, v3_3 = ap3.corners('tel', False)
    if module == 'B': 
        v2_3 *= -1
    v2_all = np.array([v2_1, v2_2, v2_3]).flatten()
    v3_all = np.array([v3_1, v3_2, v3_3]).flatten()

    # Convert to science pixel positions
    x_new, y_new = ap_siaf.tel_to_sci(v2_all, v3_all)
    # sci pixel values are use are X.5
    x1, x2 = np.array([x_new.min(), x_new.max()]).astype(np.int) + 0.5
    y1, y2 = np.array([y_new.min(), y_new.max()]).astype(np.int) + 0.5

    return (x1, x2, y1, y2)


def pickoff_image(ap_obs, v2_obj, v3_obj, flux_obj, oversample=1):
    """
    Create an unconvolved image of filled pixel values that have 
    been shifted via bilinear interpolation. The image will then
    be convolved with a PSF to create the a focal plane image that
    is the size of the NIRCam pick-off mirror. This image should
    then be cropped to generate the final detector image.

    Returns the tuple (xsci, ysci, image), where xsci and ysci are
    the science coordinates associated with the image.
    
    Parameters
    ==========
    ap_obs : str
        Name of aperture in which the observation is taking place.
        Necessary to determine pixel locations for stars.
    v2_obj : ndarray
        List of V2 coordiantes of stellar sources
    v3_obj : ndarray
        List of V3 coordinates of stellar sources
    flux_obj : ndarray
        List of fluxes (e-/sec) for each source
    
    Keyword Args
    ============
    oversample : int
        If set, the returns an oversampled version of the image to
        convolve with PSFs. If set to one, then detector pixels.
    """
    
    from scipy.interpolate import interp2d

    # xpix and ypix locations in science orientation
    ap_siaf = siaf_nrc[ap_obs]

    xpix, ypix = ap_siaf.tel_to_sci(v2_obj, v3_obj)
    x1, x2, y1, y2 = pickoff_xy(ap_obs)

    # Mask all sources that are outside pick-off mirror
    mask = ((xpix>x1) & (xpix<x2-1)) & ((ypix>y1) & (ypix<y2-1))
    xpix = xpix[mask]
    ypix = ypix[mask]
    src_flux = flux_obj[mask]

    # Create oversized and oversampled image
    ys = int((y2 - y1) * oversample)
    xs = int((x2 - x1) * oversample)
    oversized_image = np.zeros([ys,xs])
    
    # X and Y detector pixel values
    dstep = 1/oversample
    xsci = np.arange(x1, x2, dstep)
    ysci = np.arange(y1, y2, dstep)

    # Zero-based (x,y) locations for oversized images
    xvals_os = (xpix - x1) * oversample
    yvals_os = (ypix - y1) * oversample

    # separate into an integers and fractions
    intx = xvals_os.astype(np.int)
    inty = yvals_os.astype(np.int)
    fracx = xvals_os - intx
    fracy = yvals_os - inty
    
    # flip negative shift values
    ind = fracx < 0
    fracx[ind] += 1
    intx[ind] -= 1
    ind = fracy<0
    fracy[ind] += 1
    inty[ind] -= 1

    # Bilinear interpolation of all sources
    val1 = src_flux * ((1-fracx)*(1-fracy))
    val2 = src_flux * ((1-fracx)*fracy)
    val3 = src_flux * ((1-fracy)*fracx)
    val4 = src_flux * (fracx*fracy)

    # Add star-by-star in case of overlapped indices
    for i, (iy, ix) in enumerate(zip(inty,intx)):
        oversized_image[iy,   ix]   += val1[i]
        oversized_image[iy+1, ix]   += val2[i]
        oversized_image[iy,   ix+1] += val3[i]
        oversized_image[iy+1, ix+1] += val4[i]
        
    #print("NStars: {}".format(len(intx)))
    
    return xsci, ysci, oversized_image 


def gen_unconvolved_point_source_image(nrc, tel_pointing, ra_deg, dec_deg, mags, 
                                       expnum=1, osamp=1, siaf_ap_obs=None, **kwargs):
    """ Create an unconvolved image with sub-pixel shifts
    
    """    
    from .obs_nircam import attenuate_with_coron_mask, gen_coron_mask
    
    # Observation aperture
    siaf_ap_obs = nrc.siaf_ap if siaf_ap_obs is None else siaf_ap_obs
    ap_obs_name = siaf_ap_obs.AperName
    
    # Get all source fluxes
    # mags = tbl[nrc.filter].data
    flux_obj = mag_to_counts(mags, nrc.bandpass, **kwargs)

    if isinstance(expnum, str):
        expnum = int(expnum)
    
    ind = np.where(tel_pointing.exp_nums == expnum)[0][0]

    # Convert RA, Dec coordiantes into V2/V3 (arcsec)
    # ra_deg, dec_deg = (tbl['ra'], tbl['dec'])
    idl_off = [tel_pointing.position_offsets_act[ind]]
    v2_obj, v3_obj = tel_pointing.radec_to_frame((ra_deg, dec_deg), frame_out='tel', idl_offsets=idl_off)
    
    # Create initial POM image, then contract to reasonable size
    xsci, ysci, im_pom = pickoff_image(ap_obs_name, v2_obj, v3_obj, flux_obj, oversample=osamp)

    # Crop based on subarray window size
    # Maximum required size depends on PSF and detector readout size

    # Min and max sci coordinates to keep
    xmin = ymin = int(-nrc.fov_pix/2 - 1)
    xmax = int(siaf_ap_obs.XSciSize + nrc.fov_pix/2 + 1)
    ymax = int(siaf_ap_obs.YSciSize + nrc.fov_pix/2 + 1)

    xmask = (xsci>=xmin) & (xsci<xmax)
    ymask = (ysci>=ymin) & (ysci<ymax)

    # Keep only regions that contribute to final convolved image
    xsci = xsci[xmask]
    ysci = ysci[ymask]
    im_sci = im_pom[ymask][:,xmask]

    # Attenuate image by coronagraphic mask features (ND squaures and COM holder)
    if nrc.is_coron and (not nrc.ND_acq):
        try:
            cmask = nrc.mask_images['OVERSAMP']
        except:
            mask_dict = gen_coron_mask(nrc)
            cmask = mask_dict['OVERSAMP']

        if cmask is not None:
            # Make mask image same on-sky size as im_sci
            ny_over, nx_over = np.array(im_sci.shape) * nrc.oversample / osamp
            x0, y0 = (np.min(xsci), np.min(ysci))
            x1 = int(np.abs(x0*nrc.oversample))
            y1 = int(np.abs(y0*nrc.oversample))
            x2, y2 = (x1 + cmask.shape[1], y1 + cmask.shape[0])
            pad_vals = ((y1,int(ny_over-y2)), (x1,int(nx_over-x2)))
            cmask_oversized = np.pad(cmask, pad_vals, mode='edge')

            # Rebin to im_sci sampling
            cmask_oversized = frebin(cmask_oversized, dimensions=im_sci.shape, total=False)
            # Perform attenuation
            im_sci = attenuate_with_coron_mask(nrc, im_sci, cmask_oversized)
    
    # Make science image HDUList from 
    hdul_sci_image = fits.HDUList([fits.PrimaryHDU(im_sci)])
    hdul_sci_image[0].header['PIXELSCL'] = nrc.pixelscale / osamp
    hdul_sci_image[0].header['OSAMP'] = osamp
    hdul_sci_image[0].header['INSTRUME'] = nrc.name
    hdul_sci_image[0].header['APERNAME'] = ap_obs_name
    
    # Get X and Y indices corresponding to aperture reference
    xind_ref = np.argmin(np.abs(xsci - siaf_ap_obs.XSciRef))
    yind_ref = np.argmin(np.abs(ysci - siaf_ap_obs.YSciRef))
    hdul_sci_image[0].header['XIND_REF'] = (xind_ref, "x index of aperture reference")
    hdul_sci_image[0].header['YIND_REF'] = (yind_ref, "y index of aperture reference")
    hdul_sci_image[0].header['XSCI0']    = (np.min(xsci), "xsci value at (x,y)=(0,0) corner")
    hdul_sci_image[0].header['YSCI0']    = (np.min(ysci), "ysci value at (x,y)=(0,0) corner")
    hdul_sci_image[0].header['CFRAME'] = 'sci'

    # print(im_pom.shape, im_sci.shape)
    
    return hdul_sci_image

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
                 'F200W': 2.229, 'F277W': 3.144, 'F356W': 3.971, 'F444W': 4.992}
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
    nd_squares=True, shift_x=None, shift_y=None, filter=None):
    """
    Build a transmission image of a coronagraphic mask spanning
    the 20" coronagraphic FoV.

    oversample is used only if pixelscale is set to None.

    Returns the intensity transmission (square of the amplitude transmission). 
    """

    from webbpsf.optics import NIRCam_BandLimitedCoron

    shifts = {'shift_x': shift_x, 'shift_y': shift_y}

    bar_offset = None
    if name=='MASK210R':
        pixscale = pixscale_SW
        channel = 'short'
        filter = 'F210M' if filter is None else filter
    elif name=='MASK335R':
        pixscale = pixscale_LW
        channel = 'long'
        filter = 'F335M' if filter is None else filter
    elif name=='MASK430R':
        pixscale = pixscale_LW
        channel = 'long'
        filter = 'F430M' if filter is None else filter
    elif name=='MASKSWB':
        pixscale = pixscale_SW
        channel = 'short'
        filter = 'F210M' if filter is None else filter
        bar_offset = 0
    elif name=='MASKLWB':
        pixscale = pixscale_LW
        channel = 'long'
        filter = 'F430M' if filter is None else filter
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
    
    # Square the amplitude transmission to get intensity transmission
    im = mask.get_transmission(wave)**2    

    return im


def build_mask(module='A', pixscale=None, filter=None, nd_squares=True):
    """Create coronagraphic mask image

    Return a truncated image of the full coronagraphic mask layout
    for a given module. Assumes each mask is exactly 20" across.

    +V3 is up, and +V2 is to the left.
    """
    if module=='A':
        names = ['MASK210R', 'MASK335R', 'MASK430R', 'MASKSWB', 'MASKLWB']
    elif module=='B':
        names = ['MASKSWB', 'MASKLWB', 'MASK430R', 'MASK335R', 'MASK210R']

    if pixscale is None:
        pixscale=pixscale_LW

    npix = int(20 / pixscale + 0.5)
    allims = []
    for name in names:
        res = coron_trans(name, module=module, pixelscale=pixscale, npix=npix, nd_squares=nd_squares)
        allims.append(res)
    im_out = np.concatenate(allims, axis=1)

    # Multiply COM throughputs sampled at filter wavelength
    if filter is not None:
        bandpass = read_filter(filter)
        w_um = bandpass.avgwave() / 1e4
        com_th = nircam_com_th(wave_out=w_um)
        com_nd = 10**(-1*nircam_com_nd(wave_out=w_um))

        ind_nd = (im_out<0.0011) & (im_out>0.0009)
        im_out[ind_nd] = com_nd
        im_out *= com_th

    return im_out


def build_mask_detid(detid, oversample=1, ref_mask=None, pupil=None, filter=None, 
    nd_squares=True, mask_holder=True):
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
    allims = []
    for cname in cnames:
        res = coron_trans(cname, module=module, pixelscale=pixscale_over, npix=npix_large, 
                          filter=filter, nd_squares=nd_squares)
        allims.append(res)
    
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

    # Multiply COM throughputs sampled at filter wavelength
    if filter is not None:
        bandpass = read_filter(filter)
        w_um = bandpass.avgwave() / 1e4
        com_th = nircam_com_th(wave_out=w_um)
        com_nd = 10**(-1*nircam_com_nd(wave_out=w_um))

        ind_nd = (cmask<0.0011) & (cmask>0.0009)
        cmask[ind_nd] = com_nd
        cmask *= com_th

    # Place cmask in detector coords
    cmask = sci_to_det(cmask, detid)

    ############################################
    # Place blocked region from coronagraph holder
    # Also ensure region outside of COM has throughput=1
    if mask_holder:
        if detid=='A2':
            if 'CIRCLYOT' in pupil:
                i1, i2 = [int(920*oversample), int(390*oversample)]
                cmask[0:i1,0:i2] = 0
                cmask[i1:,0:i2]  = 1
                i1 = int(220*oversample)
                cmask[0:i1,:] = 0
                i2 = int(974*oversample)
                cmask[i2:,:] = 1
            else:
                i1, i2 = [int(935*oversample), int(393*oversample)]
                cmask[0:i1,0:i2] = 0
                cmask[i1:, 0:i2] = 1
                i1 = int(235*oversample)
                cmask[0:i1,:] = 0
                i2 = int(985*oversample)
                cmask[i2:,:] = 1
                
        elif detid=='A4':
            if 'CIRCLYOT' in pupil:
                i1, i2 = [int(920*oversample), int(1463*oversample)]
                cmask[0:i1,i2:] = 0
                cmask[i1:, i2:] = 1
                i1 = int(220*oversample)
                cmask[0:i1,:] = 0
                i2 = int(974*oversample)
                cmask[i2:,:] = 1
            else:
                i1, i2 = [int(935*oversample), int(1465*oversample)]
                cmask[0:i1,i2:] = 0
                cmask[i1:, i2:] = 1
                i1 = int(235*oversample)
                cmask[0:i1,:] = 0
                i2 = int(985*oversample)
                cmask[i2:,:] = 1
                
        elif detid=='A5':
            if 'CIRCLYOT' in pupil:
                i1, i2 = [int(1480*oversample), int(270*oversample)]
                cmask[i1:,0:i2]  = 0
                cmask[0:i1,0:i2] = 1
                i1, i2 = [int(1480*oversample), int(1880*oversample)]
                cmask[i1:,i2:]  = 0
                cmask[0:i1,i2:] = 1
                i1 = int(1825*oversample)
                cmask[i1:,:] = 0
                i2 = int(1452*oversample)
                cmask[0:i2,:] = 1
            else:
                i1, i2 = [int(1485*oversample), int(275*oversample)]
                cmask[i1:,0:i2]  = 0
                cmask[0:i1,0:i2] = 1
                i1, i2 = [int(1485*oversample), int(1883*oversample)]
                cmask[i1:,i2:]  = 0
                cmask[0:i1,i2:] = 1
                i1 = int(1830*oversample)
                cmask[i1:,:] = 0
                i2 = int(1462*oversample)
                cmask[0:i2,:] = 1
                
        elif detid=='B1':
            if 'CIRCLYOT' in pupil:
                i1, i2 = [int(910*oversample), int(1615*oversample)]
                cmask[0:i1,i2:] = 0
                cmask[i1:,i2:]  = 1
                i1 = int(210*oversample)
                cmask[0:i1,:] = 0
                i2 = int(956*oversample)
                cmask[i2:,:] = 1
            else:
                i1, i2 = [int(905*oversample), int(1609*oversample)]
                cmask[0:i1,i2:] = 0
                cmask[i1:,i2:]  = 1
                i1 = int(205*oversample)
                cmask[0:i1,:] = 0
                i2 = int(951*oversample)
                cmask[i2:,:] = 1

        elif detid=='B3':
            if 'CIRCLYOT' in pupil:
                i1, i2 = [int(920*oversample), int(551*oversample)]
                cmask[0:i1,0:i2] = 0
                cmask[i1:,0:i2]  = 1
                i1 = int(210*oversample)
                cmask[0:i1,:] = 0
                i2 = int(966*oversample)
                cmask[i2:,:] = 1
            else:
                i1, i2 = [int(920*oversample), int(548*oversample)]
                cmask[0:i1,0:i2] = 0
                cmask[i1:,0:i2]  = 1
                i1 = int(210*oversample)
                cmask[0:i1,:] = 0
                i2 = int(963*oversample)
                cmask[i2:,:] = 1

        elif detid=='B5':
            if 'CIRCLYOT' in pupil:
                i1, i2 = [int(555*oversample), int(207*oversample)]
                cmask[0:i1,0:i2] = 0
                cmask[i1:, 0:i2] = 1
                i1, i2 = [int(545*oversample), int(1815*oversample)]
                cmask[0:i1,i2:] = 0
                cmask[i1:, i2:] = 1
                i1 = int(215*oversample)
                cmask[0:i1,:] = 0
                i2 = int(578*oversample)
                cmask[i2:,:] = 1
            else:
                i1, i2 = [int(555*oversample), int(211*oversample)]
                cmask[0:i1,0:i2] = 0 
                cmask[i1:, 0:i2] = 1
                i1, i2 = [int(545*oversample), int(1819*oversample)]
                cmask[0:i1,i2:] = 0
                cmask[i1:, i2:] = 1
                i1 = int(215*oversample)
                cmask[0:i1,:] = 0
                i2 = int(578*oversample)
                cmask[i2:,:] = 1

    ############################################
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
