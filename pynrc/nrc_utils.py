"""pyNRC utility functions"""
from copy import deepcopy
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

# Progress bar
from tqdm.auto import trange, tqdm

from astropy.io import fits, ascii
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
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
    from webbpsf_ext.utils import get_one_siaf
except ImportError:
    raise ImportError('webbpsf_ext is not installed. pyNRC depends on its inclusion.')

# Some useful functions for displaying and measuring PSFs
import webbpsf, poppy
from poppy import (radial_profile, measure_radial, measure_fwhm, measure_ee)
from poppy import (measure_sharpness, measure_centroid) #, measure_strehl)

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
from webbpsf_ext.utils import siaf_nrc as siaf_nrc_wext
siaf_nrc = deepcopy(siaf_nrc_wext)
siaf_nrc.generate_toc()

#__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
#__location__ += '/'

__epsilon = np.finfo(float).eps


###########################################################################
#
#    Bandpasses
#
###########################################################################

from webbpsf_ext.bandpasses import bp_igood, bp_wise, bp_2mass, bp_gaia
from webbpsf_ext.bandpasses import nircam_filter as read_filter
from webbpsf_ext.bandpasses import nircam_com_th, nircam_com_nd


###########################################################################
#
#    Sensitivities and Saturation Limits
#
###########################################################################

from webbpsf_ext.bandpasses import nircam_grism_res as grism_res
from webbpsf_ext.bandpasses import nircam_grism_wref as grism_wref
from webbpsf_ext.maths import radial_std
from webbpsf_ext.utils import get_detname

def channel_select(bp):
    """Select wavelength channel

    Based on input bandpass, return the pixel scale, dark current, and
    excess read noise parameters. These values are typical for either
    a SW or LW NIRCam detector.

    Parameters
    ----------
    bp : :class:`webbpsf_ext.synphot_ext.Bandpass`
        NIRCam filter bandpass.
    """

    if bp.avgwave().to_value('um') < 2.3:
        pix_scale = pixscale_SW # pixel scale (arcsec/pixel)
        idark = 0.003      # dark current (e/sec)
        pex = (1.0,5.0)
    else:
        pix_scale = pixscale_LW
        idark = 0.03
        pex = (1.5,10.0)

    return (pix_scale, idark, pex)

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
#    Spectrum Wrappers
#
###########################################################################

from webbpsf_ext.spectra import BOSZ_spectrum, stellar_spectrum, source_spectrum
from webbpsf_ext.spectra import planets_sb12, sp_accr, jupiter_spec, companion_spec
from webbpsf_ext.spectra import linder_table, linder_filter, cond_table, cond_filter
from webbpsf_ext.spectra import bin_spectrum, mag_to_counts

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
    :mod:`webbpsf_ext.synphot_ext..Spectrum`
        Output is a synphot spectrum with default units of flam (erg/s/cm^2/A/sr).
        Note: synphot doesn't recognize that it's per steradian, but we must keep
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

    from webbpsf_ext.synphot_ext import ArraySpectrum, BlackBody, convert_flux
    
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
                        _log.warning("The input calendar day {}".format(thisday)+" is not available. Choosing closest visible day.")
                        diff = np.abs(calendar-thisday)
                        ind = np.argmin(diff)
                        ftot = farr[ind]

                sp = ArraySpectrum(wave=wvals*1e4, flux=ftot*1e6, fluxunits='Jy')
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
    # in order to work with synphot's blackbody function.
    # synphot's BB function is normalized to 1Rsun at 1kpc by default.
    f1 *= 4.0e7
    f2 *= 2.0e13

    bb1 = f1 * BlackBody(5300.0)
    bb2 = f2 * BlackBody(282.0)

    # Query Euclid Background Model
    locstr = kwargs.get('locstr')
    year  = kwargs.get('year')
    day   = kwargs.get('day')
    if (locstr is not None) and (year is not None) and (day is not None):

        # Wavelengths in um and values in MJy
        waves = np.array([1.0,5.5])
        vals = zodi_euclid(locstr, year, day, waves, **kwargs)

        # MJy at wavelength locations
        uwaves = waves * u.um
        f_bb1 = convert_flux(uwaves, bb1(uwaves), 'MJy').value
        f_bb2 = convert_flux(uwaves, bb2(uwaves), 'MJy').value

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
# 	bb1 = BlackBody(5800.); bb2 = BlackBody(300.)
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
    sp_bg : :mod:`webbpsf_ext.synphot_ext.Spectrum`
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

    from webbpsf_ext.synphot_ext import Observation
    
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
        obs_bg = Observation(sp_bg, bp, bp.wave)

        # Total background flux per pixel (not dispersed)
        area_scale = (pix_scale/206265.0)**2
        fbg_tot = obs_bg.countrate() * area_scale
        # Total counts/sec within each wavelength bin
        binwave = obs_bg.binset.to_value('um')
        binflux = area_scale * obs_bg.sample_binned(flux_unit='count').value
            
        # Interpolation function
        fint = interp1d(binwave, binflux, kind='cubic')
        # Wavelengths at each pixel to interpolate
        wave_vals = np.arange(binwave.min(), binwave.max(), dw)
        # Get flux values and preserve total flux
        flux_vals = fint(wave_vals)
        flux_vals = fbg_tot * flux_vals / flux_vals.sum()
    
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
    
    from webbpsf_ext.synphot_ext import Observation
    
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
    obs_bg = Observation(sp_bg, bp, bp.wave)

    # Total background flux per pixel (not dispersed)
    area_scale = (pix_scale/206265.0)**2
    fbg_tot = obs_bg.countrate() * area_scale
    # Total counts/sec within each wavelength bin
    binwave = obs_bg.binset.to_value('um')
    binflux = area_scale * obs_bg.sample_binned(flux_unit='count').value

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


def make_grism_slope(nrc, src_tbl, tel_pointing, expnum, add_offset=None, spec_ang=0, **kwargs):
    """ Create grism slope image 
    Parameters
    """
    
    # Set SIAF aperture info
    ap_siaf = tel_pointing.siaf_ap_obs
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
    for i in trange(xpix.size, leave=False, desc='Spectra'):
        # Get stellar spectrum
        teff_i = teff[i]   if teff   is not None else None
        sptp_i = sptype[i] if sptype is not None else 'G2V'
        sp = stellar_spectrum(sptp_i, mags_field[i], 'vegamag', nrc.bandpass,
                              Teff=teff_i, metallicity=0, log_g=4.5)

        # Create spectral image
        xr, yr = xpix[i], ypix[i]
        cen_rot = (ap_siaf.XSciRef, ap_siaf.YSciRef)
        wspec, imspec = place_grism_spec(nrc, sp, xr, yr, wref=wref, return_oversample=False,
                                         spec_ang=spec_ang, cen_rot=cen_rot)

        im_slope += imspec
        wspec_all.append(wspec)
        del imspec

    wspec_all = np.asarray(wspec_all)

    return wspec_all, im_slope

def place_grism_spec(nrc, sp, xpix, ypix, wref=None, return_oversample=False, 
                     spec_ang=0, cen_rot=None):
    """ Create spectral image and place ref wavelenght at (x,y) location 
    
    Given a NIRCam instrument object and input spectrum, create a dispersed
    PSF and place the undeviated reference wavelength at the specified
    (xpix,ypix) coordinates (assuming 'sci' coords). 

    Returned values will be a tuple of (wspec, imspec) 

    Parameters
    ==========
    nrc : :class:`~pynrc.NIRCam`
        pynrc.NIRCam class
    sp : :mod:`webbpsf_ext.synphot_ext.Spectrum`
        A synphot spectrum of target.
        Should already be normalized to the apparent flux.
    xpix : float
        Pixel position along x-axis to place reference wavelength.
        Specified in 'sci' coordinates. Assumes no spectral rotation.
    xpix : float
        Pixel position along y-axis to place reference wavelength.
        Specified in 'sci' coordinates. Assumes no spectral rotation.

    Keyword Args
    ============
    wref : float
        Undeviated reference wavelength in mircons associated with grism.
        Automatically determined from :func:`grism_wref` if not specified.
    return_oversample : bool
        Return 
    spec_ang : float
        Optionally set a rotation angle (deg) of the dispersion.
        Rotates in clockwise direction.
    cen_rot : tuple
        Position in 'sci' coordinates to rotate around if
        `spec_ang` is set. If not specified, then will rotate
        around center of image.

    """

    nx, ny = (nrc.Detector.xpix, nrc.Detector.ypix)
    oversample = nrc.oversample
    nx_over = nx * oversample
    ny_over = ny * oversample

    pupil_mask = nrc.pupil_mask
    if pupil_mask is None:
        _log.warning('place_grism_spec: NIRCam pupil mask set to None. Should be GRISMR or GRISMC.')
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
    # xr, yr = (np.array([xpix, ypix]) - 0.5) * oversample
    xr, yr = oversampled_coords(np.array([xpix, ypix]), oversample)
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

    # Expand to image
    wspec = wspec.reshape([1,-1]).repeat(imspec.shape[0], axis=0)

    # Clock spectrum
    if spec_ang!=0:
        cen_rot = np.asarray(cen_rot)
        cen_rot_over = oversampled_coords(cen_rot, oversample) #(cen_rot - 0.5) * oversample
        imspec = rotate_offset(imspec, spec_ang, cen=cen_rot_over, 
                               recenter=False, reshape=False)
        wspec = rotate_offset(wspec, spec_ang, cen=cen_rot_over, 
                              recenter=False, reshape=False, cval=np.nan)

        # Ensure we are croppsed to the correct size
        sh = (ny_over, nx_over)
        wspec = pad_or_cut_to_size(wspec, sh)
        imspec = pad_or_cut_to_size(imspec, sh)

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
    x1, x2 = np.array([x_new.min(), x_new.max()]).astype(int) + 0.5
    y1, y2 = np.array([y_new.min(), y_new.max()]).astype(int) + 0.5

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
    intx = xvals_os.astype(int)
    inty = yvals_os.astype(int)
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


def gen_unconvolved_point_source_image(nrc, tel_pointing, ra_deg, dec_deg, mags, expnum=1, 
    osamp=1, siaf_ap_obs=None, add_offset=None, **kwargs):
    """ Create an unconvolved image with sub-pixel shifts
    
    Parameters
    ==========
    nrc : :class:`~pynrc.NIRCam`
        NIRCam instrument class for PSF generation.
    tel_pointing : :class:`webbpsf_ext.jwst_point`
        JWST telescope pointing information. Holds pointing coordinates 
        and dither information for a given telescope visit.
    ra_deg : ndarray
        Array of RA positions of point sources in degrees.
    dec_deg : ndarray
        Array of Declination of points sourcesin degrees.
    mags : ndarray
        Magnitudes associated with each RA/Dec position.
        Corresponds to ``nrc.bandpass``.

    Keyword Args
    ============
    expnum : int
        Exposure number to use in ``tel_pointing``.
    osamp : int
        Output sampling of image.
    siaf_ap_obs : pysiaf Aperture
        Option to specify observed SIAF aperture. Otherwise defaults to ``nrc.siaf_ap``.
    add_offset : tuple or None
        If specififed, then will add an additional 'idl' offset to source positions.
    sp_type : str
        Spectral type to assume when calculating total counts. 
        Defaults to 'G0V'.
    mag_units : str
        Assumed magnitude units of ``mags``. 
        Default assumes 'vegamag'.
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
    # Include a offset shift to better reposition spectrum?
    add_offset = np.array([0,0]) if add_offset is None else np.asarray(add_offset)
    idl_off = np.array([(tel_pointing.position_offsets_act[ind])]) + add_offset
    # idl_off = [tel_pointing.position_offsets_act[ind]]
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

from webbpsf_ext.coron_masks import coron_trans, coron_ap_locs, coron_detector
from webbpsf_ext.coron_masks import build_mask, build_mask_detid

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
        #w0 = bp.avgwave().to_value('um')
        w0 = float(filt[1:-1])/100

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

#############################

def segment_pupil_opd(hdu, segment_name, npix=1024):
    """Extract single segment pupil from input full OPD map
    
    
    Input a OPD map HDU and name of segment (or ``segment_name=="ALL"``).
    Returns both the pupil mask and OPD image as separate HDULists.
    """
    
    from webbpsf.webbpsf_core import segname, one_segment_pupil
    webbpsf_data_path = webbpsf.utils.get_webbpsf_data_path()

    # Pupil and segment information
    # pupil_file = os.path.join(webbpsf_data_path, "jwst_pupil_RevW_npix1024.fits.gz")

    # get the master pupil file, which may or may not be gzipped
    pupil_file = os.path.join(webbpsf_data_path, f"jwst_pupil_RevW_npix{npix}.fits")
    if not os.path.exists(pupil_file):
        # try with .gz
        pupil_file = os.path.join(webbpsf_data_path, f"jwst_pupil_RevW_npix{npix}.fits.gz")
    pupil_hdul = fits.open(pupil_file)

    if segment_name.upper()=='ALL':
        opd_im, opd_header = (hdu.data, hdu.header)

        # New Pupil HDUList
        hdu = fits.PrimaryHDU(pupil_hdul[0].data)
        hdu.header = pupil_hdul[0].header.copy()
        pupil_all_hdul = fits.HDUList([hdu])
        
        # New OPD HDUList
        hdu = fits.PrimaryHDU(opd_im * pupil_all_hdul[0].data)
        hdu.header = opd_header.copy()
        opd_all_hdul = fits.HDUList([hdu])
        
        return (pupil_all_hdul, opd_all_hdul)

    else:
        # Parse out segment number
        # segment_official_name = segname(segment_name)
        # Parse out the segment number
        # num = int(segment_official_name.split('-')[1])
        # Index of segment to grab
        # iseg = num - 1

        # Pupil mask of segment only
        pupil_seg_hdul = one_segment_pupil(segment_name, npix=npix)
        
        opd_im, opd_header = (hdu.data, hdu.header)
        opd_im_seg = opd_im * pupil_seg_hdul[0].data
        
        # New Pupil HDUList
        hdu = fits.PrimaryHDU(pupil_seg_hdul[0].data)
        hdu.header = pupil_seg_hdul[0].header.copy()
        pupil_seg_hdul = fits.HDUList([hdu])

        # New OPD HDUList
        hdu = fits.PrimaryHDU(opd_im_seg * pupil_seg_hdul[0].data)
        hdu.header = opd_header.copy()
        opd_seg_hdul = fits.HDUList([hdu])
        
        return (pupil_seg_hdul, opd_seg_hdul)

def bias_dark_high_temp(darks_80K_dir, T=80, sca=485):
    """Extrapolate a bias and dark at some temperature """

    from .reduce.ref_pixels import NRC_refs
    
    # Grab the appropriate dark ramp
    dark_files = [f for f in os.listdir(darks_80K_dir)]
    matching = [s for s in dark_files if (("_{}_".format(sca) in s) and (s.endswith(".fits")))]
    fname = os.path.join(darks_80K_dir, matching[0])

    # Open the 80K dark fits
    hdul = fits.open(fname)

    header = hdul[0].header
    nint   = header['NINT']
    ng     = header['NGROUP']

    # Reference pixel correction
    data_mn = np.zeros([ng,2048,2048])
    for i in range(nint):
        data_int = hdul[0].data[i*ng:(i+1)*ng]
        ref = NRC_refs(data_int, header)
        ref.calc_avg_amps()
        ref.correct_amp_refs()

        data_mn += ref.data

    data_mn /= nint
    hdul.close()

    # Perform linear fit to averaged ramps
    det = ref.detector
    tarr = (np.arange(det.multiaccum.ngroup)+1) * ref.detector.time_group
    bias_80K, dark_80K = jl_poly_fit(tarr, data_mn)
    
    # pynrc data path
    pynrc_data_path = conf.PYNRC_PATH
    
    # calib sub-directory info for dark and bias
    subdir_dark = os.path.join('calib', f'{sca}', 'SUPER_DARK')
    subdir_bias = os.path.join('calib', f'{sca}', 'SUPER_BIAS')

    # file names
    file_dark0 = f'SUPER_DARK_{sca}.FITS'
    file_bias0 = f'SUPER_BIAS_{sca}.FITS'

    # Full paths
    path_dark0 = os.path.join(pynrc_data_path, subdir_dark, file_dark0)
    path_bias0 = os.path.join(pynrc_data_path, subdir_bias, file_bias0)
    
    hdul_dark0 = fits.open(path_dark0)
    hdul_bias0 = fits.open(path_bias0)
    
    dark0 = hdul_dark0[0].data
    bias0 = hdul_bias0[0].data
    
    # Some interpolation/extrapolation
    # Assume linear with log(dark) vs 1/T (Rule07 relationship)
    f = ((1/T - 1/39) / (1/80 - 1/39))
    dark_new = 10**(np.log10(dark_80K)*f + np.log10(dark0)*(1-f))

    # Assume bias offset is linear with temperature 
    f = ((T-39.0) / (80.0-39.0))
    bias_new = (bias_80K*f + bias0*(1-f))

    # Make sure reference pixels are 0
    dark_new[det.mask_ref] = 0
    # Fix any additional NaNs
    dark_new[np.isnan(dark_new)] = 0

    # Close HDULists
    hdul_dark0.close()
    hdul_bias0.close()

    return (bias_new, dark_new)

def dark_ramp_80K(darks_80K_dir, sca=485):
    """Return dark ramp over time"""

    from .reduce.ref_pixels import NRC_refs

    # Grab the appropriate dark ramp
    dark_dir = darks_80K_dir
    dark_files = [f for f in os.listdir(dark_dir)]
    matching = [s for s in dark_files if (("_{}_".format(sca) in s) and (s.endswith(".fits")))]
    fname = dark_dir + matching[0]

    # Open the 80K dark fits
    hdul = fits.open(fname)

    header = hdul[0].header
    nint   = header['NINT']
    ng     = header['NGROUP']

    # Reference pixel correction
    data_mn = np.zeros([ng,2048,2048])
    for i in range(nint):
        data_int = hdul[0].data[i*ng:(i+1)*ng]
        ref = NRC_refs(data_int, header)
        ref.calc_avg_amps()
        ref.correct_amp_refs()

        data_mn += ref.data

    data_mn /= nint
    hdul.close()

    det = ref.detector
    tarr = (np.arange(det.multiaccum.ngroup)+1) * det.time_group
    
    return tarr, data_mn

def do_charge_migration(image, satmax=1.5, niter=5, corners=True, **kwargs):
    """Migration of charge from saturated pixels to immediate neighbors"""

    ny, nx = image.shape
    ind_lock = np.zeros([ny,nx], dtype=bool)

    sat_image = image.copy()
    for j in range(niter):
        ind_sat = (sat_image >= satmax) & (~ind_lock)
        # Break if no saturated pixels
        if np.sum(ind_sat)==0: 
            break

        yxind = np.where(ind_sat)
        # for each saturated pixel, migrate charge to neighbors
        # starting with the brightest
        isort = np.argsort(sat_image[ind_sat])[::-1]
        for i in isort:
            y, x = yxind[0][i], yxind[1][i]
            # Find neighbors
            xind = np.array([x-1, x+1, x, x])
            yind = np.array([y, y, y-1, y+1])
            # Find corners
            xind_corn = np.array([x-1, x+1, x-1, x+1])
            yind_corn = np.array([y-1, y-1, y+1, y+1])

            # Remove neighbors outside of array or locked
            ind_good = (xind>=0) & (xind<nx) & (yind>=0) & (yind<ny) & \
                        (ind_lock[yind,xind]==False)
            xind, yind = (xind[ind_good], yind[ind_good])
            ind_good = (xind_corn>=0) & (xind_corn<nx) & (yind_corn>=0) & (yind_corn<ny) & \
                        (ind_lock[yind_corn,xind_corn]==False)
            xind_corn, yind_corn = (xind_corn[ind_good], yind_corn[ind_good])

            # Total charge to distribute
            qtot = sat_image[y,x] - 1
            # Distribute charge to neighbors and corners
            n_neighbors = len(xind)
            n_corners = len(xind_corn) if corners else 0
            if n_neighbors + n_corners <= 0:
                continue

            # Corners get sqrt(2) less charge
            q_neighbors = qtot / (n_neighbors + n_corners / np.sqrt(2))
            q_corners = q_neighbors / np.sqrt(2)

            # Add charge to neighbors
            if n_neighbors>0:
                sat_image[yind,xind] += q_neighbors
            # Add charge to corners
            if n_corners>0:
                sat_image[yind_corn,xind_corn] += q_corners
            # Lock this pixel from further charge migration
            ind_lock[y,x] = True
            # Remove charge from this pixel
            sat_image[y,x] = satmax

    return sat_image

# Option to implement MKL FFT
try:
    import mkl_fft
    _MKLFFT_AVAILABLE = True
except ImportError:
    _MKLFFT_AVAILABLE = False

def do_fft(a, n=None, axis=-1, norm=None, inverse=False, real=False, use_mkl=True):
    """Perform FFT using either numpy or MKL FFT"""

    if _MKLFFT_AVAILABLE and use_mkl:
        fft_func = mkl_fft.ifft if inverse else mkl_fft.fft
        if n is None:
            n = a.shape[axis]
        return fft_func(a, n=n, axis=axis, norm=norm)
    else:
        if real:
            fft_func = np.fft.irfft if inverse else np.fft.rfft
        else:
            fft_func = np.fft.ifft if inverse else np.fft.fft
        return fft_func(a, n=n, axis=axis, norm=norm)
