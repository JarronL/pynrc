"""pyNRC utility functions"""

from __future__ import absolute_import, division, print_function, unicode_literals

# The six library is useful for Python 2 and 3 compatibility
import six
import os, re

# Import libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Update
on_rtd = os.environ.get('READTHEDOCS') == 'True'
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
from astropy import units

#from scipy.optimize import least_squares#, leastsq
#from scipy.ndimage import fourier_shift
from scipy.interpolate import griddata, RegularGridInterpolator, interp1d
from numpy.polynomial import legendre

from . import conf
from .logging_utils import setup_logging

from .maths import robust
from .maths.image_manip import *
from .maths.fast_poly import *
from .maths.coords import *

###########################################################################
#
#    Logging info
#
###########################################################################

import logging
_log = logging.getLogger('pynrc')

###########################################################################
#
#    WebbPSF Stuff
#
###########################################################################

try:
    import webbpsf
except ImportError:
    raise ImportError('WebbPSF is not installed. pyNRC depends on its inclusion.')
# Check that minimum required version meets requirements
if not on_rtd:
    _webbpsf_version_min = (0,9,0)
    _ = webbpsf.utils.get_webbpsf_data_path(_webbpsf_version_min)

# Link to WebbPSF's instance of poppy
from webbpsf.webbpsf_core import poppy

# Set up some poppy and webbpsf defaults
# Turn off multiprocessing, which is faster now due to improved
# underlying vectorization of poppy and numpy. Swapping large
# amount of data between processes is now the bottleneck for mp (5/18/2020).
poppy.conf.use_multiprocessing = False 
# Only use this if you have the FFTW C library installed
# In general, numpy fft is actually pretty fast now, so default use_fftw=False
# It also doesn't play well with multiprocessing
poppy.conf.use_fftw = False

# Make sure we can use multiprocessing!
# Apple's Accelerate framework in 2.7 doesn't work with mp
d = np.__config__.blas_opt_info
accel_bool = ('extra_link_args' in d.keys() and ('-Wl,Accelerate' in d['extra_link_args']))
if (sys.version_info < (3,4,0)) and (platform.system()=='Darwin') and accel_bool:
    poppy.conf.use_multiprocessing = False
# If the machine has 2 or less CPU cores, then no mp
if mp.cpu_count()<3:
    poppy.conf.use_multiprocessing = False
# n_processes will be considered the max number of processors we use for multiprocessing
poppy.conf.n_processes = int(0.75 * mp.cpu_count()) #if poppy.conf.use_multiprocessing else 1

webbpsf.conf.default_output_mode = u'detector'

# Some useful functions for displaying and measuring PSFs
from poppy import radial_profile, measure_fwhm
#from poppy import (display_PSF, display_PSF_difference, display_EE, display_profiles, radial_profile,
#        measure_EE, measure_radial, measure_fwhm, measure_sharpness, measure_centroid, measure_strehl,
#        specFromSpectralType, fwcentroid)

# Detector geometry stuff
import pysiaf
siaf = pysiaf.Siaf('NIRCam')
siaf.generate_toc()
from webbpsf.webbpsf_core import DetectorGeometry

import pysynphot as S
# Extend default wavelength range to 5.6 um
S.refs.set_default_waveset(minwave=500, maxwave=56000, num=10000.0, delta=None, log=False)
# JWST 25m^2 collecting area
# Flux loss from masks and occulters are taken into account in WebbPSF
S.refs.setref(area = 25.4e4) # cm^2

# Default OPD info
opd_default = ('OPD_RevW_ote_for_NIRCam_requirements.fits', 0)

# The following won't work on readthedocs compilation
if not on_rtd:
    # Grab WebbPSF assumed pixel scales
    nc_temp = webbpsf.NIRCam()
    pixscale_SW = nc_temp._pixelscale_short
    pixscale_LW = nc_temp._pixelscale_long
    del nc_temp

    # .fits or .fits.gz?
    opd_dir = os.path.join(webbpsf.utils.get_webbpsf_data_path(),'NIRCam','OPD')
    opd_file = os.path.join(opd_dir,opd_default[0])
    if not os.path.exists(opd_file):
        opd_file_alt = opd_file + '.gz'
        if not os.path.exists(opd_file_alt):
            f1 = os.path.basename(opd_file)
            f2 = os.path.basename(opd_file_alt)
            err_msg = 'Cannot find either {} or {} in directory {}'.format(f1, f2, opd_dir)
            raise OSError(err_msg)
        else:
            opd_default = ('OPD_RevW_ote_for_NIRCam_requirements.fits.gz', 0)


        #import errno
        #raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), opd_file)

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


def bp_igood(bp, min_trans=0.001, fext=0.05):
    """
    Given a bandpass with transmission 0.0-1.0, return the indices that
    cover only the region of interest and ignore those wavelengths with
    very low transmission less than and greater than the bandpass width.
    """
    # Select which wavelengths to use
    igood = bp.throughput > min_trans
    # Select the "good" wavelengths
    wgood = (bp.wave)[igood]
    w1 = wgood.min()
    w2 = wgood.max()
    wr = w2 - w1

    # Extend by 5% on either side
    w1 -= fext*wr
    w2 += fext*wr

    # Now choose EVERYTHING between w1 and w2 (not just th>0.001)
    ind = ((bp.wave >= w1) & (bp.wave <= w2))
    return ind


def read_filter(filter, pupil=None, mask=None, module=None, ND_acq=False,
    ice_scale=None, nvr_scale=None, ote_scale=None, nc_scale=None,
    grism_order=1, coron_substrate=False, **kwargs):
    """Read filter bandpass.

    Read in filter throughput curve from file generated by STScI.
    Includes: OTE, NRC mirrors, dichroic, filter curve, and detector QE.

    To Do: Account for pupil size reduction for DHS and grism observations.

    Parameters
    ----------
    filter : str
        Name of a filter.
    pupil : str, None
        NIRCam pupil elements such as grisms or lyot stops.
    mask : str, None
        Specify the coronagraphic occulter (spots or bar).
    module : str
        Module 'A' or 'B'.
    ND_acq : bool
        ND acquisition square in coronagraphic mask.
    ice_scale : float
        Add in additional OTE H2O absorption. This is a scale factor
        relative to 0.0131 um thickness. Also includes about 0.0150 um of
        photolyzed Carbon.
    nvr_scale : float
        Modify NIRCam non-volatile residue. This is a scale factor relative 
        to 0.280 um thickness already built into filter throughput curves. 
        If set to None, then assumes a scale factor of 1.0. 
        Setting nvr_scale=0 will remove these contributions.
    ote_scale : float
        Scale factor of OTE contaminants relative to End of Life model. 
        This is the same as setting ice_scale. Will override ice_scale value.
    nc_scale : float
        Scale factor for NIRCam contaminants relative to End of Life model.
        This model assumes 0.189 um of NVR and 0.050 um of water ice on
        the NIRCam optical elements. Setting this keyword will remove all
        NVR contributions built into the NIRCam filter curves.
        Overrides nvr_scale value.
    grism_order : int
        Option to use 2nd order grism throughputs instead. Useful if
        someone wanted to overlay the 2nd order contributions onto a 
        wide field observation.
    coron_substrate : bool
        Explicit option to include coronagraphic substrate transmission
        even if mask=None. Gives the option of using LYOT pupilt with
        or without coron substrate.

    Returns
    -------
    :mod:`pysynphot.obsbandpass`
        A Pysynphot bandpass object.
    """

    if module is None: module = 'A'

    # Select filter file and read
    f = filter.lower(); m = module.lower()
    #filt_dir = __location__ + 'throughputs_stsci/'
    filt_dir = conf.PYNRC_PATH + 'throughputs/'
    filt_file = filter + '_nircam_plus_ote_throughput_mod' + m + '_sorted.txt'
    bp = S.FileBandpass(filt_dir+filt_file)
    bp_name = filter

    _log.debug('Reading file: '+filt_file)

    # Select channel (SW or LW) for minor decisions later on
    channel = 'SW' if bp.avgwave()/1e4 < 2.3 else 'LW'

    # Select which wavelengths to keep
    igood = bp_igood(bp, min_trans=0.005, fext=0.1)
    wgood = (bp.wave)[igood]
    w1 = wgood.min()
    w2 = wgood.max()
    wrange = w2 - w1

    # Read in grism throughput and multiply filter bandpass
    if (pupil is not None) and ('GRISM' in pupil):
        # Grism transmission curve follows a 3rd-order polynomial
        # The following coefficients assume that wavelength is in um
        if (module == 'A') and (grism_order==1):
            cf_g = np.array([0.068695897, -0.943894294, 4.1768413, -5.306475735])
        elif (module == 'B') and (grism_order==1):
            cf_g = np.array([0.050758635, -0.697433006, 3.086221627, -3.92089596])
        elif (module == 'A') and (grism_order==2):
            cf_g = np.array([0.05172, -0.85065, 5.22254, -14.18118, 14.37131])
        elif (module == 'B') and (grism_order==2):
            cf_g = np.array([0.03821, -0.62853, 3.85887, -10.47832, 10.61880])

        # Create polynomial function for grism throughput from coefficients
        p = np.poly1d(cf_g)
        th_grism = p(bp.wave/1e4)
        th_grism[th_grism < 0] = 0

        # Multiply filter throughput by grism
        th_new = th_grism * bp.throughput
        bp = S.ArrayBandpass(bp.wave, th_new)

        # spectral resolution in um/pixel
        # res is in pixels/um and dw is inverse
        res, dw = grism_res(pupil, module, m=grism_order)
        # Convert to Angstrom
        dw *= 10000 # Angstrom

        npts = np.int(wrange/dw)+1
        warr = np.linspace(w1, w1+dw*npts, npts)
        bp = bp.resample(warr)

    # Read in DHS throughput and multiply filter bandpass
    elif (pupil is not None) and ('DHS' in pupil):
        # DHS transmission curve follows a 3rd-order polynomial
        # The following coefficients assume that wavelength is in um
        cf_d = np.array([0.3192, -3.4719, 14.972, -31.979, 33.311, -12.582])
        p = np.poly1d(cf_d)
        th_dhs = p(bp.wave/1e4)
        th_dhs[th_dhs < 0] = 0
        th_dhs[bp.wave > 3e4] = 0

        # Multiply filter throughput by DHS
        th_new = th_dhs * bp.throughput
        bp = S.ArrayBandpass(bp.wave, th_new)

        # Mean spectral dispersion (dw/pix)
        res = 290.0
        dw = 1. / res # um/pixel
        dw *= 10000   # Angstrom/pixel

        npts = np.int(wrange/dw)+1
        warr = np.linspace(w1, w1+dw*npts, npts)
        bp = bp.resample(warr)

    # Coronagraphic throughput modifications
    # Substrate transmission
    if ((mask  is not None) and ('MASK' in mask)) or coron_substrate or ND_acq:
        # Sapphire mask transmission values for coronagraphic substrate
        hdulist = fits.open(conf.PYNRC_PATH + 'throughputs/jwst_nircam_moda_com_substrate_trans.fits')
        wtemp = hdulist[1].data['WAVELENGTH']
        ttemp = hdulist[1].data['THROUGHPUT']
        # Estimates for w<1.5um
        wtemp = np.insert(wtemp, 0, [0.5, 0.7, 1.2, 1.40])
        ttemp = np.insert(ttemp, 0, [0.2, 0.2, 0.5, 0.15])
        # Estimates for w>5.0um
        wtemp = np.append(wtemp, [6.00])
        ttemp = np.append(ttemp, [0.22])

        # Did we explicitly set the ND acquisition square?
        # This is a special case and doesn't necessarily need to be set.
        # WebbPSF has a provision to include ND filters in the field, but we include
        # this option if the user doesn't want to figure out offset positions.
        if ND_acq:
            fname = 'NDspot_ODvsWavelength.txt'
            path_ND = conf.PYNRC_PATH + 'throughputs/' + fname
            data = ascii.read(path_ND)

            wdata = data[data.colnames[0]].data # Wavelength (um)
            odata = data[data.colnames[1]].data # Optical Density
            # Estimates for w<1.5um
            wdata = np.insert(wdata, 0, [0.5])
            odata = np.insert(odata, 0, [3.8])
            # Estimates for w>5.0um
            wdata = np.append(wdata, [6.00])
            odata = np.append(odata, [2.97])

            # CV3 data suggests OD needs to be multiplied by 0.93
            # compared to Barr measurements
            odata *= 0.93

            otemp = np.interp(wtemp, wdata, odata, left=0, right=0)
            ttemp *= 10**(-1*otemp)

        # Interpolate substrate transmission onto filter wavelength grid and multiply
        th_coron_sub = np.interp(bp.wave/1e4, wtemp, ttemp, left=0, right=0)
        th_new = th_coron_sub * bp.throughput
        bp = S.ArrayBandpass(bp.wave, th_new)


    # Lyot stop wedge modifications
    # Substrate transmission
    if (pupil is not None) and ('LYOT' in pupil):

        # Transmission values for wedges in Lyot stop
        if 'SW' in channel:
            fname = 'jwst_nircam_sw-lyot_trans_modmean.fits'
            hdulist = fits.open(conf.PYNRC_PATH + 'throughputs/' + fname)
            wtemp = hdulist[1].data['WAVELENGTH']
            ttemp = hdulist[1].data['THROUGHPUT']
            # Estimates for w<1.5um
            wtemp = np.insert(wtemp, 0, [0.50, 1.00])
            ttemp = np.insert(ttemp, 0, [0.95, 0.95])
            # Estimates for w>2.3um
            wtemp = np.append(wtemp, [2.50,3.00])
            ttemp = np.append(ttemp, [0.85,0.85])
            # Interpolate substrate transmission onto filter wavelength grid
            th_wedge = np.interp(bp.wave/1e4, wtemp, ttemp, left=0, right=0)

        elif 'LW' in channel:
            fname = 'jwst_nircam_lw-lyot_trans_modmean.fits'
            hdulist = fits.open(conf.PYNRC_PATH + 'throughputs/' + fname)
            wtemp = hdulist[1].data['WAVELENGTH']
            ttemp = hdulist[1].data['THROUGHPUT']
            ttemp *= 100 # Factors of 100 error in saved values

            # Smooth the raw data
            ws = 200
            s = np.r_[ttemp[ws-1:0:-1],ttemp,ttemp[-1:-ws:-1]]
            w = np.blackman(ws)
            y = np.convolve(w/w.sum(),s,mode='valid')
            ttemp = y[int((ws/2-1)):int(-(ws/2))]

            # Estimates for w<2.3um
            wtemp = np.insert(wtemp, 0, [1.00])
            ttemp = np.insert(ttemp, 0, [0.95])
            # Estimates for w>5.0um
            wtemp = np.append(wtemp, [6.0])
            ttemp = np.append(ttemp, [0.9])
            # Interpolate substrate transmission onto filter wavelength grid
            th_wedge = np.interp(bp.wave/1e4, wtemp, ttemp, left=0, right=0)

        th_new = th_wedge * bp.throughput
        bp = S.ArrayBandpass(bp.wave, th_new, name=bp.name)


    # Weak Lens substrate transmission
    if (pupil is not None) and (('WL' in pupil) or ('WEAK LENS' in pupil)):

        if 'WL' in pupil:
            wl_alt = {'WLP4' :'WEAK LENS +4', 
                      'WLP8' :'WEAK LENS +8', 
                      'WLP12':'WEAK LENS +12 (=4+8)', 
                      'WLM4' :'WEAK LENS -4 (=4-8)',
                      'WLM8' :'WEAK LENS -8'}
            wl_name = wl_alt.get(pupil, pupil)
        else:
            wl_name = pupil

        # Throughput for WL+4
        hdulist = fits.open(conf.PYNRC_PATH + 'throughputs/jwst_nircam_wlp4.fits')
        wtemp = hdulist[1].data['WAVELENGTH']
        ttemp = hdulist[1].data['THROUGHPUT']
        th_wl4 = np.interp(bp.wave/1e4, wtemp, ttemp, left=0, right=0)

        # Throughput for WL+/-8
        hdulist = fits.open(conf.PYNRC_PATH + 'throughputs/jwst_nircam_wlp8.fits')
        wtemp = hdulist[1].data['WAVELENGTH']
        ttemp = hdulist[1].data['THROUGHPUT']
        th_wl8 = np.interp(bp.wave/1e4, wtemp, ttemp, left=0, right=0)

        # If two lenses
        wl48_list = ['WEAK LENS +12 (=4+8)', 'WEAK LENS -4 (=4-8)']
        if (wl_name in wl48_list):
            th_wl = th_wl4 * th_wl8
            bp_name = 'F212N'
        elif 'WEAK LENS +4' in wl_name:
            th_wl = th_wl4
            bp_name = 'F212N'
        else:
            th_wl = th_wl8
            
        th_new = th_wl * bp.throughput
        bp = S.ArrayBandpass(bp.wave, th_new)

        # Select which wavelengths to keep
        igood = bp_igood(bp, min_trans=0.005, fext=0.1)
        wgood = (bp.wave)[igood]
        w1 = wgood.min()
        w2 = wgood.max()
        wrange = w2 - w1

    # OTE scaling (use ice_scale keyword)
    if ote_scale is not None:
        ice_scale = ote_scale
    if nc_scale is not None:
        nvr_scale = 0
    # Water ice and NVR additions (for LW channel only)
    if ((ice_scale is not None) or (nvr_scale is not None)) and ('LW' in channel):
        fname = conf.PYNRC_PATH + 'throughputs/ote_nc_sim_1.00.txt'
        names = ['Wave', 't_ice', 't_nvr', 't_sys']
        data  = ascii.read(fname, data_start=1, names=names)

        wtemp = data['Wave']
        wtemp = np.insert(wtemp, 0, [1.0]) # Estimates for w<2.5um
        wtemp = np.append(wtemp, [6.0])    # Estimates for w>5.0um

        th_new = bp.throughput
        if ice_scale is not None:
            ttemp = data['t_ice']
            ttemp = np.insert(ttemp, 0, [1.0]) # Estimates for w<2.5um
            ttemp = np.append(ttemp, [1.0])    # Estimates for w>5.0um
            # Interpolate transmission onto filter wavelength grid
            ttemp = np.interp(bp.wave/1e4, wtemp, ttemp)#, left=0, right=0)
            
            # Scale is fraction of absorption feature depth, not of layer thickness
            th_new = th_new * (1 - ice_scale * (1 - ttemp))
            # th_ice = np.exp(ice_scale * np.log(ttemp))
            # th_new = th_ice * th_new

        if nvr_scale is not None:
            ttemp = data['t_nvr']
            ttemp = np.insert(ttemp, 0, [1.0]) # Estimates for w<2.5um
            ttemp = np.append(ttemp, [1.0])    # Estimates for w>5.0um
            # Interpolate transmission onto filter wavelength grid
            ttemp = np.interp(bp.wave/1e4, wtemp, ttemp)#, left=0, right=0)
            
            # Scale is fraction of absorption feature depth, not of layer thickness
            # First, remove NVR contributions already included in throughput curve
            th_new = th_new / ttemp
            th_new = th_new * (1 - nvr_scale * (1 - ttemp))
            
            # The "-1" removes NVR contributions already included in
            # NIRCam throughput curves
            # th_nvr = np.exp((nvr_scale-1) * np.log(ttemp))
            # th_new = th_nvr * th_new
            
        if nc_scale is not None:
            names = ['Wave', 'coeff'] # coeff is per um path length
            path = conf.PYNRC_PATH
            data_ice  = ascii.read(path + 'throughputs/h2o_abs.txt', names=names)
            data_nvr  = ascii.read(path + 'throughputs/nvr_abs.txt', names=names)
    
            w_ice = data_ice['Wave']
            a_ice = data_ice['coeff']
            a_ice = np.interp(bp.wave/1e4, w_ice, a_ice)

            w_nvr = data_nvr['Wave']
            a_nvr = data_nvr['coeff']
            a_nvr = np.interp(bp.wave/1e4, w_nvr, a_nvr)

            ttemp = np.exp(-0.189 * a_nvr - 0.050 * a_ice)
            th_new = th_new * (1 - nc_scale * (1 - ttemp))
            
            # ttemp = np.exp(-nc_scale*(a_nvr*0.189 + a_ice*0.05))
            # th_new = ttemp * th_new

        

        # Create new bandpass
        bp = S.ArrayBandpass(bp.wave, th_new)


    # Resample to common dw to ensure consistency
    dw_arr = bp.wave[1:] - bp.wave[:-1]
    #if not np.isclose(dw_arr.min(),dw_arr.max()):
    dw = np.median(dw_arr)
    warr = np.arange(w1,w2, dw)
    bp = bp.resample(warr)

    # Need to place zeros at either end so Pysynphot doesn't extrapolate
    warr = np.concatenate(([bp.wave.min()-dw],bp.wave,[bp.wave.max()+dw]))
    tarr = np.concatenate(([0],bp.throughput,[0]))
    bp   = S.ArrayBandpass(warr, tarr, name=bp_name)

    return bp


###########################################################################
#
#    Create WebbPSF Coefficients and Images
#
###########################################################################


# Subclass of the WebbPSF NIRCam class to modify aberrations
from webbpsf import NIRCam as webbpsf_NIRCam
class webbpsf_NIRCam_mod(webbpsf_NIRCam):
    def __init__(self):
        webbpsf_NIRCam.__init__(self)

    # Remove limits for detector position
    # Values outside of [0,2047] will get transformed to the correct V2/V3 location
    @webbpsf_NIRCam.detector_position.setter
    def detector_position(self, position):
        try:
            x, y = map(int, position)
        except ValueError:
            raise ValueError("Detector pixel coordinates must be a pair of numbers, not {}".format(position))
        # if x < 0 or y < 0:
        #     raise ValueError("Detector pixel coordinates must be nonnegative integers")
        # if x > self._detector_npixels - 1 or y > self._detector_npixels - 1:
        #     raise ValueError("The maximum allowed detector pixel coordinate value is {}".format(
        #         self._detector_npixels - 1))

        self._detector_position = (int(position[0]), int(position[1]))


def nproc_use(fov_pix, oversample, nwavelengths, coron=False):
    """Estimate Number of Processors

    Attempt to estimate a reasonable number of processors to use
    for a multi-wavelength calculation. One really does not want
    to end up swapping to disk with huge arrays.

    NOTE: Requires ``psutil`` package. Otherwise defaults to ``mp.cpu_count() / 2``

    Parameters
    -----------
    fov_pix : int
        Square size in detector-sampled pixels of final PSF image.
    oversample : int
        The optical system that we will be calculating for.
    nwavelengths : int
        Number of wavelengths.
    coron : bool
        Is the nproc recommendation for coronagraphic imaging?
        If so, the total RAM usage is different than for direct imaging.
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
    avail_GB = mem.available / 1024**3
    # Leave 10% for other things
    avail_GB *= 0.9

    fov_pix_over = fov_pix * oversample

    # For multiprocessing, memory accumulates into the main process
    # so we have to subtract the total from the available amount
    reserve_GB = nwavelengths * fov_pix_over**2 * 8 / 1024**3
    # If not enough available memory, then just return nproc=1
    if avail_GB < reserve_GB:
        _log.warn('Not enough available memory ({} GB) to \
                   to hold resulting PSF info ({} GB)!'.\
                   format(avail_GB,reserve_GB))
        return 1

    avail_GB -= reserve_GB

    # Memory formulas are based on fits to memory usage stats for:
    #   fov_arr = np.array([16,32,128,160,256,320,512,640,1024,2048])
    #   os_arr = np.array([1,2,4,8])
    if coron:  # Coronagraphic Imaging (in MB)
        mem_total = (oversample*1024*2.4)**2 * 16 / (1024**2) + 500
        if fov_pix > 1024: mem_total *= 1.6
    else:  # Direct Imaging (also spectral imaging)
        mem_total = 5*(fov_pix_over)**2 * 8 / (1024**2) + 300.

    # Convert to GB
    mem_total /= 1024

    # How many processors to split into?
    nproc = int(avail_GB / mem_total)
    nproc = np.min([nproc, mp.cpu_count(), poppy.conf.n_processes])

    # Each PSF calculation will constantly use multiple processors
    # when not oversampled, so let's divide by 2 for some time
    # and memory savings on those large calculations
    if oversample==1:
        nproc = np.ceil(nproc / 2)

    _log.debug('avail mem {}; mem tot: {}; nproc_init: {:.0f}'.\
        format(avail_GB, mem_total, nproc))

    nproc = np.min([nproc, nwavelengths])
    # Resource optimization:
    # Split iterations evenly over processors to free up minimally used processors.
    # For example, if there are 5 processes only doing 1 iteration, but a single
    #	processor doing 2 iterations, those 5 processors (and their memory) will not
    # 	get freed until the final processor is finished. So, to minimize the number
    #	of idle resources, take the total iterations and divide by two (round up),
    #	and that should be the final number of processors to use.
    np_max = np.ceil(nwavelengths / nproc)
    nproc = int(np.ceil(nwavelengths / np_max))

    if nproc < 1: nproc = 1

    # Multiprocessing can only swap up to 2GB of data from the child
    # process to the master process. Return nproc=1 if too much data.
    im_size = (fov_pix_over)**2 * 8 / (1024**3)
    nproc = 1 if (im_size * np_max) >=2 else nproc

    _log.debug('avail mem {}; mem tot: {}; nproc_fin: {:.0f}'.\
        format(avail_GB, mem_total, nproc))

    return int(nproc)


def OPDFile_to_HDUList(file, slice=0):
    """
    Make a picklable HDUList for ingesting into multiproccessor WebbPSF
    helper function.
    """

    try:
        hdul = fits.open(file)
    except FileNotFoundError:
        opd_dir = os.path.join(webbpsf.utils.get_webbpsf_data_path(),'NIRCam','OPD')
        hdul = fits.open(os.path.join(opd_dir, file))
    ndim = len(hdul[0].data.shape)

    if ndim==3:
        opd_im = hdul[0].data[slice,:,:]
    else:
        opd_im = hdul[0].data

    hdu_new = fits.PrimaryHDU(opd_im)
    hdu_new.header = hdul[0].header.copy()
    opd_hdul = fits.HDUList([hdu_new])

    hdul.close()

    return opd_hdul


def _wrap_coeff_for_mp(args):
    """
    Internal helper routine for parallelizing computations across multiple processors
    for multiple WebbPSF monochromatic calculations.

    args => (inst,w,fov_pix,oversample)
    """
    # No multiprocessing for monochromatic wavelengths
    mp_prev = poppy.conf.use_multiprocessing
    poppy.conf.use_multiprocessing = False

    inst,w,fov_pix,oversample = args
    fov_pix_orig = fov_pix # Does calc_psf change fov_pix??
    try:
        hdu_list = inst.calc_psf(fov_pixels=fov_pix, oversample=oversample, monochromatic=w*1e-6,
                                 add_distortion=False, crop_psf=True)
        # Distortions are ignored here. It's prefered do perform these later.
        # See the WebbPSF functions in webbpsf.distortion

    except Exception as e:
        print('Caught exception in worker thread (w = {}):'.format(w))
        # This prints the type, value, and stack trace of the
        # current exception being handled.
        traceback.print_exc()

        print('')
        #raise e
        poppy.conf.use_multiprocessing = mp_prev
        return None

    # Return to previous setting
    poppy.conf.use_multiprocessing = mp_prev
    # return pad_or_cut_to_size(hdu_list[2].data, fov_pix_orig*oversample)
    return hdu_list[0]


def gen_psf_coeff(filter_or_bp, pupil=None, mask=None, module='A',
    fov_pix=11, oversample=None, npsf=None, ndeg=None, tel_pupil=None,
    offset_r=None, offset_theta=None, jitter=None, jitter_sigma=0.007,
    opd=None, wfe_drift=None, drift_file=None, include_si_wfe=False,
    detector=None, detector_position=None, apname=None, bar_offset=None, 
    force=False, save=True, save_name=None, return_save_name=False, 
    quick=False, return_webbpsf=False, add_distortion=False, crop_psf=True, 
    use_legendre=True, pynrc_mod=True, **kwargs):
    """Generate PSF coefficients

    Creates a set of coefficients that will generate a simulated PSF at any
    arbitrary wavelength. This function first uses ``WebbPSF`` to simulate
    a number of evenly spaced PSFs throughout some specified bandpass.
    An nth-degree polynomial is then fit to each oversampled pixel using
    a linear-least square fitting routine. The final set of coefficients
    for each pixel is returned as an image cube. The returned set of
    coefficient can be used to produce a set of PSFs by:

    >>> psfs = pynrc.nrc_utils.jl_poly(waves, coeffs)

    where 'waves' can be a scalar, nparray, list, or tuple. All wavelengths
    are in microns.

    Distortions should be applied after creation of an image scene. For
    NIRCam, this involves first rotating the focal plane and then 
    applying the distortions (see `webbpsf.distortion`). 

    >>> psf_rotated = distortion.apply_rotation(psf, crop=True)  # apply rotation
    >>> psf_distorted = distortion.apply_distortion(psf_rotated)  # apply siaf distortion


    Parameters
    ----------
    filter_or_bp : str, :mod:`pysynphot.obsbandpass`
        Either the name of a filter or a Pysynphot bandpass.
    pupil : str, None
        NIRCam pupil elements such as grisms or lyot stops.
    mask : str, None
        Specify the coronagraphic occulter (spots or bar).
    module : str
        Module 'A' or 'B'.
    fov_pix : int
        Size of the FoV in pixels (real SW or LW pixels)
    oversample : int
        Factor to oversample pixels (in one dimension).
        The resulting coefficients will have x/y dimensions of
        fov_pix*oversample. Default 2 for coronagraphy and 4 otherwise.
    npsf : int
        Number of evenly-spaced (with wavelength) monochromatic PSFs to
        generate with webbPSF. If not specified, then the default is to
        produce 20 PSFs/um. The wavelength range is determined by
        choosing those wavelengths where throughput is >0.001.
        There's a minimum of 5 monochromatic PSFs calculated over
        the bandpass.
    ndeg : int
        Polynomial degree for PSF fitting.
        Default = 10 (7 if quick=True).
    offset_r : float
        Radial offset from the center in arcsec.
    offset_theta : float
        Position angle for radial offset, in degrees CCW.
    bar_offset : float
        Offset along coronagraphic bar (wedge) occulter, in arcseconds.
        Keeps the resulting PSF at zero tip/tilt, unlike `offset_r` and `offset_theta`.
        Defaults to 0 offset. Use :func:`offset_bar` for filter-dependent offsets.
    opd : str, tuple, HDUList
        OPD specifications. If a tuple, then it should contain two elements
        (filename, slice index). Can also specify just the filename, which
        will default to the first image slice.  Can also pass an HDUList
        where the OPD data is stored at HDUList[0].data.
    wfe_drift : float
        Wavefront error drift amplitude in nm.
    drift_file : str, None
        Delta OPD file to use for WFE drift.
    include_si_wfe : bool
        Include SI WFE measurements? Default=False.
    detector : str, None
        Name of detector [NRCA1, ..., NRCA5, NRCB1, ..., NRCB5].
    detector_position : tuple, None
        The pixel position in (X, Y) on the detector ("science" coordinates)
    tel_pupil : str, HDUList, None
        Telescope entrance pupil mask.
        Should either be a filename string or HDUList.
        If None, then default: jwst_pupil_RevW_npix1024.fits.gz.
    jitter : str or None
        Currently either 'gaussian' or None.
    jitter_sigma : float
        If ``jitter = 'gaussian'``, then this is the size of the blurring effect.
    force : bool
        Forces a recalcuation of PSF even if saved PSF exists. (default: False)
    save : bool
        Save the resulting PSF coefficients to a file? (default: True)
    save_name : str, None
        Full path name of FITS file to save/load coefficents.
        If None, then a name is automatically generated.
    quick : bool
        Only perform a fit over the filter bandpass with a smaller default
        polynomial degree fit. Auto filename will have filter name appended.
    return_save_name : bool
        Return only the name for saving.
    return_webbpsf : bool
        Return the WebbPSF generated PSF rather than coefficients.
    add_distortion : bool
        Add instrument distortions to the PSF. Includes 2 new extensions to 
        the PSF HDUlist object.
    crop_psf : bool
        Crop distorted PSF to match undistorted pixel shape.
    use_legendre : bool
        Use Legendre polynomials for coefficient fitting.
    """

    from .version import __version__

    grism_obs = (pupil is not None) and ('GRISM' in pupil)
    dhs_obs   = (pupil is not None) and ('DHS'   in pupil)
    coron_obs = (pupil is not None) and ('LYOT'  in pupil)

    if oversample is None:
        oversample = 2 if coron_obs else 4

    if opd is None:  # Default OPD
        opd = opd_default
    elif isinstance(opd, six.string_types):
        opd = (opd, 0)
    # Default WFE drift
    wfe_drift = 0 if wfe_drift is None else wfe_drift
    assert wfe_drift >= 0, "wfe_drift must not be negative"

    # Update module in case detector is specific
    if detector is not None:
        module = 'A' if 'A' in detector else 'B'

    # Get filter throughput and create bandpass
    if isinstance(filter_or_bp, six.string_types):
        filter = filter_or_bp
        bp = read_filter(filter, pupil=pupil, mask=mask, module=module, **kwargs)
    else:
        bp = filter_or_bp
        filter = bp.name

    chan_str = 'SW' if bp.avgwave() < 24000 else 'LW'

    # Change log levels to WARNING for pyNRC, WebbPSF, and POPPY
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)

    # Create a simulated PSF with WebbPSF
    inst = webbpsf_NIRCam_mod() if pynrc_mod else webbpsf.NIRCam()
    # inst = webbpsf_NIRCam_mod()
    #inst.options['output_mode'] = 'oversampled'
    # The fov_pix keyword overrides parity
    #inst.options['parity'] = 'odd'
    inst.filter = filter

    # Check if mask and pupil names exist in WebbPSF lists.
    # We don't want to pass values that WebbPSF does not recognize
    # but are otherwise completely valid in the pynrc framework.
    if mask in list(inst.image_mask_list): inst.image_mask = mask
    if pupil in list(inst.pupil_mask_list): inst.pupil_mask = pupil

    # If WLP4 is used, always using a narrow band, so turn on quick flag
    wl4_list = ['WEAK LENS +12 (=4+8)', 'WEAK LENS -4 (=4-8)', 'WEAK LENS +4',
                'WLP12', 'WLM8', 'WLP4']
    if (pupil in wl4_list):
        quick = True

    # Should we include field-dependent aberrations?
    inst.include_si_wfe = include_si_wfe

    # Set the SIAF apertur name
    if apname is not None:
        inst.auto_apname = False
        inst.apname = apname

    # Detector position
    # define defaults
    det_switch = {'SWA': 'A1', 'SWB':'B1', 'LWA':'A5', 'LWB':'B5'}
    detpos_switch = {'SW':(1024,1024), 'LW':(1024,1024)}
    if (detector is None) and (detector_position is None):
        inst.detector = 'NRC' + det_switch.get(chan_str+module)
        inst.detector_position = detpos_switch.get(chan_str)
    elif detector is None:
        inst.detector = 'NRC' + det_switch.get(chan_str+module)
        inst.detector_position = detector_position
    elif detector_position is None:
        inst.detector_position = detpos_switch.get(chan_str)
        inst.detector = detector
    else:
        inst.detector = detector
        inst.detector_position = detector_position

    # Print aperture and detector info
    _log.debug(inst.apname, inst.detector, inst.detector_position)

    # Telescope Pupil
    if tel_pupil is not None:
        inst.pupil = tel_pupil

    mtemp = 'NONE' if mask is None else mask
    ptemp = 'CLEAR' if pupil is None else pupil
    # Get source offset positions
    # 1. Round masks - Always assume theta=0 due to symmetry.
    # 2. Bar Masks - PSF positioning is different depending on r and theta.
    # 3. All other imaging - Just perform nominal r=theta=0.
    #    Any PSF movement is more quickly applied with sub-pixel shifting routines.
    # NB: Implementation of field-dependent OPD maps may change these settings.
    if offset_r is None: offset_r = 0
    if offset_theta is None: offset_theta = 0
    rtemp, ttemp = (offset_r, offset_theta)
    inst.options['source_offset_r']     = rtemp
    inst.options['source_offset_theta'] = ttemp

    # Bar offsets (auto_offset not supported)
    # If observing with bar mask, default to 0 offset
    if 'B' in mtemp:
        bar_offset = 0 if bar_offset is None else bar_offset
    else:
        # Set to None if not observing with bar mask
        bar_offset = None
    bar_str = '' if bar_offset is None else '_bar{:.1f}'.format(bar_offset)
    inst.options['bar_offset'] = bar_offset

    jitter_sigma = 0 if jitter is None else jitter_sigma
    inst.options['jitter'] = jitter
    inst.options['jitter_sigma'] = jitter_sigma

    setup_logging(log_prev, verbose=False)

    # Deal with OPD file name
    #print(opd)
    if isinstance(opd, tuple):
        if not len(opd)==2:
            raise ValueError("opd passed as tuple must have length of 2.")
        # Filename info
        opd_name = opd[0] # OPD file name
        opd_num  = opd[1] # OPD slice
        rev = [s for s in opd_name.split('_') if "Rev" in s]
        rev = '' if len(rev)==0 else rev[0]
        otemp = '{}slice{:.0f}'.format(rev,opd_num)
        opd = OPDFile_to_HDUList(opd_name, opd_num)
    elif isinstance(opd, fits.HDUList):
        # A custom OPD is passed. Consider using force=True.
        otemp = 'OPDcustom'
        opd_name = 'OPD from supplied FITS HDUlist object'
        opd_num = 0
    else:
        raise ValueError("OPD must be a string, tuple, or HDUList.")

    if wfe_drift>0:
        otemp = '{}-{:.0f}nm'.format(otemp,wfe_drift)

    if save_name is None:
        # Name to save array of oversampled coefficients
        save_dir = conf.PYNRC_PATH + 'psf_coeffs/'
        # Create directory if it doesn't already exist
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # Build final filename to save coeff

        # Append filter name if using quick keyword
        fstr = '_{}'.format(filter) if quick else ''
        # fname = '{}{}{}_{}_{}'.format(chan_str,module,fstr,ptemp,mtemp)
        # fname = fname + '_pix{}_os{}'.format(fov_pix,oversample)
        # fname = fname + '_jsig{:.0f}_r{:.1f}_th{:.1f}'.format(jitter_sigma*1000,rtemp,ttemp)
        # fname = fname + '{}_{}'.format(bar_str,otemp)
        
        fname = '{}{}{}_{}_{}_pix{}_os{}_jsig{:.0f}_r{:.2f}_th{:+.1f}{}_{}'.\
            format(chan_str,module,fstr, ptemp,mtemp,fov_pix,oversample,\
                   jitter_sigma*1000,rtemp,ttemp,bar_str,otemp)

        # Add SI WFE tag if included
        if inst.include_si_wfe:
            fname = fname + '_siwfe'

        if use_legendre:
            fname = fname + '_legendre'
                                      
        fname = fname + '.fits'
        save_name = save_dir + fname

    if return_save_name:
        return save_name

    # Load data from already saved FITS file
    if os.path.exists(save_name) and (not force) and (not return_webbpsf):
        #return np.load(save_name)
        # return fits.getdata(save_name)
        hdul = fits.open(save_name)
        data = hdul[0].data.astype(np.float)
        header = hdul[0].header
        hdul.close()
        return data, header

    if return_webbpsf:
        _log.info('Generating and returning WebbPSF HDUList')
    else:
        temp_str = 'and saving' if save else 'but not saving'
        _log.info('Generating {} new PSF coefficient'.format(temp_str))

    # Only drift OPD if PSF is in nominal position (rtemp=0).
    # Anything that is in an offset position is currently considered
    # to be a faint companion source that we're trying to detect, so
    # the PSF WFE difference has negligible bearing on the outcome.
    if (wfe_drift > 0): # and (rtemp == 0):

        from . import speckle_noise as sn
        _log.debug('Performing WFE drift of {}nm'.format(wfe_drift))

        if isinstance(opd, fits.HDUList):
            _log.debug('OPD is HDUList.')
            opd_im = opd[0].data
            header = opd[0].header.copy()
        else: # Read in a specified OPD file and slice
            _log.debug('OPD is tuple {}.'.format(opd))
            opd_im, header = sn.read_opd_slice(opd, header=True)

        # Read in delta OPD file
        opd_dir = os.path.join(conf.PYNRC_PATH, 'opd_mod/')
        if drift_file is None:
            drift_file = 'wfedrift_case1_var2.fits'
        delta_hdul = fits.open(opd_dir + drift_file)

        delta_rms = delta_hdul[0].header['RMS_WFE']
        scale_val = wfe_drift / delta_rms

        delta_data = delta_hdul[0].data * scale_val
        delta_hdul.close()

        opd_im += delta_data
        hdu = fits.PrimaryHDU(opd_im)
        hdu.header = header

        hdu.header.add_history("Modified OPD by adding delta")
        hdu.header.add_history(" from " + drift_file)
        hdu.header.add_history(" scaled by {}".format(scale_val))

        hdu.header['ORIGINAL'] = (opd_name,   "Original OPD source")
        hdu.header['SLICE']    = (opd_num,    "Slice index of original OPD")
        hdu.header['DFILE']    = (drift_file, "Source file for OPD drift")
        hdu.header['OCASE']    = (delta_hdul[0].header['CASE'], "Oscillation model case")
        hdu.header['OVARIANT'] = (delta_hdul[0].header['VARIANT'], "Oscillation model variant")
        hdu.header['OAMP']     = (scale_val, "Amplitude scale factor")
        hdu.header['WFEDRIFT'] = (wfe_drift, "WFE drift amount [nm]")

        opd_hdulist = fits.HDUList([hdu])
        inst.pupilopd = opd_hdulist
    else:
        inst.pupilopd = opd

    # By default, WebbPSF has wavelength limits depending on the channel
    # which can interfere with pynrc calculations, so set these to low/high values
    inst.SHORT_WAVELENGTH_MIN = inst.LONG_WAVELENGTH_MIN = 1e-7
    inst.SHORT_WAVELENGTH_MAX = inst.LONG_WAVELENGTH_MAX = 10e-6

    # Select which wavelengths to use
    # If doing a "quick" PSF, only fit the filter wavelength range.
    # Otherwise, we fit the full channel wavelength.
    if quick:
        w1 = bp.wave.min() / 1e4
        w2 = bp.wave.max() / 1e4
    else:
        w1, w2 = (0.5,2.5) if 'SW' in chan_str else (2.4,5.1)

    # First determine polynomial fit degree
    # Must ensure npsf>ndeg
    if ndeg is None:
        if use_legendre:
            # TODO: Quantify these better
            ndeg = 7 if quick else 9
        else:
            # TODO: Quantify these better
            ndeg = 7 if quick else 9

    # Create set of monochromatic PSFs to fit.
    if npsf is None:
        dn = 20 # 20 PSF simulations per um
        npsf = np.ceil(dn * (w2-w1))
    npsf = 5 if npsf<5 else int(npsf)
    npsf = ndeg+1 if npsf<=ndeg else int(npsf)
    waves = np.linspace(w1, w2, npsf)

    # Change log levels to WARNING for pyNRC, WebbPSF, and POPPY
    if return_webbpsf:
        setup_logging('WARN', verbose=False)

        if 'sp_norm' in list(kwargs.keys()):
            sp_norm = kwargs['sp_norm']
        else:
            waveset = waves * 1e4
            sp_flat = S.ArraySpectrum(waveset, 0*waveset + 10.)
            sp_flat.name = 'Flat spectrum in flam'

            # Bandpass unit response is the flux (in flam) of a star that
            # produces a response of one count per second in that bandpass
            sp_norm = sp_flat.renorm(bp.unit_response(), 'flam', bp)

        t0 = time.time()
        hdu_list = inst.calc_psf(source=sp_norm, fov_pixels=fov_pix, oversample=oversample, 
                                 add_distortion=add_distortion, crop_psf=crop_psf)
        t1 = time.time()
        setup_logging(log_prev, verbose=False)

        time_string = 'Took {:.2f} seconds to generate WebbPSF images'.format(t1-t0)
        _log.info(time_string)

        # Take into account reduced beam factor for grism data
        # Account for the circular pupil that does not allow all grism grooves to have their
        # full length illuminated (Erickson & Rabanus 2000), effectively broadening the FWHM.
        # It's actually a hexagonal pupil, so the factor is 1.07, not 1.15.
        # We want to stretch the PSF in the dispersion direction
        if grism_obs:
            wfact = 1.07
            scale = (1,wfact) if 'GRISM0' in pupil else (wfact,1)
            for hdu in hdu_list:
                im_scale = frebin(hdu.data, scale=scale)
                hdu.data = pad_or_cut_to_size(im_scale, hdu.data.shape)

        return hdu_list


    # How many processors to split into?
    nproc = nproc_use(fov_pix, oversample, npsf)#, coron=coron_obs) #if poppy.conf.use_multiprocessing else 1
    _log.debug('nprocessors: {}; npsf: {}'.format(nproc, npsf))

    setup_logging('WARN', verbose=False)
    t0 = time.time()
    # Setup the multiprocessing pool and arguments to pass to each pool
    worker_arguments = [(inst, wlen, fov_pix, oversample) for wlen in waves]
    if nproc > 1:
        pool = mp.Pool(nproc)
        # Pass arguments to the helper function

        try:
            hdu_arr = pool.map(_wrap_coeff_for_mp, worker_arguments)
            if hdu_arr[0] is None:
                raise RuntimeError('Returned None values. Issue with multiprocess or WebbPSF??')

        except Exception as e:
            _log.error('Caught an exception during multiprocess.')
            _log.error('Closing multiprocess pool.')
            pool.terminate()
            pool.close()
            raise e

        else:
            _log.debug('Closing multiprocess pool.')
            pool.close()
    else:
        # Pass arguments to the helper function
        hdu_arr = []
        for wa in worker_arguments:
            hdu = _wrap_coeff_for_mp(wa)
            if hdu is None:
                raise RuntimeError('Returned None values. Issue with WebbPSF??')
            hdu_arr.append(hdu)
    t1 = time.time()

    # Reset to original log levels
    setup_logging(log_prev, verbose=False)
    time_string = 'Took {:.2f} seconds to generate WebbPSF images'.format(t1-t0)
    _log.info(time_string)

    # Extract image data from HDU array
    images = []
    for hdu in hdu_arr:
        images.append(hdu.data)

    # Take into account reduced beam factor for grism data
    # Account for the circular pupil that does not allow all grism grooves to have their
    # full length illuminated (Erickson & Rabanus 2000), effectively broadening the FWHM.
    # It's actually a hexagonal pupil, so the factor is 1.07, not 1.15.
    # We want to stretch the PSF in the dispersion direction
    if grism_obs:
        wfact = 1.07
        scale = (1,wfact) if 'GRISM0' in pupil else (wfact,1)
        for i,im in enumerate(images):
            im_scale = frebin(im, scale=scale)
            images[i] = pad_or_cut_to_size(im_scale, im.shape)

    # Turn results into an numpy array (npsf,ny,nx)
    images = np.array(images)

    # Simultaneous polynomial fits to all pixels using linear least squares
    coeff_all = jl_poly_fit(waves, images, deg=ndeg, use_legendre=use_legendre, lxmap=[w1,w2])

    hdu = fits.PrimaryHDU(coeff_all)
    hdr = hdu.header
    head_temp = hdu_arr[0].header

    hdr['DESCR']    = ('PSF Coeffecients', 'File Description')
    hdr['NWAVES']   = (npsf, 'Number of wavelengths used in calculation')
    hdr['PUPILOPD'] = (opd_name, 'Pupil wavefront OPD source')
    hdr['OPDSLICE'] = (opd_num, 'OPD slice index')

    copy_keys = ['PUPILINT', 'EXTNAME', 'OVERSAMP', 'DET_SAMP',
                    'PIXELSCL', 'FOV', 'JITRTYPE', 'JITRSIGM',
                    'INSTRUME', 'CHANNEL', 'DET_NAME', 'TEL_WFE',
                    'DET_X', 'DET_Y', 'DET_V2', 'DET_V3',
                    'DATE', 'AUTHOR', 'VERSION', 'DATAVERS']
    for key in copy_keys:
        try:
            hdr[key] = (head_temp[key], head_temp.comments[key])
        except (AttributeError, KeyError):
            hdr[key] = ('none', 'No key found')

    # gen_psf_coeff() Keyword Values
    hdr['FILTER'] = (filter, 'Filter Input')
    hdr['PUPIL']  = (ptemp, 'Pupil Setting')
    hdr['MASK']   = (mtemp, 'Coronagraphic Mask Setting')
    hdr['MODULE'] = (module, 'NIRCam Module A or B')
    hdr['FOVPIX'] = (fov_pix, 'WebbPSF pixel FoV')
    hdr['OSAMP']  = (oversample, 'WebbPSF pixel oversample')
    hdr['NPSF']   = (npsf, 'Number of wavelengths to calc')
    hdr['NDEG']   = (ndeg, 'Polynomial fit degree')
    hdr['WAVE1']  = (w1, 'First wavelength in calc')
    hdr['WAVE2']  = (w2, 'Last of wavelength in calc')
    hdr['LEGNDR'] = (use_legendre, 'Legendre polynomial fit?')
    if tel_pupil is None:
        hdr['TELPUP'] = ('None', 'Telescope pupil')
    elif isinstance(tel_pupil, fits.HDUList):
        hdr['TELPUP'] = ('HDUList', 'Telescope pupil')
    elif isinstance(tel_pupil, six.string_types):
        hdr['TELPUP'] = (tel_pupil, 'Telescope pupil')
    else:
        hdr['TELPUP'] = ('UNKNOWN', 'Telescope pupil')
    hdr['OFFR']  = (offset_r, 'Radial offset')
    hdr['OFFTH'] = (offset_theta, 'Position angle OFFR (CCW)')
    if jitter is None:
        hdr['JITTER'] = ('None', 'Jitter type')
    else:
        hdr['JITTER'] = (jitter, 'Jitter type')
    hdr['JITSIG'] = (jitter_sigma, 'Jitter sigma')
    if opd is None:
        hdr['OPD'] = ('None', 'Telescope OPD')
    elif isinstance(opd, fits.HDUList):
        hdr['OPD'] = ('HDUList', 'Telescope OPD')
    elif isinstance(opd, six.string_types):
        hdr['OPD'] = (opd, 'Telescope OPD')
    else:
        hdr['OPD'] = ('UNKNOWN', 'Telescope OPD')
    if drift_file is None:
        hdr['DFILE'] = ('None', "Source file for OPD drift")
    else:
        hdr['DFILE'] = (drift_file, "Source file for OPD drift")
    hdr['WFEDRIFT'] = (wfe_drift, "WFE drift amount [nm]")
    hdr['SIWFE']    = (include_si_wfe, "Was SI WFE included?")
    hdr['FORCE']    = (force, "Forced calculations?")
    hdr['SAVE']     = (save, "Save file?")
    hdr['FILENAME'] = (os.path.basename(save_name), "File save name")
    hdr['PYNRCVER'] = (__version__, "pyNRC version")

    hdr.insert('DATAVERS', '', after=True)
    hdr.insert('DATAVERS', ('','psf_coeff() Keyword Values'), after=True)
    hdr.insert('DATAVERS', '', after=True)

    hdr.add_history(time_string)

    if save:
        #np.save(save_name, coeff_all)
        hdu.writeto(save_name, overwrite=True)

    return coeff_all, hdr

def gen_webbpsf_psf(filter_or_bp, pupil=None, mask=None, module='A', 
                    fov_pix=11, oversample=None, tel_pupil=None, opd=None,
                    wfe_drift=None, drift_file=None, include_si_wfe=False,
                    offset_r=None, offset_theta=None, jitter=None, jitter_sigma=0.007,
                    detector=None, detector_position=None, apname=None, bar_offset=None,
                    add_distortion=False, crop_psf=True, pynrc_mod=True, **kwargs):

    """Create WebbPSF PSF

    Kind of clunky way of generating a single PSF directly from WebbPSF
    by passing all the different options through keyword arguments.
    """

    return gen_psf_coeff(filter_or_bp, pupil=pupil, mask=mask, module=module,
        fov_pix=fov_pix, oversample=oversample, tel_pupil=tel_pupil, opd=opd, 
        wfe_drift=wfe_drift, drift_file=drift_file, include_si_wfe=include_si_wfe,
        offset_r=offset_r, offset_theta=offset_theta, jitter=jitter, jitter_sigma=jitter_sigma,
        detector=detector, detector_position=detector_position, apname=apname, bar_offset=bar_offset, 
        add_distortion=add_distortion, crop_psf=crop_psf, pynrc_mod=pynrc_mod, 
        return_webbpsf=True, **kwargs)


def gen_webbpsf_siwfe(filter_or_bp, coords, pynrc_mod=True, **kwargs):
    """ Generate Location-specific PSF from WebbPSF

    Parameters
    ----------
    filter_or_bp : str, :mod:`pysynphot.obsbandpass`
        Either the name of a filter or a Pysynphot bandpass.
    coords : tuple
        (V2,V3) coordinates in (arcmin)

    Keyword Args
    ------------
    pynrc_mod : bool
        Use `webbpsf_NIRCam_mod` instead of `webbpsf.NIRCam`.
        This is a slightly modified version of NIRCam to fix
        minor coordinate issues.
    """
    
    # Get filter throughput and create bandpass
    if isinstance(filter_or_bp, six.string_types):
        filter = filter_or_bp
        bp = read_filter(filter, **kwargs)
    else:
        bp = filter_or_bp
        filter = bp.name

    chan_str = 'SW' if bp.avgwave() < 24000 else 'LW'

    coords_asec = np.array(coords)*60
    detector, detector_position, apname = Tel2Sci_info(chan_str, coords_asec, output='sci', return_apname=True, **kwargs)
    print(detector, detector_position, apname)

    kwargs['include_si_wfe'] = True
    kwargs['apname'] = apname
    kwargs['detector'] = detector
    kwargs['detector_position'] = detector_position

    kwargs['add_distortion'] = False
        
    return gen_webbpsf_psf(filter, pynrc_mod=pynrc_mod, **kwargs)


def _wrap_wfed_coeff_for_mp(arg):
    args, kwargs = arg

    wfe = kwargs['wfe_drift']
    print('WFE Drift: {} nm'.format(wfe))

    cf, _ = gen_psf_coeff(*args, **kwargs)
    return cf

def wfed_coeff(filter_or_bp, force=False, save=True, save_name=None, nsplit=None, **kwargs):
    """PSF Coefficient Mod for WFE Drift

    This function finds a relationship between PSF coefficients
    in the presence of WFE drift. For a series of WFE drift values,
    we generate corresponding PSF coefficients and fit a polynomial
    relationship to the residual values. This allows us to quickly
    modify a nominal set of PSF image coefficients to generate a
    new PSF where the WFE has drifted by some amplitude.

    Keyword Arguments match those in :func:`gen_psf_coeff`.

    Parameters
    ----------
    filter_or_bp : str
        Name of a filter.
    force : bool
        Forces a recalcuation of coefficients even if saved
        PSF already exists. (default: False)
    save : bool
        Save the resulting WFE drift coefficents to a file?
        (default: True)
    save_name : str, None
        Full path name of save file (.npy) to save/load.
        If None, then a name is automatically generated,
        matching the :func:`gen_psf_coeff` function.
    nsplit : int
        Number of processors to split over. There are checks to 
        make sure you're not requesting more processors than the 
        current machine has available.

    Example
    -------
    Generate PSF coefficient, WFE drift modifications, then
    create an undrifted and drifted PSF.

    >>> from pynrc import nrc_utils
    >>> fpix, osamp = (128, 4)
    >>> coeff  = nrc_utils.gen_psf_coeff('F210M', fov_pix=fpix, oversample=osamp)
    >>> wfe_cf = nrc_utils.wfed_coeff('F210M', fov_pix=fpix, oversample=osamp)
    >>> psf0   = nrc_utils.gen_image_coeff('F210M', coeff=coeff, fov_pix=fpix, oversample=osamp)

    >>> # Drift the coefficients
    >>> wfe_drift = 5   # nm
    >>> cf_fit = wfe_cf.reshape([wfe_cf.shape[0], -1])
    >>> cf_mod = nrc_utils.jl_poly(np.array([wfe_drift]), cf_fit).reshape(coeff.shape)
    >>> cf_new = coeff + cf_mod
    >>> psf5nm = nrc_utils.gen_image_coeff('F210M', coeff=cf_new, fov_pix=fpix, oversample=osamp)
    """

    kwargs['force']     = True
    kwargs['save']      = False
    kwargs['save_name'] = None
    #kwargs['opd']       = opd_default

    # Get filter throughput and create bandpass
    if isinstance(filter_or_bp, six.string_types):
        filter = filter_or_bp
        bp = read_filter(filter, **kwargs)
    else:
        bp = filter_or_bp
        filter = bp.name

    # defaults
    fov_pix = kwargs['fov_pix'] if 'fov_pix' in list(kwargs.keys()) else 33
    oversample = kwargs['oversample'] if 'oversample' in list(kwargs.keys()) else 4

    # Final filename to save coeff
    if save_name is None:
        # Name to save array of oversampled coefficients
        save_dir = conf.PYNRC_PATH + 'psf_coeffs/'
        # Create directory if it doesn't already exist
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # Final filename to save coeff
        save_name = gen_psf_coeff(bp, return_save_name=True, **kwargs)
        save_name = os.path.splitext(save_name)[0] + '_wfedrift.npz'

    # Load file if it already exists
    if (not force) and os.path.exists(save_name):
        out = np.load(save_name)
        return out['arr_0'], out['arr_1']

    _log.warn('Generating WFE Drift coefficients. This may take some time...')
    # _log.warn('{}'.format(save_name))

    # Cycle through WFE drifts for fitting
    wfe_list = np.array([0,1,2,5,10,20,40])
    npos = len(wfe_list)

    # Split over multiple processors?
    nsplit_max = nproc_use(fov_pix, oversample, npos)#, coron=coron_obs)
    if nsplit is None:
        pupil = kwargs['pupil'] if 'pupil' in list(kwargs.keys()) else None
        coron_obs = (pupil is not None) and ('LYOT' in pupil)
        nsplit = nproc_use(fov_pix, oversample, npos)#, coron=coron_obs)

        # Compare to number of PSFs
        if ('quick' in list(kwargs.keys())) and (kwargs['quick']==True):
            w1 = bp.wave.min() / 1e4
            w2 = bp.wave.max() / 1e4
            dw = w2 - w1
        else:
            dw = 2.5
        npsf = np.ceil(20 * dw)
        npsf = 5 if npsf<5 else int(npsf)
        nsplit_psf = nproc_use(fov_pix, oversample, npsf)#, coron=coron_obs)
        if nsplit_psf > nsplit:
            nsplit = 1

    # Double check we're not requesting too many processors
    nsplit = nsplit_max if nsplit > nsplit_max else nsplit
    nsplit = npos

    # Create worker arguments with kwargs as an argument input
    worker_args = []
    args = [bp]
    for wfe in wfe_list:
        kw = kwargs.copy()
        kw['wfe_drift'] = wfe
        worker_args.append((args, kw))

    if nsplit>1:
        # If splitting, then cannot split in subprocess for each PSF (ie., across wavelengths)
        poppy_nproc_prev = poppy.conf.n_processes
        poppy.conf.n_processes = 1

        pool = mp.Pool(nsplit)
        try:
            cf_wfe = pool.map(_wrap_wfed_coeff_for_mp, worker_args)
            if cf_wfe[0] is None:
                raise RuntimeError('Returned None values. Issue with multiprocess or WebbPSF??')
        except Exception as e:
            _log.error('Caught an exception during multiprocess.')
            _log.error('Closing multiprocess pool.')
            pool.terminate()
            pool.close()
            poppy.conf.n_processes = poppy_nproc_prev
            raise e
        else:
            _log.debug('Closing multiprocess pool.')
            pool.close()

        poppy.conf.n_processes = poppy_nproc_prev
    else:
        # No multiprocessor
        cf_wfe = [_wrap_wfed_coeff_for_mp(wa) for wa in worker_args]

    # Get residuals
    cf_wfe = np.array(cf_wfe) - cf_wfe[0]

    # Fit each pixel with a polynomial and save the coefficient
    cf_shape = cf_wfe.shape[1:]
    cf_wfe = cf_wfe.reshape([npos, -1])
    lxmap = np.array([np.min(wfe_list), np.max(wfe_list)])
    cf_fit = jl_poly_fit(wfe_list, cf_wfe, deg=4, use_legendre=True, lxmap=lxmap)
    cf_fit = cf_fit.reshape([-1, cf_shape[0], cf_shape[1], cf_shape[2]])

    if save:
        np.savez(save_name, cf_fit, lxmap)
    _log.info('Done.')

    return cf_fit, lxmap

def _wrap_field_coeff_for_mp(arg):
    args, kwargs = arg

    apname  = kwargs['apname']
    det     = kwargs['detector']
    det_pos = kwargs['detector_position']
    v2, v3  = kwargs['coords']

    print('V2/V3 Coordinates and det pixel (sci) on {}/{}: ({:.2f}, {:.2f}), ({:.1f}, {:.1f})'
        .format(det, apname, v2/60, v3/60, det_pos[0], det_pos[1]))

    cf, _ = gen_psf_coeff(*args, **kwargs)
    return cf

def field_coeff_resid(filter_or_bp, coeff0, force=False, save=True, save_name=None, 
    return_raw=False, nsplit=None, **kwargs):
    """PSF Coefficient Residuals w.r.t. Field Position

    Keyword Arguments match those in :func:`gen_psf_coeff`.

    Parameters
    ----------
    filter_or_bp : str
        Name of a filter.
    ceoff0 : ndarray
        PSF coefficient to perform relative comparison.
    force : bool
        Forces a recalcuation of coefficients even if saved
        PSF already exists. (default: False)
    save : bool
        Save the resulting WFE drift coefficents to a file?
        (default: True)
    save_name : str, None
        Full path name of save file (.npy) to save/load.
        If None, then a name is automatically generated,
        matching the :func:`gen_psf_coeff` function.
    nsplit : int
        Number of processors to split over. There are checks to 
        make sure you're not requesting more processors than the 
        current machine has available.


    Example
    -------
    Generate PSF coefficient, field position modifications, then
    create a PSF at some (V2,V3) location.

    >>> from pynrc import nrc_utils
    >>> fpix, osamp = (128, 4)
    >>> coeff    = nrc_utils.gen_psf_coeff('F210M', fov_pix=fpix, oversample=osamp)
    >>> cf_resid = nrc_utils.field_coeff('F210M', coeff, fov_pix=fpix, oversample=osamp)

    >>> # Some (V2,V3) location (arcmin)
    >>> v2, v3 = (1.2, -7)
    >>> cf_mod = nrc_utils.field_model(v2, v3, cf_resid)
    >>> cf_new = coeff + cf_mod
    >>> psf    = nrc_utils.gen_image_coeff('F210M', coeff=cf_new, fov_pix=fpix, oversample=osamp)
    """

    from astropy.table import Table

    kwargs['force']     = True
    kwargs['save']      = False
    kwargs['save_name'] = None

    if return_raw:
        save = False
        force = True
        _log.warn('return_raw=True; Setting save=False, force=True')

    # Get filter throughput and create bandpass
    if isinstance(filter_or_bp, six.string_types):
        filter = filter_or_bp
        bp = read_filter(filter, **kwargs)
    else:
        bp = filter_or_bp
        filter = bp.name
    channel = 'SW' if bp.avgwave() < 24000 else 'LW'

    # fov_pix should not be more than some size, otherwise memory issues
    fov_pix = kwargs['fov_pix'] if 'fov_pix' in list(kwargs.keys()) else 33
    oversample = kwargs['oversample'] if 'oversample' in list(kwargs.keys()) else 4

    # Final filename to save coeff
    if save_name is None:
        # Name to save array of oversampled coefficients
        save_dir = conf.PYNRC_PATH + 'psf_coeffs/'
        # Create directory if it doesn't already exist
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # Final filename to save coeff
        save_name = gen_psf_coeff(bp, return_save_name=True, **kwargs)
        save_name = os.path.splitext(save_name)[0] + '_cffields.npz'

    # Load file if it already exists
    if (not force) and os.path.exists(save_name):
        out = np.load(save_name)
        return out['arr_0'], out['arr_1'], out['arr_2']

    _log.warn('Generating field-dependent coefficients. This may take some time...')

    # Cycle through a list of field points
    # These are the measured CV3 field positions
    module = kwargs.get('module', 'A') # If not specified, choose 'A'
    kwargs['module'] = module
    # Check if coronagraphy
    pupil = kwargs.get('pupil', 'CLEAR') # If not specified, choose 'CLEAR'
    kwargs['pupil'] = pupil
    # Read in measured SI Zernike data
    if (pupil is not None) and ('LYOT' in pupil):
        zfile = 'si_zernikes_coron_wfe.fits'
        if module=='B':
            raise NotImplementedError("There are no Full Frame SIAF apertures defined for Mod B coronagraphy")
    else:
        zfile = 'si_zernikes_isim_cv3.fits'
    data_dir = webbpsf.utils.get_webbpsf_data_path() + '/'
    zernike_file = data_dir + zfile
    ztable_full = Table.read(zernike_file)

    mod = channel + module  
    ind_nrc = ['NIRCam'+mod in row['instrument'] for row in ztable_full]  
    ind_nrc = np.where(ind_nrc)

    v2_all = np.array(ztable_full[ind_nrc]['V2'].tolist())
    v3_all = np.array(ztable_full[ind_nrc]['V3'].tolist())

    # Add detector corners
    v2_min, v2_max, v3_min, v3_max = NIRCam_V2V3_limits(module, channel=channel, pupil=pupil, rederive=True, border=1)
    igood = v3_all > v3_min
    v2_all = np.append(v2_all[igood], [v2_min, v2_max, v2_min, v2_max])
    v3_all = np.append(v3_all[igood], [v3_min, v3_min, v3_max, v3_max])
    npos = len(v2_all)

    # First is default value
    #kwargs['detector'] = None
    #kwargs['detector_position'] = None
    #kwargs['include_si_wfe'] = False
    #cf0 = gen_psf_coeff(filter, **kwargs)
    kwargs['include_si_wfe'] = True

    # Split over multiple processors?
    nsplit_max = nproc_use(fov_pix, oversample, npos)#, coron=coron_obs)
    if nsplit is None:
        pupil = kwargs['pupil'] if 'pupil' in list(kwargs.keys()) else None
        coron_obs = (pupil is not None) and ('LYOT' in pupil)
        nsplit = nproc_use(fov_pix, oversample, npos)#, coron=coron_obs)

        # Compare to number of PSFs
        if ('quick' in list(kwargs.keys())) and (kwargs['quick']==True):
            w1 = bp.wave.min() / 1e4
            w2 = bp.wave.max() / 1e4
            dw = w2 - w1
        else:
            dw = 2.5
        npsf = np.ceil(20 * dw)
        npsf = 5 if npsf<5 else int(npsf)
        nsplit_psf = nproc_use(fov_pix, oversample, npsf, coron=coron_obs)
        if nsplit_psf > nsplit:
            nsplit = 1

    # Double check we're not requesting too many processors
    nsplit = nsplit_max if nsplit > nsplit_max else nsplit

    # Create worker arguments with kwargs as an input dict
    worker_args = []
    args = [filter]
    for (v2, v3) in zip(v2_all, v3_all):
        # Get the detector and pixel position
        coords = (v2*60, v3*60) # in arcsec
        det, det_pos, apname = Tel2Sci_info(channel, coords, pupil=pupil, output="sci", return_apname=True)

        kw = kwargs.copy()
        kw['apname'] = apname
        kw['detector'] = det
        kw['detector_position'] = det_pos
        kw['coords'] = coords
        worker_args.append((args, kw))

    # Multiprocessing?
    if nsplit > 1:
        # If splitting, then cannot split in subprocess for each PSF (ie., across wavelengths)
        poppy_nproc_prev = poppy.conf.n_processes
        poppy.conf.n_processes = 1

        pool = mp.Pool(nsplit)
        try:
            cf_fields = pool.map(_wrap_field_coeff_for_mp, worker_args)
            if cf_fields[0] is None:
                raise RuntimeError('Returned None values. Issue with multiprocess or WebbPSF??')
        except Exception as e:
            _log.error('Caught an exception during multiprocess.')
            _log.error('Closing multiprocess pool.')
            pool.terminate()
            pool.close()
            poppy.conf.n_processes = poppy_nproc_prev
            raise e
        else:
            _log.debug('Closing multiprocess pool.')
            pool.close()

        poppy.conf.n_processes = poppy_nproc_prev
    else:  # No multiprocessor
        cf_fields = [_wrap_field_coeff_for_mp(wa) for wa in worker_args]

    # Get residuals
    cf_fields = np.array(cf_fields) - coeff0

    if return_raw:
        return cf_fields, v2_all, v3_all

    # Create an evenly spaced grid of V2/V3 coordinates
    nv23 = 8
    v2grid = np.linspace(v2_min, v2_max, num=nv23)
    v3grid = np.linspace(v3_min, v3_max, num=nv23)

    # Interpolate onto an evenly space grid
    res = make_coeff_resid_grid(v2_all, v3_all, cf_fields, v2grid, v3grid)
    if save: 
        np.savez(save_name, *res)

    _log.warn('Done.')
    return res

def make_coeff_resid_grid(xin, yin, cf_resid, xgrid, ygrid):

    # Create 2D grid arrays of coordinates
    xnew, ynew = np.meshgrid(xgrid,ygrid)
    nx, ny = len(xgrid), len(ygrid)

    _log.warn("Interpolating coefficient residuals onto regular grid...")

    sh = cf_resid.shape
    cf_resid_grid = np.zeros([ny,nx,sh[1],sh[2],sh[3]])

    # Cycle through each coefficient to interpolate onto V2/V3 grid
    for i in range(sh[1]):
        cf_resid_grid[:,:,i,:,:] = griddata((xin, yin), cf_resid[:,i,:,:], (xnew, ynew), method='cubic')

    return (cf_resid_grid, xgrid, ygrid)



def field_coeff_func(v2grid, v3grid, cf_fields, v2_new, v3_new):
    """Interpolation function for PSF coefficient residuals

    Uses RegularGridInterpolator to quickly determine new coefficient
    residulas at specified points.

    Parameters
    ----------
    v2grid : ndarray
        V2 values corresponding to `cf_fields`.
    v3grid : ndarray
        V3 values corresponding to `cf_fields`.
    cf_fields : ndarray
        Coefficient residuals at different field points
        Shape is (nV3, nV2, ncoeff, ypix, xpix)
    v2_new : ndarray
        New V2 point(s) to interpolate on. Same units as v2grid.
    v3_new : ndarray
        New V3 point(s) to interpolate on. Same units as v3grid.
    """

    func = RegularGridInterpolator((v3grid, v2grid), cf_fields, method='linear', 
                                   bounds_error=False, fill_value=None)

    pts = np.array([v3_new,v2_new]).transpose()
    
    # If only 1 point, remove first axes
    res = func(pts)
    res = res.squeeze() if res.shape[0]==1 else res
    return res


def wedge_coeff(filter, pupil, mask, force=False, save=True, save_name=None, **kwargs):
    """PSF Coefficient Mod w.r.t. Wedge Coronagraph Location

    Keyword Arguments match those in :func:`gen_psf_coeff`.

    Parameters
    ----------
    filter : str
        Name of a filter.
    force : bool
        Forces a recalcuation of coefficients even if saved
        PSF already exists. (default: False)
    save : bool
        Save the resulting WFE drift coefficents to a file?
        (default: True)
    save_name : str, None
        Full path name of save file (.npy) to save/load.
        If None, then a name is automatically generated,
        matching the :func:`gen_psf_coeff` function.


    Example
    -------
    Generate PSF coefficient at bar_offset=0, generate position modifications,
    then use these results to create a PSF at some arbitrary offset location.

    >>> from pynrc import nrc_utils
    >>> fpix, osamp = (320, 2)
    >>> filt, pupil, mask = ('F430M', 'WEDGELYOT', 'MASKLWB')
    >>> coeff    = nrc_utils.gen_psf_coeff(filt, pupil, mask, fov_pix=fpix, oversample=osamp)
    >>> cf_resid = nrc_utils.wedge_coeff(filt, pupil, mask, fov_pix=fpix, oversample=osamp)

    >>> # The narrow location (arcsec)
    >>> bar_offset = 8
    >>> cf_fit = nrc_utils.cf_resid.reshape([cf_resid.shape[0], -1])
    >>> cf_mod = nrc_utils.jl_poly(np.array([bar_offset]), cf_fit).reshape(coeff.shape)
    >>> cf_new = coeff + cf_mod
    >>> psf    = nrc_utils.gen_image_coeff(filt, pupil, mask, coeff=cf_new, fov_pix=fpix, oversample=osamp)

    """

    kwargs['force']     = True
    kwargs['save']      = False
    kwargs['save_name'] = None

    kwargs['pupil'] = pupil
    kwargs['mask'] = mask

    module = kwargs.get('module', 'A')
    kwargs['module'] = module

    # Get filter throughput and create bandpass
    bp = read_filter(filter)
    channel = 'SW' if bp.avgwave() < 24000 else 'LW'

    # Final filename to save coeff
    if save_name is None:
        # Name to save array of oversampled coefficients
        save_dir = conf.PYNRC_PATH + 'psf_coeffs/'
        # Create directory if it doesn't already exist
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # Final filename to save coeff
        save_name = gen_psf_coeff(filter, return_save_name=True, **kwargs)
        save_name = os.path.splitext(save_name)[0] + '_wedge.npy'

    # Load file if it already exists
    if (not force) and os.path.exists(save_name):
        return np.load(save_name)

    _log.warn('Generating wedge field-dependent coefficients. This may take some time...')

    # Cycle through a list of bar offset locations
    values = np.arange(-8,8,1)
    nvals = len(values)

    # First is default value
    kwargs['include_si_wfe'] = False
    cf0, _ = gen_psf_coeff(filter, bar_offset=0, **kwargs)

    cf_offset = []
    for val in values:
        _log.debug('Bar Offset: {:.1f} arcsec'.format(val))
        kwargs['bar_offset'] = val

        cf, _ = gen_psf_coeff(filter, **kwargs)
        cf_offset.append(cf)

    cf_offset = np.array(cf_offset)

    # Get residuals
    cf_offset -= cf0

    # Fit each pixel with a polynomial and save the coefficient
    cf_offset = cf_offset.reshape([nvals, -1])
    cf_fit = jl_poly_fit(values, cf_offset, 4)
    cf_fit = cf_fit.reshape([-1, cf.shape[0], cf.shape[1], cf.shape[2]])

    if save:
        np.save(save_name, cf_fit)

    _log.warn('Done.')
    return cf_fit


def gen_image_from_coeff(coeff, ceoff_hdr, sp_norm=None, return_oversample=False,
    nwaves=None, use_sp_waveset=False, **kwargs):
    """Generate PSF from coefficient

    Wrapper for :func:`gen_image_coeff` that uses information in the header to
    populate certain input parameters (filter, mask, pupil, fov_pix, oversample)
    so as to avoid any confusion.
    """

    kwargs['pupil']      = None if 'NONE' in coeff_hdr['PUPIL'] else coeff_hdr['PUPIL']
    kwargs['mask']       = coeff_hdr['MASK']
    kwargs['module']     = coeff_hdr['MODULE']
    kwargs['fov_pix']    = coeff_hdr['FOVPIX']
    kwargs['oversample'] = coeff_hdr['OSAMP']

    return gen_image_coeff(coeff_hdr['FILTER'], **kwargs)



def gen_image_coeff(filter_or_bp, pupil=None, mask=None, module='A',
    coeff=None, coeff_hdr=None, sp_norm=None, nwaves=None,
    fov_pix=11, oversample=4, return_oversample=False, use_sp_waveset=False,
    **kwargs):
    """Generate PSF

    Create an image (direct, coronagraphic, grism, or DHS) based on a set of
    instrument parameters and PSF coefficients. The image is noiseless and
    doesn't take into account any non-linearity or saturation effects, but is
    convolved with the instrument throughput. Pixel values are in counts/sec.
    The result is effectively an idealized slope image.

    If no spectral dispersers (grisms or DHS), then this returns a single
    image or list of images if sp_norm is a list of spectra.

    Parameters
    ----------
    filter_or_bp : str, :mod:`pysynphot.obsbandpass`
        Either the name of a filter or a Pysynphot bandpass.
    pupil : str, None
        NIRCam pupil elements such as grisms or lyot stops.
    mask : str, None
        Specify the coronagraphic occulter (spots or bar).
    module : str
        Module 'A' or 'B'.
    sp_norm : :mod:`pysynphot.spectrum`
        A normalized Pysynphot spectrum to generate image. If not specified,
        the default is flat in phot lam (equal number of photons per spectral bin).
        The default is normalized to produce 1 count/sec within that bandpass,
        assuming the telescope collecting area. Coronagraphic PSFs will further
        decrease this flux.
    coeff : ndarray
        A cube of polynomial coefficients for generating PSFs. This is
        generally oversampled with a shape (fov_pix*oversamp, fov_pix*oversamp, deg).
        If not set, this will be calculated using the :func:`gen_psf_coeff` function.
    coeff_hdr : FITS header
        Header information saved while generating coefficients.
    nwaves : int
        Option to specify the number of evenly spaced wavelength bins to
        generate and sum over to make final PSF. Useful for wide band filters
        with large PSFs over continuum sorurce.
    use_sp_waveset : bool
        Set this option to use `sp_norm` waveset instead of bandpass waveset.
        Useful if user inputs a high-resolution spectrum with line emissions,
        so may wants to keep a grism PSF (for instance) at native resolution
        rather than blurred with the bandpass waveset. TODO: Test.  
    fov_pix : int
        Number of detector pixels in the image coefficient and PSF.
    oversample : int
        Factor of oversampling of detector pixels.
    return_oversample: bool
        If True, then also returns the oversampled version of the PSF.

    Keyword Args
    ------------
    grism_order : int
        Grism spectral order (default=1).
    npsf : int
        Number of evenly-spaced (with wavelength) monochromatic PSFs to
        generate with webbPSF. If not specified, then the default is to
        produce 20 PSFs/um. The wavelength range is determined by
        choosing those wavelengths where throughput is >0.001.
    ndeg : int
        Polynomial degree for PSF fitting.
        read_filter - ND_acq
    ND_acq : bool
        ND acquisition square in coronagraphic mask.
    """

    is_grism = ((pupil is not None) and ('GRISM' in pupil))
    is_dhs   = ((pupil is not None) and ('DHS' in pupil))
    if is_dhs:
        raise NotImplementedError('DHS has yet to be fully included')

    t0 = time.time()
    # Get filter throughput and create bandpass
    if isinstance(filter_or_bp, six.string_types):
        filter = filter_or_bp
        bp = read_filter(filter, pupil=pupil, mask=mask, module=module, **kwargs)
    else:
        bp = filter_or_bp
        filter = bp.name

    if (coeff is not None) and (coeff_hdr is not None):
        fov_pix = coeff_hdr['FOVPIX']
        oversample = coeff_hdr['OSAMP']
        module = coeff_hdr['MODULE']
    elif (coeff is None) and (coeff_hdr is not None):
        raise AttributeError("`coeff_hdr` parameter set, but `coeff` is None")
    elif ((coeff is not None) and (coeff_hdr is None)):
        raise AttributeError("`coeff` parameter set, but `coeff_hdr` is None")
    else:
        coeff, coeff_hdr = gen_psf_coeff(bp, pupil=pupil, mask=mask, module=module, 
            fov_pix=fov_pix, oversample=oversample, **kwargs)

    t1 = time.time()
    waveset = np.copy(bp.wave)
    if nwaves is not None:
        # Evenly spaced waves 
        waveset = np.linspace(waveset.min(), waveset.max(), nwaves)
    elif not (is_grism or is_dhs):
        # For generating the PSF, let's save some time and memory by not using
        # ever single wavelength in the bandpass.
        # Do NOT do this for dispersed modes.
        binsize = 1
        if coeff.shape[-1]>2000:
            binsize = 7
        elif coeff.shape[-1]>1000:
            binsize = 5
        elif coeff.shape[-1]>700:
            binsize = 3

        if binsize>1:
            excess = waveset.size % binsize
            waveset = waveset[:waveset.size-excess]
            waveset = waveset.reshape(-1,binsize) # Reshape
            waveset = waveset[:,binsize//2] # Use the middle values
            waveset = np.concatenate(([bp.wave[0]],waveset,[bp.wave[-1]]))

    wgood = waveset / 1e4
    w1 = wgood.min()
    w2 = wgood.max()
    wrange = w2 - w1
    # print('nwaves: {}'.format(len(wgood)))

    t2 = time.time()
    # Flat spectrum with equal photon flux in each spectal bin
    if sp_norm is None:
        sp_flat = S.ArraySpectrum(waveset, 0*waveset + 10.)
        sp_flat.name = 'Flat spectrum in flam'

        # Bandpass unit response is the flux (in flam) of a star that
        # produces a response of one count per second in that bandpass
        sp_norm = sp_flat.renorm(bp.unit_response(), 'flam', bp)

    # Make sp_norm a list of spectral objects if it already isn't
    if not isinstance(sp_norm, list): 
        sp_norm = [sp_norm]
    nspec = len(sp_norm)

    t3 = time.time()
    # Set up an observation of the spectrum using the specified bandpass
    if use_sp_waveset:
        if nspec>1:
            raise AttributeError("Only 1 spectrum allowed when use_sp_waveset=True.")
        # Modify waveset if use_sp_waveset=True
        obs_list = []
        for sp in sp_norm:
            # Select only wavelengths within bandpass
            waveset = sp.wave
            waveset = waveset[(waveset>=w1*1e4) and (waveset<=w2*1e4)]
            obs_list.append(S.Observation(sp, bp, binset=waveset))
        # Update wgood
        wgood = waveset / 1e4
        w1 = wgood.min()
        w2 = wgood.max()
        wrange = w2 - w1
    else:
        # Use the bandpass wavelength set to bin the fluxes
        obs_list = [S.Observation(sp, bp, binset=waveset) for sp in sp_norm]

    # Convert to count rate
    for obs in obs_list: 
        obs.convert('counts')

    t4 = time.time()
    # Create a PSF for each wgood wavelength
    use_legendre = True if coeff_hdr['LEGNDR'] else False
    lxmap = [coeff_hdr['WAVE1'], coeff_hdr['WAVE2']]
    psf_fit = jl_poly(wgood, coeff, dim_reorder=True, use_legendre=use_legendre, lxmap=lxmap)
    # Just in case weird coeff gives negative values
    # psf_fit[psf_fit<=0] = np.min(psf_fit[psf_fit>0]) / 10

    t5 = time.time()
    # Multiply each monochromatic PSFs by the binned e/sec at each wavelength
    # Array broadcasting: [nx,ny,nwave] x [1,1,nwave]
    # Do this for each spectrum/observation
    if nspec==1:
        psf_fit *= obs_list[0].binflux
        psf_list = [psf_fit]
    else:
        psf_list = [psf_fit*obs.binflux for obs in obs_list]
        del psf_fit

    # The number of pixels to span spatially
    fov_pix = int(fov_pix)
    oversample = int(oversample)
    fov_pix_over = int(fov_pix * oversample)

    t6 = time.time()
    # Grism spectroscopy
    if is_grism:
        # spectral resolution in um/pixel
        # res is in pixels per um and dw is inverse
        grism_order = kwargs['grism_order'] if ('grism_order' in kwargs.keys()) else 1
        res, dw = grism_res(pupil, module, grism_order)

        # Number of real pixels that spectra will span
        npix_spec = int(wrange // dw + 1 + fov_pix)
        npix_spec_over = int(npix_spec * oversample)

        spec_list = []
        spec_list_over = []
        for psf_fit in psf_list:
            # If GRISM90 (along columns) rotate image by 90 deg CW 
            if 'GRISM90' in pupil:
                psf_fit = np.rot90(psf_fit, k=1) 

            # Create oversampled spectral image
            spec_over = np.zeros([fov_pix_over, npix_spec_over])
            # Place each PSF at its dispersed location
            for i, w in enumerate(wgood):
                # Separate shift into an integer and fractional shift
                delx = oversample * (w-w1) / dw # Number of oversampled pixels to shift
                intx = int(delx)
                fracx = delx - intx
                if fracx < 0:
                    fracx = fracx + 1
                    intx = intx - 1

                #spec_over[:,intx:intx+fov_pix_over] += fshift(psf_fit[:,:,i], fracx)
                im = psf_fit[:,:,i]
                spec_over[:,intx:intx+fov_pix_over] += im*(1.-fracx) + np.roll(im,1,axis=1)*fracx

            spec_over[spec_over<__epsilon] = 0 #__epsilon

            # Rotate spectrum to its V2/V3 coordinates
            spec_bin = poppy.utils.krebin(spec_over, (fov_pix,npix_spec))
            if 'GRISM90' in pupil: # Rotate image 90 deg CCW
                spec_over = np.rot90(spec_over, k=-1)
                spec_bin = np.rot90(spec_bin, k=-1)
            elif module=='B': # Flip right to left
                spec_over = spec_over[:,::-1]
                spec_bin = spec_bin[:,::-1]

            # Rebin ovesampled spectral image to real pixels
            spec_list.append(spec_bin)
            spec_list_over.append(spec_over)

        # Wavelength solutions
        dw_over = dw/oversample
        w1_spec = w1 - dw_over*fov_pix_over/2
        wspec_over = np.arange(npix_spec_over)*dw_over + w1_spec
        wspec = wspec_over.reshape((npix_spec,-1)).mean(axis=1)

        if nspec == 1: 
            spec_list = spec_list[0]
            spec_list_over = spec_list_over[0]
        # Return list of wavelengths for each horizontal pixel
        # as well as spectral image

        t7 = time.time()
        print('jl_poly: {:.2f} sec; binflux: {:.2f} sec; disperse: {:.2f} sec'.format(t5-t4, t6-t5, t7-t6))
        if return_oversample:
            return (wspec, spec_list), (wspec_over, spec_list_over)
        else:
            return (wspec, spec_list)

    # DHS spectroscopy
    elif is_dhs:
        raise NotImplementedError('DHS has yet to be fully included')

    # Imaging
    else:
        # Create source image slopes (no noise)
        data_list = []
        data_list_over = []
        for psf_fit in psf_list:
            data_over = psf_fit.sum(axis=2)
            data_over[data_over<=__epsilon] = data_over[data_over>__epsilon].min() / 10
            data_list_over.append(data_over)
            data_list.append(poppy.utils.krebin(data_over, (fov_pix,fov_pix)))

        if nspec == 1: 
            data_list = data_list[0]
            data_list_over = data_list_over[0]

        t7 = time.time()
        _log.debug('jl_poly: {:.2f} sec; binflux: {:.2f} sec; PSF sum: {:.2f} sec'.format(t5-t4, t6-t5, t7-t6))
        if return_oversample:
            return data_list, data_list_over
        else:
            return data_list

def radial_std(im_diff, pixscale=None, oversample=None, supersample=False, func=np.std):
    """Generate contrast curve of PSF difference

    Find the standard deviation within fixed radial bins of a differenced image.
    Returns two arrays representing the 1-sigma contrast curve at given distances.

    Parameters
    ==========
    im_diff : ndarray
        Differenced image of two PSFs, for instance.

    Keywords
    ========
    pixscale : float  
        Pixel scale of the input image
    oversample : int
        Is the input image oversampled compared to detector? If set, then
        the binsize will be pixscale*oversample (if supersample=False).
    supersample : bool
        If set, then oversampled data will have a binsize of pixscale,
        otherwise the binsize is pixscale*oversample.
    func_std : func
        The function to use for calculating the radial standard deviation.

    """

    from astropy.convolution import convolve, Gaussian1DKernel

    # Set oversample to 1 if supersample keyword is set
    oversample = 1 if supersample or (oversample is None) else oversample

    # Rebin data
    data_rebin = frebin(im_diff, scale=1/oversample)

    # Determine pixel scale of rebinned data
    pixscale = 1 if pixscale is None else oversample*pixscale

    # Pixel distances
    rho = dist_image(data_rebin, pixscale=pixscale)

    # Get radial profiles
    binsize = pixscale
    bins = np.arange(rho.min(), rho.max() + binsize, binsize)
    nan_mask = np.isnan(data_rebin)
    igroups, _, rr = hist_indices(rho[~nan_mask], bins, True)
    stds = binned_statistic(igroups, data_rebin[~nan_mask], func=func)
    stds = convolve(stds, Gaussian1DKernel(1))

    # Ignore corner regions
    arr_size = np.min(data_rebin.shape) * pixscale
    mask = rr < (arr_size/2)

    return rr[mask], stds[mask]

###########################################################################
#
#    Sensitivities and Saturation Limits
#
###########################################################################


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

def grism_res(pupil='GRISM', module='A', m=1):
    """Grism information

    Based on the pupil input and module, return the spectral
    dispersion and resolution as a tuple (res, dw).

    Parameters
    ----------
    pupil : str
        'GRISM0' or 'GRISM90', otherwise assume res=1000 pix/um
    module : str
        'A' or 'B'
    m : int
        Spectral order (1 or 2).
    """
    # Mean spectral dispersion in number of pixels per um
    res = 1000.0
    if ('GRISM90' in pupil) and (module == 'A'):
        res = 1003.12
    if ('GRISM0' in pupil)  and (module == 'A'):
        res = 996.48
    if ('GRISM90' in pupil) and (module == 'B'):
        res = 1008.64
    if ('GRISM0' in pupil)  and (module == 'B'):
        res = 1009.13

    if m==2:
        res *= 2

    # Spectral resolution in um/pixel
    dw = 1. / res

    return (res, dw)


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
    if pix_scale is None: pix_scale = args[0] # Pixel scale (arcsec/pixel)

    # Spectrum and bandpass to report magnitude that saturates NIRCam band
    if sp is None: sp = stellar_spectrum('G2V')

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
#         return {'satlim':res_point, 'units':units, 'Spectrum':sp_norm.name,
#             'bp_lim':bp_lim.name}


def pix_noise(ngroup=2, nf=1, nd2=0, tf=10.737, rn=15.0, ktc=29.0, p_excess=(0,0),
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
        Frame timel
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
        otherwise MULTIACCUM equation is used?
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
    var_ex = 12. * (n - 1.)/(n + 1.) * p_excess[0]**2 - p_excess[1] / m**0.5

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

def bp_2mass(filter):
    """2MASS Bandpass

    Create a 2MASS J, H, or Ks filter bandpass used to generate
    synthetic photometry.

    Parameters
    ----------
    filter : str
        Filter 'j', 'h', or 'k'.

    Returns
    -------
    :mod:`pysynphot.obsbandpass`
        A Pysynphot bandpass object.

    """

    dir = conf.PYNRC_PATH + 'throughputs/2MASS/'
    if 'j' in filter.lower():
        file = '2mass_j.txt'
        name = 'J-Band'
    elif 'h' in filter.lower():
        file = '2mass_h.txt'
        name = 'H-Band'
    elif 'k' in filter.lower():
        file = '2mass_ks.txt'
        name = 'Ks-Band'
    else:
        raise ValueError('{} not a valid 2MASS filter'.format(filter))

    tbl = ascii.read(dir + file, names=['Wave', 'Throughput'])
    bp = S.ArrayBandpass(tbl['Wave']*1e4, tbl['Throughput'], name=name)

    return bp

def bp_wise(filter):
    """WISE Bandpass

    Create a WISE W1-W4 filter bandpass used to generate
    synthetic photometry.

    Parameters
    ----------
    filter : str
        Filter 'w1', 'w2', 'w3', or 'w4'.

    Returns
    -------
    :mod:`pysynphot.obsbandpass`
        A Pysynphot bandpass object.

    """

    dir = conf.PYNRC_PATH + 'throughputs/WISE/'
    if 'w1' in filter.lower():
        file = 'RSR-W1.txt'
        name = 'W1'
    elif 'w2' in filter.lower():
        file = 'RSR-W2.txt'
        name = 'W2'
    elif 'w3' in filter.lower():
        file = 'RSR-W3.txt'
        name = 'W3'
    elif 'w4' in filter.lower():
        file = 'RSR-W4.txt'
        name = 'W4'
    else:
        raise ValueError('{} not a valid WISE filter'.format(filter))

    tbl = ascii.read(dir + file, data_start=0)
    bp = S.ArrayBandpass(tbl['col1']*1e4, tbl['col2'], name=name)

    return bp


def bin_spectrum(sp, wave, waveunits='um'):
    """Rebin spectrum

    Rebin a :mod:`pysynphot.spectrum` to a lower wavelength grid.
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
    binflux = binned_statistic(sp.wave, sp.flux, np.mean, bins=edges)

    sp2 = S.ArraySpectrum(wave, binflux, waveunits=waveunits, fluxunits='flam')
    sp2.convert(waveunits0)
    sp2.convert(fluxunits0)

    # Put back units of original input spectrum
    sp.convert(waveunits0)
    sp.convert(fluxunits0)

    return sp2


def BOSZ_spectrum(Teff, metallicity, log_g, res=2000, interpolate=True, **kwargs):
    """BOSZ stellar atmospheres (Bohlin et al 2017).

    Read in a spectrum from the BOSZ stellar atmosphere models database.
    Returns a Pysynphot spectral object. Wavelength values range between
    1000-32000 Angstroms. Teff range from 3500K to 36000K.

    This function interoplates the model grid by reading in those models
    closest in temperature, metallicity, and log g to the desired parameters,
    then takes the weighted average of these models based on their relative
    offsets. Can also just read in the closest model by setting interpolate=False.

    Different spectral resolutions can also be specified, currently only
    res=200 or 2000 or 20000.

    Parameters
    ----------
    Teff : float
        Effective temperature ranging from 3500K to 30000K.
    metallicity : float
        Metallicity [Fe/H] value ranging from -2.5 to 0.5.
    log_g : float
        Surface gravity (log g) from 0 to 5.

    Keyword Args
    ------------
    res : str
        Spectral resolution to use (200 or 2000 or 20000).
    interpolate : bool
        Interpolate spectrum using a weighted average of grid points
        surrounding the desired input parameters.


    References
    ----------
    https://archive.stsci.edu/prepds/bosz/
    """

    model_dir = conf.PYNRC_PATH + 'bosz_grids/'
    res_dir = model_dir + 'R{}/'.format(res)
    if not os.path.isdir(model_dir):
        raise IOError('BOSZ model directory does not exist: {}'.format(model_dir))
    if not os.path.isdir(res_dir):
        raise IOError('Resolution directory does not exist: {}'.format(res_dir))

    # Grid of computed temperature steps
    teff_grid = list(range(3500,12000,250)) \
                + list(range(12000,20000,500)) \
                + list(range(20000,36000,1000))
    teff_grid = np.array(teff_grid)

    # Grid of log g steps for desired Teff
    lg_max = 5
    lg_step = 0.5
    if   Teff <   6250: lg_min = 0
    elif Teff <   8250: lg_min = 1
    elif Teff <  12500: lg_min = 2
    elif Teff <  21000: lg_min = 3
    elif Teff <= 30000: lg_min = 4
    else: raise ValueError('Teff must be less than or equal to 30000.')

    if log_g<lg_min:
        raise ValueError('log_g must be >={}'.format(lg_min))
    if log_g>lg_max:
        raise ValueError('log_g must be <={}'.format(lg_max))

    # Grid of log g values
    logg_grid = np.arange(lg_min, lg_max+lg_step, lg_step)

    # Grid of metallicity values
    metal_grid = np.arange(-2.5,0.75,0.25)

    # First, choose the two grid points closest in Teff
    teff_diff = np.abs(teff_grid - Teff)
    ind_sort = np.argsort(teff_diff)
    if teff_diff[ind_sort[0]]==0: # Exact
        teff_best = np.array([teff_grid[ind_sort[0]]])
    else: # Want to interpolate
        teff_best = teff_grid[ind_sort[0:2]]

    # Choose the two best log g values
    logg_diff = np.abs(logg_grid - log_g)
    ind_sort = np.argsort(logg_diff)
    if logg_diff[ind_sort[0]]==0: # Exact
        logg_best = np.array([logg_grid[ind_sort[0]]])
    else: # Want to interpolate
        logg_best = logg_grid[ind_sort[0:2]]

    # Choose the two best metallicity values
    metal_diff = np.abs(metal_grid - metallicity)
    ind_sort = np.argsort(metal_diff)
    if metal_diff[ind_sort[0]]==0: # Exact
        metal_best = np.array([metal_grid[ind_sort[0]]])
    else: # Want to interpolate
        metal_best = metal_grid[ind_sort[0:2]]

    # Build files names for all combinations
    teff_names = np.array(['t{:04.0f}'.format(n) for n in teff_best])
    logg_names = np.array(['g{:02.0f}'.format(int(n*10)) for n in logg_best])
    metal_names = np.array(['mp{:02.0f}'.format(int(abs(n*10)+0.5)) for n in metal_best])
    ind_n = np.where(metal_best<0)[0]
    for i in range(len(ind_n)):
        j = ind_n[i]
        s = metal_names[j]
        metal_names[j] = s.replace('p', 'm')

    # Build final file names
    fnames = []
    rstr = 'b{}'.format(res)
    for t in teff_names:
        for l in logg_names:
            for m in metal_names:
                fname = 'a{}cp00op00{}{}v20modrt0{}rs.fits'.format(m,t,l,rstr)
                fnames.append(fname)

    # Weight by relative distance from desired value
    weights = []
    teff_diff = np.abs(teff_best - Teff)
    logg_diff = np.abs(logg_best - log_g)
    metal_diff = np.abs(metal_best - metallicity)
    for t in teff_diff:
        wt = 1 if len(teff_diff)==1 else t / np.sum(teff_diff)
        for l in logg_diff:
            wl = 1 if len(logg_diff)==1 else l / np.sum(logg_diff)
            for m in metal_diff:
                wm = 1 if len(metal_diff)==1 else m / np.sum(metal_diff)
                weights.append(wt*wl*wm)
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    if interpolate:
        wave_all = []
        flux_all = []
        for i, f in enumerate(fnames):
            d = fits.getdata(res_dir+f, 1)
            wave_all.append(d['Wavelength'])
            flux_all.append(d['SpecificIntensity'] * weights[i])

        wfin = wave_all[0]
        ffin = np.pi * np.array(flux_all).sum(axis=0) # erg/s/cm^2/A
    else:
        ind = np.where(weights==weights.max())[0][0]
        f = fnames[ind]
        d = fits.getdata(res_dir+f, 1)
        wfin = d['Wavelength']
        ffin = np.pi * d['SpecificIntensity'] # erg/s/cm^2/A

        Teff = teff_best[ind]
        log_g = logg_best[ind]
        metallicity = metal_best[ind]

    name = 'BOSZ(Teff={},z={},logG={})'.format(Teff,metallicity,log_g)
    sp = S.ArraySpectrum(wfin[:-1], ffin[:-1], 'angstrom', 'flam', name=name)

    return sp

def stellar_spectrum(sptype, *renorm_args, **kwargs):
    """Stellar spectrum

    Similar to specFromSpectralType() in WebbPSF/Poppy, this function uses
    a dictionary of fiducial values to determine an appropriate spectral model.
    If the input spectral type is not found, this function interpolates the
    effective temperature, metallicity, and log g values .

    You can also specify renormalization arguments to pass to ``sp.renorm()``.
    The order (after ``sptype``) should be (``value, units, bandpass``):

    >>> sp = stellar_spectrum('G2V', 10, 'vegamag', bp)

    Flat spectrum (in photlam) are also allowed via the 'flat' string.

    Use ``catname='bosz'`` for BOSZ stellar atmosphere (ATLAS9) (default)
    Use ``catname='ck04models'`` keyword for ck04 models
    Use ``catname='phoenix'`` keyword for Phoenix models

    Keywords exist to directly specify Teff, metallicity, an log_g rather
    than a spectral type.

    Parameters
    ----------
    sptype : str
        Spectral type, such as 'A0V' or 'K2III'.
    renorm_args : tuple
        Renormalization arguments to pass to ``sp.renorm()``.
        The order (after ``sptype``) should be (``value, units, bandpass``)
        Bandpass should be a :mod:`pysynphot.obsbandpass` type.

    Keyword Args
    ------------
    catname : str
        Catalog name, including 'bosz', 'ck04models', and 'phoenix'.
        Default is 'bosz', which comes from :func:`BOSZ_spectrum`.
    Teff : float
        Effective temperature ranging from 3500K to 30000K.
    metallicity : float
        Metallicity [Fe/H] value ranging from -2.5 to 0.5.
    log_g : float
        Surface gravity (log g) from 0 to 5.
    res : str
        BOSZ spectral resolution to use (200 or 2000 or 20000).
        Default: 2000.
    interpolate : bool
        Interpolate BOSZ spectrum using a weighted average of grid points
        surrounding the desired input parameters. Default is True.
        Default: True
    """

    Teff = kwargs.pop('Teff', None)
    metallicity = kwargs.pop('metallicity', None)
    log_g = kwargs.pop('log_g', None)

    catname = kwargs.get('catname')
    if catname is None: catname = 'bosz'
    lookuptable = {
        "O0V": (50000, 0.0, 4.0), # Bracketing for interpolation
        "O3V": (45000, 0.0, 4.0),
        "O5V": (41000, 0.0, 4.5),
        "O7V": (37000, 0.0, 4.0),
        "O9V": (33000, 0.0, 4.0),
        "B0V": (30000, 0.0, 4.0),
        "B1V": (25000, 0.0, 4.0),
        "B3V": (19000, 0.0, 4.0),
        "B5V": (15000, 0.0, 4.0),
        "B8V": (12000, 0.0, 4.0),
        "A0V": (9500, 0.0, 4.0),
        "A1V": (9250, 0.0, 4.0),
        "A3V": (8250, 0.0, 4.0),
        "A5V": (8250, 0.0, 4.0),
        "F0V": (7250, 0.0, 4.0),
        "F2V": (7000, 0.0, 4.0),
        "F5V": (6500, 0.0, 4.0),
        "F8V": (6250, 0.0, 4.5),
        "G0V": (6000, 0.0, 4.5),
        "G2V": (5750, 0.0, 4.5),
        "G5V": (5650, 0.0, 4.5),
        "G8V": (5500, 0.0, 4.5),
        "K0V": (5250, 0.0, 4.5),
        "K2V": (4750, 0.0, 4.5),
        "K5V": (4250, 0.0, 4.5),
        "K7V": (4000, 0.0, 4.5),
        "M0V": (3750, 0.0, 4.5),
        "M2V": (3500, 0.0, 4.5),
        "M5V": (3500, 0.0, 5.0),
        "M9V": (3000, 0.0, 5.0),   # Bracketing for interpolation
        "O0IV": (50000, 0.0, 3.8), # Bracketing for interpolation
        "B0IV": (30000, 0.0, 3.8),
        "B8IV": (12000, 0.0, 3.8),
        "A0IV": (9500, 0.0, 3.8),
        "A5IV": (8250, 0.0, 3.8),
        "F0IV": (7250, 0.0, 3.8),
        "F8IV": (6250, 0.0, 4.3),
        "G0IV": (6000, 0.0, 4.3),
        "G8IV": (5500, 0.0, 4.3),
        "K0IV": (5250, 0.0, 4.3),
        "K7IV": (4000, 0.0, 4.3),
        "M0IV": (3750, 0.0, 4.3),
        "M9IV": (3000, 0.0, 4.7),    # Bracketing for interpolation
        "O0III": (55000, 0.0, 3.5), # Bracketing for interpolation
        "B0III": (29000, 0.0, 3.5),
        "B5III": (15000, 0.0, 3.5),
        "G0III": (5750, 0.0, 3.0),
        "G5III": (5250, 0.0, 2.5),
        "K0III": (4750, 0.0, 2.0),
        "K5III": (4000, 0.0, 1.5),
        "M0III": (3750, 0.0, 1.5),
        "M6III": (3000, 0.0, 1.0), # Bracketing for interpolation
        "O0I": (45000, 0.0, 5.0),  # Bracketing for interpolation
        "O6I": (39000, 0.0, 4.5),
        "O8I": (34000, 0.0, 4.0),
        "B0I": (26000, 0.0, 3.0),
        "B5I": (14000, 0.0, 2.5),
        "A0I": (9750, 0.0, 2.0),
        "A5I": (8500, 0.0, 2.0),
        "F0I": (7750, 0.0, 2.0),
        "F5I": (7000, 0.0, 1.5),
        "G0I": (5500, 0.0, 1.5),
        "G5I": (4750, 0.0, 1.0),
        "K0I": (4500, 0.0, 1.0),
        "K5I": (3750, 0.0, 0.5),
        "M0I": (3750, 0.0, 0.0),
        "M2I": (3500, 0.0, 0.0),
        "M5I": (3000, 0.0, 0.0)} # Bracketing for interpolation

    def sort_sptype(typestr):
        letter = typestr[0]
        lettervals = {'O': 0, 'B': 1, 'A': 2, 'F': 3, 'G': 4, 'K': 5, 'M': 6}
        value = lettervals[letter] * 1.0
        value += (int(typestr[1]) * 0.1)
        if "III" in typestr:
            value += 30
        elif "I" in typestr:
            value += 10
        elif "V" in typestr:
            value += 50
        return value

    # Generate list of spectral types
    sptype_list = list(lookuptable.keys())

    # Test if the user wants a flat spectrum (in photlam)
    # Check if Teff, metallicity, and log_g are specified
    if (Teff is not None) and (metallicity is not None) and (log_g is not None):
        v0, v1, v2 = (Teff, metallicity, log_g)
        if 'bosz' in catname.lower():
            sp = BOSZ_spectrum(v0, v1, v2, **kwargs)
        else:
            if ('ck04models' in catname.lower()) and (v0<3500):
                _log.warn("ck04 models stop at 3500K. Setting Teff=3500.")
                v0 = 3500
            sp = S.Icat(catname, v0, v1, v2)
        sp.name = '({:.0f},{:0.1f},{:0.1f})'.format(v0,v1,v2)
    elif 'flat' in sptype.lower():
        # waveset = S.refs._default_waveset
        # sp = S.ArraySpectrum(waveset, 0*waveset + 10.)
        sp = S.FlatSpectrum(10, fluxunits='photlam')
        sp.name = 'Flat spectrum in photlam'
    elif sptype in sptype_list:
        v0,v1,v2 = lookuptable[sptype]

        if 'bosz' in catname.lower():
            sp = BOSZ_spectrum(v0, v1, v2, **kwargs)
        else:
            if ('ck04models' in catname.lower()) and (v0<3500):
                _log.warn("ck04 models stop at 3500K. Setting Teff=3500.")
                v0 = 3500
            sp = S.Icat(catname, v0, v1, v2)
        sp.name = sptype
    else: # Interpolate values for undefined sptype
        # Sort the list and return their rank values
        sptype_list.sort(key=sort_sptype)
        rank_list = np.array([sort_sptype(st) for st in sptype_list])
        # Find the rank of the input spec type
        rank = sort_sptype(sptype)
        # Grab values from tuples and interpolate based on rank
        tup_list0 = np.array([lookuptable[st][0] for st in sptype_list])
        tup_list1 = np.array([lookuptable[st][1] for st in sptype_list])
        tup_list2 = np.array([lookuptable[st][2] for st in sptype_list])
        v0 = np.interp(rank, rank_list, tup_list0)
        v1 = np.interp(rank, rank_list, tup_list1)
        v2 = np.interp(rank, rank_list, tup_list2)

        if 'bosz' in catname.lower():
            sp = BOSZ_spectrum(v0, v1, v2, **kwargs)
        else:
            if ('ck04models' in catname.lower()) and (v0<3500):
                _log.warn("ck04 models stop at 3500K. Setting Teff=3500.")
                v0 = 3500
            sp = S.Icat(catname, v0, v1, v2)
        sp.name = sptype

    #print(int(v0),v1,v2)

    # Renormalize if those args exist
    if len(renorm_args) > 0:
        sp_norm = sp.renorm(*renorm_args)
        sp_norm.name = sp.name
        sp = sp_norm

    return sp




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


    if zfact is None: zfact = 2.5
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

    The program relies on ``urllib2`` to download the page in XML format.
    However, the website only allows single wavelength queries, so
    this program implements a multithreaded procedure to query
    multiple wavelengths simultaneously. However, due to the nature
    of ``urllib2`` library, only so many requests are allowed to go
    out at a time, so this process can take some time to complete.
    Testing shows about 500 wavelengths in 10 seconds as a rough ballpark.

    Recommended to grab only a few wavelengths for normalization purposes.

    References
    ----------
    See the `Euclid Help Website
    <http://irsa.ipac.caltech.edu/applications/BackgroundModel/docs/dustProgramInterface.html>`_
    for more details.

    """

    from urllib2 import urlopen
    import xmltodict
    from multiprocessing.pool import ThreadPool

    def fetch_url(url):
        """
        TODO: Add error handling.
        """
        response = urlopen(url)
        response = response.read()
        d = xmltodict.parse(response, xml_attribs=True)
        fl_str = d['results']['result']['statistics']['zody']
        return float(fl_str.split(' ')[0])


    #locstr="17:26:44 -73:19:56"
    locstr = locstr.replace(' ', '+')
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

# Class for creating an input source spectrum
class source_spectrum(object):
    """Model source spectrum

    The class ingests spectral information of a given target
    and generates :mod:`pysynphot.spectrum` model fit to the
    known photometric SED. Two model routines can fit. The
    first is a very simple scale factor that is applied to the
    input spectrum, while the second takes the input spectrum
    and adds an IR excess modeled as a modified blackbody function.

    Parameters
    ----------
    name : string
        Source name.
    sptype : string
        Assumed stellar spectral type. Not relevant if Teff, metallicity,
        and log_g are specified.
    mag_val : float
        Magnitude of input bandpass for initial scaling of spectrum.
    bp : :mod:`pysynphot.obsbandpass`
        Bandpass to apply initial mag_val scaling.
    votable_file: string
        VOTable name that holds the source's photometry. The user can
        find the relevant data at http://vizier.u-strasbg.fr/vizier/sed/
        and click download data.

    Keyword Args
    ------------
    Teff : float
        Effective temperature ranging from 3500K to 30000K.
    metallicity : float
        Metallicity [Fe/H] value ranging from -2.5 to 0.5.
    log_g : float
        Surface gravity (log g) from 0 to 5.
    catname : str
        Catalog name, including 'bosz', 'ck04models', and 'phoenix'.
        Default is 'bosz', which comes from :func:`BOSZ_spectrum`.
    res : str
        Spectral resolution to use (200 or 2000 or 20000).
    interpolate : bool
        Interpolate spectrum using a weighted average of grid points
        surrounding the desired input parameters.

    Example
    -------
    Generate a source spectrum and fit photometric data

    >>> import pynrc
    >>> from pynrc.nrc_utils import source_spectrum
    >>>
    >>> name = 'HR8799'
    >>> vot = 'votables/{}.vot'.format(name)
    >>> bp_k = pynrc.bp_2mass('k')
    >>>
    >>> # Read in stellar spectrum model and normalize to Ks = 5.24
    >>> src = source_spectrum(name, 'F0V', 5.24, bp_k, vot,
    >>>                       Teff=7430, metallicity=-0.47, log_g=4.35)
    >>> # Fit model to photometry from 0.1 - 30 micons
    >>> # Saves pysynphot spectral object at src.sp_model
    >>> src.fit_SED(wlim=[0.1,30])
    >>> sp_sci = src.sp_model

    """

    def __init__(self, name, sptype, mag_val, bp, votable_file,
                 Teff=None, metallicity=None, log_g=None, Av=None, **kwargs):

        self.name = name

        # Setup initial spectrum
        kwargs['Teff']        = Teff
        kwargs['metallicity'] = metallicity
        kwargs['log_g']       = log_g
        self.sp0 = stellar_spectrum(sptype, mag_val, 'vegamag', bp, **kwargs)

        # Read in a low res version for photometry matching
        kwargs['res'] = 200
        self.sp_lowres = stellar_spectrum(sptype, mag_val, 'vegamag', bp, **kwargs)

        if Av is not None:
            Rv = 4
            self.sp0 = self.sp0 * S.Extinction(Av/Rv,name='mwrv4')
            self.sp_lowres = self.sp_lowres * S.Extinction(Av/Rv,name='mwrv4')

            self.sp0 = self.sp0.renorm(mag_val, 'vegamag', bp)
            self.sp_lowres = self.sp_lowres.renorm(mag_val, 'vegamag', bp)

            self.sp0.name = sptype
            self.sp_lowres.name = sptype

        # Init model to None
        self.sp_model = None

        # Readin photometry
        self.votable_file = votable_file
        self._gen_table()
        self._combine_fluxes()

    def _gen_table(self):
        """Read VOTable and convert to astropy table"""
        # Import source SED from VOTable
        from astropy.io.votable import parse_single_table
        table = parse_single_table(self.votable_file)
        # Convert to astropy table
        tbl = table.to_table()

        freq = tbl['sed_freq'] * 1e9 # Hz
        wave_m = 2.99792458E+08 / freq
        wave_A = 1e10 * wave_m

        # Add wavelength column
        col = tbl.Column(wave_A, 'sed_wave')
        col.unit = 'Angstrom'
        tbl.add_column(col)

        # Sort flux monotomically with wavelength
        tbl.sort(['sed_wave', 'sed_flux'])

        self.table = tbl

    def _combine_fluxes(self):
        """Average duplicate data points

        Creates average of duplicate point stored in self.sp_phot.
        """

        table = self.table

        wave = table['sed_wave']
        flux = table["sed_flux"]
        eflux = table["sed_eflux"]

        # Average duplicate data points
        uwave, ucnt = np.unique(wave, return_counts=True)
        uflux = []
        uflux_e = []
        for i, w in enumerate(uwave):
            ind = (wave==w)
            flx = np.median(flux[ind]) if ucnt[i]>1 else flux[ind][0]
            uflux.append(flx)

            eflx = robust.medabsdev(flux[ind]) if ucnt[i]>1 else eflux[ind][0]
            uflux_e.append(eflx)
        uflux = np.array(uflux)
        uflux_e = np.array(uflux_e)

        # Photometric data points
        sp_phot = S.ArraySpectrum(uwave, uflux,
                                  waveunits=wave.unit.name,
                                  fluxunits=flux.unit.name)
        sp_phot.convert('Angstrom')
        sp_phot.convert('Flam')

        sp_phot_e = S.ArraySpectrum(uwave, uflux_e,
                                    waveunits=wave.unit.name,
                                    fluxunits=eflux.unit.name)
        sp_phot_e.convert('Angstrom')
        sp_phot_e.convert('Flam')


        self.sp_phot = sp_phot
        self.sp_phot_e = sp_phot_e


    def bb_jy(self, wave, T):
        """Blackbody function (Jy)

        For a given wavelength set (in um) and a Temperature (K),
        return the blackbody curve in units of Jy.

        Parameters
        ----------
        wave : array_like
            Wavelength array in microns
        T : float
            Temperature of blackbody (K)
        """

        # Physical Constants
        #H  = 6.62620000E-27  # Planck's constant in cgs units
        HS = 6.62620000E-34  # Planck's constant in standard units
        C  = 2.99792458E+08  # speed of light in standard units
        K  = 1.38064852E-23  # Boltzmann constant in standard units

        # Blackbody coefficients (SI units)
        C1 = 2.0 * HS * C    # Power * unit area / steradian
        C2 = HS * C / K

        w_m = wave * 1e-6

        exponent = C2 / (w_m * T)
        expfactor = np.exp(exponent)

        return 1.0E+26 * C1 * (w_m**-3.0) / (expfactor - 1.0)


    def model_scale(self, x, sp=None):
        """Simple model to scale stellar spectrum"""

        sp = self.sp_lowres if sp is None else sp
        return x[0] * sp

    def model_IRexcess(self, x, sp=None):
        """Model for stellar spectrum with IR excesss

        Model of a stellar spectrum plus IR excess, where the
        excess is a modified blackbody. The final model follows
        the form:

        .. math::

            x_0 BB(\lambda, x_1) \lambda^{x_2}
        """

        sp = self.sp_lowres if sp is None else sp

        bb_flux = x[0] * self.bb_jy(sp.wave/1e4, x[1]) * (sp.wave/1e4)**x[2] / 1e17
        sp_bb = S.ArraySpectrum(sp.wave, bb_flux, fluxunits='Jy')
        sp_bb.convert('Flam')

        return sp + sp_bb


    def func_resid(self, x, IR_excess=False, wlim=[0.1, 30], use_err=True):
        """Calculate model residuals

        Parameters
        ----------
        x : array_like
            Model parameters for either `model_scale` or `model_IRexcess`.
            See these two functions for more details.
        IR_excess: bool
            Include IR excess in model fit? This is a simple modified blackbody.
        wlim : array_like
            Min and max limits for wavelengths to consider (microns).
        use_err : bool
            Should we use the uncertainties in the SED photometry for weighting?
        """

        # Star model and photometric data
        sp_star = self.sp_lowres
        sp_phot = self.sp_phot
        sp_phot_e = self.sp_phot_e

        # Which model are we using?
        func_model = self.model_IRexcess if IR_excess else self.model_scale

        sp_model = func_model(x, sp_star)

        wvals = sp_phot.wave
        wmin, wmax = np.array(wlim)*1e4
        ind = (wvals >= wmin) & (wvals <= wmax)

        wvals = wvals[ind]
        yvals = sp_phot.flux[ind]
        evals = sp_phot_e.flux[ind]

        # Instead of interpolating on a high-resolution grid,
        # we should really rebin onto a more coarse grid.
        mod_interp = np.interp(wvals, sp_star.wave, sp_model.flux)

        # Normalize values so the residuals aren't super small/large
        norm = np.mean(yvals)

        resid = (mod_interp - yvals)
        if use_err: resid /= evals

        # Return non-NaN normalized values
        return resid[~np.isnan(resid)] / norm

    def fit_SED(self, x0=None, robust=True, use_err=True, IR_excess=False,
                 wlim=[0.3,10], verbose=True):

        """Fit a model function to photometry

        Use :func:`scipy.optimize.least_squares` to find the best fit
        model to the observed photometric data. If not parameters passed,
        then defaults are set.

        Keyword Args
        ------------
        x0 : array_like
            Initial guess of independent variables.
        robust : bool
            Perform an outlier-resistant fit.
        use_err : bool
            Should we use the uncertainties in the SED photometry for weighting?
        IR_excess: bool
            Include IR excess in model fit? This is a simple modified blackbody.
        wlim : array_like
            Min and max limits for wavelengths to consider (microns).
        verbose : bool
            Print out best-fit model parameters. Defalt is True.
        """

        # Default initial starting parameters
        if x0 is None:
            x0 = [1.0, 2000.0, 0.5] if IR_excess else [1.0]

        # Robust fit?
        loss = 'soft_l1' if robust else 'linear'

        # Perform least-squares fit
        kwargs={'IR_excess':IR_excess, 'wlim':wlim, 'use_err':use_err}
        res = least_squares(self.func_resid, x0, bounds=(0,np.inf), loss=loss,
                            kwargs=kwargs)
        out = res.x
        if verbose: print(out)

        # Which model are we using?
        func_model = self.model_IRexcess if IR_excess else self.model_scale
        # Create final model spectrum
        sp_model = func_model(out, self.sp0)
        sp_model.name = self.name

        self.sp_model = sp_model

    def plot_SED(self, ax=None, return_figax=False, xr=[0.3,30], yr=None,
                     units='Jy', **kwargs):

        sp0 = self.sp0
        sp_phot = self.sp_phot
        sp_phot_e = self.sp_phot_e
        sp_model = self.sp_model

        # Convert to Jy and save original units
        sp0_units = sp0.fluxunits.name
        sp_phot_units = sp_phot.fluxunits.name

        # nuFnu or lamFlam?
        if (units=='nufnu') or (units=='lamflam'):
            units = 'flam'
            lfl = True
        else:
            lfl = False

        sp0.convert(units)
        sp_phot.convert(units)

        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(8,5))

        w = sp0.wave / 1e4
        f = sp0.flux
        if lfl:
            f = f * sp0.wave
        if xr is not None:
            ind = (w>=xr[0]) & (w<=xr[1])
            w, f = (w[ind], f[ind])
        ax.loglog(w, f, lw=1, label='Photosphere', **kwargs)

        w = sp_phot.wave / 1e4
        f = sp_phot.flux
        f_err = sp_phot_e.flux
        if lfl:
            f = f * sp_phot.wave
            f_err = f_err * sp_phot.wave
        if xr is not None:
            ind = (w>=xr[0]) & (w<=xr[1])
            w, f, f_err = (w[ind], f[ind], f_err[ind])
        ax.errorbar(w, f, yerr=f_err, marker='.', ls='none', label='Photometry')

        if sp_model is not None:
            sp_model_units = sp_model.fluxunits.name
            sp_model.convert(units)

            w = sp_model.wave / 1e4
            f = sp_model.flux
            if lfl:
                f = f * sp_model.wave
            if xr is not None:
                ind = (w>=xr[0]) & (w<=xr[1])
                w, f = (w[ind], f[ind])

            ax.plot(w, f, lw=1, label='Model Fit')
            sp_model.convert(sp_model_units)

        # Labels for various units
        ulabels = {'photlam': u'photons s$^{-1}$ cm$^{-2}$ A$^{-1}$',
                   'photnu' : u'photons s$^{-1}$ cm$^{-2}$ Hz$^{-1}$',
                   'flam'   : u'erg s$^{-1}$ cm$^{-2}$ A$^{-1}$',
                   'fnu'    : u'erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$',
                   'counts' : u'photons s$^{-1}$',
                  }
        if lfl: # Special case nuFnu or lamFlam
            yunits = u'erg s$^{-1}$ cm$^{-2}$'
        else:
            yunits = ulabels.get(units, units)

        ax.set_xlabel('Wavelength (microns)')
        ax.set_ylabel('Flux ({})'.format(yunits))
        ax.set_title(self.name)

        if xr is not None:
            ax.set_xlim(xr)
        if yr is not None:
            ax.set_ylim(yr)

        # Better formatting of ticks marks
        from matplotlib.ticker import LogLocator, AutoLocator, NullLocator
        from matplotlib.ticker import FuncFormatter, NullFormatter
        formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))

        xr = ax.get_xlim()
        if xr[1] < 10*xr[0]:
            ax.xaxis.set_major_locator(AutoLocator())
            ax.xaxis.set_minor_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(LogLocator())
        ax.xaxis.set_major_formatter(formatter)

        yr = ax.get_ylim()
        if yr[1] < 10*yr[0]:
            ax.yaxis.set_major_locator(AutoLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.get_major_locator().set_params(nbins=10, steps=[1,10])
        else:
            ax.yaxis.set_major_locator(LogLocator())
        ax.yaxis.set_major_formatter(formatter)

        ax.legend()

        # Convert back to original units
        sp0.convert(sp0_units)
        sp_phot.convert(sp_phot_units)

        if ax is None:
            fig.tight_layout()
            if return_figax: return (fig,ax)



# Class for reading in planet spectra
class planets_sb12(object):
    """Exoplanet spectrum from Spiegel & Burrows (2012)

    This contains 1680 files, one for each of 4 atmosphere types, each of
    15 masses, and each of 28 ages.  Wavelength range of 0.8 - 15.0 um at
    moderate resolution (R ~ 204).

    The flux in the source files are at 10 pc. If the distance is specified,
    then the flux will be scaled accordingly. This is also true if the distance
    is changed by the user. All other properties (atmo, mass, age, entropy) are
    not adjustable once loaded.

    Parameters
    ----------
    atmo: str
        A string consisting of one of four atmosphere types:

            - 'hy1s' = hybrid clouds, solar abundances
            - 'hy3s' = hybrid clouds, 3x solar abundances
            - 'cf1s' = cloud-free, solar abundances
            - 'cf3s' = cloud-free, 3x solar abundances

    mass: float
        A number 1 to 15 Jupiter masses.
    age: float
        Age in millions of years (1-1000)
    entropy: float
        Initial entropy (8.0-13.0) in increments of 0.25
    distance: float
        Assumed distance in pc (default is 10pc)
    accr : bool
        Include accretion (default: False)?
    mmdot : float
        From Zhu et al. (2015), the Mjup^2/yr value.
        If set to None then calculated from age and mass.
    mdot : float
        Or use mdot (Mjup/yr) instead of mmdot.
    accr_rin : float
        Inner radius of accretion disk (units of RJup; default: 2)
    truncated: bool
         Full disk or truncated (ie., MRI; default: False)?
    base_dir: str, None
        Location of atmospheric model sub-directories.
    """

	# Define default self.base_dir
    _base_dir = conf.PYNRC_PATH + 'spiegel/'

    def __init__(self, atmo='hy1s', mass=1, age=100, entropy=10.0, distance=10,
                 accr=False, mmdot=None, mdot=None, accr_rin=2.0, truncated=False,
                 base_dir=None, **kwargs):

        self._atmo = atmo
        self._mass = mass
        self._age = age
        self._entropy = entropy

        if base_dir is not None:
            self._base_dir = base_dir
        self.sub_dir = self._base_dir  + 'SB.' + self.atmo + '/'

        self._get_file()
        self._read_file()
        self.distance = distance

        self.accr = accr
        if not accr:
            self.mmdot = 0
        elif mmdot is not None:
            self.mmdot = mmdot
        elif mdot is not None:
            self.mmdot = self.mass * mdot # MJup^2/yr
        else:
            mdot = self.mass / (1e6 * self.age) # Assumed MJup/yr
            self.mmdot = self.mass * mdot # MJup^2/yr

        self.rin = accr_rin
        self.truncated = truncated

    def _get_file(self):
        """Find the file closest to the input parameters"""
        files = []; masses = []; ages = []
        for file in os.listdir(self.sub_dir):
            files.append(file)
            fsplit = re.split('[_\.]',file)
            ind_mass = fsplit.index('mass') + 1
            ind_age = fsplit.index('age') + 1
            masses.append(int(fsplit[ind_mass]))
            ages.append(int(fsplit[ind_age]))
        files = np.array(files)
        ages = np.array(ages)
        masses = np.array(masses)

        # Find those indices closest in mass
        mdiff = np.abs(masses - self.mass)
        ind_mass = mdiff == np.min(mdiff)

        # Of those masses, find the closest age
        adiff = np.abs(ages - self.age)
        ind_age = adiff[ind_mass] == np.min(adiff[ind_mass])

        # Get the final file name
        self.file = ((files[ind_mass])[ind_age])[0]

    def _read_file(self):
        """Read in the file data"""
        # Read in the file's content row-by-row (saved as a string)
        with open(self.sub_dir + self.file) as f:
            content = f.readlines()
        content = [x.strip('\n') for x in content]

        # Parse the strings into an array
        #   Row #, Value
        #   1      col 1: age (Myr);
        #          cols 2-601: wavelength (in microns, in range 0.8-15.0)
        #   2-end  col 1: initial S;
        #          cols 2-601: F_nu (in mJy for a source at 10 pc)

        ncol = len(content[0].split())
        nrow = len(content)
        arr = np.zeros([nrow,ncol])
        for i,row in enumerate(content):
            arr[i,:] = np.array(content[i].split(), dtype='float64')

        # Find the closest entropy and save
        entropy = arr[1:,0]
        diff = np.abs(self.entropy - entropy)
        ind = diff == np.min(diff)
        self._flux = arr[1:,1:][ind,:].flatten()
        self._fluxunits = 'mJy'

        # Save the wavelength information
        self._wave = arr[0,1:]
        self._waveunits = 'um'

        # Distance (10 pc)
        self._distance = 10

    @property
    def mdot(self):
        """Accretion rate in MJup/yr"""
        return self.mmdot / self.mass

    @property
    def wave(self):
        """Wavelength of spectrum"""
        return self._wave
    @property
    def waveunits(self):
        """Wavelength units"""
        return self._waveunits

    @property
    def flux(self):
        """Spectral flux"""
        return self._flux
    @property
    def fluxunits(self):
        """Flux units"""
        return self._fluxunits

    @property
    def distance(self):
        """Assumed distance to source (pc)"""
        return self._distance
    @distance.setter
    def distance(self, value):
        self._flux *= (self._distance/value)**2
        self._distance = value

    @property
    def atmo(self):
        """Atmosphere type
        """
        return self._atmo
    @property
    def mass(self):
        """Mass of planet (MJup)"""
        return self._mass
    @property
    def age(self):
        """Age in millions of years"""
        return self._age
    @property
    def entropy(self):
        """Initial entropy (8.0-13.0)"""
        return self._entropy

    def export_pysynphot(self, waveout='angstrom', fluxout='flam'):
        """Output to :mod:`pysynphot.spectrum` object

        Export object settings to a :mod:`pysynphot.spectrum`.

        Parameters
        ----------
        waveout : str
            Wavelength units for output
        fluxout : str
            Flux units for output
        """
        w = self.wave; f = self.flux
        name = (re.split('[\.]', self.file))[0]#[5:]
        sp = S.ArraySpectrum(w, f, name=name, waveunits=self.waveunits, fluxunits=self.fluxunits)

        sp.convert(waveout)
        sp.convert(fluxout)

        if self.accr and (self.mmdot>0):
            sp_mdot = sp_accr(self.mmdot, rin=self.rin,
                              dist=self.distance, truncated=self.truncated,
                              waveout=waveout, fluxout=fluxout)
            # Interpolate accretion spectrum at each wavelength
            # and create new composite spectrum
            fnew = np.interp(sp.wave, sp_mdot.wave, sp_mdot.flux)
            sp_new = S.ArraySpectrum(sp.wave, sp.flux+fnew,
                                     waveunits=waveout, fluxunits=fluxout)
            return sp_new
        else:
            return sp



#class planets_sb11(planets_sb12):
#    """Deprecated class. Use :class:`planets_sb12` instead."""
#    # Turns out the paper is Spiegel & Burrows (2012), not 2011
#    def __init__(self, *args, **kwargs):
#
#        _log.warning('planets_sb11 is depcrecated. Use planets_sb12 instead.')
#        planets_sb12.__init__(self, *args, **kwargs)

def sp_accr(mmdot, rin=2, dist=10, truncated=False,
            waveout='angstrom', fluxout='flam', base_dir=None):

    """Exoplanet accretion flux values (Zhu et al., 2015).

    Calculated the wavelength-dependent flux of an exoplanet accretion disk/shock
    from Zhu et al. (2015). A

    Note
    ----
    This function only uses the table of photometric values to calculate
    photometric brightness from a source, so not very useful for simulating
    spectral observations.


    Parameters
    ----------
    mmdot : float
        Product of the exoplanet mass and mass accretion rate (MJup^2/yr).
        Values range from 1e-7 to 1e-2.
    rin : float
        Inner radius of accretion disk (units of RJup; default: 2).
    dist : float
        Distance to object (pc).
    truncated: bool
        If True, then the values are for a disk with Rout=50 RJup,
        otherwise, values were calculated for a full disk (Rout=1000 RJup).
        Accretion from a "tuncated disk" is due mainly to MRI.
        Luminosities for full and truncated disks are very similar.
    waveout : str
        Wavelength units for output
    fluxout : str
        Flux units for output
    base_dir: str, None
        Location of accretion model sub-directories.
    """

    base_dir = conf.PYNRC_PATH + 'spiegel/' if base_dir is None else base_dir
    fname = base_dir + 'zhu15_accr.txt'

    names = ('MMdot', 'Rin', 'Tmax', 'J', 'H', 'K', 'L', 'M', 'N', 'J2', 'H2', 'K2', 'L2', 'M2', 'N2')
    tbl = ascii.read(fname, guess=True, names=names)

    # Inner radius values and Mdot values
    rin_vals = np.unique(tbl['Rin'])
    mdot_vals = np.unique(tbl['MMdot'])
    nmdot = len(mdot_vals)

    assert (rin >=rin_vals.min())  & (rin <=rin_vals.max()), "rin is out of range"
    assert (mmdot>=mdot_vals.min()) & (mmdot<=mdot_vals.max()), "mmdot is out of range"

    if truncated:
        mag_names = ('J2', 'H2', 'K2', 'L2', 'M2', 'N2')
    else:
        mag_names = ('J', 'H', 'K', 'L', 'M', 'N')
    wcen = np.array([ 1.2,  1.6, 2.2, 3.8, 4.8, 10.0])
    zpt  = np.array([1600, 1020, 657, 252, 163, 39.8])

    mag_arr = np.zeros([6,nmdot])
    for i, mv in enumerate(mdot_vals):
        for j, mag in enumerate(mag_names):
            tbl_sub = tbl[tbl['MMdot']==mv]
            rinvals = tbl_sub['Rin']
            magvals = tbl_sub[mag]

            mag_arr[j,i] = np.interp(rin, rinvals, magvals)

    mag_vals = np.zeros(6)
    for j in range(6):
        xi = 10**(mmdot)
        xp = 10**(mdot_vals)
        yp = 10**(mag_arr[j])
        mag_vals[j] = np.log10(np.interp(xi, xp, yp))

    mag_vals += 5*np.log10(dist/10)
    flux_Jy = 10**(-mag_vals/2.5) * zpt

    sp = S.ArraySpectrum(wcen*1e4, flux_Jy, fluxunits='Jy')
    sp.convert(waveout)
    sp.convert(fluxout)

    return sp


def jupiter_spec(dist=10, waveout='angstrom', fluxout='flam', base_dir=None):
    """Jupiter as an Exoplanet
    
    Read in theoretical Jupiter spectrum from Irwin et al. 2014 and output
    as a :mod:`pysynphot.spectrum`.
    
    Parameters
    ===========
    dist : float
        Distance to Jupiter (pc).
    waveout : str
        Wavelength units for output.
    fluxout : str
        Flux units for output.
    base_dir: str, None
        Location of tabulated file irwin_2014_ref_spectra.txt.
    """

    base_dir = conf.PYNRC_PATH + 'solar_system/' if base_dir is None else base_dir
    fname = base_dir + 'irwin_2014_ref_spectra.txt'

    # Column 1: Wavelength (in microns)
    # Column 2: 100*Ap/Astar (Earth-Sun Primary Transit)
    # Column 3: 100*Ap/Astar (Earth-Mdwarf Primary Transit)
    # Column 4: 100*Ap/Astar (Jupiter-Sun Primary Transit)
    # Column 5: Fp/Astar (Earth-Sun Secondary Eclipse)
    # Column 6: Disc-averaged radiance of Earth (W cm-2 sr-1 micron-1)
    # Column 7: Fp/Astar (Jupiter-Sun Secondary Eclipse)
    # Column 8: Disc-averaged radiance of Jupiter (W cm-2 sr-1 micron-1)
    # Column 9: Solar spectral irradiance spectrum (W micron-1)
    #            (Solar Radius = 695500.0 km)
    # Column 10: Mdwarf spectral irradiance spectrum (W micron-1)
    #            (Mdwarf Radius = 97995.0 km)

    data = ascii.read(fname, data_start=14)

    wspec = data['col1'] * 1e4 # Angstrom
    fspec = data['col8'] * 1e3 # erg s-1 cm^-2 A^-1 sr^-1
    
    # Steradians to square arcsec
    sr_to_asec2 = (3600*180/np.pi)**2
    fspec /= sr_to_asec2       # *** / arcsec^2

    # Angular size of Jupiter at some distance
    RJup_km = 71492.0
    au_to_km = 149597870.7
    # Angular size (arcsec) of Jupiter radius
    RJup_asec = RJup_km / au_to_km / dist
    area = np.pi * RJup_asec**2
    
    # flux in f_lambda
    fspec *= area        # erg s-1 cm^-2 A^-1

    sp = S.ArraySpectrum(wspec, fspec, fluxunits='flam')
    sp.convert(waveout)
    sp.convert(fluxout)
    
    return sp


def linder_table(file=None, **kwargs):
    """Load Linder Model Table

    Function to read in isochrone models from Linder et al. 2019.
    Returns an astropy Table.

    Parameters
    ----------
    age : float
        Age in Myr. If set to None, then an array of ages from the file 
        is used to generate dictionary. If set, chooses the closest age
        supplied in table.
    file : string
        Location and name of COND file. See isochrones stored at
        https://phoenix.ens-lyon.fr/Grids/.
        Default is model.AMES-Cond-2000.M-0.0.JWST.Vega
    """

    # Default file to read and load
    if file is None:
        base_dir = conf.PYNRC_PATH + 'linder/isochrones/'
        file = base_dir + 'BEX_evol_mags_-3_MH_0.00.dat'

    with open(file) as f:
        content = f.readlines()

    content = [x.strip('\n') for x in content]

    cnames = content[2].split(',')
    cnames = [name.split(':')[1] for name in cnames]
    ncol = len(cnames)
    
    content_arr = []
    for line in content[4:]:
        arr = np.array(line.split()).astype(np.float)
        if len(arr)>0: 
            content_arr.append(arr)
    
    content_arr = np.array(content_arr)

    # Convert to Astropy Table
    tbl = Table(rows=content_arr, names=cnames)
    
    return tbl
    
def linder_filter(table, filt, age, dist=None, 
    cond_interp=True, cond_file=None, **kwargs):
    """Linder Mags vs Mass Arrays
    
    Given a Linder table, NIRCam filter, and age, return arrays of MJup 
    and Vega mags. If distance (pc) is provided, then return the apparent 
    magnitude, otherwise absolute magnitude at 10pc.
    
    This function takes the isochrones tables from Linder et al 2019 and
    creates a irregular contour grid of filter magnitude and lag(age)
    where the z-axis is log(mass). This is mapped onto a regular grid
    that is interpolated within the data boundaries and linearly
    extrapolated outside of the region of available data.
    
    Parameters
    ==========
    table : astropy table
        Astropy table output from `linder_table`
    filt : string
        Name of NIRCam filter
    age : float
        Age of planet mass
    """    
    
    def _trim_nan_image(xgrid, ygrid, zgrid):
        """NaN Trimming of Image
    
        Remove rows/cols with NaN's while trying to preserve
        the maximum footprint of real data.
        """
    
        xgrid2, ygrid2, zgrid2 = xgrid, ygrid, zgrid
    
        # Create a mask of NaN'ed values
        nan_mask = np.isnan(zgrid2)
        nrows, ncols = nan_mask.shape
        # Determine number of NaN's along each row and col
        num_nans_cols = nan_mask.sum(axis=0)
        num_nans_rows = nan_mask.sum(axis=1)
    
        # First, crop all rows/cols that are only NaN's
        xind_good = np.where(num_nans_cols < nrows)[0]
        yind_good = np.where(num_nans_rows < ncols)[0]
        # get border limits
        x1, x2 = (xind_good.min(), xind_good.max()+1)
        y1, y2 = (yind_good.min(), yind_good.max()+1)
        # Trim of NaN borders
        xgrid2 = xgrid2[x1:x2]
        ygrid2 = ygrid2[y1:y2]
        zgrid2 = zgrid2[y1:y2,x1:x2]
    
        # Find a optimal rectangule subsection free of NaN's
        # Iterative cropping
        ndiff = 5
        while np.isnan(zgrid2.sum()):
            # Make sure ndiff is not negative
            if ndiff<0:
                break

            npix = zgrid2.size

            # Create a mask of NaN'ed values
            nan_mask = np.isnan(zgrid2)
            nrows, ncols = nan_mask.shape
            # Determine number of NaN's along each row and col
            num_nans_cols = nan_mask.sum(axis=0)
            num_nans_rows = nan_mask.sum(axis=1)

            # Look for any appreciable diff row-to-row/col-to-col
            col_diff = num_nans_cols - np.roll(num_nans_cols,-1) 
            row_diff = num_nans_rows - np.roll(num_nans_rows,-1)
            # For edge wrapping, just use last minus previous
            col_diff[-1] = col_diff[-2]
            row_diff[-1] = row_diff[-2]
        
            # Keep rows/cols composed mostly of real data 
            # and where number of NaN's don't change dramatically
            xind_good = np.where( ( np.abs(col_diff) <= ndiff  ) & 
                                  ( num_nans_cols < 0.5*nrows ) )[0]
            yind_good = np.where( ( np.abs(row_diff) <= ndiff  ) & 
                                  ( num_nans_rows < 0.5*ncols ) )[0]
            # get border limits
            x1, x2 = (xind_good.min(), xind_good.max()+1)
            y1, y2 = (yind_good.min(), yind_good.max()+1)
    
            # Trim of NaN borders
            xgrid2 = xgrid2[x1:x2]
            ygrid2 = ygrid2[y1:y2]
            zgrid2 = zgrid2[y1:y2,x1:x2]
        
            # Check for convergence
            # If we've converged, reduce 
            if npix==zgrid2.size:
                ndiff -= 1
                
        # Last ditch effort in case there are still NaNs
        # If so, remove rows/cols 1 by 1 until no NaNs
        while np.isnan(zgrid2.sum()):
            xgrid2 = xgrid2[1:-1]
            ygrid2 = ygrid2[1:-1]
            zgrid2 = zgrid2[1:-1,1:-1]
            
        return xgrid2, ygrid2, zgrid2

    try:
        x = table[filt]
    except KeyError:
        # In case specific filter doesn't exist, interpolate
        x = []
        cnames = ['SPHEREY','NACOJ', 'NACOH', 'NACOKs', 'NACOLp', 'NACOMp',
                  'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W', 'F560W']
        wvals = np.array([1.04, 1.27, 1.66, 2.20, 3.80, 4.80,
                          1.15, 1.50, 2.00, 2.76, 3.57, 4.41, 5.60])
                          
        # Sort by wavelength 
        isort = np.argsort(wvals)
        cnames = list(np.array(cnames)[isort])
        wvals = wvals[isort]

        # Turn table data into array and interpolate at filter wavelength
        tbl_arr = np.array([table[cn].data for cn in cnames]).transpose()
        bp = read_filter(filt)
        wint = bp.avgwave() / 1e4
        x = np.array([np.interp(wint, wvals, row) for row in tbl_arr])
        
    y = table['log(Age/yr)'].data
    z = table['Mass/Mearth'].data
    zlog = np.log10(z)

    #######################################################
    # Grab COND model data to fill in higher masses
    base_dir = conf.PYNRC_PATH + 'cond_models/'
    if cond_file is None: 
        cond_file = base_dir + 'model.AMES-Cond-2000.M-0.0.JWST.Vega'
        
    npsave_file = cond_file + '.{}.npy'.format(filt)
    
    try:
        mag2, age2, mass2_mjup = np.load(npsave_file)
    except:
        d_tbl2 = cond_table(file=cond_file) # Dictionary of ages
        mass2_mjup = []
        mag2 = []
        age2 = []
        for k in d_tbl2.keys():
            tbl2 = d_tbl2[k]
            mass2_mjup = mass2_mjup + list(tbl2['MJup'].data)
            mag2 = mag2 + list(tbl2[filt+'a'].data)
            age2 = age2 + list(np.ones(len(tbl2))*k)
    
        mass2_mjup = np.array(mass2_mjup)
        mag2 = np.array(mag2)
        age2 = np.array(age2)
    
        mag_age_mass = np.array([mag2,age2,mass2_mjup])
        np.save(npsave_file, mag_age_mass)    

    # Irregular grid
    x2 = mag2
    y2 = np.log10(age2 * 1e6)
    z2 = mass2_mjup * 318
    zlog2 = np.log10(z2)
    

    #######################################################
    
    xlim = np.array([x2.min(),x.max()+5]) 
    ylim = np.array([6,10])  # 10^6 to 10^10 yrs
    dx = (xlim[1] - xlim[0]) / 200
    dy = (ylim[1] - ylim[0]) / 200
    xgrid = np.arange(xlim[0], xlim[1]+dx, dx)
    ygrid = np.arange(ylim[0], ylim[1]+dy, dy)
    X, Y = np.meshgrid(xgrid, ygrid)
    
    zgrid = griddata((x,y), zlog, (X, Y), method='cubic')
    zgrid_cond = griddata((x2,y2), zlog2, (X, Y), method='cubic')

    # There will be NaN's along the border that need to be replaced
    ind_nan = np.isnan(zgrid)
    # First replace with COND grid
    zgrid[ind_nan] = zgrid_cond[ind_nan]
    ind_nan = np.isnan(zgrid)
    
    # Remove rows/cols with NaN's
    xgrid2, ygrid2, zgrid2 = _trim_nan_image(xgrid, ygrid, zgrid)

    # Create regular grid interpolator function for extrapolation at NaN's
    func = RegularGridInterpolator((ygrid2,xgrid2), zgrid2, method='linear',
                                   bounds_error=False, fill_value=None)

    # Fix NaN's in zgrid and rebuild func
    pts = np.array([Y[ind_nan], X[ind_nan]]).transpose()
    zgrid[ind_nan] = func(pts)

    func = RegularGridInterpolator((ygrid,xgrid), zgrid, method='linear',
                                   bounds_error=False, fill_value=None)
    
    # Get mass limits for series of magnitudes at a given age                                
    age_log = np.log10(age*1e6)
    mag_abs_arr = xgrid
    pts = np.array([(age_log,xval) for xval in mag_abs_arr])
    mass_arr = 10**func(pts) / 318.0 # Convert to MJup
    
    mag_app_arr = mag_abs_arr + 5*np.log10(dist/10.0)
    return mass_arr, mag_app_arr
    

def cond_table(age=None, file=None, **kwargs):
    """Load COND Model Table

    Function to read in the COND model tables, which have been formatted
    in a very specific way. Has the option to return a dictionary of
    astropy Tables, where each dictionary element corresponds to
    the specific ages within the COND table. Or, if the age keyword is
    specified, then this function only returns a single astropy table.

    Parameters
    ----------
    age : float
        Age in Myr. If set to None, then an array of ages from the file 
        is used to generate dictionary. If set, chooses the closest age
        supplied in table.
    file : string
        Location and name of COND file. See isochrones stored at
        https://phoenix.ens-lyon.fr/Grids/.
        Default is model.AMES-Cond-2000.M-0.0.JWST.Vega
    """

    def make_table(*args):
        i1, i2 = (ind1[i]+4, ind2[i])

        rows = []
        for line in content[i1:i2]:
            if (line=='') or ('---' in line):
                continue
            else:
                vals = np.array(line.split(), dtype='float64')
                rows.append(tuple(vals))
        tbl = Table(rows=rows, names=cnames)

        # Convert to Jupiter masses
        newcol = tbl['M/Ms'] * 1047.348644
        newcol.name = 'MJup'
        tbl.add_column(newcol, index=1)
        tbl['MJup'].format = '.2f'

        return tbl

    # Default file to read and load
    if file is None:
        base_dir = conf.PYNRC_PATH + 'cond_models/'
        file = base_dir + 'model.AMES-Cond-2000.M-0.0.JWST.Vega'

    with open(file) as f:
        content = f.readlines()

    content = [x.strip('\n') for x in content]

    # Column names
    cnames = content[5].split()
    cnames = ['M/Ms', 'Teff'] + cnames[1:]
    ncol = len(cnames)

    # Create a series of tables for each time
    times_gyr = []
    ind1 = []
    for i, line in enumerate(content):
        if 't (Gyr)' in line:
            times_gyr.append(line.split()[-1])
            ind1.append(i)
    ntimes = len(times_gyr)

    # Create start and stop indices for each age value
    ind2 = ind1[1:] + [len(content)]
    ind1 = np.array(ind1)
    ind2 = np.array(ind2)-1

    # Everything is Gyr, but prefer Myr
    ages_str = np.array(times_gyr)
    ages_gyr = np.array(times_gyr, dtype='float64')
    ages_myr = np.array(ages_gyr * 1000, dtype='int')
    #times = ['{:.0f}'.format(a) for a in ages_myr]

    # Return all tables if no age specified
    if age is None:
        tables = {}
        for i in range(ntimes):
            tbl = make_table(i, ind1, ind2, content)
            tables[ages_myr[i]] = tbl
        return tables
    else:
        # This is faster if we only want one table
        ages_diff = np.abs(ages_myr - age)
        i = np.where(ages_diff==ages_diff.min())[0][0]

        tbl = make_table(i, ind1, ind2, content)
        return tbl

def cond_filter(table, filt, module='A', dist=None, **kwargs):
    """
    Given a COND table and NIRCam filter, return arrays of MJup and Vega mags.
    If distance (pc) is provided, then return the apparent magnitude,
    otherwise absolute magnitude at 10pc.
    """

    mcol = 'MJup'
    fcol = filt + module.lower()

    # Table Data
    mass_data = table[mcol].data
    mag_data  = table[fcol].data

    # Data to interpolate onto
    mass_arr = list(np.arange(0.1,1,0.1)) + list(np.arange(1,10)) \
        + list(np.arange(10,200,10)) + list(np.arange(200,1400,100))
    mass_arr = np.array(mass_arr)

    # Interpolate
    mag_arr = np.interp(mass_arr, mass_data, mag_data)

    # Extrapolate
    cf = jl_poly_fit(np.log(mass_data), mag_data)
    ind_out = (mass_arr < mass_data.min()) | (mass_arr > mass_data.max())
    mag_arr[ind_out] = jl_poly(np.log(mass_arr), cf)[ind_out]

    # Distance modulus for apparent magnitude
    if dist is not None:
        mag_arr = mag_arr + 5*np.log10(dist/10)

    return mass_arr, mag_arr

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

    #print(r, theta)
    return r, theta


def coron_trans(name, module='A', pixscale=None, fov=20, nd_squares=True):
    """
    Build a transmission image of a coronagraphic mask spanning
    the 20" coronagraphic FoV.

    Pulled from WebbPSF
    """

    import scipy.special
    import scipy

    if name=='MASK210R':
        sigma = 5.253
        pixscale = pixscale_SW if pixscale is None else pixscale
    elif name=='MASK335R':
        sigma=3.2927866
        pixscale = pixscale_LW if pixscale is None else pixscale
    elif name=='MASK430R':
        sigma=2.58832
        pixscale = pixscale_LW if pixscale is None else pixscale
    elif name=='MASKSWB':
        pixscale = pixscale_SW if pixscale is None else pixscale
    elif name=='MASKLWB':
        pixscale = pixscale_LW if pixscale is None else pixscale

    #pixscale=0.03

    s = int(fov/pixscale + 0.5)
    shape = (s,s)
    y, x = np.indices(shape, dtype=float)
    y -= shape[0] / 2.0
    x -= shape[1] / 2.0
    y,x = (pixscale * y, pixscale * x)

    if 'WB' in name: # Wedge Masks
        scalefact = (2 + (-x + 7.5) * 4 / 15).clip(2, 6)
        if name == 'MASKSWB':
            polyfitcoeffs = np.array([2.01210737e-04, -7.18758337e-03, 1.12381516e-01,
                                      -1.00877701e+00, 5.72538509e+00, -2.12943497e+01,
                                      5.18745152e+01, -7.97815606e+01, 7.02728734e+01])
            scalefact = scalefact[:, ::-1] # flip orientation left/right for SWB mask
        elif name == 'MASKLWB':
            polyfitcoeffs = np.array([9.16195583e-05, -3.27354831e-03, 5.11960734e-02,
                                      -4.59674047e-01, 2.60963397e+00, -9.70881273e+00,
                                      2.36585911e+01, -3.63978587e+01, 3.20703511e+01])

        sigmas = scipy.poly1d(polyfitcoeffs)(scalefact)
        sigmar = sigmas * np.abs(y)
        # clip sigma: The minimum is to avoid divide by zero
        #             the maximum truncates after the first sidelobe to match the hardware
        sigmar.clip(min=np.finfo(sigmar.dtype).tiny, max=2*np.pi, out=sigmar)
        transmission = (1 - (np.sin(sigmar) / sigmar) ** 2)
        #transmission[x==0] = 0
        woutside = np.where(np.abs(x) > 10)
        transmission[woutside] = 1.0

    else: # Circular Masks
        r = np.sqrt(x ** 2 + y ** 2)
        sigmar = sigma * r
        #sigmar.clip(np.finfo(sigmar.dtype).tiny, 2*np.pi, out=sigmar)  # avoid divide by zero -> NaNs
        sigmar.clip(np.finfo(sigmar.dtype).tiny, 7.1559, out=sigmar)  # avoid divide by zero -> NaNs
        transmission = (1 - (2 * scipy.special.jn(1, sigmar) / sigmar) ** 2)
        transmission[r==0] = 0   # special case center point (value based on L'Hopital's rule)

    if nd_squares:
        # add in the ND squares. Note the positions are not exactly the same in the two wedges.
        # See the figures  in Krist et al. of how the 6 ND squares are spaced among the 5
        # corongraph regions
        # Note: 180 deg rotation needed relative to Krist's figures for the flight SCI orientation:
        # We flip the signs of X and Y here as a shortcut to avoid recoding all of the below...
        x *= -1
        y *= -1
        #x = x[::-1, ::-1]
        #y = y[::-1, ::-1]
        if ((module=='A' and name=='MASKLWB') or
            (module=='B' and name=='MASK210R')):
            wnd_5 = np.where(
                ((y > 5)&(y<10)) &
                (
                    ((x < -5) & (x > -10)) |
                    ((x > 7.5) & (x < 12.5))
                )
            )
            wnd_2 = np.where(
                ((y > -10)&(y<-8)) &
                (
                    ((x < -8) & (x > -10)) |
                    ((x > 9) & (x < 11))
                )
            )
        elif ((module=='A' and name=='MASK210R') or
              (module=='B' and name=='MASKSWB')):
            wnd_5 = np.where(
                ((y > 5)&(y<10)) &
                (
                    ((x > -12.5) & (x < -7.5)) |
                    ((x > 5) & (x <10))
                )
            )
            wnd_2 = np.where(
                ((y > -10)&(y<-8)) &
                (
                    ((x > -11) & (x < -9)) |
                    ((x > 8) & (x<10))
                )
            )
        else:
            wnd_5 = np.where(
                ((y > 5)&(y<10)) &
                (np.abs(x) > 7.5) &
                (np.abs(x) < 12.5)
            )
            wnd_2 = np.where(
                ((y > -10)&(y<-8)) &
                (np.abs(x) > 9) &
                (np.abs(x) < 11)
            )


        transmission[wnd_5] = np.sqrt(1e-3)
        transmission[wnd_2] = np.sqrt(1e-3)

        # Add in the opaque border of the coronagraph mask holder.
        if ((module=='A' and name=='MASKLWB') or
            (module=='B' and name=='MASK210R')):
            # left edge
            woutside = np.where((x < -10) & (y < 11.5 ))
            transmission[woutside] = 0.0
        elif ((module=='A' and name=='MASK210R') or
              (module=='B' and name=='MASKSWB')):
            # right edge
            woutside = np.where((x > 10) & (y < 11.5))
            transmission[woutside] = 0.0
        # mask holder edge
        woutside = np.where(y < -10)
        transmission[woutside] = 0.0

        # edge of mask itself
        # TODO the mask edge is complex and partially opaque based on CV3 images?
        # edge of glass plate rather than opaque mask I believe. To do later.
        # The following is just a temporary placeholder with no quantitative accuracy.
        # but this is outside the coronagraph FOV so that's fine - this only would matter in
        # modeling atypical/nonstandard calibration exposures.

        wedge = np.where(( y > 11.5) & (y < 13))
        transmission[wedge] = 0.7

    if not np.isfinite(transmission.sum()):
        _log.warn("There are NaNs in the BLC mask - correcting to zero. (DEBUG LATER?)")
        transmission[np.where(np.isfinite(transmission) == False)] = 0

    return transmission


def build_mask(module='A', pixscale=0.03):
    """Create coronagraphic mask image

    Return a truncated image of the full coronagraphic mask layout
    for a given module.

    +V3 is up, and +V2 is to the left.
    """
    if module=='A':
        names = ['MASK210R', 'MASK335R', 'MASK430R', 'MASKSWB', 'MASKLWB']
    elif module=='B':
        names = ['MASKSWB', 'MASKLWB', 'MASK430R', 'MASK335R', 'MASK210R']
    allims = [coron_trans(name,module,pixscale) for name in names]

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
        Which Lyot pupil stop is being used?
        If None, then defaults based on ref_mask.
    """

    # from jwxml import siaf

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
        raise ValueError("Invalid detid: {0} \n\tValid names are: {1},\n\t{2}" \
              .format(detid, ', '.join(names)))

    # These detectors don't see any of the mask structure
    names_ret0 = ['A1', 'A3', 'B2', 'B4']
    if detid in names_ret0:
        return None

    pixscale = pixscale_LW if '5' in detid else pixscale_SW
    pixscale_over = pixscale / oversample

    # Build the full mask
    xpix = ypix = 2048
    xpix_over = int(xpix * oversample)
    ypix_over = int(ypix * oversample)

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
    allims = [coron_trans(cname, module, pixscale_over) for cname in cnames]
    
    if pupil is None:
        pupil = 'WEDGELYOT' if 'WB' in ref_mask else 'CIRCLYOT'

    channel = 'LW' if '5' in detid else 'SW'
    cdict = coron_ap_locs(module, channel, ref_mask, pupil=pupil, full=False)
    xdet, ydet = cdict['cen']

    # Add an offset value before expanding to full size
    cmask = np.concatenate(allims, axis=1) + 999

    # A5 mask names need to be reversed for detector orientation
    # along horizontal direction
    if detid=='A5':
        cnames = cnames[::-1]
    xf_arr = np.arange(1,2*len(cnames)+1,2) / (2*len(cnames))
    xf = xf_arr[np.array(cnames)==ref_mask][0]
    xc = cmask.shape[1] * xf
    xc += (ypix_over - cmask.shape[1]) / 2
    yc = xpix_over / 2

    # Cut to final image size
    cmask = pad_or_cut_to_size(cmask, (ypix_over,xpix_over))

    # Place cmask in detector coords
    cmask = sci_to_det(cmask, detid)

    # Shift cmask to appropriate location
    # ie., move MASK430R from center
    xdet_over, ydet_over = np.array([xdet,ydet]) * oversample
    delx = xdet_over - xc
    dely = ydet_over - yc

    #print((xdet_over, ydet_over), (xc, yc), (delx, dely))

    cmask = fshift(cmask, int(delx), int(dely), pad=True) + 1
    cmask[cmask>10] = cmask[cmask>10] - 1000

    # Place blocked region from coronagraph holder
    if detid=='A2':
        if 'CIRCLYOT' in pupil:
            i1, i2 = [int(920*oversample), int(360*oversample)]
            cmask[0:i1,0:i2]=0
            i1 = int(220*oversample)
            cmask[0:i1,:] = 0
        else:
            i1, i2 = [int(935*oversample), int(360*oversample)]
            cmask[0:i1,0:i2]=0
            i1 = int(235*oversample)
            cmask[0:i1,:] = 0
            
    elif detid=='A4':
        if 'CIRCLYOT' in pupil:
            i1, i2 = [int(920*oversample), int(1490*oversample)]
            cmask[0:i1,i2:]=0
            i1 = int(220*oversample)
            cmask[0:i1,:] = 0
        else:
            i1, i2 = [int(935*oversample), int(1490*oversample)]
            cmask[0:i1,i2:]=0
            i1 = int(235*oversample)
            cmask[0:i1,:] = 0
            
    elif detid=='A5':
        if 'CIRCLYOT' in pupil:
            i1, i2 = [int(1480*oversample), int(260*oversample)]
            cmask[i1:,0:i2]=0
            i1, i2 = [int(1480*oversample), int(1890*oversample)]
            cmask[i1:,i2:]=0
            i1 = int(1825*oversample)
            cmask[i1:,:] = 0
        else:
            i1, i2 = [int(1485*oversample), int(265*oversample)]
            cmask[i1:,0:i2]=0
            i1, i2 = [int(1485*oversample), int(1895*oversample)]
            cmask[i1:,i2:]=0
            i1 = int(1830*oversample)
            cmask[i1:,:] = 0
            
    elif detid=='B1':
        if 'CIRCLYOT' in pupil:
            i1, i2 = [int(910*oversample), int(1635*oversample)]
            cmask[0:i1,i2:]=0
            i1 = int(210*oversample)
            cmask[0:i1,:] = 0
        else:
            i1, i2 = [int(905*oversample), int(1630*oversample)]
            cmask[0:i1,i2:]=0
            i1 = int(205*oversample)
            cmask[0:i1,:] = 0

    elif detid=='B3':
        if 'CIRCLYOT' in pupil:
            i1, i2 = [int(920*oversample), int(500*oversample)]
            cmask[0:i1,0:i2]=0
            i1 = int(210*oversample)
            cmask[0:i1,:] = 0
        else:
            i1, i2 = [int(920*oversample), int(500*oversample)]
            cmask[0:i1,0:i2]=0
            i1 = int(210*oversample)
            cmask[0:i1,:] = 0
    elif detid=='B5':
        if 'CIRCLYOT' in pupil:
            i1, i2 = [int(560*oversample), int(185*oversample)]
            cmask[0:i1,0:i2]=0
            i1, i2 = [int(550*oversample), int(1830*oversample)]
            cmask[0:i1,i2:]=0
            i1 = int(215*oversample)
            cmask[0:i1,:] = 0
        else:
            i1, i2 = [int(560*oversample), int(190*oversample)]
            cmask[0:i1,0:i2]=0
            i1, i2 = [int(550*oversample), int(1835*oversample)]
            cmask[0:i1,i2:]=0
            i1 = int(215*oversample)
            cmask[0:i1,:] = 0

    # Convert back to 'sci' orientation
    cmask = det_to_sci(cmask, detid)

    return cmask


def coron_ap_locs(module, channel, mask, pupil=None, full=False):
    """Coronagraph mask aperture locations and sizes

    Returns a dictionary of the detector aperture sizes
    and locations. Attributes `cen` and `loc` are in terms
    of (x,y) detector pixels.
    """
    
    if pupil is None:
        pupil = 'WEDGELYOT' if 'WB' in mask else 'CIRCLYOT'

    if module=='A':
        if channel=='SW':
            if '210R' in mask:
                cdict_rnd = {'det':'A2', 'cen':(712,526), 'size':640}
                cdict_bar = {'det':'A2', 'cen':(716,538), 'size':640}
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
                cdict_rnd = {'det':'A5', 'cen':(1720, 1670), 'size':320}
                cdict_bar = {'det':'A5', 'cen':(1725, 1681), 'size':320}
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
                cdict_rnd = {'det':'B1', 'cen':(1293,515), 'size':640}
                cdict_bar = {'det':'B1', 'cen':(1287,509), 'size':640}
            elif '335R' in mask:
                cdict_rnd = {'det':'B1', 'cen':(637,513), 'size':640}
                cdict_bar = {'det':'B1', 'cen':(632,508), 'size':640}
            elif '430R' in mask:
                cdict_rnd = {'det':'B1', 'cen':(-20,513), 'size':640}
                cdict_bar = {'det':'B1', 'cen':(-25,508), 'size':640}
            elif 'SWB' in mask:
                cdict_rnd = {'det':'B3', 'cen':(874,519), 'size':640}
                cdict_bar = {'det':'B3', 'cen':(870,518), 'size':640}
            elif 'LWB' in mask:
                cdict_rnd = {'det':'B3', 'cen':(1532,519), 'size':640}
                cdict_bar = {'det':'B3', 'cen':(1526,510), 'size':640}
            else:
                raise ValueError('Mask {} not recognized for {} channel'\
                                 .format(mask, channel))
        elif channel=='LW':
            if '210R' in mask:
                cdict_rnd = {'det':'B5', 'cen':(1656,359), 'size':320}
                cdict_bar = {'det':'B5', 'cen':(1660,359), 'size':320}
            elif '335R' in mask:
                cdict_rnd = {'det':'B5', 'cen':(1334,360), 'size':320}
                cdict_bar = {'det':'B5', 'cen':(1338,360), 'size':320}
            elif '430R' in mask:
                cdict_rnd = {'det':'B5', 'cen':(1012,362), 'size':320}
                cdict_bar = {'det':'B5', 'cen':(1015,361), 'size':320}
            elif 'SWB' in mask:
                cdict_rnd = {'det':'B5', 'cen':(366,364), 'size':320}
                cdict_bar = {'det':'B5', 'cen':(370,364), 'size':320}
            elif 'LWB' in mask:
                cdict_rnd = {'det':'B5', 'cen':(689,363), 'size':320}
                cdict_bar = {'det':'B5', 'cen':(693,364), 'size':320}
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


###########################################################################
#
#    Miscellaneous
#
###########################################################################

def nrc_header(det_class, filter=None, pupil=None, obs_time=None, header=None,
               DMS=True, targ_name=None):
    """Simulated header

    Create a generic NIRCam FITS header from a detector_ops class.

    Parameters
    ----------
    filter : str
        Name of filter element.
    pupil : str
        Name of pupil element.
    DMS : bool
        Make the header in a format used by Data Management Systems.
    obs_time : datetime
        Specifies when the observation was considered to be executed.
        If not specified, then it will choose the current time.
        This must be a datetime object:

            >>> datetime.datetime(2016, 5, 9, 11, 57, 5, 796686)

    header : obj
        Can pass an existing header that will be updated.
        This has not been fully tested.
    targ_name : str
        Standard astronomical catalog name for a target.
        Otherwise, it will be UNKNOWN.
    """

    from .version import __version__

    filter = 'UNKNOWN' if filter is None else filter
    pupil  = 'UNKNOWN' if pupil  is None else pupil
    targ_name = 'UNKNOWN' if targ_name is None else targ_name

    d = det_class
    # MULTIACCUM ramp information
    ma = d.multiaccum

    # How many axes?
    naxis = 2 if ma.ngroup == 1 else 3
    if naxis == 3:
        naxis3 = ma.ngroup
    naxis1 = d.xpix
    naxis2 = d.ypix

    # Select Detector ID based on SCA ID
    detector = d.detname

    # Are we in subarray?
    sub_bool = True if d.wind_mode != 'FULL' else False
    # Horizontal window mode?
    hwinmode = 'ENABLE' if d.wind_mode=='WINDOW' else 'DISABLE'

    # Window indices (0-indexed)
    x1 = d.x0; x2 = x1 + d.xpix
    y1 = d.y0; y2 = y1 + d.ypix

    # Ref pixel info
    ref_all = d.ref_info

    # Dates and times
    obs_time = datetime.datetime.utcnow() if obs_time is None else obs_time
    # Total time to complete obs = (ramp_time+reset_time)*nramps
    # ramp_time does not include reset frames!!
    tdel = ma.nint * (d.time_int + d.time_frame) + d._exp_delay
    dtstart = obs_time.isoformat()
    aTstart = Time(dtstart)
    dtend = (obs_time + datetime.timedelta(seconds=tdel)).isoformat()
    aTend = Time(dtend)
    dstart = dtstart[:10]; dend = dtend[:10]
    tstart = dtstart[11:-3]; tend = dtend[11:-3]
    tsample = 1e6/d._pixel_rate

    ################################################################
    # Create blank header
    hdr_update = False  if header is None else True
    hdr = fits.Header() if header is None else header

    # Add in basic header info
    hdr['SIMPLE']  = (True,   'conforms to FITS standard')
    hdr['BITPIX']  = (16,     'array data type')
    if DMS == True:
        hdr['SUBSTRT1'] = (x1+1, 'Starting pixel in axis 1 direction')
        hdr['SUBSTRT2'] = (y1+1, 'Starting pixel in axis 2 direction')
        hdr['SUBSIZE1'] = naxis1
        hdr['SUBSIZE2'] = naxis2
        hdr['NAXIS'] = (naxis,  'number of array dimensions')
    else:
        hdr['NAXIS']   = (naxis,  'number of array dimensions')
        hdr['NAXIS1']  = naxis1
        hdr['NAXIS2']  = naxis2

        if hdr_update: hdr.pop('NAXIS3', None)
        if naxis == 3: hdr['NAXIS3']  = (naxis3, 'length of third data axis')
    hdr['EXTEND']  = True

    hdr['DATE']    = ('',   'date file created (yyyy-mm-ddThh:mm:ss,UTC)')
    hdr['BSCALE']  = (1,     'scale factor for array value to physical value')
    hdr['BZERO']   = (32768, 'physical value for an array value of zero')
    hdr['UNITS']   = ('',  'Units for the data type (ADU, e-, etc.)')
    hdr['ORIGIN']  = ('UAz',  'institution responsible for creating FITS file')
    hdr['FILENAME']= ('',   'name of file')
    hdr['FILETYPE']= ('raw', 'type of data found in data file')

    # Observation Description
    hdr['TELESCOP']= ('JWST',    'telescope used to acquire data')
    hdr['INSTRUME']= ('NIRCAM',  'instrument identifier used to acquire data')
    hdr['OBSERVER']= ('UNKNOWN', 'person responsible for acquiring data')
    hdr['DATE-OBS']= (dstart, 'UT date of observation (yyyy-mm-dd)')
    hdr['TIME-OBS']= (tstart, 'Approximate UT time of start of observation (hh:mm:ss.sss)')
    if DMS == True:
        if 'GRISM' in pupil:
            exp_type = 'NRC_GRISM'
        elif pupil == None:
            exp_type = 'UNKNOWN'
        else:
            exp_type = 'NRC_IMAGE'
        hdr['EXP_TYPE'] = (exp_type,'Type of data in the exposure')
    hdr['DATE-END']= (dend,   'UT date of end of observation(yyyy-mm-dd)')
    hdr['TIME-END']= (tend,   'UT time of end of observation (hh:mm:ss.sss)')
    hdr['SCA_ID']  = (d.scaid,   'Unique SCA identification in ISIM')
    hdr['DETECTOR']= (d.detname, 'ASCII Mnemonic corresponding to the SCA_ID')
    hdr['PIXELSCL']= (d.pixelscale, 'Detector Pixel Scale (arcsec/pixel)')

    nx_noref = naxis1 - ref_all[2] - ref_all[3]
    ny_noref = naxis2 - ref_all[0] - ref_all[1]
    fovx = nx_noref * d.pixelscale
    fovy = ny_noref * d.pixelscale
    hdr['FOV']     = ('{:.2f}x{:.2f}'.format(fovx,fovy), 'Field of view in arcsec')

    if DMS == True:
        hdr['TARG_RA']=  (0.0, 'Target RA at mid time of exposure') #arbitrary position
        hdr['TARG_DEC']= (0.0, 'Target Dec at mid time of exposure') #arbitrary position

        hdr['PROGRAM'] = ('12345', 'Program number')
        hdr['OBSERVTN']= ('001',   'Observation number')
        hdr['VISIT']   = ('001',   'Visit Number')
        hdr['VISITGRP']= ('01',  'Visit Group Identifier')

        hdr['SEQ_ID']  = ('1', 'Parallel sequence identifier')
        hdr['ACT_ID']  = ('1', 'Activity identifier')
        hdr['EXPOSURE']= ('1', 'Exposure request number')
        hdr['OBSLABEL']= ('Target 1 NIRCam Observation 1', 'Proposer label for the observation')
        hdr['EXPSTART']= (aTstart.mjd, 'UTC exposure start time')
        hdr['EXPEND']  = (aTend.mjd, 'UTC exposure end time')
        hdr['EFFEXPTM']= (d.time_int_eff*d.time_int, 'Effective exposure time (sec)')
        hdr['NUMDTHPT']= ('1','Total number of points in pattern')
        hdr['PATT_NUM']= (1,'Position number in primary pattern')

    hdr['TARGNAME'] = (targ_name, 'Standard astronomical catalog name for target')
    hdr['OBSMODE'] = ('UNKNOWN', 'Observation mode')

    if DMS == True:
        if d.channel == 'LW':
            headerChannel = 'LONG'
        elif d.channel == 'SW':
            headerChannel = 'SHORT'
        else:
            headerChannel = 'UNKNOWN'
        hdr['CHANNEL'] = headerChannel

        hdr['GRATING'] = ('N/A - NIRCam', 'Name of the grating element used')
        hdr['BAND']    = ('N/A - NIRCam', 'MRS wavelength band')
        hdr['LAMP']    = ('N/A - NIRCam', 'Internal lamp state')
        hdr['GWA_XTIL']= ('N/A - NIRCam', 'Grating X tilt angle relative to mirror')
        hdr['GWA_YTIL']= ('N/A - NIRCam', 'Grating Y tilt angle relative to mirror')
        hdr['GWA_TILT']= ('N/A - NIRCam', 'GWA TILT (avg/calib) temperature (K)')
        hdr['MSAMETFL']= ('N/A - NIRCam', 'MSA metadata file name')
        hdr['MSAMETID']= ('N/A - NIRCam', 'MSA metadata ID')

    # Positions of optical elements
    hdr['FILTER']  = (filter, 'Module ' + d.module + ' ' + d.channel + ' FW element')
    hdr['PUPIL']   = (pupil, 'Module ' + d.module + ' ' + d.channel + ' PW element')
    hdr['PILSTATE']= ('RETRACTED', 'Module ' + d.module + ' PIL deploy state')

    # Readout Mode
    hdr['NSAMPLE'] = (1,            'A/D samples per read of a pixel')
    if DMS == True:
        frmName = 'NFRAMES'
        grpName = 'NGROUPS'
        intName = 'NINTS'
    else:
        frmName = 'NFRAME'
        grpName = 'NGROUP'
        intName = 'NINT'
    hdr[frmName]   = (ma.nf,         'Number of frames in group')
    hdr[grpName]   = (ma.ngroup,     'Number groups in an integration')
    hdr[intName]   = (ma.nint,     'Number of integrations in an exposure')

    # Timing information
    hdr['TSAMPLE'] = (tsample,        'Delta time between samples in microsec')
    hdr['TFRAME']  = (d.time_frame,   'Time in seconds between frames')
    hdr['TGROUP']  = (d.time_group,   'Delta time between groups')
    hdr['DRPFRMS1']= (ma.nd1, 'Number of frame skipped prior to first integration')
    hdr['GROUPGAP']= (ma.nd2, 'Number of frames skipped')
    hdr['DRPFRMS3']= (ma.nd3, 'Number of frames skipped between integrations')
    hdr['FRMDIVSR']= (ma.nf,  'Divisor applied to each group image')
    hdr['INTAVG']  = (1, 'Number of integrations averaged in one image')
    hdr['NRESETS1']= (ma.nr1, 'Number of reset frames prior to first integration')
    hdr['NRESETS2']= (ma.nr2, 'Number of reset frames between each integration')
    hdr['INTTIME'] = (d.time_int, 'Total integration time for one MULTIACCUM')
    hdr['EXPTIME'] = (d.time_exp, 'Exposure duration (seconds) calculated')
    
    # Subarray names
    if DMS == True:
        if (d.xpix == 2048) & (d.ypix == 2048):
            subName = 'FULL'
        elif (d.xpix == 640) & (d.ypix == 640):
            subName = 'SUB640'
        elif (d.xpix == 320) & (d.ypix == 320):
            subName = 'SUB320'
        elif (d.xpix == 400) & (d.ypix == 400):
            subName = 'SUB400P'
        elif (d.xpix == 64) & (d.ypix == 64):
            subName = 'SUB64P'
        elif (d.xpix == 2048) & (d.ypix == 256):
            subName = 'SUBGRISM256'
        elif (d.xpix == 2048) & (d.ypix == 128):
            subName = 'SUBGRISM128'
        elif (d.xpix == 2048) & (d.ypix == 64):
            subName = 'SUBGRISM64'
        else:
            subName = 'UNKNOWN'
        hdr['SUBARRAY']= (subName, 'Detector subarray string')
    else:
        hdr['SUBARRAY']= (sub_bool, 'T if subarray used, F if not')
        hdr['HWINMODE']= (hwinmode, 'If enabled, single output mode used')

    # Readout Patterns
    if DMS == True:
        hdr['READPATT']= (ma.read_mode, 'Readout pattern name')
        hdr['ZROFRAME']= (True,       'T if zeroth frame present, F if not')
    else:
        hdr['READOUT'] = (ma.read_mode, 'Readout pattern name')
        hdr['ZROFRAME']= (False,       'T if zeroth frame present, F if not')

    #Reference Data
    hdr['TREFROW'] = (ref_all[1], 'top reference pixel rows')
    hdr['BREFROW'] = (ref_all[0], 'bottom reference pixel rows')
    hdr['LREFCOL'] = (ref_all[2], 'left col reference pixels')
    hdr['RREFCOL'] = (ref_all[3], 'right col reference pixels')
    hdr['NREFIMG'] = (0, 'number of reference rows added to end')
    hdr['NXREFIMG']= (0, 'reference image columns')
    hdr['NYREFIMG']= (0, 'reference image rows')
    hdr['COLCORNR']= (x1+1, 'The Starting Column for ' + detector)
    hdr['ROWCORNR']= (y1+1, 'The Starting Row for ' + detector)

    hdr.insert('EXTEND', '', after=True)
    hdr.insert('EXTEND', '', after=True)
    hdr.insert('EXTEND', '', after=True)

    hdr.insert('FILETYPE', '', after=True)
    hdr.insert('FILETYPE', ('','Observation Description'), after=True)
    hdr.insert('FILETYPE', '', after=True)

    hdr.insert('OBSMODE', '', after=True)
    hdr.insert('OBSMODE', ('','Optical Mechanisms'), after=True)
    hdr.insert('OBSMODE', '', after=True)

    hdr.insert('PILSTATE', '', after=True)
    hdr.insert('PILSTATE', ('','Readout Mode'), after=True)
    hdr.insert('PILSTATE', '', after=True)

    hdr.insert('ZROFRAME', '', after=True)
    hdr.insert('ZROFRAME', ('','Reference Data'), after=True)
    hdr.insert('ZROFRAME', '', after=True)

    hdr.insert('ROWCORNR', '', after=True)
    hdr.insert('ROWCORNR', '', after=True)

    hdr['comment'] = 'Simulated data generated by {} v{}'\
                      .format(__package__,__version__)

    return hdr

def config2(input, intype='int'):
    """NIRCam CONFIG2 (0x4011) Register

    Return a dictionary of configuration parameters depending on the
    value of CONFIG2 register (4011).

    Parameters
    ----------
    input : int, str
        Value of CONFIG2, nominally as an int. Binary and Hex values
        can also be passed as strings.
    intype: str
        Input type (int, hex, or bin) for integer, hex, string,
        or binary string.

    """
    if 'hex' in intype:
        if '0x' in input:
            input = int(input, 0)
        else:
            input = int(input, 16)
    if 'bin' in intype:
        if '0b' in input:
            input = int(input, 0)
        else:
            input = int(input, 2)

    # Convert to 16-bit binary string
    input = "{0:016b}".format(input)

    # Config2 Bits (Right to Left)
    # ----------------------------
    # 0 : Vertical Enable
    # 1 : Horizontal Enable
    # 2 : Global reset per integration
    # 3 : Enable Fast row-by-row reset (only in window/stripe)
    # 6-4 : Number of fast row resets per int
    # 7 : Window mode in Idle when window enabled?
    # 8 : 0 = Preamp reset per frame; 1 = reset per row
    # 9 : Permanent Reset
    # 10 : Single step mode
    # 11 : Test pattern
    # 12 : FGS window mode
    # 13 : Power down preamp, adc, and ap during Idle
    # 14 : Power down preamp, adc, and ap during Drop
    # 15 : 0 = Preamp reset per frame; 1 = reset per integration

    # NFF Rows Reset
    # --------------
    # 000 = 1
    # 001 = 4
    # 010 = 16
    # 011 = 64
    # 100 = 256
    # 101 = 512
    # 110 = 1024
    # 111 = 2048

    nff_dict = {'000':   1, '001':   4, '010':  16, '011':  64,
                '100': 256, '101': 512, '110':1024, '111':2048}

    # Reverse for easier indexing of single values
    input2 = input[::-1]

    d = {}
    d['00_window_vert']  = True if bool(int(input2[0])) else False
    d['01_window_horz']  = True if bool(int(input2[1])) else False
    d['02_global_reset'] = True if bool(int(input2[2])) else False
    d['03_rows_reset']   = True if bool(int(input2[3])) else False
    d['04_rows_nff']     = nff_dict.get(input2[4:7][::-1])
    d['07_idle_window']  = True if bool(int(input2[7])) else False
    d['08_pa_reset']     = 'row' if bool(int(input2[8])) else 'frame'
    d['09_perm_reset']   = True if bool(int(input2[9])) else False
    d['10_single_step']  = True if bool(int(input2[10])) else False
    d['11_test_patt']    = True if bool(int(input2[11])) else False
    d['12_fgs_wind']     = True if bool(int(input2[12])) else False
    d['13_power_idl']    = True if bool(int(input2[13])) else False
    d['14_power_drop']   = True if bool(int(input2[14])) else False
    d['15_pa_reset']     = 'int' if bool(int(input2[15])) else 'frame'

    return d

def create_detops(header, DMS=False, read_mode=None, nint=None, ngroup=None,
    detector=None, wind_mode=None, xpix=None, ypix=None, x0=None, y0=None,
    nff=None):
    """Detector class from header

    Create a detector class based on header settings.
    Can override settings with a variety of keyword arguments.

    Parameters
    ----------
    header : obj
        Header from NIRCam FITS file
    DMS : bool
        Is header format from Data Management Systems? Otherwises, ISIM-like.

    Keyword Args
    ------------
    read_mode : str
        NIRCam Ramp Readout mode such as 'RAPID', 'BRIGHT1', etc.
    nint : int
        Number of integrations (ramps).
    ngroup : int
        Number of groups in a integration.
    detector : int, str
        NIRCam detector ID (481-490) or SCA ID (A1-B5).
    wind_mode : str
        Window mode type 'FULL', 'STRIPE', 'WINDOW'.
    xpix : int
        Size of window in x-pixels for frame time calculation.
    ypix : int
        Size of window in y-pixels for frame time calculation.
    x0 : int
        Lower-left x-coord position of detector window.
    y0 : int
        Lower-left y-coord position of detector window.
    nff : int
        Number of fast row resets.

    """
    # Create detector class
    from pynrc.pynrc_core import DetectorOps

    # Detector ID
    detector = header['SCA_ID'] if detector is None else detector

    # Detector size
    xpix = header['SUBSIZE1'] if DMS else header['NAXIS1'] if xpix is None else xpix
    ypix = header['SUBSIZE2'] if DMS else header['NAXIS2'] if ypix is None else ypix

    # Subarray position
    # Headers are 1-indexed, while detector class is 0-indexed
    if x0 is None:
        x1 = header['SUBSTRT1'] if DMS else header['COLCORNR']
        x0 = x1 - 1
    if y0 is None:
        y1 = header['SUBSTRT2'] if DMS else header['ROWCORNR']
        y0 = y1 - 1

    # Subarray setting, Full, Stripe, or Window
#     if wind_mode is None:
#         if DMS:
#             if 'FULL' in header['SUBARRAY']:
#                 wind_mode = 'FULL'
#             elif 'GRISM' in header['SUBARRAY']:
#                 wind_mode = 'STRIPE'
#             else:
#                 wind_mode = 'WINDOW'
#         else:
#             if not header['SUBARRAY']:
#                 wind_mode = 'FULL'
#             elif 'DISABLE' in header['HWINMODE']:
#                 wind_mode = 'STRIPE'
#             else:
#                 wind_mode = 'WINDOW'

    # Subarray setting: Full, Stripe, or Window
    if wind_mode is None:
        if DMS and ('FULL' in header['SUBARRAY']):
            wind_mode = 'FULL'
        elif (not DMS) and (not header['SUBARRAY']):
            wind_mode = 'FULL'
        else:
            # Test if STRIPE or WINDOW
            det_stripe = DetectorOps(detector, 'STRIPE', xpix, ypix, x0, y0)
            det_window = DetectorOps(detector, 'WINDOW', xpix, ypix, x0, y0)
            dt_stripe = np.abs(header['TFRAME'] - det_stripe.time_frame)
            dt_window = np.abs(header['TFRAME'] - det_window.time_frame)
            wind_mode = 'STRIPE' if dt_stripe<dt_window else 'WINDOW'


    # Add MultiAccum info
    if DMS: hnames = ['READPATT', 'NINTS', 'NGROUPS']
    else:   hnames = ['READOUT',  'NINT',  'NGROUP']

    read_mode = header[hnames[0]] if read_mode is None else read_mode
    nint      = header[hnames[1]] if nint      is None else nint
    ngroup    = header[hnames[2]] if ngroup    is None else ngroup

    ma_args = {'read_mode':read_mode, 'nint':nint, 'ngroup':ngroup}

    return DetectorOps(detector, wind_mode, xpix, ypix, x0, y0, nff, **ma_args)


