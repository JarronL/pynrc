"""pyNRC utility functions"""

from __future__ import absolute_import, division, print_function, unicode_literals

# The six library is useful for Python 2 and 3 compatibility
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

# OPD info
from .opds import opd_default, OPDFile_to_HDUList

###########################################################################
#
#    Logging info
#
###########################################################################

import logging
_log = logging.getLogger('pynrc')

try:
    import webbpsf
except ImportError:
    raise ImportError('WebbPSF is not installed. pyNRC depends on its inclusion.')

# Some useful functions for displaying and measuring PSFs
import poppy
from poppy import (radial_profile, measure_radial, measure_fwhm, measure_ee)
from poppy import (measure_sharpness, measure_centroid, measure_strehl)
#from poppy import (display_PSF, display_PSF_difference, display_EE, display_profiles, radial_profile,
#        measure_EE, measure_radial, measure_fwhm, measure_sharpness, measure_centroid, measure_strehl,
#        specFromSpectralType, fwcentroid)

import pysynphot as S
# Extend default wavelength range to 5.6 um
S.refs.set_default_waveset(minwave=500, maxwave=56000, num=10000.0, delta=None, log=False)
# JWST 25m^2 collecting area
# Flux loss from masks and occulters are taken into account in WebbPSF
S.refs.setref(area = 25.4e4) # cm^2

# The following won't work on readthedocs compilation
if not on_rtd:
    # Grab WebbPSF assumed pixel scales
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)
    nc_temp = webbpsf.NIRCam()
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
        even if mask=None. Gives the option of using LYOT or grism pupils 
        with or without coron substrate.

    Returns
    -------
    :mod:`pysynphot.obsbandpass`
        A Pysynphot bandpass object.
    """

    if module is None: module = 'A'

    # Select filter file and read
    filter = filter.upper()
    mod = module.lower()
    filt_dir = conf.PYNRC_PATH + 'throughputs/'
    filt_file = filter + '_nircam_plus_ote_throughput_mod' + mod + '_sorted.txt'

    _log.debug('Reading file: '+filt_file)
    bp = S.FileBandpass(filt_dir+filt_file)
    bp_name = filter

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
    # Substrate transmission (off-axis substrate with occulting masks)
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
    # Substrate transmission (located in pupil wheel to deflect beam)
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

def grism_wref(pupil='GRISM', module='A'):
    """Grism undeviated wavelength"""

    # Option for GRISMR/GRISMC
    if 'GRISMR' in pupil:
        pupil = 'GRISM0'
    elif 'GRISMC' in pupil:
        pupil = 'GRISM90'

    # Mean spectral dispersion in number of pixels per um
    if ('GRISM90' in pupil) and (module == 'A'):
        wref = 3.978
    elif ('GRISM0' in pupil)  and (module == 'A'):
        wref = 3.937
    elif ('GRISM90' in pupil) and (module == 'B'):
        wref = 3.923
    elif ('GRISM0' in pupil)  and (module == 'B'):
        wref = 3.960
    else:
        wref = 3.95

    return wref

def grism_res(pupil='GRISM', module='A', m=1):
    """Grism resolution

    Based on the pupil input and module, return the spectral
    dispersion and resolution as a tuple (res, dw).

    Parameters
    ----------
    pupil : str
        'GRISM0' or 'GRISM90', otherwise assume res=1000 pix/um.
        'GRISM0' is GRISMR; 'GRISM90' is GRISMC
    module : str
        'A' or 'B'
    m : int
        Spectral order (1 or 2).
    """

    # Option for GRISMR/GRISMC
    if 'GRISMR' in pupil:
        pupil = 'GRISM0'
    elif 'GRISMC' in pupil:
        pupil = 'GRISM90'

    # Mean spectral dispersion in number of pixels per um
    if ('GRISM90' in pupil) and (module == 'A'):
        res = 1003.12
    elif ('GRISM0' in pupil)  and (module == 'A'):
        res = 996.48
    elif ('GRISM90' in pupil) and (module == 'B'):
        res = 1008.64
    elif ('GRISM0' in pupil)  and (module == 'B'):
        res = 1009.13
    else:
        res = 1000.0

    if m==2:
        res *= 2

    # Spectral resolution in um/pixel
    dw = 1. / res

    return (res, dw)

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
    if wref is None: 
        wref = grism_wref(pupil, module) 
    
    # Background spectrum
    if sp_bg is None:
        sp_bg = zodi_spec(**kwargs)

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

    catname = kwargs.get('catname', 'bosz')
    lookuptable = {
        # https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
        "O0V": (50000, 0.0, 4.0), # Bracketing for interpolation
        "O3V": (46000, 0.0, 4.0),
        "O5V": (43000, 0.0, 4.5),
        "O7V": (36500, 0.0, 4.0),
        "O9V": (32500, 0.0, 4.0),
        "B0V": (31500, 0.0, 4.0),
        "B1V": (26000, 0.0, 4.0),
        "B3V": (17000, 0.0, 4.0),
        "B5V": (15700, 0.0, 4.0),
        "B8V": (12500, 0.0, 4.0),
        "A0V": (9700, 0.0, 4.0),
        "A1V": (9200, 0.0, 4.0),
        "A3V": (8550, 0.0, 4.0),
        "A5V": (8080, 0.0, 4.0),
        "F0V": (7220, 0.0, 4.0),
        "F2V": (6810, 0.0, 4.0),
        "F5V": (6510, 0.0, 4.0),
        "F8V": (6170, 0.0, 4.5),
        "G0V": (5920, 0.0, 4.5),
        "G2V": (5770, 0.0, 4.5),
        "G5V": (5660, 0.0, 4.5),
        "G8V": (5490, 0.0, 4.5),
        "K0V": (5280, 0.0, 4.5),
        "K2V": (5040, 0.0, 4.5),
        "K5V": (4410, 0.0, 4.5),
        "K7V": (4070, 0.0, 4.5),
        "M0V": (3870, 0.0, 4.5),
        "M2V": (3550, 0.0, 4.5),
        "M5V": (3030, 0.0, 5.0),
        "M9V": (2400, 0.0, 5.0),   # Bracketing for interpolation
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
        model to the observed photometric data. If no parameters passed,
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

        from scipy.optimize import least_squares

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
    from Zhu et al. (2015). 

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
        Location and name of Linder et al file. 
        Default is 'BEX_evol_mags_-3_MH_0.00.dat'
    """

    # Default file to read and load
    if file is None:
        indir = os.path.join(conf.PYNRC_PATH, 'linder/isochrones/')
        file = indir + 'BEX_evol_mags_-3_MH_0.00.dat'

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
    
def linder_filter(table, filt, age, dist=10, cond_interp=True, cond_file=None, **kwargs):
    """Linder Mags vs Mass Arrays
    
    Given a Linder table, NIRCam filter, and age, return arrays of MJup 
    and Vega mags. If distance (pc) is provided, then return the apparent 
    magnitude, otherwise absolute magnitude at 10pc.
    
    This function takes the isochrones tables from Linder et al 2019 and
    creates a irregular contour grid of filter magnitude and log(age)
    where the z-axis is log(mass). This is mapped onto a regular grid
    that is interpolated within the data boundaries and linearly
    extrapolated outside of the region of available data.
    
    Parameters
    ==========
    table : astropy table
        Astropy table output from `linder_table`.
    filt : string
        Name of NIRCam filter.
    age : float
        Age of planet mass.
    dist : float
        Distance in pc. Default is 10pc (abs mag).
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
    z2 = mass2_mjup * 318 # Convert to Earth masses
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
    
    # TODO: Rewrite this function to better extrapolate to lower and higher masses
    # For now, fit low order polynomial
    isort = np.argsort(mag_abs_arr)
    mag_abs_arr = mag_abs_arr[isort]
    mass_arr = mass_arr[isort]
    ind_fit = mag_abs_arr<x.max()
    lxmap = [mag_abs_arr.min(), mag_abs_arr.max()]
    xfit = np.append(mag_abs_arr[ind_fit], mag_abs_arr[-1])
    yfit = np.log10(np.append(mass_arr[ind_fit], mass_arr[-1]))
    cf = jl_poly_fit(xfit, yfit, deg=4, use_legendre=False, lxmap=lxmap)
    mass_arr = 10**jl_poly(mag_abs_arr,cf)


    mag_app_arr = mag_abs_arr + 5*np.log10(dist/10.0)

    # Sort by mass
    isort = np.argsort(mass_arr)
    mass_arr = mass_arr[isort]
    mag_app_arr = mag_app_arr[isort]


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

    # Want th_bar to be -90 so that r matches webbpsf
    if theta>0: 
        r  = -1 * r
        theta = -1 * theta

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

    ### Wedge Masks
    if 'WB' in name: 
        scalefact = (2 + (-x + 7.5) * 4 / 15).clip(2, 6)
        wedgesign = 1 if name == 'MASKSWB' else -1
        scalefact = (2 + (x * wedgesign + 7.5) * 4 / 15).clip(2, 6)
        if name == 'MASKSWB':
            polyfitcoeffs = np.array([2.01210737e-04, -7.18758337e-03, 1.12381516e-01,
                                      -1.00877701e+00, 5.72538509e+00, -2.12943497e+01,
                                      5.18745152e+01, -7.97815606e+01, 7.02728734e+01])
            # scalefact = scalefact[:, ::-1] # flip orientation left/right for SWB mask
        elif name == 'MASKLWB':
            polyfitcoeffs = np.array([9.16195583e-05, -3.27354831e-03, 5.11960734e-02,
                                      -4.59674047e-01, 2.60963397e+00, -9.70881273e+00,
                                      2.36585911e+01, -3.63978587e+01, 3.20703511e+01])

        sigma_func = np.poly1d(polyfitcoeffs)
        sigmas = sigma_func(scalefact)
        sigmar = sigmas * np.abs(y)
        # clip sigma: The minimum is to avoid divide by zero
        #             the maximum truncates after the first sidelobe to match the hardware
        sigmar.clip(min=np.finfo(sigmar.dtype).tiny, max=2*np.pi, out=sigmar)
        transmission = (1 - (np.sin(sigmar) / sigmar) ** 2)
        # the bar should truncate at +- 10 arcsec
        woutside = np.where(np.abs(x) > 10)
        transmission[woutside] = 1.0
    ### Circular Masks
    else: 
        r = poppy.accel_math._r(x, y)
        sigmar = sigma * r

        # clip sigma: The minimum is to avoid divide by zero
        #             the maximum truncates after the first sidelobe to match the hardware
        bessel_j1_zero2 = scipy.special.jn_zeros(1, 2)[1]
        sigmar.clip(np.finfo(sigmar.dtype).tiny, bessel_j1_zero2, out=sigmar)  # avoid divide by zero -> NaNs
        if poppy.accel_math._USE_NUMEXPR:
            import numexpr as ne
            # jn1 = scipy.special.j1(sigmar)
            jn1 = scipy.special.jv(1,sigmar)
            transmission = ne.evaluate("(1 - (2 * jn1 / sigmar) ** 2)")
        else:
            # transmission = (1 - (2 * scipy.special.j1(sigmar) / sigmar) ** 2)
            transmission = (1 - (2 * scipy.special.jv(1,sigmar) / sigmar) ** 2)

        # r = np.sqrt(x ** 2 + y ** 2)
        # sigmar = sigma * r
        # #sigmar.clip(np.finfo(sigmar.dtype).tiny, 2*np.pi, out=sigmar)  # avoid divide by zero -> NaNs
        # sigmar.clip(np.finfo(sigmar.dtype).tiny, 7.1559, out=sigmar)  # avoid divide by zero -> NaNs
        # transmission = (1 - (2 * scipy.special.jn(1, sigmar) / sigmar) ** 2)
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
        if ((module == 'A' and name == 'MASKLWB') or
                (module == 'B' and name == 'MASK210R')):
            wnd_5 = np.where(
                ((y > 5) & (y < 10)) &
                (
                    ((x < -5) & (x > -10)) |
                    ((x > 7.5) & (x < 12.5))
                )
            )
            wnd_2 = np.where(
                ((y > -10) & (y < -8)) &
                (
                    ((x < -8) & (x > -10)) |
                    ((x > 9) & (x < 11))
                )
            )
        elif ((module == 'A' and name == 'MASK210R') or
              (module == 'B' and name == 'MASKSWB')):
            wnd_5 = np.where(
                ((y > 5) & (y < 10)) &
                (
                    ((x > -12.5) & (x < -7.5)) |
                    ((x > 5) & (x < 10))
                )
            )
            wnd_2 = np.where(
                ((y > -10) & (y < -8)) &
                (
                    ((x > -11) & (x < -9)) |
                    ((x > 8) & (x < 10))
                )
            )
        else:
            wnd_5 = np.where(
                ((y > 5) & (y < 10)) &
                (np.abs(x) > 7.5) &
                (np.abs(x) < 12.5)
            )
            wnd_2 = np.where(
                ((y > -10) & (y < -8)) &
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
