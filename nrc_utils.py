"""
pyNRC utility functions
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# The six library is useful for Python 2 and 3 compatibility
import six

# Import libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
rcvals = {'xtick.minor.visible': True, 'ytick.minor.visible': True,
          'xtick.direction': 'in', 'ytick.direction': 'in', 
          'xtick.top': True, 'ytick.right': True, 'font.family': ['serif'],
          'xtick.major.size': 6, 'ytick.major.size': 6,
          'xtick.minor.size': 3, 'ytick.minor.size': 3,
          'image.interpolation': 'none', 'image.origin': 'lower',
          'figure.figsize': [8,6]}#,
          #'text.usetex': True, 'text.latex.preamble': ['\usepackage{gensymb}']}
matplotlib.rcParams.update(rcvals)
cmap_pri, cmap_alt = ('viridis', 'gist_heat')
matplotlib.rcParams['image.cmap'] = cmap_pri if cmap_pri in plt.colormaps() else cmap_alt


import datetime, time
import yaml, re, os
import sys, platform
import multiprocessing as mp
import traceback

from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
from astropy.time import Time

#from scipy.optimize import least_squares#, leastsq
#from scipy.ndimage import fourier_shift

from . import conf
from .logging_utils import setup_logging

from .maths import robust
from .maths.image_manip import *
from .maths.fast_poly import *

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
_webbpsf_version_min = (0,5,0)
_ = webbpsf.utils.get_webbpsf_data_path(_webbpsf_version_min)

# Link to WebbPSF's instance of poppy
from webbpsf.webbpsf_core import poppy

# Set up some poppy and webbpsf defaults
poppy.conf.use_multiprocessing = True # Assume multiprocessing
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
poppy.conf.n_processes = int(0.75 * mp.cpu_count()) if poppy.conf.use_multiprocessing else 1

webbpsf.conf.default_output_mode = u'detector'

# Some useful functions for displaying and measuring PSFs
from poppy import radial_profile, measure_fwhm
#from poppy import (display_PSF, display_PSF_difference, display_EE, display_profiles, radial_profile,
#        measure_EE, measure_radial, measure_fwhm, measure_sharpness, measure_centroid, measure_strehl,
#        specFromSpectralType, fwcentroid)

import pysynphot as S
# Extend default wavelength range to 5.6 um
S.refs.set_default_waveset(minwave=500, maxwave=56000, num=10000.0, delta=None, log=False)
# JWST 25m^2 collecting area
# Flux loss from masks and occulters are taken into account in WebbPSF
S.refs.setref(area = 25.4e4) # cm^2

# Grab WebbPSF assumed pixel scales 
nc_temp = webbpsf.NIRCam()
pixscale_SW = nc_temp._pixelscale_short
pixscale_LW = nc_temp._pixelscale_long
del nc_temp

###########################################################################
#
#    Logging info
#
###########################################################################


import logging
_log = logging.getLogger('pynrc')

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
    ice_scale=None, nvr_scale=None, **kwargs):
    """
    Read in filter throughput curve from file generated by STScI.
    Includes: OTE, NRC mirrors, dichroic, filter curve, and detector QE.
    
    Additional Keywords
    ===================
    ND_acq    : ND acquisition square in coronagraphic mask.
    ice_scale : Add in additional OTE H2O absorption. This is a scale factor 
                relative to 0.0131 um thickness
    nvr_scale : Add in additiona NIRCam non-volatile residue. This is scale factor 
                relative to 0.280 um thickness.

    Returns a Pysynphot bandpass object.
    """

    if module is None: module = 'A'

    # Select filter file and read
    f = filter.lower(); m = module.lower()
    #filt_dir = __location__ + 'throughputs_stsci/'
    filt_dir = conf.PYNRC_PATH + 'throughputs/'
    filt_file = filter + '_nircam_plus_ote_throughput_mod' + m + '_sorted.txt'
    bp = S.FileBandpass(filt_dir+filt_file)

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
        if module == 'A':
            cf_g = np.array([0.068695897, -0.943894294, 4.1768413, -5.306475735])
        else:
            cf_g = np.array([0.050758635, -0.697433006, 3.086221627, -3.92089596])
    
        # Create polynomial function for grism throughput from coefficients
        p = np.poly1d(cf_g)
        th_grism = p(bp.wave/1e4)
        th_grism[th_grism < 0] = 0
    
        # Multiply filter throughput by grism
        th_new = th_grism * bp.throughput
        bp = S.ArrayBandpass(bp.wave, th_new, name=filter)
    
        # spectral resolution in um/pixel
        # res is in pixels/um and dw is inverse
        res, dw = grism_res(pupil, module)
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
        bp = S.ArrayBandpass(bp.wave, th_new, name=filter)
    
        # Mean spectral dispersion (dw/pix)
        res = 290.0
        dw = 1. / res # um/pixel
        dw *= 10000   # Angstrom/pixel
    
        npts = np.int(wrange/dw)+1
        warr = np.linspace(w1, w1+dw*npts, npts)
        bp = bp.resample(warr)

    # Coronagraphic throughput modifications
    # Substrate transmission
    if ((mask is not None) and ('MASK' in mask)) or ND_acq:
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
        
            # CV3 data shows OD needs to be multiplied by 0.93 
            # compared to Barr measurements
            odata *= 0.93
        
            otemp = np.interp(wtemp, wdata, odata, left=0, right=0)
            ttemp *= 10**(-1*otemp)
    
        # Interpolate substrate transmission onto filter wavelength grid and multiply
        th_coron_sub = np.interp(bp.wave/1e4, wtemp, ttemp, left=0, right=0)		
        th_new = th_coron_sub * bp.throughput
        bp = S.ArrayBandpass(bp.wave, th_new, name=filter)
    

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
        bp = S.ArrayBandpass(bp.wave, th_new, name=filter)


    # Weak Lens substrate transmission
    if (pupil is not None) and ('WEAK LENS' in pupil):
        # Even though this says WL+8, this should work for all lenses
        hdulist = fits.open(conf.PYNRC_PATH + 'throughputs/jwst_nircam_wlp8.fits')
        wtemp = hdulist[1].data['WAVELENGTH']
        ttemp = hdulist[1].data['THROUGHPUT']

        # If two lenses, then we need to multiply throughput twice
        wl_list = ['WEAK LENS +12 (=4+8)', 'WEAK LENS -4 (=4-8)']
        pow = 2 if pupil in wl_list else 1

        th_wl = np.interp(bp.wave/1e4, wtemp, ttemp, left=0, right=0)
        th_new = th_wl**pow * bp.throughput
        bp = S.ArrayBandpass(bp.wave, th_new, name=filter)
        
    
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
            ttemp = np.interp(bp.wave/1e4, wtemp, ttemp, left=0, right=0)
            th_ice = np.exp(ice_scale * np.log(ttemp))
            th_new = th_ice * th_new
        if nvr_scale is not None:
            ttemp = data['t_nvr']
            ttemp = np.insert(ttemp, 0, [1.0]) # Estimates for w<2.5um
            ttemp = np.append(ttemp, [1.0])    # Estimates for w>5.0um
            # Interpolate transmission onto filter wavelength grid
            ttemp = np.interp(bp.wave/1e4, wtemp, ttemp, left=0, right=0)
            th_nvr = np.exp(nvr_scale * np.log(ttemp))
            th_new = th_nvr * th_new
        
        # Create new bandpass
        bp = S.ArrayBandpass(bp.wave, th_new, name=filter)


    # Resample to common dw to make ensure consistency
    dw_arr = bp.wave[1:] - bp.wave[:-1]
    #if not np.isclose(dw_arr.min(),dw_arr.max()):
    dw = np.median(dw_arr)
    warr = np.arange(w1,w2, dw)
    bp = bp.resample(warr)

    # Need to place zeros at either end so Pysynphot doesn't extrapolate
    warr = np.concatenate(([bp.wave.min()-dw],bp.wave,[bp.wave.max()+dw]))
    tarr = np.concatenate(([0],bp.throughput,[0]))
    bp   = S.ArrayBandpass(warr, tarr, name=filter)

    return bp


###########################################################################
#
#    Create WebbPSF Coefficients and Images
#
###########################################################################

# Subclass of the WebbPSF NIRCam class to fix coronagraphy bug
from webbpsf import NIRCam as NIRCam_webbpsf
class NIRCam(NIRCam_webbpsf):
    def __init__(self):
        NIRCam_webbpsf.__init__(self)

    def _addAdditionalOptics(self,optsys, oversample=2):
        """ Slight re-write of the webbpsf version of this function -JML
        
        Add coronagraphic optics for NIRCam

        See Krist et al. 2007, 2009 SPIE

        Three circular occulters: HWHM = 6 lambda/D at 2.1, 3.35, 4.3
                                       = 0.4, 0.64, 0.8 arcsec (avg)
                                       assuming D_tel=6.5m exactly:
                                        = 0.3998, 0.6378, 0.8187 arcsec

        Two linear bar occulters: Wedges vary from HWHM = 2 lam/D to 6 lam/D at 2.1 and 4.6 micron
                    2.1e-6:    HWHM = 0.13327 to 0.3998
                    4.6e-6:    HWHM = 0.27290 to 0.8187
            The matching Lyot stop for the wedges are tuned for 4 lam/D.
            The linear ones have a fixed width at either side: maybe ~ 3-4 arcsec. Then a linear taper
            in between.


        Values of Sigma:
            For circular occulters, 0.3998 requires sigma = 5.253
                                    0.8187 requires sigma = 2.5652
                                    sigma = 2.10013932 / loc
                                    vs. Krist's statement sigma = 2.1001/hwhm

            For linear occulters, 0.3998 requires sigma = 4.5012
                                  0.13327 requires sigma = 13.5078

                        # This is NOT a linear relationship! It's a tricky inverse sin nonlinear thing.

        Empirical checks against John Krist's provided 430R and LWB files:
            430R should have sigma = 2.588496


        Since the Weak Lenses go in the pupil too, this function provides a convenient place to implement those as well.

        """

        #optsys.add_image(name='null for debugging NIRcam _addCoron') # for debugging
        from webbpsf.optics import NIRCam_BandLimitedCoron

        if ((self.image_mask == 'MASK210R') or (self.image_mask == 'MASK335R') or
                (self.image_mask == 'MASK430R')):
            optsys.add_image( NIRCam_BandLimitedCoron( name=self.image_mask, module=self.module),
                    index=2)
            trySAM = False # FIXME was True - see https://github.com/mperrin/poppy/issues/169
            SAM_box_size = 5.0
        elif ((self.image_mask == 'MASKSWB') or (self.image_mask == 'MASKLWB')):
            optsys.add_image( NIRCam_BandLimitedCoron(name=self.image_mask, module=self.module),
                    index=2)
            trySAM = False #True FIXME
            SAM_box_size = [5,20]
        #elif ((self.pupil_mask is not None) and (self.pupil_mask.startswith('MASK'))):
        else:
            # no occulter selected but coronagraphic mode anyway. E.g. off-axis PSF
            # but don't add this image plane for weak lens calculations
            optsys.add_image(poppy.ScalarTransmission(name='No Image Mask Selected!'), index=2)
            trySAM = False
            SAM_box_size = 1.0 # irrelevant but variable still needs to be set.

        # add pupil plane mask
        if ('pupil_shift_x' in self.options and self.options['pupil_shift_x'] != 0) or \
           ('pupil_shift_y' in self.options and self.options['pupil_shift_y'] != 0):
            shift = (self.options['pupil_shift_x'], self.options['pupil_shift_y'])
        else: shift = None


        #NIRCam as-built weak lenses, from WSS config file
        WLP4_diversity =  8.27398 # microns
        WLP8_diversity = 16.4554  # microns
        WLM8_diversity =-16.4143  # microns
        WL_wavelength =   2.12    # microns

        #optsys.add_pupil( name='null for debugging NIRcam _addCoron') # debugging
        if self.pupil_mask == 'CIRCLYOT':
            optsys.add_pupil(transmission=self._datapath+"/optics/NIRCam_Lyot_Somb.fits", name=self.pupil_mask,
                    flip_y=True, shift=shift, index=3)
            optsys.planes[3].wavefront_display_hint='intensity'
        elif self.pupil_mask == 'WEDGELYOT':
            optsys.add_pupil(transmission=self._datapath+"/optics/NIRCam_Lyot_Sinc.fits", name=self.pupil_mask,
                    flip_y=True, shift=shift, index=3)
            optsys.planes[3].wavefront_display_hint='intensity'
        elif self.pupil_mask == 'WEAK LENS +4':
            optsys.add_pupil(poppy.ThinLens(
                name='Weak Lens +4',
                nwaves=WLP4_diversity / WL_wavelength,
                reference_wavelength=WL_wavelength*1e-6, #convert microns to meters
                radius=self.pupil_radius
            ), index=3)
        elif self.pupil_mask == 'WEAK LENS +8':
            optsys.add_pupil(poppy.ThinLens(
                name='Weak Lens +8',
                nwaves=WLP8_diversity / WL_wavelength,
                reference_wavelength=WL_wavelength*1e-6,
                radius=self.pupil_radius
            ), index=3)
        elif self.pupil_mask == 'WEAK LENS -8':
            optsys.add_pupil(poppy.ThinLens(
                name='Weak Lens -8',
                nwaves=WLM8_diversity / WL_wavelength,
                reference_wavelength=WL_wavelength*1e-6,
                radius=self.pupil_radius
            ), index=3)
        elif self.pupil_mask == 'WEAK LENS +12 (=4+8)':
            stack = poppy.CompoundAnalyticOptic(name='Weak Lens Pair +12', opticslist=[
                poppy.ThinLens(
                    name='Weak Lens +4',
                    nwaves=WLP4_diversity / WL_wavelength,
                    reference_wavelength=WL_wavelength*1e-6,
                    radius=self.pupil_radius
                ),
                poppy.ThinLens(
                    name='Weak Lens +8',
                    nwaves=WLP8_diversity / WL_wavelength,
                    reference_wavelength=WL_wavelength*1e-6,
                    radius=self.pupil_radius
                )]
            )
            optsys.add_pupil(stack, index=3)
        elif self.pupil_mask == 'WEAK LENS -4 (=4-8)':
            stack = poppy.CompoundAnalyticOptic(name='Weak Lens Pair -4', opticslist=[
                poppy.ThinLens(
                    name='Weak Lens +4',
                    nwaves=WLP4_diversity / WL_wavelength,
                    reference_wavelength=WL_wavelength*1e-6,
                    radius=self.pupil_radius
                ),
                poppy.ThinLens(
                    name='Weak Lens -8',
                    nwaves=WLM8_diversity / WL_wavelength,
                    reference_wavelength=WL_wavelength*1e-6,
                    radius=self.pupil_radius
                )]
            )
            optsys.add_pupil(stack, index=3)


        elif (self.pupil_mask is None and self.image_mask is not None):
            optsys.add_pupil(poppy.ScalarTransmission(name='No Lyot Mask Selected!'), index=3)

        return (optsys, trySAM, SAM_box_size)


def nproc_use(fov_pix, oversample, nwavelengths=None, coron=False):
    """ 
    Attempt to estimate a reasonable number of processes to use for a multi-wavelength calculation.

    Here we attempt to estimate how many such calculations can happen in
    parallel without swapping to disk, with a mixture of empiricism and conservatism.
    One really does not want to end up swapping to disk with huge arrays.

    NOTE: Requires psutil package. Otherwise defaults to mp.cpu_count() / 2

    Parameters
    -----------
    fov_pix      : Square size in detector-sampled pixels of final PSF image.
    oversample   : The optical system that we will be calculating for.
    nwavelengths : Number of wavelengths. Sets maximum # of processes.
    coron        : Is the nproc recommendation for coronagraphic imaging? 
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
    avail_GB = mem.available / (1024**3) - 1.0 # Leave 1 GB

    fov_pix_over = fov_pix * oversample

    # Memory formulas are based on fits to memory usage stats for:
    #   fov_arr = np.array([16,32,128,160,256,320,512,640,1024,2048])
    #   os_arr = np.array([1,2,4,8])
    # Coronagraphic Imaging (in MB)
    if coron: 
        mem_total = (oversample*1024*2.4)**2 * 16 / (1024**2) + 500
        if fov_pix > 1024: mem_total *= 1.6
    # Direct Imaging (also spectral imaging)
    else: mem_total = 5*(fov_pix_over)**2 * 8 / (1024**2) + 300. 

    # Convert to GB
    mem_total /= 1024

    # How many processors to split into?
    nproc = avail_GB // mem_total
    nproc = np.min([nproc, mp.cpu_count(), poppy.conf.n_processes])
    if nwavelengths is not None:
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

    return int(nproc)

def _wrap_coeff_for_mp(args):
    """
    Internal helper routine for parallelizing computations across multiple processors.
    """
    # No multiprocessing for monochromatic wavelengths
    mp_prev = poppy.conf.use_multiprocessing
    poppy.conf.use_multiprocessing = False

    inst,w,fov_pix,oversample = args
    fov_pix_orig = fov_pix # Does calc_psf change fov_pix??
    try:
        hdu_list = inst.calc_psf(outfile=None, save_intermediates=False, \
                                 oversample=oversample, rebin=True, \
                                 fov_pixels=fov_pix, monochromatic=w*1e-6)
    except Exception as e:
        print('Caught exception in worker thread (w = {}):'.format(w))
        # This prints the type, value, and stack trace of the
        # current exception being handled.
        traceback.print_exc()

        print()
        raise e

    # Return to previous setting
    poppy.conf.use_multiprocessing = mp_prev
    return pad_or_cut_to_size(hdu_list[0].data, fov_pix_orig*oversample)
    #return hdu_list[0].data


def psf_coeff(filter_or_bp, pupil=None, mask=None, module='A', 
    fov_pix=11, oversample=None, npsf=None, ndeg=7, opd=None, tel_pupil=None,
    offset_r=0, offset_theta=0, save=True, force=False, **kwargs):
    """
    Creates a set of coefficients that will generate a simulated PSF at any
    arbitrary wavelength. This function first uses webbPSF to simulate
    a number of evenly spaced PSFs throughout some specified bandpass.
    A 7th-degree polynomial is then fit to each pixel (oversampled, by default)
    using numpy's linear-least square fitting routine. The final set of
    coefficients for each pixel is returned as an image cube. The returned
    set of coefficient can be used to produce a PSFs by:

        psfs = pynrc.nrc_utils.jl_poly(waves, coeffs)
    
    where 'waves' can be a scalar, nparray, list, or tuple. All wavelengths
    are in microns.

    Parameters
    -------------------
    filter_or_bp : Either the name of a filter or a Pysynphot bandpass.
    pupil        : NIRCam pupil elements such as grisms or lyot stops
    mask         : Specify the coronagraphic occulter (spots or bar)
    module       : 'A' or 'B'
    fov_pix      : Size of the FoV in pixels (real SW or LW pixels)
    oversample   : Factor to oversample pixels (in one dimension). 
                   The resulting coefficients will have x/y dimensions 
                   of fov_pix*oversample. Default 2 for coronagraphy and 
                   4 otherwise.
                   
    npsf : Number of evenly-spaced (with wavelength) monochromatic PSFs to 
           generate with webbPSF. If not specified, then the default is to 
           produce 20 PSFs/um. The wavelength range is determined by
           choosing those wavelengths where throughput is >0.001.
    ndeg : Polynomial degree for PSF fitting.
    
    opd  : OPD file info. Acceptable forms:
           1. ('OPD_RevV_nircam_150.fits', 0)
           2. ('OPD_RevV_nircam_150.fits', 0, 10) - specifies 10 nm WFE drift
           3. HDUlist
           
    tel_pupil : Telescope entrance pupil mask. By default pupil_RevV.fits. 
        Should either be a filename string or HDUList.
           
    offset_r     :
    offset_theta :
    
    save  :
    force :
    """

    grism_obs = (pupil is not None) and ('GRISM' in pupil)
    dhs_obs   = (pupil is not None) and ('DHS'   in pupil)
    coron_obs = (pupil is not None) and ('LYOT'  in pupil)
    
    if oversample is None:
        oversample = 2 if coron_obs else 4

    # Default OPD
    if opd is None: opd = ('OPD_RevV_nircam_132.fits', 0)
       
    # Get filter throughput and create bandpass 
    if isinstance(filter_or_bp, six.string_types):
        filter = filter_or_bp
        bp = read_filter(filter, pupil=pupil, mask=mask, module=module)
    else:
        bp = filter_or_bp
        filter = bp.name

    # Change log levels to WARNING for pyNRC, WebbPSF, and POPPY
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)

    # Create a simulated PSF with WebbPSF
    inst = NIRCam()
    inst.options['output_mode'] = 'oversampled'
    inst.options['parity'] = 'odd'
    #inst.options['source_offset_r'] = offset_r
    #inst.options['source_offset_theta'] = offset_theta
    #inst.pupilopd = opd
    inst.filter = filter
    setup_logging(log_prev, verbose=False)

    # Check if mask and pupil names exist in WebbPSF lists.
    # We don't want to pass values that WebbPSF does not recognize,
    # but are otherwise completely valid in the NIRCam framework.
    if mask in list(inst.image_mask_list): inst.image_mask = mask
    if pupil in list(inst.pupil_mask_list): inst.pupil_mask = pupil
    
    # Telescope Pupil
    if tel_pupil is not None:
        #print('Adding telescope pupil')
        inst.pupil = tel_pupil


    mtemp = 'none' if mask is None else mask
    ptemp = 'none' if pupil is None else pupil
    # Get source offset positions
    # 1. Round masks - Always assume theta=0 due to symmetry.
    # 2. Bar Masks - PSF positioning is different depending on r and theta.
    # 3. All other imaging - Just perform nominal r=theta=0.
    #    Any PSF movement is more quickly applied with sub-pixel shifting routines.
    # NB: Implementation of field-dependent OPD maps may change these settings.
    rtemp, ttemp = (offset_r, offset_theta)
    if ('210R' in mtemp) or ('335R' in mtemp) or ('430R' in mtemp):
        rtemp, ttemp = (offset_r, 0)
    elif ('MASKSWB' in mtemp) or ('MASKLWB' in mtemp):
        rtemp, ttemp = (offset_r, offset_theta)
    else:
        rtemp = ttemp = 0
    inst.options['source_offset_r']     = rtemp
    inst.options['source_offset_theta'] = ttemp

    # Deal with OPD file name
    #print(opd)
    wfe_drift = 0
    if isinstance(opd, tuple):
        if len(opd)==2: # No drift
            pass
        elif len(opd)==3: #3rd element is the nm drifted.
            wfe_drift = opd[2]
            opd = (opd[0], opd[1])
        else:
            raise ValueError("OPD passed as tuple must have length of 2 or 3.")
         # Filename info
        opd_nm = opd[0][-8:-5] # RMS WFE (e.g., 132)
        opd_num = opd[1]       # OPD slice
        otemp = 'OPD{}nm{:.0f}'.format(opd_nm, opd_num)
        if wfe_drift > 0:
            otemp = '{}-{:.0f}nm'.format(otemp, wfe_drift)
    elif isinstance(opd, fits.HDUList):
        # A custom OPD is passed. Consider using force=True.
        otemp = 'OPDcustom'
    else:
        raise ValueError("opd must be a tuple or HDUList.")
    
    # Name to save array of oversampled coefficients
    save_dir = conf.PYNRC_PATH + 'psf_coeffs/'
    # Create directory if it doesn't already exist
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    # Final filename to save coeff
    fname = '{}_{}_{}_{}_{}_{}_{:.1f}_{:.1f}_{}.npy'.\
        format(filter,mtemp,ptemp,module,fov_pix,oversample,rtemp,ttemp,otemp)
    save_name = save_dir + fname

    if (not force) and (save and os.path.exists(save_name)):
        return np.load(save_name)

    # Only drift OPD if PSF is in nominal position (rtemp=0).
    # Anything that is in an offset position is currently considered
    # to be a companion source that we're trying to detect, so the
    # PSF WFE difference has negligible bearing on the outcome.
    # 
    # Need to look at functions inside speckle_noise to see if
    # I can speed up this process. Might be running some unnecessary
    # processes.
    if (rtemp == 0) and (not isinstance(opd, fits.HDUList)):
        from . import speckle_noise as sn
        pupilopd = opd
        # Read in a specified OPD file and slice
        opd_im, header = sn.read_opd_slice(pupilopd, header=True)
        # Create an object that extracts the Zernike components for the OPD file
        # We don't really care about the Hexike terms for individual segments 
        opd_obj = sn.OPD_extract(opd_im, header, seg_terms=5)
        # Standard deviation for each pupil Zernike
        # Piston, Tip, Tilt, Focus, Astig, Coma, Trefoil,
        #   Spherical, 2nd Astig, and Quadrafoil
        pup_cf_std = np.array([ 0.0,  0.0,  0.0,  0.03258147,  0.02023745,
            0.02036721,  0.02493373,  0.02338593,  0.01750089,  0.00750398,
            0.01938736,  0.01712082,  0.01359327,  0.01458599,  0.00956585])
        seg_cf_std = np.array([ 0.0,  0.0,  0.0,  0.01140949,  0.01712216,
            0.01586312,  0.0234882 ,  0.02331315,  0.02316506,  0.02164931,
            0.02015695,  0.01749708,  0.01672629,  0.01263545,  0.01471005,
            0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,  0.0,  0.0,  0.0,  0.0])
        # Generate science OPD image and residuals for use in reference drift.
        opd_sci, opd_resid = sn.opd_sci_gen(opd_obj)
        if wfe_drift > 0:
            # Generate reference OPD image
            args = (opd_obj, wfe_drift, pup_cf_std, seg_cf_std, opd_resid, 1)
            opd_ref = sn.opd_ref_gen(args)
            # Create OPD inside an HDUList to pass to WebbPSF
            hdu = fits.PrimaryHDU(opd_ref)
        else:
            hdu = fits.PrimaryHDU(opd_sci)

        hdu.header = header.copy()
        opd_hdulist = fits.HDUList([hdu]) 
        inst.pupilopd = opd_hdulist
    else:
        inst.pupilopd = opd

    # Select which wavelengths to use
    wgood = bp.wave / 1e4
    w1 = wgood.min()
    w2 = wgood.max()

    # WebbPSF has wavelength limits depending on the channel by default
    # We don't care about this, so set these to low/high values
    inst.SHORT_WAVELENGTH_MIN = inst.LONG_WAVELENGTH_MIN = 0
    inst.SHORT_WAVELENGTH_MAX = inst.LONG_WAVELENGTH_MAX = 10e-6

    # Create set of monochromatic PSFs to fit.
    if npsf is None:
        dn = 20 # 20 PSF simulations per um
        npsf = np.ceil(dn * (w2-w1))
    npsf = int(npsf)
    waves = np.linspace(w1, w2, npsf)

    # How many processors to split into?
    nproc = nproc_use(fov_pix, oversample, npsf) if poppy.conf.use_multiprocessing else 1
    _log.debug('nprocessors: %.0f; npsf: %.0f' % (nproc, npsf))
    # Change log levels to WARNING for pyNRC, WebbPSF, and POPPY
    setup_logging('WARN', verbose=False)
    
    t0 = time.time()
    # Setup the multiprocessing pool and arguments to pass to each pool
    worker_arguments = [(inst, wlen, fov_pix, oversample) for wlen in waves]
    if nproc > 1: 
        pool = mp.Pool(nproc)
        # Pass arguments to the helper function
        images = pool.map(_wrap_coeff_for_mp, worker_arguments)

        try:
            images = pool.map(_wrap_coeff_for_mp, worker_arguments)
        except Exception as e:
            print('Caught an exception during multiprocess.')
            raise e
        finally:
            _log.debug('Closing multiprocess pool.')
            pool.close()
    else:
        # Pass arguments to the helper function
        images = map(_wrap_coeff_for_mp, worker_arguments)	
    t1 = time.time()
    
    # Reset to original log levels
    setup_logging(log_prev, verbose=False)
    _log.debug('Took %.2f seconds to generate WebbPSF images' % (t1-t0))

    # Take into account reduced beam factor for grism data
    # Account for the circular pupil that does not allow all grism grooves to have their 
    # full length illuminated (Erickson & Rabanus 2000), effectively broadening the FWHM.
    # It's actually a hexagonal pupil, so the factor is 1.07, not 1.15.
    wfact = 1.07
    # We want to stretch the PSF in the dispersion direction
    if grism_obs:
        scale = (1,wfact) if 'GRISM0' in pupil else (wfact,1)
        for i,im in enumerate(images):
            im_scale = frebin(im, scale=scale)
            images[i] = pad_or_cut_to_size(im_scale, im.shape)
    
    # Turn results into an numpy array (npsf,nx,ny)
    #   Or is it (npsf,ny,nx)? Depends on WebbPSF's coord system...
    images = np.array(images)

###     # Simultaneous polynomial fits to all pixels using linear least squares
###     # 7th-degree polynomial seems to do the trick
###     #ndeg = 8 # 7+1
###     x = waves
###     #a = np.vstack((np.ones(npsf), x, x**2, x**3, x**4, x**5, x**6, x**7)).T
###     a = np.array([x**num for num in range(ndeg+1)]).T
###     #b = images.reshape([-1,npsf]).T
###     b = images.reshape([npsf,-1])
###     coeff_all, _, _, _ = np.linalg.lstsq(a, b)
### 
###     # The number of pixels to span spatially
###     fov_pix_over = fov_pix * oversample	
###     coeff_all = coeff_all.reshape([ndeg+1,fov_pix_over,fov_pix_over])
    
    # Simultaneous polynomial fits to all pixels using linear least squares
    # 7th-degree polynomial seems to do the trick
    coeff_all = jl_poly_fit(waves, images, ndeg)

    if save:
        np.save(save_name, coeff_all)

    return coeff_all


def gen_image_coeff(filter_or_bp, pupil=None, mask=None, module='A', 
    sp_norm=None, coeff=None, fov_pix=11, oversample=4, 
    return_oversample=False, **kwargs):
    """
    Create an image (direct, coronagraphic, grism, or DHS) based on a set of
    instrument parameters and PSF coefficients. The image is noiseless and
    doesn't take into account any non-linearity or saturation effects, but is
    convolved with the instrument throughput. Pixel values are in counts/sec.
    The result is effectively an idealized slope image.

    If no spectral dispersers (grisms or DHS), then this returns a single
    image or list of images if sp_norm is a list of spectra.

    Parameters
    -------------------
    filter_or_bp : Either the name of a filter or a pre-computed Pysynphot bandpass.
    pupil : NIRCam pupil elements such as grisms or lyot stops
    mask : Specify the coronagraphic occulter (spots or bar)
    module : 'A' or 'B'
    sp_norm : A normalized Pysynphot spectrum to generate image. If not specified, 
        the default is flat in phot lam (equal number of photons per spectral bin).
        The default is normalized to produce 1 count/sec within that bandpass,
        assuming the telescope collecting area. Coronagraphic PSFs will further
        decrease this flux.

    coeff : A cube of polynomial coefficients for generating PSFs. This is
        generally oversampled and has the shape: 
    
            [fov_pix*oversample, fov_pix*oversample, deg]
    
        If not set, this this will be calculated using the psf_coeff() function.
    fov_pix : Number of detector pixels in the image coefficient and PSF.
    oversample : Factor of oversampling of detector pixels.

    return_oversample: If True, then also returns the oversampled version of the PSF

    Keyword Args
    -------------------
    Allows the user to pass additional (optional) arguments to:
        psf_coeff   - npsf and ndeg
        read_filter - ND_acq
    """

    # Get filter throughput and create bandpass 
    if isinstance(filter_or_bp, six.string_types):
        filter = filter_or_bp
        bp = read_filter(filter, pupil=pupil, mask=mask, module=module, **kwargs)
    else:
        bp = filter_or_bp
        filter = bp.name
    
    if coeff is None:
        coeff = psf_coeff(bp, pupil, mask, module, fov_pix=fov_pix, oversample=oversample, **kwargs)
    
    waveset = np.copy(bp.wave)
    # For generating the PSF, let's save some time and memory by not using
    # ever single wavelength in the bandpass. Instead, cut by 1/3
    if coeff.shape[-1]>2000:
        binsize = 7
    elif coeff.shape[-1]>1000:
        binsize = 5
    elif coeff.shape[-1]>700:
        binsize = 3

    if coeff.shape[-1]>700:
        excess = waveset.size % binsize
        waveset = waveset[:waveset.size-excess]
        waveset = waveset.reshape(-1,binsize) # Reshape
        waveset = waveset[:,binsize//2] # Use the middle values
        waveset = np.concatenate(([bp.wave[0]],waveset,[bp.wave[-1]]))

    wgood = waveset / 1e4
    w1 = wgood.min(); w2 = wgood.max()
    wrange = w2 - w1

    # Flat spectrum with equal photon flux in each spectal bin
    if sp_norm is None:
        sp_flat = S.ArraySpectrum(waveset, 0*waveset + 10.)
        sp_flat.name = 'Flat spectrum in photlam'

        # Bandpass unit response is the flux (in flam) of a star that 
        # produces a response of one count per second in that bandpass
        sp_norm = sp_flat.renorm(bp.unit_response(), 'flam', bp)
    

    #elif isinstance(sp_norm, dict): # dictionary with position offsets?
    # TODO: Want to deal with possibility of multiple spectra passed through as
    # a dictionary???

    # Make sp_norm a list of spectral objects if it already isn't
    if not isinstance(sp_norm, list): sp_norm = [sp_norm]
    nspec = len(sp_norm)

    # Set up an observation of the spectrum using the specified bandpass
    # Use the bandpass wavelength set to bin the fluxes
    obs_list = [S.Observation(sp, bp, binset=waveset) for sp in sp_norm]
    for obs in obs_list: obs.convert('counts')

    # Create a PSF for each wgood wavelength
    psf_fit = jl_poly(wgood, coeff)

    # Multiply each monochromatic PSFs by the binned e/sec at each wavelength
    # Array broadcasting: [nx,ny,nwave] x [0,0,nwave]
    # Do this for each spectrum/observation
    psf_list = [psf_fit*obs.binflux for obs in obs_list]

    # The number of pixels to span spatially
    fov_pix = int(fov_pix)
    oversample = int(oversample)
    fov_pix_over = int(fov_pix * oversample)

    # Grism spectroscopy
    if (pupil is not None) and ('GRISM' in pupil):
        # spectral resolution in um/pixel
        # res is in pixels per um and dw is inverse
        res, dw = grism_res(pupil, module)

        # Number of real pixels that spectra will span
        npix_spec = int(wrange // dw + 1 + fov_pix)
        npix_spec_over = int(npix_spec * oversample)

        spec_list = []
        for psf_fit in psf_list:  
            # If GRISM90 (along columns) rotate by 90 deg CW (270 deg CCW)
            if 'GRISM90' in pupil:
                psf_fit = np.rot90(psf_fit, k=3) # Rotate PSFs by 3*90 deg CCW

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

            # Rebin ovesampled spectral image to real pixels
            spec_list.append(poppy.utils.krebin(spec_over, (fov_pix,npix_spec)))

        # Wavelength solutions
        dw_over = dw/oversample
        w1_spec = w1 - dw_over*fov_pix_over/2
        wspec_over = np.arange(npix_spec_over)*dw_over + w1_spec
        wspec = wspec_over.reshape((npix_spec,-1)).mean(axis=1)

        if nspec == 1: spec_list = spec_list[0]
        # Return list of wavelengths for each horizontal pixel
        # as well as spectral image
        if return_oversample:
            return (wspec, spec_list), (wspec_over, spec_over)
        else:
            return (wspec, spec_list)
    
    # DHS spectroscopy
    elif (pupil is not None) and ('DHS' in pupil):
        raise NotImplementedError('DHS has yet to be fully included')

    # Imaging
    else:
        # Create source image slopes (no noise)
        data_list = []
        for psf_fit in psf_list:
            data_over = psf_fit.sum(axis=2)
            data_over[data_over<__epsilon] = 0
            data_list.append(poppy.utils.krebin(data_over, (fov_pix,fov_pix)))
        
        if nspec == 1: data_list = data_list[0]
        if return_oversample:
            return data_list, data_over
        else:
            return data_list



###########################################################################
#
#    Sensitivities and Saturation Limits
#
###########################################################################


def channel_select(bp):
    if bp.avgwave()/1e4 < 2.3:
        pix_scale = pixscale_SW # pixel scale (arcsec/pixel)
        idark = 0.003      # dark current (e/sec)
        pex = (1.0,5.0)
    else:
        pix_scale = pixscale_LW
        idark = 0.03
        pex = (1.5,10.0)

    return (pix_scale, idark, pex)

def grism_res(pupil='GRISM', module='A'):
    """
    Based on the pupil input and module, return the spectral
    dispersion and resolution as a tuple (res, dw).
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
    # Spectral resolution in um/pixel
    dw = 1. / res

    return (res, dw)


def get_SNR(filter_or_bp, pupil=None, mask=None, module='A', pix_scale=None,
    sp=None, tf=10.737, ngroup=2, nf=1, nd2=0, nint=1,
    coeff=None, fov_pix=11, oversample=4, quiet=True, **kwargs):
    """
    Obtain the SNR of an input source spectrum with specified instrument setup.
    This is simply a wrapper for bg_sensitivity(forwardSNR=True).
    """

    return bg_sensitivity(filter_or_bp, \
        pupil=pupil, mask=mask, module=module, pix_scale=pix_scale, \
        sp=sp, tf=tf, ngroup=ngroup, nf=nf, nd2=ngroup, nint=nint, \
        coeff=coeff, fov_pix=fov_pix, oversample=oversample, \
        quiet=quiet, forwardSNR=True, **kwargs)

def _mlim_helper(sub_im, mag_norm=10, mag_arr=np.arange(5,35,1), 
    nsig=5, nint=1, snr_fact=1, forwardSNR=False, **kwargs):
    """
    Helper function for determining grism sensitivities
    """

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
    coeff=None, fov_pix=11, oversample=4, quiet=True, forwardSNR=False, 
    offset_r=0, offset_theta=0, return_image=False, image=None, 
    dw_bin=None, ap_spec=None, rad_EE=None, **kwargs):
    """
    Estimates the sensitivity for a set of instrument parameters.
    By default, a flat spectrum is convolved with the specified bandpass.
    For imaging, this function also returns the surface brightness sensitivity.

    The number of photo-electrons are computed for a source at some magnitude
    as well as the noise from the detector readout and some average zodiacal 
    background flux. Detector readout noise follows an analytical form that
    matches extensive long dark observations during cryo-vac testing.

    This function returns the n-sigma background limit in units of uJy (unless
    otherwise specified; valid units can be found on the Pysynphot webpage).

    For imaging, a single value is given assuming aperture photometry with a
    radius of ~1 FWHM rounded to the next highest integer pixel (or 2.5 pixels,
    whichever is larger). For spectral observtions, this function returns an 
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
        generally oversampled and has the shape: 

            [fov_pix*oversample, fov_pix*oversample, deg]

        If not set, this this will be calculated from fov_pix, oversample,
        and npsf by generating a number of webbPSF images within the bandpass
        and fitting a high-order polynomial.
    fov_pix      : Number of detector pixels in the image coefficient and PSF.
    oversample   : Factor of oversampling of detector pixels.
    offset_r     : Radial offset of the target from center.
    offset_theta : Position angle for that offset, in degrees CCW (+Y).

    Misc.
    -------------------
    image        : Explicitly pass image data rather than calculating from coeff.
    return_image : Instead of calculating sensitivity, return the image calced from coeff.
    rad_EE       : Extraction aperture radius (in pixels) for imaging mode.
    dw_bin       : Delta wavelength to calculate spectral sensitivities (grisms & DHS).
    ap_spec      : Instead of dw_bin, specify the spectral extraction aperture in pixels.
                   Takes priority over dw_bin. Value will get rounded up to nearest int.

    Keyword Args
    -------------------
    **kwargs : Allows the user to pass additional (optional) arguments:
        zodi_spec   - zfact, locstr, year, day
        pix_noise   - rn, ktc, idark, and p_excess
        psf_coeff   - npsf and ndeg
        read_filter - ND_acq
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
    # Generate the PSF image for analysis
    t0 = time.time()
    # This process can take a while if being done over and over again. 
    # Let's provide the option to skip this with a pre-generated image.
    # Remember, this is for a very specific NORMALIZED spectrum
    if image is None:
        image = gen_image_coeff(bp, pupil, mask, module, sp_norm, coeff, fov_pix, oversample, 
            offset_r=offset_r, offset_theta=offset_theta, **kwargs)
    t1 = time.time()
    _log.debug('fov_pix={0}, oversample={1}'.format(fov_pix,oversample))
    _log.debug('Took %.2f seconds to generate images' % (t1-t0))
    if return_image:
        return image
            
    # Cosmic Ray Loss (JWST-STScI-001721)
    # SNR with cosmic ray events depends directly on ramp integration time
    tint = (ngroup*nf + (ngroup-1)*nd2) * tf
    snr_fact = 1.0 - tint*6.7781e-5

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

        # Create HDU list to pass to radial_profile()
        flux_hdu = fits.PrimaryHDU(image)
        flux_hdu.header.append(('PIXELSCL', pix_scale))
        flux_hdu_list = fits.HDUList(flux_hdu)
        radius, _, EE_flux = radial_profile(flux_hdu_list, EE=True, center=center)
        rad_pix = radius / pix_scale + 0.5

        # How many pixels do we want?
        fwhm_pix = 1.2 * efflam * 0.206265 / 6.5 / pix_scale
        if rad_EE is None:
            rad_EE = np.max([fwhm_pix,2.5])
        npix_EE = np.pi * rad_EE**2

        ####TEMPORARY
# 		tgrp = (nf + nd2) * tf
# 		tint = ngroup*tgrp - nd2*tf #(ngroup*nf + (ngroup-1)*nd2) * tf
# 		texp_orig = nint*tint
# 		# Cosmic ray influence
# 		texp = texp_orig#*(1. - 0.32*tgrp/tint)*(1. - tf/tint)
# 		idark = 0.01
# 		rn = 15.0
# 		
# 		var_const = fzodi_pix/tint + idark/tint + rn**2 / tint**2
# 		_log.debug('tgrp:{0:.2f}, tint:{1:.2f}, texp_orig:{2:.2f}, texp:{3:.2f}'\
# 			.format(tgrp,tint,texp_orig,texp))
# 		_log.debug('idark:{0:.4f}, rn:{1:.1f}'.format(idark,rn))
# 		_log.debug('Extraction radius: {0:.2f} pixels'.format(rad_EE))
        ####TEMPORARY

        # For surface brightness sensitivity (extended object)
        # Assume the fiducial (sp_norm) to be in terms of mag/arcsec^2
        # Multiply countrate() by pix_scale^2 to get in terms of per pixel (area)
        # This is the count rate per pixel for the fiducial starting point
        image_ext = obs.countrate() * pix_scale**2 # e-/sec/pixel
        #print(image_ext)
    
        if forwardSNR:
            im_var = pix_noise(ngroup=ngroup, nf=nf, nd2=nd2, tf=tf, 
                fzodi=fzodi_pix, fsrc=image, **kwargs)**2
            # Create HDU list to pass to radial_profile()
            var_hdu = fits.PrimaryHDU(im_var)
            var_hdu.header.append(('PIXELSCL', pix_scale))
            var_hdu_list = fits.HDUList(var_hdu)
            _, _, EE_var = radial_profile(var_hdu_list, EE=True, center=center)
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

                    # Create HDU list to pass to radial_profile()
                    var_hdu = fits.PrimaryHDU(im_var)
                    var_hdu.header.append(('PIXELSCL', pix_scale))
                    var_hdu_list = fits.HDUList(var_hdu)
                    _, _, EE_var = radial_profile(var_hdu_list, EE=True, center=center)
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


def sat_limit_webbpsf(filter_or_bp, pupil=None, mask=None, module='A', 
    sp=None, bp_lim=None, int_time=21.47352, full_well=81e3, well_frac=0.8, 
    coeff=None, fov_pix=11, oversample=4, quiet=True, units='vegamag', 
    offset_r=0, offset_theta=0, **kwargs):
    """
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
    full_well : Detector well level in electrons (by default, 80% of this number
        will be considered the saturation level).
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
    **kwargs : Allows the user to pass additional (optional) arguments:
        psf_coeff   - npsf and ndeg
        read_filter - ND_acq
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
    result = gen_image_coeff(bp, pupil, mask, module, sp_norm, coeff, fov_pix, oversample, 
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
        _log.debug('Approximate Time to {1:.2f} of Saturation: {0:.2f} sec'.\
            format(sat_time,well_frac))

        # Magnitude necessary to saturate a given pixel
        ratio = int_time/sat_time
        sat_mag = mag_norm + 2.5*np.log10(ratio)
    
        # Convert to desired unit
        sp_temp = sp.renorm(sat_mag, 'vegamag', bp_lim)
        obs_temp = S.Observation(sp_temp, bp_lim, binset=bp_lim.wave)
        sat_mag = obs_temp.effstim(units)

        if not quiet:
            if bp_lim.name == bp.name:
                print('{} Saturation Limit assuming {} source: {:.2f} {}'.\
                    format(bp_lim.name, sp_norm.name, sat_mag, units) )
            else:
                print('{} Saturation Limit for {} assuming {} source: {:.2f} {}'.\
                    format(bp_lim.name, bp.name, sp_norm.name, sat_mag, units) )

        return {'satmag':sat_mag, 'units':units, 'Spectrum':sp_norm.name, 
            'bp_lim':bp_lim.name}


def pix_noise(ngroup=2, nf=1, nd2=0, tf=10.737, rn=15.0, ktc=29.0, p_excess=(0,0),
    fsrc=0.0, idark=0.003, fzodi=0, fbg=0, ideal_Poisson=False, **kwargs):
    """
    Theoretical noise calculation of a generalized MULTIACCUM ramp in terms of e-/sec.
    Includes flat field errors from JWST-CALC-003894.

    Parameters
    ===========
    n (int) : Number of groups in integration ramp
    m (int) : Number of frames in each group
    s (int) : Number of dropped frames in each group
    tf (float) : Frame time
    rn (float) : Read Noise per pixel
    ktc (float) : kTC noise only valid for single frame (n=1)
    p_excess: An array or list of two elements that holding the
        parameters that describe the excess variance observed in
        effective noise plots. By default these are both 0.
        Recommended values are [1.0,5.0] or SW and [1.5,10.0] for LW.

    fsrc  (float) : Flux of source in e-/sec/pix
    idark (float) : Dark current in e-/sec/pix
    fzodi (float) : Zodiacal light emission in e-/sec/pix
    fbg   (float) : Any additional background (telescope emission or scattered light?)

    ideal_Poisson : If set to True, use total signal for noise estimate,
                    otherwise MULTIACCUM equation is used?

    Various parameters can either be single values or numpy arrays.
    If multiple inputs are arrays, make sure their array sizes match.
    Variables that need to have the same array sizes (or a single value):
        - n, m, s, & tf
        - rn, idark, ktc, fsrc, fzodi, & fbg

    Array broadcasting also works:
        For Example
        n = np.arange(50)+1 # An array of groups to test out

        # Create 2D Gaussian PSF with FWHM = 3 pix
        npix = 20 # Number of pixels in x and y direction
        x = np.arange(0, npix, 1, dtype=float)
        y = x[:,np.newaxis]
        x0 = y0 = npix // 2 # Center position
        fwhm = 3.0
        fsrc = np.exp(-4*np.log(2.) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
        fsrc /= fsrc.max()
        fsrc *= 10 # Total source counts/sec (arbitrarily scaled)
        fsrc = fsrc.reshape(npix,npix,1) # Necessary for broadcasting

        # Represents pixel array w/ different RN/pix
        rn = np.ones([npix,npix,1])*15. 
        # Results is a (20x20)x50 showing the noise in e-/sec/pix at each group
        noise = pix_noise(ngroup=n, rn=rn, fsrc=fsrc) 
    """

    n = np.array(ngroup)
    m = np.array(nf)
    s = np.array(nd2)
    tf = np.array(tf)
    max_size = np.max([n.size,m.size,s.size,tf.size])
    if n.size  != max_size: n  = n.repeat(max_size)
    if m.size  != max_size: m  = m.repeat(max_size)
    if s.size  != max_size: s  = s.repeat(max_size)
    if tf.size != max_size: tf = tf.repeat(max_size)

    # Total flux (e-/sec/pix)
    ftot = fsrc + idark + fzodi + fbg

    # Special case if n=1
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
    
    # Noise floor
    var = var_rn + var_poisson + var_ex
    sig = np.sqrt(var)

    # Noise in e-/sec
    noise = sig / tint
    #print(ind_n1_all.shape,noise.shape,noise_n1.shape)
    if (n==1).any():
        noise[ind_n1_all] = noise_n1[ind_n1_all]

    # Include flat field noise
    # JWST-CALC-003894
    noise_ff = 1E-4 # Uncertainty in the flat field
    factor = 1 + noise_ff*np.sqrt(ftot)
    noise *= factor

    return noise


###########################################################################
#
#    Image Manipulation and Maths
#
###########################################################################

def image_rescale(HDUlist_or_filename, args_in, args_out, cen_star=True):
    """
    Scale the flux and rebin the image with a give pixel scale and distance
    to some output pixel scale and distance. The object's physical units (AU)
    are assumed to be constant, so the angular size changes if the distance
    to the object changes.

    IT IS RECOMMENDED THAT UNITS BE IN PHOTONS/SEC/PIXEL (not mJy/arcsec)

    Parameters
    ==========
    args_in  : Two parameters consisting of the input image pixel scale and distance
        assumed to be in units of arcsec/pixel and parsecs, respectively
    args_out : Same as above, but the new desired outputs
    cen_star : Is the star placed in the central pixel?

    Returns an HDUlist of the new image
    """
    im_scale, dist = args_in
    pixscale_out, dist_new = args_out

    if isinstance(HDUlist_or_filename, six.string_types):
        hdulist = fits.open(HDUlist_or_filename)
    elif isinstance(HDUlist_or_filename, fits.HDUList):
        hdulist = HDUlist_or_filename
    else:
        raise ValueError("Input must be a filename or HDUlist")

    # By moving the image closer, we increased the flux (inverse square law)
    image = (hdulist[0].data) * (dist / dist_new)**2
    #hdulist.close()

    # We also increased the angle that the image subtends
    # So, each pixel would have a large angular size
    # New image scale in arcsec/pixel
    imscale_new = im_scale * dist / dist_new

    # Before rebinning, we want the flux in the central pixel to
    # always be in the central pixel (the star). So, let's save
    # and remove that flux then add back after the rebinning.
    if cen_star:
        mask_max = image==image.max()
        star_flux = image[mask_max][0]
        image[mask_max] = 0

    # Rebin the image to get a pixel scale that oversamples the detector pixels
    fact = imscale_new / pixscale_out
    image_new = frebin(image, scale=fact)

    # Restore stellar flux to the central pixel.
    ny,nx = image_new.shape
    if cen_star:
        image_new[ny//2, nx//2] += star_flux

    hdu_new = fits.PrimaryHDU(image_new)
    hdu_new.header = hdulist[0].header.copy()
    hdulist_new = fits.HDUList([hdu_new])
    hdulist_new[0].header['PIXELSCL'] = (pixscale_out, 'arcsec/pixel')
    hdulist_new[0].header['DISTANCE'] = (dist_new, 'parsecs')

    return hdulist_new


def scale_ref_image(im1, im2, mask=None, smooth_imgs=False):
    """
    Find value to scale a reference image by minimizing residuals.
    
    Inputs
    ======
    im1 - Science star observation.
    im2 - Reference star observation.
    mask - Use this mask to exclude pixels for performing standard deviation.
           Boolean mask where True is included and False is excluded
    """
    
    # Mask for generating standard deviation
    if mask is None:
        mask = np.ones(im1.shape, dtype=np.bool)
    nan_mask = ~(np.isnan(im1) | np.isnan(im2))
    mask = (mask & nan_mask)

    # Spatial averaging to remove bad pixels
    if smooth_imgs:
        im1_smth = []#np.zeros_like(im1)
        im2_smth = []#np.zeros_like(im2)
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                im1_smth.append(fshift(im1, i, j))
                im2_smth.append(fshift(im2, i, j))

        im1 = np.nanmedian(np.array(im1_smth), axis=0)
        im2 = np.nanmedian(np.array(im2_smth), axis=0)

    ind = np.where(im1==im1[mask].max())
    ind = [ind[0][0], ind[1][0]]

    # Initial Guess
    scl = np.nanmean(im1[ind[0]-3:ind[0]+3,ind[1]-3:ind[1]+3]) / \
          np.nanmean(im2[ind[0]-3:ind[0]+3,ind[1]-3:ind[1]+3])
          
    # Wider range
    # Check a range of scale values
    # Want to minimize the standard deviation of the differenced images
    scl_arr = np.linspace(0.2*scl,2*scl,10)
    mad_arr = []
    for val in scl_arr:
        diff = im1 - val*im2
        mad_arr.append(robust.medabsdev(diff[mask]))
    mad_arr = np.array(mad_arr)
    scl = scl_arr[mad_arr==mad_arr.min()][0]

    # Check a range of scale values
    # Want to minimize the standard deviation of the differenced images
    scl_arr = np.linspace(0.85*scl,1.15*scl,50)
    mad_arr = []
    for val in scl_arr:
        diff = im1 - val*im2
        mad_arr.append(robust.medabsdev(diff[mask]))
    mad_arr = np.array(mad_arr)

    #plt.plot(scl_arr,mad_arr)
    return scl_arr[mad_arr==mad_arr.min()][0]





def dist_image(image, pixscale=None, center=None, return_theta=False):
    """
    Returns radial distance in units of pixels, unless pixscale is specified.
    Use the center keyword to specify the position (in pixels) to measure from.
    If not set, then the center of the image is used.

    return_theta will also return the angular position of each pixel relative 
    to the specified center
    
    center should be entered as (x,y)
    """
    y, x = np.indices(image.shape)
    if center is None:
        center = tuple((a - 1) / 2.0 for a in image.shape[::-1])
    x = x - center[0]
    y = y - center[1]

    rho = np.sqrt(x**2 + y**2)
    if pixscale is not None: rho *= pixscale

    if return_theta:
        return rho, np.arctan2(-x,y)*180/np.pi
    else:
        return rho

def xy_to_rtheta(x, y):
    """
    Input (x,y) coordinates and return polar cooridnates that use
    the WebbPSF convention (theta is CCW of +Y)
    
    Input can either be a single value or numpy array.
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(-x,y)*180/np.pi

    if np.size(r)==1:
        if np.abs(x) < __epsilon: x = 0
        if np.abs(y) < __epsilon: y = 0
    else:
        r[np.abs(r) < __epsilon] = 0
        theta[np.abs(theta) < __epsilon] = 0

    return r, theta

def rtheta_to_xy(r, theta):
    """
    Input polar cooridnates (WebbPSF convention) and return Carteesian coords
    in the imaging coordinate system (as opposed to RA/DEC)

    Input can either be a single value or numpy array.

    r     : Radial offset from the center in pixels
    theta : Position angle for offset in degrees CCW (+Y).
    """
    x = -r * np.sin(theta*np.pi/180.)
    y =  r * np.cos(theta*np.pi/180.)

    if np.size(x)==1:
        if np.abs(x) < __epsilon: x = 0
        if np.abs(y) < __epsilon: y = 0
    else:
        x[np.abs(x) < __epsilon] = 0
        y[np.abs(y) < __epsilon] = 0

    return x, y

###########################################################################
#
#    Coordinate Systems
#
###########################################################################

def det_to_V2V3(image, detid):
    """
    Reorient image from detector coordinates to V2/V3 coordinate system.
    This places +V3 up and +V2 to the LEFT. Detector pixel (0,0) is assumed 
    to be in the bottom left. For now, we're simply performing axes flips. 
    """
    
    # Check if SCA ID (481-489) where passed through detname rather than A1-B5
    try:
        detid = int(detid)
    except ValueError:
        detname = detid
    else:
        scaids = {481:'A1', 482:'A2', 483:'A3', 484:'A4', 485:'A5',
                  486:'B1', 487:'B2', 488:'B3', 489:'B4', 490:'B5'}
        detname = scaids[detid]
    
    xflip = ['A1','A3','A5','B2','B4']
    yflip = ['A2','A4','B1','B3','B5']
    
    for s in xflip:
        if detname in s:
            image = image[:,::-1] 
    for s in yflip:
        if detname in s:
            image = image[::-1,:] 
    
    return image
    
def V2V3_to_det(image, detid):
    """
    Reorient image from V2/V3 coordinates to detector coordinate system.
    Assumes +V3 up and +V2 to the LEFT. The result plances the detector
    pixel (0,0) in the bottom left. For now, we're simply performing 
    axes flips.
    """
    
    # Flips occur along the same axis and manner as in det_to_V2V3()
    return det_to_V2V3(image, detid)
    

###########################################################################
#
#    Pysynphot Spectrum Wrappers
#
###########################################################################


def bin_spectrum(sp, wave, waveunits='um'):
    """
    Rebin a Pysynphot spectrum to a lower wavelenght grid.
    This function first converts the input spectrum to units
    of photlam then combines the photon flux onto the 
    desired wavelength bin
    
    Output spectrum units are the same as the input spectrum.
    
    sp        - Pysynphot spectrum to rebin.
    wave      - Wavelength grid to rebin onto.
    waveunits - Units of wave input. Must be recognizeable by Pysynphot.
    
    Returns rebinned Pysynphot spectrum in same units as input spectrum.
    """

    waveunits0 = sp.waveunits
    fluxunits0 = sp.fluxunits

    # Convert wavelength of input spectrum to desired output units
    sp.convert(waveunits)
    # We also want input to be in terms of photlam
    sp.convert('photlam')

    edges = S.binning.calculate_bin_edges(wave)
    indices = np.searchsorted(sp.wave, edges)
    i1_arr = indices[:-1]
    i2_arr = indices[1:]

    # This assumes the original wavelength grid is uniform
    binflux = np.empty(shape=wave.shape, dtype=np.float64)
    for i in range(len(wave)):
        i1 = i1_arr[i]
        i2 = i2_arr[i]
        binflux[i] = sp.flux[i1:i2].sum() / (i2-i1)
    
    sp2 = S.ArraySpectrum(wave, binflux, waveunits=waveunits, fluxunits='photlam')
    sp2.convert(waveunits0)
    sp2.convert(fluxunits0)
    
    # Put back units of original input spectrum
    sp.convert(waveunits0)
    sp.convert(fluxunits0)

    return sp2



def stellar_spectrum(sptype, *renorm_args, **kwargs):
    """
    Get Pysynphot Spectrum object from a user-friendly spectral type string.

    Similar to specFromSpectralType() in WebbPSF/Poppy, this function uses
    a fixed dictionary to determine an appropriate spectral model using the
    Phoenix grid in the Pysynphot dataset (CDBS). However, instead of using
    a fixed dictionary where certain spectral types are invalid, this function
    interpolates the effective temperature, metallicity, and log g values if
    the input spectral type is not found.

    You can also specify renormalization arguments to pass to sp.renorm. The
    order (after `sptype`) should be (value, units, bandpass):
        ie., sp = stellar_spectrum('G2V', 10, 'vegamag', bp)
    
    Flat spectrum (in photlam) are also allowed via the 'flat' string.
    
    Use catname='ck04models' keyword for ck04 models
    
    """

    catname = kwargs.get('catname')
    if catname is None: catname = 'phoenix'
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
        "G5V": (5750, 0.0, 4.5),
        "G8V": (5500, 0.0, 4.5),
        "K0V": (5250, 0.0, 4.5),
        "K2V": (4750, 0.0, 4.5),
        "K5V": (4250, 0.0, 4.5),
        "K7V": (4000, 0.0, 4.5),
        "M0V": (3750, 0.0, 4.5),
        "M2V": (3500, 0.0, 4.5),
        "M5V": (3500, 0.0, 5.0),
        "M9V": (3000, 0.0, 5.0),    # Bracketing for interpolation
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

    # First test if the user wants a flat spectrum (in photlam)
    if 'flat' in sptype.lower():
        waveset = S.refs._default_waveset
        sp = S.ArraySpectrum(waveset, 0*waveset + 10.)
        sp.name = 'Flat spectrum in photlam'
    elif sptype in sptype_list:
        v0,v1,v2 = lookuptable[sptype]
    
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
        
        if ('ck04models' in catname.lower()) and (v0<3500): v0 = 3500
        sp = S.Icat(catname, v0, v1, v2)
        sp.name = sptype
        
    #print(int(v0),v1,v2)

    # Renormalize if those args exist
    if len(renorm_args) > 0:
        sp_norm = sp.renorm(*renorm_args)
        sp_norm.name = sp.name
        sp = sp_norm

    return sp

        
def zodi_spec(zfact=None, locstr=None, year=None, day=None, **kwargs):
    """
    Create a spectrum of the zodiacal light emission in order to estimate the
    in-band sky background flux. This is simply the addition of two blackbodies
    at T=5300K (solar scattered light) and T=282K (thermal dust emission)
    that have been scaled to match the literature flux values.

    In reality, the intensity of the zodiacal dust emission varies as a
    function of viewing position. In this case, we have added the option
    to scale the zodiacal level (or each component individually) by some
    user-defined factor 'zfact'. The user can set zfact as a scalar in order
    to scale the entire spectrum. If defined as a list, tuple, or np array,
    then the each component gets scaled where T=5300K corresponds to the first
    elements and T=282K is the second element of the array.

    Output is a Pysynphot spectrum with default units of flam (erg/s/cm^2/A/sr).
    Note: Pysynphot doesn't recognize that it's per steradian, but we must keep 
    that in mind when integrating the flux per pixel.

    Representative values for zfact:
        0.0 - No zodiacal emission
        1.0 - Minimum zodiacal emission from JWST-CALC-003894 (Figure 2-2)
        1.2 - Required NIRCam performance
        2.5 - Average (default)
        5.0 - High
        10. - Maximum
    
    Added the ability to query the Euclid background model for a specific
    location and observing time. The two blackbodies will be scaled to the
    1.0 and 5.5 um emission. See the help website for more details:
        http://irsa.ipac.caltech.edu/applications/BackgroundModel/docs/dustProgramInterface.html
        locstr - Object name, RA/DEC in decimal degrees or sexigesimal input
        year   - Year of observation
        day    - Day of observation
        
    """

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
    """
    Queries the IPAC Euclid Background Model at
    http://irsa.ipac.caltech.edu/applications/BackgroundModel/
    in order to get date and position-specific zodiacal dust emission.

    The program relies on urllib2 to download the page in XML format.
    However, the website only allows single wavelength queries, so
    this program implements a multithreaded procedure to query
    multiple wavelengths simultaneously. However, due to the nature
    of urllib2 library, only so many requests are allowed to go out
    at a time, so this process can take some time to complete.
    Testing shows about 500 wavelengths in 10 seconds as a rough
    ballpark.

    Recommended to grab only a few wavelengths and use for normalization
    purposes. 
    """

    from urllib2 import urlopen
    import xmltodict
    from multiprocessing.pool import ThreadPool

    def fetch_url(url):
        """
        Need to add error handling.
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

# Class for reading in planet spectra
class planets_sb12(object):
    """
    Exoplanet spectrum from Spiegel & Burrows (2012)

    This contains 1680 files, one for each of 4 atmosphere types, each of
    15 masses, and each of 28 ages.  Wavelength range of 0.8 - 15.0 um at
    moderate resolution (R ~ 204).

    The flux in the source files are at 10 pc. If the distance is specified,
    then the flux will be scaled accordingly. This is also true if the distance
    is changed by the user. All other properties (atmo, mass, age, entropy) are 
    not adjustable once loaded.

    Arguments:
        atmo: A string consisting of one of four atmosphere types:
            hy1s = hybrid clouds, solar abundances
            hy3s = hybrid clouds, 3x solar abundances
            cf1s = cloud-free, solar abundances
            cf3s = cloud-free, 3x solar abundances
        mass: Integer number 1 to 15 Jupiter masses.
        age: Age in millions of years (1-1000)
        entropy: Initial entropy (8.0-13.0) in increments of 0.25
        distance: Assumed distance in pc (default is 10pc)
        base_dir: Location of atmospheric model sub-directories.
    """

    base_dir = conf.PYNRC_PATH + 'spiegel/'

    def __init__(self, atmo='hy1s', mass=1, age=100, entropy=10.0, 
                 distance=10, base_dir=None, **kwargs):

        self._atmo = atmo
        self._mass = mass
        self._age = age
        self._entropy = entropy

        if base_dir is not None:
            self.base_dir = base_dir
        self.sub_dir = self.base_dir  + 'SB.' + self.atmo + '/'

        self.get_file()
        self.read_file()
        self.distance = distance

    def get_file(self):
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

    def read_file(self):
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
    def wave(self):
        return self._wave
    @property
    def waveunits(self):
        return self._waveunits

    @property
    def flux(self):
        return self._flux
    @property
    def fluxunits(self):
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
        """
        A string consisting of one of four atmosphere types:
            hy1s = hybrid clouds, solar abundances
            hy3s = hybrid clouds, 3x solar abundances
            cf1s = cloud-free, solar abundances
            cf3s = cloud-free, 3x solar abundances
        """
        return self._atmo
    @property
    def mass(self):
        """Jupiter masses"""
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
        w = self.wave; f = self.flux        
        name = (re.split('[\.]', self.file))[5:]        
        sp = S.ArraySpectrum(w, f, name=name, waveunits=self.waveunits, fluxunits=self.fluxunits)

        sp.convert(waveout)
        sp.convert(fluxout)

        return sp
        
# Turns out the paper is Spiegel & Burrows (2012), not 2011
class planets_sb11(planets_sb12):

    """
    Deprecated version of planets_sb12 class. Use that instead.
    """

    def __init__(self, *args, **kwargs):
                 
        _log.warning('planets_sb11 is depcrecated. Use planets_sb12 instead.')
        planets_sb12.__init__(self, *args, **kwargs)




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
    nwavelengths : int
        Number of wavelengths. Sets maximum # of processes.
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
#    OPD Stuff for High Contrast Imaging Models
#
###########################################################################



###########################################################################
#
#    Coronagraphic Mask Transmission
#
###########################################################################


def coron_trans(name, module='A', pixscale=None, fov=20):
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

    s = int(round(fov/pixscale))
    shape = (s,s)
    y, x = np.indices(shape, dtype=float)
    y -= shape[0] / 2.0
    x -= shape[1] / 2.0
    y,x = (pixscale * y, pixscale * x)

    if 'WB' in name:
        scalefact = (2 + (-x + 7.5) * 4 / 15).clip(2, 6)
        if name == 'MASKSWB': #np.abs(self.wavelength - 2.1e-6) < 0.1e-6:
            polyfitcoeffs = np.array([2.01210737e-04, -7.18758337e-03, 1.12381516e-01,
                                      -1.00877701e+00, 5.72538509e+00, -2.12943497e+01,
                                      5.18745152e+01, -7.97815606e+01, 7.02728734e+01])
            scalefact = scalefact[:, ::-1] # flip orientation left/right for SWB mask
        elif name == 'MASKLWB': #elif np.abs(self.wavelength - 4.6e-6) < 0.1e-6:
            polyfitcoeffs = np.array([9.16195583e-05, -3.27354831e-03, 5.11960734e-02,
                                      -4.59674047e-01, 2.60963397e+00, -9.70881273e+00,
                                      2.36585911e+01, -3.63978587e+01, 3.20703511e+01])

        sigmas = scipy.poly1d(polyfitcoeffs)(scalefact)
        sigmar = sigmas * np.abs(y)
        sigmar.clip(min=np.finfo(sigmar.dtype).tiny, max=2*np.pi, out=sigmar)
        transmission = (1 - (np.sin(sigmar) / sigmar) ** 2)
        #transmission[x==0] = 0
        woutside = np.where(np.abs(x) > 10)
        transmission[woutside] = 1.0

    else:
        r = np.sqrt(x ** 2 + y ** 2)
        sigmar = sigma * r
        sigmar.clip(np.finfo(sigmar.dtype).tiny, 2*np.pi, out=sigmar)  # avoid divide by zero -> NaNs
        transmission = (1 - (2 * scipy.special.jn(1, sigmar) / sigmar) ** 2)
        transmission[r==0] = 0   # special case center point (value based on L'Hopital's rule)


    # add in the ND squares
    x = x[::-1, ::-1]
    y = y[::-1, ::-1]
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
        woutside = np.where((x < -10) & (y < 11.5 ))
        transmission[woutside] = 0.0
    elif ((module=='A' and name=='MASK210R') or
          (module=='B' and name=='MASKSWB')):
        woutside = np.where((x > 10) & (y < 11.5))
        transmission[woutside] = 0.0
    woutside = np.where(y < -10)
    transmission[woutside] = 0.0

    # edge of mask itself
    wedge = np.where(( y > 11.5) & (y < 13))
    transmission[wedge] = 0.7

    return transmission


def build_mask(module='A', pixscale=0.03):
    """
    Return an image of the full coronagraphic mask layout for a given module.
    +V3 is up, and +V2 is to the left.
    """
    if module=='A':
        names = ['MASK210R', 'MASK335R', 'MASK430R', 'MASKSWB', 'MASKLWB']
    elif module=='B':
        names = ['MASKSWB', 'MASKLWB', 'MASK430R', 'MASK335R', 'MASK210R']
    allims = [coron_trans(name,module,pixscale) for name in names]

    return np.concatenate(allims, axis=1)




###########################################################################
#
#    Miscellaneous
#
###########################################################################

def nrc_header(det_class, filter=None, pupil=None, obs_time=None, header=None,
               DMS=True,targ_name=None):
    """
    Create a generic NIRCam FITS header from a detector_ops class.

    Parameters
    ===========
    filter (str) : Name of filter element.
    pupil  (str) : Name of pupil element.
    DMS    (bool): 
        Make the header in a format used by Data Management Systems
    
    obs_time (datetime): 
        Specifies when the observation was considered to be executed.
        If not specified, then it will choose the current time.
        This must be a datetime object:
            datetime.datetime(2016, 5, 9, 11, 57, 5, 796686)
        
    header (obj) : Can pass an existing header that will be updated.
        This has not been fully tested
               
    targ_name (str) :
        Standard astronomical catalog name for a target
        Otherwise, it will be UNKOWN
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
            exp_type = 'UNKOWN'
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
        hdr['TARG_RA']=  (80.4875, 'Target RA at mid time of exposure') #arbitrary position
        hdr['TARG_DEC']= (-69.498333, 'Target Dec at mid time of exposure') #arbitrary position
        
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
        hdr['EFFEXPTM']= (d.time_total_int, 'Effective exposure time (sec)')
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

    
    hdr['TSAMPLE'] = (tsample,           'Delta time between samples in microsec')
    hdr['TFRAME']  = (d.time_frame,   'Time in seconds between frames')
    hdr['TGROUP']  = (d.time_group,     'Delta time between groups')
    hdr['DRPFRMS1']= (ma.nd1, 'Number of frame skipped prior to first integration')
    hdr['GROUPGAP']= (ma.nd2, 'Number of frames skipped')
    hdr['DRPFRMS3']= (ma.nd3, 'Number of frames skipped between integrations')
    hdr['FRMDIVSR']= (ma.nf,  'Divisor applied to each group image')
    hdr['INTAVG']  = (1,   'Number of integrations averaged in one image')
    hdr['NRESETS1']= (1,   'Number of reset frames prior to first integration')
    hdr['NRESETS2']= (1,   'Number of reset frames between each integration')
    hdr['INTTIME'] = (d.time_int,   'Total integration time for one MULTIACCUM')
    hdr['EXPTIME'] = (d.time_exp,    'Exposure duration (seconds) calculated')
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
        hdr['SUBARRAY']= (subName,     'Detector subarray string')
    else:
        hdr['SUBARRAY']= (sub_bool,    'T if subarray used, F if not')
    
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



# Unused function
# def lazy_thunkif y(f):
#     """
#     Make a function immediately return a function of no args which, when called,
#     waits for the result, which will start being processed in another thread.
# 
#     This decorator will cause any function to, instead of running its code, 
#     start a thread to run the code, returning a thunk (function with no args) that 
#     waits for the function's completion and returns the value (or raises the exception).
# 
#     Useful if you have Computation A that takes x seconds and then uses Computation B, 
#     which takes y seconds. Instead of x+y seconds you only need max(x,y) seconds.
# 
#     Example:
# 
#     @lazy_thunkify
#     def slow_double(i):
#         print "Multiplying..."
#         time.sleep(5)
#         print "Done multiplying!"
#         return i*2
# 
#     def maybe_multiply(x):
#         double_thunk = slow_double(x)
#         print "Thinking..."
#         time.sleep(3)
#         time.sleep(3)
#         time.sleep(1)
#         if x == 3:
#             print "Using it!"
#             res = double_thunk()
#         else:
#             print "Not using it."
#             res = None
#         return res
# 
#     #both take 7 seconds
#     maybe_multiply(10)
#     maybe_multiply(3)
# 
#     """
#
#    import threading, functools, traceback
# 
#     @functools.wraps(f)
#     def lazy_thunked(*args, **kwargs):
#         wait_event = threading.Event()
# 
#         result = [None]
#         exc = [False, None]
# 
#         def worker_func():
#             try:
#                 func_result = f(*args, **kwargs)
#                 result[0] = func_result
#             except Exception as e:
#                 exc[0] = True
#                 exc[1] = sys.exc_info()
#                 print("Lazy thunk has thrown an exception (will be raised on thunk()):\n%s" % 
#                     (traceback.format_exc()))
#             finally:
#                 wait_event.set()
# 
#         def thunk():
#             wait_event.wait()
#             if exc[0]:
#                 raise exc[1][0], exc[1][1], exc[1][2]
# 
#             return result[0]
# 
#         threading.Thread(target=worker_func).start()
# 
#         return thunk
# 
#     return lazy_thunked