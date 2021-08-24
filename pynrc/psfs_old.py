from __future__ import absolute_import, division, print_function, unicode_literals

# The six library is useful for Python 2 and 3 compatibility
import six, os

# Import libraries
import numpy as np

import datetime, time
import sys, platform
import multiprocessing as mp
import traceback

import scipy
from scipy.interpolate import griddata, RegularGridInterpolator

from astropy.io import fits
import astropy.units as u

import logging
_log = logging.getLogger('pynrc')

from . import conf
from .logging_utils import setup_logging

from .nrc_utils import read_filter, S, grism_res
from .opds import opd_default, OPDFile_to_HDUList
from .maths.image_manip import fshift, frebin, pad_or_cut_to_size
from .maths.fast_poly import jl_poly_fit, jl_poly
from .maths.coords import Tel2Sci_info, NIRCam_V2V3_limits, dist_image

# Program bar
from tqdm.auto import trange, tqdm

__epsilon = np.finfo(float).eps

from webbpsf_ext.psfs import nproc_use, gen_image_from_coeff

###########################################################################
#
#    WebbPSF Stuff
#
###########################################################################

try:
    import webbpsf_ext
except ImportError:
    raise ImportError('webbpsf_ext is not installed. pyNRC depends on its inclusion.')
import webbpsf, poppy
from webbpsf.opds import OTE_Linear_Model_WSS

# Check that version meets minimum requirements
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if not on_rtd:
    _webbpsf_version_min = (0,9,0)
    _ = webbpsf.utils.get_webbpsf_data_path(_webbpsf_version_min)

# Set up some poppy and webbpsf defaults
# Turn off multiprocessing, which is faster now due to improved
# underlying vectorization of poppy and numpy. Swapping large
# amount of data between processes is now the bottleneck for mp (5/18/2020).
poppy.conf.use_multiprocessing = False 
# Only use this if you have the FFTW C library installed
# In general, numpy fft is actually pretty fast now, so default use_fftw=False
# It also doesn't play well with multiprocessing
poppy.conf.use_fftw = False

# If the machine has 2 or less CPU cores, then no mp
if mp.cpu_count()<3:
    poppy.conf.use_multiprocessing = False
# n_processes will be considered the max number of processors we use for multiprocessing
poppy.conf.n_processes = int(0.75 * mp.cpu_count()) #if poppy.conf.use_multiprocessing else 1

webbpsf.conf.default_output_mode = u'detector'

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
    # fov_pix_orig = fov_pix # Does calc_psf change fov_pix??
    try:
        hdu_list = inst.calc_psf(fov_pixels=fov_pix, oversample=oversample, monochromatic=w*1e-6,
                                 add_distortion=False, crop_psf=True)
        # Distortions are ignored here. It's preferred do perform these later.
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
    fov_pix=11, oversample=None, npsf=None, ndeg=None, nproc=None, 
    offset_r=None, offset_theta=None, jitter=None, jitter_sigma=0.007,
    tel_pupil=None, opd=None, wfe_drift=None, include_si_wfe=False,
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

    # Set the SIAF aperture name
    if apname is not None:
        inst.auto_aperturename = False
        inst.aperturename = apname

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
    _log.debug(inst.aperturename, inst.detector, inst.detector_position)

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
        opd_name = 'OPD from FITS HDUlist'
        opd_num = 0
    elif isinstance(opd, poppy.OpticalElement):
        # OTE Linear Model
        # No need to do anything
        pass
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

    # If there is wfe_drift, create a OTE Linear Model
    if (wfe_drift > 0):
        _log.debug('Performing WFE drift of {}nm'.format(wfe_drift))

        # OPD should already be an HDUList or OTE LM by now
        # If we want more realistic time evolution, then need to use
        # procedure in dev_utils/WebbPSF_OTE_LM.ipynb to create a time
        # series of OPDs then pass those OPDs directly to create unique PSFs
        if isinstance(opd, fits.HDUList):
            hdul = opd

            header = hdul[0].header

            header['ORIGINAL'] = (opd_name,   "Original OPD source")
            header['SLICE']    = (opd_num,    "Slice index of original OPD")
            header['WFEDRIFT'] = (wfe_drift, "WFE drift amount [nm]")

            name = 'Modified from ' + opd_name
            opd = OTE_Linear_Model_WSS(name=name, opd=hdul, opd_index=opd_num, transmission=inst.pupil)

        # Apply WFE drift to OTE Linear Model (Amplitude of frill drift)
        inst.pupilopd = opd
        inst.pupil = opd

        # Split WFE drift amplitude between three processes
        # 1) IEC Heaters; 2) Frill tensioning; 3) OTE Thermal perturbations
        # Give IEC heaters 1 nm 
        wfe_iec = 1 if np.abs(wfe_drift) > 2 else 0

        # Split remainder evenly between frill and OTE thermal slew
        wfe_remain_var = wfe_drift**2 - wfe_iec**2
        wfe_frill = np.sqrt(0.8*wfe_remain_var)
        wfe_therm = np.sqrt(0.2*wfe_remain_var)
        # wfe_th_frill = np.sqrt((wfe_drift**2 - wfe_iec**2) / 2)

        # Negate amplitude if supplying negative wfe_drift
        if wfe_drift < 0:
            wfe_frill *= -1
            wfe_therm *= -1
            wfe_iec *= -1

        # Apply IEC
        opd.apply_iec_drift(wfe_iec, delay_update=True)
        # Apply frill
        opd.apply_frill_drift(wfe_frill, delay_update=True)

        # Apply OTE thermal slew amplitude
        # This is slightly different due to how thermal slews are specified
        import astropy.units as u
        delta_time = 14*24*60 * u.min
        wfe_scale = (wfe_therm / 24)
        if wfe_scale == 0:
            delta_time = 0
        opd.thermal_slew(delta_time, case='BOL', scaling=wfe_scale)
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
    if nproc is None:
        nproc = nproc_use(fov_pix, oversample, npsf)
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
    elif isinstance(opd, poppy.OpticalElement):
        hdr['OPD'] = ('OTE Linear Model', 'Telescope OPD')
    else:
        hdr['OPD'] = ('UNKNOWN', 'Telescope OPD')
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
    # print('WFE Drift: {} nm'.format(wfe))

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
    create an undrifted and drifted PSF. (pseudo-code)

    >>> fpix, osamp = (128, 4)
    >>> coeff, _  = gen_psf_coeff('F210M', fov_pix=fpix, oversample=osamp)
    >>> wfe_cf    = wfed_coeff('F210M', fov_pix=fpix, oversample=osamp)
    >>> psf0      = gen_image_coeff('F210M', coeff=coeff, fov_pix=fpix, oversample=osamp)

    >>> # Drift the coefficients
    >>> wfe_drift = 5   # nm
    >>> cf_fit = wfe_cf.reshape([wfe_cf.shape[0], -1])
    >>> cf_mod = jl_poly(np.array([wfe_drift]), cf_fit).reshape(coeff.shape)
    >>> cf_new = coeff + cf_mod
    >>> psf5nm = gen_image_coeff('F210M', coeff=cf_new, fov_pix=fpix, oversample=osamp)
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

        cf_wfe = []
        # pool = mp.Pool(nsplit)
        try:
            # cf_wfe = pool.map(_wrap_wfed_coeff_for_mp, worker_args)
            with mp.Pool(nsplit) as pool:
                for res in tqdm(pool.imap(_wrap_wfed_coeff_for_mp, worker_args), total=npos, leave=False):
                    cf_wfe.append(res)
                pool.close()

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
            # pool.close()

        poppy.conf.n_processes = poppy_nproc_prev
    else:
        # No multiprocessor
        cf_wfe = []
        for wa in tqdm(worker_args):
            cf = _wrap_wfed_coeff_for_mp(wa)
            cf_wfe.append(cf)
        # cf_wfe = [_wrap_wfed_coeff_for_mp(wa) for wa in worker_args]

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

    _log.info('V2/V3 Coordinates and det pixel (sci) on {}/{}: ({:.2f}, {:.2f}), ({:.1f}, {:.1f})'
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
    return_raw : bool
        Return PSF coefficients of unevenly sampled V2/V3 grid
        along with the V2/V3 coordinates (cf_resid, v2_all, v3_all).


    Example
    -------
    Generate PSF coefficient, field position modifications, then
    create a PSF at some (V2,V3) location. (pseudo-code)

    >>> fpix, osamp = (128, 4)
    >>> coeff, _ = gen_psf_coeff('F210M', fov_pix=fpix, oversample=osamp)
    >>> cf_resid = field_coeff_resid('F210M', coeff, fov_pix=fpix, oversample=osamp)

    >>> # Some (V2,V3) location (arcmin)
    >>> v2, v3 = (1.2, -7)
    >>> cf_mod = field_model(v2, v3, cf_resid)
    >>> cf_new = coeff + cf_mod
    >>> psf    = gen_image_coeff('F210M', coeff=cf_new, fov_pix=fpix, oversample=osamp)
    """

    from astropy.table import Table

    kwargs['force']     = True
    kwargs['save']      = False
    kwargs['save_name'] = None

    if return_raw:
        save = False
        force = True
        _log.warn("return_raw=True; Setting 'save=False' and 'force=True'")

    # Get filter throughput and create bandpass
    if isinstance(filter_or_bp, six.string_types):
        filter = filter_or_bp
        bp = read_filter(filter, **kwargs)
    else:
        bp = filter_or_bp
        filter = bp.name
    channel = 'SW' if bp.avgwave() < 24000 else 'LW'

    # Set a default fov_pix and oversample
    fov_pix = kwargs['fov_pix'] if 'fov_pix' in list(kwargs.keys()) else 33
    oversample = kwargs['oversample'] if 'oversample' in list(kwargs.keys()) else 4

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
        if (nsplit <= 2) and (nsplit_psf > nsplit):
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

        cf_fields = []
        ntot = len(worker_args)
        # pool = mp.Pool(nsplit)
        try:
            with mp.Pool(nsplit) as pool:
                for res in tqdm(pool.imap(_wrap_field_coeff_for_mp, worker_args), total=ntot, leave=False):
                    cf_fields.append(res)
                pool.close()

            # cf_fields = pool.map(_wrap_field_coeff_for_mp, worker_args)
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
            # pool.close()

        poppy.conf.n_processes = poppy_nproc_prev
    else:  # No multiprocessor
        cf_fields = []
        for wa in tqdm(worker_args):
            cf = _wrap_field_coeff_for_mp(wa)
            cf_fields.append(cf)
        # cf_fields = [_wrap_field_coeff_for_mp(wa) for wa in worker_args]

    # Get residuals
    cf_fields_resid = np.array(cf_fields) - coeff0

    if return_raw:
        return cf_fields_resid, v2_all, v3_all

    # Create an evenly spaced grid of V2/V3 coordinates
    nv23 = 8
    v2grid = np.linspace(v2_min, v2_max, num=nv23)
    v3grid = np.linspace(v3_min, v3_max, num=nv23)

    # Interpolate onto an evenly space grid
    res = make_coeff_resid_grid(v2_all, v3_all, cf_fields_resid, v2grid, v3grid)
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
    (pseudo-code)

    >>> fpix, osamp = (320, 2)
    >>> filt, pupil, mask = ('F430M', 'WEDGELYOT', 'MASKLWB')
    >>> coeff    = gen_psf_coeff(filt, pupil, mask, fov_pix=fpix, oversample=osamp)
    >>> cf_resid = wedge_coeff(filt, pupil, mask, fov_pix=fpix, oversample=osamp)

    >>> # The narrow location (arcsec)
    >>> bar_offset = 8
    >>> cf_fit = cf_resid.reshape([cf_resid.shape[0], -1])
    >>> cf_mod = jl_poly(np.array([bar_offset]), cf_fit).reshape(coeff.shape)
    >>> cf_new = coeff + cf_mod
    >>> psf    = gen_image_coeff(filt, pupil, mask, coeff=cf_new, fov_pix=fpix, oversample=osamp)

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
    for val in tqdm(values):
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


def gen_image_from_coeff(coeff, coeff_hdr, **kwargs):
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

    kwargs['coeff']     = coeff
    kwargs['coeff_hdr'] = coeff_hdr

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
        with large PSFs over continuum source.
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
    psf_fit = jl_poly(wgood, coeff, dim_reorder=False, use_legendre=use_legendre, lxmap=lxmap)
    # Just in case weird coeff gives negative values
    # psf_fit[psf_fit<=0] = np.min(psf_fit[psf_fit>0]) / 10

    t5 = time.time()
    # Multiply each monochromatic PSFs by the binned e/sec at each wavelength
    # Array broadcasting: [nx,ny,nwave] x [1,1,nwave]
    # Do this for each spectrum/observation
    if nspec==1:
        psf_fit *= obs_list[0].binflux.reshape([-1,1,1])
        psf_list = [psf_fit]
    else:
        psf_list = [psf_fit*obs.binflux.reshape([-1,1,1]) for obs in obs_list]
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
            elif module=='B': # Flip right to left to disperse in correct orientation
                psf_fit = psf_fit[:,:,::-1]

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

                spec_over[:,intx:intx+fov_pix_over] += fshift(psf_fit[i], delx=fracx, interp='cubic')
                # im = psf_fit[i]
                # spec_over[:,intx:intx+fov_pix_over] += im*(1.-fracx) + np.roll(im,1,axis=1)*fracx

            spec_over[spec_over<__epsilon] = 0 #__epsilon

            # Rotate spectrum to its V2/V3 coordinates
            spec_bin = poppy.utils.krebin(spec_over, (fov_pix,npix_spec))
            if 'GRISM90' in pupil: # Rotate image 90 deg CCW
                spec_over = np.rot90(spec_over, k=-1)
                spec_bin = np.rot90(spec_bin, k=-1)
            elif module=='B': # Flip right to left for sci coords
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
        if ('GRISM0' in pupil) and (module=='B'): # Flip for sci coords
            wspec = wspec[::-1]

        if nspec == 1: 
            spec_list = spec_list[0]
            spec_list_over = spec_list_over[0]
        # Return list of wavelengths for each horizontal pixel
        # as well as spectral image

        t7 = time.time()
        _log.debug('jl_poly: {:.2f} sec; binflux: {:.2f} sec; disperse: {:.2f} sec'.format(t5-t4, t6-t5, t7-t6))
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
            data_over = psf_fit.sum(axis=0)
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
