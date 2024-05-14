import numpy as np
import os

import json

from tqdm import trange, tqdm

from astropy.io import ascii, fits
from astropy.convolution import convolve, Gaussian2DKernel

from webbpsf_ext.utils import get_one_siaf, get_detname
from webbpsf_ext.analysis_tools import ipc_info, ppc_info
from webbpsf_ext.imreg_tools import read_sgd_files, get_coron_apname
from webbpsf_ext.image_manip import fractional_image_shift, fourier_imshift
from webbpsf_ext.image_manip import apply_pixel_diffusion, add_ipc, add_ppc
from webbpsf_ext import robust

from jwst import datamodels
from jwst.datamodels import dqflags
from spaceKLIP.utils import get_dqmask

nrc_siaf = get_one_siaf(instrument='NIRCam')

import logging
# Define logging
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

from ..logging_utils import setup_logging

#################################################################
# Functions for generating NIRCam class
#################################################################

def subtract_psf(image, psf, osamp=1, bpmask=None, rin=None, rout=None,
                 xyshift=(0,0), psf_scale=None, psf_offset=0,
                 method='fourier', pad=True, cval=0, interp='lanczos', 
                 kipc=None, kppc=None, diffusion_sigma=None, 
                 weights=None, return_sum2=False, **kwargs):
    """ Subtract PSF from image

    Provide scale, offset, and shift values to PSF before subtraction.
    Uses `fractional_image_shift` function to shift PSF.
    
    Parameters
    ----------
    image: ndarray
        Observed science image.
    psf: ndarray
        Oversampled PSF.
    osamp: int
        Oversampling factor of PSF.
    xyshift: tuple
        Shift values in (x,y) directions. Units of pixels.
    psf_scale: float
        Scale factor to apply to PSF. If set to None, then will
        find the best scaling factor.
    psf_offset: float
        Offset to apply to PSF.
    method : str
        Method to use for shifting. Options are:
        - 'fourier' : Shift in Fourier space
        - 'fshift' : Shift using interpolation
        - 'opencv' : Shift using OpenCV warpAffine
    kipc: ndarray
        3x3 array of IPC kernel values. If None, then no IPC is applied.
    kppc: ndarray
        3x3 array of PPC kernel values. If None, then no PPC is applied.
        Should already be oriented along readout direction of PSF.
    diffusion_sigma: float
        Sigma value for Gaussian diffusion kernel. If None, then
        no diffusion is applied. In units of detector pixels.
    weights: ndarray
        Array of weights to use during the fitting process.
        Useful if you have bad pixels to mask out (ie.,
        set them to zero). Default is None (no weights).
        Should be same size as image.
        Recommended is inverse variance map.
    return_sum2 : bool
        Return the sum of the squared difference between the image
        and PSF. Default is False.

    Keyword Args
    ------------
    pad : bool
        Should we pad the array before shifting, then truncate?
        Otherwise, the image is wrapped.
    cval : sequence or float, optional
        The values to set the padded values for each axis. Default is 0.
        ((before_1, after_1), ... (before_N, after_N)) unique pad constants for each axis.
        ((before, after),) yields same before and after constants for each axis.
        (constant,) or int is a shortcut for before = after = constant for all axes.
    interp : str
        Interpolation method to use for shifting using 'fshift' or 'opencv. 
        Default is 'cubic'.
        For 'opencv', valid options are 'linear', 'cubic', and 'lanczos'.
        for 'fshift', valid options are 'linear', 'cubic', and 'quintic'.
    """
    
    from webbpsf_ext.image_manip import frebin, crop_image
    from webbpsf_ext.image_manip import fractional_image_shift
    from webbpsf_ext.image_manip import apply_pixel_diffusion, add_ipc, add_ppc
    from webbpsf_ext.coords import dist_image

    # Shift oversampled PSF and 
    xsh_over, ysh_over = np.array(xyshift) * osamp
    if method is not None:
        kwargs_shift = {}
        kwargs_shift['pad'] = pad
        kwargs_shift['cval'] = cval
        if method in ['fshift', 'opencv']:
            kwargs_shift['interp'] = interp
        # Scale Gaussian std dev by oversampling factor
        gstd_pix = kwargs.get('gstd_pix')
        if gstd_pix is not None:
            kwargs_shift['gstd_pix'] = gstd_pix * osamp
        psf_over = fractional_image_shift(psf, xsh_over, ysh_over, method=method, **kwargs_shift)

    # Charge diffusion
    if diffusion_sigma is not None:
        sigma_osamp = diffusion_sigma * osamp
        psf_over = apply_pixel_diffusion(psf_over, sigma_osamp)

    # Rebin to detector sampling
    psf_det = frebin(psf_over, scale=1/osamp) if osamp!=1 else psf_over
        
    # Add IPC to detector-sampled PSF
    if kipc is not None:
        psf_det = add_ipc(psf_det, kernel=kipc)

    if kppc is not None:
        psf_det = add_ppc(psf_det, kernel=kppc, nchans=1)
    
    # Crop image
    if psf_det.shape != image.shape:
        psf_det = crop_image(psf_det, image.shape)

    if psf_scale is None:
        # Get optimal scale factor between images
        # Ignore NaNs and zeros
        good_mask = ~np.isnan(image) & ~np.isnan(psf_det)
        good_mask = good_mask & (~np.isclose(image,0)) & (~np.isclose(psf_det,0))
        if bpmask is not None:
            good_mask &= ~bpmask

        if (rin is not None) or (rout is not None):
            rho = dist_image(image)
            rin = 0 if rin is None else rin
            rout = np.inf if rout is None else rout
            good_mask &= (rho >= rin) & (rho <= rout)

        im_good = image[good_mask].flatten() - psf_offset
        psf_good = psf_det[good_mask].flatten()
        cf = np.linalg.lstsq(psf_good.reshape([1,-1]).T, im_good, rcond=None)[0]
        psf_scale = cf[0]

    psf_det = psf_det * psf_scale + psf_offset

    # Subtract PSF from image
    diff = image - psf_det

    if weights is not None:
        diff = diff * weights

    if return_sum2:
        # Set anything that are 0 in either image as zero in difference
        mask = np.isclose(image,0) | np.isclose(psf_det,0)
        diff[mask] = 0
        return np.sum(diff**2)
    else:
        return diff

# NOTE (2/13/2024): A lot of issues with this function finding the best solution. Best to just use
# a coarse grid search, then zoom in with a finer grid search. 
def align_leastsq(image, psf_over, osamp=1, bpmask=None, psf_bpmask=None, weights=None,
                  params0=[0.0,0.0,1.0,0.0], params_fixed=None, 
                  kipc=None, kppc=None, diffusion_sigma=None,
                  shift_method='fourier', interp='cubic', pad=True, 
                  method='COBYLA', kwargs_pass={}, **kwargs):
    """Find best shift value
    
    LSQ optimization with option of shift alignment algorithm.
    In practice, the 'reference' image gets shifted to match
    the 'target' image.
    
    Parameters
    ----------
    image : ndarray
        Observed science image
    psf_over : ndarray
        Input oversampled PSF to fit and align
        
    Keyword Args
    ------------
    osamp : int
        Oversampling factor of PSF
    bpmask : ndarray, None
        Bad pixel mask indicating what pixels in input
        image to ignore.
    weights : ndarray, None
        Array of weights to use during the fitting process.
    params0 : list
        Initial guess for (x, y, scale, offset) values. If only two
        values are given, then scale=1 and offset=0 are fixed. If
        three values are given, then offset=0 is fixed.
    params_fixed : list
        List of booleans indicating which parameters to fix.
    pad : bool
        Should we pad the array before shifting, then truncate?
        Otherwise, the image is wrapped.
    shift_method : func
        Which function to use for sub-pixel shifting.
        Options are fourier_imshift, fshift, or cv_shift.
    interp : str
        Interpolation method to use for shifting. Default is 'cubic'.
        Options are 'nearest', 'linear', 'cubic', and 'quadratic'
        for `fshift`.  Consider 'lanczos' for cv_shift.
    method : str
        Optimization method to use. Default is 'COBYLA'. 'Powell' also
        seems to work well when aligning data images, but should fix the
        offset parameter to some value (e.g., 0.0).

    Returns
    -------
    list
        (x, y, scale, offset) values from LSQ optimization, where (x, y) 
        are the misalignment of target from reference and scale
        is the fraction by which the target intensity must be
        reduced to match the intensity of the reference. Offset gives
        the difference in the mean intensity of the two images.
    """
    from scipy.optimize import least_squares, leastsq, minimize
    from astropy.convolution import convolve, Gaussian2DKernel
    
    def calc_psf_diff(params, image, psf, kwargs_pass):
        """PSF differencing helper"""

        # Get fixed parameters
        fixed_dict = kwargs_pass['fixed_dict']
        params_all = np.zeros(4)
        j = 0
        for i in range(4):
            d = fixed_dict[i]
            if d['fixed']:
                params_all[i] = d['value']
            else:
                params_all[i] = params[j]
                j += 1

        # Shift values always come first
        xsh, ysh = params_all[0:2]

        # Get scale and offset values
        if len(params_all)==2:
            psf_scale = kwargs_pass.get('psf_scale', 1.0)
            psf_offset = kwargs_pass.get('psf_offset', 0.0)
        elif len(params_all)==3:
            psf_scale = params_all[-1]
            psf_offset = kwargs_pass.get('psf_offset', 0.0)
        elif len(params_all)==4:
            psf_scale  = params_all[-2]
            psf_offset = params_all[-1]
        else:
            raise ValueError("params must be length 2, 3, or 4")
        
        # If psf_bpmask exists, it needs to be shifted as well
        # Then update weights
        psf_bpmask = kwargs_pass.pop('psf_bpmask', None)
        if psf_bpmask is not None:
            from webbpsf_ext.image_manip import fshift
            bp = psf_bpmask.astype('float')
            bp = fshift(bp, xsh, ysh, method='linear', pad=True, cval=1)
            psf_bpmask = bp > 0
            # Update weights
            weights = kwargs_pass.get('weights', np.ones_like(image))
            weights[psf_bpmask] = 0
            kwargs_pass['weights'] = weights

        kwargs_pass['xyshift']    = (xsh, ysh)
        kwargs_pass['psf_scale']  = psf_scale
        kwargs_pass['psf_offset'] = psf_offset
        return subtract_psf(image, psf, return_sum2=True, **kwargs_pass)


    # Account for any NaNs in input image
    # The input image, bpmask, and weights are all stationary
    # Only the psf image and its associated bpmask get shifted
    nan_mask = np.isnan(image)
    if np.any(nan_mask):
        bpmask = nan_mask if bpmask is None else bpmask | nan_mask

    # Set weights image to pass to differencing function
    if bpmask is not None:
        weights = np.ones_like(image) if weights is None else weights.copy()
        weights[bpmask] = 0

    # Are we fixing any of the parameters?
    if params_fixed is None:
        params_fixed = [False, False, False, False]
    params_fixed = np.array(params_fixed, dtype='bool')
    fixed_dict = { 0 : {}, 1 : {}, 2 : {}, 3 : {} }
    for i in range(len(params0)):
        d = fixed_dict[i]
        d['fixed'] = params_fixed[i]
        d['value'] = params0[i]

    # Keywords to pass
    kwargs2 = {
        'psf_bpmask' : psf_bpmask,
        'weights' : weights,
        'osamp'   : osamp,
        'method'  : shift_method,
        'interp'  : interp,
        'pad'     : pad,
        'kipc' : kipc, 'kppc' : kppc,
        'diffusion_sigma' : diffusion_sigma,
        'fixed_dict' : fixed_dict,
        'gstd_pix' : kwargs.pop('gstd_pix', None)
    }
    # kwargs_pass = kwargs.copy()
    kwargs_pass.update(kwargs2)

    # Apply Gaussian blurring to image as well
    gstd_pix = kwargs_pass.get('gstd_pix')
    if (gstd_pix is not None) and (gstd_pix > 0):
        kernel = Gaussian2DKernel(x_stddev=gstd_pix)
        image = convolve(image, kernel)
    else:
        # Set any NaNs to zero
        image = image.copy()
        image[nan_mask] = 0
        psf_over = psf_over.copy()
        psf_over[nan_mask] = 0

    params0 = np.array(params0)
    params0_free = params0[~params_fixed]

    bounds = kwargs.get('bounds', None)
    if bounds is not None:
        npar = len(params0)
        bounds = [bounds[i] for i in range(npar) if not params_fixed[i]]
        kwargs['bounds'] = bounds

    res = minimize(calc_psf_diff, params0_free, args=(image, psf_over, kwargs_pass), 
                   method=method, **kwargs)
    params_free = res['x']

    # res = leastsq(calc_psf_diff, params0_free, args=(image, psf_over, kwargs_pass), **kwargs)
    # params_free = res[0]

    out = params0.copy()
    out[~params_fixed] = params_free
    return out


    # Use loss='soft_l1' for least squares robust against outliers
    # May want to play around with f_scale...
    # res = least_squares(psf_diff, params0, #diff_step=0.1, loss='soft_l1', f_scale=1.0, 
    #                     args=(image, psf_over, kwargs_pass), **kwargs)
    # out = res.x

    # for i in range(len(params0)):
    #     out[i] = pfix[i] if params_fixed[i] else out[i]

    # return out

def _gen_nrc_class(filt, apname, date, fov_pix, oversample, autogen_coeffs=False,
                   quick_grid=False, obs_hci=False, **kwargs):

    import pynrc, time

    # Create NIRCam object
    if obs_hci:
        sp_sci   = kwargs.pop('sp_sci')
        dist_sci = kwargs.pop('dist_sci')
        sp_ref   = kwargs.pop('sp_ref', None)

        nrc = pynrc.obs_hci(sp_sci, dist_sci, sp_ref=sp_ref, use_ap_info=True,
                            filter=filt, apname=apname, autogen_coeffs=autogen_coeffs, **kwargs)
    else:
        nrc = pynrc.NIRCam(filter=filt, apname=apname, autogen_coeffs=autogen_coeffs, **kwargs)

    # Load date-specific OPD map
    if not quick_grid:
        retries = 0
        retry_limit = 5
        while retries < retry_limit:
            try:
                nrc.load_wss_opd_by_date(date=date, choice='before', plot=False, verbose=False)
                break
            except Exception as e:
                # Wait 5 seconds before retrying
                time.sleep(5)
                # log the error
                retries += 1
                if retries == retry_limit:
                    _log.error(f'Failed to load OPD map after {retry_limit} retries')
                    raise e

    # Set fov_pix and oversample
    ap = nrc_siaf[apname]
    nrc.fov_pix = np.min([fov_pix, ap.XSciSize, ap.YSciSize])
    nrc.oversample = oversample

    if quick_grid:
        nrc.gen_psf_coeff()
        nrc.gen_wfemask_coeff()

    nrc.include_distortions = kwargs.get('include_distortions', True)
    nrc.include_ote_field_dependence = kwargs.get('include_ote_field_dependence', True)
    nrc.include_si_wfe = kwargs.get('include_si_wfe', True)

    # If there's a filter specified in the apname (e.g. SWB/LWB)
    # then nrc.filter matches that by default. Instead, we want to
    # update nrc.filter to match self.filter.
    if nrc.filter != filt:
        nrc.update_psf_coeff(filter=filt)

    # Remove image mask for TACONF1 apertures
    if ('FULL_TA' in apname) or ('FULL_FSTA' in apname):
        nrc.update_psf_coeff(image_mask='CLEAR')
        # Add in ND throughput depending on aperture name
        nrc.ND_acq = False if 'FSTA' in apname else True

    nrc._update_bg_class(fov_bg_match=True)

    # Update default webbpsf detector position (sci coords) to correspond 
    # to requested aperture reference point
    ap_full = nrc_siaf[nrc.aperturename]
    xsci_full, ysci_full = ap_full.det_to_sci(ap.XDetRef, ap.YDetRef)
    nrc.detector_position = (xsci_full, ysci_full)
    nrc._nrc_bg.detector_position = (xsci_full, ysci_full)

    return nrc

def nrc_from_file(fpath, fov_pix, oversample=None, **kwargs):
    """Create NIRCam object from a given file"""

    data_model = datamodels.open(fpath)

    # apname = data_model.meta.aperture.name
    # Do a better job of parsing aperture name
    apname = get_coron_apname(data_model)
    date = data_model.meta.observation.date_beg

    filt = data_model.meta.instrument.filter
    pupil = data_model.meta.instrument.pupil

    resample_step = data_model.meta.cal_step.resample
    if (resample_step is None) or (resample_step=='SKIPPED'):
        default_distortions = True
    else:
        default_distortions = False
    kwargs['include_distortions'] = kwargs.get('include_distortions', default_distortions)

    # Check if filter in pupil wheel
    if pupil[0]=='F' and (pupil[-1]=='N' or pupil[-1]=='M'):
        filt = pupil

    _log.info("Creating NIRCam object...")
    if oversample is None:
        # Check if coronagraphic observation
        oversample = 2 if 'MASK' in apname else 4
    nrc = _gen_nrc_class(filt, apname, date, fov_pix, oversample, **kwargs)

    # Update detector readout parameters
    read_mode = data_model.meta.exposure.readpatt
    ngroup = data_model.meta.exposure.ngroups
    nint = data_model.meta.exposure.nints
    nrc.update_detectors(read_mode=read_mode, ngroup=ngroup, nint=nint)

    # Close data model
    data_model.close()

    return nrc

#################################################################
# Functions for generating NIRCam PSFs
#################################################################

def gen_defocus_psf(nrc, defocus_waves_2um, return_oversample=False, xyoffpix=(0,0), 
                    include_distortions=None, include_ote_field_dependence=None,
                    include_si_wfe=None, return_hdul=False, quick_grid=False, **kwargs):
    """ Generate a single defocused PSF
    
    return_oversample : int
        Return an oversampled version of the PSF. Only valid if
        return_hdul=False, otherwise always returns both detector
        and oversampled versions.
    """

    from webbpsf_ext.image_manip import fourier_imshift, frebin

    # Make sure defocus_waves_2um is not None
    if defocus_waves_2um is None:
        defocus_waves_2um = 0

    use_bg_psf = kwargs.pop('use_bg_psf', False)
    if use_bg_psf:
        nrc = nrc._nrc_bg

    # Get initial attribute values
    idist_orig = nrc.include_distortions
    iote_orig = nrc.include_ote_field_dependence
    isiwfe_orig = nrc.include_si_wfe
    idefocus = nrc.options.get('defocus_waves')
    idefocus_wavelength = nrc.options.get('defocus_wavelength')
    
    # Update attributes
    if include_distortions is not None:
        nrc.include_distortions = include_distortions
    if include_ote_field_dependence is not None:
        nrc.include_ote_field_dependence = include_ote_field_dependence
    if include_si_wfe is not None:
        nrc.include_si_wfe = include_si_wfe
    nrc.options['defocus_waves'] = defocus_waves_2um
    nrc.options['defocus_wavelength'] = 2e-6

    if return_hdul:
        if quick_grid:
            hdul = nrc.calc_psf_from_coeff(return_oversample=return_oversample, 
                                           return_hdul=True, **kwargs)
        else:
            hdul = nrc.calc_psf(return_hdul=True, **kwargs)
        # Perform shifts on each HDU
        if (xyoffpix is not None) and (not np.allclose(xyoffpix, 0)):
            for hdu in hdul:
                data = hdul.data
                osamp = hdu.header['DET_SAMP']
                hdul.data = fourier_imshift(data, xyoffpix[0]*osamp, xyoffpix[1]*osamp)
        output = hdul
    else:
        # Generate oversampled PSF and downsample
        if quick_grid:
            output = nrc.calc_psf_from_coeff(return_oversample=True, return_hdul=False, **kwargs)
        else:
            output = nrc.calc_psf(return_oversample=True, return_hdul=False, **kwargs)
        # Shift if xyoffpix is not (0,0)
        osamp = nrc.oversample
        if (xyoffpix is not None) and (not np.allclose(xyoffpix, 0)):
            output = fourier_imshift(output, xyoffpix[0]*osamp, xyoffpix[1]*osamp)
        if not return_oversample:
            output = frebin(output, scale=1/osamp)
    
    # Return attributes to original
    nrc.include_distortions = idist_orig
    nrc.include_ote_field_dependence = iote_orig
    nrc.include_si_wfe = isiwfe_orig
    # Remove from options if originally set to None
    if idefocus is None:
        nrc.options.pop('defocus_waves', None)
    else:
        nrc.options['defocus_waves'] = idefocus
    if idefocus_wavelength is None:
        nrc.options.pop('defocus_wavelength', None)
    else:
        nrc.options['defocus_wavelength'] = idefocus_wavelength
    
    return output

def gen_diffusion_psf(nrc, diffusion_sigma, return_oversample=False, xyoffpix=(0,0), 
                      include_distortions=None, include_ote_field_dependence=None,
                      defocus_waves_2um=0, include_si_wfe=None, return_hdul=False, 
                      quick_grid=False, **kwargs):
    """Generate a single PSF with charge diffusion applied
    
    Parameters
    ==========
    nrc : pynrc.NIRCam instance
        NIRCam object
    diffusion_sigma : float or list
        Charge diffusion value to apply to PSF in terms of detector pixels.
        Effectively applies a Gaussian filter.
    return_oversample : int
        Return an oversampled version of the PSF. Only valid if
        return_hdul=False, otherwise always returns both detector
        and oversampled versions.
    xyoffpix : tuple
        (x,y) shift offset to apply to PSF (fourier_imshift). Units of pixels.

    Keyword Args
    ============
    sp : pysynphot.Spectrum
        Source spectrum to use for PSF generation.
    """

    kwargs['return_oversample'] = return_oversample
    kwargs['return_hdul']       = return_hdul
    kwargs['xyoffpix']          = xyoffpix
    kwargs['include_distortions']          = include_distortions
    kwargs['include_ote_field_dependence'] = include_ote_field_dependence
    kwargs['include_si_wfe']               = include_si_wfe
    kwargs['quick_grid']                   = quick_grid
    res = gen_defocus_psf(nrc, defocus_waves_2um, **kwargs)

    if return_hdul:
        # Apply each charge diffusion value to PSF
        for hdu in res:
            osamp = hdu[0].header['OSAMP'] # Image oversampling
            hdu[0].data = apply_pixel_diffusion(hdu[0].data, diffusion_sigma * osamp)
    else:
        df = diffusion_sigma * nrc.oversample if return_oversample else diffusion_sigma
        res = apply_pixel_diffusion(res, df)

    return res


# def klip_subtraction(imarr, imref):

#     from .pca import run_pca_subtraction

#     res_asec = 206265 * nrc_obs.nrc.bandpass.pivot().to_value('m') / 6.5
#     res_pix = res_asec / nrc_obs.nrc.pixelscale
#     fwhm_pix = 1.025 * res_pix

#     # Subtraction regions config
#     kwargs_sub = {
#         'IWA_nfwhm': 2,
#         'OWA_nfwhm': 33,
#         'sub_ann_rad': 3,
#         'sub_ann_width': 3,
#         'annuli_spacing' : 'constant',
#         'constant_theta' : False,
#     }
#     # Optimization regions config
#     kwargs_opt = {
#         'opt_ann_rad' : kwargs_sub['sub_ann_rad']+1,
#         'opt_ann_width' : 3,
#         'nfwhm_sep'   : 0,
#         'exclude_sub' : True,
#     }

def stellar_arguments(name, votdir='../votables/', fname=None, **kwargs):

    from webbpsf_ext import bp_2mass
    from webbpsf_ext.synphot_ext import ObsBandpass

    # Define bandpasses and source information
    bp_k = bp_2mass('k')

    # Remove spaces, dashes, and underscores
    name_key = name.replace(' ','').replace('-','').replace('_','')

    # Science   source,  dist, age, sptype, Teff, [Fe/H], log_g, mag, band
    # dist in units of pc and age in units of Myr
    stellar_dict = {
        'MWC758' : {
            'name': 'MWC-758', 'fname': 'MWC758.vot',
            'dist': 160, 'age': 5, 'sptype': 'A5V', 
            'mag_val': 5.7, 'bp': bp_k, 
        },
        'HLTAU' : {
            'name': 'HL-Tau', 'fname': 'HLTau.vot',
            'dist': 140, 'age': 5, 'sptype': 'K5V', 
            'mag_val': 7.4, 'bp': bp_k, 
        },
        'SAO206462' : {
            'name': 'SAO-206462', 'fname': 'SAO206462.vot',
            'dist': 135, 'age': 10, 'sptype': 'F8V', 
            'mag_val': 5.8, 'bp': bp_k, 
        },
        'PDS70' : {
            'name': 'PDS-70', 'fname': 'PDS70.vot',
            'dist': 112, 'age': 10, 'sptype': 'K7IV', 
            'mag_val': 8.8, 'bp': bp_k, 
        },
        'HD107146' : {
            'name': 'HD 107146', 'fname': 'HD107146.vot',
            'dist': 27.47, 'age': 200, 'sptype': 'G2V', 
            'Teff': 5850, 'metallicity': +0.00, 'log_g': 4.5,
            'Av': 0.0, 'mag_val': 5.54, 'bp': bp_k, 
        },
        'HD111398' : {
            'name': 'HD 111398', 'fname': 'HD111398.vot',
            'sptype': 'G5V', 'Teff': 5689, 'metallicity': +0.07, 'log_g': 4.5,
            'Av': 0.0, 'mag_val': 5.53, 'bp': bp_k, 
        },

    }

    try:
        dict_sci = stellar_dict[name_key]
    except KeyError:
        raise ValueError(f"Source '{name}' not found in stellar dictionary.")
    
    fname = dict_sci.pop('fname')
    dict_sci['votable_input'] = os.path.join(votdir, fname)

    # Add any kwargs
    dict_sci.update(kwargs)
    
    return dict_sci


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

class nrc_analyze():

    siaf = nrc_siaf
    _mastdir = os.getenv('JWSTDOWNLOAD_OUTDIR')

    def __init__(self, pid, obsids, filter, sca, obsids_ref=None, basedir=None):
        """ Initialize NIRCam analysis class 
        
        Parameters
        ----------
        pid : int
            Program ID
        obsids : list
            List of observation IDs
        filter : str
            NIRCam filter
        sca : int
            Detector SCA number
        obsids_ref : list
            List of reference observation IDs. Will create additional classes for each
            set of reference observations.
        """
        self.pid = pid
        self.obsids = obsids
        self.obsids_ref = obsids_ref
        self.filter = filter
        self.sca = get_detname(sca, use_long=True)

        # Diffusion info
        self._best_diffusion = None
        self._diffusion_file = 'P330E_best_diffusion.txt'
        self._diffusion_table = ascii.read(self._diffusion_file)

        # Filter table directory
        self._filter_offset_dir = None

        # Default to MAST directory if not specified
        self.basedir = self._mastdir if basedir is None else basedir
        self._uncal_dir = None
        self._rate_dir = None
        self._cal_dir = None

        # Save locations
        self.figdir = 'figures_analyze/'
        self.tbldir = 'output_analyze/'

        # Create directories if they don't exist
        for d in [self.figdir, self.tbldir]:
            os.makedirs(d, exist_ok=True)

        self.obs_dict = {}
        self.sp_sci = None
        self._sp_kws = {}

        # Attribute to determine to set if simulated data
        self.is_sim = False

        # Init PSF grid
        self.nrc = None
        self.psfs_over = None
        self.xyoff_psfs_over = None
        self.xy_loc_ind = None
        self.xyshift = None
        self.xy_mask_offset = None
        self.shift_matrix = None

        # Create objects for each reference observation
        if obsids_ref is not None:
            self.ref_objs = []
            for oid in obsids_ref:
                ref_obj = nrc_analyze(pid, [oid], filter, sca, basedir=basedir)
                self.ref_objs.append(ref_obj)
        else:
            self.ref_objs = None

    @property
    def uncal_dir(self):
        """Directory housing uncal.fits data"""
        out = os.path.join(self.basedir, f'{self.pid:05d}/') if self._uncal_dir is None else self._uncal_dir
        return out
    @uncal_dir.setter
    def uncal_dir(self, value):
        self._uncal_dir = value
    @property
    def rate_dir(self):
        """Directory housing rate & rateints data"""
        out = os.path.join(self.basedir, f'{self.pid:05d}_proc/stage1/') if self._rate_dir is None else self._rate_dir
        return out
    @rate_dir.setter
    def rate_dir(self, value):
        self._rate_dir = value
    @property
    def cal_dir(self):
        """Directory housing cal & calints data"""
        out = os.path.join(self.basedir, f'{self.pid:05d}_proc/stage2/') if self._rate_dir is None else self._rate_dir
        return out
    @cal_dir.setter
    def cal_dir(self, value):
        self._cal_dir = value

    @property
    def apname(self):
        """Aperture name"""
        try:
            hdr0 = self.obs_dict[self.obsids[0]][0]['hdr0']
        except:
            fpath = self.obs_dict[self.obsids[0]][0]['file']
            hdr0 = fits.getheader(fpath)
        apname = get_coron_apname(hdr0)
        return apname
    
    @property
    def siaf_ap(self):
        """SIAF aperture object"""
        return self.siaf[self.apname]
    
    @property
    def is_sgd(self):
        """Is this a SGD reference observation?"""
        try:
            hdr0 = self.obs_dict[self.obsids[0]][0]['hdr0']
        except:
            fpath = self.obs_dict[self.obsids[0]][0]['file']
            hdr0 = fits.getheader(fpath)
        is_sgd = hdr0.get('SUBPXPAT') == 'SMALL-GRID-DITHER'
        return is_sgd
    @property
    def sgd_pattern(self):
        """SGD pattern"""
        try:
            hdr0 = self.obs_dict[self.obsids[0]][0]['hdr0']
        except:
            fpath = self.obs_dict[self.obsids[0]][0]['file']
            hdr0 = fits.getheader(fpath)
        sgd_pattern = hdr0.get('SMGRDPAT', None)
        return sgd_pattern

    @property
    def rvals(self):
        """Inner and outer radii for PSF fitting"""
        from webbpsf_ext.bandpasses import nircam_filter

        rin = 5
        rout = 25
        # Scale rout by wavelength
        if self.filter in ['F210M', 'F335M']:
            scale = 1
        else:
            bp = nircam_filter(self.filter)
            wave = bp.avgwave().to_value('um')
            scale = wave / 2.1 if wave<2.5 else wave / 3.35

        rout = int(rout * scale)
        return (rin, rout)
    
    @property
    def kipc(self):
        """IPC kernel"""
        (a1, a2), kipc = ipc_info(self.sca)
        return kipc
    @property
    def kppc(self):
        """PPC kernel"""
        ppc_frac, kppc = ppc_info(self.sca)
        return kppc
    
    @property
    def best_diffusion(self):
        """PSF diffusion value in pixels"""
        if self._best_diffusion is None:
            tbl = self._diffusion_table
            ind = (tbl['Filter']==self.filter) & (tbl['SCA']==self.sca)
            diffusion = 0 if ind.sum()==0 else tbl[ind]['BestSig_sub'][0]
            return diffusion
        else:
            return 0
    @best_diffusion.setter
    def best_diffusion(self, value):
        self._best_diffusion = value

    def generate_obs_dict(self, file_type='calints.fits'):
        """Generate dictionary of observations"""
        if 'cal' in file_type or 'i2d' in file_type:
            indir = self.cal_dir
        else:
            indir = self.rate_dir

        if len(self.obs_dict)>0:
            obs_dict = self.obs_dict
            self.obs_dict = {}
            del obs_dict

        for oid in self.obsids:
            obs_dict = read_sgd_files(indir, self.pid, oid, self.filter, self.sca,
                                      file_type=file_type, combine_same_dithers=True)
            self.obs_dict[oid] = obs_dict

        # Call this same function in the reference objects
        if self.ref_objs is not None:
            for ref_obj in self.ref_objs:
                ref_obj.generate_obs_dict(file_type=file_type)

    def _flag_bad_pixels(self, imarr, dqarr, nsig_spatial=10, nsig_temporal=10, ntemporal_limit=10, niter=3):
        """Flag bad pixels in a single image or stack of images
        
        Returns updated dqarr
        """

        from webbpsf_ext import robust
        from webbpsf_ext.image_manip import bp_fix
        from webbpsf_ext.coords import dist_image

        # 1. For each dither position, find consistent bad pixels in the median data.
        # 2. For each dither position, flag pixels that are always the same value.
        if len(imarr.shape)==3:
            im_med = np.nanmedian(imarr, axis=0)
            _, bp_med = bp_fix(im_med, sigclip=nsig_spatial, niter=niter,
                               in_place=False, return_mask=True)
            im_std = np.nanstd(imarr, axis=0)
            bp_std = (im_std==0)
            bp_all = bp_med | bp_std

            # Flag all images in DQ array as NO_NOT_USE
            for dq in dqarr:
                dq[bp_all] |= dqflags.pixel['DO_NOT_USE']
                
        # 3. For a single image, flag pixels that are nsig times the standard deviation.
        if len(imarr.shape)==3 and imarr.shape[0]>=ntemporal_limit:
            good_mask = robust.mean(imarr, Cut=nsig_temporal, axis=0, return_mask=True)
            bp_mask = ~good_mask
            # Flag bad pixels in DQ array as NO_NOT_USE 
            dqarr[bp_mask] |= dqflags.pixel['DO_NOT_USE']

        # 4. Find bad pixels in individual images
        if len(imarr.shape)==3:
            for im, dq in zip(imarr, dqarr):
                _, bp = bp_fix(im, sigclip=nsig_spatial, niter=niter, 
                                in_place=False, return_mask=True)
                dq[bp] |= dqflags.pixel['DO_NOT_USE']
        elif len(imarr.shape)==2:
            _, bp = bp_fix(imarr, sigclip=nsig_spatial, niter=niter, 
                            in_place=False, return_mask=True)
            dqarr[bp] |= dqflags.pixel['DO_NOT_USE']
        else:
            raise ValueError(f"Unexpected shape for imarr: {imarr.shape}")
        
        # 5. Flag pixels that are 5-sigma below bg level
        if len(imarr.shape)==3:
            im_med = np.nanmedian(imarr, axis=0)
            dq = np.bitwise_and.reduce(dqarr, axis=0)
        else:
            im_med = imarr
            dq = dqarr
        bp = get_dqmask(dq, ['DO_NOT_USE']) > 0
        rval = 0.8 * np.max([imarr.shape[-2:]]) / 2
        mask = ~bp & (dist_image(im_med) > rval) & ~np.isnan(im_med)
        bg_val = robust.mode(im_med[mask])
        bg_sig = robust.medabsdev(im_med[mask])
        bg_bad = (im_med < bg_val - 5*bg_sig) & ~np.isnan(im_med)
        # Flag all images in DQ array as NO_NOT_USE
        if len(imarr.shape)==3:
            for dq in dqarr:
                dq[bg_bad] |= dqflags.pixel['DO_NOT_USE']
        else:
            dqarr[bg_bad] |= dqflags.pixel['DO_NOT_USE']

        return dqarr

    def flag_bad_pixels(self, nsig_spatial=10, nsig_temporal=10, ntemporal_limit=10, niter=3,
                        save_dq_flags=True, save_suffix='_newdqflags.fits', force=False):
        """ Flag bad pixels in each observation's DQ array

        Will first search for already saved files with updated dq flags. If found, will not
        re-run the flagging process unless force=True.

        The process:
        1. For each dither position, find consistent bad pixels in the median data.
        2. For each dither position, flag pixels that are always the same value.
        3. For a single image, flag pixels that are nsig times the standard deviation.
        4. Find bad pixels in individual images.
        5. Flag pixels that are 5-sigma below bg level

        Parameters
        ----------
        nsig_spatial : float
            Number of sigma for spatial bad pixel flagging
        nsig_temporal : float
            Number of sigma for temporal bad pixel flagging
        ntemporal_limit : int
            Minimum number of images for temporal flagging to be applied
        niter : int
            Number of iterations for sigma clipping
        save_dq_flags : bool
            Save the DQ flags to a new file
        save_suffix : str
            Suffix to append to the file name when saving DQ flags
        force : bool
            Force the function to run even if new DQ flags have already been saved
        """

        # Check if generate_obs_dict has been run
        if len(self.obs_dict)==0:
            raise ValueError("Run generate_obs_dict() first.")

        kwargs = {
            'nsig_spatial'    : nsig_spatial,
            'nsig_temporal'   : nsig_temporal,
            'ntemporal_limit' : ntemporal_limit,
            'niter': niter,
        }

        obs_dict = self.obs_dict
        for oid in self.obsids:
            odict = obs_dict[oid]
            for k in tqdm(odict.keys(), desc=f'Flagging bad pixels for Obs {oid}', leave=False):
                # Search for existing DQ flags file
                file = odict[k]['file']
                file_dq = file.replace('.fits', save_suffix)
                if os.path.exists(file_dq) and (force==False):
                    # Load existing DQ flags
                    dq = fits.getdata(file_dq, extname='DQ')
                else:
                    # Update dq array
                    dq = self._flag_bad_pixels(odict[k]['data'], odict[k]['dq'], **kwargs)
                    if save_dq_flags:
                        hdr_dq = fits.getheader(file, extname='DQ')
                        hdu = fits.PrimaryHDU(header=odict[k]['hdr0'])
                        hdul = fits.HDUList([hdu])
                        hdul.append(fits.ImageHDU(dq, header=hdr_dq))
                        hdul.writeto(file_dq, overwrite=True)
                odict[k]['dq'] = dq

        # Call this same function in the reference objects
        if self.ref_objs is not None:
            for ref_obj in self.ref_objs:
                ref_obj.flag_bad_pixels(nsig_spatial=nsig_spatial, nsig_temporal=nsig_temporal,
                                        ntemporal_limit=ntemporal_limit, niter=niter,
                                        save_dq_flags=save_dq_flags, save_suffix=save_suffix, force=force)

    def get_expected_pos(self):
        """Get the expected stellar positions based on header info"""
        from webbpsf_ext.imreg_tools import get_expected_loc

        # Get the expected location for each observation
        if len(self.obs_dict)==0:
            raise ValueError("Run generate_obs_dict() first.")
        
        for oid in self.obsids:
            # Observation dictionary
            oid_dict = self.obs_dict[oid]
            for k in oid_dict.keys():
                # Exposure dictionary
                d = oid_dict[k]
                ap = d['ap']
                hdr0 = d['hdr0']
                xind, yind = get_expected_loc(hdr0)
                xsci, ysci = (xind+1, yind+1)
                loc_dict = {}
                for frame in ['det', 'sci', 'tel', 'idl']:
                    if frame=='sci':
                        loc = [xsci, ysci]
                    elif frame=='det':
                        loc = ap.sci_to_det(xsci, ysci)
                    elif frame=='tel':
                        loc = ap.sci_to_tel(xsci, ysci)
                    elif frame=='idl':
                        loc = ap.sci_to_idl(xsci, ysci)
                    loc_dict[frame] = np.asarray(loc)
                d['loc_exp'] = loc_dict

        # Call this same function in the reference objects
        if self.ref_objs is not None:
            for ref_obj in self.ref_objs:
                ref_obj.get_expected_pos()

    def expected_pos(self, obsid, dither, frame='sci'):
        """Return expected position for a given observation"""
        return self.obs_dict[obsid][dither]['loc_exp'][frame]

    def get_filter_offset(self, arcsec=False, **kwargs):
        """Read in known offsets between the TA filter and the science filter
        
        Only applicable for coronagraphic observations.
        """

        if self.nrc is None:
            self.create_nircam_object()

        filter = self.filter
        image_mask = self.nrc.image_mask

        # If already in TA filter, don't need to shift
        if filter in ['F335M', 'F210M']:
            dx_filt = dy_filt = 0
        elif image_mask is None:
            dx_filt = dy_filt = 0
        else:
            is_lwb = 'LWB' in image_mask
            is_swb = 'SWB' in image_mask

            if is_lwb:
                filt_file = 'filter_offsets_lwb.txt' 
            elif is_swb:
                filt_file = 'filter_offsets_swb.txt'
            else:
                filt_file = 'filter_offsets_rnd.txt'
            
            tbl_dir = self._filter_offset_dir
            filt_path = filt_file if tbl_dir is None else os.path.join(tbl_dir, filt_file)
            tbl_filts = ascii.read(filt_path)

            # Pixel or arcsec columns?
            dx_key = 'dx_asec' if arcsec else 'dx_pix'
            dy_key = 'dy_asec' if arcsec else 'dy_pix'

            # Get filter offsets from TA filter to observed filter
            ind = np.where(tbl_filts['filter'] == filter)[0]
            # If filter not found and is LWB or SWB, default to filter_offsets.txt
            if len(ind) == 0 and (is_lwb or is_swb):
                filt_path = os.path.join(tbl_dir, 'filter_offsets.txt')
                tbl_filts = ascii.read(filt_path)
                ind = np.where(tbl_filts['filter'] == filter)[0]

            # If still not found, continue with no offset
            if len(ind) == 0:
                # _log.warning(f'No filter offset found for {image_mask} {filter}')
                dx_filt = dy_filt = 0
            elif len(ind)>1:
                # _log.warning(f'Multiple filter offsets found for {image_mask} {filter}')
                dx_filt = tbl_filts[dx_key][ind[0]]
                dy_filt = tbl_filts[dy_key][ind[0]]
            else:
                dx_filt = tbl_filts[dx_key][ind[0]]
                dy_filt = tbl_filts[dy_key][ind[0]]

        return np.array([dx_filt, dy_filt])

    def create_stellar_spectrum(self, name=None, return_sp=False, **kwargs):
        """Create stellar spectrum"""
        from webbpsf_ext.spectra import source_spectrum
        from webbpsf_ext import bp_2mass

        # Check if generate_obs_dict has been run
        if len(self.obs_dict)==0:
            raise ValueError("Run generate_obs_dict() first.")

        if name is None:
            hdr = self.obs_dict[self.obsids[0]][0]['hdr0']
            try:
                name = hdr['TARGNAME']
                kwargs_src = stellar_arguments(name, **kwargs)
            except:
                name = hdr['TARGPROP']
                kwargs_src = stellar_arguments(name, **kwargs)
        else:
            kwargs_src = stellar_arguments(name, **kwargs)

        # Directory housing VOTables 
        # http://vizier.u-strasbg.fr/vizier/sed/
        # votdir = '../votables/'

        # Create spectral object and fit
        self._sp_kws = kwargs_src
        src = source_spectrum(**kwargs_src)
        src.fit_SED(use_err=False, robust=False, wlim=[1,10], IR_excess=True, verbose=False)

        # Plot SED if desired
        # src.plot_SED(xr=[0.5,30])

        # Final source spectrum
        if return_sp:
            return src.sp_model
        else:
            self.sp_sci = src.sp_model

    def create_nircam_object(self, fov_pix=65, oversample=None, obs_hci=True, **kwargs):
        """Create NIRCam object"""

        from ..nrc_utils import conf

        log_prev = conf.logging_level
        setup_logging('WARN', verbose=False)

        if self.sp_sci is None:
            self.create_stellar_spectrum()

        fpath = self.obs_dict[self.obsids[0]][0]['file']
        kwargs2 = {'sp_sci': self.sp_sci, 'dist_sci': self._sp_kws.get('dist')} if obs_hci else {}
        kwargs.update(kwargs2)
        self.nrc = nrc_from_file(fpath, fov_pix, oversample=oversample, obs_hci=obs_hci, **kwargs)

        # Reset logging level
        setup_logging(log_prev, verbose=False)

        # Call this same function in the reference objects
        if self.ref_objs is not None:
            for ref_obj in self.ref_objs:
                ref_obj.create_nircam_object(fov_pix=fov_pix, oversample=oversample, obs_hci=obs_hci, **kwargs)

        
    def _simulate_psf(self, coord_vals=None, coord_frame=None, 
                      fov_pix=None, oversample=None, sp=None,
                      focus=0, diffusion_sigma=0, return_xyoff=False):
        """Simulate PSF placed in center of array"""

        from webbpsf_ext.imreg_tools import recenter_psf
        from ..nrc_utils import conf

        if self.nrc is None:
            self.create_nircam_object()

        nrc = self.nrc
        if fov_pix is not None:
            nrc.fov_pix = fov_pix
        if oversample is not None:
            nrc.oversample = oversample

        apname = self.nrc.siaf_ap.AperName

        # Unit response to create effective PSF
        bp = nrc.bandpass
        sp = self.sp_sci
        sp_norm = sp.renorm(bp.unit_response(), 'flam', bp)

        focus = 0 if focus is None else focus
        diffusion_sigma = self.best_diffusion if diffusion_sigma is None else diffusion_sigma

        # Simulate PSF
        log_prev = conf.logging_level
        setup_logging('WARN', verbose=False)

        use_bg_psf = True if nrc.is_coron else False
        psf_over = gen_diffusion_psf(nrc, diffusion_sigma, defocus_waves_2um=focus,
                                     return_oversample=True, return_hdul=False,
                                     sp=sp_norm, coord_vals=coord_vals, coord_frame=coord_frame,
                                     use_bg_psf=use_bg_psf, normalize='exit_pupil')

        # Reposition oversampled synthetic PSF to the center of array 
        # using center of mass algorithm
        if nrc.is_coron:
            if oversample==1:
                halfwidth=1
            elif oversample<=3:
                # Prevent special case COM algorithm from not converging
                if ('LWB' in apname) and 'F4' in nrc.filter:
                    halfwidth=5
                else:
                    halfwidth=3
            elif oversample<=5:
                halfwidth=7
        else:
            halfwidth=15
        _, xyoff_psfs_over = recenter_psf(psf_over, niter=3, halfwidth=halfwidth)

        # Convert to detector pixels
        xyoff_psf = np.array(xyoff_psfs_over) / oversample
        nrc.psf_offset_to_center = xyoff_psf

        # Regenerate coronagraphic PSF
        # Don't include diffusion or IPC yet. That comes later.
        if nrc.is_coron:
            # For certain coronagraphic observations, fit occulted 
            # obs with bg PSF because of poor pointing
            if (self.pid in [1412]) and (self.obsid in [2, 4, 5]):
                use_bg_psf = True
            elif self.pid in [1536, 1537, 1538] and '_MASK' in apname:
                # These observations were intentionally offset 5" to the south
                use_bg_psf = True
            else:
                use_bg_psf = False
                
            psf_over = gen_diffusion_psf(nrc, diffusion_sigma, defocus_waves_2um=focus,
                                         return_oversample=True, return_hdul=False,
                                         sp=sp, coord_vals=coord_vals, coord_frame=coord_frame,
                                         use_bg_psf=use_bg_psf, normalize='exit_pupil')
            
        # Reset logging level
        setup_logging(log_prev, verbose=False)

        psf_over = nrc.recenter_psf(psf_over, sampling=oversample, shift_func=fourier_imshift)
        # xyoff_psfs_over is the shift required to move simulated PSF to center of array
        if return_xyoff:
            return psf_over, xyoff_psfs_over
        else:
            return psf_over
    

    def simulate_psfs(self, xysub, use_com=True, force=False, diffusion_sigma=0):
        """Simulate PSFs for each dither position
        
        For corongraphic observations, this only simulates a single PSF
        that is centered on the coronagraphic mask.
        """

        from webbpsf_ext.imreg_tools import load_cropped_files
        from webbpsf_ext.imreg_tools import get_com, get_expected_loc

        if self.nrc is None:
            self.create_nircam_object(fov_pix=xysub)

        # Ensure stellar spectrum has been created
        if self.sp_sci is None:
            self.create_stellar_spectrum()

        # Saved file
        obs_dict = self.obs_dict
        save_dir = os.path.dirname(obs_dict[self.obsids[0]][0]['file'])

        # Coronagraphic observations?
        is_coron = self.nrc.is_coron

        if is_coron:
             # No need to do multiple PSFs if coronagraphic observations
            ndith = 1
        else:
            # Get all the file names
            files = []
            ndither = 0
            for oid in self.obsids:
                odict = obs_dict[oid]
                for k in odict.keys():
                    if odict[k].get('files') is not None:
                        files.extend(odict[k]['files'])
                    else:
                        files.append(odict[k]['file'])
                    ndither += 1
            files = [os.path.basename(file) for file in files]

            # Crop array around star
            find_func = get_com if use_com else get_expected_loc
            res = load_cropped_files(save_dir, files, xysub=15, bgsub=False, find_func=find_func)
            xyind_arr = res[2].reshape([ndither,-1])

            # Get x/y loc for each observation
            # It will be the middle of the subarray; xyind_arr are subarray coords
            xloc_ind = xyind_arr[:,:2].mean(axis=1)
            yloc_ind = xyind_arr[:,2:].mean(axis=1)
            xy_ind = np.array([xloc_ind, yloc_ind]).T
            xy_sci = xy_ind + 1

            # Convert to 'tel' V2V3 coords
            ap = self.nrc.siaf_ap
            xy_tel = np.array([ap.sci_to_tel(xy[0],xy[1]) for xy in xy_sci])

            ndith = len(xy_tel)

        # Simulate PSFs
        # Create oversampled PSFs for each dither location
        osamp = self.nrc.oversample
        if (self.psfs_over is None) or (force==True):
            psfs_over = []
            xyoff_psfs_over = []

            itervals = range(ndith) if ndith==1 else trange(ndith, desc='Oversampled PSFs', leave=False)
            for i in itervals:
                if is_coron:
                    coord_vals = coord_frame = None
                else:
                    v2, v3 = xy_tel[i]
                    coord_vals, coord_frame = ((v2,v3), 'tel')

                res = self._simulate_psf(coord_vals=coord_vals, coord_frame=coord_frame,
                                         fov_pix=xysub, oversample=osamp, return_xyoff=True,
                                         diffusion_sigma=diffusion_sigma)
                psfs_over.append(res[0])
                xyoff_psfs_over.append(res[1])
            psfs_over = np.asarray(psfs_over)
            xyoff_psfs_over = np.array(xyoff_psfs_over)

            # Save for later
            self.psfs_over = psfs_over
            self.xyoff_psfs_over = xyoff_psfs_over
        else:
            psfs_over = self.psfs_over
            xyoff_psfs_over = self.xyoff_psfs_over

        return psfs_over, xyoff_psfs_over


    def get_star_positions(self, xysub=65, bgsub=False, use_com=True,
                           med_dithers=True, ideal_sgd=True, save=True, force=False):
        """Find the offset between the expected and actual position
        
        Updates self.xy_loc_ind, self.xyshift, and self.xy_mask_offset

        Parameters
        ----------
        xysub : int
            Size of subarray to use for PSF fitting
        bgsub : bool
            If True, then subtract the background from the cropped image.
            The background region is defined as r>0.7*xysub/2.
        use_com : bool
            Use center of mass algorithm to find star position. 
            Otherwise, use expected location.
        med_dithers : bool
            If True, then median combine the dithered images before fitting.
        save : bool
            Save the star positions to a file
        force : bool
            Force re-calculation of star positions.
        """
        from webbpsf_ext.imreg_tools import find_pix_offsets, load_cropped_files
        from webbpsf_ext.imreg_tools import get_com, get_expected_loc, get_sgd_offsets
        from webbpsf_ext.image_manip import get_im_cen, bp_fix

        # Check if generate_obs_dict has been run
        if len(self.obs_dict)==0:
            raise ValueError("Run generate_obs_dict() first.")

        if self.xy_loc_ind is not None and (force==False):
            _log.info("Star positions already found. Set force=True to re-run.")
            return
        
        if self.nrc is None:
            self.create_nircam_object(fov_pix=xysub)

        keys = list(self.obs_dict[self.obsids[0]][0].keys())
        if 'xyloc' in keys:
            _log.info("Data arrays already updated. Run generate_obs_dict() to start over.")
            return

        obs_dict = self.obs_dict

        # Saved file
        save_dir = os.path.dirname(obs_dict[self.obsids[0]][0]['file'])
        save_str0 = '_obs' + '.'.join([str(obs) for obs in self.obsids])
        save_str1 = '_com' if use_com else '_exp'
        save_str2 = '_med' if med_dithers else ''
        save_str3 = '_sim' if self.is_sim else ''
        save_file = f'star_positions_{self.filter}{save_str0}{save_str1}{save_str2}{save_str3}.json'
        save_path = os.path.join(save_dir, save_file)
        if os.path.exists(save_path) and (force==False):
            _log.info(f"Loading star positions from {save_path}")
            with open(save_path, 'r') as f:
                data = json.load(f)
            self.xy_loc_ind = np.array(data['xy_loc_ind'])
            self.xyshift = np.array(data['xyshift'])

        else:
            # Get all the file names
            files = []
            ndither = 0
            for oid in self.obsids:
                odict = obs_dict[oid]
                for k in odict.keys():
                    if odict[k].get('files') is not None:
                        files.extend(odict[k]['files'])
                    else:
                        files.append(odict[k]['file'])
                    ndither += 1
            files = [os.path.basename(file) for file in files]

            # Crop array around star
            find_func = get_com if use_com else get_expected_loc
            _log.info("Loading cropped files and fixing pixels...")
            res = load_cropped_files(save_dir, files, xysub=xysub, bgsub=bgsub, find_func=find_func)
            imsub_arr, dqsub_arr, xyind_arr, bp_masks = res

            # Replace imsub_arr data with simulated data
            if self.is_sim:
                # print(imsub_arr.shape, xyind_arr.shape)
                for i, oid in enumerate(self.obsids):
                    odict = self.obs_dict[oid]
                    for k in odict.keys():
                        x1, x2, y1, y2 = xyind_arr[i]
                        data_sim = odict[k]['data']
                        data_shape = imsub_arr[i].shape
                        if len(data_sim.shape) == 2:
                            imsub_arr[i] = data_sim[y1:y2, x1:x2].reshape(data_shape)
                        else:
                            imsub_arr[i] = data_sim[:, y1:y2, x1:x2].reshape(data_shape)


            if imsub_arr.shape[0] != ndither:
                nint = imsub_arr.shape[0] // ndither
                ny, nx = imsub_arr.shape[-2:]
                sh_orig = (ndither, nint, ny, nx)
                # Reshape to expected shapes
                imsub_arr = imsub_arr.reshape(sh_orig)
                dqsub_arr = dqsub_arr.reshape(sh_orig)
                bp_masks = bp_masks.reshape(sh_orig)
                xyind_arr = xyind_arr.reshape([ndither,-1,4])[:,0,:]
            else:
                sh_orig = imsub_arr.shape
                ny, nx = sh_orig[-2:]

            # Fix bad pixels
            # Flag additional bad pixels
            for i in range(ndither):
                dqsub_arr[i] = self._flag_bad_pixels(imsub_arr[i], dqsub_arr[i])
            imsub_arr = imsub_arr.reshape([-1, ny, nx])
            dqsub_arr = dqsub_arr.reshape([-1, ny, nx])
            bp_masks = bp_masks.reshape([-1, ny, nx])

            for i in range(ndither):
                bp1 = bp_masks[i]
                bp2 = get_dqmask(dqsub_arr[i], ['DO_NOT_USE']) > 0
                bp = bp1 | bp2
                im = bp_fix(imsub_arr[i], bpmask=bp, in_place=True, niter=3)
                border = get_dqmask(dqsub_arr[i], ['FLUX_ESTIMATED', 'REFERENCE_PIXEL']) > 0
                im[border] = 0
                imsub_arr[i] = im

            imsub_arr = imsub_arr.reshape(sh_orig)
            dqsub_arr = dqsub_arr.reshape(sh_orig)
            bp_masks = bp_masks.reshape(sh_orig)

            # Simulate PSFs for each dither location
            _log.info("Simulating PSFs...")
            psfs_over, _ = self.simulate_psfs(xysub, use_com=use_com, force=force)

            # return imsub_arr, psfs_over, bp_masks, xyind_arr

            # Choose region to fit PSF
            nrc = self.nrc
            apname = nrc.siaf_ap.AperName
            if nrc.is_coron and ('TAMASK' in apname):
                # Target acquisitions
                rin = 0
            elif self.pid in [1536, 1537, 1538]:
                # Certain coronagraphic observations are off the mask by ~5"
                rin = 0 
            elif nrc.is_coron:
                # Coronagraphic observations
                rin = 4
            else:
                # Direct Imaging
                rin  = 0

            apname     = obs_dict[self.obsids[0]][0]['apname']
            apname_pps = obs_dict[self.obsids[0]][0]['apname_pps']
            if self.filter in ['F200W']:
                xylim_pix = (-5,5)
            elif apname != apname_pps:
                xylim_pix = (-5,5)
            elif nrc.is_coron:
                xylim_pix = (-3,3)
            else:
                xylim_pix = (-5,5)

            # Find best sub-pixel fit location for all images
            xy_loc_all = []
            # print("Finding offsets...")
            osamp = self.nrc.oversample
            itervals = trange(ndither, desc='Finding offsets', leave=False)
            for i in itervals:
                # Get the PSF for this dither
                psf_over = psfs_over[0] if psfs_over.shape[0]==1 else psfs_over[i]
                # Get the image(s) for this dither
                imsub = imsub_arr[i]
                if len(imsub.shape)==3 and med_dithers:
                    imsub = np.nanmedian(imsub, axis=0)
                # Get the bad pixel mask(s) for this dither
                bpmask = bp_masks[i]
                if len(bpmask.shape)==3 and med_dithers:
                    bpmask = np.bitwise_and.reduce(bpmask, axis=0)
                # Find the best PSF shifts to match science images
                xysh_pix = find_pix_offsets(imsub, psf_over, psf_osamp=osamp, rin=rin, xylim_pix=xylim_pix,
                                            kipc=self.kipc, kppc=self.kppc, diffusion_sigma=self.best_diffusion,
                                            bpmask_arr=bpmask, phase=False, corr_avg=True)

                # Multiply by -1 to get current position relative to center of cropped subarray
                xysh_pix *= -1

                im = imsub if len(imsub.shape)==2 else imsub[0]
                xc_sub, yc_sub = get_im_cen(im)
                # Get locations within the subarray
                xy_loc_sub = xysh_pix + np.array([xc_sub, yc_sub])
                # Locations in full science frame
                xy_loc = xy_loc_sub + xyind_arr[i, 0::2]

                xy_loc_all.append(xy_loc)
            xy_loc_all = np.array(xy_loc_all)

            # Always make a 2D array assuming pointing is really good
            if len(xy_loc_all.shape)==3:
                xy_loc_all = np.mean(xy_loc_all, axis=1)

            # Index positions of star in reduced data
            self.xy_loc_ind = xy_loc_all

            # Get shift values necessary to center the star in reduce image array
            im = obs_dict[self.obsids[0]][0]['data']
            if len(im.shape)==3:
                im = im[0]
            self.xyshift = get_im_cen(im) - self.xy_loc_ind

            # Special case for SGD, assume ideal offsets
            # hdr0 = obs_dict[self.obsids[0]][0]['hdr0']
            # if (hdr0.get('SUBPXPAT') == 'SMALL-GRID-DITHER') and ideal_sgd:
            if self.is_sgd and ideal_sgd:
                sgd_patt = self.sgd_pattern #hdr0.get('SMGRDPAT', None)
                xoff_asec, yoff_asec = get_sgd_offsets(sgd_patt)
                xoff_pix = xoff_asec / nrc.pixelscale
                yoff_pix = yoff_asec / nrc.pixelscale

                # Update xyshift and xy_loc_ind
                self.xyshift = np.mean(self.xyshift, axis=0) - np.array([xoff_pix, yoff_pix]).T
                self.xy_loc_ind = get_im_cen(im) - self.xyshift

            # Save xy_loc_ind and xyshift to file
            if save:
                _log.info(f"Saving star positions to {save_path}")
                save_data = {'xy_loc_ind': self.xy_loc_ind, 'xyshift': self.xyshift}
                with open(save_path, 'w') as f:
                    json.dump(save_data, f, cls=NumpyArrayEncoder)

            del imsub_arr, dqsub_arr, bp_masks

        self._update_mask_offsets()

        # print(self.xy_loc_ind)
        # print(self.xyshift)

        # Call this same function in the reference objects
        if self.ref_objs is not None:
            for ref_obj in self.ref_objs:
                ref_obj.get_star_positions(xysub=xysub, bgsub=bgsub, use_com=use_com,
                                           med_dithers=med_dithers, save=save, force=force)

    def _update_mask_offsets(self):
        """Determine mask offsets and update self.xy_mask_offset and nrc.pointing_info"""

        # Get the offsets from nominal mask position for coronagraphic observations
        filt_offset = self.get_filter_offset(arcsec=False)
        siaf_ap = self.siaf_ap
        sci_ref = np.array([siaf_ap.XSciRef, siaf_ap.YSciRef])
        # Convert location to sci coords by adding 1
        # Subtract the filter offset
        if self.nrc.is_coron:
            self.xy_mask_offset = (self.xy_loc_ind + 1) - sci_ref - filt_offset
        else:
            self.xy_mask_offset = np.zeros_like(self.xy_loc_ind)

        # Update nrc.pointing_info
        if self.nrc is not None:
            # Ensure 0 random pointing error added
            self.nrc.pointing_info['slew_std'] = 0
            self.nrc.pointing_info['fsm_std'] = 0

            # Offsets in terms of arcsec
            pixelscale = self.nrc.pixelscale
            if self.is_sgd:
                self.nrc.pointing_info['ref'] = self.xy_mask_offset * pixelscale
                self.nrc.pointing_info['sgd_type'] = self.sgd_pattern 
            else:
                self.nrc.pointing_info['roll1'] = self.xy_mask_offset[0] * pixelscale
                self.nrc.pointing_info['roll2'] = self.xy_mask_offset[1] * pixelscale

    def _replace_data_with_sim(self, med_dithers=True, **kwargs):
        """Replace data with simulated PSFs"""

        replace_data_with_sim(self,**kwargs)

        # Call this same function in the reference objects
        if self.ref_objs is not None:
            for ref_obj in self.ref_objs:
                ref_obj._replace_data_with_sim(**kwargs)

    def shift_to_center_int(self, med_dithers=False, return_results=False, odd_shape=True):
        """Expand and shift images to place star roughly in center of array
        
        Does not perform any fractional shifts. Only integer shifts are applied.

        Parameters
        ----------
        med_dithers : bool
            If True, median combine images for each dither before shifting.
            Will overwrite the original data in the obs_dict with the median combined data.
        return_results : bool
            If True, return results in a dictionary and do not overwrite self.obs_dict.
        odd_shape : bool
            If True, then pad to odd dimensions. Otherwise, pad to even dimensions.
        """

        from webbpsf_ext.maths import round_int
        from webbpsf_ext.imreg_tools import crop_image
        from webbpsf_ext.image_manip import get_im_cen
        from jwst.datamodels import dqflags

        # Determine if SGD data
        # hdr0 = self.obs_dict[self.obsids[0]][0]['hdr0']
        is_sgd = self.is_sgd # hdr0.get('SUBPXPAT') == 'SMALL-GRID-DITHER'

        # Enforce med_dithers=True for SGD data
        # if is_sgd and not med_dithers:
        #     _log.warning("Forcing med_dithers=True for SGD data.")
        #     med_dithers = True

        imarr = []
        dqarr = []
        errarr = []
        bparr = []
        for oid in self.obsids:
            odict = self.obs_dict[oid]
            for k in odict.keys():
                im = odict[k]['data']
                if len(im.shape)==3 and med_dithers:
                    im = np.nanmedian(im, axis=0)
                imarr.append(im)

                err = odict[k]['err']
                if len(err.shape)==3 and med_dithers:
                    err = np.nanmedian(err, axis=0)
                errarr.append(err)

                dq = odict[k]['dq']
                if len(dq.shape)==3 and med_dithers:
                    dq = np.bitwise_and.reduce(dq, axis=0)
                dqarr.append(dq)

                bpmask = get_dqmask(dq, ['DO_NOT_USE']) > 0
                if len(bpmask.shape)==3 and med_dithers:
                    bpmask = np.bitwise_and.reduce(bpmask, axis=0)
                bparr.append(bpmask)
                
        imarr = np.asarray(imarr)
        errarr = np.asarray(errarr)
        dqarr = np.asarray(dqarr)
        bparr = np.asarray(bparr)

        imshape_orig = imarr.shape
        ny, nx = imshape_orig[-2:]
        ndith = imshape_orig[0]
        # Number of images per dither
        if len(imshape_orig)==3:
            nimg_per_dither = 1
        elif len(imshape_orig)==4:
            nimg_per_dither = imshape_orig[1]
        else:
            raise ValueError(f"imarr array has unexpected shape {imshape_orig}.")

        # Integer values to offset image to place star in center of image
        xy_loc_all = self.xy_loc_ind
        xyshift = self.xyshift
        xy_loc_shift = self.xy_loc_ind.copy()
        if (nimg_per_dither==1) and (len(xyshift.shape)==3):
            # Reduce to single shift value per dither
            xyshift = np.mean(xyshift, axis=1)
            xy_loc_shift = np.mean(xy_loc_shift, axis=1)

        # Determine pad size and number of shift values per dither
        if len(xyshift.shape)==2:
            xoff_int = round_int(xyshift[:, 0])
            yoff_int = round_int(xyshift[:, 1])
            # Number of shift values per dither
            nsh_per_dither = 1
        elif len(xyshift.shape)==3:
            xoff_int = round_int(xyshift[:, :, 0])
            yoff_int = round_int(xyshift[:, :, 1])
            # Number of shift values per dither
            nsh_per_dither = xyshift.shape[1]
        else:
            raise ValueError(f"xyshift array has unexpected shape {xyshift.shape}.")
        
        # Get padded array size for shifting
        pad_vals = 2 * int(np.max(np.abs(np.concatenate([xoff_int, yoff_int]))))
        ny_pad, nx_pad = (ny+pad_vals, nx+pad_vals)
        nxy_pad = np.max([nx_pad, ny_pad])

        is_even = np.mod(nxy_pad,2)==0
        is_odd = not is_even

        # Remove 1 pixel if desired shape is not of right type (odd/even)
        if (odd_shape and is_even) or (not odd_shape and is_odd):
            nxy_pad -= 1

        imarr_shift = []
        errarr_shift = []
        dqarr_shift = []
        bparr_shift = []
        for i in range(ndith):
            # Case of single image per dither
            if nimg_per_dither==1:
                # Use first (nominal) dither position to shift all images if SGD data
                xy_loc = xy_loc_all[0] if is_sgd else xy_loc_all[i]
                im, xy = crop_image(imarr[i], nxy_pad, xyloc=xy_loc, return_xy=True)
                err = crop_image(errarr[i], nxy_pad, xyloc=xy_loc, fill_val=np.nanmax(errarr))
                fill_val = dqflags.pixel['FLUX_ESTIMATED'] | dqflags.pixel['DO_NOT_USE']
                dq = crop_image(dqarr[i], nxy_pad, xyloc=xy_loc, fill_val=fill_val)
                bp = crop_image(bparr[i], nxy_pad, xyloc=xy_loc, fill_val=True)
                imarr_shift.append(im)
                errarr_shift.append(err)
                dqarr_shift.append(dq)
                bparr_shift.append(bp)
                xy_loc_shift[i,0] -= xy[0]
                xy_loc_shift[i,1] -= xy[2]
            else:
                imlist = []
                errlist = []
                dqlist = []
                bplist = []
                for j in range(nimg_per_dither):
                    if is_sgd:
                        # Use first (nominal) dither position to shift all images
                        xy_loc = xy_loc_all[0] if nsh_per_dither==1 else robust.mean(xy_loc_all[0,:], axis=1)
                        # raise RuntimeError("SGD should have med_dithers=True, so not sure how we got here!")
                    else:
                        xy_loc = xy_loc_all[i] if nsh_per_dither==1 else xy_loc_all[i,j]
                    im, xy = crop_image(imarr[i,j], nxy_pad, xyloc=xy_loc, return_xy=True)
                    err = crop_image(errarr[i,j], nxy_pad, xyloc=xy_loc, fill_val=np.nanmax(errarr))
                    fill_val = dqflags.pixel['FLUX_ESTIMATED'] | dqflags.pixel['DO_NOT_USE']
                    dq = crop_image(dqarr[i,j], nxy_pad, xyloc=xy_loc, fill_val=fill_val)
                    bp = crop_image(bparr[i,j], nxy_pad, xyloc=xy_loc, fill_val=True)
                    imlist.append(im)
                    errlist.append(err)
                    dqlist.append(dq)
                    bplist.append(bp)
                    if nsh_per_dither==2:
                        xy_loc_shift[i,j,0] -= xy[0]
                        xy_loc_shift[i,j,1] -= xy[2]
                imarr_shift.append(np.asarray(imlist))
                errarr_shift.append(np.asarray(errlist))
                dqarr_shift.append(np.asarray(dqlist))
                bparr_shift.append(np.asarray(bplist))
                if nsh_per_dither==1:
                    xy_loc_shift[i,0] -= xy[0]
                    xy_loc_shift[i,1] -= xy[2]

        imarr_shift = np.asarray(imarr_shift)
        errarr_shift = np.asarray(errarr_shift)
        dqarr_shift = np.asarray(dqarr_shift)
        bparr_shift = np.asarray(bparr_shift)

        ny_fin, nx_fin = imarr_shift.shape[-2:]
        im_temp = imarr_shift.reshape([-1,ny_fin,nx_fin])[0]
        xyshift_new = get_im_cen(im_temp) - xy_loc_shift

        if return_results:
            out = {
                'imarr_shift' : imarr_shift,
                'errarr_shift' : errarr_shift,
                'dqarr_shift' : dqarr_shift,
                'bparr_shift' : bparr_shift,
                'xy_loc_ind'  : xy_loc_shift,
                'xyshift'     : xyshift_new,
            }
            return out
        
        # Update odict with shifted images
        ii = 0
        for i, oid in enumerate(self.obsids):
            odict = self.obs_dict[oid]
            for k in odict.keys():
                del odict[k]['data'], odict[k]['dq'], odict[k]['err']
                odict[k]['data']  = imarr_shift[ii]
                odict[k]['err']   = errarr_shift[ii]
                odict[k]['dq']    = dqarr_shift[ii]
                odict[k]['bp']    = bparr_shift[ii]
                odict[k]['xyloc'] = xy_loc_shift[ii]
                ii += 1

        # Update class attributes
        self.xy_loc_ind = xy_loc_shift
        self.xyshift = xyshift_new

        # Call this same function in the reference objects
        if self.ref_objs is not None:
            for ref_obj in self.ref_objs:
                ref_obj.shift_to_center_int(med_dithers=med_dithers, return_results=False)

    def _get_dither_data(self, subsize=None, outer_rad=32, rebin=1, gstd_pix=None):

        from webbpsf_ext.image_manip import crop_image, frebin, bp_fix

        imarr = []
        dqarr = []
        bparr = []
        for oid in self.obsids:
            odict = self.obs_dict[oid]
            for k in odict.keys():
                # Images reduced to 2D
                im = odict[k]['data']
                if len(im.shape)==3:
                    im = np.nanmedian(im, axis=0)
                imarr.append(im)
                # DQ arrays
                dq = odict[k]['dq']
                if len(dq.shape)==3:
                    dq = np.bitwise_and.reduce(dq, axis=0)
                dqarr.append(dq)
                # Bad pixel masks
                bp = get_dqmask(dq, ['DO_NOT_USE']) > 0
                bp |= np.isnan(im)
                if 'bp' in odict[k].keys():
                    bp_temp = odict[k]['bp']
                    if len(bp_temp.shape)==3:
                        bp |= np.bitwise_and.reduce(bp_temp, axis=0)
                bparr.append(bp)

        imarr = np.asarray(imarr)
        dqarr = np.asarray(dqarr)
        bparr = np.asarray(bparr)

        # Crop images to subsize
        if subsize is None and (outer_rad is not None):
            subsize = outer_rad * 2 + 1
        if subsize is not None:
            # Ensure subsize is odd
            subsize = subsize + 1 if subsize % 2 == 0 else subsize
            imarr = crop_image(imarr, subsize)
            dqarr = crop_image(dqarr, subsize)
            bparr = crop_image(bparr, subsize)

        # Perform bad pixel fixing on all images
        ndither = imarr.shape[0]
        for i in range(ndither):
            bp = bparr[i]
            im = bp_fix(imarr[i], bpmask=bp, in_place=True, niter=10)
            border = get_dqmask(dqarr[i], ['FLUX_ESTIMATED', 'REFERENCE_PIXEL']) > 0
            im[border] = 0
            imarr[i] = im

        rebin = 1 if rebin is None else rebin
        if (rebin != 1):
            imarr = frebin(imarr, scale=rebin)
            dqarr = frebin(dqarr, scale=rebin, total=False).astype('uint32')
            bparr = frebin(bparr, scale=rebin, total=False).astype('bool')

        # Gaussian smoothing
        if gstd_pix is None:
            gstd_pix = 0
        if gstd_pix>0:
            kernel = Gaussian2DKernel(x_stddev=gstd_pix)
            for i, im in enumerate(imarr):
                imarr[i] = convolve(im, kernel, boundary='extend')

        del dqarr
        return imarr, bparr
    

    def get_dither_offsets(self, method='opencv', interp='lanczos', subsize=None, 
                           rebin=1, gstd_pix=None, inner_rad=10, outer_rad=32, weights=None,
                           coarse_limits=(-1.5,1.5), fine_limits=(-0.2,0.2), 
                           return_results=False, save=True, force=False, 
                           ideal_sgd=False, **kwargs):
        """Find the position offsets between dithered images via LSQ minimization

        Compares all dithered and roll images relative to each other, and stores the
        offsets in self.shift_matrix.

        Performs a coarse grid search to find the global minimum, and then a 
        fine grid search to find a more precise minimum. 

        Updates self.xyshift and self.xy_loc_ind with the new shift values optimized
        for PSF subtraction.

        """
        from webbpsf_ext.image_manip import crop_image, frebin, fshift, bp_fix, get_im_cen
        from webbpsf_ext.maths import dist_image
        from skimage.filters import window as sk_window
        
        def subtraction_metrics(im1, im2, bp1, bp2, xyshift):
            """Perform subtraction and return goodness of fit metrics (std, ssr)"""

            # Shift bp2 by (dx,dy)
            bp2 = bp2.astype('float')
            bp2 = fshift(bp2, xyshift[0], xyshift[1], interp='linear', pad=True, cval=1)
            bp2 = bp2 > 0
            bpmask = bp1 | bp2
            good_mask = mask & (im1 != im2) & ~bpmask

            # Get optimal scale factor between images
            im1_good = im1[good_mask]
            im2_good = im2[good_mask]
            cf = np.linalg.lstsq(im2_good.reshape([1,-1]).T, im1_good, rcond=None)[0]
            scale = cf[0]

            # Find indices within which good data exists
            xgood = np.where(np.sum(good_mask, axis=0) > 0)[0]
            x1, x2 = (xgood[0], xgood[-1])
            ygood = np.where(np.sum(good_mask, axis=1) > 0)[0]
            y1, y2 = (ygood[0], ygood[-1])

            # Crop down for faster processing
            im1_crop = im1[y1:y2+1, x1:x2+1]
            im2_crop = im2[y1:y2+1, x1:x2+1]
            good_mask_crop = good_mask[y1:y2+1, x1:x2+1]
            weights_crop = weights[y1:y2+1, x1:x2+1]

            # Subtract images after performing fractional shift
            diff = subtract_psf(im1_crop, im2_crop, xyshift=xyshift, psf_scale=scale, 
                                method=method, interp=interp)

            std = np.std(diff[good_mask_crop] * weights_crop[good_mask_crop])
            ssr = np.sum((diff[good_mask_crop] * weights_crop[good_mask_crop])**2)

            return std, ssr
        
        def build_maps(im1, im2, bp1, bp2, dx_arr, dy_arr):
            """Build map of LSQ subtraction metrics
            
            Returns a (nxy x nxy) array of std and square-summed residuals.
            """
            std_arr = []
            ssr_arr = []
            for dy in dy_arr:
                for dx in dx_arr:
                    std, ssr = subtraction_metrics(im1, im2, bp1, bp2, (dx, dy))
                    # Standard deviation of the difference
                    std_arr.append(std)
                    # Sum of squared residuals
                    ssr_arr.append(ssr)

            nxy = len(dx_arr)
            std_arr = np.array(std_arr).reshape(nxy, nxy)
            ssr_arr = np.array(ssr_arr).reshape(nxy, nxy)

            return std_arr, ssr_arr
        
        def find_best_offset(im1, im2, bp1, bp2, dxy_coarse_arr, dxy_fine_arr0,
                             use_ssr=True, use_std=True):
            """Best offset to move im2 to align with im1"""
            # Perform coarse grid search
            std_arr, ssr_arr = build_maps(im1, im2, bp1, bp2, dxy_coarse_arr, dxy_coarse_arr)
            # Find the minimum positions for std and ssr
            dy_idx, dx_idx = np.unravel_index(np.nanargmin(std_arr), std_arr.shape)
            xbest1, ybest1 = (dxy_coarse_arr[dx_idx], dxy_coarse_arr[dy_idx])
            dy_idx, dx_idx = np.unravel_index(np.nanargmin(ssr_arr), ssr_arr.shape)
            xbest2, ybest2 = (dxy_coarse_arr[dx_idx], dxy_coarse_arr[dy_idx])
            if use_std and not use_ssr:
                xbest, ybest = xbest1, ybest1
            elif use_ssr and not use_std:
                xbest, ybest = xbest2, ybest2
            elif use_ssr and use_std:
                xbest = (xbest1 + xbest2) / 2
                ybest = (ybest1 + ybest2) / 2
            else:
                raise ValueError("Must use at least one of use_std=True or use_ssr=True.")

            # Perfom fine grid search
            dx_fine_arr = xbest + dxy_fine_arr0
            dy_fine_arr = ybest + dxy_fine_arr0
            std_arr, ssr_arr = build_maps(im1, im2, bp1, bp2, dx_fine_arr, dy_fine_arr)

            # The best fit is then the weighted average of the inverse of the std and ssr maps
            xv_grid, yv_grid = np.meshgrid(dx_fine_arr, dy_fine_arr)
            inv_std_arr = 1 / std_arr
            xsh_fine1 = np.average(xv_grid, weights=inv_std_arr**2)
            ysh_fine1 = np.average(yv_grid, weights=inv_std_arr**2)
            inv_ssr_arr = 1 / ssr_arr
            xsh_fine2 = np.average(xv_grid, weights=inv_ssr_arr**2)
            ysh_fine2 = np.average(yv_grid, weights=inv_ssr_arr**2)

            if use_std and not use_ssr:
                xsh_fine, ysh_fine = xsh_fine1, ysh_fine1
            elif use_ssr and not use_std:
                xsh_fine, ysh_fine = xsh_fine2, ysh_fine2
            elif use_ssr and use_std:
                xsh_fine = (xsh_fine1 + xsh_fine2) / 2
                ysh_fine = (ysh_fine1 + ysh_fine2) / 2
            else:
                raise ValueError("Must use at least one of use_std=True or use_ssr=True.")

            return np.array([xsh_fine, ysh_fine])

        def get_ref_offset(ref_obj):
            """Determine offset of SGD reference data relative to science data"""

            imarr_sci, bparr_sci = self._get_dither_data(subsize=subsize, outer_rad=outer_rad, rebin=rebin, gstd_pix=gstd_pix)
            imarr_ref, bparr_ref = ref_obj._get_dither_data(subsize=subsize, outer_rad=outer_rad, rebin=rebin, gstd_pix=gstd_pix)

            ndither_sci = imarr_sci.shape[0]
            ndither_ref = imarr_ref.shape[0]

            shift_matrix = np.zeros((ndither_sci, ndither_ref, 2))
            for i in trange(ndither_sci, desc='Relative Offsets', leave=False):
                for j in range(ndither_ref):
                    # Get best offset within 0.01 pixels
                    im1, bp1 = imarr_sci[i], bparr_sci[i]
                    im2, bp2 = imarr_ref[j], bparr_ref[j]
                    xysh_best = find_best_offset(im1, im2, bp1, bp2, dxy_coarse_arr, dxy_fine_arr0)
                    shift_matrix[i,j] = xysh_best / rebin

            return shift_matrix

        obs_dict = self.obs_dict

        # Crop images to subsize
        if subsize is None and (outer_rad is not None):
            subsize = outer_rad * 2 + 1
        if subsize is not None:
            # Ensure subsize is odd
            subsize = subsize + 1 if subsize % 2 == 0 else subsize

        # Saved file
        save_dir = os.path.dirname(obs_dict[self.obsids[0]][0]['file'])
        save_str0 = '_obs' + '.'.join([str(obs) for obs in self.obsids])
        save_str1 = '_sim' if self.is_sim else ''
        save_str = f'_{method}_{interp}_sub{subsize}_rebin{rebin}_gstd{gstd_pix}_irad{inner_rad}_orad{outer_rad}{save_str1}'
        save_file = f'star_positions_{self.filter}{save_str0}{save_str}.json'
        save_path = os.path.join(save_dir, save_file)
        if os.path.exists(save_path) and (force==False):
            _log.info(f"Loading star positions from {save_path}")
            with open(save_path, 'r') as f:
                data = json.load(f)

            if return_results:
                return np.array(data['xyshift'])

            self.xy_loc_ind = np.array(data['xy_loc_ind'])
            self.xyshift = np.array(data['xyshift'])
            self.shift_matrix = np.array(data['shift_matrix'])
        else:

            imarr, bparr = self._get_dither_data(subsize=subsize, outer_rad=outer_rad, rebin=rebin, gstd_pix=gstd_pix)
            ndither = imarr.shape[0]

            # Set up coarse grid search
            xy1, xy2 = np.array(coarse_limits) * rebin
            dxy_coarse = 0.2 * rebin
            dxy_coarse_arr = np.arange(xy1, xy2+dxy_coarse, dxy_coarse)

            # Define a finer grid offsets
            # Make sure fine grid limits are at least 2x the coarse grid steps
            dxy_fine = 0.01 * rebin
            if 2 * dxy_coarse > fine_limits[1] - fine_limits[0]:
                fine_limits = (-dxy_coarse, dxy_coarse)
            xy1, xy2 = np.array(fine_limits) * rebin
            dxy_fine_arr0 = np.arange(xy1, xy2+dxy_fine, dxy_fine)

            # Weight via an inverse Gaussian window
            # weights = 1 - sk_window(('gaussian', 10*rebin), imarr.shape[-2:])
            if weights is None:
                weights = np.ones(imarr.shape[-2:])
            else:
                weights = crop_image(weights, imarr.shape[-2:], fill_val=0)
            
            inner_rad = 0 if inner_rad is None else inner_rad
            rho = dist_image(imarr[0])
            if outer_rad is not None:
                mask = (rho >= (inner_rad*rebin)) & (rho <= (outer_rad*rebin))
            else:
                mask = rho >= (inner_rad*rebin)

            shift_matrix = np.zeros((ndither, ndither, 2))
            for i in trange(ndither, desc='Relative Offsets', leave=False):
                for j in range(ndither):
                    if i==j:
                        continue
                    # Offset needed to move im2 to align with im1
                    im1, bp1 = imarr[i], bparr[i]
                    im2, bp2 = imarr[j], bparr[j]
                    xysh_best = find_best_offset(im1, im2, bp1, bp2, dxy_coarse_arr, dxy_fine_arr0)
                    shift_matrix[i,j] = xysh_best / rebin

            del imarr, bparr#, dqarr

            if return_results:
                return shift_matrix
            
            # Shift matrix that gives the best shift for all images relative to each other
            self.shift_matrix = shift_matrix

            # Update xyshift values to be consistent with shift_matrix offsets
            # Determine if SGD data to preserve relative offsets
            # hdr0 = obs_dict[self.obsids[0]][0]['hdr0']
            # is_sgd = hdr0.get('SUBPXPAT') == 'SMALL-GRID-DITHER'
            is_sgd = self.is_sgd

            # Don't update xyshift if SGD data
            if (not is_sgd) or (is_sgd and not ideal_sgd):
                xsh0, ysh0 = self.xyshift.T
                xsh0_mean = np.mean(xsh0)
                ysh0_mean = np.mean(ysh0)
                xysh_arr = []
                for i in range(ndither):
                    shift_avg = (shift_matrix[i,:] - shift_matrix[:,i]) / 2
                    xsh_i, ysh_i = shift_avg.T
                    xsh_i -= np.mean(xsh_i) - xsh0_mean
                    ysh_i -= np.mean(ysh_i) - ysh0_mean
                    xysh_arr.append([xsh_i, ysh_i])
                xysh_arr = np.array(xysh_arr)
                xysh_mean =  np.mean(xysh_arr, axis=0).T

            # New shifts necessary to center the star in their existing image arrays
            self.xyshift = xysh_mean

            # Update best-guess locations of star in existing image arrays
            data = self.obs_dict[self.obsids[0]][0]['data']
            im = data[0] if len(data.shape)==3 else data
            xy_cen = get_im_cen(im)
            self.xy_loc_ind = xy_cen - self.xyshift

            # Save xy_loc_ind and xyshift to file
            if save:
                _log.info(f"Saving star positions to {save_path}")
                save_data = {'xy_loc_ind': self.xy_loc_ind, 'xyshift': self.xyshift,
                            'shift_matrix': self.shift_matrix}
                with open(save_path, 'w') as f:
                    json.dump(save_data, f, cls=NumpyArrayEncoder)

        # Call this same function in the reference objects
        if self.ref_objs is not None:
            for ref_obj in self.ref_objs:
                ref_obj.get_dither_offsets(method=method, interp=interp, subsize=subsize, 
                                           rebin=rebin, gstd_pix=gstd_pix, inner_rad=inner_rad, 
                                           outer_rad=outer_rad, coarse_limits=coarse_limits, 
                                           fine_limits=fine_limits, return_results=False, 
                                           save=save, force=force, ideal_sgd=ideal_sgd, **kwargs)
                
                # Align reference observations to science observations
                return get_ref_offset(ref_obj)

    def align_images(self, rebin=1, gstd_pix=None, #ref_obs=None, 
                     med_dithers=False, method='opencv', interp='lanczos', 
                     new_shape=None, **kwargs):
        """Align all images to a common reference frame
        
        Adds 'data_aligned', 'dq_aligned', 'bp_aligned' to each observation dictionary in self.obs_dict.
        The data have had their bad pixels fixed. The dq arrays and bad pixel masks have been shifted to 
        match the new image locations.

        Parameters
        ----------
        ref_obs : tuple or None
            Tuple of observation number and dither position to align all images to. 
            If None, then align all images to center of array.
        rebin : int
            Factor to rebin images before shifting. Results will be updated to new scale.
        gstd_pix : float
            Standard deviation of Gaussian kernel for smoothing images before shifting.
        new_shape : tuple
            Crop/expand images to a specific new (ny,nx) shape. If None, then shapes 
            are automatically expanded based on shift amounts.
        """

        from webbpsf_ext.image_manip import fractional_image_shift, fshift, frebin, bp_fix, crop_image

        if self.shift_matrix is None:
            _log.warning("shift_matrix attribute is None. Did you want to get_dither_offsets() first?")
            
        # Check flux units in header
        # If in surface brightness units, then set total=False
        header = self.obs_dict[self.obsids[0]][0]['hdr1']
        flux_units = header['BUNIT']
        total = True if '/sr' in flux_units.lower() else False

        ref_obs=None
        if ref_obs is None:
            xsh0 = ysh0 = 0
        # else:
        #     oid0, pos0 = ref_obs
        #     # Determine if reference observation is in the current object or in a reference object
        #     if oid0 in self.obsids:
        #         obj = self
        #     elif (self.ref_objs is not None) and (oid0 in self.obsids_ref):
        #         obsids_ref = list(self.obsids_ref)
        #         obj = self.ref_objs[obsids_ref.index(oid0)]
        #     else:
        #         raise ValueError(f"Observation {oid0} not found in self.obsids or self.obsids_ref.")

        #     # Loop through dictionary to find index of reference observations
        #     ii = 0
        #     for oid in obj.obsids:
        #         odict = obj.obs_dict[oid]
        #         for k in odict.keys():
        #             if oid==oid0 and k==pos0:
        #                 ii_ref = ii
        #                 break
        #             ii += 1
        #     xsh0, ysh0 = obj.xyshift[ii_ref]

        ii = 0
        for oid in self.obsids:
            odict = self.obs_dict[oid]
            for k in tqdm(odict.keys(), desc=f'Centering Obs {oid}', leave=False):
                # Images reduced to 2D
                im = odict[k]['data'].copy()
                if len(im.shape)==3 and med_dithers:
                    im = np.nanmedian(im, axis=0)

                err = odict[k]['err'].copy()
                if len(err.shape)==3 and med_dithers:
                    err = np.sqrt(np.nanmean(err**2, axis=0))

                # DQ arrays
                dq = odict[k]['dq']
                if len(dq.shape)==3 and med_dithers:
                    dq = np.bitwise_and.reduce(dq, axis=0)
                    
                # Bad pixel masks
                bp = get_dqmask(dq, ['DO_NOT_USE']) > 0
                bp |= np.isnan(im)
                if len(bp.shape)==3 and med_dithers:
                    bp = np.bitwise_and.reduce(bp, axis=0)

                # Fix bad pixels
                if len(im.shape)==3:
                    for i in range(im.shape[0]):
                        bpmask = bp if len(bp.shape)==2 else bp[i]
                        # im[i] = bp_fix(im[i], bpmask=bpmask, in_place=False, niter=5)
                        im[i][bpmask] = np.nan
                else:
                    # im = bp_fix(im, bpmask=bp, in_place=False, niter=5)
                    im[bp] = np.nan
                border = get_dqmask(dq, ['FLUX_ESTIMATED', 'REFERENCE_PIXEL']) > 0
                im[border] = 0
                err[border] = np.nanmax(err)
                err[np.isnan(err)] = np.nanmax(err)

                xsh, ysh = self.xyshift[ii] - np.array([xsh0, ysh0])
                im_shift = fractional_image_shift(im, xsh, ysh, method=method, interp=interp,
                                                  oversample=rebin, gstd_pix=gstd_pix, 
                                                  return_oversample=True, total=total)
                
                var_shift = fractional_image_shift(err**2, xsh, ysh, method='fshift', interp='linear',
                                                   oversample=rebin, gstd_pix=gstd_pix, 
                                                   return_oversample=True, total=total)
                err_shift = np.sqrt(var_shift)
                
                bp_shift = frebin(bp.astype('float'), scale=rebin, total=False)
                bp_shift = fshift(bp_shift, xsh*rebin, ysh*rebin, interp='linear', pad=True, cval=1)
                bp_shift = bp_shift > 0

                dq_shift = frebin(dq.astype('float'), scale=rebin, total=False)
                xsh_dq = np.sign(xsh) * np.ceil(np.abs(xsh*rebin))
                ysh_dq = np.sign(ysh) * np.ceil(np.abs(ysh*rebin))
                fill_val = dqflags.pixel['FLUX_ESTIMATED'] | dqflags.pixel['DO_NOT_USE']
                dq_shift = fshift(dq_shift, xsh_dq, ysh_dq, pad=True, cval=fill_val)
                dq_shift = dq_shift.astype(dq.dtype)

                # If new shape is requested, crop/expand images
                xyloc = self.xy_loc_ind[ii] + np.array([xsh, ysh])
                if new_shape is not None:
                    im_shift, xyarr = crop_image(im_shift, new_shape, return_xy=True)
                    err_shift = crop_image(err_shift, new_shape)
                    bp_shift = crop_image(bp_shift, new_shape)
                    dq_shift = crop_image(dq_shift, new_shape)
                    # Update xyloc to reflect new image size
                    xyloc = xyloc - np.array([xyarr[0], xyarr[2]])

                odict[k]['data_aligned'] = im_shift
                odict[k]['err_aligned'] = err_shift
                odict[k]['bp_aligned'] = bp_shift
                odict[k]['dq_aligned'] = dq_shift
                odict[k]['xy_aligned'] = xyloc
                odict[k]['bin_aligned'] = rebin

                ii += 1

        # Call this same function in the reference objects
        if self.ref_objs is not None:
            for ref_obj in self.ref_objs:
                sci_shape = self.obs_dict[self.obsids[0]][0]['data_aligned'].shape[-2:]
                ref_obj.align_images(rebin=rebin, gstd_pix=gstd_pix, med_dithers=med_dithers, 
                                     method=method, interp=interp, new_shape=sci_shape, **kwargs)


    def _get_rdi_ref(self, obsids=None, dith_pos=None, bin_ints=1,
                     med_dithers=False, data_key='data_aligned'):
        """Get reference observation data to use for RDI subtraction

        Exclusively uses observations in the self.ref_objs list of object.

        Parameters
        ----------
        obsids : list or None
            Observation ID to use as reference data. If None, uses all obsids
            in the reference objects.
        dith_pos : int or None
            Dither position to use for reference data. If None, then use all dither 
            positions in obsids.
        bin_ints : int
            Number of integrations to average together for reference data.
        med_dithers : bool
            If True, median combine integrations for each dither before subtracting.
            In this case, bin_ints is ignored.
        data_key : str
            Key in obs_dict to use for data. Default is 'data_aligned'.
        """

        if 'data' in data_key:
            dq_key = data_key.replace('data', 'dq')
            is_err = False
        if 'err' in data_key:
            dq_key = data_key.replace('err', 'dq')
            is_err = True

        obsids_ref = self.obsids_ref
        if obsids is None:
            obsids = obsids_ref

        if isinstance(obsids, int):
            obsids = [obsids]


        ref_obs = []
        for obsid in obsids:
            # Select the correct reference object
            ref_obj = self.ref_objs[obsids_ref.index(obsid)]
            odict = ref_obj.obs_dict[obsid]

            # Get dither positions to grab data from
            if dith_pos is None:
                dith_keys = odict.keys()
            elif isinstance(dith_pos, int):
                dith_keys = [dith_pos]

            for k in dith_keys:
                imref = odict[k][data_key].copy()
                dqref = odict[k][dq_key]
                dqmask = get_dqmask(dqref, ['DO_NOT_USE']) > 0
                imref[dqmask] = np.nan

                ny, nx = imref.shape[-2:]

                sh_orig = imref.shape
                ndim = len(sh_orig)
                if ndim==3 and med_dithers:
                    if is_err:
                        imref = np.sqrt(np.nanmean(imref**2, axis=0))
                    else:
                        imref = np.nanmean(imref, axis=0)
                elif ndim==3 and bin_ints>1:
                    nbins = sh_orig[0] // bin_ints
                    imref = imref[:nbins*bin_ints]
                    if is_err:
                        imref = np.sqrt(np.nanmean(imref.reshape(nbins,bin_ints,ny,nx)**2, axis=1))
                    else:
                        imref = np.nanmean(imref.reshape(nbins,bin_ints,ny,nx), axis=1)

                ref_obs.append(imref)

        ref_obs = np.array(ref_obs).reshape([-1,ny,nx])
        return ref_obs.squeeze()

    def _get_roll_ref(self, obsid, dith_pos=None, bin_ints=1, 
                      med_dithers=False, data_key='data_aligned'):
        """Get reference observation data to use for roll subtraction
        
        Parameters
        ----------
        obsid : int
            Observation ID to use as science data
        dith_pos : int or None
            Dither position to use for reference data. If None, then use all dither 
            positions in different obsid.
        bin_ints : int
            Number of integrations to average together for reference data.
        med_dithers : bool
            If True, median combine integrations for each dither before subtracting.
            In this case, bin_ints is ignored.
        data_key : str
            Key in obs_dict to use for data. Default is 'data_aligned'.
        """

        if 'data' in data_key:
            dq_key = data_key.replace('data', 'dq')
            is_err = False
        if 'err' in data_key:
            dq_key = data_key.replace('err', 'dq')
            is_err = True

        ny, nx = self.obs_dict[self.obsids[0]][0][data_key].shape[-2:]

        ref_obs = []
        for oid2 in self.obsids:
            # Skip if same observation ID (e.g., same roll angle)
            if oid2==obsid:
                continue
            odict2 = self.obs_dict[oid2]

            # Get dither positions to grab data from
            if dith_pos is None:
                dith_keys = odict2.keys()
            elif isinstance(dith_pos, int):
                dith_keys = [dith_pos]

            for k in dith_keys:
                imref = odict2[k][data_key].copy()
                dqref = odict2[k][dq_key]
                dqmask = get_dqmask(dqref, ['DO_NOT_USE']) > 0
                imref[dqmask] = np.nan

                ny, nx = imref.shape[-2:]

                sh_orig = imref.shape
                ndim = len(sh_orig)
                if ndim==3 and med_dithers:
                    if is_err:
                        imref = np.sqrt(np.nanmean(imref**2, axis=0))
                    else:
                        imref = np.nanmean(imref, axis=0)
                elif ndim==3 and bin_ints>1:
                    nbins = sh_orig[0] // bin_ints
                    imref = imref[:nbins*bin_ints]
                    if is_err:
                        imref = np.sqrt(np.nanmean(imref.reshape(nbins,bin_ints,ny,nx)**2, axis=1))
                    else:
                        imref = np.nanmean(imref.reshape(nbins,bin_ints,ny,nx), axis=1)

                ref_obs.append(imref)

        ref_obs = np.array(ref_obs).reshape([-1,ny,nx])
        return ref_obs.squeeze()


    def psf_subtraction(self, obs_sub=None, med_dithers=False, data_key='data_aligned',
                        bin_ints_sci=1, bin_ints_ref=1, all_pos=True, do_pca=True,
                        do_rdi=True, **kwargs):
        """Perform roll subtraction on centered data
        
        Parameters
        ----------
        obs_sub : tuple or None
            Tuple of observation number and dither position to perform subtraction on.
            Used for doing subtraction on a subset of the data.
        med_dithers : bool
            If True, median combine integrations for each dither before subtracting.
            Overrides `bin_ints_sci` and `bin_ints_ref` parameters.
        bin_ints_sci : int
            Number of integrations to bin together for science data.
        bin_ints_ref : int
            Number of integrations to bin together for reference data.
        all_pos : bool
            If True, use all dither positions in other roll as reference for PSF subtraction.
            If False, only use the same dither position as the science data. Setting this to
            False only makes sense for direct imaging observations that have matched dithered
            positions in rolls and references.
        """

        from . import pca

        def basic_subtraction(imarr, imref, func_mean=np.nanmean):
            """Basic subtraction of images"""
            if len(imref.shape)==3:
                imref = func_mean(imref, axis=0)

            scale = np.nansum(imarr) / np.nansum(imref)
            return imarr - imref * scale
        
        def pca_subtraction(imarr, imarr_ref, bin_pix, **kwargs):
            # Build pca params dictionary for PSF subtraction
            # If corongraphic imaging using diameter of 5.2m rather than 6.5m
            is_coron = ('CORONMSK' in self.obs_dict[self.obsids[0]][0]['hdr0'])
            diam = 5.2 if is_coron else 6.5
            res_asec = 206265 * self.nrc.bandpass.pivot().to_value('m') / diam
            res_pix = res_asec / self.nrc.pixelscale
            fwhm_pix = 1.025 * res_pix

            fwhm_pix_bin = fwhm_pix * bin_pix #odict[k]['bin_aligned']
            pca_params = pca.build_pca_dict(fwhm_pix_bin, **kwargs)
            imdiff = pca.run_pca_subtraction(imarr, imarr_ref, pca_params)

            return imdiff, pca_params

        if do_rdi and self.ref_objs is None:
            _log.warning("No reference objects found for do_rdi=True. Setting do_rdi=False.")
            do_rdi = False

        # Check if obs_dict exists
        if len(self.obs_dict)==0:
            raise ValueError("Run generate_obs_dict() first.")

        # Check if data_key exists in obs_dict
        if data_key not in self.obs_dict[self.obsids[0]][0].keys():
            if data_key=='data_aligned':
                raise ValueError("Run align_images() first.")
            else:
                raise KeyError(f"Key '{data_key}' not found in obs_dict.")
            
        dq_key = data_key.replace('data', 'dq')
        bp_key = data_key.replace('data', 'bp')

        ny, nx = self.obs_dict[self.obsids[0]][0][data_key].shape[-2:]

        if obs_sub is not None:
            oid_sub, dith_pos_sub = obs_sub
        else:
            oid_sub = dith_pos_sub = None

        for oid in self.obsids:
            odict = self.obs_dict[oid]

            # Skip if obs id not in obs_sub
            if (oid_sub is not None) and (oid != oid_sub):
                continue

            # Compile all reference observations
            if all_pos and do_rdi:
                imarr_ref = self._get_rdi_ref(obsids=None, dith_pos=None, bin_ints=bin_ints_ref,
                                              med_dithers=med_dithers, data_key=data_key)
            elif all_pos and (not do_rdi):
                imarr_ref = self._get_roll_ref(oid, dith_pos=None, bin_ints=bin_ints_ref,
                                               med_dithers=med_dithers, data_key=data_key)

            for k in odict.keys():
                # Skip if dither position not in ref_obs
                if (dith_pos_sub is not None) and (k != dith_pos_sub):
                    continue

                # Get reference observations in this dither position only
                if (not all_pos) and do_rdi:
                    imarr_ref = self._get_rdi_ref(obsids=None, dith_pos=k, bin_ints=bin_ints_ref,
                                                  med_dithers=med_dithers, data_key=data_key)
                elif (not all_pos) and (not do_rdi):
                    imarr_ref = self._get_roll_ref(oid, dith_pos=k, bin_ints=bin_ints_ref,
                                                   med_dithers=med_dithers, data_key=data_key)

                imarr = odict[k][data_key].copy()
                sh_orig = imarr.shape
                ndim = len(sh_orig)

                # NaN out the bad pixels
                # bparr = odict[k][bp_key]
                dqarr = odict[k][dq_key]
                dqmask = get_dqmask(dqarr, ['DO_NOT_USE']) > 0
                imarr[dqmask] = np.nan

                if ndim==3 and med_dithers:
                    imarr = np.nanmedian(imarr, axis=0)
                elif ndim==3 and bin_ints_sci>1:
                    nbins = sh_orig[0] // bin_ints_sci
                    imarr = imarr[:nbins*bin_ints_sci]
                    imarr = np.nanmean(imarr.reshape(nbins,bin_ints_sci,ny,nx), axis=1)

                # print(imarr.shape, imarr_ref.shape)
                if do_pca:
                    imdiff, pca_params = pca_subtraction(imarr, imarr_ref, odict[k]['bin_aligned'], **kwargs)
                    odict[k]['pca_params'] = pca_params
                else:
                    imdiff = basic_subtraction(imarr, imarr_ref)

                # Clean up zeros
                # if len(imdiff.shape)==4:
                #     nim = imdiff.shape[0]
                #     for i in range(nim):
                #         mask_zeros = imarr[i]==0
                #         imdiff[i, :, mask_zeros] = np.nan
                # elif len(imdiff.shape)==3:
                #     nim = imdiff.shape[0]
                #     for i in range(nim):
                #         mask_zeros = imarr[i]==0
                #         imdiff[i, mask_zeros] = np.nan

                odict[k]['data_diff'] = imdiff

                del imarr

        del imarr_ref

    def get_data_arr(self, obsid, data_key='data', squeeze=True):
        """Get all data for a given observation ID"""

        # Check if obsid is in reference objects
        if (self.obsids_ref is not None) and (obsid in self.obsids_ref):
            obs_dict = self.ref_objs[self.obsids_ref.index(obsid)].obs_dict
        else:
            obs_dict = self.obs_dict

        odict = obs_dict[obsid]
        data_arr = []
        for k in odict.keys():
            data_arr.append(odict[k][data_key])

        if squeeze:
            return np.array(data_arr).squeeze()
        else:
            return np.array(data_arr)

    def get_header(self, obsid, ext=0):
        """Get header for a given observation ID"""

        # Check if obsid is in reference objects
        if (self.obsids_ref is not None) and (obsid in self.obsids_ref):
            obs_dict = self.ref_objs[self.obsids_ref.index(obsid)].obs_dict
        else:
            obs_dict = self.obs_dict

        return obs_dict[obsid][0][f'hdr{str(ext)}']


    # def shift_to_center_frac(self, med_dithers=True):
    #     """Fractional shifts to align arrays"""
        
    # def align_images(self, method='fourier', lsq=True, return_offsets=False, med_dithers=True):
    #     """Shift and align all images
        
        
    #     """
        
def replace_data_with_sim(self, **kwargs):
    """Replace observed data with simulated data"""

    def gen_sim_rate_images(data_shape, xyoff_asec, nrc, gain_scale=True, photmjsr=1, **kwargs):
        """Generate simulated data with noise in DN/sec
        
        If data cube, then creates a repeat of the same image such that
        subsequent mean or median of the cube generates the same result
        with reducing noise properties. This is because the noise is already
        generated to take into account the number of integrations; that is,
        it divides by sqrt(nints).

        gain_scale : bool
            Convert from e-/sec to DN/sec
        photmjsr : float
            Converstion factor from DN/sec to MJy/sr. Set to 1 if no conversion needed.
        """
        if len(data_shape)==2:
            nint = 1
        elif len(data_shape)==3:
            nint = data_shape[0]
        else:
            raise ValueError(f"Data array has unexpected shape {data_shape}.")
        
        im_sim = nrc.gen_slope_image(xyoff_asec=xyoff_asec, exclude_noise=False,
                                     shift_func=fourier_imshift, **kwargs)
        
        # Convert from e-/sec to DN/sec
        if gain_scale:
            im_sim /= nrc.Detector.gain
        im_sim *= photmjsr

        return np.repeat(im_sim[None,:,:], nint, axis=0).reshape(data_shape)


    nrc = self.nrc
    hdr0 = self.obs_dict[self.obsids[0]][0]['hdr0']
    hdr1 = self.obs_dict[self.obsids[0]][0]['hdr1']

    # Convert from e-/sec to DN/sec
    gain_scale = True if 'SKIPPED' in hdr0.get('S_GANSCL','') else False
    
    # Convert from DN/sec to MJy/sr
    photmjsr = hdr1['PHOTMJSR'] if hdr1['BUNIT'] == 'MJy/sr' else 1

    imarr = []
    for oid in self.obsids:
        odict = self.obs_dict[oid]
        for k in odict.keys():
            im = odict[k]['data']
            if len(im.shape)==3:
                im = np.nanmean(im, axis=0)
            imarr.append(im)
            
    imarr = np.asarray(imarr)

    imshape_orig = imarr.shape
    ndith = imshape_orig[0]
    # Number of images per dither
    if len(imshape_orig)==3:
        nimg_per_dither = 1
    elif len(imshape_orig)==4:
        nimg_per_dither = imshape_orig[1]
    else:
        raise ValueError(f"imarr array has unexpected shape {imshape_orig}.")

    # Integer values to offset image to place star in center of image
    xy_mask_offset = self.xy_mask_offset.copy()
    if (nimg_per_dither==1) and (len(xy_mask_offset.shape)==3):
        # Reduce to single shift value per dither
        xy_mask_offset = np.mean(xy_mask_offset, axis=1)

    # Determine number of shift values per dither
    if len(xy_mask_offset.shape)==2:
        # Number of shift values per dither
        nsh_per_dither = 1
    elif len(xy_mask_offset.shape)==3:
        # Number of shift values per dither
        nsh_per_dither = xy_mask_offset.shape[1]
    else:
        raise ValueError(f"xy_mask_offset array has unexpected shape {xy_mask_offset.shape}.")


    from ..nrc_utils import conf
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)

    imarr_sim = []
    for i in trange(ndith, leave=False, desc='Simulating Data'):
        # Case of single image per dither
        if nimg_per_dither==1:
            xyoff_asec = xy_mask_offset[i] * self.nrc.pixelscale
            data_cube = gen_sim_rate_images(imarr[i].shape, xyoff_asec, nrc, 
                                            gain_scale=gain_scale, photmjsr=photmjsr, 
                                            **kwargs)
            imarr_sim.append(data_cube)
        else:
            imlist = []
            for j in range(nimg_per_dither):
                xyoff_asec = xy_mask_offset[i] if nsh_per_dither==1 else xy_mask_offset[i,j] 
                xyoff_asec = xyoff_asec * self.nrc.pixelscale
                data_cube = gen_sim_rate_images(imarr[i,j].shape, xyoff_asec, nrc, 
                                                gain_scale=gain_scale, photmjsr=photmjsr, 
                                                **kwargs)
                imlist.append(data_cube)
            imarr_sim.append(np.asarray(imlist))
    imarr_sim = np.asarray(imarr_sim)

    setup_logging(log_prev, verbose=False)

    # Update odict with shifted images
    ii = 0
    for i, oid in enumerate(self.obsids):
        odict = self.obs_dict[oid]
        for k in odict.keys():
            data_sim = imarr_sim[ii]
            data_shape = odict[k]['data'].shape
            data_sim_shape = data_sim.shape

            # print(data_shape, data_sim_shape)
            if data_shape != data_sim_shape:
                data_sim = data_sim[None,:,:] if len(data_sim_shape)==2 else data_sim[None,:,:,:]
                data_sim = np.repeat(data_sim, data_shape[0], axis=0)
            data_sim = data_sim.reshape(data_shape)

            del odict[k]['data']
            odict[k]['data'] = data_sim

            ii += 1

