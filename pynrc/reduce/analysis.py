import numpy as np
import os

import json

from tqdm import trange, tqdm

from astropy.io import ascii, fits

from webbpsf_ext import robust
from webbpsf_ext.analysis_tools import ipc_info, ppc_info
from webbpsf_ext.imreg_tools import read_sgd_files, get_coron_apname

from ..nrc_utils import get_one_siaf, get_detname

from ..maths.image_manip import fractional_image_shift, image_shift_with_nans, replace_nans
from ..maths.image_manip import frebin, zrebin, crop_image, get_im_cen
from ..maths.image_manip import fshift, fourier_imshift
from ..maths.image_manip import apply_pixel_diffusion, add_ipc, add_ppc
from ..maths.image_manip import image_convolution, subtract_psf
from ..maths.coords import dist_image

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

def subtraction_metrics(im1, im2, bp1, bp2, xyshift, 
                        method='fourier', interp='lanczos',
                        mask=None, weights=None, **kwargs):
    """Perform subtraction and return goodness of fit metrics (std, ssr)
    

    Keyword Args
    ------------
    method : str
        Method to use for shifting. Options are:
        - 'fourier' : Shift in Fourier space
        - 'fshift' : Shift using interpolation
        - 'opencv' : Shift using OpenCV warpAffine
    interp : str
        Interpolation method to use for shifting using 'fshift' or 'opencv. 
        Default is 'cubic'.
        For 'opencv', valid options are 'linear', 'cubic', and 'lanczos'.
        for 'fshift', valid options are 'linear', 'cubic', and 'quintic'.
    mask : bool array
        Mask of good pixels to consider. Default is None.
    weights: ndarray
        Array of weights to use during the fitting process.
        Useful if you have bad pixels to mask out (ie.,
        set them to zero). Default is None (no weights).
        Should be same size as image.
        Recommended is inverse variance map.
    """

    # Shift bp2 by (dx,dy)
    bp2 = bp2.astype('float')
    bp2 = fshift(bp2, xyshift[0], xyshift[1], interp='linear', pad=True, cval=1)
    bp2 = bp2 > 0
    bpmask = bp1 | bp2
    good_mask = (im1 != im2) & ~bpmask
    if mask is not None:
        good_mask &= mask

    # NOTE: This can only be done after shifting im2
    # # Get optimal scale factor between images
    # im1_good = im1[good_mask]
    # im2_good = im2[good_mask]
    # try:
    #     cf = np.linalg.lstsq(im2_good.reshape([1,-1]).T, im1_good, rcond=None)[0]
    # except:
    #     print(xyshift)
    #     print(np.sum(np.isnan(im1_good)))
    #     print(np.sum(np.isnan(im2_good)))
    #     cf = 1.0
    # scale = cf[0]

    # Find indices within which good data exists
    xgood = np.where(np.sum(good_mask, axis=0) > 0)[0]
    x1, x2 = (xgood[0], xgood[-1])
    ygood = np.where(np.sum(good_mask, axis=1) > 0)[0]
    y1, y2 = (ygood[0], ygood[-1])

    # Crop down for faster processing
    im1_crop = im1[y1:y2+1, x1:x2+1]
    im2_crop = im2[y1:y2+1, x1:x2+1]
    good_mask_crop = good_mask[y1:y2+1, x1:x2+1]
    weights_crop = np.ones_like(im1_crop) if weights is None else weights[y1:y2+1, x1:x2+1]
    bp_crop = ~good_mask_crop

    # Subtract images after performing fractional shift
    diff = subtract_psf(im1_crop, im2_crop, xyshift=xyshift, psf_scale=None, 
                        bpmask=bp_crop, method=method, interp=interp, **kwargs)

    std = np.std(diff[good_mask_crop] * weights_crop[good_mask_crop])
    ssr = np.sum((diff[good_mask_crop] * weights_crop[good_mask_crop])**2)

    return std, ssr

def build_subtraction_maps(im1, im2, bp1, bp2, dx_arr, dy_arr,
               method='fourier', interp='lanczos',
               mask=None, weights=None, verbose=False, **kwargs):
    """Build map of LSQ subtraction metrics
    
    Returns a (nxy x nxy) array of std and square-summed residuals.

    Keyword Args
    ------------
    method : str
        Method to use for shifting. Options are:
        - 'fourier' : Shift in Fourier space
        - 'fshift' : Shift using interpolation
        - 'opencv' : Shift using OpenCV warpAffine
    interp : str
        Interpolation method to use for shifting using 'fshift' or 'opencv. 
        Default is 'cubic'.
        For 'opencv', valid options are 'linear', 'cubic', and 'lanczos'.
        for 'fshift', valid options are 'linear', 'cubic', and 'quintic'.
    mask : bool array
        Mask of good pixels to consider. Default is None.
    weights: ndarray
        Array of weights to use during the fitting process.
        Useful if you have bad pixels to mask out (ie.,
        set them to zero). Default is None (no weights).
        Should be same size as image.
        Recommended is inverse variance map.
    """
    std_arr = []
    ssr_arr = []

    dy_iter = tqdm(dy_arr, desc='dy', leave=False) if verbose else dy_arr
    for dy in dy_iter:
        for dx in dx_arr:
            std, ssr = subtraction_metrics(im1, im2, bp1, bp2, (dx, dy), 
                                           mask=mask, weights=weights,
                                           method=method, interp=interp, **kwargs)
            # Standard deviation of the difference
            std_arr.append(std)
            # Sum of squared residuals
            ssr_arr.append(ssr)

    nxy = len(dx_arr)
    std_arr = np.array(std_arr).reshape(nxy, nxy)
    ssr_arr = np.array(ssr_arr).reshape(nxy, nxy)

    return std_arr, ssr_arr

def find_best_offset(im1, im2, bp1, bp2, dxy_coarse_arr, dxy_fine_arr0,
                     mask=None, weights=None, method='fourier', interp='lanczos',
                     use_ssr=True, use_std=True, verbose=False, **kwargs):
    """Best offset to move im2 to align with im1
    
    Keyword Args
    ------------
    method : str
        Method to use for shifting. Options are:
        - 'fourier' : Shift in Fourier space
        - 'fshift' : Shift using interpolation
        - 'opencv' : Shift using OpenCV warpAffine
    interp : str
        Interpolation method to use for shifting using 'fshift' or 'opencv. 
        Default is 'cubic'.
        For 'opencv', valid options are 'linear', 'cubic', and 'lanczos'.
        for 'fshift', valid options are 'linear', 'cubic', and 'quintic'.
    mask : bool array
        Mask of good pixels to consider. Default is None.
    weights: ndarray
        Array of weights to use during the fitting process. Useful if you 
        have bad pixels to mask out (ie., set them to zero). 
        Default is None (no weights). Should be same size as image.
        Recommended is inverse variance map.
    use_ssr : bool
        Use sum of squared residuals as a metric for alignment. Default is True.
        If combined with use_std=True, then the best fit is the average of the two.
    use_std : bool
        Use standard deviation of the difference as a metric for alignment. Default is True.
        If combined with use_ssr=True, then the best fit is the average of the two.
    """
    # Perform coarse grid search
    std_arr, ssr_arr = build_subtraction_maps(im1, im2, bp1, bp2, dxy_coarse_arr, dxy_coarse_arr,
                                              mask=mask, weights=weights, method=method, interp=interp, 
                                              verbose=verbose, **kwargs)
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
    std_arr, ssr_arr = build_subtraction_maps(im1, im2, bp1, bp2, dx_fine_arr, dy_fine_arr,
                                              mask=mask, weights=weights, method=method, interp=interp,
                                              verbose=verbose, **kwargs)

    # Find the minimum positions for std and ssr
    dy_idx, dx_idx = np.unravel_index(np.nanargmin(std_arr), std_arr.shape)
    xsh_fine1, ysh_fine1 = (dx_fine_arr[dx_idx], dy_fine_arr[dy_idx])
    dy_idx, dx_idx = np.unravel_index(np.nanargmin(ssr_arr), ssr_arr.shape)
    xsh_fine2, ysh_fine2 = (dx_fine_arr[dx_idx], dy_fine_arr[dy_idx])

    # NOTE: This actually doesn't work well. Positional offsets changes the final answer. Just report min location.
    # The best fit is then the weighted average of the inverse of the std and ssr maps
    # Take the weight average of the 5x5 grid around the minimum
    # xv_grid, yv_grid = np.meshgrid(dx_fine_arr[dx_idx-2:dx_idx+3], dy_fine_arr[dy_idx-2:dy_idx+3])
    # inv_std_arr = 1 / std_arr[dy_idx-2:dy_idx+3, dx_idx-2:dx_idx+3]
    # xsh_fine1 = np.average(xv_grid, weights=inv_std_arr**2)
    # ysh_fine1 = np.average(yv_grid, weights=inv_std_arr**2)
    # inv_ssr_arr = 1 / ssr_arr[dy_idx-2:dy_idx+3, dx_idx-2:dx_idx+3]
    # xsh_fine2 = np.average(xv_grid, weights=inv_ssr_arr**2)
    # ysh_fine2 = np.average(yv_grid, weights=inv_ssr_arr**2)

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

def find_best_offset_wrapper(im1, im2, bp1=None, bp2=None, pixel_binning=1,
                             coarse_limits=(-1.5,1.5),fine_limits=(-0.2,0.2),
                             dxy_coarse=0.2, dxy_fine=0.01, rin=None, rout=None, 
                             method='fourier', interp='lanczos',
                             verbose=False, **kwargs):
    """Simple wrapper for find_best_offset function
    
    Returns the best shift values for im2 to align with im1 in units of *pixels*. 
    The pixel_binning keyword defines the pixel oversampling of im1 and im2 images.
    The coarse_limits and fine_limits define the search space for the coarse and 
    fine grid searches in units of *pixels*. 
    """
    
    bp1 = np.zeros_like(im1).astype('bool') if bp1 is None else bp1
    bp2 = np.zeros_like(im2).astype('bool') if bp2 is None else bp2

    # Set up coarse grid search
    xy1, xy2 = np.array(coarse_limits) * pixel_binning
    dxy_coarse = 0.2 * pixel_binning
    dxy_coarse_arr = np.arange(xy1, xy2+dxy_coarse, dxy_coarse)

    # Define a finer grid offsets
    # Make sure fine grid limits are at least 2x the coarse grid steps
    dxy_fine = 0.01 * pixel_binning
    if 2 * dxy_coarse > fine_limits[1] - fine_limits[0]:
        fine_limits = (-dxy_coarse, dxy_coarse)
    xy1, xy2 = np.array(fine_limits) * pixel_binning
    dxy_fine_arr0 = np.arange(xy1, xy2+dxy_fine, dxy_fine)

    rho_pix = dist_image(im1) / pixel_binning
    rin = 0 if rin is None else rin
    if rout is not None:
        mask = (rho_pix >= rin) & (rho_pix <= rout)
    else:
        mask = rho_pix >= rin

    res = find_best_offset(im1, im2, bp1, bp2, dxy_coarse_arr, dxy_fine_arr0,
                           mask=mask, method=method, interp=interp, 
                           verbose=verbose, **kwargs)
    
    return res / pixel_binning

def flag_outliers_in_diff(diff, nsig=10, rin=15):
    """Find pixel outliers in difference image
    
    Input images are assumed to have NaNs where known bad pixels are located.
    The 2nd image is shifted by xyshift, then subtracted from the 1st image.
    Outlier pixels in the differenced image are flagged and returned. Only
    new bad pixels are flagged. Pixels within a certain radius of the center
    (`rout`) are ignored.
    """

    # Flag bad pixels
    bpmask = np.isnan(diff)

    # Mask off pixels within rin of the center
    rho = dist_image(diff)
    rmask = rho>rin

    # Pixels to get statistics
    good_mask = ~bpmask & rmask
    med = np.nanmedian(diff[good_mask])
    std = robust.medabsdev(diff[good_mask])

    # Create new (additional) bad pixel mask
    bp_new = np.zeros_like(diff).astype('bool')
    ind = diff > med + nsig*std
    bp_new[ind] = True
    bp_new[bpmask] = False
    bp_new[~rmask] = False

    return bp_new

def flag_outliers(im1_nans, im2_nans, xyshift, nsig=10, rin=15,
                  shift_method='fshift', interp='linear', 
                  oversample=4, order=1, return_diff=False, **kwargs):
    """Find pixel outliers in difference image
    
    Input images are assumed to have NaNs where known bad pixels are located.
    The 2nd image is shifted by xyshift, then subtracted from the 1st image.
    Outlier pixels in the differenced image are flagged and returned. Only
    new bad pixels are flagged. Pixels within a certain radius of the center
    (`rout`) are ignored.
    """

    im1_sh = image_shift_with_nans(im1_nans, 0, 0, oversample=oversample, order=order, 
                                   shift_method=shift_method, interp=interp,
                                   return_oversample=False, preserve_nans=True,
                                   **kwargs)

    # Shift im2 by xyshift and interpolate NaNs
    xsh, ysh = xyshift
    im2_sh = image_shift_with_nans(im2_nans, xsh, ysh, oversample=oversample, order=order, 
                                   shift_method=shift_method, interp=interp,
                                   return_oversample=False, preserve_nans=False,
                                   **kwargs)

    # Flag new additional bad pixels (not already NaNs in diff images)
    diff = im1_sh - im2_sh
    bp_new = flag_outliers_in_diff(diff, nsig=nsig, rin=rin)

    if return_diff:
        return bp_new, diff
    else:
        return bp_new


def fit_gauss_image(array, bpmask=None, cropsize=15, cen=None, 
                    fwhm=4, theta=0, threshold=False, sigfactor=5,
                    full_output=False, debug=False):
    """ Fitting a 2D Gaussian to the 2D distribution of the data.

    Parameters
    ----------
    array : numpy ndarray
        Input frame with a single PSF.
    cen : tuple of int, optional
        X,Y integer position of source in the array for extracting the subimage.
        If None the center of the frame is used for cropping the subframe (the
        PSF is assumed to be ~ at the center of the frame).
    cropsize : int, optional
        Size of the subimage.
    fwhmx, fwhmy : float, optional
        Initial values for the standard deviation of the fitted Gaussian, in px.
    theta : float, optional
        Angle of inclination of the 2d Gaussian counting from the positive X
        axis.
    threshold : bool, optional
        If True the background pixels (estimated using sigma clipped statistics)
        will be replaced by small random Gaussian noise.
    sigfactor : int, optional
        The background pixels will be thresholded before fitting a 2d Gaussian
        to the data using sigma clipped statistics. All values smaller than
        (MEDIAN + sigfactor*STDDEV) will be replaced by small random Gaussian
        noise.
    bpm : 2D numpy ndarray, optional
        Mask of bad pixels to not consider for the fit.
    full_output : bool, optional
        If False it returns just the centroid, if True also returns the
        FWHM in X and Y (in pixels), the amplitude and the rotation angle,
        and the uncertainties on each parameter.
    debug : bool, optional
        If True, the function prints out parameters of the fit and plots the
        data, model and residuals.

    Returns
    -------
    mean_x : float
        Source centroid y position on input array from fitting.
    mean_y : float
        Source centroid x position on input array from fitting.

    If ``full_output`` is True it returns a Pandas dataframe containing the
    following columns:
        'centroid_y': Y coordinate of the centroid.
        'centroid_x': X coordinate of the centroid.
        'fwhm_y': Float value. FWHM in X [px].
        'fwhm_x': Float value. FWHM in Y [px].
        'amplitude': Amplitude of the Gaussian.
        'theta': Float value. Rotation angle.
        # and fit uncertainties on the above values:
        'centroid_y_err'
        'centroid_x_err'
        'fwhm_y_err'
        'fwhm_x_err'
        'amplitude_err'
        'theta_err'

    """

    from astropy.stats import (gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma, sigma_clipped_stats)
    from astropy.modeling import models, fitting
    from photutils.centroids import centroid_com

    if bpmask is None:
        bpmask = np.zeros_like(array).astype('bool')

    if cropsize is not None:
        if cen is None:
            cenx, ceny = get_im_cen(array)
        else:
            cenx, ceny = cen

        imside = np.min(array.shape[-2:])
        psf_subimage, xyvals = crop_image(array, min(cropsize, imside),
                                          xyloc=(cenx, ceny), return_xy=True)
        subx = xyvals[0]
        suby = xyvals[2]
        bpm_subimage = crop_image(bpmask, min(cropsize, imside),  xyloc=(cenx, ceny))
    else:
        psf_subimage = array.copy()
        bpm_subimage = bpmask.copy()

    # Include any NaNs in the bad pixel mask
    bpm_subimage |= np.isnan(psf_subimage)

    if threshold:
        _, clipmed, clipstd = sigma_clipped_stats(psf_subimage, sigma=2)
        indi = np.where(psf_subimage <= clipmed + sigfactor * clipstd)
        subimnoise = np.random.randn(psf_subimage.shape[0],
                                     psf_subimage.shape[1]) * clipstd
        psf_subimage[indi] = subimnoise[indi]

    # Get initial values for the fit
    try:
        fwhm_x0, fwhm_y0 = fwhm
    except:
        fwhm_x0 = fwhm_y0 = fwhm
    x_stddev = fwhm_x0 * gaussian_fwhm_to_sigma
    y_stddev = fwhm_y0 * gaussian_fwhm_to_sigma

    # Creating the 2D Gaussian model
    init_amplitude = np.ptp(psf_subimage[~bpm_subimage])
    xcom, ycom = centroid_com(psf_subimage)
    gauss = models.Gaussian2D(amplitude=init_amplitude, theta=theta,
                              x_mean=xcom, y_mean=ycom,
                              x_stddev=x_stddev, y_stddev=y_stddev)
    # Levenberg-Marquardt algorithm
    fitter = fitting.LevMarLSQFitter()
    y, x = np.indices(psf_subimage.shape)
    fit = fitter(gauss, x[~bpm_subimage], y[~bpm_subimage],
                 psf_subimage[~bpm_subimage])

    if cropsize is not None:
        mean_x = fit.x_mean.value + subx
        mean_y = fit.y_mean.value + suby
    else:
        mean_x = fit.x_mean.value
        mean_y = fit.y_mean.value
    fwhm_x = fit.x_stddev.value * gaussian_sigma_to_fwhm
    fwhm_y = fit.y_stddev.value * gaussian_sigma_to_fwhm
    amplitude = fit.amplitude.value
    theta = np.rad2deg(fit.theta.value)

    # compute uncertainties
    if fitter.fit_info['param_cov'] is not None:
        with np.errstate(invalid='raise'):
            try:
                perr = np.sqrt(np.diag(fitter.fit_info['param_cov']))
                amplitude_e, mean_x_e, mean_y_e, fwhm_x_e, fwhm_y_e, theta_e = perr
                fwhm_x_e *= gaussian_sigma_to_fwhm
                fwhm_y_e *= gaussian_sigma_to_fwhm
            except:
                # this means the fit failed
                mean_y, mean_x = np.nan, np.nan
                fwhm_y, fwhm_x = np.nan, np.nan
                amplitude, theta = np.nan, np.nan
                mean_y_e, mean_x_e = np.nan, np.nan
                fwhm_y_e, fwhm_x_e = np.nan, np.nan
                amplitude_e, theta_e = np.nan, np.nan
    else:
        amplitude_e, theta_e, mean_x_e = np.nan, np.nan, np.nan
        mean_y_e, fwhm_x_e, fwhm_y_e = np.nan, np.nan, np.nan
        # the following also means the fit failed
        if fwhm_y == fwhm_y0 and fwhm_x == fwhm_x0 and amplitude == init_amplitude:
            mean_y, mean_x = np.nan, np.nan
            fwhm_y, fwhm_x = np.nan, np.nan
            amplitude, theta = np.nan, np.nan

    if debug:
        from hciplot import plot_frames

        if threshold:
            label = ('Subimage thresholded', 'Model', 'Residuals')
        else:
            label = ('Subimage', 'Model', 'Residuals')
        plot_frames((psf_subimage, fit(x, y), psf_subimage-fit(x, y)),
                    grid=True, grid_spacing=1, label=label)
        print('FWHM_x =', fwhm_x)
        print('FWHM_y =', fwhm_y, '\n')
        print('centroid x =', mean_x)
        print('centroid y =', mean_y)
        print('centroid x subim =', fit.x_mean.value)
        print('centroid y subim =', fit.y_mean.value, '\n')
        print('amplitude =', amplitude)
        print('theta =', theta)

    if full_output:
        import pandas as pd
        return pd.DataFrame({'centroid_s': mean_x, 'centroid_y': mean_y,
                             'fwhm_x': fwhm_x, 'fwhm_y': fwhm_y,
                             'amplitude': amplitude, 'theta': theta,
                             'centroid_y_err': mean_y_e,
                             'centroid_x_err': mean_x_e,
                             'fwhm_y_err': fwhm_y_e, 'fwhm_x_err': fwhm_x_e,
                             'amplitude_err': amplitude_e,
                             'theta_err': theta_e}, index=[0],
                            dtype=np.float64)
    else:
        return np.array([mean_x, mean_y])



def lsq_subtraction(imarr, imarr_ref, rin=0, rout=None, th_range=None, verbose=False):
    """ Perform least squares optimization to subtract reference image(s) from science image(s)"""

    # Reshape arrays to 3D if necessary
    sh_sci = imarr.shape
    ndim_sci = len(sh_sci)
    ndim_ref = len(imarr_ref.shape)

    if ndim_sci==2:
        imarr = imarr.reshape((1, imarr.shape[0], imarr.shape[1]))
    if ndim_ref==2:
        imarr_ref = imarr_ref.reshape((1, imarr_ref.shape[0], imarr_ref.shape[1]))

    nz_sci = imarr.shape[0]
    nz_ref = imarr_ref.shape[0]

    # Pixel regions to perform optimizationif ndim_sci == 3:
    rho, th = dist_image(imarr[0], return_theta=True)

    mask_good = np.ones_like(rho, dtype=bool)
    # Ignore bad pixels from science image
    bpmask_sci = np.logical_or.reduce(np.isnan(imarr), axis=0)
    mask_good[bpmask_sci] = False
    # Ignore bad pixels from reference image(s)
    bpmask_ref = np.logical_or.reduce(np.isnan(imarr_ref), axis=0)
    mask_good[bpmask_ref] = False
    # Mask out the central region
    mask_good[rho < rin] = False
    # Mask out outer regions
    if rout is not None:
        mask_good[rho > rout] = False

    # Mask out regions with specific theta values
    if th_range is not None:
        if th_range[0] < th_range[1]:
            th_mask = (th >= th_range[0]) & (th <= th_range[1])
        else:
            th_mask = (th >= th_range[0]) | (th <= th_range[1])
        mask_good &= th_mask

    # Optimization regions
    sci_opt = imarr[:,mask_good]
    bg_opt = imarr_ref[:, mask_good]

    # Subtract regions
    sci_sub = imarr.reshape((nz_sci, -1))
    bg_sub = imarr_ref.reshape((nz_ref, -1))

    a, b = bg_opt, sci_opt

    # Perform least squares optimization
    q, r = np.linalg.qr(a.T, 'reduced')
    qTb = np.matmul(q.T, b.T)
    coeff_all = np.linalg.lstsq(r, qTb, rcond=None)[0]

    if verbose:
        print(coeff_all.squeeze())

    # Subtract reference images
    imdiff = sci_sub - np.dot(bg_sub.T, coeff_all).T
    imdiff = imdiff.reshape(sh_sci)

    return imdiff


def pca_subtraction(imarr, imarr_ref, fwhm, **kwargs):

    from . import pca

    # Build pca params dictionary for PSF subtraction
    pca_params = pca.build_pca_dict(fwhm, **kwargs)
    imdiff = pca.run_pca_subtraction(imarr, imarr_ref, pca_params)

    return imdiff, pca_params


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
    from astropy.convolution import Gaussian2DKernel, convolve
    
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
        kernel = Gaussian2DKernel(gstd_pix)
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
    nrc.fov_pix = np.min([ap.XSciSize, ap.YSciSize]) if fov_pix is None else fov_pix
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
                    include_si_wfe=None, return_hdul=False, quick_grid=False, 
                    **kwargs):
    """ Generate a single defocused PSF
    
    return_oversample : int
        Return an oversampled version of the PSF. Only valid if
        return_hdul=False, otherwise always returns both detector
        and oversampled versions.
    """

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
    if defocus_waves_2um !=0:
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
                osamp = hdu.header['OSAMP']
                hdul.data = fourier_imshift(data, xyoffpix[0]*osamp, xyoffpix[1]*osamp)
        output = hdul
    else:
        # Generate oversampled PSF and downsample
        if quick_grid:
            output = nrc.calc_psf_from_coeff(return_oversample=True, 
                                             return_hdul=False, **kwargs)
        else:
            # print(kwargs)
            # print(nrc.options)
            # print(nrc.include_distortions, nrc.include_ote_field_dependence, nrc.include_si_wfe)
            output = nrc.calc_psf(return_oversample=True, return_hdul=False, **kwargs)
        # Shift if xyoffpix is not (0,0)
        osamp = nrc.oversample
        if (xyoffpix is not None) and (not np.allclose(xyoffpix, 0)):
            # print(xyoffpix)
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
                      quick_grid=False, psf_corr_image=None, **kwargs):
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

    kwargs['return_oversample'] = True
    kwargs['return_hdul']       = return_hdul
    kwargs['xyoffpix']          = xyoffpix
    kwargs['include_distortions']          = include_distortions
    kwargs['include_ote_field_dependence'] = include_ote_field_dependence
    kwargs['include_si_wfe']               = include_si_wfe
    kwargs['quick_grid']                   = quick_grid
    res = gen_defocus_psf(nrc, defocus_waves_2um, **kwargs)

    # Alway returns oversampled data
    osamp = nrc.oversample

    if return_hdul:
        hdu = res['OVERDIST'] if include_distortions else res['OVERSAMP']
        if diffusion_sigma>0:
            hdu.data = apply_pixel_diffusion(hdu.data, diffusion_sigma*osamp)
        # Apply PSF correction factor
        if psf_corr_image is not None:
            hdu.data *= crop_image(psf_corr_image, hdu.data.shape, fill_val=1)
        if not return_oversample:
            hdu.data = frebin(hdu.data, scale=1/osamp)
            # Convert header info to detector pixels
            hdu.header['EXTNAME'] = hdu.header['EXTNAME'].replace('OVER','DET_')
            hdu.header['OSAMP'] = 1
            hdu.header['OVERSAMP'] = 1
            hdu.header['PIXELSCL'] *= osamp

        res = fits.HDUList([hdu])
    else:
        if diffusion_sigma>0:
            res = apply_pixel_diffusion(res, diffusion_sigma * osamp)
        # Apply PSF correction factor
        if psf_corr_image is not None:
            res *= crop_image(psf_corr_image, hdu.data.shape, fill_val=1)
        if not return_oversample:
            res = frebin(res, scale=1/osamp)

    return res


def quick_bfe(im, frac_min=0, frac_max=0.1):
    """Add brighter fatter effect to an image"""

    from astropy.convolution import convolve

    # Shift fractional amount of signal to neighboring pixels that depends on the signal
    # Larger signals values move more signal to neighboring pixels
    im_frac = im.copy()
    im_frac[im_frac<0] = 0
    im_frac /= im_frac.max()
    im_frac *= (frac_max - frac_min)
    im_frac += frac_min

    sig_shift = im_frac * im

    # Create a 3x3 kernel based on inverse distance^2 from center
    kernel = 1 / dist_image(np.zeros((3,3)))**2
    kernel[1,1] = 0
    kernel /= kernel.sum()

    # Apply kernel to the shifted image
    sig_shift_convolve = image_convolution(sig_shift, kernel)

    # Subtract shifted signal levels from original and add convolved image
    im_final = im - sig_shift + sig_shift_convolve

    return im_final

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
    """Get star information from a dictionary of sources
    
    Removes spaces, dashes, and underscores in name.
    """

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
        'P330E' : { # G. Rieke et al. 2024
            'name': 'GSPC P330-E', 'fname': 'P330-E.vot',
            'sptype': 'G5V', 'Teff': 5850, 'metallicity': -0.23, 'log_g': 4.4,
            'Av': 0.0, 'mag_val': 11.428, 'bp': bp_k, 
        },
        'P177D' : { # G. Rieke et al. 2024
            'name': 'GSPC P177-D', 'fname': 'P177-D.vot',
            'sptype': 'G0V', 'Teff': 5870, 'metallicity': -0.05, 'log_g': 4.23,
            'Av': 0.0, 'mag_val': 11.856, 'bp': bp_k, 
        },
        'BD+601753' : { # G. Rieke et al. 2024
            'name': 'BD+60-1753', 'fname': 'BD+60-1753.vot',
            'sptype': 'A1V', 'Av': 0.0, 'mag_val': 9.64, 'bp': bp_k, 
        },
        'J1743045' : { 
            'name': 'TIC 233205654', 'fname': 'J17430448.vot',
            'sptype': 'A5V', 'Av': 0.0, 'mag_val': 12.8, 'bp': bp_k, 
        },
        'J1802272' : { 
            'name': 'TIC 233067231', 'fname': 'J18022716.vot',
            'sptype': 'A2V', 'Av': 0.0, 'mag_val': 11.8, 'bp': bp_k, 
        },
        'J1757132' : { 
            'name': 'TIC 219094190', 'fname': 'J17571324.vot',
            'sptype': 'A3V', 'Av': 0.0, 'mag_val': 11.2, 'bp': bp_k, 
        },
        'EMAS209' : { 
            'name': 'AS 209', 'fname': 'AS209.vot',
            'dist': 121.2, 'age': 2, 'sptype': 'K4V', 
            'Av': 0.0, 'mag_val': 6.96, 'bp': bp_k, 
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

def disk_model_grater_2hg(r0, h0, ain, aout, pa, incl, g1, g2, wg1,
                           e=0., omega=0., gamma=2., beta=1.,
                           distance=10., nx=320, ny=320, pxscale=0.063, cent=None,
                           accuracy=None, rmax_accuracy=None,
                           halfNbSlices=25, polar=False, flux_max=None):
    """
    A simple ring-like disk morphology based on Augereau et al. (1999) and 
    assuming a linear combo of two H-G SPFs as the scattering phase function.
    
    Parameters
    ----------
    r0 : float
        fiducial radius in au
    h0 : float
        technically h0/r0 — the ratio of scale height to 
        fiducial radius at the fiducial radius.
    ain : float
        radial density power law exponent interior to r0
    aout : float
        radial density power law exponent exterior to r0
    pa : float
        disk position angle in degrees
    incl : float
        disk inclination wrt the line of sight in degrees 
        (0 means pole-on, 90 means edge-on)
    g1 : float
        1st Henyey-Greenstein asymmetry parameter.
        Slope of the power-low distribution in the inner disk.
    g2 : float
        2nd Henyey-Greenstein asymmetry parameter.
        Slope of the power-low distribution in the outer disk.
    wg1 : float
        Weight for the SPF term with asymmetry parameter g1 
        (value in range 0-1); wg2 is 1-wg1

    Keyword Arguments
    -----------------
    e : float
        eccentricity
    omega : float
        argument of pericenter in degrees
    gamma : float 
        vertical density exponent (gamma = 2 for gaussian)
    beta : float
        disk radial flaring exponent (beta = 1 for linear)
    distance : float
        distance to the target in parsecs
    nx : int
        number of x-axis pixels for the image
    ny : int
        number of y-axis pixels for the image
    cent : ndarray or None
        The (x,y) pixel position for the center of the disk. 
        Generally the location of the star in the data.
        Set to None to place at the center of the image.
    pxscale : float or astropy.units.quantity.Quantity
        The pixel scale for the data; either a float (must be arcsec/pixel)
        or astropy units (any units that can be cast to arcsec/pixel)
    accuracy : float
        the numerical accuracy for the model; 
        pixels with density below this value will be set to zero
    rmax_accuracy : float
        if accuracy is None, the model's accuracy is set such that
        non-zero values are achieved to this separation (in au)
    halfNbSlices : int
        the number of planar slices to compute above and below the disk midplane.
        Default is 25.
    polar : bool
        if True, a simple bell-shaped polarization curve is used to 
        generate a polarized intensity image
    flux_max : float
        if not None, normalize the model image so that this is the maximum value.
    """


    from vip_hci.fm import ScatteredLightDisk
    from vip_hci.var import frame_center

    import astropy.units as u
    from astropy import convolution
    

    def ang_size_to_proj_sep(ang_size, distance):
        """
        Converts angular separation (any angular unit, e.g. arcsec, degrees, radians) to projected separation (au).
        ang_size and distance can be provided as a float/int (in which case units of arcsec and parsec are assumed
        respectively).
        
        If not specified, units for ang_size and distance are assumed to be arcseconds and parsecs respectively.
        
        Example:
            1) r = ang_size_to_proj_sep(0.25, 156) 
            # Returns the proj separation in astropy units of au for an angular separation of 0.25 arcsec at 156 pc
            
            2) r = ang_size_to_proj_sep(250*u.mas, 508.8*u.lightyear) 
            # Returns roughly the same value as example 1, but with different input units.
            
        Note: returns an astropy unit value. 
            ang_size_to_proj_sep(ang_size, distance).value will give you a float instead.
        """
        ang_size = ang_size << u.arcsec # If units aren't provided, sets unit to arcsec. Else converts unit to arcsec
        d = distance << u.pc
        return (d * np.tan(ang_size.to('rad'))).to('AU')

    def px_size_to_ang_size(px_size, pxscale):
        """
        Converts a pixel size (in pixels) to an angular separation (in arcsec).
        If not specified, units for px_size and pxscale are assumed to be pixels and arcsec/pixel respectively.
        """
        px_size = px_size << u.pixel
        pxscale = pxscale << u.arcsec / u.pixel
        return px_size * pxscale

    def dont_actually_check_inclination():
        """
        VIP's ScatteredLightDisk will normally decrease disk inclination if too high.
        This will cause issues for optimization, so we'd rather live with the artifacts / etc.
        Instead, we just tell VIP that the incl is good whenever it checks.
        """
        return None
 
    if cent is None:
        cent = get_im_cen(np.zeros([ny, nx]))

    if accuracy is None:
        if rmax_accuracy is None:
            corners = np.array([[0, 0], [0, ny-1], [nx-1, 0], [nx-1, ny-1]])
            im_corner_dists = np.hypot(*(corners - cent).T)
            rmax_pix = np.max(im_corner_dists)
            rmax_asec = px_size_to_ang_size(rmax_pix, pxscale)
            rmax_accuracy = ang_size_to_proj_sep(rmax_asec, distance).value # In AU
        accuracy = (rmax_accuracy/r0)**(aout)
            
    ksi0 = h0*r0
    spf = {'name':'DoubleHG', 'g': [g1, g2], 'weight': wg1, 'polar': polar}
    dens = {'name': '2PowerLaws', 'ain': ain, 'aout': aout, 'a': r0, 'ksi0':ksi0,
            'e': e, 'gamma': gamma, 'beta': beta, 'accuracy': accuracy}
    
    pxInArcsec = (pxscale << u.arcsec/u.pixel).value
    vip_disk = ScatteredLightDisk(nx=nx, ny=ny, distance=distance,
                                  itilt=incl, pxInArcsec=pxInArcsec,
                                  pa=pa-180., omega=omega, flux_max=flux_max,
                                  density_dico=dens, spf_dico=spf)

    vip_disk.dust_density.accuracy = accuracy
    a = vip_disk.dust_density.dust_distribution_calc.a
    aout = vip_disk.dust_density.dust_distribution_calc.aout
    rmax = a * accuracy**(1/aout)    
    vip_disk.dust_density.dust_distribution_calc.rmax = rmax 
    vip_disk.check_inclination = dont_actually_check_inclination
    disk = vip_disk.compute_scattered_light(halfNbSlices=halfNbSlices)

    # return disk

    # For disks with material coincident with the stellar position along the line of sight, 
    # VIP returns an image with a plus-sign shaped region of zeros at the star's position
    # The below is a quick work-around to correct this. 
    # Very likely makes zero difference following PSF-convolution
    c_vip = frame_center(disk)
    disk[c_vip[1]-1:c_vip[1]+2, c_vip[0]] = np.nan
    disk[c_vip[1], c_vip[0]-1:c_vip[0]+2] = np.nan
    disk = convolution.interpolate_replace_nans(disk, np.ones((3,3)))

    # VIP also defines the center differencely that we do, 
    # depending on odd or even, so perform an offset to correct if needed 
    dx, dy = cent[0]-c_vip[0], cent[1]-c_vip[1]
    # print('Shape:', disk.shape)
    # print('Centers (frame, VIP):', cent, c_vip)
    # print('Offsets (dx, dy):', (dx, dy))
    if not np.allclose([dx, dy], 0):
        # Set 0 pixels back to zero after fourier shifting
        zero_mask = disk<=0 
        disk = fshift(disk, dx, dy, interp='cubic', pad=True)
        zero_mask_sh = fshift(zero_mask.astype('float'), dx, dy, pad=True) > 0
        disk[zero_mask_sh] = 0

    disk[disk<0] = 0

    return disk


def make_grater_disk(nrc, npix=None, scale_flux=1, return_oversample=True, **kwargs):

    if npix is None:
        ny, nx = np.array([nrc.det_info['ypix'], nrc.det_info['xpix']]) // 2 + 1
    else:
        nx = ny = npix

    osamp = nrc.oversample
    pixscale = nrc.pixelscale

    dstar = nrc.distance

    nx_pix = nx*osamp if return_oversample else nx
    ny_pix = ny*osamp if return_oversample else ny
    pxscale = pixscale/osamp if return_oversample else pixscale

    # Default parameters
    kwargs_def = {
        'r0': 75., 'h0': 5/75., 'ain': 12, 'aout': -12, 
        'pa': 158.5, 'incl': 51.7, 'g1': 0.85, 'g2': 10, 'wg1': 0.7, 
        'e': 0.0, 'omega': 0.0, 'gamma': 2.0, 'beta': 1.3, 'distance': dstar, 
        'nx': nx_pix, 'ny': ny_pix, 'pxscale': pxscale, 
        'cent': None, 'accuracy': None, 'rmax_accuracy': None, 
        'halfNbSlices': 25, 'polar': False, 'flux_max': 1.
    }

    # Update default parameters with user input
    kwargs = {**kwargs_def, **kwargs}

    return scale_flux * disk_model_grater_2hg(**kwargs)


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
        self.psf_corr_dict = {} # PSF correction factor info
        self.psf_corr_over = None

        # Various location offserts
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
    def is_coron(self):
        """Is this a coronagraphic observation?"""
        try:
            hdr0 = self.obs_dict[self.obsids[0]][0]['hdr0']
        except:
            fpath = self.obs_dict[self.obsids[0]][0]['file']
            hdr0 = fits.getheader(fpath)
        is_coron = ('CORONMSK' in hdr0)
        return is_coron
    
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
    def has_sb_units(self):
        """Check if flux values are in surface brightness units"""
        try:
            header = self.obs_dict[self.obsids[0]][0]['hdr1']
        except:
            fpath = self.obs_dict[self.obsids[0]][0]['file']
            header = fits.getheader(fpath, ext=1)

        is_sb = '/sr' in header.get('BUNIT', 'none').lower()
        return is_sb

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
        """IPC kernel
        
        Returns None if IPC correcton was performed on data.
        """
        # Check if IPC correction was performed
        try:
            hdr0 = self.obs_dict[self.obsids[0]][0]['hdr0']
        except:
            fpath = self.obs_dict[self.obsids[0]][0]['file']
            hdr0 = fits.getheader(fpath)

        ipc_corr = hdr0.get('S_IPC', 'N/A') == 'COMPLETE'
        if ipc_corr:
            return None
        else:
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
            return self._best_diffusion
    @best_diffusion.setter
    def best_diffusion(self, value):
        self._best_diffusion = value

    def load_psf_correction(self, verbose=True):
        """Load PSF correction factor"""

        self.psf_corr_dict = load_psf_correction(self.filter, self.apname, verbose=verbose)
        self.psf_corr_over = self.psf_corr_dict.get('psf_scale_data', None)

    def generate_obs_dict(self, file_type='calints.fits', combine_same_dithers=True):
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
                                      file_type=file_type, combine_same_dithers=combine_same_dithers)

            if len(obs_dict)==0:
                raise FileNotFoundError(f'No {file_type} files found for PID {self.pid}, Obs {oid}, Filter {self.filter}, SCA {self.sca}')

            self.obs_dict[oid] = obs_dict

        # Call this same function in the reference objects
        if self.ref_objs is not None:
            for ref_obj in self.ref_objs:
                ref_obj.generate_obs_dict(file_type=file_type)

    def _flag_bad_pixels(self, imarr, dqarr, nsig_spatial=10, nsig_temporal=10, ntemporal_limit=10, niter=3):
        """Flag bad pixels in a single image or stack of images
        
        Returns updated dqarr
        """

        from webbpsf_ext.image_manip import bp_fix

        sh_orig = imarr.shape
        imarr = imarr.squeeze()
        dqarr = dqarr.squeeze()

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
        try:
            bg_val = robust.mode(im_med[mask])
        except:
            print(np.sum(mask))
            bg_val = 0
        bg_sig = robust.medabsdev(im_med[mask])
        bg_bad = (im_med < bg_val - 5*bg_sig) & ~np.isnan(im_med)
        # Flag all images in DQ array as NO_NOT_USE
        if len(imarr.shape)==3:
            for dq in dqarr:
                dq[bg_bad] |= dqflags.pixel['DO_NOT_USE']
        else:
            dqarr[bg_bad] |= dqflags.pixel['DO_NOT_USE']

        return dqarr.reshape(sh_orig)

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

    def expected_pos(self, obsid=None, dither=None, frame='sci'):
        """Return expected position for a given observation
        
        Parameters
        ----------
        obsid : int
            Observation ID
        dither : int
            Dither position
        frame : str
            Frame to return location. Options are 'det', 'sci', 'tel', 'idl'
        """

        def _get_loc(obsid, dither):
            return self.obs_dict[obsid][dither]['loc_exp'][frame]
        
        def _get_obs_loc(obsid):
            return np.array([_get_loc(obsid, d) for d in self.obs_dict[obsid].keys()])
        
        def _get_all_loc():
            return np.concatenate([_get_obs_loc(oid) for oid in self.obsids], axis=0)

        if obsid is None:
            return _get_all_loc()
        elif dither is None:
            return _get_obs_loc(obsid)
        else:
            return _get_loc(obsid, dither)

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

    def create_stellar_spectrum(self, name=None, return_sp=False, return_src=False, plot=False, 
                                kwargs_fit={'use_err':False, 'robust':False, 'wlim':[1,10], 
                                            'IR_excess':True, 'verbose':False},
                                **kwargs):
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
        src.fit_SED(**kwargs_fit)

        # Plot SED if desired
        if plot:
            src.plot_SED(xr=[0.5,30])

        # Final source spectrum
        if return_src:
            return src
        elif return_sp:
            return src.sp_model
        else:
            self.sp_sci = src.sp_model

    def create_nircam_object(self, fov_pix=321, oversample=None, obs_hci=True, **kwargs):
        """Create NIRCam object"""

        from ..nrc_utils import conf

        log_prev = conf.logging_level
        setup_logging('WARN', verbose=False)

        if self.sp_sci is None:
            self.create_stellar_spectrum(**kwargs)

        fpath = self.obs_dict[self.obsids[0]][0]['file']
        kwargs2 = {'sp_sci': self.sp_sci, 'dist_sci': self._sp_kws.get('dist')} if obs_hci else {}
        kwargs.update(kwargs2)
        self.nrc = nrc_from_file(fpath, fov_pix, oversample=oversample, obs_hci=obs_hci, **kwargs)

        if self.nrc.aperturename != self.nrc.siaf_ap.AperName:
            self.nrc.aperturename = self.nrc.siaf_ap.AperName

        # Reset logging level
        setup_logging(log_prev, verbose=False)

        # Call this same function in the reference objects
        if self.ref_objs is not None:
            for ref_obj in self.ref_objs:
                ref_obj.create_nircam_object(fov_pix=fov_pix, oversample=oversample, obs_hci=obs_hci, **kwargs)

        
    def _simulate_psf(self, coord_vals=None, coord_frame=None, 
                      fov_pix=None, oversample=None, sp=None,
                      focus=0, diffusion_sigma=0, return_xyoff=False,
                      psf_corr_image=None):
        """Simulate PSF placed in center of array"""

        from ..nrc_utils import conf

        if self.nrc is None:
            self.create_nircam_object()

        nrc = self.nrc
        if fov_pix is not None:
            nrc.fov_pix = fov_pix
        if oversample is not None:
            nrc.oversample = oversample
        else:
            oversample = nrc.oversample

        apname = self.nrc.siaf_ap.AperName

        # Unit response to create effective PSF
        bp = nrc.bandpass
        if sp is None:
            sp = self.sp_sci
        sp_norm = sp.renorm(bp.unit_response(), 'flam', bp)

        focus = 0 if focus is None else focus
        diffusion_sigma = self.best_diffusion if diffusion_sigma is None else diffusion_sigma

        if (diffusion_sigma == 0) and (psf_corr_image is not None):
            _log.warning("PSF correction image provided, but diffusion_sigma=0. Correction should be applied after diffusion.")

        # Simulate PSF
        log_prev = conf.logging_level
        setup_logging('WARN', verbose=False)

        use_bg_psf = True if nrc.is_coron else False
        # print(coord_vals, coord_frame)
        psf_over = gen_diffusion_psf(nrc, diffusion_sigma, defocus_waves_2um=focus,
                                     return_oversample=True, return_hdul=False,
                                     sp=sp_norm, coord_vals=coord_vals, coord_frame=coord_frame,
                                     use_bg_psf=use_bg_psf, normalize='exit_pupil')

        xyoff_psf = nrc.psf_offset_to_center
        xyoff_psfs_over = xyoff_psf * oversample

        # Regenerate coronagraphic PSF
        # Don't include IPC or PPC yet. That comes later.
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

        # gen_diffusion_psf uses calc_psf, which can have slight offsets from center of array
        # Recenter PSF to place in center of array
        psf_over = nrc.recenter_psf(psf_over, sampling=oversample, shift_func=fourier_imshift)
        if psf_corr_image is not None:
            psf_over *= crop_image(psf_corr_image, psf_over.shape, fill_val=1)

        # xyoff_psfs_over is the shift required to move simulated PSF to center of array
        if return_xyoff:
            return psf_over, xyoff_psfs_over
        else:
            return psf_over
    

    def simulate_psfs(self, xysub, fov_pix=None, use_com=True, 
                      force=False, sp=None, diffusion_sigma=0):
        """Simulate PSFs for each dither position
        
        For coronagraphic observations, this only simulates a single PSF
        that is centered on the coronagraphic mask.
        """

        from webbpsf_ext.imreg_tools import load_cropped_files
        from webbpsf_ext.imreg_tools import get_com, get_expected_loc

        if self.nrc is None:
            self.create_nircam_object(fov_pix=fov_pix)

        # Simulate PSFs
        # Create oversampled PSFs for each dither location
        osamp = self.nrc.oversample
        if (self.psfs_over is None) or (force==True):

            # Ensure stellar spectrum has been created
            if self.sp_sci is None:
                self.create_stellar_spectrum()

            # Saved file
            obs_dict = self.obs_dict
            save_dir = os.path.dirname(obs_dict[self.obsids[0]][0]['file'])

            # Coronagraphic observations?
            is_coron = self.is_coron

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

                # Get location of star
                if self.is_sim:
                    xy_idl = self.expected_pos(frame='idl')
                else:
                    find_func = get_com if use_com else get_expected_loc
                    res = load_cropped_files(save_dir, files, xysub=15, bgsub=False, find_func=find_func)
                    xyind_arr = res[2].reshape([ndither,-1])

                    # Get x/y loc for each observation
                    # It will be the middle of the subarray; xyind_arr are subarray coords
                    xloc_ind = xyind_arr[:,:2].mean(axis=1)
                    yloc_ind = xyind_arr[:,2:].mean(axis=1)
                    xy_ind = np.array([xloc_ind, yloc_ind]).T
                    xy_sci = xy_ind + 1

                    # Convert to 'idl' offset coords in arcsec
                    ap = self.nrc.siaf_ap
                    xy_idl = np.array([ap.sci_to_idl(xy[0],xy[1]) for xy in xy_sci])

                ndith = len(xy_idl)

            psfs_over = []
            xyoff_psfs_over = []

            itervals = range(ndith) if ndith==1 else trange(ndith, desc='Oversampled PSFs', leave=False)
            for i in itervals:
                if is_coron:
                    coord_vals = coord_frame = None
                else:
                    coord_vals, coord_frame = (xy_idl[i], 'idl')

                res = self._simulate_psf(coord_vals=coord_vals, coord_frame=coord_frame,
                                         fov_pix=fov_pix, oversample=osamp, return_xyoff=True,
                                         sp=sp, diffusion_sigma=diffusion_sigma)
                psfs_over.append(res[0])
                xyoff_psfs_over.append(res[1])
            psfs_over = np.asarray(psfs_over)
            xyoff_psfs_over = np.array(xyoff_psfs_over)

            # Save for later
            self.psfs_over = psfs_over
            self.xyoff_psfs_over = xyoff_psfs_over

        psfs_over = crop_image(self.psfs_over, xysub*osamp, pad=True,fill_val=0)
        xyoff_psfs_over = self.xyoff_psfs_over

        return psfs_over, xyoff_psfs_over

    def get_star_positions(self, xysub=65, rin=None, bgsub=False, use_com=True, gauss_fit=False,
                           med_dithers=True, ideal_sgd=True, save=True, force=False, **kwargs):
        """Find the offset between the expected and actual position
        
        Updates self.xy_loc_ind, self.xyshift, and self.xy_mask_offset

        Parameters
        ----------
        xysub : int
            Size of subarray to use for PSF fitting
        rin : int
            Perform PSF fitting to pixels outside of this radius
        bgsub : bool
            If True, then subtract the background from the cropped image.
            The background region is defined as r>0.7*xysub/2.
        use_com : bool
            Use center of mass algorithm to first find star position. 
            Otherwise, use expected location.
        gauss_fit : bool
            Fit a 2D Gaussian to the cropped image to find the star position.
            Otherwise, perform cross correlation with PSF.
        med_dithers : bool
            If True, then median combine the dithered images before fitting.
        save : bool
            Save the star positions to a file
        force : bool
            Force re-calculation of star positions.
        """
        from webbpsf_ext.imreg_tools import find_pix_offsets, load_cropped_files
        from webbpsf_ext.imreg_tools import get_com, get_expected_loc, get_sgd_offsets
        from webbpsf_ext.image_manip import bp_fix

        def saved_path_name(save_dir):
            # Saved file
            save_str0 = '_obs' + '.'.join([str(obs) for obs in self.obsids])
            save_str1 = '_com' if use_com else '_exp'
            save_str2 = '_med' if med_dithers else ''
            save_str3 = '_gfit' if gauss_fit else ''
            save_str4 = '_sim' if self.is_sim else ''
            save_file = f'star_positions_{self.filter}{save_str0}{save_str1}{save_str2}{save_str3}{save_str4}.json'
            save_path = os.path.join(save_dir, save_file)
            return save_path

        def load_saved_file(save_dir):
            # if os.path.exists(save_path) and (force==False):
            save_path = saved_path_name(save_dir)
            _log.info(f"Loading star positions from {save_path}")
            with open(save_path, 'r') as f:
                data = json.load(f)
            self.xy_loc_ind = np.array(data['xy_loc_ind'])
            self.xyshift = np.array(data['xyshift'])

        # Check if generate_obs_dict has been run
        if len(self.obs_dict)==0:
            raise ValueError("Run generate_obs_dict() first.")
        obs_dict = self.obs_dict

        if (self.xy_loc_ind is not None) and (force==False):
            _log.info("Star positions already found. Set force=True to re-run.")
            return
        
        if self.nrc is None:
            self.create_nircam_object()

        keys = list(obs_dict[self.obsids[0]][0].keys())
        if 'xyloc' in keys:
            _log.info("Data arrays already updated. Run generate_obs_dict() to start over.")
            return

        # Saved file
        save_dir = os.path.dirname(obs_dict[self.obsids[0]][0]['file'])
        save_path = saved_path_name(save_dir)
        if os.path.exists(save_path) and (force==False):
            load_saved_file(save_dir)
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
                data_sim_all = self.get_data_arr(data_key='data')
                for i in range(len(data_sim_all)):
                    data_sim = data_sim_all[i]
                    data_shape = imsub_arr[i].shape
                    x1, x2, y1, y2 = xyind_arr[i]
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
            if not self.is_sim:
                for i in range(ndither):
                    dqsub_arr[i] = self._flag_bad_pixels(imsub_arr[i], dqsub_arr[i])
            imsub_arr = imsub_arr.reshape([-1, ny, nx])
            dqsub_arr = dqsub_arr.reshape([-1, ny, nx])
            bp_masks = bp_masks.reshape([-1, ny, nx])

            for i in range(len(imsub_arr)):
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
            if not gauss_fit:
                _log.info("Simulating PSFs...")
                psfs_over, _ = self.simulate_psfs(xysub, use_com=use_com, force=force)

            # return imsub_arr, psfs_over, bp_masks, xyind_arr

            # Choose region to fit PSF
            nrc = self.nrc
            apname = nrc.siaf_ap.AperName
            if rin is None:
                if nrc.is_coron and ('TAMASK' in apname):
                    # Target acquisitions
                    rin = 0
                elif nrc.is_coron and (self.pid in [1536, 1537, 1538]):
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
            xylim_pix = kwargs.pop('xylim_pix', None)
            if xylim_pix is None:
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
                # Get the image(s) for this dither
                imsub = imsub_arr[i]
                if len(imsub.shape)==3 and med_dithers:
                    imsub = np.nanmedian(imsub, axis=0)
                # Get the bad pixel mask(s) for this dither
                bpmask = bp_masks[i]
                if len(bpmask.shape)==3 and med_dithers:
                    bpmask = np.bitwise_and.reduce(bpmask, axis=0)

                if gauss_fit:
                    # Fit a 2D Gaussian to the image
                    fwhm_pix = self.psf_fwhm_pix() # Initial guess
                    cropsize = int(10*fwhm_pix)
                    if len(imsub.shape)==2:
                        xy_loc_sub = fit_gauss_image(imsub, bpmask=bpmask, cropsize=cropsize)
                    else:
                        xy_loc_sub = np.array([fit_gauss_image(im, bpmask=bpm, cropsize=cropsize) for im, bpm in zip(imsub, bpmask)])
                else:
                    # Get the PSF for this dither
                    psf_over = psfs_over[0] if psfs_over.shape[0]==1 else psfs_over[i]

                    # Find the best PSF shifts to match science images
                    xysh_pix = find_pix_offsets(imsub, psf_over, psf_osamp=osamp, rin=rin, xylim_pix=xylim_pix,
                                                kipc=self.kipc, kppc=self.kppc, diffusion_sigma=self.best_diffusion,
                                                psf_corr_image=self.psf_corr_over, bpmask_arr=bpmask, phase=False,
                                                **kwargs)
                    
                    # xysh_pix is the shift necessary to move a perfectly centered PSF to the star location
                    # Add to the subarray center to get the star location

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
        if self.is_coron:
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

    def shift_to_center_int(self, med_dithers=False, return_results=False, 
                            odd_shape=True, xysub=None):
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
        if xysub is None:
            pad_vals = 2 * int(np.max(np.abs(np.concatenate([xoff_int, yoff_int]))))
            ny_pad, nx_pad = (ny+pad_vals, nx+pad_vals)
            nxy_pad = np.max([nx_pad, ny_pad])
        else:
            nxy_pad = xysub

        is_even = np.mod(nxy_pad,2)==0
        is_odd = not is_even

        # Add 1 pixel if desired shape is not of right type (odd/even)
        if (odd_shape and is_even) or (not odd_shape and is_odd):
            nxy_pad += 1

        imarr_shift = []
        errarr_shift = []
        dqarr_shift = []
        bparr_shift = []
        xy0_list = []
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

            xy0_list.append(np.array([-1*xy[0], -1*xy[2]]))

        imarr_shift = np.asarray(imarr_shift)
        errarr_shift = np.asarray(errarr_shift)
        dqarr_shift = np.asarray(dqarr_shift)
        bparr_shift = np.asarray(bparr_shift)

        # Update xyshift values
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
                odict[k]['sci00'] = xy0_list[ii] # Index of (0,0) in science frame
                ii += 1

        # Update class attributes
        self.xy_loc_ind = xy_loc_shift
        self.xyshift = xyshift_new

        # Call this same function in the reference objects
        if self.ref_objs is not None:
            for ref_obj in self.ref_objs:
                ref_obj.shift_to_center_int(med_dithers=med_dithers, return_results=False)

    def _get_dither_data(self, obsids=None, subsize=None, outer_rad=32, gstd_pix=None, 
                         bpfix=False, rebin=1, order=1, data_key='data', **kwargs):

        from astropy.convolution import Gaussian2DKernel, convolve
        from webbpsf_ext.image_manip import bp_fix

        # Check flux units in header
        # If in surface brightness units, then set total=False
        total = False if self.has_sb_units else True

        dq_key = data_key.replace('data', 'dq')
        bp_key = data_key.replace('data', 'bp')

        if obsids is None:
            obsids = self.obsids
        else:
            obsids = np.array([obsids]).flatten()

        imarr = []
        dqarr = []
        bparr = []
        for oid in obsids:
            odict = self.obs_dict[oid]
            for k in odict.keys():
                # Images reduced to 2D
                im = odict[k][data_key]
                if len(im.shape)==3:
                    im = np.nanmedian(im, axis=0)
                imarr.append(im)
                # DQ arrays
                dq = odict[k][dq_key]
                if len(dq.shape)==3:
                    dq = np.bitwise_and.reduce(dq, axis=0)
                dqarr.append(dq)
                # Bad pixel masks
                bp = get_dqmask(dq, ['DO_NOT_USE']) > 0
                bp |= np.isnan(im)
                if 'bp' in odict[k].keys():
                    bp_temp = odict[k][bp_key]
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
        if bpfix:
            ndither = imarr.shape[0]
            for i in range(ndither):
                bp = bparr[i]
                im = bp_fix(imarr[i], bpmask=bp, in_place=True, niter=10)
                border = get_dqmask(dqarr[i], ['FLUX_ESTIMATED', 'REFERENCE_PIXEL']) > 0
                im[border] = 0
                imarr[i] = im

        rebin = 1 if rebin is None else rebin
        if (rebin != 1):
            imarr = image_shift_with_nans(imarr, 0, 0, oversample=rebin, return_oversample=True,
                                          order=order, total=total,  gstd_pix=gstd_pix, 
                                          preserve_nans=True, **kwargs)
            # dqarr = frebin(dqarr, scale=rebin, total=False).astype('uint32')
            bparr = frebin(bparr, scale=rebin, total=False)

        # del dqarr
        return imarr, bparr
    

    def get_dither_offsets(self, method='fourier', interp='lanczos', 
                           oversample=4, order=3, rescale_pix=True, gstd_pix=None, 
                           subsize=None, inner_rad=None, outer_rad=32, 
                           xylim_pix=(-1,1), dxy_coarse=0.1, dxy_fine=0.01,
                           lsq_diff=True, return_results=False, save=True, force=False, 
                           ideal_sgd=False, verbose=False, **kwargs):
        """Find the position offsets between dithered images via LSQ minimization

        Compares all dithered and roll images relative to each other, and stores the
        offsets in self.shift_matrix.

        Performs a coarse grid search to find the global minimum, and then a 
        fine grid search to find a more precise minimum. 

        Updates self.xyshift and self.xy_loc_ind with the new shift values optimized
        for PSF subtraction.

        """
        from webbpsf_ext.maths import round_int
        from skimage.filters import window as sk_window
        from webbpsf_ext.imreg_tools import find_pix_offsets
        
        def get_ref_offset(ref_obj):
            """Determine offset of SGD reference data relative to science data"""

            # imarr_sci, bparr_sci = self._get_dither_data(subsize=subsize, outer_rad=orad, rebin=1, gstd_pix=0)
            # imarr_ref, bparr_ref = ref_obj._get_dither_data(subsize=subsize, outer_rad=orad, rebin=1, gstd_pix=0)

            # imall_sci_over = image_shift_with_nans(imarr_sci, 0, 0, oversample=oversample, return_oversample=True, 
            #                                        order=order, rescale_pix=rescale_pix, gstd_pix=gstd_pix,
            #                                        preserve_nans=False, mean_func=None)

            # imall_ref_over = image_shift_with_nans(imarr_ref, 0, 0, oversample=oversample, return_oversample=True, 
            #                                        order=order, rescale_pix=rescale_pix, gstd_pix=gstd_pix,
            #                                        preserve_nans=False, mean_func=None)

            imall_sci_over, bparr_over_sci = self._get_dither_data(subsize=subsize, outer_rad=orad, rebin=oversample, 
                                                                   order=order, rescale_pix=rescale_pix, gstd_pix=gstd_pix)
            imall_ref_over, bparr_over_ref = self._get_dither_data(subsize=subsize, outer_rad=orad, rebin=oversample, 
                                                                   order=order, rescale_pix=rescale_pix, gstd_pix=gstd_pix)
            bparr_sci = frebin(bparr_over_sci, scale=1/oversample, total=False)
            bparr_ref = frebin(bparr_over_ref, scale=1/oversample, total=False)

            ndither_sci = imall_sci_over.shape[0]
            ndither_ref = imall_ref_over.shape[0]

            shift_matrix = np.zeros((ndither_sci, ndither_ref, 2))
            for i in trange(ndither_sci, desc='Relative Offsets', leave=False):
                im1, bp1 = (frebin(imall_sci_over[i], scale=1/oversample), bparr_sci[i])
                im1[bp1] = np.nan
                for j in range(ndither_ref):
                    im2_over, bp2 = (imall_ref_over[j], bparr_ref[j])
                    bpmask = bp1 | bp2
                    xysh_best = find_pix_offsets(im1, im2_over, psf_osamp=oversample, bpmask_arr=bpmask, 
                                                 crop=subsize, rin=inner_rad, xcorr=xcorr, lsq_diff=lsq_diff,
                                                 xylim_pix=xylim_pix, dxy_coarse=dxy_coarse, dxy_fine=dxy_fine)

                    # xysh_best = find_best_offset_wrapper(im1, im2, bp1=bp1, bp2=bp2, pixel_binning=rebin,
                    #                                      coarse_limits=coarse_limits,fine_limits=fine_limits,
                    #                                      rin=inner_rad, rout=outer_rad, method=method, interp=interp,
                    #                                      oversample=oversample, order=1,
                    #                                      weights=weights, verbose=verbose, **kwargs)
                    shift_matrix[i,j] = xysh_best

            return shift_matrix


        obs_dict = self.obs_dict

        # Check if coronagrpahic observations
        if inner_rad is None:
            inner_rad=10 if self.is_coron else 0

        # Ensure there is enough data to perform the shifts
        if outer_rad is not None:
            orad = round_int(outer_rad + np.max(np.abs(xylim_pix)) + 0.5)
        else:
            orad = None

        # Crop images to subsize
        if subsize is None and (outer_rad is not None):
            subsize = orad * 2 + 1
        if subsize is not None:
            # Ensure subsize is odd
            subsize = subsize + 1 if subsize % 2 == 0 else subsize

        # Saved file
        save_dir = os.path.dirname(obs_dict[self.obsids[0]][0]['file'])
        save_str0 = '_obs' + '.'.join([str(obs) for obs in self.obsids])
        save_str1 = '_sim' if self.is_sim else ''
        save_str = f'_{method}_{interp}_sub{subsize}_osamp{oversample}_gstd{gstd_pix}_irad{inner_rad}_orad{outer_rad}{save_str1}'
        save_file = f'star_positions_{self.filter}{save_str0}{save_str}.json'
        save_path = os.path.join(save_dir, save_file)
        if os.path.exists(save_path) and (force==False):
            _log.info(f"Loading dither positions from {save_path}")
            with open(save_path, 'r') as f:
                data = json.load(f)

            if return_results:
                return np.array(data['xyshift'])

            self.xy_loc_ind = np.array(data['xy_loc_ind'])
            self.xyshift = np.array(data['xyshift'])
            self.shift_matrix = np.array(data['shift_matrix'])
        else:

            # imarr, bparr = self._get_dither_data(subsize=subsize, outer_rad=orad, rebin=1, gstd_pix=0)
            # imall_over = image_shift_with_nans(imarr, 0, 0, oversample=oversample, return_oversample=True, 
            #                                    order=order, rescale_pix=rescale_pix, gstd_pix=gstd_pix,
            #                                    preserve_nans=False, mean_func=None)

            imall_over, bparr_over = self._get_dither_data(subsize=subsize, outer_rad=orad, rebin=oversample, 
                                                              order=order, rescale_pix=rescale_pix, gstd_pix=gstd_pix)
            bparr = frebin(bparr_over, scale=1/oversample, total=False)

            ndither = imall_over.shape[0]

            # Weight via an inverse Gaussian window
            # weights = 1 - sk_window(('gaussian', 10*rebin), imarr.shape[-2:])
            # if weights is None:
            #     weights = np.ones(imarr.shape[-2:])
            # else:
            #     weights = crop_image(weights, imarr.shape[-2:], fill_val=0)

            xcorr = False if lsq_diff else True
            shift_matrix = np.zeros((ndither, ndither, 2), dtype='float')
            for i in trange(ndither, desc='Relative Offsets', leave=False):
                im1, bp1 = (frebin(imall_over[i], scale=1/oversample), bparr[i])
                im1[bp1] = np.nan
                for j in range(ndither):
                    if i==j:
                        continue
                    # Offset needed to move im2 to align with im1
                    im2_over, bp2 = (imall_over[j], bparr[j])
                    bpmask = bp1 | bp2
                    xysh_best = find_pix_offsets(im1, im2_over, psf_osamp=oversample, bpmask_arr=bpmask, 
                                                 crop=subsize, rin=inner_rad, xcorr=xcorr, lsq_diff=lsq_diff,
                                                 xylim_pix=xylim_pix, dxy_coarse=dxy_coarse, dxy_fine=dxy_fine)

                    # xysh_best = find_best_offset_wrapper(im1, im2, bp1=bp1, bp2=bp2, pixel_binning=rebin,
                    #                                      coarse_limits=coarse_limits, fine_limits=fine_limits,
                    #                                      rin=inner_rad, rout=outer_rad, method=method, interp=interp,
                    #                                      weights=weights, verbose=verbose, **kwargs)
                    shift_matrix[i,j] = xysh_best

            del imall_over, bparr#, dqarr

            if return_results:
                return shift_matrix
            
            # Shift matrix that gives the best shift for all images relative to each other
            self.shift_matrix = shift_matrix

            # Update xyshift values to be consistent with shift_matrix offsets
            # Determine if SGD data to preserve relative offsets
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
                _log.info(f"Saving dither positions to {save_path}")
                save_data = {'xy_loc_ind': self.xy_loc_ind, 'xyshift': self.xyshift,
                            'shift_matrix': self.shift_matrix}
                with open(save_path, 'w') as f:
                    json.dump(save_data, f, cls=NumpyArrayEncoder)

        # Call this same function in the reference objects
        if self.ref_objs is not None:
            kwargs_ref = {
                'method':method, 'interp':interp, 'oversample':oversample, 
                'order':order, 'rescale_pix':rescale_pix, 'gstd_pix':gstd_pix, 
                'subsize':subsize, 'inner_rad':inner_rad, 'outer_rad':outer_rad, 
                'xylim_pix':xylim_pix, 'dxy_coarse':dxy_coarse, 'dxy_fine':dxy_fine, 
                'lsq_diff':lsq_diff, 'ideal_sgd':ideal_sgd, 'return_results':False, 
                'save':save, 'force':force, 'verbose':verbose,
            }
            
            for ref_obj in self.ref_objs:
                # ref_obj.get_dither_offsets(method=method, interp=interp, subsize=subsize, 
                #                            rebin=rebin, gstd_pix=gstd_pix, inner_rad=inner_rad, 
                #                            outer_rad=outer_rad, coarse_limits=coarse_limits, 
                #                            fine_limits=fine_limits, return_results=False, 
                #                            save=save, force=force, ideal_sgd=ideal_sgd, **kwargs)
                ref_obj.get_dither_offsets(**kwargs_ref, **kwargs)

            # Align reference observations to science observations
            # Only does the first reference object...
            return get_ref_offset(self.ref_objs[0])
            
    def _get_shift_vals(self, oid1, oid2, dith_pos1, dith_pos2, shift_matrix=None):
        """Get shift values between two dither positions"""

        if shift_matrix is None:
            shift_matrix = self.shift_matrix

        cnt=0
        for i, oid in enumerate(self.obsids):
            dith_keys = self.obs_dict[oid].keys()
            for j, dith_key in enumerate(dith_keys):
                if oid==oid1 and dith_key==dith_pos1:
                    ii = cnt
                cnt += 1

        cnt=0
        for i, oid in enumerate(self.obsids):
            dith_keys = self.obs_dict[oid].keys()
            for j, dith_key in enumerate(dith_keys):
                if oid==oid2 and dith_key==dith_pos2:
                    jj = cnt
                cnt += 1

        xysh = shift_matrix[ii,jj]

        return xysh
    
    def pixelscale(self, return_all=False):

        xscale = self.nrc.siaf_ap.XSciScale
        yscale = self.nrc.siaf_ap.YSciScale
        pixscale = 0.5 * (xscale + yscale)

        if return_all:
            return pixscale, xscale, yscale
        else:
            return pixscale

    def psf_fwhm_pix(self, return_sigma=False, return_arcsec=False, 
                     psf_to_measure=None, psf_pixscale=None, threshold=0.1,
                     gfit=True, plot=False):
        """Determine the FWHM of the PSF
        
        Calculate the FWHM of the PSF based on the bandpass and telescope diameter.
        Accounts for effectively smaller diameter for coronagraphic observations.
        Optionally return the Gaussian sigma value instead of FWHM.
        Returned values are in detector pixels.

        Parameters
        ----------
        return_sigma : bool
            Return the sigma value of the Gaussian PSF instead of FWHM.
        return_arcsec : bool
            Return the FWHM in arcseconds instead of detector pixels.
            Default is False (ie., detector pixels).
        psf_to_measure : ndarray
            Image of PSF to measure FWHM. If None, then use the theoretical
            JWST value based on average wavelength of bandpass. FWHM and sigma values
            will be return in terms of *detector* pixels even if the input PSF is
            oversampled.
        psf_pixscale : float
            Pixel scale of PSF image. If None, then use the average pixel scale
            of the NIRCam detector.
        threshold : float
            Fractional threshold above which to measure the FWHM. Default is 0.1.
        plot : bool
            Plot the radial profile of the PSF image if provided.
        """
        from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
        from astropy.modeling import models, fitting
        from scipy.interpolate import interp1d

        import matplotlib
        import matplotlib.pyplot as plt
        # plt.style.use('webbpsf_ext.wext_style')

        # Smaller effective diameter for coronagraphic observations
        diam = 5.2 if self.is_coron else 6.5

        # Averge pixel scale
        pixscale = 0.5 * (self.nrc.siaf_ap.XSciScale + self.nrc.siaf_ap.YSciScale)

        # Get telescope resolution in arcsec
        res_asec = 206265 * self.nrc.bandpass.pivot().to_value('m') / diam
        # Convert to detector pixels
        res_pix = res_asec / pixscale

        # Convert to FWHM
        fwhm_asec, fwhm_pix = 1.025 * np.array([res_asec, res_pix])

        # Convert to sigma
        sigma_asec, sigma_pix = gaussian_fwhm_to_sigma * np.array([fwhm_asec, fwhm_pix])

        if psf_to_measure is None:
            if return_sigma:
                res = sigma_asec if return_arcsec else sigma_pix
            else:
                res = fwhm_asec if return_arcsec else fwhm_pix
        else:
            psf_pixscale = self.nrc.pixelscale if psf_pixscale is None else psf_pixscale
            image = psf_to_measure.copy()
            image /= image.max()

            rho = dist_image(image, pixscale=psf_pixscale)
            ind_fit = image > threshold

            rho_fit = rho[ind_fit]
            image_fit = image[ind_fit]

            # Fit a Gaussian to the radial profile
            if gfit:
                g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=sigma_asec)
                g_init.mean.fixed = True

                fit_g = fitting.LevMarLSQFitter()
                g = fit_g(g_init, rho_fit, image_fit)

                fit_std_asec = np.abs(g.stddev.value)
                fit_fwhm_asec = gaussian_sigma_to_fwhm * fit_std_asec

            else:
                # Average duplicates
                rho_uniq = np.unique(rho_fit)
                imvals = np.array([np.mean(image_fit[rho_fit==r]) for r in rho_uniq])

                isort = imvals.argsort()
                f = interp1d(imvals[isort], rho_uniq[isort], kind='cubic', fill_value='extrapolate')
                fit_fwhm_asec = 2 * f([0.5])[0]  # FWHM in arcsec (Twice HFHM)
                fit_std_asec = gaussian_fwhm_to_sigma * fit_fwhm_asec

            # Convert from arcsec to detector pixels
            fit_std_pix = fit_std_asec / pixscale
            fit_fwhm_pix = fit_fwhm_asec / pixscale

            if return_sigma:
                res = fit_std_asec if return_arcsec else fit_std_pix
            else:
                res = fit_fwhm_asec if return_arcsec else fit_fwhm_pix

            if plot:
                plt.loglog(rho.flat, image.flat, linestyle='none', marker='o', alpha=0.5)
                plt.plot(rho_fit, image_fit, linestyle='none', marker='.', alpha=0.5)

                if gfit:
                    rmin = rho_fit[rho_fit != 0].min()
                    rfit = np.linspace(rmin, rho_fit.max(), 30)
                    plt.plot(rfit, g(rfit))
                else:
                    tfit = np.linspace(threshold, 1, 30)
                    plt.plot(f(tfit), tfit)
                plt.xlabel("Radius [arcsec]")
                plt.ylabel("Intensity relative to peak")

                plt.axhline(0.5, ls=":")
                plt.axvline(fit_fwhm_asec / 2, ls=':')
                plt.text(0.1, 0.2, f'FWHM={fit_fwhm_asec:.3f} arcsec', 
                         transform=plt.gca().transAxes)

                plt.gca().set_ylim(threshold /10, 2)
                xlim = plt.gca().get_xlim()
                plt.gca().set_xlim(xlim[0], fit_fwhm_asec * 5)

        return res
        
        
    def flag_custom_bppix(self, bp_indices, obsids=None):
        """Flag custom bad pixel mask

        Parameters
        ----------
        bp_indices : list
            List of indices to flag as bad pixels.
            [[ix1, iy1], [ix2, iy2], ...]
        obsids : list
            List of observation IDs to apply mask to
        data_key : str
            Key in obs_dict to apply mask to
        """

        if obsids is None:
            obsids = self.obsids

        for oid in obsids:
            odict = self.obs_dict[oid]
            for k in odict.keys():
                for ix, iy in bp_indices:
                    odict[k]['bp'][iy, ix] = True
            
    def flag_outlier_pix_diff(self, nsig=10, rin_flag=15, rin_fit=0, rout_fit=None, gauss_fit=True, 
                              oversample=4, shift_method='fshift', interp='linear', grid_method='nearest',
                              verbose=False, **kwargs):
        """Flag outlier pixels in difference images

        Parameters
        ----------
        nsig : float
            Number of standard deviations to flag pixels
        """

        diff_all = []
        bp_all = []

        # PSF FWHM
        fwhm_pix = self.psf_fwhm_pix()

        obs_dict = self.obs_dict
        obsids = self.obsids
        for oid1 in obsids:
            imarr1, bparr1 = self._get_dither_data(obsids=oid1, subsize=None, outer_rad=None)

            imarr2, oids2 = self._get_roll_ref(oid1, dith_pos=None, bin_ints=1, med_dithers=True, 
                                                data_key='data', return_refoids=True)

            # Get centers of all images
            if not self.is_coron and gauss_fit:
                im1_cen_arr = np.array([fit_gauss_image(im, fwhm=fwhm_pix) for im in imarr1])
                im2_cen_arr = np.array([fit_gauss_image(im, fwhm=fwhm_pix) for im in imarr2])

            for i, im1 in enumerate(imarr1):
                # Ensure bad pixels are set to NaNs
                bp1 = bparr1[i]
                im1_nans = im1.copy()
                im1_nans[bp1] = np.nan

                # Get relative offsets of reference images relative to science images
                if not self.is_coron and gauss_fit:
                    xysh_arr2 = im1_cen_arr[i] - im2_cen_arr
                else:
                    nim2 = len(imarr2)
                    xysh_arr2 = np.array([self._get_shift_vals(oid1, oids2[j], i, j) for j in range(nim2)])

                # Shift reference images to align with science image
                im1_sh = image_shift_with_nans(im1_nans, 0, 0, oversample=oversample,
                                               shift_method=shift_method, interp=interp,
                                               return_oversample=False, preserve_nans=True, 
                                               grid_method=grid_method, order=1, **kwargs)

                # Shift im2 by xyshift and interpolate NaNs
                im2_sh_arr = []
                for j, im2 in enumerate(imarr2):
                    xsh, ysh = xysh_arr2[j]
                    im2_sh = image_shift_with_nans(im2, xsh, ysh, oversample=oversample,
                                                   shift_method=shift_method, interp=interp,
                                                   return_oversample=False, preserve_nans=False,  
                                                   grid_method=grid_method, order=1, **kwargs)
                    im2_sh_arr.append(im2_sh)
                im2_sh_arr = np.array(im2_sh_arr)

                # Flag pixels that are outliers in diff image
                diff = lsq_subtraction(im1_sh, im2_sh_arr, rin=rin_fit, rout=rout_fit)
                bp_new = flag_outliers_in_diff(diff, nsig=nsig, rin=rin_flag)

                if verbose:
                    print(f"ObsID {oid1} Dither {i}: flagged {np.sum(bp_new)} additional pixels")

                diff_all.append(diff)
                bp_all.append(bp_new)

                # Assume these new bad pixels are always bad in all unshifted images
                # x0, y0 = self.obs_dict[oid1][i]['sci00']
                # for oid in obsids:
                #     for j in range(len(obs_dict[oid])):
                #         x0j, y0j = obs_dict[oid][j]['sci00']
                #         shift_ij = np.array([x0j-x0, y0j-y0])
                #         bp_new_j = fshift(bp_new, shift_ij[0], shift_ij[1], cval=False, pad=True)
                #         obs_dict[oid][j]['bp'] |= bp_new_j

                # Update bp mask in data
                obs_dict[oid1][i]['bp'] |= bp_new

        # diff_all = np.array(diff_all)
        # bp_all = np.array(bp_all)
        # diff_all[bp_all] = np.nan

        # return diff_all, bp_all

    def _shift_and_subtract_refs(self, imsci_nans, imrefs_nans, ref_shifts, rebin=1, return_oversample=True,
                                 method='opencv', interp='lanczos', preserve_nans=True, order=3, gstd_pix=None,
                                 do_pca=False, do_loci=True, rin=0, rout=10, **kwargs):
        """Align reference images to science image and subtract"""

        from . import pca

        # Check flux units in header
        # If in surface brightness units, then set total=False
        total = False if self.has_sb_units else True

        # Align reference images to science image using ref_shifts
        ref_shape = imrefs_nans.shape
        ndim_shape = len(ref_shape)
        if ndim_shape==2:
            # Reshepae to 3D array
            # imrefs_nans = np.expanad_dims(imrefs_nans, axis=0)
            imrefs_nans = imrefs_nans.reshape([-1,ref_shape[0],ref_shape[1]])
        elif ndim_shape==3:
            pass
        else:
            raise ValueError(f"Unexpected shape of im_refs: {imrefs_nans.shape}")
        
        nz, ny, nx = imrefs_nans.shape

        if nz<3 and do_pca:
            _log.warning(f"Very few reference images ({nz}) to perform PCA.")
            do_pca=False

        im_refs_shift = []
        for i, im_ref in enumerate(imrefs_nans):
            xsh, ysh = ref_shifts[i]
            im_sh = image_shift_with_nans(im_ref, xsh, ysh, oversample=rebin, return_oversample=True, 
                                          shift_method=method, interp=interp, order=order, total=total,
                                          preserve_nans=preserve_nans, gstd_pix=gstd_pix, **kwargs)
            im_refs_shift.append(im_sh)
        im_refs_shift = np.asarray(im_refs_shift)

        # Subtract reference images from science image
        if rebin==1:
            im_sci_over = imsci_nans
        else:
            im_sci_over = image_shift_with_nans(imsci_nans, 0, 0, oversample=rebin, return_oversample=True, 
                                                shift_method=method, interp=interp, order=order, total=total,
                                                preserve_nans=preserve_nans, gstd_pix=gstd_pix, **kwargs)
            
        bpmask_sci = np.isnan(im_sci_over)
        bpmask_ref = np.logical_or.reduce(np.isnan(im_refs_shift))
        bpmask_over = bpmask_sci | bpmask_ref

        if do_pca:
            fwhm_pix_bin = self.psf_fwhm_pix() * rebin
            imdiff, pca_params = pca_subtraction(im_sci_over, im_refs_shift, fwhm_pix_bin, 
                                                 loci=do_loci, **kwargs)
        else:
            imdiff = lsq_subtraction(im_sci_over, im_refs_shift)

        return imdiff

    # def align_diff_images(self, rebin=1, return_oversample=True, gstd_pix=None, #ref_obs=None, 
    #                  med_dithers=False, method='opencv', interp='lanczos', 
    #                  preserve_nans=True, order=3, new_shape=None,
    #                  obs_sub=None, med_dithers=False, data_key='data_aligned',
    #                     bin_ints_sci=1, bin_ints_ref=1, all_pos=True, do_pca=True,
    #                     do_rdi=True, **kwargs):):

    def align_images(self, rebin=1, return_oversample=True, gstd_pix=None, #ref_obs=None, 
                     med_dithers=False, method='opencv', interp='lanczos', 
                     preserve_nans=True, order=3, new_shape=None, **kwargs):
        """Align all images to a common reference frame
        
        Adds 'data_aligned', 'err_aligned', 'dq_aligned', 'bp_aligned', 'xy_aligned', and 'bin_aligned' 
        to each observation dictionary in self.obs_dict. The data have had their bad pixels fixed. 
        The dq arrays and bad pixel masks have been shifted to match the new image locations.

        Parameters
        ----------
        ref_obs : tuple or None
            Tuple of observation number and dither position to align all images to. 
            If None, then align all images to center of array.
        rebin : int
            Factor to rebin images before shifting. Results will be updated to new scale.
        return_oversample : bool
            Save oversampled version of image after shifting.
        gstd_pix : float
            Standard deviation of Gaussian kernel for smoothing images before shifting.
        med_dithers : bool
            Median combine dithered images before aligning.
        method : str
            Method for shifting images. Options are 'fshift', 'opencv', or 'fourier'.
        interp : str
            Type of interpolation to use during the sub-pixel shift. Valid values are
            'linear', 'cubic', and 'quintic' for `fshift` method (default: 'linear').
            For `opencv` method, valid values are 'linear', 'cubic', and 'lanczos' 
            (default: 'lanczos'). Not valid for `fourier` method.
        preserve_nans : bool
            Preserve NaN values in the image when shifting. If False, then NaNs are
            replaced with interpolated values.
        order : int
            The order of the spline interpolation for `zrebin` function, Default is 3. 
            Only used if rebin>1. If order=0, then `frebin` is used.
        new_shape : tuple
            Crop/expand images to a specific new (ny,nx) shape. If None, then shapes 
            are automatically expanded based on shift amounts.
        """

        if self.shift_matrix is None:
            _log.warning("shift_matrix attribute is None. Did you want to get_dither_offsets() first?")
            
        # Check flux units in header
        # If in surface brightness units, then set total=False
        total = False if self.has_sb_units else True

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
                bp = odict[k]['bp']
                bp |= (get_dqmask(dq, ['DO_NOT_USE']) > 0)
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
                # im[border] = 0
                err[border] = np.nanmax(err)
                err[np.isnan(err)] = np.nanmax(err)

                xsh, ysh = self.xyshift[ii] - np.array([xsh0, ysh0])
                # im_shift = fractional_image_shift(im, xsh, ysh, method=method, interp=interp,
                #                                   oversample=rebin, gstd_pix=gstd_pix, 
                #                                   return_oversample=True, total=total)
                
                # Get shifted image arrays
                im_shift = image_shift_with_nans(im, xsh, ysh, oversample=rebin, order=order, 
                                                 shift_method=method, interp=interp, gstd_pix=gstd_pix,
                                                 return_oversample=return_oversample, 
                                                 preserve_nans=preserve_nans,
                                                 pad=True, total=total, **kwargs)

                # Get shifted error arrays
                var_shift = fractional_image_shift(err**2, xsh, ysh, method='fshift', interp='linear',
                                                   oversample=rebin, gstd_pix=gstd_pix, 
                                                   return_oversample=return_oversample, total=total)
                err_shift = np.sqrt(var_shift)
                
                # Get shifted bad pixel mask
                bp_shift = frebin(bp.astype('float'), scale=rebin, total=False)
                bp_shift = fshift(bp_shift, xsh*rebin, ysh*rebin, interp='linear', pad=True, cval=1)
                if not return_oversample:
                    bp_shift = frebin(bp_shift, scale=1/rebin, total=False)
                bp_shift = bp_shift > 0

                # Get shifted DQ array
                dq_shift = frebin(dq.astype('float'), scale=rebin, total=False)
                xsh_dq = np.sign(xsh) * np.ceil(np.abs(xsh*rebin))
                ysh_dq = np.sign(ysh) * np.ceil(np.abs(ysh*rebin))
                fill_val = dqflags.pixel['FLUX_ESTIMATED'] | dqflags.pixel['DO_NOT_USE']
                dq_shift = fshift(dq_shift, xsh_dq, ysh_dq, pad=True, cval=fill_val)
                if not return_oversample:
                    dq_shift = frebin(dq_shift, scale=1/rebin, total=False)
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
                odict[k]['method_aligned'] = method
                odict[k]['interp_aligned'] = interp

                ii += 1

        # Call this same function in the reference objects
        if self.ref_objs is not None:
            for ref_obj in self.ref_objs:
                sci_shape = self.obs_dict[self.obsids[0]][0]['data_aligned'].shape[-2:]
                ref_obj.align_images(rebin=rebin, gstd_pix=gstd_pix, med_dithers=med_dithers, 
                                     method=method, interp=interp, new_shape=sci_shape, **kwargs)

    def find_best_diffusion(self, subsize=15, data_key='data_aligned', force_psf=False,
                            imall=None, bpall=None, use_mean=True, psf_corr_over=None,
                            bounds=(0,3), return_ims=False, verbose=False):
        """ Find the best diffusion value that matches PSF to data
        
        Assume observed data is a point source.

        Updates self.best_diffusion.

        Parameters
        ----------
        subsize : int
            Size of subarray to use for fitting.
        data_key : str
            Key in obs_dict to use for data. Default is 'data_aligned'.
        force_psf : bool
            Force PSF to be recalculated.
        imall : ndarray, optional
            Combined data array to use for fitting. If None, then use data from obs_dict.
            If these data are not aligned, then update `data_key` to 'data'.
        bpall : ndarray, optional
            Bad pixel mask to use for fitting. If None, then use bad pixel mask from obs_dict,
            or set bad pixel mask based on NaNs in `imall`.
        use_mean : bool
            Use the mean of all frames for fitting. Otherwise, use individual dithers.
            Only valid for aligned images.
        psf_corr_over : ndarray
            PSF correctioni factor to apply to oversampled PSFs.
        bounds : tuple
            Bounds for diffusion value fitting.
        return_ims : bool
            Return the difference images along with best-fit diffusion value.
        verbose : bool
            Print out additional information during fitting.
        """
        from scipy.optimize import minimize_scalar

        osamp = self.nrc.oversample

        def _make_diff(df_sigma, image, psf_over, bpmask, xyshift=(0,0), return_ims=False):
            """ Difference between single image and PSF 
            
            `xyshift` corresponds to the shift required to align the PSF to the data.
            """

            return_sum2 = not return_ims
            diff = subtract_psf(image, psf_over, osamp=osamp, bpmask=bpmask, rin=None, rout=None,
                                xyshift=xyshift, psf_scale=None, psf_offset=0,
                                method='fourier', pad=True, cval=0, interp='lanczos', 
                                kipc=self.kipc, kppc=self.kppc, diffusion_sigma=df_sigma, 
                                psf_corr_over=psf_corr_over, weights=None, return_sum2=return_sum2)
            
            if return_ims:
                return diff
            else:
                if verbose and use_mean:
                    print(df_sigma, np.sum(diff**2), np.std(diff), robust.medabsdev(diff))
                return diff
        
        def _make_diff_arr(df_sigma, images, psfs_over, bpmasks, return_ims=False):
            """ Differences on individual dithers with shifts"""

            diff_arr = []
            for i, (image, psf_over, bpmask) in enumerate(zip(images, psfs_over, bpmasks)):
                xysh = (0,0) if 'aligned' in data_key else -1*self.xyshift[i]
                diff = _make_diff(df_sigma, image, psf_over, bpmask, xyshift=xysh,
                                  return_ims=return_ims)
                diff_arr.append(diff)
            diff_arr = np.array(diff_arr)
            
            if return_ims:
                return diff_arr
            else:
                if verbose:
                    print(df_sigma, diff_arr.sum(), diff_arr)
                return diff_arr.sum()

        # Get all data
        if imall is None:
            imall = self.get_data_arr(data_key=data_key)
            # Set bad pixels to NaN
            bp_key = data_key.replace('data', 'bp')
            bpall = self.get_data_arr(data_key=bp_key)
            imall[bpall] = np.nan
        else:
            bpall = np.isnan(imall) if bpall is None else bpall

        # Oversampled PSFs centered in array; no diffusion, IPC, or PPC has been applied
        psfs_over, _ = self.simulate_psfs(65, diffusion_sigma=0, force=force_psf)

        # Use averages for psf fitting
        if use_mean and ('aligned' in data_key):
            # Take mean of all frames and PSFs
            im_mean = np.nanmean(imall, axis=0)
            psf_over = np.mean(psfs_over, axis=0)

            # Crop to subarray size
            im_mean = crop_image(im_mean, subsize)
            bp_mask = crop_image(np.logical_and.reduce(bpall, axis=0), subsize)
            psf_over = crop_image(psf_over, subsize*osamp)

            args = (im_mean, psf_over, bp_mask)
            res = minimize_scalar(_make_diff, args=args, bounds=bounds, method='bounded')

            if return_ims:
                diff = _make_diff(res.x, im_mean, psf_over, bp_mask, return_ims=True)
                return res.x, diff
        else:
            if use_mean:
                _log.warning("`use_mean=True`, but images are not aligned, so using individual dithers instead.")

            # Use cropped individual frames
            imall_crop = crop_image(imall, subsize)
            bpall_crop = crop_image(bpall, subsize)
            psfs_over_crop = crop_image(psfs_over, subsize*osamp)

            args = (imall_crop, psfs_over_crop, bpall_crop)
            res = minimize_scalar(_make_diff_arr, args=args, bounds=bounds, method='bounded')
            if return_ims:
                diffs = _make_diff_arr(res.x, imall_crop, psfs_over_crop, bpall_crop, return_ims=True)
                return res.x, diffs

        self.best_diffusion = res.x
        # return res.x


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
            bp_key = data_key.replace('data', 'bp')
            is_err = False
        if 'err' in data_key:
            dq_key = data_key.replace('err', 'dq')
            bp_key = data_key.replace('err', 'bp')
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
                bpmask = odict[k][bp_key]
                dqmask = get_dqmask(dqref, ['DO_NOT_USE']) > 0
                imref[dqmask | bpmask] = np.nan

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
                      med_dithers=False, data_key='data_aligned',
                      return_refoids=False):
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
            bp_key = data_key.replace('data', 'bp')
            is_err = False
        if 'err' in data_key:
            dq_key = data_key.replace('err', 'dq')
            bp_key = data_key.replace('err', 'bp')
            is_err = True

        ny, nx = self.obs_dict[self.obsids[0]][0][data_key].shape[-2:]

        ref_obs = []
        ref_oids = []
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
                bpmask = odict2[k][bp_key]
                dqmask = get_dqmask(dqref, ['DO_NOT_USE']) > 0
                imref[dqmask | bpmask] = np.nan

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
                ref_oids.append(oid2)

        ref_obs = np.array(ref_obs).reshape([-1,ny,nx])
        ref_oids = np.array(ref_oids)
        if return_refoids:
            return ref_obs.squeeze(), ref_oids
        else:
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

        def basic_subtraction(imarr, imref, func_mean=np.nanmean):
            """Basic subtraction of images"""
            if len(imref.shape)==3:
                imref = func_mean(imref, axis=0)

            scale = np.nansum(imarr) / np.nansum(imref)
            return imarr - imref * scale
        
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
                    fwhm_pix_bin = self.psf_fwhm_pix() * odict[k]['bin_aligned']
                    imdiff, pca_params = pca_subtraction(imarr, imarr_ref, fwhm_pix_bin, **kwargs)
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

    def get_data_arr(self, obsid=None, data_key='data', dither=None, squeeze=True):
        """Get all data for a given observation ID"""

        def _get_data(obsid, dither):
            return obs_dict[obsid][dither][data_key]
        
        def _get_obs_data(obsid):
            return np.array([_get_data(obsid, d) for d in obs_dict[obsid].keys()])
        
        def _get_all_data():
            return np.concatenate([_get_obs_data(oid) for oid in self.obsids], axis=0)

        # Check if obsid is in reference objects
        if (obsid is not None) and (self.obsids_ref is not None) and (obsid in self.obsids_ref):
            obs_dict = self.ref_objs[self.obsids_ref.index(obsid)].obs_dict
        else:
            obs_dict = self.obs_dict

        if obsid is None:
            # Get all science data if obsid is not specified
            data_arr = _get_all_data()
        elif dither is None:
            # Get data from specified obsid
            data_arr = _get_obs_data(obsid)
        else:
            # Get data from specified obsid and dither position
            data_arr = _get_data(obsid, dither)

        if squeeze:
            return data_arr.squeeze()
        else:
            return data_arr

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
        
def replace_data_with_sim(self, return_sims=False, use_coeff=False, **kwargs):
    """Replace observed data with simulated data"""

    def gen_sim_rate_images(data_shape, nrc, PA, xyoff_asec, gain_scale=True, photmjsr=1, **kwargs):
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
                
        im_sim = nrc.gen_slope_image(PA=PA, xyoff_asec=xyoff_asec, #exclude_noise=False,
                                     shift_method='fourier', return_oversample=False, 
                                     use_coeff=use_coeff, kipc=self.kipc, kppc=self.kppc, 
                                     diffusion_sigma=self.best_diffusion,
                                     psf_corr_over=self.psf_corr_over, **kwargs)

        im_sim = crop_image(im_sim, data_shape[-2:])

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
    photmjsr = hdr1['PHOTMJSR'] if self.has_sb_units else 1

    imarr = []
    pa_arr = []
    for oid in self.obsids:
        odict = self.obs_dict[oid]
        for k in odict.keys():
            im = odict[k]['data']
            if len(im.shape)==3:
                im = np.nanmean(im, axis=0)
            imarr.append(im)

            hdr1 = odict[k]['hdr1']
            pa = hdr1['ROLL_REF'] + hdr1['V3I_YANG']
            pa_arr.append(pa)

    imarr = np.asarray(imarr)
    pa_arr = np.asarray(pa_arr)

    # imarr = self.get_data_arr(data_key='data')
    # if len(imarr.shape)==4:
    #     imarr = np.nanmean(imarr, axis=1)

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


    xy_idl = self.expected_pos(frame='idl')

    from ..nrc_utils import conf
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)

    # Add WFE drift between different rolls
    pa_temp = (pa_arr*10).astype('int') / 10 # uniq with 0.1 deg precision
    pa_uniq = np.unique(pa_temp)
    wfe_start = np.linspace(0, 5, len(pa_uniq))
    wfe_drift = np.zeros_like(pa_temp)
    for i, pa in enumerate(pa_uniq):
        ind = pa_temp==pa
        wfe_drift[ind] = wfe_start[i] + np.linspace(0, 2, np.sum(ind))
    # Add in about 20 nm of drift to simulate drift relative to OPD measurement from WFS data
    wfe_drift += 20

    imarr_sim = []
    for i in trange(ndith, leave=False, desc='Simulating Data'):
        # Case of single image per dither
        if nimg_per_dither==1:
            xyoff_asec = xy_idl[i] + xy_mask_offset[i] * self.nrc.pixelscale
            data_cube = gen_sim_rate_images(imarr[i].shape, nrc, pa_arr[i], xyoff_asec, 
                                            gain_scale=gain_scale, photmjsr=photmjsr, 
                                            wfe_drift0=wfe_drift[i], **kwargs)
            imarr_sim.append(data_cube)
        else:
            imlist = []
            for j in range(nimg_per_dither):
                xyoff_asec = xy_mask_offset[i] if nsh_per_dither==1 else xy_mask_offset[i,j] 
                xyoff_asec = xy_idl[i] + xyoff_asec * self.nrc.pixelscale
                data_cube = gen_sim_rate_images(imarr[i,j].shape, nrc, pa_arr[i], xyoff_asec, 
                                                gain_scale=gain_scale, photmjsr=photmjsr, 
                                                wfe_drift0=wfe_drift[i], **kwargs)
                imlist.append(data_cube)
            imarr_sim.append(np.asarray(imlist))
    imarr_sim = np.asarray(imarr_sim)

    setup_logging(log_prev, verbose=False)

    if return_sims:
        return imarr_sim

    # Update odict with shifted images
    ii = 0
    for i, oid in enumerate(self.obsids):
        odict = self.obs_dict[oid]
        for k in odict.keys():
            data_sim = imarr_sim[ii]
            data_shape = odict[k]['data'].shape
            data_sim_shape = data_sim.shape

            # print(data_shape, data_sim_shape)
            if len(data_shape) != len(data_sim_shape):
                data_sim = data_sim[None,:,:] if len(data_sim_shape)==2 else data_sim[None,:,:,:]
                data_sim = np.repeat(data_sim, data_shape[0], axis=0)
            data_sim = data_sim.reshape(data_shape)

            del odict[k]['data']
            odict[k]['data'] = data_sim

            ii += 1

    # Set self.xy_loc_ind to None so we can re-run get_star_positions()
    self.xy_loc_ind = None

def create_offset_psfs(nrc_obs, xysize, fov_pix=None, 
                       diffusion_sigma=0, kipc=None, kppc=None,
                       psf_scale_over=None, return_oversample=False,
                       no_offset=False, force=None, gstd_pix=0):
    
    """Create a set of PSFs with offsets corresponding to the dither locations
    
    Can specify charge diffusion, IPC, and PPC kernels to apply to the PSFs.

    If kipc and kppc are set to None, then they will use properties from the `nrc_obs` object.

    If `return_oversample` is True, then the function will return the oversampled PSFs
    without kipc and kppc applied. Any specified diffusion will still be applied.

    Set `no_offset` to True to skip the offsetting of the PSFs.

    Use psf_scale_over to apply a scale factor to the oversampled PSFs to better match
    empirical PSFs. 

    Order of operations:
    - Apply diffusion to oversampled PSFs
    - Apply PSF scale factor to oversampled PSFs
    - Shift oversampled PSFs 
    - Apply additional Gaussian smoothing
    - Rebin to detector sampling
    - Apply IPC and PPC kernels
    """

    osamp = nrc_obs.nrc.oversample
    # Oversampled PSF shape
    sh_imover = (xysize*osamp, xysize*osamp)
    if force is None:
        if fov_pix is not None:
            force = True if fov_pix >= nrc_obs.nrc.fov_pix else False
        else:
            force = False

        # if nrc_obs.psfs_over is None:
        #     ny_psf_over = nx_psf_over = 0
        # else:
        #     ny_psf_over, nx_psf_over = nrc_obs.psfs_over.shape[-2:]
        # if (ny_psf_over >= sh_imover[0]) and (nx_psf_over >= sh_imover[1]):
        #     force = False
        # else:
        #     force = True

    # Replaces nrc_obs.psfs_over
    psfs_over, _ = nrc_obs.simulate_psfs(xysize, fov_pix=fov_pix, diffusion_sigma=0, force=force)
    psfs_over = crop_image(psfs_over, sh_imover)
    xyshifts_over = -1 * nrc_obs.xyshift * osamp

    # Apply a scale factor to the oversampled PSFs?
    if psf_scale_over is not None:
        shape_over = psfs_over.shape[-2:]
        psf_scale_over = crop_image(psf_scale_over, shape_over, fill_val=1)

    kipc = nrc_obs.kipc if kipc is None else kipc
    kppc = nrc_obs.kppc if kppc is None else kppc

    psf_res_arr = []
    for i, psf in enumerate(psfs_over):
        # Apply diffusion
        if (diffusion_sigma is not None) and (diffusion_sigma >= 0):
            psf = apply_pixel_diffusion(psf, diffusion_sigma*osamp)

        # Apply a PSF scale factor
        if psf_scale_over is not None:
            psf *= psf_scale_over

        # Shift oversampled PSF
        xsh_over, ysh_over = (0,0) if no_offset else xyshifts_over[i]
        # Even if no shift, still apply additional Gaussian smoothing if requested
        psf = fractional_image_shift(psf, xsh_over, ysh_over, method='fourier', 
                                     pad=True, gstd_pix=gstd_pix*osamp)

        if return_oversample:
            psf_res_arr.append(psf)
        else:
            # Rebin to detector sampling
            psf_det = frebin(psf, scale=1/osamp) if osamp!=1 else psf
            if kipc is not None:
                psf_det = add_ipc(psf_det, kernel=kipc)
            if kppc is not None:
                psf_det = add_ppc(psf_det, kernel=kppc, nchans=1)
            psf_res_arr.append(psf_det)

    return np.asarray(psf_res_arr)

def load_psf_correction(filter, apname, verbose=True, file=None):
    """Load PSF correction factor"""

    fstart = f'psf_scale_factor_{filter}_{apname}_'
    psf_scale_dir = 'psf_scale_factors'
    if file is None:
        psf_scale_files = [f for f in os.listdir(psf_scale_dir) if 
                            f.startswith(fstart) and f.endswith('.fits')]
    else:
        psf_scale_files = [file]
    
    if verbose:
        _log.info(f'Found {len(psf_scale_files)} PSF correction files for {filter} and {apname}')
        print(psf_scale_files)

    if len(psf_scale_files)==0:
        _log.warning(f'No PSF correction files found for {filter} and {apname}')
        return {}

    psf_corr_dict = {}

    hdulist = fits.HDUList()
    for f in psf_scale_files:
        hdul_temp = fits.open(os.path.join(psf_scale_dir, f))
        hdul_temp[0].header['EXTNAME'] = hdul_temp[0].header['PID']
        hdulist.append(hdul_temp[0])

    psf_corr_dict['files'] = psf_scale_files
    psf_corr_dict['hdulist'] = hdulist

    # Read and average PSF scale factors
    psf_scales_data_all = np.array([hdu.data for hdu in hdulist]) 
    
    # First replace 1's with NaNs
    psf_scales_data_all[psf_scales_data_all==1] = np.nan

    # Take the average of all PSF scale factors
    psf_scale_data = np.nanmean(psf_scales_data_all, axis=0)
    psf_scale_data[np.isnan(psf_scale_data)] = 1

    psf_corr_dict['psf_scale_data'] = psf_scale_data

    _log.info(f'Successfuly Loaded PSF correction image for {filter} and {apname}')
    return psf_corr_dict


class nrc_rdi():

    def __init__(self, nrc_obs):
        self.nrc_obs = nrc_obs
        self.nrc = nrc_obs.nrc

        self._imcube_sci = None
        self._bpcube_sci = None
        self._errcube_sci = None
        self._data_key_sci = None
        self._offsets_sci = None

        self._imcube_css = None

        self._posangs_sci = None
        
        self._imcube_ref = None
        self._bpcube_ref = None
        self._errcube_ref = None

        self.cropped_shape = None
        self._crop_indices = None

        self._gstd_pix = None
        self.psfs_over = None
        self.psfs_det = None

        self.output_ext = 'psfsub'

        self.fixed_rdi_settings = {}
        self.rdi_settings = {}
        self._optzones = None
        self._subzones = None

    @property
    def angles(self):
        return self._posangs_sci
    @angles.setter
    def angles(self, value):
        self._posangs_sci = value

    @property
    def _im_shape_orig(self):
        if self._imcube_sci is None:
            oid1 = self.nrc_obs.obsids[0]
            im = self.nrc_obs.get_data_arr(oid1, data_key='data_aligned', dither=0)
            return im.shape[-2:]
        else:
            return self._imcube_sci.shape[-2:]

    @property
    def ndith(self):
        """Number of dither positions in each roll"""
        oid1 = self.nrc_obs.obsids[0]
        odict = self.nrc_obs.obs_dict[oid1]
        ndith = 0
        for k in odict.keys():
            # Check if item is a dictionary
            if isinstance(odict[k], dict):
                ndith += 1
        return ndith
        
    @property
    def gstd_pix(self):
        if self._gstd_pix is None:
            return self.nrc_obs.psf_fwhm_pix(return_sigma=True)
        else:
            return self._gstd_pix
        
    @property
    def _fwhm(self):
        return self.nrc_obs.psf_fwhm_pix()
        
    # Science data
    @property
    def imcube_sci(self):
        return self._crop_image(self._imcube_sci)
    @property
    def bpcube_sci(self):
        if self.imcube_sci is None:
            _log.warning("No science data found. Returning None...")
            return None
        elif self._bpcube_sci is None:
            return np.isnan(self.imcube_sci)
        else:
            return self._crop_image(self._bpcube_sci)
    @property
    def errcube_sci(self):
        if self.imcube_ref is None:
            _log.warning("No science data found. Returning None...")
            return None
        elif self._errcube_sci is None:
            return np.zeros_like(self.imcube_sci) #np.ones_like(self.imcube_sci)
        else:
            return self._crop_image(self._errcube_sci)
        
    @property
    def offsets_sci(self):
        """Offset values for each dither position"""
        if self._offsets_sci is not None:
            return self._offsets_sci
        elif self._data_key_sci is None:
            return None
        elif 'aligned' in self._data_key_sci:
            nz = len(self._imcube_sci)
            return np.zeros((nz,2))
        else:
            return -1 * self.nrc_obs.xyshift
    @property
    def star_cens(self):
        """Star position for each dither position"""
        if self.offsets_sci is None:
            return None
        else:
            sci_shape = self.imcube_sci.shape[-2:]
            im_cen = get_im_cen(np.zeros(sci_shape))
            return im_cen + self.offsets_sci
        
    # Circumstellar disk model (convolved with PSF)
    @property
    def imcube_css(self):
        return self._crop_image(self._imcube_css)
    
    # Reference Data
    @property
    def imcube_ref(self):
        if isinstance(self._imcube_ref, (list,tuple)):
            return [self._crop_image(im) for im in self._imcube_ref]
        else:
            return self._crop_image(self._imcube_ref)
    @property
    def bpcube_ref(self):
        if self.imcube_ref is None:
            _log.warning("No reference data found. Returning None...")
            return None
        elif self._bpcube_ref is None:
            if isinstance(self._imcube_ref, (list,tuple)):
                return [np.isnan(im) for im in self.imcube_ref]
            else:
                return np.isnan(self.imcube_ref)
        elif isinstance(self._bpcube_ref, (list,tuple)):
            return [self._crop_image(im) for im in self._bpcube_ref]
        else:
            return self._crop_image(self._bpcube_ref)
    @property
    def errcube_ref(self):
        if self.imcube_ref is None:
            _log.warning("No reference data found. Returning None...")
            return None
        elif self._errcube_ref is None:
            if isinstance(self._imcube_ref, (list,tuple)):
                return [np.zeros_like(im) for im in self.imcube_ref] # [np.ones_like(im) for im in self.imcube_ref]
            else:
                return np.zeros_like(self.imcube_ref) # np.ones_like(self.imcube_ref)
        elif isinstance(self._errcube_ref, (list,tuple)):
            return [self._crop_image(im) for im in self._errcube_ref]
        else:
            return self._crop_image(self._errcube_ref)
    
    @property
    def optzones(self):
        return self._crop_image(self._optzones)
    @property
    def subzones(self):
        return self._crop_image(self._subzones)

    def _crop_image(self, im):
        """Generic function to perform image cropping depending on set shapes and indices"""
        if (self._crop_indices is None) or (im is None):
            return im
        
        sh_orig = self._im_shape_orig
        y1, y2, x1, x2 = self._crop_indices

        # Make sure im shape matches the science image shape
        if not (im.shape[-2:] == sh_orig):
            action_str = "Cropping" if im.shape[-1:] > sh_orig[-1] else "Padding"
            _log.warning(f"Image shape does not match science image shape. {action_str} image to match.")
            im = crop_image(im, sh_orig)

        # Ensure cropped indices are within bounds
        if x1<0 or y1<0 or x2>sh_orig[1] or y2>sh_orig[0]:
            return crop_image(im, self.cropped_shape)
        else:
            return im[..., y1:y2, x1:x2]

    def set_crop(self, cropped_shape=None, auto_pad_nfwhm=5):
        """Specify a shape to crop the data to in pixels (ny,nx)"""
        sh_orig = self._im_shape_orig

        if cropped_shape == 'auto':
            rmax = np.max(dist_image(np.zeros(sh_orig))[np.any(self._optzones, axis=0)])
            new_nx = new_ny = int(rmax*2 + auto_pad_nfwhm*self._fwhm)
            cropped_shape = (new_ny, new_nx)

        if cropped_shape is not None:
            _, (x1,x2,y1,y2) = crop_image(self._imcube_sci[0], cropped_shape, return_xy=True)
            if x1<0 or y1<0 or x2>sh_orig[1] or y2>sh_orig[0]:
                raise ValueError("Cropped indices are out of bounds of original shape.")

            self._crop_indices = (y1,y2,x1,x2)
            self.cropped_shape = cropped_shape

        else:
            self._crop_indices = None
            self.cropped_shape = None

    def gen_sci_images(self, data_key='data', oid_ref=None, remove_med_bg=False,
                       correct_nans=True, gstd_pix=None, verbose=False):

        from astropy.convolution import Gaussian2DKernel
        # from .image_manip import image_convolution

        nrc_obs = self.nrc_obs
        self._data_key_sci = data_key

        oids = nrc_obs.obsids
        ny, nx = nrc_obs.get_data_arr(obsid=oids[0], data_key=data_key).shape[-2:]
        imall = nrc_obs.get_data_arr(data_key=data_key).reshape(-1,ny,nx)
        if oid_ref is not None:
            imarr_ref = nrc_obs.get_data_arr(oid_ref, data_key=data_key)

        data_key = data_key.replace('data', 'err')
        errall = nrc_obs.get_data_arr(data_key=data_key).reshape(-1,ny,nx)
        if oid_ref is not None:
            imerr_ref = nrc_obs.get_data_arr(oid_ref, data_key=data_key)

        data_key = data_key.replace('err', 'bp')
        bpall = nrc_obs.get_data_arr(data_key=data_key).reshape(-1,ny,nx)
        if oid_ref is not None:
            bparr_ref = nrc_obs.get_data_arr(oid_ref, data_key=data_key)

        imall[bpall] = np.nan

        if verbose:
            print('sci shapes:', imall.shape, errall.shape, bpall.shape)
            if oid_ref is not None:
                print('ref shapes:', imarr_ref.shape, imerr_ref.shape, bparr_ref.shape)


        angles = []
        for oid in nrc_obs.obsids:
            hdr = nrc_obs.get_header(oid, ext=1)
            pa = hdr['ROLL_REF'] + hdr['V3I_YANG']

            nim = len(nrc_obs.obs_dict[oid])
            pa_arr = np.ones([nim]) * pa
            angles.append(pa_arr)
        angles = np.concatenate(angles)

        # Subtraction local background
        if remove_med_bg:
            imall = imall.copy()
            rho = dist_image(imall[0])
            for i, im in enumerate(imall):
                std = robust.medabsdev(im[rho>5])
                ind = (im>-20*std) & (im<20*std) & (rho>5)
                imall[i] -= np.nanmedian(im[ind])

            if oid_ref is not None:
                imarr_ref = imarr_ref.copy()
                rho = dist_image(imarr_ref[0])
                for i, im in enumerate(imarr_ref):
                    std = robust.medabsdev(im[rho>5])
                    ind = (im>-20*std) & (im<20*std) & (rho>5)
                    imarr_ref[i] -= np.nanmedian(im[ind])

        # Correct NaNs using interpolation
        if correct_nans:
            # Replace NaN with interpolated / extrapolated values
            # Even if not aligned, better to correct NaNs from stack than from neighbors
            imall = replace_nans(imall)
            # if 'aligned' in data_key:
            #     imall = replace_nans(imall)
            # else:
            #     imall = np.array([replace_nans(im) for im in imall])

        # Apply Gaussian smoothing
        if (gstd_pix is not None) and (gstd_pix>0):
            kernel = Gaussian2DKernel(x_stddev=gstd_pix)
            if len(imall.shape)==3:
                imall = np.array([image_convolution(im, kernel) for im in imall])
            else:
                imall = image_convolution(imall, kernel)

        self._imcube_sci  = imall
        self._bpcube_sci  = bpall
        self._errcube_sci = errall
        self._posangs_sci = angles

        if oid_ref is not None:
            if correct_nans:
                # Replace NaN with interpolated / extrapolated values
                # Even if not aligned, better to correct NaNs from stack than from neighbors
                imarr_ref = replace_nans(imarr_ref)
                # if 'aligned' in data_key:
                #     self._imcube_ref = replace_nans(imarr_ref)
                # else:
                #     self._imcube_ref = np.array([replace_nans(im) for im in imarr_ref])

            # Apply Gaussian smoothing
            if (gstd_pix is not None) and (gstd_pix>0):
                kernel = Gaussian2DKernel(x_stddev=gstd_pix)
                if len(imarr_ref.shape)==3:
                    imarr_ref = np.array([image_convolution(im, kernel) for im in imarr_ref])
                else:
                    imarr_ref = image_convolution(imarr_ref, kernel)

            self._imcube_ref = imarr_ref
            self._bpcube_ref = bparr_ref
            self._errcube_ref = imerr_ref


    def gen_roll_ref(self, align_ref=True, gstd_pix=0):
        """Set off roll position as references"""

        if 'aligned' not in self._data_key_sci:
            _log.warning("Images are not aligned and may not produce optimal subtraction results. Will try our best!")

        # For each science image, select the off-roll images as references
        im_ref_list = []
        bp_ref_list = []
        err_ref_list = []
        for i in range(len(self._imcube_sci)):
            pa = self._posangs_sci[i]
            im_cen = self.star_cens[i]

            # Select images with different PA values
            ind_ref = np.abs(self._posangs_sci - pa) > 1
            im_ref_cube = self._imcube_sci[ind_ref]
            bp_ref_cube = self._bpcube_sci[ind_ref]
            err_ref_cube = self._errcube_sci[ind_ref]

            # Shift images to align with science image
            if align_ref:
                ref_cens = self.star_cens[ind_ref]
                for j in range(len(im_ref_cube)):
                    xsh, ysh = im_cen - ref_cens[j]
                    im_ref_cube[j] = image_shift_with_nans(im_ref_cube[j], xsh, ysh, oversample=4, 
                                                           pad=True, gstd_pix=gstd_pix)
                    bp_ref_cube[j] = fshift(bp_ref_cube[j], xsh, ysh, pad=True, cval=np.nan)
                    var_shift = image_shift_with_nans(err_ref_cube[j]**2, xsh, ysh, pad=True,
                                                      method='fshift', interp='linear',
                                                      oversample=4, order=0)
                    err_ref_cube[j] = np.sqrt(var_shift)

            im_ref_list.append(im_ref_cube)
            bp_ref_list.append(bp_ref_cube)
            err_ref_list.append(err_ref_cube)

        self._imcube_ref = im_ref_list
        self._bpcube_ref = bp_ref_list
        self._errcube_ref = err_ref_list


    def gen_synthetic_ref(self, xysize=None, fov_pix=None, diffusion_sigma=None,
                          psf_scale_over=None, exclude_psf_scaling=False,
                          gen_shifted_list=False, shift_offset=0.05, nshift=5,
                          force=None, gstd_pix=0):
        """Generate a synthetic reference image from using pynrc
        
        Parameters
        ----------
        xysize : int
            Size of the PSF in pixels. Defaults to the size of the original science images.
        fov_pix : int
            Number of detector pixels to use in simulations. If not set, uses exisiting values
            in `self.nrc_obs.nrc.fov_pix`.
        diffusion_sigma : float
            Diffusion kernel sigma value. If not set, performs best-fit to science data and
            updates `self.nrc_obs.best_diffusion` parameter.
        psf_scale_over : ndarray
            PSF scaling factor to apply to oversampled PSFs. If not set, uses the value in
            `self.nrc_obs.psf_corr_over`.
        exclude_psf_scaling : bool
            Exclude PSF scaling factor from oversampled PSFs.
        """
        if self._imcube_sci is None:
            raise ValueError("Run gen_aligned_images() first.")

        nrc_obs = self.nrc_obs
        nrc = nrc_obs.nrc
        osamp = nrc.oversample

        # Exclude PSF correction scaling?
        if exclude_psf_scaling:
            psf_scale_over = None
        elif psf_scale_over is None:
            psf_scale_over = nrc_obs.psf_corr_over

        # Find best diffusion if not set explicitly
        if diffusion_sigma is None:
            use_mean = True if 'aligned' in self._data_key_sci else False
            nrc_obs.find_best_diffusion(subsize=11, data_key=self._data_key_sci, 
                                        imall=self.imcube_sci, bpall=self.bpcube_sci,
                                        use_mean=use_mean, psf_corr_over=psf_scale_over)
            diffusion_sigma = nrc_obs.best_diffusion
            print(f"Diffusion self.imcube_sci: {diffusion_sigma:.3f}")

        kipc = nrc_obs.kipc
        kppc = nrc_obs.kppc

        xysize = np.max(self._im_shape_orig) if xysize is None else xysize
        aligned = 'aligned' in self._data_key_sci

        if gen_shifted_list:
            # Oversampled PSFs 
            psfs_over = create_offset_psfs(nrc_obs, xysize=xysize, fov_pix=fov_pix,
                                           diffusion_sigma=diffusion_sigma,
                                           no_offset=aligned, psf_scale_over=psf_scale_over,
                                           return_oversample=True, force=force, gstd_pix=gstd_pix)

            # Shift and rebin PSFs
            psf_list = []
            xy_shifts = np.linspace(-shift_offset*osamp, shift_offset*osamp, nshift)
            # Make sure 0 is included
            if 0 not in xy_shifts:
                xy_shifts = np.concatenate(([0], xy_shifts))
            for i in trange(len(self.imcube_sci), desc='Generating PSF Refs'):
                psf_over_i = psfs_over[i]
                # Scale PSFs to match total flux such that coefficients are closer to unity
                psf_scale = np.nansum(self.imcube_sci[i]) / np.nansum(psf_over_i)
                psf_shifts = []
                for dy in xy_shifts:
                    for dx in xy_shifts:
                        psf_over_sh = fourier_imshift(psf_over_i, dx, dy, pad=True)
                        psf_det_sh = psf_scale * frebin(psf_over_sh, scale=1/osamp, total=True)
                        if kipc is not None:
                            psf_det_sh = add_ipc(psf_det_sh, kernel=kipc)
                        if kppc is not None:
                            psf_det_sh = add_ppc(psf_det_sh, kernel=kppc, nchans=1)
                        psf_shifts.append(psf_det_sh)

                psf_list.append(np.array(psf_shifts))
            self._imcube_ref = psf_list

        else:
            # Detector sampled PSFs 
            psfs_ref = create_offset_psfs(nrc_obs, xysize=xysize, fov_pix=fov_pix,
                                          diffusion_sigma=diffusion_sigma,
                                          kipc=kipc, kppc=kppc, no_offset=aligned,
                                          psf_scale_over=psf_scale_over, force=force, gstd_pix=gstd_pix)
            
            # Scale PSFs to match total flux such that coefficients are closer to unity
            psf_scale = np.nanmedian(np.nansum(self.imcube_sci, axis=(1,2))) / np.nanmedian(np.nansum(psfs_ref, axis=(1,2)))
            
            # Normalize PSF max to 1.0
            # psfs_ref = np.array([psf/psf.max() for psf in psfs_ref])

            # Discard any duplicate reference images
            psfs_ref = np.unique(psfs_ref, axis=0) * psf_scale

            self._imcube_ref = crop_image(psfs_ref, self._im_shape_orig)


    def gen_psf_models(self, xysize=65, diffusion_sigma=None,
                       psf_scale_over=None, exclude_psf_scaling=False,
                       gstd_pix=0):
        """Create PSF models for PSF convolution
        
        Saves oversampled and detector sampled PSFs to self.psfs_over and self.psfs_det.
        """

        nrc_obs = self.nrc_obs

        # Exclude PSF correction scaling?
        if exclude_psf_scaling:
            psf_scale_over = None
        elif psf_scale_over is None:
            psf_scale_over = nrc_obs.psf_corr_over

        # Find best diffusion if not set explicitly
        if diffusion_sigma is None:
            use_mean = True if 'aligned' in self._data_key_sci else False
            nrc_obs.find_best_diffusion(subsize=11, data_key=self._data_key_sci,
                                        imall=self.imcube_sci, bpall=self.bpcube_sci,
                                        use_mean=use_mean, psf_corr_over=psf_scale_over)
            diffusion_sigma = nrc_obs.best_diffusion
            print(f"Diffusion self.imcube_sci: {diffusion_sigma:.3f}")

        # Oversampled PSFs centered in array
        # Diffusion applied, but not IPC or PPC
        psfs_over = create_offset_psfs(nrc_obs, xysize, diffusion_sigma=diffusion_sigma,
                                       return_oversample=True, no_offset=True,
                                       psf_scale_over=psf_scale_over, gstd_pix=gstd_pix)

        # Detector sampled PSFs 
        # Diffusion, IPC, and PPC are all applied (diffusion on oversampled PSF)
        psfs_det = create_offset_psfs(nrc_obs, xysize, diffusion_sigma=diffusion_sigma,
                                      kipc=nrc_obs.kipc, kppc=nrc_obs.kppc, no_offset=True,
                                      psf_scale_over=psf_scale_over, gstd_pix=gstd_pix)
        
        self.psfs_over = psfs_over
        self.psfs_det = psfs_det

    def run_rdi(self, save_products=False, return_res_only=False,
                forward_model=False, collapse_rolls=True, derotate=True,
                prop_err=True, pad_before_derot=False, 
                use_gpu=False, ncores=-2, show_progress=False, **extra_rdi_settings):
        
        from winnie.rdi import rdi_residuals
        from winnie.utils import median_combine
        from copy import copy

        if pad_before_derot:
            _log.error("`pad_before_derot` is not correctly implemented for derotation. Setting to False.")
            pad_before_derot = False

        # Make sure optzones and subzones are set
        if (self._optzones is None) or (self._subzones is None):
            raise ValueError("""
                Optimization and subtraction zones must first be set 
                using update_annular_zones() or set_zones().
                """)

        output_ext = copy(self.output_ext)
        if forward_model:
            if (self.rdi_settings.get('coeffs_in', None) is not None) or \
                    (extra_rdi_settings.get('coeffs_in', None) is not None):
                raise ValueError("""
                    Forward modeling with run_rdi is not valid when using fixed
                    RDI coefficients. For classical RDI, the output from the
                    derotate_and_combine_cssmodel method is likely more
                    appropriate.
                    """)
            if self.imcube_css is None:
                raise ValueError("""
                    Prior to executing "run_rdi" with forward_model=True you
                    must first set a circumstellar model using the
                    set_circumstellar_model method.
                    """)
            imcube_sci = self.imcube_css
            prop_err = False # Never propagate error when forward modeling
            output_ext = output_ext + '_fwdmod'
        else:
            imcube_sci = self.imcube_sci
        
        if not prop_err or return_res_only:
            err_hcube = err_hcube_ref = None
        else:
            if self._errcube_sci is None and self._errcube_ref is None:
                prop_err = False

            if prop_err:
                err_hcube = self.errcube_sci
                err_hcube_ref = self.errcube_ref
            else:
                err_hcube = err_hcube_ref = None

        imcube_css = self.rdi_settings.pop('hcube_css', None)
        if imcube_css is not None:
            imcube_css = imcube_css.squeeze()

        residuals = []
        residuals_err = []
        residuals_der = []
        residuals_err_der = []
        coeffs = []
        cent_der = []
        for i, im_sci in enumerate(imcube_sci):
            # Science data
            hcube = im_sci[np.newaxis, np.newaxis]

            # CSS data
            hcube_css = imcube_css[[i], np.newaxis] if imcube_css is not None else None

            # Science error data
            if err_hcube is not None:
                err_hcube = self.errcube_sci[[i], np.newaxis]

            # Reference data (could be tuple or list)
            if isinstance(self.imcube_ref, (list,tuple)):
                hcube_ref = self.imcube_ref[i][:, np.newaxis]
            else:
                hcube_ref = self.imcube_ref[:, np.newaxis]

            # Reference error data (could be tuple or list)
            if err_hcube_ref is None:
                pass
            elif isinstance(self.errcube_ref, (list,tuple)):
                err_hcube_ref = self.errcube_ref[i][:, np.newaxis]
            else:
                err_hcube_ref = self.errcube_ref[:, np.newaxis]

            # If only one error array is missing, then assume errors are 0.0
            if (err_hcube is not None) and (err_hcube_ref is None):
                err_hcube_ref = np.zeros_like(hcube_ref)
            elif (err_hcube is None) and (err_hcube_ref is not None):
                err_hcube = np.zeros_like(hcube)

            if derotate:
                posangs = self.angles[[i]]
                cent = self.star_cens[i]
            else:
                posangs = None
                cent=None

            res = self._rdi_resids(hcube=hcube,
                                   hcube_ref=hcube_ref,
                                   optzones=self.optzones, subzones=self.subzones,
                                   hcube_css=hcube_css,
                                   posangs=posangs, cent=cent, 
                                   use_gpu=use_gpu, ncores=ncores,
                                   err_hcube=err_hcube,
                                   err_hcube_ref=err_hcube_ref,
                                   pad_before_derot=pad_before_derot,
                                   show_progress=show_progress,
                                   **self.rdi_settings, **extra_rdi_settings)
            
            resid_i, resid_err_i, resid_der_i, resid_err_der_i, coeffs_i, cent_der_i = res
            residuals.append(resid_i)
            residuals_err.append(resid_err_i)
            residuals_der.append(resid_der_i)
            residuals_err_der.append(resid_err_der_i)
            coeffs.append(coeffs_i)
            cent_der.append(cent_der_i)

        residuals = np.array(residuals).squeeze()
        residuals_err = None if residuals_err[0] is None else np.array(residuals_err).squeeze()
        residuals_der = None if residuals_der[0] is None else np.array(residuals_der).squeeze()
        residuals_err_der = None if residuals_err_der[0] is None else np.array(residuals_err_der).squeeze()
        coeffs = np.array(coeffs).squeeze()
        cent_der = np.array(cent_der).squeeze()

        res = self._package_products(residuals, residuals_err=residuals_err, 
                                     residuals_der=residuals_der, 
                                     residuals_err_der=residuals_err_der,
                                     coeffs=coeffs, c_derot=cent_der, 
                                     collapse_rolls=collapse_rolls, 
                                     output_ext=output_ext)
        
        return res

        imcube_ref = self.imcube_ref[:, np.newaxis]
        cent = get_im_cen(imcube_sci[0,0])
        res = rdi_residuals(hcube=imcube_sci,
                            hcube_ref=imcube_ref,
                            optzones=self.optzones, subzones=self.subzones,
                            posangs=posangs, cent=cent,
                            use_gpu=use_gpu, ncores=ncores,
                            err_hcube=err_hcube,
                            err_hcube_ref=err_hcube_ref,
                            pad_before_derot=pad_before_derot,
                            show_progress=show_progress,
                            **self.rdi_settings, **extra_rdi_settings)
        
        if return_res_only:
            return res
        
        residuals, residuals_err, c_derot = res
        
        residuals = residuals[:,0] # dropping unused wavelength axis
        if residuals_err is not None: 
            residuals_err = residuals_err[:,0]
        
        res = self._package_products(residuals, residuals_err=residuals_err, c_derot=c_derot, 
                                     derotate=derotate, collapse_rolls=collapse_rolls, prop_err=prop_err, 
                                     output_ext=output_ext)
        
        return res

        if derotate:
            im_col, err_col = median_combine(residuals, residuals_err)
        else:
            im_col, err_col = None, None
            
        if collapse_rolls:
            # Breatk out the two roll positions
            resid1 = residuals[:self.ndith]
            resid2 = residuals[self.ndith:]

            if residuals_err is not None:
                resid1_err = residuals_err[:self.ndith]
                resid2_err = residuals_err[self.ndith:]
            else:
                resid1_err = resid2_err = None

            im_roll1, err_roll1 = median_combine(resid1, resid1_err)
            im_roll2, err_roll1 = median_combine(resid2, resid2_err)

            im_rolls = np.asarray([im_roll1, im_roll2])
            err_rolls = np.asarray([err_roll1, err_roll1]) if prop_err else None

            # im_rolls, err_rolls = [],[]
            # uni_visit_ids, uni_visit_inds = np.unique(self._visit_ids_sci, return_index=True)
            # uni_visit_ids = uni_visit_ids[np.argsort(uni_visit_inds)]
            # for visit_id in uni_visit_ids:
            #     visit = self._visit_ids_sci == visit_id
            #     im_roll, err_roll = median_combine(residuals[visit], (residuals_err[visit] if prop_err else None))
            #     im_rolls.append(im_roll)
            #     err_rolls.append(err_roll)x
            # im_rolls = np.asarray(im_rolls)
            # err_rolls = np.asarray(err_rolls) if prop_err else None
        else:
            im_rolls = err_rolls = None

        products = {'im': im_col, 'rolls':im_rolls, 'err': err_col, 'err_rolls':err_rolls, 
                    'c_star_out':c_derot, 'output_ext':output_ext, 'derotated':derotate,
                    'residuals':residuals, 'residuals_err':residuals_err}
        
        # Return results dictionary as a class object
        return nrc_rdi_results(products)
    
    def _rdi_resids(self, *args, **extra_rdi_settings):

        from winnie.rdi import rdi_residuals
        from winnie.utils import rotate_hypercube, pad_and_rotate_hypercube, free_gpu
        from copy import copy

        # Get position angles and center locations
        posangs = extra_rdi_settings.pop('posangs', None)
        cent = extra_rdi_settings.pop('cent', None)
        cent_out = None

        # Number of cores and GPU settings
        ncores = extra_rdi_settings.get('ncores', -2)
        use_gpu = extra_rdi_settings.get('use_gpu', False)

        # Pad before derotation?
        pad_before_derot = extra_rdi_settings.get('pad_before_derot', False)

        # Run RDI residuals
        extra_rdi_settings.update(self.rdi_settings)
        res = rdi_residuals(*args, **extra_rdi_settings)
        hcube_res, err_hcube_res, _ = res

        # Run again to return coefficients
        coeffs = rdi_residuals(*args, return_coeffs=True, **extra_rdi_settings)

        # Set up rotation function and keywords
        rot_kwargs = {'cent':cent, 'ncores':ncores, 'use_gpu':use_gpu, 'cval0':np.nan}
        if pad_before_derot:
            rot_func = pad_and_rotate_hypercube
        else:
            rot_func = rotate_hypercube
            # Make sure to reposition all residuals to common center
            new_cent = get_im_cen(np.zeros(hcube_res.shape[-2:]))
            rot_kwargs['new_cent'] = new_cent

        # Derotate residuals
        if posangs is None:
            hcube_res_rot = None 
        else:
            res = rot_func(hcube_res, -posangs, **rot_kwargs)
            # pad_and_rotate_hypercube also returns the new center
            hcube_res_rot, cent_out = res if pad_before_derot else (res, new_cent)

        # Derotate error residuals
        if (err_hcube_res is None) or (posangs is None):
            err_hcube_res_rot = None
        else:
            var_hcube_res = err_hcube_res**2
            res = rot_func(var_hcube_res, -posangs, **rot_kwargs)
            # pad_and_rotate_hypercube also returns the new center
            var_hcube_res_rot = res[0] if pad_before_derot else res
            err_hcube_res_rot = np.sqrt(var_hcube_res_rot)
            del var_hcube_res, var_hcube_res_rot

        if use_gpu:
            free_gpu()

        # residuals, residuals_err, residuals_der, residuals_err_der, coeffs, cent_out
        return hcube_res, err_hcube_res, hcube_res_rot, err_hcube_res_rot, coeffs, cent_out

    def _package_products(self, residuals, residuals_err=None, 
                          residuals_der=None, residuals_err_der=None,
                          coeffs=None, c_derot=None, collapse_rolls=True, 
                          prop_err=None, output_ext=None):

        from winnie.utils import median_combine, rotate_image
        from webbpsf_ext.image_manip import rotate_offset

        output_ext = self.output_ext if output_ext is None else output_ext

        prop_err = False if residuals_err is None else True
        derotate = False if residuals_der is None else True

        if derotate:
            im_rot, err_rot = median_combine(residuals_der, residuals_err_der)
        else:
            im_rot, err_rot = None, None

        products = {
            'im': im_rot, 'err': err_rot, 'rolls': None,  'err_rolls': None, 
            'coeffs': coeffs, 'residuals': residuals, 'residuals_err': residuals_err,
            'residuals_der': residuals_der, 'residuals_err_der': residuals_err_der,
            'c_star_out': c_derot, 'star_cens': self.star_cens,
            'roll_angles': np.unique(self.angles), 'angles': self.angles, 
            'output_ext': output_ext, 'prop_err': prop_err, 'derotated': derotate,
            'pixelscale': self.nrc.pixelscale, 
        }
        
        # Return results dictionary as a class object
        res = nrc_rdi_results(products)

        # Collapse rolls if desired (create res.rolls and res.err_rolls)
        if collapse_rolls:
            res.collapse_rolls()
            res.create_roll_sub()

        return res

    def set_zones(self, optzones, subzones, exclude_opt_nans=True):
        """
        Set the optimization and subtraction zones for RDI PSF-subtraction. See
        winnie.rdi.rdi_residuals for more information on the format and
        function of optzones and subzones. 

        ___________
        Parameters:

        optzones: ndarray
            Optimization zones for the reduction. 
        subzones: ndarray
            Subtraction zones for the reduction.
        exclude_opt_nans: bool
            If True, excludes from the optimization zones any pixels that are
            NaN in either the science or reference data.

        Raises:

            ValueError: If some pixels are included in multiple subtraction zones.

        Notes: 

        - If the spatial dimensions of the zones match those of the uncropped
          data, the zones are directly assigned.
         
        - If the zones are already cropped, the corresponding uncropped zones
          are constructed from them.
        
        """
        if np.any(subzones.sum(axis=0) > 1):
            raise ValueError("Subtraction zones are invalid; some pixels are included in multiple subtraction zones.")
        
        optzones = np.asarray(optzones)
        subzones = np.asarray(subzones)

        # If spatial dims of zones match those of the uncropped data:
        if optzones.shape[-2:] == self._im_shape_orig:
            self._optzones = optzones
            self._subzones = subzones
        else: 
            self._optzones = crop_image(optzones, self._im_shape_orig, fill_val=False)
            self._subzones = crop_image(subzones, self._im_shape_orig, fill_val=False)

        if exclude_opt_nans:
            # Mask out pixels that are NaN in either the science or reference data
            nans_sci = np.logical_or.reduce(np.isnan(self._imcube_sci), axis=0)
            try:
                if isinstance(self._imcube_ref, (list,tuple)):
                    nans_cube_ref = np.logical_or.reduce([np.isnan(imarr) for imarr in self._imcube_ref], axis=0)
                    nans_ref = np.logical_or.reduce(np.isnan(nans_cube_ref), axis=0)
                    del nans_cube_ref
                else:
                    nans_ref = np.logical_or.reduce(np.isnan(self._imcube_ref), axis=0)
            except TypeError:
                _log.warning("No reference data has been defined! NaNs will only be excluded from the science data.")
                nans_ref = np.zeros(self._im_shape_orig, dtype=bool)
            self._optzones[:, nans_ref | nans_sci] = False

    def update_annular_zones(self, exclude_opt_nans=True, r_opt=50, r_sub=None):
        """
        Set annular RDI zones based on current values for self.r_opt and
        self.r_sub.

        Parameters
        ----------
        exclude_opt_nans : bool
            If True, excludes from the optimization zones any pixels that are
            NaN in either the science or reference data.
        r_opt : array, float, None
            The optimization zone radii to pass as 'r_opt' to
            winnie.rdi.build_annular_rdi_zones when loading each concatenation.
            See winnie.rdi.build_annular_rdi_zones doctstring for more info on
            permitted formats. Defaults to 3*u.arcsec (producing a single
            optimization zone spanning 0-3 arcsec from the star).
        r_sub : array, float, None
            The subtraction zone radii to pass as 'r_sub' to
            winnie.rdi.build_annular_rdi_zones when loading each concatenation.
            See winnie.rdi.build_annular_rdi_zones doctstring for more info on
            permitted formats. Defaults to None (producing a single subtraction
            zone spanning the field of view).

        """
        from winnie.rdi import build_annular_rdi_zones

        ny, nx = self._im_shape_orig
        xcen, ycen = get_im_cen(np.zeros([ny,nx]))
        pixelscale = self.nrc.pixelscale

        optzones, subzones = build_annular_rdi_zones(nx, ny, (xcen,ycen), r_opt=r_opt, r_sub=r_sub, pxscale=pixelscale)
        self.set_zones(optzones, subzones, exclude_opt_nans=exclude_opt_nans)


    def report_current_config(self, show_plots=False):
        """
        Print a summary of the current configuration of the SpaceRDI instance.
        If show_plots is True, also plots the current optimization and
        subtraction zones for the first science exposure.
        """
        from winnie.plot import quick_implot, mpl_centered_extent
        import matplotlib as mpl


        try:
            nz, ny, nx = self.imcube_sci.shape
            print(f'Science data:   {nz} exposures of shape ({ny},{nx})')
        except:
            print('No science data loaded.')

        try:
            if isinstance(self.imcube_ref, (list)):
                for i,im in enumerate(self.imcube_ref):
                    nz, ny, nx = im.shape
                    print(f'Reference data {i}: {nz} exposures of shape ({ny},{nx})')
                print('')
            else:
                nz, ny, nx = self.imcube_ref.shape
                print(f'Reference data: {nz} exposures of shape ({ny},{nx})\n')
        except:
            print('No reference data loaded.\n')

        print(f'RDI Settings:')
        for key in self.rdi_settings:
            if isinstance(self.rdi_settings[key], np.ndarray):
                desc = f'{type(self.rdi_settings[key])} of shape {self.rdi_settings[key].shape}'
            else:
                desc = self.rdi_settings[key]
            print(f"'{key}': {desc}")
        print(f"Extension for output files: '{self.output_ext}'")
        try:
            print(f"{len(self.optzones)} optimization zone(s)")
            print(f"{len(self.subzones)} subtraction zone(s)")
        except:
            print("No optimization or subtraction zones set.")
            return
        
        if show_plots:
            ny, nx = self.imcube_sci[0].shape
            xcen, ycen = get_im_cen(np.zeros([ny,nx]))
            pixelscale = self.nrc.pixelscale

            fig,axes = quick_implot(np.array([np.where(self.optzones, self.imcube_sci[0], np.nan),
                                              np.where(self.subzones, self.imcube_sci[0], np.nan)]
                                             ).transpose((1,0,2,3)),
                                    norm=mpl.colors.LogNorm,
                                    norm_kwargs=dict(clip=True),
                                    clim='0.001*99.99%, 99.99%',
                                    extent=mpl_centered_extent((ny, nx), (xcen, ycen), pixelscale),
                                    show_ticks=True,
                                    show=False, panelsize=(4,4))
            axes[0].set_title('Optimization Zones')
            axes[1].set_title('Subtraction Zones')
            for i,ax in enumerate(axes):
                for axis in [ax.xaxis, ax.yaxis]:
                    axis.set_major_formatter("${x:0.0f}''$")
            fig.tight_layout()

    def set_fixed_rdi_settings(self, **settings):
        """
        Set RDI settings that will be added to (and overwrite where duplicated)
        any settings managed by set_presets, rdi_presets, hpfrdi_presets, or
        mcrdi_presets. These must be re-set if a new concatenation is loaded.

        Some settings that may be useful:
        
        ref_mask: ndarray
            2D boolean array of shape (len(optzones), len(self.imcube_ref))
            that indicates which reference images should be considered for
            which optimization regions. E.g., if ref_mask[i,j] is False, then
            for the ith optimization zone (optzones[i]), the jth reference
            image (imcube_ref[j]) will NOT be used for construction of the PSF
            model. This can be useful if some reference exposures have
            anomalous features that make them problematic for some regions
            while still being suitable for others; e.g., an image with a bright
            background source near the edge of the FOV may still be useful for
            nulling the PSF near the inner working angle.

        zero_nans: bool
            If True, any nans in the optimization zones will be replaced with
            zeros for the procedure.

        return_coeffs: bool
            If True, returns only the array of PSF model coefficients.

        coeffs_in: ndarray
            If provided, these coefficients will be used to construct the PSF
            model instead of computing coefficients.

        opt_smoothing_fn: callable or None
            If not None, this argument indicates the function with which to
            smooth the sequences. This should be a function that takes a
            hypercube along with some keyword arguments and returns a smoothed
            hypercube, i.e.: hcube_filt = opt_smoothing_fn(hcube,
            **opt_smoothing_kwargs).
            
        opt_smoothing_kwargs: dict
            If opt_smoothing_fn is not None, arguments to pass to
            opt_smoothing_fn when it is called.

        large_arrs: bool
            If True (and if use_gpu=False), PSF model reconstruction will use
            smaller parallelized calculations. If False, larger vectorized
            calculations will be used instead. See the docstring for
            winnie.rdi.reconstruct_psf_model_cpu for more information. Default
            is False.
        """
        self.fixed_rdi_settings = settings
        self.rdi_settings.update(self.fixed_rdi_settings)

    def set_presets(self, presets={}, output_ext='psfsub', verbose=False):
        """
        Generic method to quickly assign a set of arguments to use for
        winnie.rdi.rdi_residuals, while also setting the extension for saved
        files, repopulating any settings in self.fixed_rdi_settings, and
        reporting the configuration if verbose is True.
        """
        self.output_ext = output_ext
        self.rdi_settings = presets
        self.rdi_settings.update(self.fixed_rdi_settings)
        if verbose:
            self.report_current_config()

    def rdi_presets(self, output_ext='rdi_psfsub', verbose=False):
        """
        Set presets to perform a standard RDI reduction.

        ___________
        Parameters:
            output_ext (str, optional): Output file extension for FITS
                products. Defaults to 'rdi_psfsub'.
        """
        self.set_presets(presets={}, output_ext=output_ext, verbose=verbose)

    def hpfrdi_presets(self, filter_size=None, filter_size_adj=1, 
                       output_ext='hpfrdi_psfsub', verbose=False):
        """
        Set presets for High-Pass Filtering RDI (HPFRDI), in which coefficients
        are computed by comparing high-pass filtered science and reference
        data.

        ___________
        Parameters:
            filter_size (float, optional): Size of the high-pass filter. If not
                provided, it defaults to the value of self._sigma.
            filter_size_adj (float, optional): Adjustment factor for the filter
                size. Defaults to 1. output_ext (str, optional): Output file
                extension for FITS products. Defaults to 'hpfrdi_psfsub'.

        Notes:
            - This method checks if there will be any NaN values in the
              optimization zones after applying the specified filtering.
            - If so, it also sets 'zero_nans' to True to avoid a crash when
              run_rdi is called.
        """
        from winnie.utils import high_pass_filter_sequence

        if filter_size is None:
            filter_size = self.gstd_pix
        presets = {}
        presets['opt_smoothing_fn'] = high_pass_filter_sequence
        presets['opt_smoothing_kwargs'] = dict(filtersize=filter_size_adj*filter_size)

        # See if there's any NaNs in our optzones after filtering is applied. 
        # If so, add zero_nans=True to our settings to avoid a crash.
        sci_filt = high_pass_filter_sequence(self.imcube_sci, filter_size)
        if isinstance(self.imcube_ref, (list,tuple)):
            ref_filt = np.concatenate([high_pass_filter_sequence(im, 2) for im in self.imcube_ref], axis=0)
        else:
            ref_filt = high_pass_filter_sequence(self.imcube_ref, filter_size)

        allopt = np.any(self.optzones, axis=0)
        nans = np.any([*np.isnan(sci_filt[..., allopt]), *np.isnan(ref_filt[..., allopt])])
        if nans:
            presets['zero_nans'] = True
        self.set_presets(presets=presets, output_ext=output_ext, verbose=verbose)

        del sci_filt, ref_filt

    
    def mcrdi_presets(self, output_ext='mcrdi_psfsub', verbose=False):
        """
        Set presets for Model Constrained RDI (MCRDI), in which coefficients
        are computed by comparing reference data to science data from which an
        estimate of the circumstellar scene has been subtracted.

        ___________
        Parameters:
            output_ext (str, optional): Output file extension for FITS
                products. Defaults to 'mcrdi_psfsub'.

        Raises:
            ValueError: If a circumstellar model has not been set using the
                set_circumstellar_model method.
        """
        if self.imcube_css is None:
            raise ValueError(
                """
                Prior to executing mcrdi_presets,
                you must first set a circumstellar model using 
                set_circumstellar_model.
                """)

        self.set_presets(presets={'hcube_css': self.imcube_css[:, np.newaxis]},
                         output_ext=output_ext, verbose=verbose)

    def set_circumstellar_model(self, model_cube=None, raw_model=None, raw_model_osamp=None,
                                ncores=-2, use_gpu=False):
        """
        Sets a circumstellar scene model to be used in various procedures
        (e.g., RDI forward modeling or MCRDI.) 
        """

        from vip_hci.preproc import cube_derotate
        from webbpsf_ext.image_manip import rotate_offset
        from winnie.utils import rotate_hypercube

        if raw_model is not None:
            # Derotate model to match observed orientation
            nframes = len(self.angles)

            # Resample model to match oversampled PSFs
            if raw_model_osamp is None:
                raise ValueError("Must provide `raw_model_osamp` if raw_model is provided.")

            osamp = self.nrc.oversample
            resampling = osamp / raw_model_osamp
            if resampling != 1:
                raw_model = zrebin(raw_model, resampling, total=False)

            # Generate multiple frames for each dither position and roll angle
            model_cube = np.repeat(raw_model[np.newaxis,:,:], nframes, axis=0)
            # model_cube = rotate_hypercube(model_cube, self.angles, cent=None, new_cent=None, 
            #                               ncores=ncores, use_gpu=use_gpu, cval0=0., prefilter=True)


            # Rotate and convolve with oversampled PSF
            for i, im in enumerate(model_cube):
                # Derotate by correct angles
                im_rot = rotate_offset(im, self.angles[i], order=3, reshape=False)
                # Offset to align with science data
                xy_off_over = self.offsets_sci[i] * 4
                if np.abs(xy_off_over).max() > 0:
                    im_rot_offset = image_shift_with_nans(im_rot, xy_off_over[0], xy_off_over[1], pad=True, cval=0)
                    im_rot_offset[im_rot_offset<0] = 0
                else:
                    im_rot_offset = im_rot

                # hcube = im[np.newaxis,np.newaxis,:,:]
                # angles = self.angles[[i]]
                # cent = get_im_cen(im)
                # new_cent = cent + xy_off_over
                # im_rot_offset = rotate_hypercube(hcube, angles, cent=cent, new_cent=new_cent, 
                #                                  ncores=-1, use_gpu=False, cval0=0., prefilter=True)

                model_cube[i] = image_convolution(im_rot_offset, self.psfs_over[i], method='scipy')

            # Rebin to detector sampling
            model_cube = frebin(model_cube, scale=1/osamp, total=False)

            # Throw a warning if model cube and science data have different even/odd shapes
            modely_is_even = model_cube.shape[-2] % 2 == 0
            modelx_is_even = model_cube.shape[-1] % 2 == 0
            sciy_is_even = self.imcube_sci.shape[-2] % 2 == 0
            scix_is_even = self.imcube_sci.shape[-1] % 2 == 0
            if modely_is_even != sciy_is_even or modelx_is_even != scix_is_even:
                _log.warning("Model and science data have different even/odd shapes. "
                             "This may cause issues with centering.")


        self._imcube_css = crop_image(model_cube, self._imcube_sci.shape[-2:], fill_val=0)


    def circumstellar_model_rescale(self, return_scale=False, sig=None, mask=None, 
                                    hpf=False, filter_size=None, filter_size_adj=1,
                                    image_rdi=None):
        """
        Rescale the circumstellar model to match the median flux of the science
        data. This is useful for forward modeling in RDI, where the model
        should be scaled to match the observed flux level.
        """
        from winnie.utils import median_filter_sequence
        from copy import copy, deepcopy

        if self.imcube_css is None:
            raise ValueError(
                """
                Prior to executing circumstellar_model_rescale, you must first
                set a circumstellar model using set_circumstellar_model.
                """)

        # Save current presets
        output_ext_prev = deepcopy(self.output_ext)
        presets_prev = copy(self.rdi_settings)

        if hpf:
            self.hpfrdi_presets(verbose=False, filter_size=filter_size, filter_size_adj=filter_size_adj)
        else:
            self.rdi_presets(verbose=False)

        # Run RDI to get residual image
        if image_rdi is None:
            rdi_res = self.run_rdi(collapse_rolls=False, prop_err=True)
            image_rdi = rdi_res.im
        else:
            rdi_res = None

        if (sig is None) and (rdi_res is not None):
            sig = rdi_res.err

        # Foward model circumstellar disk model
        fmrdi_res = self.run_rdi(forward_model=True, collapse_rolls=False)
        image_rdi = crop_image(image_rdi, fmrdi_res.im.shape)

        footprint = np.array([[0,1,0], [1,1,1], [0,1,0]])
        args = median_filter_sequence(np.array([image_rdi, fmrdi_res.im]), 
                                      footprint=footprint, 
                                      prop_threshold=0.8)
        sfac = model_rescale_factor(*args, sig=sig, mask=mask)
        self._imcube_css *= sfac

        self.set_presets(presets=presets_prev, output_ext=output_ext_prev, verbose=False)

        if return_scale:
            return sfac


    def derotate_and_combine_circumstellar_model(self, 
                                                 pad_before_derot=False,
                                                 collapse_rolls=True,
                                                 output_ext='cssmodel', 
                                                 save_products=False,
                                                 ncores=-2, use_gpu=False):
        """
        Derotates the current circumstellar model and averages over all rolls;
        provides output in a SpaceReduction object to match the output of
        run_rdi. If a circumstellar model is not set, this method will raise a
        ValueError.
        """
        if self.imcube_css is None:
            raise ValueError(
                """
                Prior to executing derotate_and_combine_circumstellar_model, you must
                first set a circumstellar model using `set_circumstellar_model()`.
                """
            )
        
        from winnie.utils import pad_and_rotate_hypercube, rotate_hypercube

        if pad_before_derot:
            _log.error("`pad_before_derot` is not yet correctly implemented for derotation. Setting to False.")
            pad_before_derot = False
        
        csscube = self.imcube_css
        cent = self.star_cens

        # Set up rotation function
        rot_func = pad_and_rotate_hypercube if pad_before_derot else rotate_hypercube
        rot_kwargs = {'ncores':ncores, 'use_gpu':use_gpu, 'cval0':np.nan}
        if not pad_before_derot:
            # Make sure to reposition all residuals to common center
            new_cent = get_im_cen(np.zeros(csscube.shape[-2:]))
            rot_kwargs['new_cent'] = new_cent


        residuals = []
        residuals_der = []
        c_derot = []
        for i, im in enumerate(csscube):
            rot_kwargs['cent'] = cent[i]
            res = rot_func(im[np.newaxis,np.newaxis], -self.angles[[i]], **rot_kwargs)
            resid_der, cnew = res if pad_before_derot else (res, new_cent)

            residuals.append(im.squeeze())
            residuals_der.append(resid_der.squeeze())
            c_derot.append(cnew)

        residuals = np.array(residuals)
        residuals_der = np.array(residuals_der)
        c_derot = np.array(c_derot)
        
        res = self._package_products(residuals, 
                                     residuals_der=residuals_der, 
                                     c_derot=c_derot, 
                                     collapse_rolls=collapse_rolls, 
                                     output_ext=output_ext)

        return res

        if save_products:
            try:
                products.save(overwrite=self.overwrite)
            
            except OSError:
                raise OSError("""
                      A FITS file for this output_ext + output_dir + concat
                      already exists! To overwrite existing files, set the
                      overwrite attribute for your Winnie SpaceRDI instance to
                      True. Alternatively, either change the output_ext
                      attribute for your SpaceRDI instance, or select a
                      different output directory when initializing your
                      SpaceKLIP database object.
                      """)


# Convert dictionary keys to a class attribute
class nrc_rdi_results:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

    def collapse_rolls(self, gstd_pix=None):
        """Collapse residuals into their roll positions and derotate if desired."""

        from webbpsf_ext.image_manip import rotate_offset, image_shift_with_nans, get_im_cen
        from winnie.utils import median_combine

        residuals = self.residuals
        c_derot = self.c_star_out

        derotate = self.derotated
        prop_err = self.prop_err

        # Align residuals to common center
        # residuals_aligned = []
        # residuals_err_aligned = [] if prop_err else None
        # for i, im_resid in enumerate(residuals):
        #     cent = self.star_cens[i]
        #     new_cent = get_im_cen(im_resid) if c_derot is None else c_derot[i]
        #     im_aligned = rotate_image(im_resid, 0, cent=cent, new_cent=new_cent, cval0=0)
        #     residuals_aligned.append(im_aligned)
        # residuals_aligned = np.array(residuals_aligned)

        # if prop_err:
        #     residuals_err_aligned = []
        #     for i, im_err in enumerate(residuals_err):
        #         cent = self.star_cens[i]
        #         new_cent = get_im_cen(im_err) if c_derot is None else c_derot[i]
        #         var_aligned = rotate_image(im_err**2, 0, cent=cent, new_cent=new_cent, cval0=np.nan)
        #         var_aligned[np.isnan(var_aligned)] = np.nanmax(var_aligned)
        #         residuals_err_aligned.append(np.sqrt(var_aligned))
        #     residuals_err_aligned = np.array(residuals_err_aligned)

        # Align residuals to common center
        residuals_aligned = []
        residuals_err_aligned = [] if prop_err else None
        for i, im_resid in enumerate(residuals):
            cent = self.star_cens[i]
            new_cent = get_im_cen(im_resid) if c_derot is None else c_derot[i]
            xsh, ysh = new_cent - cent
            im_aligned = image_shift_with_nans(im_resid, xsh, ysh, oversample=4, order=3,
                                                shift_method='fourier', interp='lanczos',
                                                gstd_pix=gstd_pix)
            residuals_aligned.append(im_aligned)

            if prop_err:
                im_err = self.residuals_err[i]
                var_aligned = image_shift_with_nans(im_err**2, xsh, ysh, oversample=4, order=0,
                                                    shift_method='fshift', interp='linear',
                                                    gstd_pix=gstd_pix, preserve_nans=True)
                var_aligned[np.isnan(var_aligned)] = np.nanmax(var_aligned)
                residuals_err_aligned.append(np.sqrt(var_aligned))

        residuals_aligned = np.array(residuals_aligned)
        residuals_err_aligned = np.array(residuals_err_aligned) if prop_err else None


        # Break out the unique roll positions
        uangles = np.unique(self.angles)
        im_rolls = []
        err_rolls = []
        for angle in uangles:
            ind = (self.angles == angle)
            resid = residuals_aligned[ind]
            resid_err = residuals_err_aligned[ind] if prop_err else None
            im_roll, err_roll = median_combine(resid, resid_err)
            im_rolls.append(im_roll)
            err_rolls.append(err_roll)

        self.rolls = np.asarray(im_rolls)
        self.err_rolls = np.asarray(err_rolls) if prop_err else None

        # Derotate rolls
        if derotate:
            self.rolls_der = np.array([rotate_offset(im, -uangles[i], reshape=False, cval=np.nan)
                                       for i, im in enumerate(self.rolls)])
            if prop_err:
                var_rolls_der = np.array([rotate_offset(err**2, -uangles[i], reshape=False, cval=np.nan)
                                          for i, err in enumerate(self.err_rolls)])
                self.err_rolls_der = np.sqrt(var_rolls_der)
            else:
                self.err_rolls_der = None

    def create_roll_sub(self, gstd_pix=None):
        """
        Create a subtraction image from the two roll positions.
        """
        from webbpsf_ext.image_manip import rotate_offset

        if self.rolls is None:
            raise ValueError("Rolls need to be collapsed first using self.collapse_rolls()")
        if self.derotated==False:
            _log.warning('No prior derotation of images has been performed on data products.')

        im1, im2 = self.rolls
        pa1, pa2 = self.roll_angles

        diff1 = im1 - im2
        diff2 = -1 * diff1

        if (gstd_pix is not None) and (gstd_pix > 0):
            diff1 = image_shift_with_nans(diff1, 0, 0, gstd_pix=gstd_pix, oversample=4, order=3, preserve_nans=True)
            diff2 = image_shift_with_nans(diff2, 0, 0, gstd_pix=gstd_pix, oversample=4, order=3, preserve_nans=True)
        
        diff1_der = rotate_offset(diff1, -pa1, reshape=False, cval=np.nan)
        diff2_der = rotate_offset(diff2, -pa2, reshape=False, cval=np.nan)
        diff_der = np.nanmean([diff1_der, diff2_der], axis=0)
        diff_der_abs = np.nanmean([np.abs(diff1_der), np.abs(diff2_der)], axis=0)

        self.roll_sub = diff_der
        self.roll_sub_abs = diff_der_abs

        if self.prop_err:
            err1, err2 = self.err_rolls
            err_diff = np.sqrt(err1**2 + err2**2)
            var_diff = err_diff**2
            if (gstd_pix is not None) and (gstd_pix > 0):
                var_diff = image_shift_with_nans(var_diff, 0, 0, gstd_pix=gstd_pix, oversample=4, order=1, preserve_nans=True)

            var_diff1_der = rotate_offset(var_diff, -pa1, reshape=False, cval=np.nan)
            var_diff2_der = rotate_offset(var_diff, -pa2, reshape=False, cval=np.nan)
            self.err_roll_sub = np.sqrt(np.nanmean([var_diff1_der, var_diff2_der], axis=0))
        else:
            self.err_roll_sub = None

    def plot_overview(self, gstd_pix=None, vmin_list=None, vmax_list=None, 
                      log=False, labels=None, pl_rth_asec=None, return_fig=False):

        from webbpsf_ext.coords import rtheta_to_xy
        from hciplot import plot_frames

        im_der, diff_der_abs, diff_der = (self.im, self.roll_sub_abs, self.roll_sub)
        if (gstd_pix is not None) and (gstd_pix > 0):
            im_der = image_shift_with_nans(im_der, 0, 0, gstd_pix=gstd_pix, oversample=4, order=3, preserve_nans=True)
            _log.warning(f'Updating self.roll_sub, self.roll_sub_abs, and self.err_roll_sub with gstd_pix={gstd_pix}')
            self.create_roll_sub(gstd_pix=gstd_pix)
            diff_der_abs, diff_der = (self.roll_sub_abs, self.roll_sub)
            # diff_der_abs = image_shift_with_nans(diff_der_abs, 0, 0, gstd_pix=gstd_pix, oversample=4, order=3, preserve_nans=True)
            # diff_der = image_shift_with_nans(diff_der, 0, 0, gstd_pix=gstd_pix, oversample=4, order=3, preserve_nans=True)

        xcen, ycen = get_im_cen(im_der)
        pixelscale = self.pixelscale

        if pl_rth_asec is None:
            pl_locs = None
        else:
            pl_locs = []
            for r, th in pl_rth_asec:
                x, y = np.array(rtheta_to_xy(r, th)) / pixelscale
                x_fin, y_fin = np.array([x, y]) + np.array([xcen, ycen])
                pl_locs.append((x_fin, y_fin))
            pl_locs = tuple(pl_locs)

        vmin_list = list([-50, -10, -10]) if vmin_list is None else list(vmin_list)
        vmax_list = list([np.nanmax(im_der), np.nanmax(diff_der_abs)/3, 40]) if vmax_list is None else list(vmax_list)


        labels = ('RDI Residuals', 'Roll Sub (Abs Diff)', 'Roll Sub (Diff)') if labels is None else tuple(labels)
        fig, axes = plot_frames([im_der, diff_der_abs, diff_der], log=log,
                                rows=1, cmap=('inferno', 'inferno', 'inferno'), 
                                vmin=vmin_list, vmax=vmax_list, size_factor=4,
                                show_center=True, cross_color='grey', cross_alpha=1,
                                label=labels, label_color='C2', label_size=10,
                                circle=pl_locs, circle_radius=1.7, circle_color='C2', circle_alpha=1,
                                horsp=0.25, versp=0, ang_scale=True, ang_ticksep=8,
                                pxscale=pixelscale, return_fig_ax=True)
        
        fig.tight_layout()

        if return_fig:
            return fig


def model_rescale_factor(A, B, sig=None, mask=None):
    """
    Determines the value of scalar c such that:
        chi^2 = sum [ (A-c*B)^2 / sig^2 ]
    is minimized.
    
    Parameters
    ----------
    A : numpy.ndarray
        Array of measurements
    B : numpy.ndarray
        Array of model values. Shape must match A and B
    sig : numpy.ndarray, optional
        The 1 sigma uncertainty for the measurements of A.
    mask : numpy.ndarray, optional
        A boolean mask with False for entries of A, B, and sig not to be
        utilized, and True for entries that are. Defaults to None.
    Returns
    -------
    c : float
        The scaling factor to multiply the model (B) by to achieve the minimum chi^2
        for measurements (A) having the given uncertainties (sig).
    """
    if np.shape(A) != np.shape(B):
        raise ValueError("A and B must be arrays of the same shape!")
    if sig is not None:
        if np.shape(A) != np.shape(sig):
            raise ValueError("A, B, and sig must be arrays of the same shape if sig is specified!")
    else:
        sig = 1
    if mask is None:
        c = np.nansum(A * B / (sig ** 2)) / np.nansum((B ** 2) / (sig ** 2))
    elif np.shape(mask)[-2:] != np.shape(A)[-2:]:
        raise ValueError("If provided, mask's shape must match the final axes of A, B, and sig!")
    else:
        Amsk, Bmsk = A[..., mask], B[..., mask]
        if np.ndim(sig) != 0:
            Smsk = sig[..., mask]
        else:
            Smsk = sig
        c = np.nansum(Amsk * Bmsk / (Smsk ** 2)) / np.nansum((Bmsk ** 2) / (Smsk ** 2))
    return c
