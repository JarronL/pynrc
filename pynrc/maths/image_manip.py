import numpy as np
import logging
_log = logging.getLogger('pynrc')

from poppy.utils import krebin

from .coords import dist_image
from scipy.ndimage import fourier_shift, rotate
from astropy.io import fits

from webbpsf_ext.image_manip import pad_or_cut_to_size
from webbpsf_ext.image_manip import fshift, fourier_imshift, cv_shift
from webbpsf_ext.image_manip import frebin, zrebin
from webbpsf_ext.image_manip import fractional_image_shift, image_shift_with_nans, replace_nans
from webbpsf_ext.image_manip import rotate_offset, rotate_shift_image
from webbpsf_ext.image_manip import image_rescale, model_to_hdulist
from webbpsf_ext.image_manip import convolve_image, crop_zero_rows_cols
from webbpsf_ext.image_manip import get_im_cen
from webbpsf_ext.image_manip import add_ipc, add_ppc, apply_pixel_diffusion
from webbpsf_ext.image_manip import image_convolution
from webbpsf_ext.imreg_tools import subtract_psf
from webbpsf_ext.maths import hist_indices, binned_statistic, fit_bootstrap

from webbpsf_ext.image_manip import crop_observation as _crop_observation

def crop_observation(im_full, ap, xysub, xyloc=None, delx=0, dely=0, 
                     shift_func=fourier_imshift, interp='cubic',
                     return_xy=False, fill_val=0, **kwargs):
    
    """Crop around aperture reference location

    `xysub` specifies the desired crop size.
    if `xysub` is an array, dimension order should be [nysub,nxsub]

    `xyloc` provides a way to manually supply the central position. 
    Set `ap` to None will crop around `xyloc` or center of array.

    delx and delx will shift array by some offset before cropping
    to allow for sub-pixel shifting. To change integer crop positions,
    recommend using `xyloc` instead.

    Shift function can be fourier_imshfit, fshift, or cv_shift.
    The interp keyword only works for the latter two options.
    Consider 'lanczos' for cv_shift.

    Setting `return_xy` to True will also return the indices 
    used to perform the crop.

    Parameters
    ----------
    im_full : ndarray
        Input image.
    ap : pysiaf aperture
        Aperture to use for cropping. Will crop around the aperture
        reference point by default. Will be overridden by `xyloc`.
    xysub : int, tuple, or list
        Size of subarray to extract. If a single integer is provided,
        then a square subarray is extracted. If a tuple or list is
        provided, then it should be of the form (ny, nx).
    xyloc : tuple or list
        (x,y) pixel location around which to crop the image. If None,
        then the image aperture refernece point is used.
    
    Keyword Args
    ------------
    delx : int or float
        Integer pixel offset in x-direction. This shifts the image by
        some number of pixels in the x-direction. Positive values shift
        the image to the right.
    dely : int or float
        Integer pixel offset in y-direction. This shifts the image by
        some number of pixels in the y-direction. Positive values shift
        the image up.
    shift_func : function
        Function to use for shifting. Default is `fourier_imshift`.
        If delx and dely are both integers, then `fshift` is used.
    interp : str
        Interpolation method to use for shifting. Default is 'cubic'.
        Options are 'nearest', 'linear', 'cubic', and 'quadratic'
        for `fshift`.
    return_xy : bool
        If True, then return the x and y indices used to crop the
        image prior to any shifting from `delx` and `dely`. 
        Default is False.
    fill_val : float
        Value to use for filling in the empty pixels after shifting.
        Default = 0.
    """
    return _crop_observation(im_full, ap, xysub, xyloc=xyloc, delx=delx, dely=dely,
                              shift_func=shift_func, interp=interp,
                              return_xy=return_xy, fill_val=fill_val, **kwargs)


def crop_image(imarr, xysub, xyloc=None, **kwargs):
    """Crop input image around center using integer offsets only

    If size is exceeded, then the image is expanded and filled with 0s by default.

    Parameters
    ----------
    imarr : ndarray
        Input image or image cube [nz,ny,nx].
    xysub : int, tuple, or list
        Size of subarray to extract. If a single integer is provided,
        then a square subarray is extracted. If a tuple or list is
        provided, then it should be of the form (ny, nx).
    xyloc : tuple or list
        (x,y) pixel location around which to crop the image. If None,
        then the image center is used.
    
    Keyword Args
    ------------
    delx : int or float
        Integer pixel offset in x-direction. This shifts the image by
        some number of pixels in the x-direction. Positive values shift
        the image to the right.
    dely : int or float
        Integer pixel offset in y-direction. This shifts the image by
        some number of pixels in the y-direction. Positive values shift
        the image up.
    shift_func : function
        Function to use for shifting. Default is `fourier_imshift`.
        If delx and dely are both integers, then `fshift` is used.
    interp : str
        Interpolation method to use for shifting. Default is 'cubic'.
        Options are 'nearest', 'linear', 'cubic', and 'quadratic'
        for `fshift`.
    return_xy : bool
        If True, then return the x and y indices used to crop the
        image prior to any shifting from `delx` and `dely`; 
        (x1, x2, y1, y2). Default is False.
    fill_val : float
        Value to use for filling in the empty pixels after shifting.
        Default = 0.
    """

    sh = imarr.shape
    if len(sh) == 2:
        return crop_observation(imarr, None, xysub, xyloc=xyloc, **kwargs)
    elif len(sh) == 3:
        return_xy = kwargs.pop('return_xy', False)
        res = np.asarray([crop_observation(im, None, xysub, xyloc=xyloc, **kwargs) for im in imarr])
        if return_xy:
            _, xy = crop_observation(imarr[0], None, xysub, xyloc=xyloc, return_xy=True, **kwargs)
            return (res, xy)
        else:
            return res 
    else:
        raise ValueError(f'Found {len(sh)} dimensions {sh}. Only 2 or 3 dimensions allowed.')


def shift_subtract(params, reference, target, mask=None, pad=False, interp='cubic',
                   shift_function=fshift):
    """Shift and subtract image
    
    Subpixel shifts for input into least-square optimizer.
    
    Parameters
    ----------
    params : tuple
        xshift, yshift, beta
    reference : ndarray
        See align_fourierLSQ
    target : ndarray
        See align_fourierLSQ
    mask : ndarray, optional
        See align_fourierLSQ
    pad : bool
        Should we pad the array before shifting, then truncate?
        Otherwise, the image is wrapped.
    interp : str
        Interpolation for fshift function. Default is 'cubic'.
        Options are 'linear', 'cubic', or 'quintic'.
    shift_function : func
        which function to use for sub-pixel shifting
            
    Returns
    -------
    ndarray
        1D array of target-reference residual after
        applying shift and intensity fraction.
    """
    xshift, yshift, beta = params

    if shift_function is not None:
        offset = shift_function(reference, xshift, yshift, pad=pad, interp=interp)
    else:
        offset = reference
    
    if mask is not None:
        return ( (target - beta * offset) * mask ).ravel() #.flatten()
    else:
        return ( target - beta * offset ).ravel() #.flatten()


def align_leastsq(image, psf_over, osamp=1, bpmask=None, weights=None,
                  params0=[0.0,0.0,1.0,0.0], kipc=None,
                  func_shift=fourier_imshift, interp='cubic', pad=True, **kwargs):
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
        Initial guess for (x, y, offset) values. Optimal scaling 
        factor is calculated automatically in `subtract_psf`.
    func_shift : func
        Which function to use for sub-pixel shifting.
        Options are fourier_imshift, fshift, or cv_shift.
    interp : str
        Interpolation for fshift or cv_shift functions. 
        Options are 'linear', 'cubic', or 'quintic'.
        Default is 'cubic'. 
    pad : bool
        Should we pad the array before shifting, then truncate?
        Otherwise, the image is wrapped.

    Returns
    -------
    list
        (x, y, scale, offset) values from LSQ optimization, where (x, y) 
        are the misalignment of target from reference and scale
        is the fraction by which the target intensity must be
        reduced to match the intensity of the reference. Offset gives
        the difference in the mean intensity of the two images.
    """
    from scipy.optimize import least_squares#, leastsq

    def psf_diff(params, image, psf, **kwargs):
        """PSF differencing helper"""

        kwargs['xyshift'] = params[0:2]

        if len(params)==1:
            kwargs['xyshift'] = (0,0)
            psf_offset = 0.0
        elif len(params)==2:
            psf_offset = 0.0
        elif len(params)==3:
            psf_offset = params[-1]
        else:
            raise ValueError("params must be length 2, 3, or 4")

        kwargs['psf_offset'] = psf_offset

        return subtract_psf(image, psf, **kwargs).ravel()

    # Set weights image to pass to differencing function
    if bpmask is not None:
        weights = np.ones_like(image) if weights is None else weights
        weights[bpmask] = 0
    
    # Keywords to pass 
    kwargs2 = {
        'weights'    : weights,
        'osamp'      : osamp,
        'func_shift' : func_shift,
        'interp'     : interp,
        'pad'        : pad,
        'kipc'       : kipc,
    }
    kwargs_pass = kwargs.copy()
    kwargs_pass.update(kwargs2)
    
    # Use loss='soft_l1' for least squares robust against outliers
    # May want to play around with f_scale...
    res = least_squares(psf_diff, params0, #diff_step=0.1, loss='soft_l1', f_scale=1.0, 
                        args=(image, psf_over), kwargs=kwargs_pass, **kwargs)
    out = res.x

    if len(out)==1:
        return out[0]
    else:
        return out


def align_LSQ(reference, target, mask=None, pad=False, interp='cubic',
              shift_function=fshift, init_pars=[0.0, 0.0, 1.0]):
    """Find best shift value
    
    LSQ optimization with option of shift alignment algorithm.
    In practice, the 'reference' image gets shifted to match
    the 'target' image.
    
    Parameters
    ----------
    reference : ndarray
        N x K image to be aligned to
    target : ndarray
        N x K image to align to reference
    mask : ndarray, optional
        N x K image indicating pixels to ignore when
        performing the minimization. The masks acts as
        a weighting function in performing the fit.
    pad : bool
        Should we pad the array before shifting, then truncate?
        Otherwise, the image is wrapped.
    interp : str
        Interpolation for fshift function. Default is 'cubic'.
        Options are 'linear', 'cubic', or 'quintic'.
    shift_function : func
        which function to use for sub-pixel shifting.
        Options are fourier_imshift or fshift.
        fshift tends to be 3-5 times faster for similar results.

    Returns
    -------
    list
        (x, y, beta) values from LSQ optimization, where (x, y) 
        are the misalignment of target from reference and beta
        is the fraction by which the target intensity must be
        reduced to match the intensity of the reference.
    """
    from scipy.optimize import least_squares#, leastsq

    # Use loss='soft_l1' for least squares robust against outliers
    # May want to play around with f_scale...
    res = least_squares(shift_subtract, init_pars, diff_step=0.1,
                        loss='soft_l1', f_scale=1.0, args=(reference,target), 
                        kwargs={'mask':mask,'pad':pad,'shift_function':shift_function,'interp':interp})
    out = res.x
    #out,_ = leastsq(shift_subtract, init_pars, 
    #                args=(reference,target,mask,pad,shift_function))

    #results = [out[0],out[1],out[2]] #x,y,beta
    return out


# Fix NaN values
def fix_nans_with_med(im, niter_max=5, verbose=False, **kwargs):
    """Iteratively fix NaNs with surrounding Real data"""
    sh_orig = im.shape
    
    nan_mask = np.isnan(im)
    n_nans = np.where(nan_mask)[0].size
    if verbose: print('{} NaNs to start'.format(n_nans))
    
    for ii in range(niter_max):
        im = im.flatten()
        nan_mask = np.isnan(im)
        im = im.reshape(sh_orig)

        # Return if we no NaNs
        if not np.any(nan_mask): return im

        if verbose: print('Iter {}'.format(ii))

        # Shift
        im_smth = []
        for i in np.arange(-1,2):
            for j in np.arange(-1,2):
                im_smth.append(fshift(im, i, j))
        im_smth = np.array(im_smth)
        
        # Flatten arrays for indexing of NaNs
        im_smth = im_smth.reshape([im_smth.shape[0],-1])
        im = im.flatten()
        
        # Take median of only the NaN'ed pixels
        im[nan_mask] = np.nanmedian(im_smth[:,nan_mask], axis=0)
        im = im.reshape(sh_orig)
        
    nan_mask = np.isnan(im)
    if np.any(nan_mask):
        n_nans = np.where(nan_mask)[0].size
        if verbose:
            print('{} NaNs left after {} iterations.'.format(n_nans, niter_max))
        
    return im



def scale_ref_image(im1, im2, mask=None, smooth_imgs=False,
                    return_shift_values=False):
    """Reference image scaling
    
    Find value to scale a reference image by minimizing residuals.
    Assumes everything is already aligned if return_shift_values=False.
    
    Or simply turn on return_shift_values to return (dx,dy,scl). 
    Then fshift(im2,dx,dy) to shift the reference image.
    
    Parameters
    ----------
    im1 : ndarray
        Science star observation.
    im2 : ndarray
        Reference star observation.
    mask : bool array or None
        Use this mask to exclude pixels.
        Boolean mask where True is included and False is excluded.
    smooth_imgs : bool
        Smooth the images with nearest neighbors to remove bad pixels?
    return_shift_values : bool
        Option to return x and y shift values
    """
    
    # Mask for generating standard deviation
    if mask is None:
        mask = np.ones(im1.shape, dtype=bool)
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
        
    scale_init = np.nanmedian(im1[mask]) / np.nanmedian(im2[mask])
    if return_shift_values:
        params = [0.0, 0.0, scale_init]
    else:
        params = [scale_init]

    # Perform linear least squares fit on difference function
    return align_leastsq(im1, im2, bpmask=~mask, weights=None,
                         params0=params, func_shift=fshift, interp='linear')


    # if return_shift_values:
    #     return align_LSQ(im2[mask], im1[mask], shift_function=fshift)
    # else:
    #     _, _, scl = align_LSQ(im2[mask], im1[mask], shift_function=None)
    #     return scl

    # ind = np.where(im1==im1[mask].max())
    # ind = [ind[0][0], ind[1][0]]

    # # Initial Guess
    # scl = np.nanmean(im1[ind[0]-3:ind[0]+3,ind[1]-3:ind[1]+3]) / \
    #       np.nanmean(im2[ind[0]-3:ind[0]+3,ind[1]-3:ind[1]+3])
          
    # # Wider range
    # # Check a range of scale values
    # # Want to minimize the standard deviation of the differenced images
    # scl_arr = np.linspace(0.2*scl,2*scl,10)
    # mad_arr = []
    # for val in scl_arr:
    #     diff = im1 - val*im2
    #     mad_arr.append(robust.medabsdev(diff[mask]))
    # mad_arr = np.array(mad_arr)
    # scl = scl_arr[mad_arr==mad_arr.min()][0]

    # # Check a range of scale values
    # # Want to minimize the standard deviation of the differenced images
    # scl_arr = np.linspace(0.85*scl,1.15*scl,50)
    # mad_arr = []
    # for val in scl_arr:
    #     diff = im1 - val*im2
    #     mad_arr.append(robust.medabsdev(diff[mask]))
    # mad_arr = np.array(mad_arr)

    # #plt.plot(scl_arr,mad_arr)
    # return scl_arr[mad_arr==mad_arr.min()][0]


def optimal_difference(im_sci, im_ref, scale, binsize=1, center=None, 
                       mask_good=None, sub_mean=True, std_func=np.std):
    """Optimize subtraction of ref PSF
    
    Scale factors from scale_ref_image work great for subtracting
    a reference PSF from a science image where there are plenty
    of photons, but perform poorly in the noise-limited regime. If
    we simply perform a difference by scaling the reference image,
    then we also amplify the noise. In the background, it's better to
    simply subtract the unscaled reference pixels. This routine finds
    the radial cut-off of the dominant noise source.

    Parameters
    ----------
    im_sci : ndarray
        Science star observation.
    im_ref : ndarray
        Reference star observation.
    scale : float
        Scale factor from :func:`scale_ref_image`
    binsize : int
        Radial binsize (in pixels) to perform calculations
    center : tuple or None
        Location (x,y) to calculate radial distances.
        Default is center of image.
    mask_good : bool array
        Only perform operations on pixels where mask_good=True.
    sub_mean : bool
        Subtract mean (median, actually) of pixels in each
        radial bin? Basically a background subtraction.
    std_func : func
        What function do we want to use for calculating
        the standard deviation in each radial bin?
        After comparing the standard deviation between the
        two scaled differences in each radial bin, we only
        keep the better of the two.
    """

    diff1 = im_sci - im_ref
    diff2 = im_sci - im_ref * scale
    
    rho = dist_image(im_sci, center=center)
    
    # Only perform operations on pixels where mask_good=True
    if mask_good is None:
        mask_good = np.ones(rho.shape, dtype=bool)
    nan_mask1 = np.isnan(diff1)
    nan_mask2 = np.isnan(diff2)
    mask_good = mask_good & (~nan_mask1) & (~nan_mask2)
        
    rho_good = rho[mask_good]
    diff1_good = diff1[mask_good]
    diff2_good = diff2[mask_good]

    # Get the histogram indices
    bins = np.arange(rho_good.min(), rho_good.max() + binsize, binsize)
    igroups = hist_indices(rho_good, bins)
    nbins = len(igroups)

    # Standard deviation for each bin
    std1 = binned_statistic(igroups, diff1_good, func=std_func)
    std2 = binned_statistic(igroups, diff2_good, func=std_func)

    # Subtract the mean at each radius
    if sub_mean:
        med1 = binned_statistic(igroups, diff1_good, func=np.median)
        med2 = binned_statistic(igroups, diff2_good, func=np.median)
        for i in range(nbins):
            diff1_good[igroups[i]] -= med1[i]
            diff2_good[igroups[i]] -= med2[i]

    # Replace values in diff1 with better ones in diff2
    ibin_better = np.where(std2 < std1)[0]
    for ibin in ibin_better:
        diff1_good[igroups[ibin]] = diff2_good[igroups[ibin]]
            
    diff1[mask_good] = diff1_good
    return diff1

