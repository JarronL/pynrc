import numpy as np
import logging
_log = logging.getLogger('pynrc')

from poppy.utils import krebin

from .coords import dist_image
from scipy.ndimage import fourier_shift
from scipy.ndimage.interpolation import rotate
from astropy.io import fits

from webbpsf_ext.image_manip import pad_or_cut_to_size, fshift, fourier_imshift, frebin
from webbpsf_ext.image_manip import rotate_offset, rotate_shift_image
from webbpsf_ext.image_manip import image_rescale, model_to_hdulist
from webbpsf_ext.image_manip import convolve_image, crop_zero_rows_cols
from webbpsf_ext.maths import hist_indices, binned_statistic, fit_bootstrap

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

def align_LSQ(reference, target, mask=None, pad=False, interp='cubic',
              shift_function=fshift):
    """Find best shift value
    
    LSQ optimization with option of shift alignment algorithm
    
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

    init_pars = [0.0, 0.0, 1.0]

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
        Use this mask to exclude pixels for performing standard deviation.
        Boolean mask where True is included and False is excluded.
    smooth_imgs : bool
        Smooth the images with nearest neighbors to remove bad pixels?
    return_shift_values : bool
        Option to return x and y shift values
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
        
    # Perform linear least squares fit on difference function
    if return_shift_values:
        return align_LSQ(im2[mask], im1[mask], shift_function=fshift)
    else:
        _, _, scl = align_LSQ(im2[mask], im1[mask], shift_function=None)
        return scl

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
        mask_good = np.ones(rho.shape, dtype=np.bool)
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

