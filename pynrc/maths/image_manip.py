from __future__ import absolute_import, division, print_function, unicode_literals

# The six library is useful for Python 2 and 3 compatibility
import six

#__all__ = ['pad_or_cut_to_size', 'frebin', \
#           'fshift', 'fourier_imshift', 'shift_subtract', 'align_LSQ']
import numpy as np
import logging
_log = logging.getLogger('pynrc')

from poppy.utils import krebin

from pynrc.maths.coords import dist_image
#from pynrc.nrc_utils import (hist_indices, binned_statistics)
#    igroups = hist_indices(rho_good, bins)
#    nbins = len(igroups)

    # Standard deviation for each bin
#    std1 = binned_statistic(igroups, diff1_good, func=std_func)


from scipy.optimize import least_squares#, leastsq
from scipy.ndimage import fourier_shift

from astropy.io import fits


def pad_or_cut_to_size(array, new_shape):
    """
    Resize an array to a new shape by either padding with zeros
    or trimming off rows and/or columns. The ouput shape can
    be of any arbitrary amount.

    Parameters
    ----------
    array :  ndarray
        A 1D or 2D array representing some image
    padded_shape :  tuple of 2 elements
        Desired size for the output array. For 2D case, if a single value, 
        then will create a 2-element tuple of the same value.

    Returns
    -------
    output : ndarray
        An array of size new_shape that preserves the central information 
        of the input array.
    """

    ndim = len(array.shape)
    if ndim == 1:
        is_1d = True
        # Reshape array to a 2D array with nx=1
        array = array.reshape((-1,1))
        ny, nx = array.shape
        if isinstance(new_shape, float) or isinstance(new_shape, int):
            ny_new = int(round(new_shape))
            nx_new = 1
            new_shape = (ny_new, nx_new)
        elif len(new_shape) < 2:
            ny_new = nx_new = new_shape[0]
            new_shape = (ny_new, nx_new)
        else:
            ny_new = new_shape[0]
            nx_new = new_shape[1]
        output = np.zeros(shape=new_shape, dtype=array.dtype)
    elif ndim == 2:	
        is_1d = False
        ny, nx = array.shape
        if isinstance(new_shape, float) or isinstance(new_shape, int):
            ny_new = nx_new = int(round(new_shape))
            new_shape = (ny_new, nx_new)
        elif len(new_shape) < 2:
            ny_new = nx_new = new_shape[0]
            new_shape = (ny_new, nx_new)
        else:
            ny_new = new_shape[0]
            nx_new = new_shape[1]
        output = np.zeros(shape=new_shape, dtype=array.dtype)
    else:
        raise ValueError('Input image can only have 1 or 2 dimensions. \
                          Found {} dimensions.'.format(ndim))

    if nx_new>nx:
        n0 = (nx_new - nx) / 2
        n1 = n0 + nx
    elif nx>nx_new:
        n0 = (nx - nx_new) / 2
        n1 = n0 + nx_new
    else:
        n0 = 0; n1 = nx		
    n0 = int(round(n0))
    n1 = int(round(n1))

    if ny_new>ny:
        m0 = (ny_new - ny) / 2
        m1 = m0 + ny
    elif ny>ny_new:
        m0 = (ny - ny_new) / 2
        m1 = m0 + ny_new
    else:
        m0 = 0; m1 = ny		
    m0 = int(round(m0))
    m1 = int(round(m1))

    if (nx_new>=nx) and (ny_new>=ny):
        #print('Case 1')
        output[m0:m1,n0:n1] = array
    elif (nx_new<=nx) and (ny_new<=ny):
        #print('Case 2')
        output = array[m0:m1,n0:n1]
    elif (nx_new<=nx) and (ny_new>=ny):
        #print('Case 3')
        output[m0:m1,:] = array[:,n0:n1]
    elif (nx_new>=nx) and (ny_new<=ny):
        #print('Case 4')
        output[:,n0:n1] = array[m0:m1,:]
        
    # Flatten if input and output arrays are 1D
    if (ndim==1) and (nx_new==1):
        output = output.flatten()

    return output


def fshift(image, delx=0, dely=0, pad=False):
    """
    Ported from IDL function fshift.pro.
    Routine to shift an image by non-integer values.

    INPUTS:
        image - 2D image to be shifted
        delx  - shift in x (same direction as IDL SHIFT function)
        dely  - shift in y
        pad   - Should we pad the array before shifting, then truncate?
                Otherwise, the image is wrapped.
    OUTPUTS:
        shifted image is returned as the function results

    """
    
    if len(image.shape) == 1:
        # separate shift into an integer and fraction shift
        intx = np.int(delx)
        fracx = delx - intx
        if fracx < 0:
            fracx += 1
            intx -= 1

        # Pad ends with zeros
        if pad:
            padx = np.abs(intx) + 1
            x = np.pad(image,np.abs(intx),'constant')
        else:
            padx = 0
            x = image.copy()

        # shift by integer portion
        x = np.roll(x, intx)
        # if significant fractional shift...
        if not np.isclose(fracx, 0, atol=1e-5):
            x = x * (1.-fracx) + np.roll(x,1) * fracx

        x = x[padx:padx+image.size]
        return x

    elif len(image.shape) == 2:	
        # separate shift into an integer and fraction shift
        intx = np.int(delx)
        inty = np.int(dely)
        fracx = delx - intx
        fracy = dely - inty
        if fracx < 0:
            fracx += 1
            intx -= 1
        if fracy < 0:
            fracy += 1
            inty -= 1

        # Pad ends with zeros
        if pad:
            padx = np.abs(intx) + 1
            pady = np.abs(inty) + 1
            pad_vals = ([pady]*2,[padx]*2)
            x = np.pad(image,pad_vals,'constant')
        else:
            padx = 0; pady = 0
            x = image.copy()

        # shift by integer portion
        x = np.roll(np.roll(x, intx, axis=1), inty, axis=0)
    
        # Check if fracx and fracy are effectively 0
        fxis0 = np.isclose(fracx,0, atol=1e-5)
        fyis0 = np.isclose(fracy,0, atol=1e-5)
        # If fractional shifts are significant
        # use bi-linear interpolation between four pixels
        if not (fxis0 and fyis0):
            # Break bi-linear interpolation into four parts
            # to avoid NaNs unnecessarily affecting integer shifted dimensions
            x1 = x * ((1-fracx)*(1-fracy))
            x2 = 0 if fyis0 else np.roll(x,1,axis=0)*((1-fracx)*fracy)
            x3 = 0 if fxis0 else np.roll(x,1,axis=1)*((1-fracy)*fracx)
            x4 = 0 if (fxis0 or fyis0) else np.roll(np.roll(x, 1, axis=1), 1, axis=0) * fracx*fracy
    
            x = x1 + x2 + x3 + x4
    
        x = x[pady:pady+image.shape[0], padx:padx+image.shape[1]]
        return x
            

        #if not np.allclose([fracx,fracy], 0, atol=1e-5):
        #	x = x * ((1-fracx)*(1-fracy)) + \
        #		np.roll(x,1,axis=0) * ((1-fracx)*fracy) + \
        #		np.roll(x,1,axis=1) * (fracx*(1-fracy)) + \
        #		np.roll(np.roll(x, 1, axis=1), 1, axis=0) * fracx*fracy

        #x = x[pady:pady+image.shape[0], padx:padx+image.shape[1]]
        #return x

    else:
        raise ValueError('Input image can only have 1 or 2 dimensions. \
                          Found {} dimensions.'.format(len(image.shape)))
                          
                          
def fourier_imshift(image, xshift, yshift, pad=False):
    '''
    Shift an image by use of Fourier shift theorem
    Parameters:
        image : nd array
            N x K image
        xshift : float
            Pixel value by which to shift image in the x direction
        yshift : float
            Pixel value by which to shift image in the x direction
        pad : bool
            Should we pad the array before shifting, then truncate?
            Otherwise, the image is wrapped.
    Returns:
        offset : nd array
            Shifted image
    '''
    # Pad ends with zeros
    if pad:
        padx = np.abs(np.int(xshift)) + 1
        pady = np.abs(np.int(yshift)) + 1
        pad_vals = ([pady]*2,[padx]*2)
        im = np.pad(image,pad_vals,'constant')
    else:
        padx = 0; pady = 0
        im = image
    
    offset = fourier_shift( np.fft.fft2(im), (yshift,xshift) )
    offset = np.fft.ifft2(offset).real
    
    offset = offset[pady:pady+image.shape[0], padx:padx+image.shape[1]]
    
    return offset
    
def shift_subtract(params, reference, target, mask=None, pad=False, 
                   shift_function=fshift):
    '''
    Use Fourier Shift theorem for subpixel shifts for 
    input into least-square optimizer.
    
    Parameters:
        params : tuple
            xshift, yshift, beta
        reference : nd array
            See align_fourierLSQ
        target : nd array
            See align_fourierLSQ
        mask : nd array, optional
            See align_fourierLSQ
        pad  : bool
            Should we pad the array before shifting, then truncate?
            Otherwise, the image is wrapped.
            
        shift_function : which function to use for sub-pixel shifting
            
    Returns:
        1D nd array of target-reference residual after
        applying shift and intensity fraction.
    '''
    xshift, yshift, beta = params

    if shift_function is not None:
        offset = shift_function(reference, xshift, yshift, pad)
    else:
        offset = reference
    
    if mask is not None:
        return ( (target - beta * offset) * mask ).ravel() #.flatten()
    else:
        return ( target - beta * offset ).ravel() #.flatten()

def align_LSQ(reference, target, mask=None, pad=False, 
              shift_function=fshift):
    '''
    LSQ optimization with option of shift alignment algorithm
    
    Parameters:
        reference : nd array
            N x K image to be aligned to
        target : nd array
            N x K image to align to reference
        mask : nd array, optional
            N x K image indicating pixels to ignore when
            performing the minimization. The masks acts as
            a weighting function in performing the fit.
        shift_function : which function to use for sub-pixel shifting.
            Options are fourier_imshift or fshift.
            fshift tends to be 3-5 times faster for similar results.
    Returns:
        results : list
            [x, y, beta] values from LSQ optimization, where (x, y) 
            are the misalignment of target from reference and beta
            is the fraction by which the target intensity must be
            reduced to match the intensity of the reference.
    '''

    init_pars = [0.0, 0.0, 1.0]

    # Use loss='soft_l1' for least squares robust against outliers
    # May want to play around with f_scale...
    res = least_squares(shift_subtract, init_pars, diff_step=0.1,
                        loss='soft_l1', f_scale=1.0, args=(reference,target), 
                        kwargs={'mask':mask,'pad':pad,'shift_function':shift_function})
    out = res.x
    #out,_ = leastsq(shift_subtract, init_pars, 
    #                args=(reference,target,mask,pad,shift_function))

    results = [out[0],out[1],out[2]] #x,y,beta
    return res.x


def frebin(image, dimensions=None, scale=None, total=True):
    """
    Python port from the IDL frebin.pro
    Shrink or expand the size of a 1D or 2D array by an arbitary amount 
    using bilinear interpolation. Conserves flux by ensuring that each 
    input pixel is equally represented in the output array.

    Parameters
    ==========
    image      : Input image, 1-d or 2-d ndarray
    dimensions : Size of output array (take priority over scale)
    scale      : Factor to scale output array
    total      : Conserves the surface flux. If True, the output pixels 
                 will be the sum of pixels within the appropriate box of 
                 the input image. Otherwise, they will be the average.
         
    Returns the binned ndarray
    """

    if dimensions is not None:
        if isinstance(dimensions, float):
            dimensions = [int(dimensions)] * len(image.shape)
        elif isinstance(dimensions, int):
            dimensions = [dimensions] * len(image.shape)
        elif len(dimensions) != len(image.shape):
            raise RuntimeError("The number of input dimensions don't match the image shape.")
    elif scale is not None:
        if isinstance(scale, float) or isinstance(scale, int):
            dimensions = map(int, map(round, map(lambda x: x*scale, image.shape)))
        elif len(scale) != len(image.shape):
            raise RuntimeError("The number of input dimensions don't match the image shape.")
        else:
            dimensions = [scale[i]*image.shape[i] for i in range(len(scale))]
    else:
        raise RuntimeError('Incorrect parameters to rebin.\n\frebin(image, dimensions=(x,y))\n\frebin(image, scale=a')
    #print(dimensions)


    shape = image.shape
    if len(shape)==1:
        nlout = 1
        nsout = dimensions[0]
        nsout = int(round(nsout))
        dimensions = [nsout]
    elif len(shape)==2:
        nlout, nsout = dimensions
        nlout = int(round(nlout))
        nsout = int(round(nsout))
        dimensions = [nlout, nsout]
    if len(shape) > 2:
        raise ValueError('Input image can only have 1 or 2 dimensions. Found {} dimensions.'.format(len(shape)))
    

    if nlout != 1:
        nl = shape[0]
        ns = shape[1]
    else:
        nl = nlout
        ns = shape[0]

    sbox = ns / float(nsout)
    lbox = nl / float(nlout)
    #print(sbox,lbox)

    # Contract by integer amount
    if (sbox.is_integer()) and (lbox.is_integer()):
        image = image.reshape((nl,ns))
        result = krebin(image, (nlout,nsout))
        if not total: result /= (sbox*lbox)
        if nl == 1:
            return result[0,:]
        else:
            return result

    ns1 = ns - 1
    nl1 = nl - 1

    if nl == 1:
        #1D case
        _log.debug("Rebinning to Dimension: %s" % nsout)
        result = np.zeros(nsout)
        for i in range(nsout):
            rstart = i * sbox
            istart = int(rstart)
            rstop = rstart + sbox

            if int(rstop) < ns1:
                istop = int(rstop)
            else:
                istop = ns1

            frac1 = float(rstart) - istart
            frac2 = 1.0 - (rstop - istop)

            #add pixel values from istart to istop and subtract fraction pixel
            #from istart to rstart and fraction pixel from rstop to istop
            result[i] = np.sum(image[istart:istop + 1]) - frac1 * image[istart] - frac2 * image[istop]

        if total:
            return result
        else:
            return result / (float(sbox) * lbox)
    else:
        _log.debug("Rebinning to Dimensions: %s, %s" % tuple(dimensions))
        #2D case, first bin in second dimension
        temp = np.zeros((nlout, ns))
        result = np.zeros((nsout, nlout))

        #first lines
        for i in range(nlout):
            rstart = i * lbox
            istart = int(rstart)
            rstop = rstart + lbox

            if int(rstop) < nl1:
                istop = int(rstop)
            else:
                istop = nl1

            frac1 = float(rstart) - istart
            frac2 = 1.0 - (rstop - istop)

            if istart == istop:
                temp[i, :] = (1.0 - frac1 - frac2) * image[istart, :]
            else:
                temp[i, :] = np.sum(image[istart:istop + 1, :], axis=0) -\
                             frac1 * image[istart, :] - frac2 * image[istop, :]

        temp = np.transpose(temp)

        #then samples
        for i in range(nsout):
            rstart = i * sbox
            istart = int(rstart)
            rstop = rstart + sbox

            if int(rstop) < ns1:
                istop = int(rstop)
            else:
                istop = ns1

            frac1 = float(rstart) - istart
            frac2 = 1.0 - (rstop - istop)

            if istart == istop:
                result[i, :] = (1. - frac1 - frac2) * temp[istart, :]
            else:
                result[i, :] = np.sum(temp[istart:istop + 1, :], axis=0) -\
                               frac1 * temp[istart, :] - frac2 * temp[istop, :]

        if total:
            return np.transpose(result)
        else:
            return np.transpose(result) / (sbox * lbox)


# Fix NaN values
def fix_nans_with_med(im, niter_max=5, verbose=False):
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
        print('{} NaNs left after {} iterations.'.format(n_nans, niter_max))
        
    return im

    
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


def scale_ref_image(im1, im2, mask=None, smooth_imgs=False,
                    return_shift_values=False):
    """
    Find value to scale a reference image by minimizing residuals.
    This assumed everything is already aligned. Or simply turn on
    return_shift_values to return (dx,dy,scl). Then fshift(im2,dx,dy)
    to shift the reference image.
    
    Inputs
    ======
    im1 - Science star observation.
    im2 - Reference star observation.
    mask - Use this mask to exclude pixels for performing standard deviation.
           Boolean mask where True is included and False is excluded
    smooth_imgs - Smooth the images with nearest neighbors to remove bad pixels.
    return_shift_values - Option to return x and y shift values
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

###     ind = np.where(im1==im1[mask].max())
###     ind = [ind[0][0], ind[1][0]]
### 
###     # Initial Guess
###     scl = np.nanmean(im1[ind[0]-3:ind[0]+3,ind[1]-3:ind[1]+3]) / \
###           np.nanmean(im2[ind[0]-3:ind[0]+3,ind[1]-3:ind[1]+3])
###           
###     # Wider range
###     # Check a range of scale values
###     # Want to minimize the standard deviation of the differenced images
###     scl_arr = np.linspace(0.2*scl,2*scl,10)
###     mad_arr = []
###     for val in scl_arr:
###         diff = im1 - val*im2
###         mad_arr.append(robust.medabsdev(diff[mask]))
###     mad_arr = np.array(mad_arr)
###     scl = scl_arr[mad_arr==mad_arr.min()][0]
### 
###     # Check a range of scale values
###     # Want to minimize the standard deviation of the differenced images
###     scl_arr = np.linspace(0.85*scl,1.15*scl,50)
###     mad_arr = []
###     for val in scl_arr:
###         diff = im1 - val*im2
###         mad_arr.append(robust.medabsdev(diff[mask]))
###     mad_arr = np.array(mad_arr)
### 
###     #plt.plot(scl_arr,mad_arr)
###     return scl_arr[mad_arr==mad_arr.min()][0]


def optimal_difference(im_sci, im_ref, scale, binsize=1, center=None, 
                       mask_good=None, sub_mean=True, std_func=np.std):
    """
    Scale factors from scale_ref_image work great for subtracting
    a reference PSF from a science image where there are plenty
    of photons, but perform poorly in the noise-limited regime. If
    we simply perform a difference by scaling the reference image,
    then we also amplify the noise. In the background, it's better to
    simply subtract the unscaled reference pixels. This routine finds
    the radial cut-off of the dominant noise source.
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


def hist_indices(values, bins=10, return_more=False):
    """
    This function bins an input of values and returns the indices for
    each bin. This is similar to the reverse indices functionality
    of the IDL histogram routine. It's also much faster than doing
    a for loop and creating masks/indice at each iteration, because
    we utilize a sparse matrix constructor. It's kinda magical...
    
    Returns of a list of indices grouped together according to the bin.
    Only works for evenly spaced bins.
    
    Parameters
    ==========
    values  - Input numpy array. Should be a single dimension.
    bins    - If bins is an int, it defines the number of equal-width bins 
              in the given range (10, by default). If bins is a sequence, 
              it defines the bin edges, including the rightmost edge.
   
    return_more - Option to also return the values organized by bin and 
                  the value of the centers (igroups, vgroups, center_vals).
    
    Example
    ==========
        # Find the standard deviation at each radius of an image
        rho = dist_image(image)
        binsize = 1
        bins = np.arange(rho.min(), rho.max() + binsize, binsize)
        igroups, vgroups, center_vals = hist_indices(rho, bins, True)
        # Get the standard deviation of image at each bin
        std = binned_statistic(igroups, image, np.std)

    """
    
    from scipy.sparse import csr_matrix
    
    values_flat = values.ravel()

    v0 = values_flat.min()
    v1 = values_flat.max()
    N  = len(values_flat)   
    
    try: # if bins is an integer
        binsize = (v1 - v0) / bins
        bins = np.arange(v0, v1 + binsize, binsize)
    except: # otherwise assume it's already an array
        binsize = bins[1] - bins[0]
    
    # Central value of each bin
    center_vals = bins[:-1] + binsize / 2.
    nbins = center_vals.size

    digitized = ((nbins-1.0) / (v1-v0) * (values_flat-v0)).astype(np.int)
    csr = csr_matrix((values_flat, [digitized, np.arange(N)]), shape=(nbins, N))

    # Split indices into their bin groups    
    igroups = np.split(csr.indices, csr.indptr[1:-1])
    
    if return_more:
        vgroups = np.split(csr.data, csr.indptr[1:-1])
        return (igroups, vgroups, center_vals)
    else:
        return igroups
    

def binned_statistic(x, values, func=np.mean, bins=10):
    """
    Compute a binned statistic for a set of data. Drop-in replacement
    for scipy.stats.binned_statistic.

    Parameters
    ==========
    x      - A sequence of values to be binned. Or a list of binned 
             indices from hist_indices().
    values - The values on which the statistic will be computed.
    func   - The function to use for calculating the statistic. 
    bins   - If bins is an int, it defines the number of equal-width bins 
             in the given range (10, by default). If bins is a sequence, 
             it defines the bin edges, including the rightmost edge.
             This doens't do anything if x is a list of indices.
             
    Example
    ==========
        # Find the standard deviation at each radius of an image
        rho = dist_image(image)
        binsize = 1
        bins = np.arange(rho.min(), rho.max() + binsize, binsize)
        igroups, vgroups, center_vals = hist_indices(rho, bins, True)
        # Get the standard deviation of image at each bin
        std = binned_statistic(igroups, image, np.std)
    
    """

    values_flat = values.ravel()
    
    try: # This will be successful if x is not already a list of indices
    
        # Check if bins is a single value
        if (len(np.array(bins))==1) and (bins is not None):
            igroups = hist_indices(x, bins)
            res = np.array([func(values_flat[ind]) for ind in igroups])
        # Otherwise we assume bins is a list or array defining edge locations
        else:
            bins = np.array(bins)
            # Check if binsize is the same for all bins
            bsize = bins[1:] - bins[:-1]
            if np.isclose(bsize.min(), bsize.max()):
                igroups = hist_indices(x, bins)
                res = np.array([func(values_flat[ind]) for ind in igroups])
            else:
                # If non-uniform bins, just use scipy.stats.binned_statistic
                from scipy import stats 
                res, _, _ = stats.binned_statistic(x, values, func, bins)
    except:
        igroups = x
        res = np.array([func(values_flat[ind]) for ind in igroups])
    
    return res
