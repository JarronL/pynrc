from __future__ import absolute_import, division, print_function, unicode_literals

#__all__ = ['pad_or_cut_to_size', 'frebin', \
#           'fshift', 'fourier_imshift', 'shift_subtract', 'align_LSQ']
import numpy as np
import logging
_log = logging.getLogger('pynrc')

from poppy.utils import krebin

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

    offset = shift_function(reference, xshift, yshift, pad)
    
    if mask is not None:
        return ( (target - beta * offset) * mask ).flatten()
    else:
        return ( target - beta * offset ).flatten()

def align_LSQ(reference, target, mask=None, pad=False, shift_function=fshift):
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
            fshift tends to be 3-5 times faster for similar results.s
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

