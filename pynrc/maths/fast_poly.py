from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
#import logging
#_log = logging.getLogger('pynrc')

def jl_poly(xvals, coeff, dim_reorder=False):
    """Evaluate polynomial
    
    Replacement for np.polynomial.polynomial.polyval(wgood, coeff)
    to evaluate y-values given a set of xvals and coefficients.
    Uses matrix multiplication, which is much faster. Beware, the
    default output array shapes organization may differ from the 
    polyval routine for 2D and 3D results.

    Parameters
    ----------
    xvals : ndarray
        1D array (time, for instance)
    coeff : ndarray
        1D, 2D, or 3D array of coefficients from a polynomial fit.
        The first dimension should have a number of elements equal
        to the polynomial degree + 1. Order such that lower degrees
        are first, and higher degrees are last.
    dim_reorder : bool
        Reorder output shape to mimic the polyval routine,
        where the first dimensions correspond to the coeff latter 
        dimensions, and the final dimension is equal to the number 
        of xvals.
                       
    Returns
    -------
    float array
        An array of values where each xval has been evaluated at each
        set of supplied coefficients. The output shape has the first 
        dimension equal to the number of xvals, and the final dimensions
        correspond to coeff's latter dimensions. The result is flattened 
        if there is either only one xval or one set of coeff (or both).
    """

    # How many xvals?
    n = np.size(xvals)
    xdim = len(xvals.shape)
    if xdim>1:
        raise ValueError('xvals can only have 1 dimension. Found {} dimensions.'.format(xdim))

    # Check number of dimensions in coefficients
    dim = coeff.shape
    ndim = len(dim)
    if ndim>3:
        raise ValueError('coefficient can only have 1, 2, or 3 dimensions. \
                          Found {} dimensions.'.format(ndim))

    # Create an array of exponent values
    parr = np.arange(dim[0], dtype='float')
    # If 3D, this reshapes xfan to 2D
    xfan = xvals**parr.reshape((-1,1)) # Array broadcasting

    # Reshape coeffs to 2D array
    cf = coeff.reshape(dim[0],-1)
    if dim_reorder:
        # Coefficients are assumed (deg+1,nx,ny)
        # xvals have length nz
        # Result to be ordered (nx,ny,nz)
        yfit = np.dot(cf.T, xfan)

        if ndim==1 or n==1: yfit = yfit.flatten()
        if ndim==3: yfit = yfit.reshape((dim[1],dim[2],n))
    else:
        # This is the Python preferred ordering
        # Coefficients are assumed (deg+1,ny,nx)
        # xvals have length nz
        # Result to be ordered (nz,ny,nx)
        yfit = np.dot(xfan.T,cf)

        if ndim==1 or n==1: yfit = yfit.flatten()
        if ndim==3: yfit = yfit.reshape((n,dim[1],dim[2]))

    return yfit


def jl_poly_fit(x, yvals, deg=1, QR=True, robust_fit=False, niter=25):
    """Fast polynomial fitting
    
    Fit a polynomial to a function using linear least-squares.
    This function is particularly useful if you have a data cube
    and want to simultaneously fit a slope to all pixels in order
    to produce a slope image.
    
    Gives the option of performing QR decomposition, which provides
    a considerable speed-up compared to simply using np.linalg.lstsq().
    In addition to being fast, it has better numerical stability than
    linear regressions that involve matrix inversions (ie., dot(x.T,x)).
    
    Returns the coefficients of the fit for each pixel.
    
    Parameters
    ----------
    x : ndarray
        X-values of the data array (1D).
    yvals : ndarray 
        Y-values (1D, 2D, or 3D) where the first dimension
        must have equal length of x. For instance, if x is
        a time series of a data cube with size NZ, then the 
        data cube must follow the Python convention (NZ,NY,NZ).
    deg : int
        Degree of polynomial to fit to the data.
    QR : bool
        Perform QR decomposition? Default=True.
    robust_fit : bool
        Perform robust fitting, iteratively kicking out 
        outliers until convergence.
    niter : int
        Maximum number of iterations for robust fitting.
        If convergence is attained first, iterations will stop.
    
    Example
    -------
    Fit all pixels in a data cube to get slope image in terms of ADU/sec
    
    >>> nz, ny, nx = cube.shape
    >>> tvals = (np.arange(nz) + 1) * 10.737
    >>> coeff = jl_poly_fit(tvals, cube, deg=1)
    >>> bias = coeff[0]  # Bias image (y-intercept)
    >>> slope = coeff[1] # Slope image (DN/sec)
    """
    
    from pynrc.maths.robust import medabsdev
    
#     nz = 1000
#     tarr = (np.arange(nz) + 1) * 10.737
# 
#     cf_truth = np.array([3000., 1000.])
#     xpix = 10
#     ypix = 10
#     npix = xpix * ypix
#     
#     deg = len(cf_truth)-1
#     cf_all = np.broadcast_to(cf_truth, (npix,deg+1)).T.reshape(deg+1,npix)
#     yvals = jl_poly(tarr, cf_all)
#     yvals += 0.01 * np.median(yvals) * np.random.standard_normal(yvals.shape)
#     x = tarr
# 
#     # create outliers
#     outlier_prop = 0.3
#     outlier_IND = np.random.permutation(yvals.size)
#     outlier_IND = outlier_IND[0:int(np.floor(yvals.size * outlier_prop))]
#     z_noise_outlier = yvals.flatten()
#     z_noise_outlier[outlier_IND] += 10 * np.median(yvals) * np.random.standard_normal(z_noise_outlier[outlier_IND].shape)
#     z_noise_outlier = z_noise_outlier.reshape(yvals.shape)
# 
#     yvals = z_noise_outlier


    orig_shape = yvals.shape
    ndim = len(orig_shape)
    
    cf_shape = list(yvals.shape)
    cf_shape[0] = deg+1
    
    if ndim==1:
        assert len(x)==len(yvals), 'X and Y must have the same length'
    else:
        assert len(x)==orig_shape[0], 'X and Y.shape[0] must have the same length'

    a = np.array([x**num for num in range(deg+1)], dtype='float')
    b = yvals.reshape([orig_shape[0],-1])

    # Fast method, but numerically unstable for overdetermined systems
    #cov = np.linalg.pinv(np.dot(a,a.T))
    #coeff_all = np.dot(cov,np.dot(a,b))
    
    if QR:
        # Perform QR decomposition of the A matrix
        q, r = np.linalg.qr(a.T, 'reduced')
        # computing Q^T*b (project b onto the range of A)
        qTb = np.dot(q.T, b)
        # solving R*x = Q^T*b
        coeff_all = np.linalg.lstsq(r, qTb)[0]
    else:
        coeff_all = np.linalg.lstsq(a.T, b)[0]
        
    if robust_fit:
        # Normally, we would weight both the x and y (ie., a and b) values
        # then plug those into the lstsq() routine. However, this means we
        # can no longer use the same x values for a series of y values. Each
        # fit would have differently weight x-values, requiring us to fit
        # each element individually, which would be very slow. 
        # Instead, we will compromise by "fixing" outliers in order to 
        # preserve the quickness of this routine. The fixed outliers become 
        # the new data that we refit. 

        close_factor = 0.03
        close_enough = np.max([close_factor * np.sqrt(0.5/(x.size-1)), 1e-20])
        err = 0
        for i in range(niter):
            # compute absolute value of residuals (fit minus data)
            yvals_mod = jl_poly(x, coeff_all)
            abs_resid = np.abs(yvals_mod - b)

            # compute the scaling factor for the standardization of residuals
            # using the median absolute deviation of the residuals
            # 6.9460 is a tuning constant (4.685/0.6745)
            abs_res_scale = 6.9460 * np.median(abs_resid, axis=0)

            # standardize residuals
            w = abs_resid / abs_res_scale.reshape([1,-1])

            # exclude outliers
            outliers = w>1
            
            # Create a version with outliers fixed
            # Se
            yvals_fix = b.copy()
            yvals_fix[outliers] = yvals_mod[outliers]
            
            # Ignore fits with no outliers
            ind_fit = outliers.sum(axis=0) > 0
            if ind_fit[ind_fit].size == 0: break
            if QR:
                qTb = np.dot(q.T, yvals_fix[:,ind_fit])
                coeff_all[:,ind_fit] = np.linalg.lstsq(r, qTb)[0]
            else:
                coeff_all[:,ind_fit] = np.linalg.lstsq(a.T, yvals_fix[:,ind_fit])[0]

            prev_err = medabsdev(abs_resid, axis=0) if i==0 else err
            err = medabsdev(abs_resid, axis=0)
            
            diff = np.abs((prev_err - err)/err)
            #print(coeff_all.mean(axis=1), coeff_all.std(axis=1), np.nanmax(diff), ind_fit[ind_fit].size)
            if 0 < np.nanmax(diff) < close_enough: break
    
    return coeff_all.reshape(cf_shape)
