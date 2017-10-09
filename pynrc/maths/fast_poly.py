from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
#import logging
#_log = logging.getLogger('pynrc')

def jl_poly(xvals, coeff, dim_reorder=False):
    """
    Replacement for np.polynomial.polynomial.polyval(wgood, coeff)
    to evaluate y-values given a set of xvals and coefficients.
    Uses matrix multiplication, which is much faster. Beware, the
    default output array shapes organization may differ from the 
    polyval routine for 2D and 3D results.

    Inputs:
        xvals - 1D array (time, for instance)
        coeff - 1D, 2D, or 3D array of coefficients from a polynomial fit.
                The first dimension should have a number of elements equal
                to the polynomial degree + 1. Order such that lower degrees
                are first, and higher degrees are last.
        dim_reorder - Reorder output shape to mimic the polyval routine,
                where the first dimensions correspond to the coeff latter 
                dimensions, and the final dimension is equal to the number 
                of xvals.
                       
    Returns:
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
        raise ValueError('coefficient can only have 1, 2, or 3 dimensions. Found {} dimensions.'.format(ndim))

    # Create an array of exponent values
    parr = np.arange(dim[0], dtype='float')
    # If 3D, this reshapes xfan to 2D
    xfan = xvals**parr.reshape((-1,1)) # Array broadcasting

    # Reshape coeffs to 2D array
    cf = coeff.reshape(dim[0],-1)
    if not dim_reorder:
        # This is the Python preferred ordering
        # Coefficients are assumed (deg+1,ny,nx)
        # xvals have length nz
        # Result to be order (nz,ny,nx)
        yfit = np.dot(xfan.T,cf)

        if ndim==1 or n==1: yfit = yfit.flatten()
        if ndim==3: yfit = yfit.reshape((n,dim[1],dim[2]))
    else:
        # Coefficients are assumed (deg+1,nx,ny)
        # xvals have length nz
        # Result to be order (nx,ny,nz)
        yfit = np.dot(cf.T, xfan)

        if ndim==1 or n==1: yfit = yfit.flatten()
        if ndim==3: yfit = yfit.reshape((dim[1],dim[2],n))

    return yfit


def jl_poly_fit(x, yvals, deg=1, QR=True):
    """
    Fit a polynomial to a function using linear least-squares.
    
    Gives the option of performing QR decomposition, which provides
    a considerable speed-up compared to simply using np.linalg.lstsq().
    In addition to being fast, it has better numerical stability than
    linear regressions that involve matrix inversions (ie., dot(x.T,x)).
    
    Returns the coefficients of the fit for each pixel.
    
    # Example: Fit all pixels in a data cube to get slope image
    # in terms of ADU/sec
    nz, ny, nx = cube.shape
    tvals = (np.arange(nz) + 1) * 10.737
    slope = jl_poly_fit(tvals, cube)
    """

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
        coeff_all, _, _, _ = np.linalg.lstsq(r, qTb)
    else:
        coeff_all, _, _, _ = np.linalg.lstsq(a.T, b)
    
    return coeff_all.reshape(cf_shape)
