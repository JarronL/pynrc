#!/usr/bin/env python

import numpy as np
import os
from scipy import sparse
from time import time

from tqdm import tqdm, trange

from astropy.io import fits

import logging
# Define logging
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

__all__ = ['NMF', 'data_masked_only', 'data_masked_only_revert', 
           'NMFcomponents', 'NMFmodelling', 'NMFsubtraction', 'NMFbff', 'nmf_math']

""" 
nmf.py

    This piece of software is developed and maintained by Guangtun Ben Zhu, 
    It is designed to solve nonnegative matrix factorization (NMF) given a dataset with heteroscedastic 
    uncertainties and missing data with a vectorized multiplicative update rule (Zhu 2016).
    The un-vectorized (i.e., indexed) update rule for NMF without uncertainties or missing data was
    originally developed by Lee & Seung (2000), and the un-vectorized update rule for NMF
    with uncertainties or missing data was originally developed by Blanton & Roweis (2007).

    As all the codes, this code can always be improved and any feedback will be greatly appreciated.

    Note:
      -- Between W and H, which one is the basis set and which one is the coefficient
         depends on how you interpret the data, because you can simply transpose everything
         as in X-WH versus X^T - (H^T)(W^T)
      -- Everything needs to be non-negative

    Here are some small tips for using this code:
      -- The algorithm can handle heteroscedastic uncertainties and missing data.
         You can supply the weight (V) and the mask (M) at the instantiation:

         >> g = nmf.NMF(X, V=V, M=M)

         This can also be very useful if you would like to iterate the process
         so that you can exclude certain new data by updating the mask.
         For example, if you want to perform a 3-sigma clipping after an iteration
         (assuming V is the inverse variance below):

         >> chi2_red, time_used = g.SolveNMF()
         >> New_M = np.copy(M)
         >> New_M[np.fabs(np.sqrt(V)*(X-np.dot(g.W, g.H)))>3] = False
         >> New_g = nmf.NMF(X, V=V, M=New_M)

         Caveat: Currently you need to re-instantiate the object whenever you update
         the weight (V), the mask (M), W, H or n_components.
         At the instantiation, the code makes a copy of everything.
         For big jobs with many iterations, this could be a severe bottleneck.
         For now, I think this is a safer way.

      -- It has W_only and H_only options. If you know H or W, and would like
         to calculate W or H. You can run, e.g.,

         >> chi2_red, time_used = g.SolveNMF(W_only=True)

         to get the other matrix (H in this case).


    Copyright (c) 2015-2016 Guangtun Ben Zhu

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
    and associated documentation files (the "Software"), to deal in the Software without 
    restriction, including without limitation the rights to use, copy, modify, merge, publish, 
    distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or 
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
    BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
class NMF:
    """
    Nonnegative Matrix Factorization - Build a set of nonnegative basis components given 
    a dataset with Heteroscedastic uncertainties and missing data with a vectorized update rule.

    Algorithm:
      -- Iterative multiplicative update rule

    Input: 
      -- X: m x n matrix, the dataset

    Optional Input/Output: 
      -- n_components: desired size of the basis set, default 5

      -- V: m x n matrix, the weight, (usually) the inverse variance
      -- M: m x n binary matrix, the mask, False means missing/undesired data
      -- H: n_components x n matrix, the H matrix, usually interpreted as the coefficients
      -- W: m x n_components matrix, the W matrix, usually interpreted as the basis set
    
    (See README for how to retrieve the test data)
    
    Construct a new basis set with 12 components
    Instantiation: 
        >> g = nmf.NMF(flux, V=ivar, n_components=12)
    Run the solver: 
        >> chi2_red, time_used = g.SolveNMF()

    If you have a basis set, say W, you would like to calculate the coefficients:
        >> g = nmf.NMF(flux, V=ivar, W=W_known)
    Run the solver: 
        >> chi2_red, time_used = g.SolveNMF(H_only=True)

    Comments:
      -- Between W and H, which one is the basis set and which one is the coefficient 
         depends on how you interpret the data, because you can simply transpose everything
         as in X-WH versus X^T - (H^T)(W^T)
      -- Everything needs to be non-negative

    References:
      -- Guangtun Ben Zhu, 2016
         A Vectorized Algorithm for Nonnegative Matrix Factorization with 
         Heteroskedastic Uncertainties and Missing Data
         AJ/PASP, (to be submitted)
      -- Blanton, M. and Roweis, S. 2007
         K-corrections and Filter Transformations in the Ultraviolet, Optical, and Near-infrared
         The Astronomical Journal, 133, 734
      -- Lee, D. D., & Seung, H. S., 2001
         Algorithms for non-negative matrix factorization
         Advances in neural information processing systems, pp. 556-562

    To_do:
      -- 

    History:
        -- 22-May-2016, Documented, BGT, JHU
        -- 13-May-2016, Add projection mode (W_only, H_only), BGT, JHU
        -- 30-Nov-2014, Started, BGT, JHU
    """

    def __init__(self, X, W=None, H=None, V=None, M=None, n_components=5, rand_seed=None):
        """ Initialization

        Parameters
        ----------
        X : ndarray
            m x n matrix, the input data set.

        Keyword Arguments
        -----------------
        W : ndarray, optional
            m x n_components matrix, the W matrix, usually interpreted as the basis set
        H : ndarray, optional
            n_components x n matrix, the H matrix, usually interpreted as the coefficients
        V : ndarray, optional
            m x n matrix, the weight, (usually) the inverse variance
        M : ndarray, optional
            m x n binary matrix, the mask, False means missing/undesired data
        n_components : int, optional
            Desired size of the basis set. Default 5.
            Cannot be larger than the number of columns in X (n).
        rand_seed : int, optional
            random seed for reproducibility of any randomly generated matrices
        """

        # Create local random number generator to avoid global seed setting
        self._rng = np.random.default_rng(seed=rand_seed)

        self._n_components = n_components
        self.maxiters = 1000
        self.tol = 1e-5

        # Make a copy for the safety of everything; should not be a bottleneck
        self.X = np.copy(X) 
        if (np.count_nonzero(self.X<0)>0):
            _log.info("There are negative values in X. Setting them to be zero...")
            self.X[self.X<0] = 0.

        # Basis set
        if (W is None):
            self.W = self._rng.random((self.X.shape[0], n_components))
        else:
            # if (W.shape != (self.X.shape[0], n_components)):
            if W.shape[0] != self.X.shape[0]:
                raise ValueError(f"Initial W has wrong shape {W.shape}.")
            self.W = np.copy(W)

        # Coefficients
        if (H is None):
            self.H = self._rng.random((n_components, self.X.shape[1]))
        else:
            # if (H.shape != (n_components, self.X.shape[1])):
            if H.shape[1] != self.X.shape[1]:
                raise ValueError(f"Initial H has wrong shape {H.shape}")
            self.H = np.copy(H)

        # Weights
        if (V is None):
            self._V = np.ones(self.X.shape)
        else:
            if (V.shape != self.X.shape):
                raise ValueError(f"Initial V(Weight) has wrong shape {V.shape}.")
            self.V = np.copy(V)

        # Mask
        if (M is None):
            self.M = np.ones(self.X.shape, dtype='bool')
        else:
            if (M.shape != self.X.shape):
                raise ValueError(f"M(ask) has wrong shape {M.shape}.")
            if (M.dtype != bool):
                raise TypeError("M(ask) needs to be boolean.")
            self.M = np.copy(M)

        # Set masked elements to be zero
        self._V[~self.M] = 0

    @property
    def n_components(self):
        """ Number of components """
        return self._n_components
    @n_components.setter
    def n_components(self, ncomps_new):
        """ Set the number of components """

        # Add additional components to W and H if needed
        ncomps_full = self._W.shape[1]
        if ncomps_new > ncomps_full:
            # Make new arrays with the new (larger) number of components
            Wtemp = self._rng.random((self.X.shape[0], ncomps_new))
            Htemp = self._rng.random((ncomps_new, self.X.shape[1]))
            # Copy over the old values
            Wtemp[:, :(ncomps_full-1)] = self._W
            Htemp[:(ncomps_full-1), :] = self._H
            # Set the new arrays
            self.W = Wtemp
            self.H = Htemp        

        # Update the number of components
        self._n_components = ncomps_new

    @property
    def W(self):
        """ Basis set components """
        return self._W[:, 0:self.n_components]
    @W.setter
    def W(self, value):
        """ Set the basis set components """
        neg_mask = value<0
        if (np.count_nonzero(neg_mask) > 0):
            _log.warning("There are negative values in W. Setting them to be zero...")
            value[neg_mask] = 0.
        # Fortran ordering, column elements contiguous in memory.
        self._W = np.array(value, order='F') 

    @property
    def H(self):
        """ Coefficients """
        return self._H[0:self.n_components, :]
    @H.setter
    def H(self, value):
        """ Set the coefficients """
        neg_mask = value<0
        if (np.count_nonzero(neg_mask) > 0):
            _log.warning("There are negative values in H. Setting them to be zero...")
            value[neg_mask] = 0.
        # C ordering, row elements contiguous in memory.
        self._H = np.array(value, order = 'C') 

    @property
    def V(self):
        """ Weights """
        return self._V
    @V.setter
    def V(self, value):
        neg_mask = value<0
        if (np.count_nonzero(neg_mask<0) > 0):
            _log.warning("There are negative values in V. Setting them to be zero...")
            value[neg_mask<0] = 0.
        self._V = value

    @property 
    def V_size(self):
        """ Total number of valid data points """
        return np.count_nonzero(self.V)

    @property
    def cost(self):
        """  Total cost of a given set s """
        diff = self.X - np.dot(self.W, self.H)
        # chi2 = np.einsum('ij,ij', self.V*diff, diff) / self.V_size

        # Reduced chi2 metric: chi2 = sum(V*(X-WH)^2) / V_size
        chi2 = np.sum(self.V * diff**2) / self.V_size
        return chi2

    def SolveNMF(self, W_only=False, H_only=False, sparsemode=False, 
                 maxiters=None, tol=None, verbose=False, **kwargs):
        """
        Construct the NMF basis set and coefficients

        Parameters
        ----------
        W_only : bool
            If True, only update W, assuming H is known
        H_only : bool
            If True, only update H, assuming W is known
        sparsemode : bool
            If True, use sparse matrix for the calculation. Benchmarks show that
            this is only faster for very sparse data (e.g., >95% zeros).
        maxiters : int
            Maximum number of iterations. If None, use the default value
            at self.maxiters (initially 1000).
        tol : float
            Convergence criterion. If None, use the default value at self.tol
            (initially 1E-5).
        verbose: bool
            If True, print out extra information such as wall clock time and chi2

        Returns
        -------
        chi2 : float
            Reduced final cost
        time_used : float
            Time in sec to run
        """

        t0 = time()

        if (maxiters is None): 
            maxiters = self.maxiters
        if (tol is None):
            tol = self.tol

        chi2 = self.cost
        oldchi2 = np.finfo(chi2.dtype).max # 1e100

        if (W_only and H_only):
            _log.warning("Both W_only and H_only are set to be True. Returning...")
            return (chi2, 0.)

        if (sparsemode == True):
            V = sparse.csr_matrix(self.V)
            VT = sparse.csr_matrix(self.V.T)
            func_mult = sparse.csr_matrix.multiply
            func_dot = sparse.csr_matrix.dot
        else:
            V = self.V #np.copy(self.V)
            VT = V.T
            func_mult = np.multiply
            func_dot = np.dot

        if (not W_only):
            XVT = func_mult(VT, self.X.T)
        if (not H_only):
            XV = func_mult(V, self.X)

        niter = 0
        # while (niter < self.maxiters) and (chi_change > self.tol):
        tqdm_desc = f"{self.n_components}-component NMF"
        tqdm_iter = trange(maxiters, desc=tqdm_desc, leave=False)
        for niter in tqdm_iter:

            # Update H
            if (not W_only):
                H_up = func_dot(XVT, self.W)
                WHVT = func_mult(VT, np.dot(self.W, self.H).T)
                H_down = func_dot(WHVT, self.W)
                H_new = self.H * (H_up.T / H_down.T)
                self._H[0:self.n_components, :] = H_new

            # Update W
            if (not H_only):
                W_up = func_dot(XV, self.H.T)
                WHV = func_mult(V, np.dot(self.W, self.H))
                W_down = func_dot(WHV, self.H.T)
                W_new = self.W * (W_up / W_down)
                self._W[:, 0:self.n_components] = W_new

            # Update chi2
            oldchi2 = chi2
            chi2 = self.cost
            chi_change = (oldchi2 - chi2) / oldchi2

            # Some quick check. May need its error class ...
            if (not np.isfinite(chi2)):
               raise ValueError("NMF construction failed, likely due to missing data")

            # print output updates
            if (np.mod(niter, 20)==0):
                tqdm_iter.set_description(f"{tqdm_desc}: Chi2={chi2:.3f}, Change={chi_change:.1e}")
                # _log.info(f"Current Chi2={chi2:.4f}, Previous Chi2={oldchi2:.4f}, Change={100*chi_change:.4f}% @ niters={niter}")

            if (np.abs(chi_change) < tol):
                break

        if (niter == maxiters):
            _log.warning(f"SolveNMF reached maximum number of iterations ({maxiters})")

        time_total = time() - t0
        if verbose:
            _log.info(f"{self.n_components} took {time_total:.1f} seconds. Final chi2={chi2:.4f} in {niter+1} iterations.")

        return (chi2, time_total)
    
    def SolveNMF_OneByOne(self, W_only=False, H_only=False, sparsemode=False, 
                          maxiters=None, tol=None, verbose=False, ncomp_start=1, **kwargs):
        """Build NMF components one-by-one
        
        Each component is built by adding one more component to the basis set and the coefficients.
        This can be useful for checking the convergence of the NMF algorithm.
        Option to start at a specific component number in case W and H are already partially built.
        """

        if ncomp_start < 1:
            raise ValueError("ncomp_start must be at least 1")
        elif ncomp_start > self.n_components:
            raise ValueError(f"ncomp_start={ncomp_start} must be less than {self.n_components+1}")
        elif ncomp_start == 1:
            _log.info(f"Solving {self.n_components}-component NMF one-by-one")
        else:
            _log.info(f"Solving {self.n_components}-component NMF one-by-one, starting at {ncomp_start} components")

        time_total = 0
        for ncomp in range(ncomp_start, self.n_components+1):
            self.n_components = ncomp
            chi2, partial_time = self.SolveNMF(W_only=W_only, H_only=H_only, sparsemode=sparsemode, 
                                               maxiters=maxiters, tol=tol, verbose=verbose, **kwargs)
            time_total += partial_time

        if verbose:
            _log.info(f"Took {time_total:.1f} seconds to reach current solution.")

        return (chi2, time_total)


# This code is the nmf_imaging.py adjusted for pyKLIP at https://bitbucket.org/pyKLIP/pyklip/src/master/pyklip/nmf_imaging.py
# Another version is kept at https://github.com/seawander/nmf_imaging/blob/master/nmf_imaging_for_pyKLIP.py

def data_masked_only(data, mask):
    """ Return the data where the same regions are ignored in all the data

    Parameters
    ----------
    data: ndarray
        shape (p, N) where p is the number of pixels in each reference and N is the number of references,
    mask: boolean array
        shape (p) where False will be ignored in the output

    Returns
    -------
    data_focused: ndarray
        shape (p_focused, N) where p_focused is the number of True's in mask
    """   
    data_focused = data[mask]
    if len(data.shape) == 1:
        data_focused = data_focused[:, np.newaxis]
    
    return data_focused
    
def data_masked_only_revert(data, mask):
    """ Return the data where the same regions were ignored in all the data

    Parameters
    ----------
    data : ndarray
        shape (p_focused, N) where N is the number of references, p_focused is the number of 1's in mask
    mask : boolean array
        shape (p) where False was previously ignored in the input

    Returns
    -------
    data_focused_revert: ndarray
        shape (p, N) where  p is the number of pixels in each reference. Ignored data is replaced with NaNs.
    """ 
    data_focused_revert = np.zeros((len(mask), data.shape[1])) * np.nan
    data_focused_revert[mask] = data
    
    return data_focused_revert

def NMFcomponents(ref, ref_err=None, ignore_mask=None, n_components=5, 
                  maxiters=1000, oneByOne=True, save=False, save_name=None, save_dir=None, 
                  recalculate=False, rand_seed=None, **kwargs):
    """Returns the NMF components, where the rows contain the information.

    Parameters
    ----------
    ref : 2D array
        Reference images of shape (N,p) where N is the number of references, p is the number of pixels in each reference.
    ref_err : 2D array
         Uncertainty in the reference images of shape (N,p).
    ignore_mask : 1D or 2D array
        Mask pixels in each image that you don't want to use. Can either ben of size (p) or (N,p).
    n_components : int
        Number of components to be used. If None, all the components will be used. If greater than N, N will be used.
    maxiters : int
        Maximum number of iterations for the NMF algorithm.
    oneByOne : bool
        If True, the components will be built one-by-one. If False, all the components will be built at once.
    save : bool
        If True, the components and coefficients will be saved to disk for reuse.
    save_dir : str
        Path to save the NMF components (files named '_comp.fits' and '_coef.fits')
    recalculate : bool
        If True, the components will be recalculated even if they already exist on disk.
    rand_seed : int
        Random seed for reproducibility of any randomly generated matrices

    Returns
    -------
    components : 2D array
        NMf components of shape (n_components, p).
    """
    
    if save_dir is None:
        save_dir = '.'
    elif not os.path.exists(save_dir):
        raise ValueError(f"Path {save_dir} does not exist.")
    
    name_append = '' if save_name is None else f'{save_name}'
    ext_name = 'onebyone' if oneByOne else 'allatonce'
    fpath_comp = os.path.join(save_dir, f'{name_append}_comp_{ext_name}_{n_components}comps.fits')
    fpath_coef = os.path.join(save_dir, f'{name_append}_coef_{ext_name}_{n_components}comps.fits')

    # Search for file with same name but larger number of components
    if (not os.path.exists(fpath_comp)) and (not recalculate):
        for f in os.listdir(save_dir):
            if f.startswith(f'{name_append}_comp_{ext_name}_') and f.endswith('comps.fits'):
                ncomps_file = int(f.split('_')[-1].split('comps')[0])
                if ncomps_file > n_components:
                    _log.info(f'Found existing components with {ncomps_file} components. Using those instead...')
                    fpath_comp = os.path.join(save_dir, f)
                    fpath_coef = os.path.join(save_dir, f.replace(f'{name_append}_comp_', f'{name_append}_coef_'))
                    break

    # Set recalculate keyword to False if the components do not exist on disk
    # Will still calculate components, but provides correct print messages
    if recalculate and (not os.path.exists(fpath_comp)):
        recalculate = False

    if ref_err is None:
        ref_err = np.sqrt(ref)
        
    ncomp, npix = ref.shape
    if (n_components is None) or (n_components > ncomp):
        n_components = ncomp
        
    if ignore_mask is None:
        ignore_mask = np.ones_like(ref, dtype='bool')
    elif ignore_mask.shape == ref.shape[1]:
        # Create a 2D mask of shape (N, p)
        ignore_mask = np.tile(ignore_mask, (ncomp, 1))

    assert ref.shape == ref_err.shape, "Reference and error maps must have the same shape."
    assert ref.shape == ignore_mask.shape, "Reference and ignore mask must have the same shape."
    
    # ignore certain values in component construction
    ignore_mask[ref <= 0] = False # 1. negative values
    ignore_mask[~np.isfinite(ref)] = False # 2. infinite values
    ignore_mask[np.isnan(ref)] = False # 3. nan values
    
    ignore_mask[ref_err <= 0] = False # 1. negative values in input error map
    ignore_mask[~np.isfinite(ref_err)] = False # 2. infinite values in input error map
    ignore_mask[np.isnan(ref_err)] = False # 3. nan values in input error map
    
    # Ignore pixels that are always masked across all references
    mask_mark = np.bitwise_or.reduce(ignore_mask, axis=0)
    
    # Transpose the data to comply with the NMF class
    ref_columnized = data_masked_only(ref.T, mask_mark)
    ref_err_columnized = data_masked_only(ref_err.T, mask_mark)
    mask_columnized = data_masked_only(ignore_mask.T, mask_mark)

    # Update bad pixels
    ref_columnized[~mask_columnized] = 0 # assign 0 to ignored values, should not impact the final result
    ref_err_columnized[~mask_columnized] = np.nanmax(ref_err_columnized) # assign max uncertainty to ignored values, should not impact the final result

    ref_columnized[np.isnan(ref_columnized)] = 0 # assign 0 to NaN values, should not impact the final result
    ref_err_columnized[np.isnan(ref_err_columnized)] = np.nanmax(ref_err_columnized) # assign max uncertainty to NaN values, should not impact the final result

    # weights
    weights = 1.0 / ref_err_columnized**2


    # component calculation
    skip_calc = False
    if os.path.exists(fpath_comp) and (not recalculate):
        _log.info(f'Loading existing components and coefficients from {save_dir}...')
        W_assign = fits.getdata(fpath_comp)
        H_assign = fits.getdata(fpath_coef)
        if W_assign.shape[1] == n_components:
            skip_calc = True
        if W_assign.shape[1] > n_components:
            _log.info(f'You have already had {W_assign.shape[1]} components while asking for {n_components}. Returning...')
            W_assign = W_assign[:, :n_components]
            H_assign = H_assign[:n_components, :]
            skip_calc = True
        else:
            # Set oneByOne to True to get remaining components
            _log.info(f'{W_assign.shape[1]} existing components lower than requested ({n_components}). Calculating the rest...')
            oneByOne = True
    else:
        W_assign = None
        H_assign = None

    if recalculate or (not os.path.exists(fpath_comp)):
        _log.info("Building components one-by-one...")
        if recalculate:
            _log.warning('Recalculating components from scratch...')
        elif not os.path.exists(fpath_comp):
            _log.info(f'{fpath_comp} does not exist, calculating from scratch...')

    if not skip_calc:
        g_img = NMF(ref_columnized, W=W_assign, H=H_assign, V=weights, M=mask_columnized, 
                    n_components=n_components, rand_seed=rand_seed)
        
        # Ensure maxiters is an integer
        g_img.maxiters = int(maxiters)

        if oneByOne:
            ncomp_start = 1 if W_assign is None else W_assign.shape[1]
            chi2, time_used = g_img.SolveNMF_OneByOne(ncomp_start=ncomp_start, **kwargs)
        else:
            chi2, time_used = g_img.SolveNMF(**kwargs)
        W_assign, H_assign = (g_img.W, g_img.H)

        if save:
            str_write = f"overwriting" if recalculate else "writing"
            _log.info(f'\t\t\t Finished NMF for {n_components} components, {str_write} 2D component matrix to {fpath_comp}')
            fits.writeto(fpath_comp, W_assign, overwrite=True)

            _log.info(f'\t\t\t Finished NMF for {n_components} components, {str_write} 2D coefficient matrix to {fpath_coef}')
            fits.writeto(fpath_coef, H_assign, overwrite = True)

    # Normalize components
    norm_val = np.sqrt(np.nansum(W_assign**2, axis=0))
    components_column = W_assign / norm_val
    # components of shape (p, n_components)
    components = data_masked_only_revert(components_column, mask=mask_mark)   

    return components.T
    

def NMFmodelling(sci, components, n_components=None, mask_components=None, mask_data_imputation=None, 
                 sci_err=None, oneByOne=False, maxiters=1e3, sciThresh=0, rand_seed=None, **kwargs):
    """ NMF modeling of target data

    Parameters:
    -----------
    sci: 1D array
        Target image of shape (p) where p is the number of pixels
    components: 2D array
        NMF components of shape (N, p)
    n_components: int
        Number of components to be used. If None, all the components will be used. If greater than N, N will be used.
    mask_components: 1D array
        Mask pixels in component array that you don't want to use. Should be of size (p).
        False values will be ignored in the modeling.
    mask_data_imputation: 1D array
        Mask pixels in the target data that you don't want to use. Can be of size (p).
        False values will be ignored in the modeling.

    Args:
        sci: 1D array, p pixels
        components: N * p, calculated using NMFcomponents.
        n_components: how many components do you want to use. If None, all the components will be used.
        cube: whether output a cube or not (increasing the number of components).
        sciThresh: ignore the regions with low photon counts. Especially when they are ~10^-15 or smaller. I chose 0 in this case.
    
    Returns:
        NMF model of the target.
    """

    if n_components is None:
        n_components = components.shape[0]
    n_components = np.min([n_components, components.shape[0]])
        
    if sci_err is None:
        sci_err = np.sqrt(sci)

    if mask_components is None:
        masks_nans = np.isnan(components[:n_components])
        mask_components = ~np.bitwise_or.reduce(masks_nans, axis=0)

     # columnize the components, make sure NonnegMFPy returns correct results.
    components_column_all = data_masked_only(components[:n_components].T, mask=mask_components)  
    # Ensure components are normalized
    norm_val = np.sqrt(np.nansum(components_column_all**2, axis = 0))
    components_column_all = components_column_all / norm_val
    
    if mask_data_imputation is not None:
        flag_di = True
        _log.info('Data Imputation!')
    else:
        flag_di = False
        mask_data_imputation = np.ones(sci.shape, dtype='bool')
    
    # Ignore pixels that are False in either mask (ie., True in both masks)
    mask = mask_components & mask_data_imputation # will be used for modeling

    # Set values below threshold to 0
    sci[sci < sciThresh] = 0
    sci_err[sci==0] = np.nanmax(sci_err)

    # Mask invalid pixels
    mask[sci <= 0] = False
    mask[np.isnan(sci)] = False
    mask[~np.isfinite(sci)] = False
    
    # Columnize the target and its error and keep only the valid pixels
    sci_column = data_masked_only(sci, mask=mask)
    sci_err_column = data_masked_only(sci_err, mask=mask)
    weights = 1.0 / sci_err_column**2
    components_column = data_masked_only(components[:n_components].T, mask=mask)

    sci_img = NMF(sci_column, V=weights, W=components_column, 
                  n_components=n_components, rand_seed=rand_seed)
    if oneByOne:
        chi2, time_used = sci_img.SolveNMF_OneByOne(H_only=True, maxiters=maxiters, **kwargs)
    else:
        chi2, time_used = sci_img.SolveNMF(H_only=True, maxiters=maxiters, **kwargs)

    coefs = sci_img.H
    if flag_di: # do data imputation
        model_column = np.dot(components_column_all, coefs)
        model = data_masked_only_revert(model_column, mask_components)
        model[~mask_components] = np.nan
    else: # do not do data imputation
        model_column = np.dot(components_column, coefs)
        model = data_masked_only_revert(model_column, mask)
        model[~mask] = np.nan

    return model.flatten() #model_column.T.flatten()
    
def NMFsubtraction(sci, model, frac=1):
    """NMF subtraction with a correction factor, frac."""
    frac = np.atleast_1d(frac)

    result = []
    for i, fraction in enumerate(frac):
        result.append(sci - model*fraction)

    return np.array(result).squeeze()
    
def NMFbff(sci, model, fracs=None, **kwargs):
    """BFF subtraction.
    Args:
        sci:
        model:
        fracs: (if need to be).
    Returns: 
        best frac
    """

    from webbpsf_ext import robust
    
    if fracs is None:
        fracs = np.arange(0.50, 1.001, 0.001)

    # std_infos = []
    # for i, frac in enumerate(fracs):
    #     data_slice = sci - model*frac
    #     while 1:
    #         if np.nansum(data_slice > np.nanmedian(data_slice) + 3*np.nanstd(data_slice)) == 0 or np.nansum(data_slice < np.nanmedian(data_slice) -3*np.nanstd(data_slice)) == 0: # Modified from -10 on 2018/07/12
    #             break
    #         data_slice[data_slice > np.nanmedian(data_slice) + 3*np.nanstd(data_slice)] = np.nan
    #         data_slice[data_slice < np.nanmedian(data_slice) - 3*np.nanstd(data_slice)] = np.nan # Modified from -10 on 2018/07/12
    #     std_info = np.nanstd(data_slice)
    #     std_infos.append(std_info)
    # std_infos = np.array(std_infos)

    std_infos = np.array([robust.medabsdev(sci - model*frac) for frac in fracs])
    ind_min = np.where(std_infos == np.nanmin(std_infos))[0]
    return fracs[ind_min]   
   
def nmf_math(sci, ref_psfs, sci_err=None, ref_psfs_err=None, 
             ignore_mask=None, mask_data_imputation=None, sci_type='disk', 
             n_components=10, maxiters=5000, oneByOne=True, rand_seed=None,
             save=True, save_name=None, save_dir=None, recalculate=False, **kwargs):
    """ Main NMF function for high contrast imaging.

    Parameters
    ----------
    sci : 1D or 2D array
        Target image, dimension: height * width.

    Args:  
        sci (1D array): target image, dimension: height * width.
        refs (2D array): reference cube, dimension: referenceNumber * height * width.
        sci_err, ref_err: uncertainty for sci and refs, repectively. If None is given, the squareroot of the two arrays will be adopted.
    
        componentNum (integer): number of components to be used. Default: 5. Caution: choosing too many components will slow down the computation.
        maxiters (integer): number of iterations needed. Default: 10^5.
        oneByOne (boolean): whether to construct the NMF components one-by-one. Default: True.
        sci_type (string,  default: "disk" or "d" for circumsetllar disks by Bin Ren, the user can use "planet" or "p" for planets): are we aiming at finding circumstellar disks or planets?
    Returns: 
        result (1D array): NMF modeling result. Only the final subtraction result is returned.
    """

    rng = np.random.default_rng(seed=rand_seed)

    sh_sci = sci.shape
    sh_ref = ref_psfs.shape

    if len(sh_sci) == 2:
        sci = sci.ravel()
    if len(sh_ref) == 3:
        ref_psfs = ref_psfs.reshape(sh_ref[0], -1)

    # Total number of ref components and pixels
    ncomp_tot, npix = ref_psfs.shape

    # Ensure n_components is not greater than the number of references
    if n_components > ncomp_tot:
        _log.warning(f"Requested n_components={n_components} is greater than number of references ({ncomp_tot}). Setting n_components={ncomp_tot}.")
        n_components = ncomp_tot

    if sci_err is not None:
        sci_err = sci_err.reshape(sci.shape)
    if ref_psfs_err is not None:
        ref_psfs_err = ref_psfs_err.reshape(ref_psfs.shape)

    if (ignore_mask is not None):
        if ignore_mask.size == npix:
            ignore_mask = ignore_mask.ravel()
        elif ignore_mask.size == ncomp_tot * npix:
            ignore_mask = ignore_mask.reshape(ncomp_tot, npix)
        else:
            raise ValueError("ignore_mask must be same size as `sci` or `ref_psfs`.")
        
    if mask_data_imputation is not None:
        mask_data_imputation = mask_data_imputation.reshape(sci.shape)

    # Generate components
    kwargs_components = {
        'ref_err': ref_psfs_err, 'ignore_mask': ignore_mask, 
        'n_components': n_components, 'maxiters': maxiters, 'oneByOne': oneByOne, 
        'save': save, 'save_name': save_name, 'save_dir': save_dir, 
        'recalculate': recalculate, 'rand_seed' : rng.integers(0, 2**32-1),
    }
    kwargs_components.update(kwargs)
    components = NMFcomponents(ref_psfs, **kwargs_components)

    bp_mask = np.isnan(sci)
    sci[bp_mask] = 0

    # Generate PSF models for each component
    ncomp_arr = np.arange(n_components) + 1
    model_arr = []
    for ncomp in ncomp_arr:
        kwargs_model = {
            'sci_err': sci_err, 'mask_data_imputation': mask_data_imputation, 
            'n_components': ncomp, 'oneByOne': False, 
            'maxiters': maxiters, 'tol': 1e-8, 'verbose': False,
            'rand_seed' : rng.integers(0, 2**32-1),
        }
        kwargs_model.update(kwargs)
        model = NMFmodelling(sci, components, **kwargs_model)
        model_arr.append(model)
    model_arr = np.array(model_arr)

    if mask_data_imputation is None:
        if ("planet" in sci_type) or (sci_type=="p"):
            # best_frac = 1
            best_frac_arr = np.ones(len(model_arr))
        elif ("disk" in sci_type) or (sci_type=="d"):
            # Find best scale factor for subtraction
            # best_frac = NMFbff(sci, model)
            best_frac_arr = np.array([NMFbff(sci, model, **kwargs) for model in model_arr]).flatten()
        
        result_arr = sci - model_arr * best_frac_arr.reshape([-1,1])
    else:                                
        result_arr = sci - model_arr

    result_arr[:, bp_mask] = np.nan

    return result_arr


def build_nmf_dict(fwhm_pix, **kwargs):
    """ Build PCA parameter dictionary

    Returns a dictionary of settings to pass to various analysis functions.
    There are multiple keywords that will be set by default and can be modified 
    by passing via keyword arguments. 
    
    Example output:

    nmf_params = {
        'fwhm_pix'    : fwhm_pix, # Number of pixels equal to PSF FWHM
        'loci'        : True,   # Locally optimized regions?
        'numbasis'    : [1,5,10,50],  # Number of KL basis vectors to use

        'kwargs_sub' : {}, # Subtraction regions config
        'kwargs_opt' : {}, # Optimization regions config
    }


    # Subtraction regions config
    kwargs_sub = {
        'IWA_nfwhm' : 0,     # Inner working angle in units of FWHM
        'OWA_nfwhm' : None,  # Outer working angle in units of FWHM
        'sub_ann_rad'   : 3, # Radial width of annulus (units of FWHM)
        'sub_ann_width' : 3, # Arc length of annulus (units of FWHM)
        'annuli_spacing' : 'constant', # Constant, log, or linear
        'constant_theta' : False,      # Do we want constant angle for each ring?
    }    

    # Optimization regions config (only for LOCI)
    kwargs_opt = {
        'opt_ann_rad'   : None, # Radial width of annulus (units of FWHM)
        'opt_ann_width' : 4,    # Arc length of annulus (units of FWHM)
        'nfwhm_sep'   : 0,      # Separation between subtraction and optimization regions (units of FWHM)
        'exclude_sub' : True,   # Exclude the subtraction region from the optimization region?
    }

    Parameters
    ----------
    param_dict : dict
        Dictionary of parameters from the main reduction script.
    """


    # PCA Parameters
    pca_params = {
        'fwhm_pix' : fwhm_pix, # Number of pixels equal to PSF FWHM

        # 'bin' : 1,               # Image binning factor
        # 'navg_ints_sci' : None,  # Number of sci INTS to average together
        # 'navg_ints_ref' : None,  # Number of ref INTS to average together
        # 'scale_psfs'    : False, # Scale PSFs by flux before subtracting and combining?
        # 'sat_rad_asec'  : None,  # Saturation radius in arcsec

        # 'add_fake' : False,    # Add Fake Planets?

        'loci'        : True,  # Use LOCI instead of KLIP?
        'remove_mean' : False,  # Remove mean from each frame before PCA?
        'numpsfs_max' : 100,    # Maximum number of PSFs to use for decomposition
        'numbasis'    : [1,5,10,50],  # Number of KL basis vectors to use
        'svd_frac'    : 0.9, # Fraction of variance to describe with KL basis vectors
    }

    # Subtraction regions config
    kwargs_sub = {
        'IWA_nfwhm' : 0,
        'OWA_nfwhm' : None,
        'sub_ann_rad'   : 3, # In terms of FWHM
        'sub_ann_width' : 3, # In terms of FWHM
        'annuli_spacing' : 'constant',
        'constant_theta' : False,
    }
    for k in kwargs_sub.keys():
        kwargs_sub[k] = kwargs.pop(k, kwargs_sub[k])

    # Optimization regions config
    kwargs_opt = {
        'opt_ann_rad'   : None,
        'opt_ann_width' : 2,
        'nfwhm_sep'   : 0,
        'exclude_sub' : True,
    }
    for k in kwargs_opt.keys():
        kwargs_opt[k] = kwargs.pop(k, kwargs_opt[k])
    # Default the radius of the optimization region to be 1 FWHM larger than the subtraction region
    if kwargs_opt['opt_ann_rad'] is None:
        kwargs_opt['opt_ann_rad'] = kwargs_sub['sub_ann_rad']+1

    # Save to dictionary
    pca_params['kwargs_sub'] = kwargs_sub
    pca_params['kwargs_opt'] = kwargs_opt

    # Update defaults with any user inputs
    pca_params.update(**kwargs)

    return pca_params


def NMFcomponents_old(ref, ref_err = None, n_components = None, maxiters = 1e3, oneByOne = False, ignore_mask = None, path_save = None, recalculate = False):
    """Returns the NMF components, where the rows contain the information.
    Args:
        ref and ref_err should be (N, p) where N is the number of references, p is the number of pixels in each reference.
        ignore_mask: array of shape (N, p). mask pixels in each image that you don't want to use. 
        path_save: string, path to save the NMF components (at: path_save + '_comp.fits') and coeffieients (at: path_save + '_coef.fits')
        recalculate: boolean, whether to recalculate when path_save is provided
    Returns: 
        NMf components (n_components * p).
    """
    ref = ref.T # matrix transpose to comply with statistician standards on storing data
    
    if ref_err is None:
        ref_err = np.sqrt(ref)
    else:
        ref_err = ref_err.T # matrix transpose for the error map as well
        
    if (n_components is None) or (n_components > ref.shape[0]):
        n_components = ref.shape[0]
        
    if ignore_mask is None:
        ignore_mask = np.ones_like(ref)
    
    # ignore certain values in component construction
    ignore_mask[ref <= 0] = 0 # 1. negative values
    ignore_mask[~np.isfinite(ref)] = 0 # 2. infinite values
    ignore_mask[np.isnan(ref)] = 0 # 3. nan values
    
    ignore_mask[ref_err <= 0] = 0 # 1. negative values in input error map
    ignore_mask[~np.isfinite(ref_err)] = 0 # 2. infinite values in input error map
    ignore_mask[np.isnan(ref_err)] = 0 # 3. nan values in input error map
    
    # speed up component calculation by ignoring the commonly-ignored elements across all references
    mask_mark = np.nansum(ignore_mask, axis = 1)
    mask_mark[mask_mark != 0] = 1 # 1 means that there is coverage in at least one of the refs
    
    ref_columnized = data_masked_only(ref, mask = mask_mark)
    ref_err_columnized = data_masked_only(ref_err, mask = mask_mark)
    mask_columnized = data_masked_only(ignore_mask, mask = mask_mark)
    mask_columnized_boolean = np.array(data_masked_only(ignore_mask, mask = mask_mark), dtype = bool)
    ref_columnized[mask_columnized == 0] = 0 # assign 0 to ignored values, should not impact the final result given the usage of mask_columnized_boolean
    ref_err_columnized[mask_columnized == 0] = np.nanmax(ref_err_columnized) # assign max uncertainty to ignored values, should not impact the final result

    
    # component calculation
    components_column = 0
    if not oneByOne:
        g_img = NMF(ref_columnized, V=1.0/ref_err_columnized**2, M = mask_columnized_boolean, n_components=n_components)
        chi2, time_used = g_img.SolveNMF(maxiters=maxiters)
        components_column = g_img.W/np.sqrt(np.nansum(g_img.W**2, axis = 0)) #normalize the components        
        components = data_masked_only_revert(components_column, mask = mask_mark)        
    else:
        print("Building components one-by-one...")
        if path_save is None or recalculate:
            if recalculate:
                print('Recalculating no matter if you have saved previous ones.')
            for i in range(n_components):
                print("\t" + str(i+1) + " of " + str(n_components))
                n = i + 1
                if (i == 0):
                    g_img = NMF(ref_columnized, V = 1.0/ref_err_columnized**2, M = mask_columnized_boolean, n_components= n)
                else:
                    W_ini = np.random.rand(ref_columnized.shape[0], n)
                    W_ini[:, :(n-1)] = np.copy(g_img.W)
                    W_ini = np.array(W_ini, order = 'F') #Fortran ordering, column elements contiguous in memory.
                
                    H_ini = np.random.rand(n, ref_columnized.shape[1])
                    H_ini[:(n-1), :] = np.copy(g_img.H)
                    H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
                
                    g_img = NMF(ref_columnized, V = 1.0/ref_err_columnized**2, M = mask_columnized_boolean, W = W_ini, H = H_ini, n_components= n)
                chi2 = g_img.SolveNMF(maxiters=maxiters)
            
                components_column = g_img.W/np.sqrt(np.nansum(g_img.W**2, axis = 0)) #normalize the components
                components = data_masked_only_revert(components_column, mask = mask_mark) 
            if recalculate:
                print('\t\t\t Calculation for ' + str(n) + ' components done, overwriting raw 2D component matrix at ' + path_save + '_comp.fits')
                fits.writeto(path_save + '_comp.fits', g_img.W, overwrite = True)
                print('\t\t\t Calculation for ' + str(n) + ' components done, overwriting raw 2D coefficient matrix at ' + path_save + '_coef.fits')
                fits.writeto(path_save + '_coef.fits', g_img.H, overwrite = True)
        else:
            if not os.path.exists(path_save + '_comp.fits'):
                print('\t\t ' + path_save + '_comp.fits does not exist, calculating from scratch.')
                for i in range(n_components):
                    print("\t" + str(i+1) + " of " + str(n_components))
                    n = i + 1
                    if (i == 0):
                        g_img = NMF(ref_columnized, V = 1.0/ref_err_columnized**2, M = mask_columnized_boolean, n_components= n)
                    else:
                        W_ini = np.random.rand(ref_columnized.shape[0], n)
                        W_ini[:, :(n-1)] = np.copy(g_img.W)
                        W_ini = np.array(W_ini, order = 'F') #Fortran ordering, column elements contiguous in memory.
                
                        H_ini = np.random.rand(n, ref_columnized.shape[1])
                        H_ini[:(n-1), :] = np.copy(g_img.H)
                        H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
                
                        g_img = NMF(ref_columnized, V = 1.0/ref_err_columnized**2, M = mask_columnized_boolean, W = W_ini, H = H_ini, n_components= n)
                    chi2 = g_img.SolveNMF(maxiters=maxiters)
                    print('\t\t\t Calculation for ' + str(n) + ' components done, overwriting raw 2D component matrix at ' + path_save + '_comp.fits')
                    fits.writeto(path_save + '_comp.fits', g_img.W, overwrite = True)
                    print('\t\t\t Calculation for ' + str(n) + ' components done, overwriting raw 2D coefficient matrix at ' + path_save + '_coef.fits')
                    fits.writeto(path_save + '_coef.fits', g_img.H, overwrite = True)
                    components_column = g_img.W/np.sqrt(np.nansum(g_img.W**2, axis = 0)) #normalize the components
                    components = data_masked_only_revert(components_column, mask = mask_mark)
            else:
                W_assign = fits.getdata(path_save + '_comp.fits')
                H_assign = fits.getdata(path_save + '_coef.fits')
                if W_assign.shape[1] >= n_components:
                    print('You have already had ' + str(W_assign.shape[1]) + ' components while asking for ' + str(n_components) + '. Returning to your input.')
                    components_column = W_assign/np.sqrt(np.nansum(W_assign**2, axis = 0))
                    components = data_masked_only_revert(components_column, mask = mask_mark)
                else:
                    print('You are asking for ' + str(n_components) + ' components. Building the rest based on the ' + str(W_assign.shape[1]) + ' provided.')

                    for i in range(W_assign.shape[1], n_components):
                        print("\t" + str(i+1) + " of " + str(n_components))
                        n = i + 1
                        if (i == W_assign.shape[1]):
                            W_ini = np.random.rand(ref_columnized.shape[0], n)
                            W_ini[:, :(n-1)] = np.copy(W_assign)
                            W_ini = np.array(W_ini, order = 'F') #Fortran ordering, column elements contiguous in memory.
            
                            H_ini = np.random.rand(n, ref_columnized.shape[1])
                            H_ini[:(n-1), :] = np.copy(H_assign)
                            H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
            
                            g_img = NMF(ref_columnized, V = 1.0/ref_err_columnized**2, W = W_ini, H = H_ini, M = mask_columnized_boolean, n_components= n)
                        else:
                            W_ini = np.random.rand(ref_columnized.shape[0], n)
                            W_ini[:, :(n-1)] = np.copy(g_img.W)
                            W_ini = np.array(W_ini, order = 'F') #Fortran ordering, column elements contiguous in memory.
            
                            H_ini = np.random.rand(n, ref_columnized.shape[1])
                            H_ini[:(n-1), :] = np.copy(g_img.H)
                            H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
            
                            g_img = NMF(ref_columnized, V = 1.0/ref_err_columnized**2, W = W_ini, H = H_ini, M = mask_columnized_boolean, n_components= n)
                        chi2 = g_img.SolveNMF(maxiters=maxiters)
                        print('\t\t\t Calculation for ' + str(n) + ' components done, overwriting raw 2D component matrix at ' + path_save + '_comp.fits')
                        fits.writeto(path_save + '_comp.fits', g_img.W, overwrite = True)
                        print('\t\t\t Calculation for ' + str(n) + ' components done, overwriting raw 2D coefficient matrix at ' + path_save + '_coef.fits')
                        fits.writeto(path_save + '_coef.fits', g_img.H, overwrite = True)
                        components_column = g_img.W/np.sqrt(np.nansum(g_img.W**2, axis = 0)) #normalize the components
                        components = data_masked_only_revert(components_column, mask = mask_mark)            
    return components.T


def NMFmodelling_old(trg, components, n_components = None, mask_components = None, mask_data_imputation = None, trg_err = None, maxiters = 1e3, cube = False, trgThresh = 0):
    """ NMF modeling.
    Args:
        trg: 1D array, p pixels
        components: N * p, calculated using NMFcomponents.
        n_components: how many components do you want to use. If None, all the components will be used.
        cube: whether output a cube or not (increasing the number of components).
        trgThresh: ignore the regions with low photon counts. Especially when they are ~10^-15 or smaller. I chose 0 in this case.
    
    Returns:
        NMF model of the target.
    """

    
    if n_components is None:
        n_components = components.shape[0]
        
    if trg_err is None:
        trg_err = np.sqrt(trg)

    if mask_components is None:
        mask_components = np.ones(trg.shape)
        mask_components[np.where(np.isnan(components[0]))] = 0

    components_column_all = data_masked_only(components[:n_components].T, mask = mask_components)   #columnize the components, make sure NonnegMFPy returns correct results.
    components_column_all = components_column_all/np.sqrt(np.nansum(components_column_all**2, axis = 0)) #normalize the components #make sure the components are normalized.
    
    if mask_data_imputation is None:
        flag_di = 0
        mask_data_imputation = np.ones(trg.shape)
    else:
        flag_di = 1
        print('Data Imputation!')
        
    mask = mask_components*mask_data_imputation #will be used for modeling

    trg[trg < trgThresh] = 0
    trg_err[trg == 0] = np.nanmax(trg_err)

    mask[trg <= 0] = 0
    mask[np.isnan(trg)] = 0
    mask[~np.isfinite(trg)] = 0
    
    #Columnize the target and its error.
    trg_column = data_masked_only(trg, mask = mask)
    trg_err_column = data_masked_only(trg_err, mask = mask)
    components_column = data_masked_only(components.T, mask = mask)

    if not cube:
        trg_img = NMF(trg_column, V=1/trg_err_column**2, W=components_column, n_components = n_components)
        (chi2, time_used) = trg_img.SolveNMF(H_only=True, maxiters = maxiters)
        coefs = trg_img.H
        if flag_di == 0: # do not do data imputation
            model_column = np.dot(components_column, coefs)

            model = data_masked_only_revert(model_column, mask)
            model[np.where(mask == 0)] = np.nan
        elif flag_di == 1: # do data imputation
            model_column = np.dot(components_column_all, coefs)
            model = data_masked_only_revert(model_column, mask_components)
            model[np.where(mask_components == 0)] = np.nan
    else:
        print("Building models one-by-one...")

        for i in range(n_components):
            print("\t" + str(i+1) + " of " + str(n_components))
            trg_img = NMF(trg_column, V=1/trg_err_column**2, W=components_column[:, :i+1], n_components = i + 1)
            (chi2, time_used) = trg_img.SolveNMF(H_only=True, maxiters = maxiters)

            coefs = trg_img.H

            model_column = np.dot(components_column[:, :i+1], coefs)

    return model.flatten() #model_column.T.flatten()