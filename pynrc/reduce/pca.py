#!/usr/bin/env python

# Import the usual libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os, pickle
from astropy.io import fits

from webbpsf_ext.image_manip import frebin, pad_or_cut_to_size, rotate_offset
from webbpsf_ext.image_manip import fshift, fourier_imshift
from webbpsf_ext.maths import round_int, hist_indices, binned_statistic
from webbpsf_ext.coords import dist_image
from webbpsf_ext import robust

from tqdm import trange, tqdm

import logging
# Define logging
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


#################################################################
# KLIP and LOCI PSF Subtraction functions
#################################################################

def mkpsf_klip(bg_opt, sci_opt, nsvd=None, svd_frac=0.9):
    """ Create a PSF using KLIP PSF subtraction

    Parameters
    ----------
    bg_opt : ndarray
        Optimization region of reference observations ("background").
    sci_opt : ndarray
        Optimization region of science observations.
    nsvd : int, list, or ndarray
        Number of SVD modes to keep. If None, then the number of modes
        is determined by the fraction of cumulative eigenvalues.
    svd_frac : float
        Fraction of cumulative eigenvalues to keep. Default is 0.9.
    """
    
    try:
        _, s, VT = np.linalg.svd(bg_opt, full_matrices=False)
    except Exception as e:
        _log.info(f"Unexpected {e=}, {type(e)=}")
        _log.info(bg_opt)
        raise e
    
    # Eigenvalues
    if nsvd is None:
        # Eigenvalues
        evals = s**2
        evals /= evals.sum()
        evals_csum = evals.cumsum()

        # try:
        emin = evals_csum.min()
        # except ValueError:
        #     print(bg_opt.shape, evals)
        #     raise ValueError

        # Cut off at 90% cumulative power
        eval_cut = emin + svd_frac * (1.0 - emin) 
        ind = np.where(evals_csum < eval_cut)
        Z = VT[ind[0],:]
        
        res = np.dot(sci_opt, np.dot(Z.T,Z)) 
    else:
        nsvd = np.array([nsvd]).flatten()
        nres = len(nsvd)
        res = []
        for i in range(nres):
            Z = VT[0:nsvd[i],:]
            res.append(np.dot(sci_opt, np.dot(Z.T,Z)) )
        res = np.array(res)
    
    return res

def fit_loci_coeff(bg_opt, sci_opt, use_QR=True, use_old=False, 
                   nsvd=None, svd_frac=0.9, **kwargs):
    """Fit coefficients in optimization region for LOCI PSF subtraction
    
    Parameters
    ----------
    bg_opt : array_like
        Background optimization region.
    sci_opt : array_like
        Science optimization region (single vector size of bg_opt.shape[1]).
    use_QR : bool, optional
        Use QR decomposition to solve for coefficients. Only works for
        nsvd=None.
    use_old : bool, optional
        Use old method of solving for coefficients (not recommended).
        Only applicable for nsvd=None.
    nsvd : None, int, or array-like, optional
        Number of singular values to use in SVD. If None, then the number
        of singular values is chosen to be the minimum number that account 
        for svd_frac of the variance in the data.
    svd_frac : float, optional
        Fraction of variance to account for in SVD. Ignored if nsvd is not
        None.
    """
    
    import numpy.ma as ma

    # Deal with NaNs in bg array?
    inan_bg = np.isnan(bg_opt)
    if np.any(inan_bg):
        use_masked_bg = True
        bg_opt = ma.masked_invalid(bg_opt)
    else:
        use_masked_bg = False

    # Deal with NaNs in sci array?
    inan_sci = np.isnan(sci_opt)
    if np.any(inan_sci):
        use_masked_sci = True
        sci_opt = ma.masked_invalid(sci_opt)
    else:
        use_masked_sci = False

    nopt = bg_opt.shape[1]
    # Covariance Matrix
    try:
        A = ma.dot(bg_opt, bg_opt.T).data if use_masked_bg else np.dot(bg_opt, bg_opt.T)
        A /= (nopt-1.0)
    except:
        A = ma.cov(bg_opt).data if use_masked_bg else np.cov(bg_opt)

    # Data vector
    b_use_masked = use_masked_bg or use_masked_sci
    b = ma.dot(bg_opt, sci_opt.T).data if b_use_masked else np.dot(bg_opt, sci_opt.T)

    # Original SVD cut-off version
    if nsvd is not None:
        try:
            U,s,VT = np.linalg.svd(A, full_matrices=True)
        except Exception as e:
            print(f"Unexpected {e=}, {type(e)=}")
            print(A)
            raise e

        nsvd = np.array([nsvd]).flatten()
        nres = len(nsvd)
        coeff_all = []
        for ns in nsvd:
            # Pseudo-invert
            sinv = np.zeros_like(s)
            sinv[0:ns] = 1 / s[0:ns]
            Sinv = np.diag(sinv)
            Ainv = np.dot(VT.T, np.dot(Sinv, U.T)) / (nopt - 1.0)

            coeff = np.dot(Ainv, b)
            coeff_all.append(coeff)
        
        if nres == 1:
            coeff = coeff_all[0]
        else:
            coeff = np.array(coeff_all)

    elif use_old:
        try:
            U,s,VT = np.linalg.svd(A, full_matrices=True)
        except Exception as e:
            print(f"Unexpected {e=}, {type(e)=}")
            print(A)
            raise e
        
        # Eigenvalues
        evals = s**2
        evals /= evals.sum()
        evals_csum = evals.cumsum()
        emin = evals_csum.min()

        # Cut off at 90% cumulative power
        eval_cut = emin + svd_frac * (1.0 - emin) 
        ind_s = np.where(evals_csum < eval_cut)[0]

        # Pseudo-inversion of A matrix
        # with SVD cut-off
        sinv = np.zeros_like(s)
        sinv[ind_s] = 1 / s[ind_s]
        Sinv = np.diag(sinv)
        Ainv = np.dot(VT.T, np.dot(Sinv, U.T)) / (nopt - 1.0)

        # Coefficients fit to psf_lib that best fits science data
        # witin the optimization region
        coeff = np.dot(Ainv, b)

    elif use_QR:
        # QR decomposition of A matrix
        q, r = np.linalg.qr(A, 'reduced')
        # computing Q^T*b (project b onto the range of A)
        qTb = np.matmul(q.T, b)
        # solving R*x = Q^T*b
        res = np.linalg.lstsq(r, qTb, rcond=1-svd_frac)
        coeff = res[0]
    else:
        res = np.linalg.lstsq(A.T, b, rcond=1-svd_frac)
        coeff = res[0]

    # if nsvd is an array, returns array of coefficients
    return coeff

def mkpsf_loci(bg_opt, sci_opt, bg_sub, nsvd=None, svd_frac=0.9, **kwargs):
    """
    Input an array of nsvd to test variety of PCAs.
    Use svd_frac to perform a fraction power cut-off,
    which automatically selects some number of PCAs.

    Parameters
    ----------
    bg_opt : array_like
        Optimization region of reference observations.
    sci_opt : array_like
        Optimization region of science observations (single vector size of bg_opt.shape[1]).
    bg_sub : array_like
        Subtraction region of reference observations.
    """

    import numpy.ma as ma

    coeff = fit_loci_coeff(bg_opt, sci_opt, nsvd=nsvd, svd_frac=svd_frac, **kwargs)

    # Deal with NaNs
    inan_bg = np.isnan(bg_sub)
    if np.any(inan_bg):
        use_masked = True
        bg_sub = ma.masked_invalid(bg_sub)
    else:
        use_masked = False

    # These are mathematically equivalent
    # res = np.array([np.matmul(cf.T, bg_sub) for cf in coeff])
    # res = np.matmul(coeff, bg_sub)

    # If multiple dimensions, then coeff is a matrix
    cf_vals = coeff if (nsvd is not None) and (len(nsvd) > 1) else coeff.T

    # For 2D arrays, np.dot and np.matmul are equivalent
    res = ma.dot(cf_vals, bg_sub).data if use_masked else np.matmul(cf_vals, bg_sub)

    # if np.isnan(res).any():
    #     print(res.shape, res.size, np.isnan(res).sum())

    return res

def klip_subtract(sci_opt, bg_opt, numbasis=None, svd_frac=0.9,
                  remove_mean=False):
    """ Perform KLIP subtracton a sub-region of the data

    Parameters
    ----------
    sci_opt : array_like
        Science data to be subtracted
    bg_opt : array_like
        Background data to be used for PSF subtraction
    numbasis : int, optional
        Number of KL basis vectors to use
    svd_frac : float, optional
        Fraction of total variance to use for SVD cut-off
    remove_mean : bool, optional
        Remove the mean of the data before performing KLIP
    """

    # Any NaNs in the data?
    inan_bg = np.isnan(bg_opt)
    inan_sci = np.isnan(sci_opt)

    isnan_bg = np.any(inan_bg)
    isnan_sci = np.any(inan_sci)

    if remove_mean:
        # Use nanmean to ignore NaNs
        fmn_bg = np.nanmean if isnan_bg else np.mean
        fmn_sci = np.nanmean if isnan_sci else np.mean

        # Subtract mean of each frame data
        bg_mean_sub = bg_opt - fmn_bg(bg_opt, axis=1)[:, None]
        sci_mean_sub = sci_opt - fmn_sci(sci_opt)

        # Replace NaNs with zeros?
        if isnan_bg:
            bg_mean_sub[inan_bg] = 0
        if isnan_sci:
            sci_mean_sub[inan_sci] = 0
    else:
        # Replace NaNs with zeros?
        bg_mean_sub = bg_opt if not isnan_bg else np.nan_to_num(bg_opt)
        sci_mean_sub = sci_opt if not isnan_sci else np.nan_to_num(sci_opt)
    
    kpsf = mkpsf_klip(bg_mean_sub, sci_mean_sub, nsvd=numbasis, svd_frac=svd_frac)
    res = sci_mean_sub - kpsf
    
    return res

def loci_subtract(sci_opt, bg_opt, bg_sub, sci_sub, 
                  numbasis=None, svd_frac=0.9, remove_mean=False, **kwargs):
    """ Perform LOCI subtracton a sub-region of the data
    
    Parameters
    ----------
    sci_opt : array_like
        Science data within optimization region
    bg_opt : array_like
        Background (PSF) data frames within optimization region
    bg_sub : array_like
        Background (PSF) data frames within subtraction region
    sci_sub : array_like
        Science data within subtraction region
    numbasis : int, optional
        Number of KL basis vectors to use
    svd_frac : float, optional
        Fraction of total variance to use for SVD cut-off
    remove_mean : bool, optional
        Remove the mean of the data before performing fits

    Keyword Args
    ------------
    use_old : bool, optional
        Use older manual method of computing coefficients
    use_QR : bool, optional
        Use QR decomposition to compute coefficients
    """
        
    if remove_mean:
        # Any NaNs in the data?
        isnan_bg = np.any(np.isnan(bg_opt))
        isnan_sci = np.any(np.isnan(sci_opt))
        # Use nanmean to ignore NaNs
        fmn_bg = np.nanmean if isnan_bg else np.mean
        fmn_sci = np.nanmean if isnan_sci else np.mean

        bg_opt_mean = fmn_bg(bg_opt, axis=1)[:, None]
        sci_opt_mean = fmn_sci(sci_opt)

        bg_mean_opt  = bg_opt  - bg_opt_mean
        bg_mean_sub  = bg_sub  - bg_opt_mean
        sci_mean_opt = sci_opt - sci_opt_mean
        sci_mean_sub = sci_sub - sci_opt_mean
    else:
        bg_mean_opt  = bg_opt  
        bg_mean_sub  = bg_sub  
        sci_mean_opt = sci_opt 
        sci_mean_sub = sci_sub   

    # Create a PSF at the science subtraction region
    lpsf = mkpsf_loci(bg_mean_opt, sci_mean_opt, bg_mean_sub, 
                      nsvd=numbasis, svd_frac=svd_frac, **kwargs)
    
    return sci_mean_sub - lpsf

#################################################################
# Functions for masking optimization and subtraction regions
#################################################################

def define_annuli_bounds(annuli, IWA, OWA, annuli_spacing='constant'):
    """
    Defines the annuli boundaries radially

    Parameters
    ==========
    annuli: int
        number of annuli
    IWA: int 
        inner working angle (pixels)
    OWA: int 
        outer working angle (pixels)
    annuli_spacing: str
        how to distribute the annuli radially. Currently three options. 
        'constant' : (equally spaced),
        'log'      : (logarithmical expansion with r)
        'linear'   : (linearly expansion with r)

    Returns
    =======
    array of 2-element tuples that specify the beginning and end radius of that annulus

    """
    #calculate the annuli ranges
    if annuli_spacing.lower() == "constant":
        dr = float(OWA - IWA) / annuli
        rad_bounds = [(dr * rad + IWA, dr * (rad + 1) + IWA) for rad in range(annuli)]
    elif annuli_spacing.lower() == "log":
        # calculate normalization of log scaling
        unnormalized_log_scaling = np.log(np.arange(annuli) + 1) + 1
        log_coeff = float(OWA - IWA)/np.sum(unnormalized_log_scaling)
        # construct the radial spacing
        rad_bounds = []
        for i in range(annuli):
            # lower bound is either mask or end of previous annulus
            if i == 0:
                lower_bound = IWA
            else:
                lower_bound = rad_bounds[-1][1]
            upper_bound = lower_bound + log_coeff * unnormalized_log_scaling[i]
            rad_bounds.append((lower_bound, upper_bound))
    elif annuli_spacing.lower() == "linear":
        # scale linaer scaling to OWA-IWA
        linear_coeff = float(OWA - IWA) / np.sum(np.arange(annuli) + 1)
        dr = linear_coeff * (np.arange(annuli) + 1)
        bins = np.append([IWA], np.cumsum(dr) + IWA)
        r1_arr = bins[0:-1]
        r2_arr = bins[1:]
        rad_bounds = [(r1,r2) for r1, r2 in zip(r1_arr, r2_arr)]
    else:
        raise ValueError("annuli_spacing currently only supports 'constant', 'log', or 'linear'")

    # check to make sure the annuli are all greater than 1 pixel
    min_width = np.min(np.diff(rad_bounds, axis=1))
    if min_width < 1:
        raise ValueError("Too many annuli, some annuli are less than 1 pixel")

    return rad_bounds

def define_theta_bounds(subsections, degrees=True):
    """ Split angles between [-180,+180] into equal sizes"""
    
    dphi = 2 * np.pi / subsections
    phi_bounds = np.array([[dphi * phi_i - np.pi, dphi * (phi_i + 1) - np.pi] for phi_i in range(subsections)])
    phi_bounds[-1][1] = np.pi

    if degrees:
        return np.rad2deg(phi_bounds)
    else:
        return phi_bounds

def get_sub_regions(im, fwhm_pix, IWA_nfwhm=1, OWA_nfwhm=None,
                    sub_ann_rad=1, sub_ann_width=1, 
                    annuli_spacing='constant', constant_theta=False):
    """ Create an image of subtraction regions

    Parameters
    ==========
    im : ndarray
        Image template to get size
    fwhm_pix : float
        Number of pixels equal to PSF FWHM
    IWA_nfwhm : float
        Inner working angle in units of PSF FWHM
    OWA_nfwhm : float
        Outer working angle in units of PSF FWHM
    sub_ann_rad : float
        Radial width of annulus (units of fwhm)
    sub_ann_width : float
        Arc length of annulus (units of fwhm)
    annuli_spacing: str
        how to distribute the annuli radially. Currently three options. 
        'constant' : (equally spaced),
        'log'      : (logarithmical expansion with r)
        'linear'   : (linearly expansion with r)
    constant_theta : bool
        Do we want constant angle for each ring?
        Otherwise, the arc length (number of pixels) stays the same.
    """
    
    rho_im, th_im = dist_image(im, return_theta=True)
    
    # Determine outer radius of mask
    ny, nx = np.shape(im)
    OWA_nfwhm_max = np.sqrt((nx/2)**2 + (ny/2)**2) / fwhm_pix
    if (OWA_nfwhm is None) or (OWA_nfwhm > OWA_nfwhm_max):
        OWA_nfwhm = OWA_nfwhm_max
        
    assert IWA_nfwhm<OWA_nfwhm, "IWA_nfwhm must be less than OWA_nfwhm"
    
    # How many annuli to generate?
    n_annuli = round_int((OWA_nfwhm - IWA_nfwhm) / sub_ann_rad  + 0.5)
    dr_pix = sub_ann_rad * fwhm_pix
    
    # Generate list of radial bounds
    IWA_pix = IWA_nfwhm * fwhm_pix
    OWA_pix = IWA_pix + n_annuli*dr_pix
    rad_bounds = define_annuli_bounds(n_annuli, IWA_pix, OWA_pix, 
                                      annuli_spacing=annuli_spacing)
    
    # Either constant theta or constant pixel size
    if constant_theta:
        # Base number of theta sections on first radius and nfwhm_ann_width
        pix_dist = np.mean(rad_bounds[0])
        circ_npix = 2*np.pi*pix_dist
        nsect = int(circ_npix / (sub_ann_width*fwhm_pix))
        
        th_bounds = define_theta_bounds(nsect, degrees=True)
        # Repeat for each annulus
        th_bounds_all = np.tile(th_bounds, n_annuli).reshape([-1,n_annuli,2]).transpose([1,0,2])
    else:
        th_bounds_all = []
        for rvals in rad_bounds:
            pix_dist = np.mean(rvals)
            circ_npix = 2*np.pi*pix_dist
            nsect = int(circ_npix / (sub_ann_width*fwhm_pix))
            
            th_bounds = define_theta_bounds(nsect, degrees=True)
            th_bounds_all.append(th_bounds)
    
    # Create region mask
    region_mask = np.zeros_like(rho_im).astype('int')
    ii = 1
    for jj in range(n_annuli):
        r1, r2 = rad_bounds[jj]
        # print(r1,r2,r2-r1)
        th_bounds = th_bounds_all[jj]
        for th1, th2 in th_bounds:
            ind = (rho_im>=r1) & (rho_im<=r2) & (th_im>=th1) & (th_im<=th2)
            region_mask[ind] = ii
            ii += 1

    # Ensure consecutive integers
    region_mask_fin = np.zeros_like(region_mask)
    iuniq = np.unique(region_mask)
    # Exclude 0s
    iuniq = iuniq[iuniq>0]
    for ii, iu in enumerate(iuniq):
        region_mask_fin[region_mask==iu] = ii+1

    return region_mask_fin
        
def get_opt_regions(region_mask, fwhm_pix, opt_ann_rad=None, 
                    opt_ann_width=0.75, nfwhm_sep=0, exclude_sub=True):
    """ Create array of optimization region masks for each subtraction region

    For LOKI style subtractions, we utilize optimization regions that are
    adjacent to the subtraction region. This function creates a series of
    image masks definining those optimization regions for each subtraction
    region.

    Parameters
    ==========
    region_mask : image
        Input of regions generated with `get_sub_regions`.
        Values set to 0 will be ignored.
    fwhm_pix : float
        Number of pixels equal to PSF FWHM
    opt_ann_rad : float
        Radial width of annulus (units of fwhm)
        If set to None, then will match region_mask thicknesses
    opt_ann_width : float
        Arc length of annulus (units of fwhm)
    nfwhm_sep : float
        Separation between subtraction and optimization regions (units of fwhm)
    exclude_sub : bool
        Do we want to exclude the subtraction region from the optimization
        region?
    """
    
    rho_im, th_im = dist_image(region_mask, return_theta=True)
    # Used for indexing thetas <-180 or >+180
    th_im_m180 = th_im.copy()
    th_im_m180[th_im>0] = th_im[th_im>0] - 360
    th_im_p180 = th_im.copy()
    th_im_p180[th_im<0] = th_im[th_im<0] + 360
        
    mask_vals = np.unique(region_mask)
    # Exclude 0
    mask_vals = mask_vals[mask_vals>0]
    masks_opt = []
    for val in mask_vals:
        ind_sub = (region_mask == val)
        # Get center location
        rho_mn = 0.5 * (rho_im[ind_sub].max() + rho_im[ind_sub].min())
        
        # Create annulus mask
        if opt_ann_rad is None:
            rho_min, rho_max = rho_im[ind_sub].min(), rho_im[ind_sub].max()
        else:
            rho_min, rho_max = rho_mn + np.array([-1,1])*opt_ann_rad*fwhm_pix/2
        annulus_mask = (rho_im >= rho_min) & (rho_im < rho_max)

        # th_im goes from -180 to +180
        # Mask regions do not span that the division
        th_ind = th_im[ind_sub]
        
        # Record edges of subtraction region
        th_lims = np.array([th_ind.min(), th_ind.max()])
        th_mn = np.mean(th_lims)
        
        # Get regions on either side of subtraction region
        # These are the start locations
        offset_start = ((nfwhm_sep*fwhm_pix) / (2*np.pi*rho_mn)) * 360
        offset_end   = ((opt_ann_width*fwhm_pix) / (2*np.pi*rho_mn)) * 360
        # Start locations for two sides of symmetric optimization region
        th_opt_start = th_mn + np.array([-1,1]) * offset_start
        # Minor offsets to make sure opt widths go beyond subtraction region
        th_opt_start2 = th_lims + np.array([-1,1]) * offset_start
        th_opt_end    = th_opt_start2 + np.array([-1,1]) * offset_end
        
        th_ims = [th_im, th_im_m180, th_im_p180]

        # First region
        th1, th2 = th_opt_start[0], th_opt_end[0]
        if th1<=-180 or th2<=-180:
            th_ind = 1
        elif th1>=180 or th2>=180:
            th_ind = 2
        else:
            th_ind = 0

        theta_mask1 = (th_ims[th_ind]<=th1) & (th_ims[th_ind]>=th2)

        # Second region
        th1, th2 = th_opt_start[1], th_opt_end[1]
        if th1<=-180 or th2<=-180:
            th_ind = 1
        elif th1>=180 or th2>=180:
            th_ind = 2
        else:
            th_ind = 0

        theta_mask2 = (th_ims[th_ind]>=th1) & (th_ims[th_ind]<=th2)
        
        # Final optimization region mask
        ind_opt = (theta_mask1 | theta_mask2) & annulus_mask

        # Exclude anything in original ind_sub?
        if exclude_sub:
            ind_opt = ind_opt & ~ind_sub
        
        masks_opt.append(ind_opt)
    
    masks_opt = np.array(masks_opt)
    
    return masks_opt

def create_region_masks(shape, fwhm_pix, kw_sub={}, kw_opt={}):

    # Subtraction and optimization region masks
    kwargs_sub = {
        'IWA_nfwhm' : kw_sub.get('IWA_nfwhm', 0.0),
        'OWA_nfwhm' : kw_sub.get('OWA_nfwhm', None),
        'sub_ann_rad'   : kw_sub.get('sub_ann_rad', 2.0),
        'sub_ann_width' : kw_sub.get('sub_ann_width', 2.0),
        'annuli_spacing' : kw_sub.get('annuli_spacing', 'constant'),
        'constant_theta' : kw_sub.get('constant_theta', False),
    }
    kwargs_opt = {
        'opt_ann_rad'   : kw_opt.get('opt_ann_rad'),
        'opt_ann_width' : kw_opt.get('opt_ann_width', 1),
        'nfwhm_sep'   : kw_opt.get('nfwhm_sep', 0),
        'exclude_sub' : kw_opt.get('exclude_sub', True),
    }
    if kwargs_opt['opt_ann_rad'] is None:
        kwargs_opt['opt_ann_rad'] = kwargs_sub['sub_ann_rad']+1

    # Create subtraction and optimization regions (same)
    sub_masks = get_sub_regions(np.zeros(shape), fwhm_pix, **kwargs_sub)
    opt_masks = get_opt_regions(sub_masks, fwhm_pix, **kwargs_opt)

    return sub_masks, opt_masks


def plot_masks(sub_masks, opt_masks=None, omask_index=None, pixscale=None,
               save_name=None, save_dir=None, return_figax=False, 
               verbose=True, **kwargs):
    """Plot the subtraction and optimization masks

    Parameters
    ----------
    sub_masks : numpy.ndarray
        Image array of the subtraction regions
    opt_masks : numpy.ndarray, optional
        Image array of the optimization regions. If set to None,
        then the optimization regions will not be plotted.
    omask_index : int, optional
        Index of the optimization region to plot. By default, will
        choose the middle region.
    pixscale : float, optional
        Pixel scale of the image. If set to None, then the axes
        will be in units of pixels.
    save_name : str, optional
        Name of the file to save the plot. Default is None
        means that the plot will not be saved.
    save_dir : str, optional
        Directory to save the plot. Default is None, which
        means that the current working directory will be used.
    return_figax : bool, optional
        If True, return the figure and axes objects. Default is False.
    verbose : bool, optional
        If True, print out information about the saved file. Default is True.
    """

    if opt_masks is None:
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1,2,figsize=(10,5))

    if verbose:
        nmasks = len(np.unique(sub_masks[sub_masks>0]))
        print(f"Number of masks: {nmasks}")

    ny, nx = sub_masks.shape

    extent = np.array([-nx,nx,-ny,ny]) / 2
    if pixscale is not None:
        extent *= pixscale
        xlabel = ylabel = 'arcsec'
    else:
        xlabel = ylabel = 'pixels'

    axes[0].imshow(sub_masks, extent=extent)
    axes[0].set_title('Subtraction Regions')

    # Are we also plotting optimization regions?
    if opt_masks is not None:
        omask_index = int(len(opt_masks)/2) if omask_index is None else omask_index
        axes[1].imshow(opt_masks[omask_index], extent=extent)
        axes[1].set_title('Optimization Region Example')

    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')

        ax.tick_params(axis='both', color='w', which='both')
        for k in ax.spines.keys():
            ax.spines[k].set_color('w')

    fig.tight_layout()

    if save_name is not None:
        if save_dir is None:
            save_dir = os.getcwd()
        save_path = os.path.join(save_dir, save_name)
        if verbose:
            print(f"Saving: {save_path}")
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    if return_figax:
        return fig, axes


def plot_mask_images(subsize, fwhm_pix, bin=1, do_loci=True,
                     kwargs_sub={}, kwargs_opt={}, **kwargs):
    """ Plot subtraction and optimization regions
    
    Parameters
    ----------
    pca_params : dict

    Keyword Args
    ------------
    omask_index : int, optional
        Index of the optimization region to plot. By default, will
        choose the middle region.
    pixscale : float, optional
        Pixel scale of the image. If set to None, then the axes
        will be in units of pixels.
    save_name : str, optional
        Name of the file to save the plot. Default is None
        means that the plot will not be saved.
    save_dir : str, optional
        Directory to save the plot. Default is None, which
        means that the current working directory will be used.
    return_figax : bool, optional
        If True, return the figure and axes objects. Default is False.
    verbose : bool, optional
        If True, print out information about the saved file. Default is True.
    """

    # Create subtraction and optimization regions (same)
    nx = ny = int(subsize / bin)
    im = np.zeros((ny,nx))
    sub_masks = get_sub_regions(im, fwhm_pix, **kwargs_sub)

    # Only produce optimization regions if LOCI
    plot_opt = do_loci
    if plot_opt:
        opt_masks = get_opt_regions(sub_masks, fwhm_pix, **kwargs_opt)
    else:
        opt_masks = None

    # Plot masks
    return plot_masks(sub_masks, opt_masks=opt_masks, **kwargs)


def corr_all(image, psf_arr, bp_image=None, bp_psf=None, sub_size=32, inner_rad=0):
    """ Get array of corr coeff for image to PSFs """

    from webbpsf_ext.image_manip import crop_image
    from webbpsf_ext.imreg_tools import correl_images

    # Create mask to exclude central region
    if inner_rad > 0:
        rho = dist_image(np.zeros((sub_size, sub_size)))
        mask_good = rho > inner_rad
    else:
        mask_good = np.ones((sub_size, sub_size), dtype=bool)

    if bp_image is not None:
        mask_good = mask_good & ~bp_image
    if bp_psf is not None:
        if len(bp_psf.shape) == 3:
            bp_psf = np.logical_and.reduce(bp_psf, axis=0)
        mask_good = mask_good & ~bp_psf

    # Cropped versions
    imsub = crop_image(image, sub_size)
    psfsub = crop_image(psf_arr, sub_size)
    # Get correlation matrix of each original PSF to each other
    return correl_images(imsub, psfsub, mask=mask_good)


def pca_subtraction(imarr, psfarr, fwhm_pix, do_loci=True, 
                    npsf_max=None, numbasis=None, svd_frac=0.9, remove_mean=False,
                    more_progress=False, kw_sub={}, kw_opt={}, **kwargs):
    """ Perform PCA subtraction on a set of images
    
    Parameters
    ----------
    imarr : array_like
        Image cube to be subtracted
    psfarr : array_like
        PSF library to be used for subtraction
    fwhm_pix : float
        Number of pixels equal to PSF FWHM
    do_loci : bool
        Use LOCI subtraction instead of KLIP
    npsf_max : int, optional
        Maximum number of PSFs to use in library to generate PCA basis
    numbasis : int or array_like, optional
        Number of PCA basis vectors to use for generating final PSF. 
        If set to None, then the number of singular values is chosen to be 
        the minimum number that account for svd_frac of the variance in the data. 
        If set to an integer, then will use that number of basis vectors. 
        If set to an array, then will loop through each value.
    svd_frac : float, optional
        Fraction of total variance to use for SVD cut-off
    remove_mean : bool, optional
        Remove the mean of the data and PSFs prior to generating PCA basis
        and performing fits.
    more_progress : bool, optional
        Show more detailed progress bar
    kw_sub : dict, optional
        Keyword arguments for creating subtraction region masks
    kw_opt : dict, optional
        Keyword arguments for creating optimization region masks

    """

    shape_orig = imarr.shape
    ndim = len(shape_orig)
    if ndim==3:
        nint, ny, nx = shape_orig
    elif ndim==2:
        nint, ny, nx = (1, shape_orig[0], shape_orig[1])
        imarr = imarr.reshape([nint,ny,nx])

    psf_shape = psfarr.shape
    npsf = psf_shape[0]
    if len(psf_shape) != 3 or npsf==1:
        raise ValueError(f"PSF array must be image cube and have multiple frames {psf_shape}")
    
    do_klip = not do_loci

    # Create subtraction and optimization region masks
    sub_masks, opt_masks = create_region_masks(imarr[0].shape, fwhm_pix, kw_sub=kw_sub, kw_opt=kw_opt)
    if do_klip:
        opt_masks = None
    # Number of sub region masks
    nmasks = len(np.unique(sub_masks[sub_masks>0]))

    # print(numbasis, npsf_max, npsf)

    # Maximum number of PSFs to use in library
    if (npsf_max is None) or (npsf_max > npsf):
        npsf_max = npsf
    if numbasis is None:
        nbasis_sets = 1
    elif isinstance(numbasis, (tuple, list, np.ndarray)):
        numbasis = np.asarray(numbasis)
        # Filter out numbasis where greater than npsf_max
        ind_out_of_range = numbasis > npsf_max
        if np.any(ind_out_of_range):
            _log.warning(f"numbasis values {numbasis[ind_out_of_range]} are greater than npsf_max={npsf_max}")
        numbasis = numbasis[~ind_out_of_range]
        nbasis_sets = len(numbasis)
        if nbasis_sets == 0:
            raise ValueError("All numbasis values are greater than npsf_max")
    elif isinstance(numbasis, int):
        if numbasis > npsf_max:
            raise ValueError(f"numbasis={numbasis} is greater than npsf_max={npsf_max}")
        numbasis = np.array([numbasis])
        nbasis_sets = 1

    # Initialize array of de-rotated and subtracted images
    fin_arr = np.zeros([nint, nbasis_sets, ny, nx])

    # print(fin_arr.shape)

    # Loop through all images
    for j in trange(nint, desc=f"Images", leave=False):
        im = imarr[j]

        # Get top correlated PSFs
        corr_vals = corr_all(im, psfarr, sub_size=32, inner_rad=5)
        imax = np.min([npsf_max, len(corr_vals)])
        isort = np.argsort(corr_vals)[::-1][0:imax]
        psf_lib = psfarr[isort]
        # corr_im = corr_vals[isort]

        psf_mean = np.nanmean(psf_lib, axis=0)
        psf_lib_resid = psf_lib - psf_mean
        im_resid = im - psf_mean

        # Loop through all subtraction regions
        sub_res = np.zeros([nbasis_sets,ny,nx])
        jj_arr = np.arange(1, nmasks+1)
        jj_iter = tqdm(jj_arr, leave=False, desc="Subtraction Region") if more_progress else jj_arr
        for jj in jj_iter:
            # Select optimization and subtraction regions
            # For KLIP, these are the same region
            ind_sub = (sub_masks == jj)
            ind_opt = ind_sub if do_klip else opt_masks[jj-1] 

            sci_sub = im_resid[ind_sub]  # Science subtraction region
            bg_opt = psf_lib_resid[:,ind_opt] # PSF optimization region

            # KLIP or LOCI PSF subtraction
            if do_loci:
                sci_opt = im_resid[ind_opt]        # Science optimization region
                bg_sub = psf_lib_resid[:,ind_sub]  # PSF subtraction region
                sub_res[:,ind_sub] = loci_subtract(sci_opt, bg_opt, bg_sub, sci_sub, 
                                                   numbasis=numbasis,
                                                   svd_frac=svd_frac,
                                                   remove_mean=remove_mean,
                                                   **kwargs)
            else:
                sub_res[:,ind_sub] = klip_subtract(sci_sub, bg_opt, 
                                                   numbasis=numbasis, 
                                                   svd_frac=svd_frac,
                                                   remove_mean=remove_mean)

        fin_arr[j] = sub_res

    return fin_arr.squeeze()


def run_pca_subtraction(imarr, psfarr, pca_params):
    """ Run PCA subtraction on a set of parameters """

    # Unpack parameters
    fwhm_pix = pca_params['fwhm_pix']
    do_loci = pca_params['loci']
    npsf_max = pca_params['numpsfs_max']
    numbasis = pca_params['numbasis']
    svd_frac = pca_params['svd_frac']
    kw_sub = pca_params['kwargs_sub']
    kw_opt = pca_params['kwargs_opt']

    return pca_subtraction(imarr, psfarr, fwhm_pix, do_loci=do_loci, 
                           npsf_max=npsf_max, numbasis=numbasis, 
                           svd_frac=svd_frac, kw_sub=kw_sub, kw_opt=kw_opt)

# def run_pca_subtraction(imarr, psfarr, fwhm_pix, do_loci=True, 
#                         npsf_max=None, numbasis=None, svd_frac=0.9, remove_mean=False,
#                         more_progress=True, kw_sub={}, kw_opt={}, **kwargs):


def build_pca_dict(fwhm_pix, **kwargs):
    """ Build PCA parameter dictionary

    Returns a dictionary of settings to pass to various analysis functions.
    There are multiple keywords that will be set by default and can be modified 
    by passing via keyword arguments. 
    
    Example output:

    pca_params = {
        'fwhm_pix' : fwhm_pix, # Number of pixels equal to PSF FWHM
        'loci'        : True,  # Use LOCI instead of KLIP?
        'remove_mean' : False,  # Remove mean from each frame before PCA?
        'numpsfs_max' : 100,    # Maximum number of PSFs to use for decomposition
        'numbasis'    : [1,5,10,50],  # Number of KL basis vectors to use
        'svd_frac'    : 0.9, # Fraction of variance to describe with KL basis vectors

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
