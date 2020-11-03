"""
ngNRC - NIRCam Detector Noise Simulator

Modification History:

15 Feb 2016, J.M. Leisenring, UA/Steward
    - First Release
21 July 2016, J.M. Leisenring, UA/Steward
    - Updated many things and more for nghxrg (v3.0)
11 Aug 2016, J.M. Leisenring, UA/Steward
    - Modified how the detector and multiaccum info is handled
    - Copied detector and multiaccum classes from pyNRC
    - In the future, we will want to integrate this directly
      so that any changes made in the pyNRC classes are accounted.
21 Feb 2017
    - Add ngNRC to pyNRC code base
20 Oct 2020
    - Restructure det noise and ramp creation
    - DMS simulations using JWST pipeline data models
"""
# Necessary for Python 2.6 and later
#from __future__ import division, print_function
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import datetime, os
from astropy.io import fits
from astropy.table import Table

# Program bar
from tqdm.auto import trange, tqdm

# HxRG Noise Generator
from . import nghxrg as ng
from pynrc.nrc_utils import pad_or_cut_to_size, jl_poly
from pynrc.reduce.calib import ramp_resample
from pynrc.maths.coords import det_to_sci, sci_to_det

#import pdb
from copy import deepcopy

from pynrc import DetectorOps, nrc_header
from pynrc import conf

# # Set log output levels
# # webbpsf and poppy have too many unnecessary warnings
# import logging
# logging.getLogger('nghxrg').setLevel(logging.ERROR)
# 
# _log = logging.getLogger('ngNRC')
# #_log.setLevel(logging.DEBUG)
# #_log.setLevel(logging.INFO)
# _log.setLevel(logging.WARNING)
# #_log.setLevel(logging.ERROR)
# logging.basicConfig(level=logging.WARNING,format='%(name)-10s: %(levelname)-8s %(message)s')

import logging
_log = logging.getLogger('pynrc')

def SCAnoise(det=None, scaid=None, params=None, caldir=None, file_out=None, 
    dark=True, bias=True, out_ADU=False, verbose=False, use_fftw=False, ncores=None,
    **kwargs):
    """NIRCam SCA noise generator
    
    Create a data cube consisting of realistic NIRCam detector noise.

    This is essentially a wrapper for nghxrg.py that selects appropriate values
    for a specified SCA in order to reproduce realistic noise properties similiar
    to those measured during ISIM CV3.

    Parameters
    ----------
    det : object
        Option to specify already existing NIRCam detector object class
        Otherwise, use scaid and params
    scaid : int
        NIRCam SCA number (481, 482, ..., 490)
    params : dictionary
        A set of MULTIACCUM parameters such as:
        
        >>> params = {'ngroup': 2, 'wind_mode': 'FULL', 
        >>>           'xpix': 2048, 'ypix': 2048, 'x0':0, 'y0':0}

        wind_mode can be FULL, STRIPE, or WINDOW
    file_out : str
        Folder name and destination to place optional FITS output. 
        A timestamp will be appended to the end of the file name (and before .fits').
    caldir : str
        Directory location housing the super bias and super darks for each SCA.
    dark : bool
        Use super dark? If True, then reads in super dark slope image.
    bias : bool
        Use super bias? If True, then reads in super bias image.
    out_ADU : bool
        Noise values are calculated in terms of equivalent electrons. This
        gives the option of converting to ADU (True) or keeping in term of e- (False).
        ADU values are converted to 16-bit UINT. Keep in e- if applying to a ramp
        observation then convert combined data to ADU later.

    Returns 
    ----------
    Primary HDU with noise ramp in hud.data and header info in hdu.header.

    Examples 
    ----------
    
    >>> import ngNRC
    >>> params = {'ngroup': 108, 'wind_mode': 'FULL', 
    >>>           'xpix': 2048, 'ypix': 2048, 'x0':0, 'y0':0}
    
    Output to a file:
    
    >>> scaid = 481
    >>> caldir = '/data/darks_sim/nghxrg/sca_images/'
    >>> file_out = '/data/darks_sim/dark_sim_481.fits'
    >>> hdu = ngNRC.SCAnoise(scaid, params, file_out=file_out, caldir=caldir, \
    >>>    dark=True, bias=True, out_ADU=True, use_fftw=False, ncores=None, verbose=False)

    Don't save file, but keep hdu in e- for adding to simulated observation ramp:

    >>> scaid = 481
    >>> caldir = '/data/darks_sim/nghxrg/sca_images/'
    >>> hdu = ngNRC.SCAnoise(scaid, params, file_out=None, caldir=caldir, \
    >>>    dark=True, bias=True, out_ADU=False, use_fftw=False, ncores=None, verbose=False)

    """

    # Extensive testing on both Python 2 & 3 shows that 4 cores is optimal for FFTW
    # Beyond four cores, the speed improvement is small. Those other processors are
    # are better used elsewhere.
    if use_fftw and (ncores is None): ncores = 4

    if det is None:
        wind_mode = params.pop('wind_mode', 'FULL')
        xpix = params.pop('xpix', 2048)
        ypix = params.pop('ypix', 2048)
        x0 = params.pop('x0', 0)
        y0 = params.pop('y0', 0)
        det = DetectorOps(scaid, wind_mode, xpix, ypix, x0, y0, params)
    else:
        scaid = det.scaid


    # Line and frame overheads
    nroh     = det._line_overhead
    nfoh     = det._extra_lines
    nfoh_pix = det._frame_overhead_pix

    # How many total frames (incl. dropped and all) per ramp?
    # Exclude last set of nd2 and nd3 (drops that add nothing)
    ma = det.multiaccum
    naxis3 = ma.nd1 + ma.ngroup*ma.nf + (ma.ngroup-1)*ma.nd2

    # Set bias and dark files
    sca_str = np.str(scaid)
    if caldir is None:
        caldir  = conf.PYNRC_PATH + 'sca_images/'
    bias_file = caldir + 'SUPER_BIAS_'+sca_str+'.FITS' if bias else None
    dark_file = caldir + 'SUPER_DARK_'+sca_str+'.FITS' if dark else None

    # Instantiate a noise generator object
    ng_h2rg = ng.HXRGNoise(naxis1=det.xpix, naxis2=det.ypix, naxis3=naxis3, 
                 n_out=det.nout, nroh=nroh, nfoh=nfoh, nfoh_pix=nfoh_pix,
                 dark_file=dark_file, bias_file=bias_file,
                 wind_mode=det.wind_mode, x0=det.x0, y0=det.y0,
                 use_fftw=use_fftw, ncores=ncores, verbose=verbose)
 

    # Lists of each SCA and their corresponding noise info
    sca_arr = range(481,491)

    # These come from measured dark ramps acquired during ISIM CV3 at GSFC
    # Gain values (e/ADU). Everything else will be in measured ADU units
    gn_arr =  [2.07, 2.01, 2.16, 2.01, 1.83, 
               2.00, 2.42, 1.93, 2.30, 1.85]

    # Noise Values (ADU)
    ktc_arr = [18.5, 15.9, 15.2, 16.9, 20.0, 
               19.2, 16.1, 19.1, 19.0, 20.0]
    ron_arr  = [[4.8,4.9,5.0,5.3], [4.4,4.4,4.4,4.2], [4.8,4.0,4.1,4.0], [4.5,4.3,4.4,4.4],
                [4.2,4.0,4.5,5.4],
                [5.1,5.1,5.0,5.1], [4.6,4.3,4.5,4.2], [5.1,5.6,4.6,4.9], [4.4,4.5,4.3,4.0],
                [4.5,4.3,4.6,4.8]]
    # Pink Noise Values (ADU)
    cp_arr  = [ 2.0, 2.5, 1.9, 2.5, 2.1,
                2.5, 2.5, 3.2, 3.0, 2.5]
    up_arr  = [[0.9,0.9,0.9,0.9], [0.9,1.0,0.9,1.0], [0.8,0.9,0.8,0.8], [0.8,0.9,0.9,0.8],
               [1.0,1.3,1.0,1.1],
               [1.0,0.9,1.0,1.0], [0.9,0.9,1.1,1.0], [1.0,1.0,1.0,0.9], [1.1,1.1,0.8,0.9],
               [1.1,1.1,1.0,1.0]]
   
   
    # Offset Values (ADU)
    bias_avg_arr = [5900, 5400, 6400, 6150, 11650, 
                    7300, 7500, 6700, 7500, 11500]
    bias_sig_arr = [20.0, 20.0, 30.0, 11.0, 50.0, 
                    20.0, 20.0, 20.0, 20.0, 20.0]
    ch_off_arr   = [[1700, 530, -375, -2370], [-150, 570, -500, 350], [-530, 315, 460, -200],
                    [480, 775, 1040, -2280],  [560, 100, -440, -330],
                    [105, -29, 550, -735],    [315, 425, -110, -590],   [918, -270, 400, -1240],
                    [-100, 500, 300, -950],   [-35, -160, 125, -175]]
    f2f_corr_arr = [14.0, 13.8, 27.0, 14.0, 26.0,
                    14.7, 11.5, 18.4, 14.9, 14.8]
    f2f_ucorr_arr= [[18.4,11.1,10.8,9.5], [7.0,7.3,7.3,7.1], [6.9,7.3,7.3,7.5],
                    [6.9,7.3,6.5,6.7], [16.6,14.8,13.5,14.2],
                    [7.2,7.5,6.9,7.0], [7.2,7.6,7.5,7.4], [7.9,6.8,6.9,7.0],
                    [7.6,8.6,7.5,7.4], [13.3,14.3,14.1,15.1]]
    aco_a_arr    = [[770, 440, 890, 140], [800, 410, 840, 800], [210,680,730,885],
                    [595, 642, 634, 745], [-95,660,575,410],
                    [220, 600, 680, 665], [930,1112, 613, 150], [395, 340, 820, 304],
                    [112, 958, 690, 907], [495, 313, 392, 855]]
    ref_inst_arr = [1.0, 1.5, 1.0, 1.3, 1.0, 
                    1.0, 1.0, 1.0, 2.2, 1.0]


    # SCA Index
    ind = sca_arr.index(scaid)

    # Convert everything to e-
    gn = gn_arr[ind]
    # Noise Values
    ktc_noise= gn * ktc_arr[ind] * 1.15            # kTC noise in electrons
    rd_noise = gn * np.array(ron_arr[ind]) * 0.93  # White read noise per integration
    # Pink Noise
    c_pink   = gn * cp_arr[ind] * 1.6              # Correlated pink noise
    u_pink   = gn * np.array(up_arr[ind]) * 1.4    # Uncorrelated
    ref_rat  = 0.9  # Ratio of reference pixel noise to that of reg pixels

    # Offset Values
    bias_off_avg = gn * bias_avg_arr[ind] + 110  # On average, integrations start here in electrons
    bias_off_sig = gn * bias_sig_arr[ind]  # bias_off_avg has some variation. This is its std dev.
    bias_amp     = gn * 1.0     # A multiplicative factor to multiply bias_image. 1.0 for NIRCam.

    # Offset of each channel relative to bias_off_avg.
    ch_off = gn * np.array(ch_off_arr[ind]) + 110
    # Random frame-to-frame reference offsets due to PA reset
    ref_f2f_corr  = gn * f2f_corr_arr[ind] * 0.95
    ref_f2f_ucorr = gn * np.array(f2f_ucorr_arr[ind]) * 1.15 # per-amp
    # Relative offsets of altnernating columns
    aco_a = gn * np.array(aco_a_arr[ind])
    aco_b = -1 * aco_a
    #Reference Instability
    ref_inst = gn * ref_inst_arr[ind]

    # If only one output (window mode) then select first elements of each array
    if det.nout == 1:
        rd_noise = rd_noise[0]
        u_pink = u_pink[0]
        ch_off = ch_off[0]
        ref_f2f_ucorr = ref_f2f_ucorr[0]
        aco_a = aco_a[0]; aco_b = aco_b[0]

    # Run noise generator
    hdu = ng_h2rg.mknoise(None, gain=gn, rd_noise=rd_noise, c_pink=c_pink, u_pink=u_pink, 
            reference_pixel_noise_ratio=ref_rat, ktc_noise=ktc_noise,
            bias_off_avg=bias_off_avg, bias_off_sig=bias_off_sig, bias_amp=bias_amp,
            ch_off=ch_off, ref_f2f_corr=ref_f2f_corr, ref_f2f_ucorr=ref_f2f_ucorr, 
            aco_a=aco_a, aco_b=aco_b, ref_inst=ref_inst, out_ADU=out_ADU)

    hdu.header = nrc_header(det)#, header=hdu.header)
    hdu.header['UNITS'] = 'ADU' if out_ADU else 'e-'

    # Write the result to a FITS file
    if file_out is not None:
        now = datetime.datetime.now().isoformat()[:-7]
        hdu.header['DATE'] = now #datetime.datetime.now().isoformat()[:-7]
        if file_out.lower()[-5:] == '.fits':
            file_out = file_out[:-5]
        if file_out[-1:] == '_':
            file_out = file_out[:-1]

# 		file_now = now
# 		file_now = file_now.replace(':', 'h', 1)
# 		file_now = file_now.replace(':', 'm', 1)
# 		file_out = file_out + '_' + file_now + '.fits'
        file_out = file_out + '.fits'

        hdu.header['FILENAME'] = os.path.split(file_out)[1]
        hdu.writeto(file_out, clobber='True')

    return hdu

# def slope_to_ramp(det, im_slope=None, out_ADU=False, file_out=None, 
#                   filter=None, pupil=None, obs_time=None, targ_name=None,
#                   DMS=True, dark=True, bias=True, det_noise=True,
#                   return_results=True):
#     """Convert slope image to simulated ramp
    
#     For a given detector operations class and slope image, create a
#     ramp integration using Poisson noise and detector noise. 

#     Currently, this image simulator does NOT take into account:
    
#         - QE variations across a pixel's surface
#         - Intrapixel Capacitance (IPC)
#         - Post-pixel Coupling (PPC) due to ADC "smearing"
#         - Pixel non-linearity
#         - Persistence/latent image
#         - Optical distortions
#         - Zodiacal background roll off for grism edges
#         - Telescope jitter
#         - Cosmic Rays

#     Parameters
#     ----------
#     det : object
#         Detector operations object.
#     im_slope : ndarray
#         Idealized slope image.

#     out_ADU : bool
#         If true, divide by gain and convert to 16-bit UINT.
#     file_out : str
#         Name (including directory) to save FITS file
#     filter : str
#         Name of filter element for header
#     pupil : str
#         Name of pupil element for header
#     obs_time : datetime 
#         Specifies when the observation was considered to be executed.
#         If not specified, then it will choose the current time.
#         This must be a datetime object:
            
#             >>> datetime.datetime(2016, 5, 9, 11, 57, 5, 796686)
            
#         This information is added to the header.
#     targ_name : str
#         Target name (optional)
#     DMS : bool
#         Package the data in the format used by DMS?
#     dark : bool
#         Include the dark current?
#     bias : bool
#         Include the bias frame?
#     det_noise: bool
#         Include detector noise components? If set to False, then only 
#         perform Poisson noise. Darks and biases are also excluded.
#     """
    
#     if (det_noise==False) and (im_slope is None):
#         _log.warning('No input slope image and det_noise=False. Nothing to return.')
#         return

#     #import ngNRC
#     # MULTIACCUM ramp information
#     ma  = det.multiaccum

#     xpix = det.xpix
#     ypix = det.ypix

#     nd1     = ma.nd1
#     nd2     = ma.nd2
#     nf      = ma.nf
#     ngroup  = ma.ngroup
#     t_frame = det.time_frame

#     # Number of total frames up the ramp (including drops)
#     naxis3 = nd1 + ngroup*nf + (ngroup-1)*nd2

#     if im_slope is not None:
#         # Set reference pixels' slopes equal to 0
#         w = det.ref_info
#         if w[0] > 0: # lower
#             im_slope[:w[0],:] = 0
#         if w[1] > 0: # upper
#             im_slope[-w[1]:,:] = 0
#         if w[2] > 0: # left
#             im_slope[:,:w[2]] = 0
#         if w[3] > 0: # right
#             im_slope[:,-w[3]:] = 0

#         # Count accumulation for a single frame
#         frame = im_slope * t_frame
#         # Add Poisson noise at each frame step
#         sh0, sh1 = im_slope.shape
#         new_shape = (naxis3, sh0, sh1)
#         ramp = np.random.poisson(lam=frame, size=new_shape)#.astype(np.float64)
#         # Perform cumulative sum in place
#         np.cumsum(ramp, axis=0, out=ramp)
        
#         # Truncate anything above well level
#         well_max = det.well_level
#         ramp[ramp>well_max] = well_max
#     else:
#         ramp = 0

#     if det_noise==False:
#         hdu = fits.PrimaryHDU(ramp)
#     else:
#         # Create dark ramp with read noise and 1/f noise
#         hdu = SCAnoise(det=det, dark=dark, bias=bias)
#         # Add signal ramp to dark ramp
#         if im_slope is not None:
#             hdu.data += ramp.reshape(hdu.data.shape) 
#     # Update header information
#     hdu.header = det.make_header(filter, pupil, obs_time, targ_name=targ_name, DMS=DMS)
#     data = hdu.data

#     #### Add in IPC (TBI) ####

#     # Get rid of any drops at the beginning (nd1)
#     if nd1>0: data = data[nd1:,:,:]

#     # Convert to ADU (16-bit UINT)
#     if out_ADU:
#         gain = det.gain
#         data = data / gain
#         data[data < 0] = 0
#         data[data >= 2**16] = 2**16 - 1
#         data = data.astype('uint16')
#         hdu.header['UNITS'] = 'ADU'

#     ## Save the first frame (so-called ZERO frame) for the zero frame extension
#     zeroData = deepcopy(data[0,:,:])

#     # Remove drops and average grouped data
#     if nf>1 or nd2>0:
#         # Trailing drop frames already excluded, so need to pull off last group of avg'ed frames
#         data_end = data[-nf:,:,:].mean(axis=0) if nf>1 else data[-1:,:,:]
#         data_end = data_end.reshape([1,ypix,xpix])

#         # Only care about first (n-1) groups for now
#         # Last group is handled separately
#         data = data[:-nf,:,:]

#         # Reshape for easy group manipulation
#         data = data.reshape([-1,nf+nd2,ypix,xpix])

#         # Trim off the dropped frames (nd2)
#         if nd2>0: data = data[:,:nf,:,:]

#         # Average the frames within groups
#         # In reality, the 16-bit data is bit-shifted
#         data = data.reshape([-1,ypix,xpix]) if nf==1 else data.mean(axis=1)

#         # Add back the last group (already averaged)
#         data = np.append(data,data_end,axis=0)

#     hdu.data = data
    
#     if file_out is not None:
#         hdu.header['FILENAME'] = os.path.split(file_out)[1]
    
#     if DMS == True:
#         primHDU = fits.PrimaryHDU(header=hdu.header)
#         primHDU.name = 'PRIMARY'
#         sciHDU = fits.ImageHDU(data=hdu.data)
#         sciHDU.name = 'SCI'
#         sciHDU.header.comments['NAXIS1'] = 'length of first data axis (#columns)'
#         sciHDU.header.comments['NAXIS2'] = 'length of second data axis (#rows)'
#         if sciHDU.header['NAXIS'] > 2:
#             sciHDU.header.comments['NAXIS3'] = 'length of third data axis (#groups/integration '
#         if sciHDU.header['NAXIS'] > 3:
#             sciHDU.header.comments['NAXIS4'] = 'length of fourth data axis (#integrations)  '
#         sciHDU.header['BZERO'] = (32768, 'physical value for an array value of zero')
#         sciHDU.header['BUNIT'] = ('DN', 'physical units of the data array values')
        
#         zerHDU = fits.ImageHDU(data=zeroData)
#         zerHDU.name = 'ZEROFRAME'
#         zerHDU.header.comments['NAXIS1'] = 'length of first data axis (#columns)'
#         zerHDU.header.comments['NAXIS2'] = 'length of second data axis (#rows)'
        
#         outHDU = fits.HDUList([primHDU,sciHDU,zerHDU])
#     else:
#         outHDU = hdu
    
#     if file_out is not None:
#         outHDU.writeto(file_out, overwrite='True')
    
#     # Only return outHDU if return_results=True
#     if return_results: 
#         return outHDU

def slope_to_ramps(det, dark_cal_obj, im_slope=None, filter=None, pupil=None, 
                   targ_name=None, obs_time=None, file_out=None, 
                   out_ADU=True, DMS=True, return_results=True, **kwargs):
    
    """Simulate HDUList frome slope
    
    
    Parameters
    ==========
    det : Detector Class
        Desired detector class output
    dark_cal_obj: nircam_dark class
        NIRCam Dark class that holds the necessary calibration info to
        simulate a dark ramp.
    im_slope : ndarray
        Input slope image of observed scene. Assumed to be in detector
        coordinates. If an image cube, then number of images must match 
        the number of integration (`nint`) in `det` class.

    """
    import os
    import datetime
    import multiprocessing as mp
    
    
    # Pixel readout
    nchan = det.nout
    ny, nx = (det.ypix, det.xpix)
    
    # Number of saved frames in a ramp
    ma   = det.multiaccum
    nz   = ma.ngroup
    nint = ma.nint
        
    if DMS:
        is_cube = True if (im_slope is not None) and (len(im_slope.shape)==3) else False
        if is_cube:
            assert im_slope.shape[0]==nint, "`im_slope` images must match nint"

        # Save all ramps within an exposure
        data_all = []
        zframe_all = []

        for i in trange(nint, desc='Ramps', leave=True):
            im_slope_i = im_slope[i] if is_cube else im_slope

            res = simulate_detector_ramp(det, dark_cal_obj, im_slope=im_slope_i, 
                                        out_ADU=out_ADU, return_zero_frame=True, **kwargs)

            # Convert from det to sci coords
            data   = det_to_sci(res[0], det.detid)
            zframe = det_to_sci(res[1], det.detid)
            # Append to full array
            data_all.append(data)
            zframe_all.append(zframe)
        
        data_all = np.array(data_all)
        zframe_all = np.array(zframe_all)

        header = det.make_header(filter, pupil, obs_time, targ_name=targ_name, DMS=DMS)
        if file_out is not None:
            header['FILENAME'] = os.path.split(file_out)[1]
            
        # Primary exension just has most header information
        # of telescope, instrument, target, visit, exposure, either, ephemeris,et
        primHDU = fits.PrimaryHDU(header=header)
        primHDU.name = 'PRIMARY'
        
        # Science extension includes exposure data
        # Header has obs time, spacecraft pointing, WCS
        sciHDU = fits.ImageHDU(data=data_all)
        sciHDU.name = 'SCI'
        sciHDU.header.comments['NAXIS1'] = 'length of first data axis (#columns)'
        sciHDU.header.comments['NAXIS2'] = 'length of second data axis (#rows)'
        if sciHDU.header['NAXIS'] > 2:
            sciHDU.header.comments['NAXIS3'] = 'length of third data axis (#groups/integration '
        if sciHDU.header['NAXIS'] > 3:
            sciHDU.header.comments['NAXIS4'] = 'length of fourth data axis (#integrations)  '
        sciHDU.header['BZERO'] = (32768, 'physical value for an array value of zero')
        sciHDU.header['BUNIT'] = ('DN', 'physical units of the data array values')
        
        # Zeroframe extension
        zerHDU = fits.ImageHDU(data=zframe_all)
        zerHDU.name = 'ZEROFRAME'
        zerHDU.header.comments['NAXIS1'] = 'length of first data axis (#columns)'
        zerHDU.header.comments['NAXIS2'] = 'length of second data axis (#rows)'
        
        outHDU = fits.HDUList([primHDU,sciHDU,zerHDU])
            
    # FITSWriter (ISIM format)
    else:
        assert len(im_slope.shape)==2, "`im_slope` should be a single image"
        data = simulate_detector_ramp(det, dark_cal_obj, im_slope=im_slope, 
                                      out_ADU=out_ADU,  return_zero_frame=False, **kwargs)
        hdu = fits.PrimaryHDU(data)
        hdu.header = det.make_header(filter, pupil, obs_time, targ_name=targ_name, DMS=DMS)

        if file_out is not None:
            hdu.header['FILENAME'] = os.path.split(file_out)[1]
        outHDU = fits.HDUList([hdu])

    # Write file to disk
    if file_out is not None:
        outHDU.writeto(file_out, overwrite='True')
        
    # Only return outHDU if return_results=True
    if return_results: 
        return outHDU
    else:
        outHDU.close()





def add_ipc(im, alpha_min=0.0065, alpha_max=None, kernel=None):
    """Convolve image with IPC kernel
    
    Given an image in electrons, apply IPC convolution.
    NIRCam average IPC values (alpha) reported 0.005 - 0.006.
    
    Parameters
    ==========
    im : ndarray
        Input image or array of images.
    alpha_min : float
        Minimum coupling coefficient between neighboring pixels.
        If alpha_max is None, then this is taken to be constant
        with respect to signal levels.
    alpha_max : float or None
        Maximum value of coupling coefficent. If specificed, then
        coupling between pixel pairs is assumed to vary depending
        on signal values. See Donlon et al., 2019, PASP 130.
    kernel : ndarry or None
        Option to directly specify the convolution kernel. 
        `alpha_min` and `alpha_max` are ignored.
    
    Examples
    ========
    Constant Kernel

        >>> im_ipc = add_ipc(im, alpha_min=0.0065)

    Constant Kernel (manual)

        >>> alpha = 0.0065
        >>> k = np.array([[0,alpha,0], [alpha,1-4*alpha,alpha], [0,alpha,0]])
        >>> im_ipc = add_ipc(im, kernel=k)

    Signal-dependent Kernel

        >>> im_ipc = add_ipc(im, alpha_min=0.0065, alpha_max=0.0145)

    """

    from astropy.convolution import convolve
    
    sh = im.shape
    ndim = len(sh)
    if ndim==2:
        im = im.reshape([1,sh[0],sh[1]])
        sh = im.shape
    
    if kernel is None:
        xp = yp = 1
    else:
        yp, xp = np.array(kernel.shape) / 2
        yp, xp = int(yp), int(xp)

    # Pad images to have a pixel border of zeros
    im_pad = np.pad(im, ((0,0), (yp,yp), (xp,xp)), 'constant')
    
    # Check for custom kernel (overrides alpha values)
    if (kernel is not None) or (alpha_max is None):
        # Reshape to stack all images along horizontal axes
        im_reshape = im_pad.reshape([-1, im_pad.shape[-1]])
    
        if kernel is None:
            kernel = np.array([[0.0, alpha_min, 0.0],
                               [alpha_min, 1.-4*alpha_min, alpha_min],
                               [0.0, alpha_min, 0.0]])
    
        # Convolve IPC kernel with images
        im_ipc = convolve(im_reshape, kernel).reshape(im_pad.shape)
    
    # Exponential coupling strength
    # Equation 7 of Donlon et al. (2018)
    else:
        arrsqr = im_pad**2

        amin = alpha_min
        amax = alpha_max
        ascl = (amax-amin) / 2
        
        alpha_arr = []
        for ax in [1,2]:
            # Shift by -1
            diff = np.abs(im_pad - np.roll(im_pad, -1, axis=ax))
            sumsqr = arrsqr + np.roll(arrsqr, -1, axis=ax)
            
            imtemp = amin + ascl * np.exp(-diff/20000) + \
                     ascl * np.exp(-np.sqrt(sumsqr / 2) / 10000)
            alpha_arr.append(imtemp)
            # Take advantage of symmetries to shift in other direction
            alpha_arr.append(np.roll(imtemp, 1, axis=ax))
            
        alpha_arr = np.array(alpha_arr)

        # Flux remaining in parent pixel
        im_ipc = im_pad * (1 - alpha_arr.sum(axis=0))
        # Flux shifted to adjoining pixels
        for i, (shft, ax) in enumerate(zip([-1,+1,-1,+1], [1,1,2,2])):
            im_ipc += alpha_arr[i]*np.roll(im_pad, shft, axis=ax)
        del alpha_arr

    # Trim excess
    return im_ipc[:,yp:-yp,xp:-xp].squeeze()
    
    
def add_ppc(im, ppc_frac=0.002, nchans=4, kernel=None,
    same_scan_direction=False, reverse_scan_direction=False,
    in_place=False):
    """ Add Post-Pixel Coupling (PPC)
    
    This effect is due to the incomplete settling of the analog
    signal when the ADC sample-and-hold pulse occurs. The measured
    signals for a given pixel will have a value that has not fully
    transitioned to the real analog signal. Mathematically, this
    can be treated in the same way as IPC, but with a different
    convolution kernel.
    
    Parameters
    ==========
    im : ndarray
        Image or array of images
    ppc_frac : float
        Fraction of signal contaminating next pixel in readout. 
    kernel : ndarry or None
        Option to directly specify the convolution kernel, in
        which case `ppc_frac` is ignored.
    nchans : int
        Number of readout output channel amplifiers.
    same_scan_direction : bool
        Are all the output channels read in the same direction?
        By default fast-scan readout direction is ``[-->,<--,-->,<--]``
        If ``same_scan_direction``, then all ``-->``
    reverse_scan_direction : bool
        If ``reverse_scan_direction``, then ``[<--,-->,<--,-->]`` or all ``<--``
    in_place : bool
        Apply in place to input image.
    """

                       
    sh = im.shape
    ndim = len(sh)
    if ndim==2:
        im = im.reshape([1,sh[0],sh[1]])
        sh = im.shape

    nz, ny, nx = im.shape
    chsize = nx // nchans
    
    # Do each channel separately
    if kernel is None:
        kernel = np.array([[0.0, 0.0, 0.0],
                           [0.0, 1.0-ppc_frac, ppc_frac],
                           [0.0, 0.0, 0.0]])

    res = im if in_place else im.copy()
    for ch in np.arange(nchans):
        if same_scan_direction:
            k = kernel[:,::-1] if reverse_scan_direction else kernel
        elif np.mod(ch,2)==0:
            k = kernel[:,::-1] if reverse_scan_direction else kernel
        else:
            k = kernel if reverse_scan_direction else kernel[:,::-1]

        x1 = chsize*ch
        x2 = x1 + chsize
        res[:,:,x1:x2] = add_ipc(im[:,:,x1:x2], kernel=k)
    
    return res.squeeze()



def gen_col_noise(ramp_column_varations, prob_bad, nz=108, nx=2048):
    """ Generate RTN Column Noise

    This function takes the random telegraph noise templates derived from 
    CV3 data and generates a random noise set to add to an dark ramp sim.
    These column variations likely come from RTN in column-specifc FETs 
    jumping between two discrete states, possibly within the detector column bus.

    This function randomly draws a number of template column variation ramps,
    then randomly assigns them to different columns in `super_dark_ramp`.
    The nubmer of columns (and whether or not a column is assigned a random
    variation) is based on the `prob_bad` variable.
    """

    # Number of samples in ramp templates
    nz0 = ramp_column_varations.shape[0]

    if nz>nz0:
        raise ValueError('nz should not be greater than {} frames'.format(nz0))

    # Variable to store column offsets for all NX columns
    cols_all_add = np.zeros([nz0,nx])

    # Mask of random columns to include ramp excursions
    # Create set of random values between 0 and 1
    # Mark those with values less than prob_bad for 
    # adding some random empirically measured column
    xmask_random = np.random.random_sample(size=nx) <= prob_bad
    nbad_random = len(xmask_random[xmask_random])

    # Grab some random columns from the stored templates
    ntemplates = ramp_column_varations.shape[1]
    ind_rand = np.random.randint(0, high=ntemplates, size=ntemplates)
    # Make sure we get unique values (no repeats)
    _, ind_rand = np.unique(ind_rand, return_index=True)
    ind_rand = ind_rand[0:nbad_random]
    # If we don't have enough random columns, append more
    # This should be very unlikely to occur, but just in case...
    if len(ind_rand) < nbad_random:
        ndiff = nbad_random - len(ind_rand)
        ind_rand = np.append(ind_rand, np.random.randint(0, high=ntemplates, size=ndiff))
        
    # Select the set of random column variation templates
    cols_rand = ramp_column_varations[:,ind_rand]

    # Add a random phase shift to each of those template column
    tshifts = np.random.randint(0, high=nz0, size=nbad_random)
    for i in np.arange(nbad_random):
        cols_rand[:,i] = np.roll(cols_rand[:,i], tshifts[i])

    # Add to columns variable
    cols_all_add[:, xmask_random] = cols_rand

    # Reshape to (nz0,1,nx) to easily add to a ramp of size (nz,ny,nx)
    cols_all_add = cols_all_add.reshape([nz0,1,-1])

    # Only return number of request frames
    return cols_all_add[0:nz, :, :]

def add_col_noise(super_dark_ramp, ramp_column_varations, prob_bad):
    """ Add RTN Column Noise
    
    This function takes the random telegraph noise templates derived from 
    CV3 data and adds it to an idealized dark ramp. These column variations 
    likely come from noise in column-specifc FETs jumping between two discrete 
    states, possibly within the detector column bus.

    This function randomly draws a number of template column variation ramps,
    then randomly assigns them to different columns in `super_dark_ramp`.
    The nubmer of columns (and whether or not a column is assigned a random
    variation) is based on the `prob_bad` variable.
    
    Parameters
    ==========
    
    super_dark_ramp : ndarray
        Idealized ramp of size (nz,ny,nx)
    ramp_column_variations : ndarray
        The column-average ramp variations of size (nz,nx). 
        These are added to a given columnn.
    prob_bad : float
        Probability that a given column is subject to these column variations.
    """
    
    nz, ny, nx = super_dark_ramp.shape
    
    cols_all_add = gen_col_noise(ramp_column_varations, prob_bad, nz=nz, nx=nx)

    # Add to dark ramp
    data = super_dark_ramp + cols_all_add
    
    return data

def gen_ramp_biases(ref_dict, nchan=None, data_shape=(2,2048,2048), ref_border=[4,4,4,4]):
    """ Generate a ramp of bias offsets

    Parameters
    ==========
    ref_dict : dict
        Dictionary of reference behaviors.
    nchan : int
        Specify number of output channels. If not set, then will
        automatically determine from `ref_dict`. This allows us
        to set nchan=1 for Window Mode while using the first channel
        info provided in `ref_dict`.
    data_shape : array like
        Shape of output (nz,ny,nx) 
    ref_border: list
        Number of references pixels [lower, upper, left, right]
    """
    
    if nchan is None:
        nchan = len(ref_dict['amp_offset_mean'])

    cube = np.zeros(data_shape)
    nz, ny, nx = data_shape
    chsize = int(nx/nchan)
    
    ######################
    # Add overall bias
    # TODO: Add temperature dependence
    bias_off = ref_dict['master_bias_mean'] + np.random.normal(scale=ref_dict['master_bias_std'])
    cube += bias_off

    # Add amplifier offsets
    # These correlate to bias offset
    cf = ref_dict['master_amp_cf']
    amp_off = jl_poly(bias_off, cf) + np.random.normal(scale=ref_dict['amp_offset_std'])

    for ch in range(nchan):
        cube[:,:,ch*chsize:(ch+1)*chsize] += amp_off[ch]
    
    # Include frame-to-frame bias variation
    ######################
    bias_off_f2f = np.random.normal(scale=ref_dict['master_bias_f2f'], size=nz)
    amp_off_f2f = np.random.normal(scale=ref_dict['amp_offset_f2f'][0:nchan], size=(nz,nchan))

    for i, im in enumerate(cube):
        im += bias_off_f2f[i]
        for ch in range(nchan):
            im[:,ch*chsize:(ch+1)*chsize] += amp_off_f2f[i,ch]
    
    # Add some reference pixel instability relative to active pixels
    ######################

    # Mask of active pixels
    mask_act = np.zeros([ny,nx]).astype('bool')
    rb, rt, rl, rr = ref_border
    mask_act[rb:-rt,rl:-rr] = True

    # Mask of all reference pixels
    mask_ref = ~mask_act

    # ref_inst = np.random.normal(scale=ref_dict['amp_ref_inst_f2f'], size=(nz,nchan))
    for ch in range(nchan):
        mask_ch = np.zeros([ny,nx]).astype('bool')
        mask_ch[:,ch*chsize:(ch+1)*chsize] = True

        std = ref_dict['amp_ref_inst_f2f'][ch]
        ref_noise = std * pink_noise(nz)
        cube[:, mask_ref & mask_ch] += ref_noise.reshape([-1,1])

        
        # cube[:,mask_ref & mask_ch] += ref_inst[:,ch].reshape([-1,1])


    # Set even/odd offsets
    ######################
    mask_even = np.zeros([ny,nx]).astype('bool')
    mask_even[:,0::2] = True

    mask_odd = np.zeros([ny,nx]).astype('bool')
    mask_odd[:,1::2] = True

    for ch in range(nchan):
        mask_ch = np.zeros([ny,nx]).astype('bool')
        mask_ch[:,ch*chsize:(ch+1)*chsize] = True

        cube[:, mask_even & mask_ch] += ref_dict['amp_even_col_offset'][ch]
        cube[:, mask_odd & mask_ch]  += ref_dict['amp_odd_col_offset'][ch]
    
    return cube


def fft_noise(pow_spec, nstep_out=None, fmin=None, f=None, 
              pad_mode='edge', **kwargs):
    """ Random Noise from Power Spectrum
    
    Returns a noised array where the instrinsic where the distribution
    is follows that of the input power spectrum. The output has an
    instrinsic standard deviation scaled to 1.0.
    
    Parameters
    ==========
    pow_spec : ndarray
        Input power spectrum from which to generate noise distribution.
    nstep_out : int
        Desired size of the output noise array. If smaller than `pow_spec`
        then it just truncates the results to the appropriate size.
        If larger, then pow_spec gets padded by the specified `pad_mode`.
    fmin : float or None
        Low-frequency cutoff. Power spectrum values below this cut-off
        point get set equal to the power spectrum value at fmin.
    f : ndarray or None
        An array the same size as pow_spec and is only used when fmin
        is set. If set to None, then `f = np.fft.rfftfreq(n_ifft)`
        where `n_ifft` is the size of the result of `rifft(pow_spec)`
        assuming a delta time of unity.
    pad_mode : str or function
        One of the following string values or a user supplied function.
        Default is 'edge'.

        'constant' (default)
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.
        'linear_ramp'
            Pads with the linear ramp between end_value and the
            array edge value.
        'maximum'
            Pads with the maximum value of all or part of the
            vector along each axis.
        'mean'
            Pads with the mean value of all or part of the
            vector along each axis.
        'median'
            Pads with the median value of all or part of the
            vector along each axis.
        'minimum'
            Pads with the minimum value of all or part of the
            vector along each axis.
        'reflect'
            Pads with the reflection of the vector mirrored on
            the first and last values of the vector along each
            axis.
        'symmetric'
            Pads with the reflection of the vector mirrored
            along the edge of the array.
        'wrap'
            Pads with the wrap of the vector along the axis.
            The first values are used to pad the end and the
            end values are used to pad the beginning.

    """
    
    if nstep_out is None:
        nstep_out = 2 * (len(pow_spec) - 1)
        
    nstep = nstep_out
    nstep2 = int(2**np.ceil(np.log2(nstep-1))) + 1

    lin_spec = np.sqrt(pow_spec) 
    # Set cuf-off frequency
    if (fmin is not None) and (fmin>0):
        n_ifft = 2 * (len(lin_spec) - 1)
        f = np.fft.rfftfreq(n_ifft) if f is None else f
        fstep = f[1] - f[0]
        fmin = np.max([fmin, fstep])
        ix  = np.sum(f < fmin)   # Index of the cutoff
        if ix > 1 and ix < len(f):
            lin_spec[:ix] = lin_spec[ix]
    
    # Padding to add lower frequencies
    pad = nstep2-len(lin_spec)
    pad = 0 if pad <0 else pad
    if pad>0:
        lin_spec = np.pad(lin_spec, (pad,0), mode=pad_mode, **kwargs)
    
    # Build scaling factors for all frequencies
    
    # Calculate theoretical output standard deviation from scaling
    w = lin_spec[1:-1]
    n_ifft = 2 * (len(lin_spec) - 1)
    w_last = lin_spec[-1] * (1 + (n_ifft % 2)) / 2. # correct f = +-0.5
    the_std = 2 * np.sqrt(np.sum(w**2) + w_last**2) / n_ifft
    
    # Generate scaled random power + phase
    sr = lin_spec
    sr = np.random.normal(scale=lin_spec)
    si = np.random.normal(scale=lin_spec)

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if (nstep2 % 2) == 0: 
        si[-1] = 0

    # Regardless of signal length, the DC component must be real
    si[0] = 0

    # Combine power + corrected phase to Fourier components
    thefft  = sr + 1J * si

    # Apply the pinkening filter.
    result = np.fft.irfft(thefft)
    
    # Keep requested nstep and scale to unit variance
    result = result[:nstep_out] / the_std

    return result


def pink_noise(nstep_out, pow_spec=None, f=None, fmin=None, alpha=-1, **kwargs):
    """ Generate random pink noise

    Parameters
    ==========
    nstep_out : int
        Desired size of the output noise array. If smaller than `pow_spec`
        then it just truncates the results to the appropriate size.
        If larger, then pow_spec gets padded by the specified `pad_mode`.
    pow_spec : ndarray
        Option to input the power spectrum instead of regenerating it
        every time. Make sure this was generated with powers of 2 for
        faster processing.
    f : ndarray or None
        An array the same size as pow_spec. If set to None, then 
        will create an array of appropriate size assuming a delta
        time of unity.
    fmin : float or None
        Low-frequency cutoff. Power spectrum values below this cut-off
        point get set equal to the power spectrum value at fmin.
    pad_mode : str or function
        One of the following string values or a user supplied function.
        Default is 'edge'.

        'constant' (default)
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.
        'linear_ramp'
            Pads with the linear ramp between end_value and the
            array edge value.
        'maximum'
            Pads with the maximum value of all or part of the
            vector along each axis.
        'mean'
            Pads with the mean value of all or part of the
            vector along each axis.
        'median'
            Pads with the median value of all or part of the
            vector along each axis.
        'minimum'
            Pads with the minimum value of all or part of the
            vector along each axis.
        'reflect'
            Pads with the reflection of the vector mirrored on
            the first and last values of the vector along each
            axis.
        'symmetric'
            Pads with the reflection of the vector mirrored
            along the edge of the array.
        'wrap'
            Pads with the wrap of the vector along the axis.
            The first values are used to pad the end and the
            end values are used to pad the beginning.


    """
    
    if ((fmin is not None) and (fmin>0)) or (pow_spec is None):
        # Set up to a power of 2 for faster processing
        nstep2 = 2 * int(2**np.ceil(np.log2(nstep_out)))
        f = np.fft.rfftfreq(nstep2)
        f[0] = f[1] # First element should not be 0
    if pow_spec is None:
        pow_spec = f**alpha
        pow_spec[0] = 0.
        
    if f is not None:
        assert len(f)==len(pow_spec), "f and p_filter must be same size"
        
    assert len(pow_spec)>=nstep_out, "Power spectrum must be greater than nstep_out"
            
    res = fft_noise(pow_spec, nstep_out=nstep_out, fmin=fmin, f=f, **kwargs)
        
    return res

def sim_noise_data(det, rd_noise=[5,5,5,5], u_pink=[1,1,1,1], c_pink=3,
    acn=0, same_scan_direction=False, reverse_scan_direction=False,
    pow_spec_corr=None, corr_scales=None, fcorr_lim=[1,10],
    ref_ratio=0.8, **kwargs):
    
    """ Simulate Noise Ramp
    
    Simulate the noise components of a ramp, including white noise as well as
    1/f (pink) noise components that are uncorrelated and correlated between
    amplifier channels.

    Parameters
    ==========
    det : `det_timing` class
        Class holding detector operations information. See `detops.det_timing`
        for generic class, or `pynrc_core.DetectorOps` for NIRCam specific
        timing.
    rd_noise : array like or float or None
        Array of white noise values (std dev per frame) for each output channel, 
        or a single value. If an array, must match the number amplifier values 
        specified in `det.nchan`.
    u_pink : array like or float or None
        Array of uncorrelated pink noise (std dev per frame) for each output channel, 
        or a single value. If an array, must match the number amplifier values 
        specified in `det.nchan`.
    c_pink : float or None
        Standard deviation of the pink noise correlated between channels.
    pow_spec_corr : ndarray
        Option to input a custom power spectrum for the correlated noise.
    corr_scales : array like
        Instead of `pow_spec_corr`, input the scale factors of the two 1/f
        components [low freq, highfreq]).
    fcorr_lim : array like
        Low- and high- frequency cut-off points for `corr_scales` factors.
        The first element of `corr_scales` is applied to those frequencies
        below `fcorr_lim[0]`, while the second element corresponds to frequencies
        above `fcorr_lim[1]`.
    same_scan_direction : bool
        Are all the output channels read in the same direction?
        By default fast-scan readout direction is ``[-->,<--,-->,<--]``
        If ``same_scan_direction``, then all ``-->``
    reverse_scan_direction : bool
        If ``reverse_scan_direction``, then ``[<--,-->,<--,-->]`` or all ``<--``

    """
    
    from pynrc.reduce.calib import fit_corr_powspec, broken_pink_powspec
    import time

    nchan = det.nout
    nx = det.xpix
    ny = det.ypix
    chsize = det.chsize

    # Number of total frames up the ramp (including drops)
    ma     = det.multiaccum
    nd1    = ma.nd1
    nd2    = ma.nd2
    nf     = ma.nf
    ngroup = ma.ngroup
    nz     = nd1 + ngroup*nf + (ngroup-1)*nd2

    nroh = det._line_overhead
    nfoh = det._extra_lines
    
    result = np.zeros([nz,ny,nx])
                            
    # Make white read noise. This is the same for all pixels.
    if rd_noise is not None:
        # We want rd_noise to be an array or list
        if isinstance(rd_noise, (np.ndarray,list)):
            temp = np.asarray(rd_noise)
            if temp.size != nchan:
                _log.error('Number of elements in rd_noise not equal to n_out')
                return
        else: # Single value as opposed to an array or list
            rd_noise = np.ones(nchan) * rd_noise
    
        w = det.ref_info
        rr = ref_ratio #reference_pixel_noise_ratio 
        
        if np.any(rd_noise):
            _log.info('Generating read noise...')

            # Go frame-by-frame
            for z in np.arange(nz):
                here = np.zeros((ny,nx))

                # First assume no ref pixels and just add in random noise
                for ch in np.arange(nchan):
                    x1 = ch * chsize
                    x2 = x1 + chsize
                    here[:,x1:x2] = np.random.normal(scale=rd_noise[ch], size=(ny,chsize))

                # If there are reference pixels, overwrite with appropriate noise values
                # Noisy reference pixels for each side of detector
                rd_ref = rr * np.mean(rd_noise)
                if w[0] > 0: # lower
                    here[:w[0],:] = np.random.normal(scale=rd_ref, size=(w[0],nx))
                if w[1] > 0: # upper
                    here[-w[1]:,:] = np.random.normal(scale=rd_ref, size=(w[1],nx))
                if w[2] > 0: # left
                    here[:,:w[2]] = np.random.normal(scale=rd_ref, size=(ny,w[2]))
                if w[3] > 0: # right
                    here[:,-w[3]:] = np.random.normal(scale=rd_ref, size=(ny,w[3]))

                # Add the noise in to the result
                result[z,:,:] += here

                
    # Finish if no 1/f noise specified
    if (c_pink is None) and (u_pink is None) and (acn is None):
        return result
                
    #################################
    # 1/f noise
        
    ch_poh = chsize + nroh
    ny_poh = ny + nfoh

    # Compute the number of time steps per integration, per output
    nstep_frame = ch_poh * ny_poh
    nstep = nstep_frame * nz
    # Pad nsteps to a power of 2, which is much faster
    nstep2 = int(2**np.ceil(np.log2(nstep)))
    
    f2 = np.fft.rfftfreq(2*nstep2)
    f2[0] = f2[1] # First element should not be 0
    alpha = -1
    p_filter2 = np.sqrt(f2**alpha)
    p_filter2[0] = 0.
    
    # Add correlated pink noise.
    if (c_pink is not None) and (c_pink > 0):
        _log.info('Adding correlated pink noise...')

        if corr_scales is not None:
            scales = np.array(corr_scales)
            fcut1, fcut2 = np.array(fcorr_lim) / det._pixel_rate
            pf = broken_pink_powspec(f2, scales, fcut1=fcut1, fcut2=fcut2, alpha=alpha)
            pf[0] = 0
        elif pow_spec_corr is not None:
            n_ifft = 2 * (len(pow_spec_corr) - 1)
            freq_corr = np.fft.rfftfreq(n_ifft, d=1/det._pixel_rate)
            freq_corr[0] = freq_corr[1]
            # Fit power spectrum and remake for f2
            scales = fit_corr_powspec(freq_corr, pow_spec_corr, **kwargs)
            fcut1, fcut2 = np.array(fcorr_lim) / det._pixel_rate
            pf = broken_pink_powspec(f2, scales, fcut1=fcut1, fcut2=fcut2, alpha=alpha)
            pf[0] = 0
        else:
            pf = p_filter2

        tt = c_pink * pink_noise(nstep, pow_spec=pf)
        tt = tt.reshape([nz, ny_poh, ch_poh])[:,0:ny,0:chsize]
        _log.debug('  Corr Pink Noise (input, output): {:.2f}, {:.2f}'
              .format(c_pink, np.std(tt)))

        for ch in np.arange(nchan):
            x1 = ch*chsize
            x2 = x1 + chsize
            
            if same_scan_direction:
                flip = True if reverse_scan_direction else False
            elif np.mod(ch,2)==0:
                flip = True if reverse_scan_direction else False
            else:
                flip = False if reverse_scan_direction else True

            if flip: 
                result[:,:,x1:x2] += tt[:,:,::-1]
            else:
                result[:,:,x1:x2] += tt
        del tt

    # Add uncorrelated pink noise. Because this pink noise is stationary and
    # different for each output, we don't need to flip it (but why not?)
    if u_pink is not None:
        # We want u_pink to be an array or list
        if isinstance(u_pink, (np.ndarray,list)):
            temp = np.asarray(u_pink)
            if temp.size != nchan:
                _log.error('Number of elements in u_pink not equal to n_out')
                return
        else: # Single value as opposed to an array or list
            u_pink = np.ones(nchan) * u_pink

        # Only do the rest if any values are not 0
        if np.any(u_pink):
            _log.info('Adding uncorrelated pink noise...')
            
            for ch in range(nchan):
                x1 = ch*chsize
                x2 = x1 + chsize

                tt = u_pink[ch] * pink_noise(nstep, pow_spec=p_filter2)
                tt = tt.reshape([nz, ny_poh, ch_poh])[:,0:ny,0:chsize]
                _log.debug('  Ch{} Pink Noise (input, output): {:.2f}, {:.2f}'
                      .format(ch, u_pink[ch], np.std(tt)))

                if same_scan_direction:
                    flip = True if reverse_scan_direction else False
                elif np.mod(ch,2)==0:
                    flip = True if reverse_scan_direction else False
                else:
                    flip = False if reverse_scan_direction else True

                if flip: 
                    result[:,:,x1:x2] += tt[:,:,::-1]
                else:
                    result[:,:,x1:x2] += tt

                del tt

    # Add ACN
    if (acn is not None) and (acn>0):
        _log.info('Adding ACN noise...')

        facn = np.fft.rfftfreq(nstep2)
        facn[0] = facn[1] # First element should not be 0
        alpha = -2
        pf_acn = np.sqrt(facn**alpha)
        pf_acn[0] = 0.

        for ch in np.arange(nchan):
            x1 = ch*chsize
            x2 = x1 + chsize

            # Generate new pink noise for each even and odd vector.
            a = acn * pink_noise(int(nstep/2), pow_spec=pf_acn)
            b = acn * pink_noise(int(nstep/2), pow_spec=pf_acn)
            _log.debug('  Ch{} ACN Noise (input, [outa, outb]): {:.2f}, [{:.2f}, {:.2f}]'
                    .format(ch, acn, np.std(a), np.std(b)))

            # Reformat into an image.
            tt = np.reshape(np.transpose(np.vstack((a, b))),
                            (nz, ny_poh, ch_poh))[:, 0:ny, 0:chsize]

            if same_scan_direction:
                flip = True if reverse_scan_direction else False
            elif np.mod(ch,2)==0:
                flip = True if reverse_scan_direction else False
            else:
                flip = False if reverse_scan_direction else True

            if flip: 
                result[:,:,x1:x2] += tt[:,:,::-1]
            else:
                result[:,:,x1:x2] += tt

            del tt

    return result

def gen_dark_ramp(dark, out_shape, tf=10.73677, gain=1, ref_info=None,
                  avg_ramp=None):
    
    """
    Assumes a constant dark current rate, either in image form or single value.
    If gain is supplied, then input is assumed to be in DN/sec, otherwise e-/sec. 
    Output will be e-.

    Parameters
    ----------
    dark : ndarray or float
        Dark slope image or constant value. Assumed to be DN/sec. If gain=1,
        then also e-/sec. If this value is intended to be e-/sec, then simply
        set gain=1.
    out_shape : tuple, list, ndarray
        Desired shape of output ramp (nframes, ny, nx). If `dark` is an
        array, then dark.shape == out_shape[1:] == (ny,nx).
    tf : float
        Frame time in seconds
    gain : float
        Gain of detector in e-/sec. If specified to be other than 1, then we
        assume `dark` to be in units of DN/sec.
    avg_ramp : ndarray

    """

    nz, ny, nx = out_shape

    if avg_ramp is not None:
        assert len(avg_ramp)>=nz, "avg_ramp size must be >= to number of requested frames (out_shape[0])"
    
    # Count accumulation for a single frame (e-)
    dark_frame = np.ones([ny,nx]) * dark * tf * gain

    # Set negative values to median
    med = np.median(dark_frame)
    med = 0 if med<0 else med
    dark_frame[dark_frame<0] = med

    # Return an array of 0s if all dark current is 0
    if np.all(dark_frame==0):
        result = np.zeros(out_shape)
    else:
        # Add Poisson noise at each frame step
        result = np.random.poisson(lam=dark_frame, size=out_shape).astype('float')
        # Perform cumulative sum in place
        result = np.cumsum(result, axis=0, out=result)

    # Modulate "ideal" slope by emperical "average ramp" behavior
    if avg_ramp is not None:
        tarr = np.arange(1,nz+1)*tf
        avg_dark = np.median(dark)
        del_ramp = avg_ramp[0:nz] - avg_dark*tarr   # DN
        result += gain * del_ramp.reshape([-1,1,1]) # e-
            
    # Set reference pixels' dark current equal to 0
    if ref_info is not None:
        w = ref_info
        if w[0] > 0: # lower
            result[:,:w[0],:] = 0
        if w[1] > 0: # upper
            result[:,-w[1]:,:] = 0
        if w[2] > 0: # left
            result[:,:,:w[2]] = 0
        if w[3] > 0: # right
            result[:,:,-w[3]:] = 0
            
    # Return in units of e-
    return result 

def sim_dark_ramp(det, super_dark, ramp_avg_ch=None, ramp_avg_tf=10.73677, 
    out_ADU=False, verbose=False, **kwargs):
    """
    Simulate a dark current ramp based on input det class and
    super dark image.

    Parameters
    ----------
    det : Detector Class
        Desired detector class output
    super_dark : ndarray
        Dark current input image
    
    Keyword Args
    ------------
    ramp_avg_ch : ndarray or None
        Time-dependent flux of average dark ramp for each amplifier channel.
    ramp_avg_tf : float
        Delta time between between `ramp_avg_ch` points.
    out_ADU : bool
        Divide by gain to get value in ADU (float).
    verbose : bool
        Print some messages.
    """
    
    nchan = det.nout
    nx, ny = (det.xpix, det.ypix)
    chsize = det.chsize
    tf = det.time_frame
    gain = det.gain
    ref_info = det.ref_info

    # Do we need to crop out subarray?
    if super_dark.shape[0]==ny:
        y1, y2 = (0, ny)
    else: # Will crop a subarray out of super_dark image
        y1 = det.y0
        y2 = int(y1 + ny)

    # Number of total frames up the ramp (including drops)
    ma     = det.multiaccum
    nd1    = ma.nd1
    nd2    = ma.nd2
    nf     = ma.nf
    ngroup = ma.ngroup
    nz     = nd1 + ngroup*nf + (ngroup-1)*nd2

    # Interpolate ramp_avg_ch onto tarr grid
    if (ramp_avg_ch is not None):
        tarr = np.arange(1,nz+1) * tf
        if tarr.max() < ramp_avg_tf:
            if verbose:
                msg = "Max ramp time {:.1f} is less than ramp_avg_tf. \
                    Not applying ramp_avg_ch.".format(tarr.max())
                _log.warn(msg)
            ramp_avg_ch = None
        else:
            # Insert 0 DN at t=0
            tvals = np.arange(0,ramp_avg_ch.shape[1]) * ramp_avg_tf
            ramp_avg_ch = np.insert(ramp_avg_ch, 0,0, axis=0)

            # Interpolate onto new time grid
            ramp_avg_ch_new = []
            for ramp_avg in ramp_avg_ch:
                avg_interp = np.interp(tarr, tvals, ramp_avg)
                ramp_avg_ch_new.append(avg_interp)
            ramp_avg_ch = np.array(ramp_avg_ch_new)

    if verbose:
        _log.info('Generating dark current ramp...')

    res = np.zeros([nz,ny,nx])
    for ch in np.arange(nchan):
        if nchan==1: # Subarray window case
            if super_dark.shape[1]==nx:
                x1, x2 = (0, nx)
            else: # Will crop a subarray out of super_dark image
                x1 = det.x0
                x2 = int(x1 + nx)
        else: # STRIPE or FULL frame
            x1 = ch*chsize 
            x2 = x1 + chsize

        dark = super_dark[y1:y2,x1:x2]
        
        avg_ramp = None if ramp_avg_ch is None else ramp_avg_ch[ch]
        res[:,:,x1:x2] = gen_dark_ramp(dark, (nz,ny,chsize), gain=gain, tf=tf,
                                       avg_ramp=avg_ramp, ref_info=None)

    if out_ADU:
        res /= gain
        
    # Set reference pixels' dark current equal to 0
    if ref_info is not None:
        w = ref_info
        if w[0] > 0: # lower
            res[:,:w[0],:] = 0
        if w[1] > 0: # upper
            res[:,-w[1]:,:] = 0
        if w[2] > 0: # left
            res[:,:,:w[2]] = 0
        if w[3] > 0: # right
            res[:,:,-w[3]:] = 0
        
    return res

def sim_image_ramp(det, im_slope, verbose=False, **kwargs):
    """
    Simulate an image ramp based on input det class and slope image.
    Uses the `sim_dark_ramp` function.

    Parameters
    ----------
    det : Detector Class
        Desired detector class output
    im_slope : ndarray
        Input slope image
    
    Keyword Args
    ------------
    out_ADU : bool
        Divide by gain to get value in ADU (float).
    verbose : bool
        Print some messages.
    """
    if verbose:
        _log.info('Generating image acquisition ramp...')

    return sim_dark_ramp(det, im_slope, ramp_avg_ch=None, verbose=False, **kwargs)


def simulate_detector_ramp(det, dark_cal_obj, im_slope=None, out_ADU=False,
                           include_dark=True, include_bias=True, include_ktc=True, 
                           include_rn=True, include_cpink=True, include_upink=True, 
                           include_acn=True, apply_ipc=True, apply_ppc=True, 
                           include_refinst=True, include_colnoise=True, col_noise=None,
                           amp_crosstalk=True, add_crs=True, latents=None, linearity_map=None, 
                           return_zero_frame=None, return_full_ramp=False, prog_bar=False):
    
    """ Return a single simulated ramp
    
    The output will be in raw detector coordiantes.
    
    Parameters
    ==========
    det : Detector Class
        Desired detector class output
    dark_cal_obj: nircam_dark class
        NIRCam Dark class that holds the necessary calibration info to
        simulate a dark ramp.
    im_slope : ndarray
        Input slope image of observed scene. Assumed to be in detector
        coordinates.

    Keyword Args
    ============
    return_zero_frame : bool or None
        For DMS data, particularly readout patterns with averaged frames,
        this returns the very first raw read in the ramp.
    return_full_ramp : bool
        By default, we average groups and drop frames as specified in the
        `det` input. If this keyword is set to True, then return all raw
        frames within the ramp. The last set of `nd2` frames will be omitted.
    out_ADU : bool
        If true, divide by gain and convert to 16-bit UINT.
    include_dark : bool
        Add dark current?
    include_bias : bool
        Add detector bias?
    include_ktc : bool
        Add kTC noise?
    include_rn : bool
        Add readout noise per frame?
    include_cpink : bool
        Add correlated 1/f noise to all amplifiers?
    include_upink : bool
        Add uncorrelated 1/f noise to each amplifier?
    include_acn : bool
        Add alternating column noise?
    apply_ipc : bool
        Include interpixel capacitance?
    apply_ppc : bool
        Apply post-pixel coupling to linear analog signal?
    include_refinst : bool
        Include reference/active pixel instabilities?
    include_colnoise : bool
        Add in column noise per integration?
    col_noise : ndarray or None
        Option to explicitly specifiy column noise distribution in
        order to shift by one for subsequent integrations
    amp_crosstalk : bool
        Crosstalk between amplifiers?
    add_crs : bool
        Add cosmic ray events?
    latents : None
        Apply persistence.
    linearity_map : ndarray
        Add non-linearity.
    prog_bar : bool
        Show a progress bar for this ramp generation?
    """
    
    ################################
    # Dark calibration properties
    dco = dark_cal_obj

    # Super bias and darks
    super_bias = dco.super_bias_deconv # DN
    super_dark = dco.super_dark_deconv # DN/sec

    # IPC/PPC kernel information
    k_ipc = dco.kernel_ipc
    k_ppc = dco.kernel_ppc
    
    # Noise info
    cds_dict = dco.cds_act_dict
    keys = ['spat_det', 'spat_pink_corr', 'spat_pink_uncorr']
    cds_vals = [np.sqrt(np.mean(cds_dict[k]**2, axis=0)) for k in keys]
    # CDS Noise values
    rd_noise_cds, c_pink_cds, u_pink_cds = cds_vals
    # Noise per frame
    rn, cp, up = cds_vals / np.sqrt(2)
    acn = 1

    # kTC Reset Noise
    ktc_noise = dco.ktc_noise

    # Power spectrum for correlated noise
    # freq = dco.pow_spec_dict['freq']
    scales = dco._pow_spec_dict['ps_corr_scale']
    # pcorr_fit = broken_pink_powspec(freq, scales)

    # Reference noise info
    ref_ratio = np.mean(dco.cds_ref_dict['spat_det'] / dco.cds_act_dict['spat_det'])
    
    ################################
    # Detector output configuration

    # Detector Gain
    gain = det.gain
    
    # Pixel readout
    nchan = det.nout
    ny, nx = (det.ypix, det.xpix)
    x1, x2 = (det.x0, det.x0 + nx)
    y1, y2 = (det.y0, det.y0 + ny)
    
    # Crop super bias and super dark for subarray observations
    super_bias = super_bias[y1:y2,x1:x2]
    super_dark = super_dark[y1:y2,x1:x2]
    
    # Number of total frames up the ramp (including drops)
    ma     = det.multiaccum
    nd1    = ma.nd1
    nd2    = ma.nd2
    nf     = ma.nf
    ngroup = ma.ngroup
    nz     = nd1 + ngroup*nf + (ngroup-1)*nd2

    # Scan direction info
    ssd = det.same_scan_direction
    rsd = det.reverse_scan_direction

    # Number of reference pixels (lower, upper, left, right)
    ref_info = det.ref_info

    
    ################################
    # Begin...
    ################################
    if prog_bar: pbar = tqdm(total=13, leave=False)
    data = np.zeros([nz,ny,nx])

    ####################
    # Create a super dark ramp (Units of e-)
    # Average shape of ramp
    if prog_bar: pbar.set_description("Dark Current")
    ramp_avg_ch = dco.dark_ramp_dict['ramp_avg_ch']
    # Create dark (adds Poisson noise)
    if include_dark:
        data += sim_dark_ramp(det, super_dark, ramp_avg_ch=ramp_avg_ch, verbose=False)
    if prog_bar: pbar.update(1)

    ####################
    # Add on-sky source image
    if prog_bar: pbar.set_description("Sky Image")
    if im_slope is not None:
        data += sim_image_ramp(det, im_slope, verbose=False)
    if prog_bar: pbar.update(1)

    ####################
    # TODO: Add cosmic rays
    if prog_bar: pbar.set_description("Cosmic Rays")
    if add_crs:
        pass
    if prog_bar: pbar.update(1)
    
    ####################
    # TODO: Apply persistence/latent image
    if prog_bar: pbar.set_description("Persistence")
    if latents is not None:
        pass
    if prog_bar: pbar.update(1)
    
    ####################
    # Apply IPC (before or after non-linearity??)
    if prog_bar: pbar.set_description("Include IPC")
    if apply_ipc:
        data = add_ipc(data, kernel=k_ipc)
    if prog_bar: pbar.update(1)

    ####################
    # TODO: Add non-linearity
    if prog_bar: pbar.set_description("Non-Linearity")
    # Truncate anything above well level
    # This wwill be part of non-linearity after full implementation
    if linearity_map is not None:
        pass
    well_max = det.well_level
    data[data>well_max] = well_max
    if prog_bar: pbar.update(1)
    
    ####################
    # Add kTC noise:
    if prog_bar: pbar.set_description("kTC Noise")
    if include_ktc:
        ktc_offset = gain * np.random.normal(scale=ktc_noise, size=(ny,nx))
        data += ktc_offset
    if prog_bar: pbar.update(1)
        
    ####################
    # Add super bias
    if prog_bar: pbar.set_description("Super Bias")
    if include_bias:
        data += gain * super_bias
    if prog_bar: pbar.update(1)
    
    ####################
    # Apply PPC (when should this occur?)
    if prog_bar: pbar.set_description("Include PPC")
    if apply_ppc:
        data = add_ppc(data, nchans=nchan, kernel=k_ppc, in_place=True,
                       same_scan_direction=ssd, reverse_scan_direction=rsd)
    if prog_bar: pbar.update(1)
    
    ####################
    # TODO: Add channel crosstalk
    if prog_bar: pbar.set_description("Amplifier Crosstalk")
    if amp_crosstalk:
        pass
    if prog_bar: pbar.update(1)

    ####################
    # Add read and 1/f noise
    if prog_bar: pbar.set_description("Detector and ASIC Noise")
    if nchan==1:
        rn, up = (rn[0], up[0])
    rn  = None if (not include_rn)    else rn
    up  = None if (not include_upink) else up
    cp  = None if (not include_cpink) else cp*1.2
    acn = None if (not include_acn)   else acn
    data += gain * sim_noise_data(det, rd_noise=rn, u_pink=up, c_pink=cp,
                                  acn=acn, corr_scales=scales, ref_ratio=ref_ratio,
                                  same_scan_direction=ssd, reverse_scan_direction=rsd)
    if prog_bar: pbar.update(1)

    ####################
    # Add reference offsets
    if prog_bar: pbar.set_description("Ref Pixel Offsets")
    if include_refinst:
        data += gain * gen_ramp_biases(dco.ref_pixel_dict, nchan=nchan, 
                                       data_shape=data.shape, ref_border=det.ref_info)
    if prog_bar: pbar.update(1)

    ####################
    # Add column noise
    if prog_bar: pbar.set_description("Column Noise")
    # Passing col_noise allows for shifting of noise 
    # by one col ramp-to-ramp in higher level function
    if include_colnoise and (col_noise is None):
        col_noise = gain * gen_col_noise(dco.column_variations, 
                                         dco.column_prob_bad, 
                                         nz=nz, nx=nx)
    elif (include_colnoise==False):
        col_noise = 0
    # Add to data
    data += col_noise
    if prog_bar: pbar.update(1)

    # Convert to DN (16-bit int)
    if out_ADU:
        data /= gain
        data[data < 0] = 0
        data[data >= 2**16] = 2**16 - 1
        data = data.astype('uint16')

    if prog_bar: pbar.close()
    
    # return_zero_frame not set, the True if not RAPID (what about BRIGHT1??)
    if return_zero_frame is None:
        return_zero_frame = False if 'RAPID' in det.multiaccum.read_mode else True
        
    if return_full_ramp:
        if return_zero_frame:
            return data, data[0].copy()
        else:
            return data
    else:
        return ramp_resample(data, det, return_zero_frame=return_zero_frame)


# npix = 20
# arr = np.random.rand(npix) * 4000
# arr[1] = 0
# #arr[-1] = 2**16-1
# 
# pixel_time = 10. # usec
# 
# darr = np.roll(arr,-1) - arr
# tarr = np.arange(100.) / pixel_time
# tau = 1.5
# vt = darr.reshape([-1,1])*(1-np.exp(-tarr.reshape([1,-1])/tau))
# vt = vt.ravel()
# 
# arr_full = arr.repeat(len(tarr))
# arr_new = np.roll(arr_full + vt, len(tarr))
# 
# tfull = np.arange(len(arr_new)) * pixel_time / len(tarr)
# 
# # Pixel sample times
# t_sample = np.arange(npix)*pixel_time + 6.
# v_sample = np.interp(t_sample, tfull, arr_new)
# 
# fig, ax = plt.subplots(1,1, figsize=(12,5))
# ax.plot(tfull, arr_full)#[0:-20])
# ax.plot(tfull, arr_new)
# ax.xaxis.get_major_locator().set_params(nbins=9, steps=[1, 2, 5, 10])
# 
# fig.tight_layout()
# 
# from astropy.convolution import convolve
# alpha = 0.018
# kernel = np.array([0.0, 1-alpha, alpha])
# arr_ppc = convolve(arr, kernel)
# 
# 
# arr[1] = 0
# arr[-1] = 2**16-1
# # Convert to usec timestep
# arr = arr.repeat(10)
# 
# from scipy.ndimage import gaussian_filter1d
# res = gaussian_filter1d(arr, 1.5)
# 
# plt.clf()
# plt.plot(arr[0:-20])
# plt.plot(res[0:-20])
# 
# 
# res = arr.copy()
# res = res.reshape([npix,-1])
# res[:, 0:3] = np.nan
# res[:, -3:] = np.nan
# res = res.ravel()
# 
# im_mask = np.isnan(res)
# x = mask_helper() # Returns the nonzero (True) indices of a mask
# res[im_mask]= np.interp(x(im_mask), x(~im_mask), res[~im_mask])
# 
# plt.clf()
# plt.plot(arr[0:-20])
# plt.plot(res[0:-20])
# 
# 
# from scipy.signal import butter, filtfilt
# b, a = butter(3, 0.2, btype='lowpass', analog=False)
# res = filtfilt(b, a, arr)
# 
# from scipy.signal import savgol_filter
# winsize = 5
# res2 = savgol_filter(arr, winsize, 3)
# 
# plt.clf()
# plt.plot(arr)
# plt.plot(res)
# #plt.plot(res2)
# 
