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
	- 
"""
# Necessary for Python 2.6 and later
from __future__ import division, print_function

import numpy as np
from astropy.io import fits
import datetime
import os

# HxRG Noise Generator
from pynrc import nghxrg as ng

from nrc_utils import nrc_header
from pynrc_core import DetectorOps
from . import conf

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
	dark=True, bias=True, out_ADU=False, verbose=False, use_fftw=False, ncores=None):
	"""
	Create a data cube consisting of realistic NIRCam detector noise.

	This is essentially a wrapper for nghxrg.py that selects appropriate values
	for a specified SCA in order to reproduce realistic noise properties similiar
	to those measured during ISIM CV3.

	Parameters
	----------
	det    : Option to specify already existing NIRCam detector object class
		     Otherwise, use scaid and params
	scaid  : NIRCam SCA number (481, 482, ..., 490)
	params : A set of MULTIACCUM parameters such as:
		params = {'ngroup': 2, 'wind_mode': 'FULL', 
				'xpix': 2048, 'ypix': 2048, 'x0':0, 'y0':0}
		wind_mode can be FULL, STRIPE, or WINDOW
	file_out : Folder name and destination to place optional FITS output. 
		A timestamp will be appended to the end of the file name (and before .fits').
	caldir : Directory location housing the super bias and super darks for each SCA.
	dark : Use super dark? If True, then reads in super dark slope image.
	bias : Use super bias? If True, then reads in super bias image.
	out_ADU : Noise values are calculated in terms of equivalent electrons. This
		gives the option of converting to ADU (True) or keeping in term of e- (False).
		ADU values are converted to 16-bit UINT. Keep in e- if applying to a ramp
		observation then convert combined data to ADU later.
	
	Returns 
	----------
	Primary HDU with noise ramp in hud.data and header info in hdu.header.

	Examples 
	----------
	import ngNRC
	params = {'ngroup': 108, 'wind_mode': 'FULL', 
			'xpix': 2048, 'ypix': 2048, 'x0':0, 'y0':0}
			
	# Output to a file
	scaid = 481
	caldir = '/data/darks_sim/nghxrg/sca_images/'
	file_out = '/data/darks_sim/dark_sim_481.fits'
	hdu = ngNRC.SCAnoise(scaid, params, file_out=file_out, caldir=caldir, \
		dark=True, bias=True, out_ADU=True, use_fftw=False, ncores=None, verbose=False)

	# Don't save file, but keep hdu in e- for adding to simulated observation ramp
	scaid = 481
	caldir = '/data/darks_sim/nghxrg/sca_images/'
	hdu = ngNRC.SCAnoise(scaid, params, file_out=None, caldir=caldir, \
		dark=True, bias=True, out_ADU=False, use_fftw=False, ncores=None, verbose=False)

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
	nfoh     = det._extra_lines[0]
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

	hdu.header = nrc_header(det, header=hdu.header)
	hdu.header['UNITS'] = 'ADU' if out_ADU else 'e-'

	# Write the result to a FITS file
	if file_out is not None:
		now = datetime.datetime.now().isoformat()[:-7]
		hdu.header['DATE'] = datetime.datetime.now().isoformat()[:-7]
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

