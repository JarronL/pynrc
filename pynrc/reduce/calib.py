import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Import libraries
import numpy as np
import os, gzip, json
from copy import deepcopy
from scipy import ndimage

from astropy.io import fits
from astropy.modeling import models, fitting

# Multiprocessing
import multiprocessing as mp

# Program bar
from tqdm.auto import trange, tqdm

from webbpsf_ext import robust
from webbpsf_ext.image_manip import fshift, pad_or_cut_to_size, bp_fix
from webbpsf_ext.maths import hist_indices, jl_poly_fit, jl_poly
from ..maths.coords import det_to_sci, sci_to_det 

# import pynrc
from ..nrc_utils import var_ex_model
from ..reduce.ref_pixels import reffix_hxrg, channel_smooth_savgol, channel_averaging

from .. import conf, DetectorOps
from ..detops import create_detops

from ..logging_utils import setup_logging
import logging

from pynrc import logging_utils
_log = logging.getLogger('pynrc')

def detname_to_scaid(det_id):

    from ..nrc_utils import get_detname
    detname = get_detname(det_id)

    det_dict = {'NRCA1':481, 'NRCA2':482, 'NRCA3':483, 'NRCA4':484, 'NRCA5':485,
                'NRCB1':486, 'NRCB2':487, 'NRCB3':488, 'NRCB4':489, 'NRCB5':490}

    return det_dict[detname]

class nircam_dark(object):
    """NIRCam dark calibration"""

    def __init__(self, scaid, datadir, outdir, lindir=None, DMS=False, 
                 same_scan_direction=False, reverse_scan_direction=False):
        
        self.DMS = DMS

        # In case detname (A1...B5) is specified rather than scaid (481...490)
        self.scaid = detname_to_scaid(scaid)
        # Directory information
        self._create_dir_structure(datadir, outdir, lindir=lindir)

        # Get header information and create a NIRCam detector timing instance
        hdr = self._grab_single_header()
        self.det = create_detops(hdr, DMS=DMS)
        self.det.same_scan_direction = same_scan_direction
        self.det.reverse_scan_direction = reverse_scan_direction

        # Get temperature information
        self._grab_temperature_data()

        self._init_attributes()
        
    def _init_attributes(self):

        # Create masks for ref pixels, active pixels, and channels
        self._create_pixel_masks()

        # Initialize superbias and superdark attributes
        self._super_bias = None
        self._super_bias_sig = None
        self._super_dark = None
        self._super_dark_sig = None
        self._super_dark_ramp = None
        self._super_dark_deconv = None
        self._super_bias_deconv = None
        self._dark_ramp_dict = None
        self._pixel_masks = None

        # IPC info
        self._kernel_ipc = None
        self._kernel_ppc = None
        self._kernel_ipc_sig = None
        self._kernel_ppc_sig = None

        # Noise info
        self._ktc_noise = None
        self._cds_act_dict = None
        self._cds_ref_dict = None
        self._eff_noise_dict = None
        self._pow_spec_dict = None

        # Reference pixel properties
        self._ref_pixel_dict = None

        # Column variations
        self._column_variations = None
        self._column_prob_bad   = None

        # Non-linearity coefficients
        self.linear_dict = None
        self.nonlinear_dict = None

        # Flat field info
        self.lflats = None  # Low frequency spatial variations
        self.pflats = None  # High frequency variations (cross hatch)

    # Directory and files
    @property
    def datadir(self):
        return self.paths_dict['datadir']
    @property
    def lindir(self):
        return self.paths_dict['lindir']
    @property
    def outdir(self):
        return self.paths_dict['outdir']
    @property
    def allfiles(self):
        return self.paths_dict['allfiles']
    @property
    def linfiles(self):
        return self.paths_dict['linfiles']

    # Temperature information
    @property
    def temperature_dict(self):
        return self._temperature_dict

    @property
    def time_arr(self):
        return self.det.times_group_avg

    # Ramp shapes and sizes
    @property
    def dark_shape(self):
        """Shape of dark ramps"""
        nx = self.det.xpix
        ny = self.det.ypix
        nz = self.det.multiaccum.ngroup
        return (nz,ny,nx)
    @property
    def nchan(self):
        """Number of output channels"""
        return self.det.nout
    @property
    def nchans(self):
        """Number of output channels"""
        return self.det.nout
    @property
    def chsize(self):
        """Width of output channel"""
        return self.det.chsize

    # Array masks
    @property
    def mask_ref(self):
        return self._mask_ref
    @property
    def mask_act(self):
        if self.mask_ref is None:
            return None
        else:
            return ~self.mask_ref
    @property
    def mask_channels(self):
        return self._mask_channels

    # Bias and dark slope information
    @property
    def super_bias(self):
        return self._super_bias
    @property
    def super_bias_deconv(self):
        return self._super_bias_deconv 
    @property
    def super_dark(self):
        return self._super_dark
    @property
    def super_dark_deconv(self):
        return self._super_dark_deconv
    @property
    def super_dark_ramp(self):
        return self._super_dark_ramp
    @property
    def dark_ramp_dict(self):
        return self._dark_ramp_dict

    # Column variations
    @property
    def ref_pixel_dict(self):
        return self._ref_pixel_dict

    # Column variations
    @property
    def column_variations(self):
        return self._column_variations
    @property
    def column_prob_bad(self):
        return self._column_prob_bad

    # IPC/PPC Kernel info
    @property
    def kernel_ipc(self):
        return self._kernel_ipc
    @property
    def ipc_alpha_frac(self):
        """Fractional IPC value (alpha)"""
        if self.kernel_ipc is None:
            return None
        else:
            return self.kernel_ipc[1,2]
    @property
    def kernel_ppc(self):
        return self._kernel_ppc
    @property
    def ppc_frac(self):
        """Fractional PPC value"""
        if self.kernel_ppc is None:
            return None
        else:
            return self.kernel_ppc[1,2]

    @property
    def ktc_noise(self):
        return self._ktc_noise
    @property
    def cds_act_dict(self):
        return self._cds_act_dict
    @property
    def cds_ref_dict(self):
        return self._cds_ref_dict
    @property
    def eff_noise_dict(self):
        return self._eff_noise_dict
    @property
    def pow_spec_dict(self):
        return self._pow_spec_dict

    def _create_dir_structure(self, datadir, outdir, lindir=None):
        """ Directories and files"""

        scaid = self.scaid
        # Add SCA ID to output directory path
        outbase = outdir
        if str(scaid) in outbase:
            outdir = outbase
        else:
            outdir = os.path.join(outbase, str(scaid)) + '/'        

        # Directory information
        if datadir is None:
            allfiles = None
        else:
            indir = os.path.join(datadir, str(scaid)) + '/'
            # Get file names within directory
            allfits = [file for file in os.listdir(indir) if file.endswith('.fits')]
            allfits = np.sort(allfits)
            # Add directory 
            allfiles = [indir + f for f in allfits]

        if lindir is None:
            linfiles = None
        else:
            # Directory information
            indir = os.path.join(lindir, str(scaid)) + '/'
            # Get file names within directory
            linfits = [file for file in os.listdir(indir) if file.endswith('.fits')]
            linfits = np.sort(linfits)
            # Add directory 
            linfiles = [indir + f for f in linfits]

        # Directory to save figures for analysis
        figdir = os.path.join(outdir, 'FIGURES') + '/'
        # figdir = os.path.join(outdir, str(scaid)) + '/'        

        # Directories to save super bias and super dark info
        super_bias_dir = os.path.join(outdir, 'SUPER_BIAS') + '/'
        super_dark_dir = os.path.join(outdir, 'SUPER_DARK') + '/'
        noise_dir      = os.path.join(outdir, 'NOISE') + '/'
        power_spec_dir = os.path.join(outdir, 'POWER_SPEC') + '/'
        linearity_dir  = os.path.join(outdir, 'LINEARITY') + '/'


        # Make sure directories exist for writing
        for path in [outbase, outdir, figdir, super_bias_dir, super_dark_dir, noise_dir, linearity_dir]:
            if not os.path.exists(path):
                os.mkdir(path)

        self.paths_dict = {
            'datadir ' : datadir,
            'allfiles' : allfiles,
            'linfiles' : linfiles,
            'outdir'   : outdir,
            'figdir'   : figdir,
            'header_file'         : outdir + f'HEADER_{scaid}.TXT',
            'temperatures_file'   : outdir + f'TEMPERATURES_{scaid}.JSON',
            'super_bias_dir'      : super_bias_dir,
            'super_dark_dir'      : super_dark_dir,
            'super_bias_init'     : super_bias_dir + f'SUPER_BIAS_INIT_{scaid}.FITS',
            'super_bias'          : super_bias_dir + f'SUPER_BIAS_{scaid}.FITS',
            'super_dark_ramp'     : super_dark_dir + f'SUPER_DARK_RAMP_{scaid}.FITS',
            'super_dark'          : super_dark_dir + f'SUPER_DARK_{scaid}.FITS',
            'super_dark_ramp_avgs': super_dark_dir + f'SUPER_DARK_RAMP_AVGS_{scaid}.npz',
            'kernel_ipc'          : super_dark_dir + f'KERNEL_IPC_{scaid}.FITS',
            'kernel_ppc'          : super_dark_dir + f'KERNEL_PPC_{scaid}.FITS',
            'pixel_masks'         : super_dark_dir + f'PIXEL_MASKS_{scaid}.FITS.gz',
            'column_variations'   : super_dark_dir + f'SUPER_DARK_COLVAR_{scaid}.FITS',
            'ref_pix_variations'  : super_bias_dir + f'BIAS_BEHAVIOR_{scaid}.JSON',
            'cds_act_dict'        : noise_dir      + f'CDS_NOISE_ACTIVE_{scaid}.JSON',
            'cds_ref_dict'        : noise_dir      + f'CDS_NOISE_REF_{scaid}.JSON',
            'eff_noise_dict'      : noise_dir      + f'EFF_NOISE_{scaid}.JSON',
            'power_spec_cds'      : noise_dir      + f'POWER_SPEC_CDS_{scaid}.npy',
            'power_spec_full'     : noise_dir      + f'POWER_SPEC_FULL_{scaid}.npy',
            'power_spec_cds_oh'   : noise_dir      + f'POWER_SPEC_CDS_OH_{scaid}.npy',
            'power_spec_full_oh'  : noise_dir      + f'POWER_SPEC_FULL_OH_{scaid}.npy',
            'power_spec_cds_pix'  : noise_dir      + f'POWER_SPEC_CDS_PIX_{scaid}.npy',
            'power_spec_full_pix' : noise_dir      + f'POWER_SPEC_FULL_PIX_{scaid}.npy',
            'linear_coeffs'       : linearity_dir  + f'LINEAR_COEFFS_{scaid}.npz',
            'nonlinear_coeffs'    : linearity_dir  + f'NONLINEAR_COEFFS_{scaid}.npz',
            'super_flats'         : linearity_dir  + f'SUPER_FLATS_{scaid}.FITS',
        }

    def _create_pixel_masks(self):

        # Array masks
        # self.mask_act is just ~self.mask_ref
        self._mask_ref = self.det.mask_ref
        self._mask_channels = self.det.mask_channels

    def _dict_to_json(self, in_dict, savename):
        # Save reference pixel dictionary
        dtemp = deepcopy(in_dict)
        # Convert any ndarrays to lists
        for k in dtemp.keys():
            if isinstance(dtemp[k], (np.ndarray)):
                dtemp[k] = dtemp[k].tolist()
        with open(savename, 'w') as fp:
            json.dump(dtemp, fp, sort_keys=False, indent=4)

    def _json_to_dict(self, savename):
        # Load from JSON files
        with open(savename, 'r') as fp:
            d = json.load(fp)

        # Convert any lists to np.array
        for k in d.keys():
            if isinstance(d[k], (list)):
                d[k] = np.array(d[k])

        return d

    def _grab_single_header(self):
        """Read/save or Open header of first FITS file"""
        from astropy.io.fits import Header

        savename = self.paths_dict['header_file']
        file_exists = os.path.isfile(savename)

        if file_exists:
            hdr = Header.fromtextfile(savename)
        else:
            hdr = fits.getheader(self.allfiles[0])
            hdr.totextfile(savename, overwrite=True)

        return hdr

    def _grab_temperature_data(self):
        """ Grab temperature data from headers
        
        Creates a dictionary that houses the temperature
        info stored in the headers of each FITS file.
        """

        # TODO: Add DMS support for temperature
        if self.DMS:
            self._temperature_dict = None
            _log.warning("DMS data not yet supported obtaining FPA temperatures")
            return
        
        savename = self.paths_dict['temperatures_file']
        file_exists = os.path.isfile(savename)

        if file_exists:
            # Load from JSON files
            temperature_dict = self._json_to_dict(savename)
        else:
            # Get initial temperature keys
            hdr = self._grab_single_header()

            tkeys = [k for k in list(hdr.keys()) if k[0:2]=='T_'] + ['ASICTEMP']

            # Initialize lists for each temperature key
            temperature_dict = {}
            for k in tkeys:
                temperature_dict[k] = []
                
            for f in self.allfiles:
                hdul = fits.open(f)
                hdr = hdul[0].header
                for k in tkeys:
                    temperature_dict[k].append(float(hdr[k]))
                hdul.close()

            # Save temperature dictionary
            self._dict_to_json(temperature_dict, savename)

        self._temperature_dict = temperature_dict

    def get_super_bias_init(self, deg=1, nsplit=2, force=False, **kwargs):

        _log.info("Generating initial super bias")

        allfiles = self.allfiles

        savename = self.paths_dict['super_bias_init']
        file_exists = os.path.isfile(savename)

        if file_exists and (not force):
            super_bias, super_bias_sig = get_fits_data(savename)
        else:
            
            # Default ref pixel correction kw args
            kwargs_def = {
                'nchans': self.nchan, 'altcol': True, 'in_place': True,    
                'fixcol': True, 'avg_type': 'pixel', 'savgol': True, 'perint': False    
            }
            for k in kwargs_def.keys():
                if k not in kwargs:
                    kwargs[k] = kwargs_def[k]

            res = gen_super_bias(allfiles, deg=deg, nsplit=nsplit, DMS=self.DMS,
                                 return_std=True, **kwargs)
            super_bias, super_bias_sig = res

            # Save superbias frame to directory
            hdu = fits.PrimaryHDU(np.array([super_bias, super_bias_sig]))
            hdu.writeto(savename, overwrite=True)

        self._super_bias = super_bias
        self._super_bias_sig = super_bias_sig

    def get_super_bias_update(self, force=False, **kwargs):
        # Make sure initial super bias exists
        if (self._super_bias is None) or (self._super_bias_sig is None):
            self.get_super_bias_init(**kwargs)

        # File names
        fname = self.paths_dict['super_bias']
        file_exists = os.path.isfile(fname)
        if file_exists and (not force):
            # Grab updated Super Bias
            _log.info("Opening updated super bias")
            self._super_bias = get_fits_data(fname)
        else:
            # Generate Super Bias along with dark ramp and pixel masks
            self.get_super_dark_ramp(force=force, **kwargs)

    def get_super_dark_ramp(self, force=False, **kwargs):
        """Create or read super dark ramp and update super bias"""

        # Make sure initial super bias exists
        if (self._super_bias is None) or (self._super_bias_sig is None):
            self.get_super_bias_init(**kwargs)

        _log.info("Creating super dark ramp cube, updated super bias, and pixel mask info")

        # File names
        fname_super_dark_ramp = self.paths_dict['super_dark_ramp']
        fname_super_bias      = self.paths_dict['super_bias']
        fname_pixel_mask      = self.paths_dict['pixel_masks']

        file_exists = os.path.isfile(fname_super_dark_ramp)
        if file_exists and (not force):

            # Grab Super Dark Ramp
            super_dark_ramp = get_fits_data(fname_super_dark_ramp)
            
            # Grab updated Super Bias
            super_bias = get_fits_data(fname_super_bias)
            
            # Generate pixel masks dictionary
            masks_dict = {}
            hdul = fits.open(fname_pixel_mask)
            for hdu in hdul:
                key = hdu.name.lower()
                masks_dict[key] = hdu.data.astype('bool')
            hdul.close()
        else:
            allfiles = self.allfiles

            # Default kwargs to run
            kwargs_def = {
                'nchans': self.nchan, 'altcol': True, 'in_place': True,    
                'fixcol': True, 'avg_type': 'pixel', 'savgol': True, 'perint': False    
            }
            for k in kwargs_def.keys():
                if k not in kwargs:
                    kwargs[k] = kwargs_def[k]

            res = gen_super_dark(allfiles, super_bias=self.super_bias, DMS=self.DMS, **kwargs)
            super_dark_ramp, bias_off, masks_dict = res

            # Add residual bias offset
            super_bias = self.super_bias + bias_off
            
            # Save updated superbias frame to directory
            hdu = fits.PrimaryHDU(super_bias)
            hdu.writeto(fname_super_bias, overwrite=True)
            
            # Save super dark ramp
            hdu = fits.PrimaryHDU(super_dark_ramp.astype(np.float32))
            # hdu = fits.PrimaryHDU(super_dark_ramp)
            hdu.writeto(fname_super_dark_ramp, overwrite=True)
            
            # Save mask dictionary to a compressed FITS file
            hdul = fits.HDUList()

            for k in masks_dict.keys():
                data = masks_dict[k].astype('uint8')
                hdu = fits.ImageHDU(data, name=k)
                hdul.append(hdu)

            output = gzip.open(fname_pixel_mask, 'wb')
            hdul.writeto(output, overwrite=True) 
            output.close()

        # Save as class attributes
        self._super_dark_ramp = super_dark_ramp
        self._super_bias = super_bias
        self._pixel_masks = masks_dict

    def get_dark_slope_image(self, deg=1, force=False):
        """ Calculate dark slope image"""

        _log.info('Calculating dark slope image...')
        fname = self.paths_dict['super_dark']

        file_exists = os.path.isfile(fname)
        if file_exists and (not force):
            # Grab Super Dark
            super_dark = get_fits_data(fname)
        else:
            if self._super_dark_ramp is None:
                self.get_super_dark_ramp()
            # Get dark slope image
            cf = jl_poly_fit(self.time_arr, self.super_dark_ramp, deg=deg)
            super_dark = cf[1]

            # Save super dark frame to directory
            hdu = fits.PrimaryHDU(super_dark)
            hdu.writeto(fname, overwrite=True)

        self._super_dark = super_dark

    def get_pixel_slope_averages(self, force=False):
        """Get average pixel ramp"""

        _log.info('Calculating average pixel ramps...')
        fname = self.paths_dict['super_dark_ramp_avgs']

        file_exists = os.path.isfile(fname)
        if file_exists and (not force):
            out = np.load(fname)
            ramp_avg_ch  = out.get('ramp_avg_ch')
            ramp_avg_all = out.get('ramp_avg_all')
        else:
            if self._super_dark_ramp is None:
                _log.error("`super_dark_ramp` is not defined. Please run self.get_super_dark_ramp().")
                return

            nz = self.dark_shape[0]
            nchan = self.nchan
            chsize = self.chsize

            # Average slope in each channel
            ramp_avg_ch = []
            for ch in range(nchan):
                ramp_ch = self.super_dark_ramp[:,:,ch*chsize:(ch+1)*chsize]
                avg = np.median(ramp_ch.reshape([nz,-1]), axis=1)
                ramp_avg_ch.append(avg)
            ramp_avg_ch = np.array(ramp_avg_ch)

            # Average ramp for all pixels
            ramp_avg_all = np.mean(ramp_avg_ch, axis=0)

            np.savez(fname, ramp_avg_ch=ramp_avg_ch, ramp_avg_all=ramp_avg_all)

        self._dark_ramp_dict = {
            'ramp_avg_ch'  : ramp_avg_ch,
            'ramp_avg_all' : ramp_avg_all
        }

    def get_ipc(self, calc_ppc=False):
        """Calculate IPC (and PPC) kernels"""
        
        if calc_ppc:
            _log.info("Calculating IPC and PPC kernels...")
        else:
            _log.info("Calculating IPC kernels...")

        fname_ipc = self.paths_dict['kernel_ipc']
        fname_ppc = self.paths_dict['kernel_ppc']

        gen_vals = False
        if os.path.isfile(fname_ipc):
            k_ipc, k_ipc_sig = get_fits_data(fname_ipc)
        else:
            gen_vals = True

        if calc_ppc:
            if os.path.isfile(fname_ppc):
                k_ppc, k_ppc_sig = get_fits_data(fname_ppc)
            else:
                gen_vals = True
        
        # Do we need to generate IPC/PPC values?
        if gen_vals:
            if self.super_dark_ramp is None:
                _log.error("`super_dark_ramp` is not defined. Please run get_super_dark_ramp().")
                return

            dark_ramp = self.super_dark_ramp[1:] - self.super_dark_ramp[0]

            # Subtract away averaged spatial background from each frame
            dark_med = ndimage.median_filter(self.super_dark, 7)
            tarr = self.time_arr[1:] - self.time_arr[0]
            for i, im in enumerate(dark_ramp):
                im -= dark_med*tarr[i]
                
            nchan = self.nchan
            chsize = self.chsize

            ssd = self.det.same_scan_direction
            rsd = self.det.reverse_scan_direction

            # Set the average of each channel in each image to 0
            for ch in np.arange(nchan):
                x1 = int(ch*chsize)
                x2 = int(x1 + chsize)

                dark_ramp_ch = dark_ramp[:,:,x1:x2]
                dark_ramp_ch = dark_ramp_ch.reshape([dark_ramp.shape[0],-1])
                chmed_arr = np.median(dark_ramp_ch, axis=1)
                dark_ramp[:,:,x1:x2] -= chmed_arr.reshape([-1,1,1])

            k_ipc_arr = []
            k_ppc_arr = []
            for im in dark_ramp[::4]:
                diff = dark_ramp[-1] - im
                res = get_ipc_kernel(diff, bg_remove=False, boxsize=5, calc_ppc=calc_ppc,
                                    same_scan_direction=ssd, reverse_scan_direction=rsd,
                                    suppress_error_msg=True)
                if res is not None:
                    if calc_ppc:
                        k_ipc, k_ppc = res
                        k_ppc_arr.append(k_ppc)
                    else:
                        k_ipc = res
                    k_ipc_arr.append(k_ipc)
                
            # Average IPC values
            k_ipc_arr = np.array(k_ipc_arr)
            k_ipc = robust.mean(k_ipc_arr, axis=0)
            k_ipc_sig = robust.std(k_ipc_arr, axis=0)
                
            # Ensure kernels are normalized to 1
            ipc_norm = k_ipc.sum()
            k_ipc /= ipc_norm
            k_ipc_sig /= ipc_norm

            # Save IPC kernel to file
            hdu = fits.PrimaryHDU(np.array([k_ipc, k_ipc_sig]))
            hdu.writeto(fname_ipc, overwrite=True)

            # PPC values
            if calc_ppc:
                k_ppc_arr = np.array(k_ppc_arr)
                k_ppc = robust.mean(k_ppc_arr, axis=0)
                k_ppc_sig = np.std(k_ppc_arr, axis=0)
                ppc_norm = k_ppc.sum()
                k_ppc /= ppc_norm
                k_ppc_sig /= ppc_norm

                # Save IPC kernel to file
                hdu = fits.PrimaryHDU(np.array([k_ppc, k_ppc_sig]))
                hdu.writeto(fname_ppc, overwrite=True)

        # Store kernel information
        self._kernel_ipc = k_ipc
        self._kernel_ipc_sig = k_ipc_sig
        
        alpha = k_ipc[1,2]
        alpha_sig = k_ipc_sig[1,2]
        _log.info('  IPC = {:.3f}% +/- {:.3f}%'.format(alpha*100, alpha_sig*100))

        # PPC values
        if calc_ppc:
            self._kernel_ppc = k_ppc
            self._kernel_ppc_sig = k_ppc_sig

            ppc = k_ppc[1,2]
            ppc_sig = k_ppc_sig[1,2]
            _log.info('  PPC = {:.3f}% +/- {:.3f}%'.format(ppc*100, ppc_sig*100))

    def get_ktc_noise(self, **kwargs):
        """Calculate and store kTC (Reset) Noise
        
        Keyword Args
        ------------
        bias_sigma_arr : ndarray
            Image of the pixel uncertainties.
        binsize : float
            Size of the histogram bins.
        return_std : bool
            Also return the standard deviation of the 
            distribution?

        """

        if self._super_bias_sig is None:
            # Make sure super bias sigma exists
            _log.info('Obtaining sigma image for super bias...')
            self.get_super_bias_init()

        _log.info("Calculating kTC Noise for active and reference pixels...")

        # kTC Noise (DN)
        im = self._super_bias_sig[self.mask_act]
        self._ktc_noise = calc_ktc(im, **kwargs)
        # kTC Noise of reference pixels
        im = self._super_bias_sig[self.mask_ref]
        self._ktc_noise_ref= calc_ktc(im, binsize=1)

    def get_cds_dict(self, force=False):
        """Calculate CDS noise for all files
        
        Creates a dictionary of CDS noise components, including 
        total noise, amplifier 1/f noise, correlated 1/f noise, 
        white noise, and reference pixel ratios. Two different
        methods are used to calculate CDS per pixels:
        temporal and spatial.

        Creates dictionary attributes `self.cds_act_dict`
        and `self.cds_ref_dict`.
        """

        _log.info("Building CDS Noise dictionaries...")

        ssd = self.det.same_scan_direction

        outname1 = self.paths_dict['cds_act_dict']
        outname2 = self.paths_dict['cds_ref_dict']
        both_exist = os.path.exists(outname1) and os.path.exists(outname2)
        if (not both_exist) or force:
            # Create CDS dictionaries
            cds_act_dict, cds_ref_dict = gen_cds_dict(
                self.allfiles, superbias=self.super_bias,
                mask_good_arr=self._pixel_masks['mask_poly'],
                same_scan_direction=ssd, DMS=self.DMS)

            # Save active pixel dictionary
            self._dict_to_json(cds_act_dict, outname1)
            # Save reference pixel dictionary
            self._dict_to_json(cds_ref_dict, outname2)

        # Load dictionaries
        self._cds_act_dict = self._json_to_dict(outname1)
        self._cds_ref_dict = self._json_to_dict(outname2)

    def get_effective_noise(self, ideal_Poisson=False, force=False):
        "Calculate effective noise curves for each readout pattern"
        
        outname = self.paths_dict['eff_noise_dict']

        allfiles = self.allfiles
        superbias = self.super_bias

        det = self.det

        nchan = det.nout
        gain = det.gain

        patterns = list(det.multiaccum._pattern_settings.keys())

        if os.path.exists(outname) and (not force):
            # Load from JSON files
            with open(outname, 'r') as fp:
                dtemp = json.load(fp)

            # Convert to arrays
            for k in dtemp.keys():
                d2 = dtemp[k]
                out_list = [np.array(d2[patt]) for patt in patterns]
                dtemp[k] = out_list

            ng_all_list  = dtemp['ng_all_list']
            en_spat_list = dtemp['en_spat_list']

        else:
            ng_all_list = []
            en_spat_list = []
            #en_temp_list = []
            for patt in tqdm(patterns, leave=False, desc='Patterns'):
                res = calc_eff_noise(allfiles, DMS=self.DMS, superbias=superbias, read_pattern=patt, temporal=False)
                # ng_all, eff_noise_temp, eff_noise_spa = res
                ng_all, eff_noise_spat = res
                
                # List of ngroups arrays
                ng_all_list.append(ng_all)
                en_spat_list.append(eff_noise_spat)
                #en_temp_list.append(eff_noise_temp)

            # Place variables into dictionary for saving to disk
            dtemp = {'ng_all_list' : ng_all_list, 'en_spat_list' : en_spat_list}

            # Make sure everything are in list format
            for k in dtemp.keys():
                arr = dtemp[k]
                d2 = {}
                for i, patt in enumerate(patterns):
                    d2[patt] = arr[i].tolist()
                dtemp[k] = d2

            # Save to a JSON file
            with open(outname, 'w') as fp:
                json.dump(dtemp, fp, sort_keys=False, indent=4)
            
        # Suppress info logs
        log_prev = conf.logging_level
        setup_logging('WARN', verbose=False)

        tarr_all = []
        for i, patt in enumerate(patterns):
            det_new = deepcopy(det)
            ma_new = det_new.multiaccum
            ma_new.read_mode = patt
            # ma_new.ngroup = int((det.multiaccum.ngroup - ma_new.nd1 + ma_new.nd2) / (ma_new.nf + ma_new.nd2))
            # tvals_all.append(det_new.times_group_avg)
            # Times associated with each calcualted group
            ng_all = ng_all_list[i]
            tarr_all.append((ng_all-1)*det_new.time_group)

        # Determine excess variance parameters
        from scipy.optimize import least_squares#, leastsq

        en_dn_list = []
        for i in range(len(patterns)):
            # Average spatial and temporal values
        #     var_avg_ch = (en_spat_list[i]**2 + en_temp_list[i]**2) / 2
        #     var_avg_ch = en_temp_list[i]**2
            var_avg_ch = en_spat_list[i]**2
            en_dn_list.append(np.sqrt(var_avg_ch[0:nchan].mean(axis=0)))

        # Average dark current (e-/sec)
        if self.dark_ramp_dict is None:
            idark_avg = det.dark_current
        else:
            idark = []
            tarr = self.time_arr
            for ch in np.arange(nchan):
                y = self.dark_ramp_dict['ramp_avg_ch'][ch]
                cf = jl_poly_fit(tarr, y, deg=1)
                idark.append(cf[1])
            idark = np.array(idark) * gain
            idark_avg = np.mean(idark)
            
        # Average read noise per frame (e-)
        cds_var = (en_dn_list[0][0] * det.time_group * gain)**2 - (idark_avg * det.time_group)
        read_noise = np.sqrt(cds_var / 2)

        p0 = [1.5,10]  # Initial guess
        args=(det, patterns, ng_all_list, en_dn_list)
        kwargs = {'idark':idark_avg, 'read_noise':read_noise, 'ideal_Poisson':ideal_Poisson}
        res_lsq = least_squares(fit_func_var_ex, p0, args=args, kwargs=kwargs)
        p_excess = res_lsq.x

        setup_logging(log_prev, verbose=False)
        _log.info("  Best fit excess variance model parameters: {}".format(p_excess))

        self._eff_noise_dict = {
            'patterns'      : patterns,     # Readout patterns
            'ng_all_list'   : ng_all_list,  # List of groups fit
            'tarr_all_list' : tarr_all,     # Associated time values
            'en_spat_list'  : en_spat_list, # Effective noise per channel (spatial)
            'p_excess'      : p_excess      # Excess variance model parameters (best fit)
        }


    def calc_cds_noise(self, cds_type='spatial', temperature=None, temp_key='T_FPA1'):
        """ Return CDS Noise components for each channel
        
        Parameters
        ----------
        cds_type : str
            Return 'spatial', 'temporal', or 'average' noise values?
        temperature : float or None
            Option to supply temperature at which to interpolate. If None is
            provided, then returns the median of all noise values.
        temp_key : str
            Temperature key from `self.temperature_dict` to interpolate over.
            Generally, either 'T_FPA1' or 'T_FPA2' as those most closely
            represent the detector operating temperatures.
        """

        def cds_fit(tval, temps, cds_per_ch):
            """Fit """
            cds_arr = []
            for ch in np.arange(self.nchan):
                cf = jl_poly_fit(temp_arr, cds_per_ch[:,ch])
                cds_arr.append(jl_poly(temperature, cf))
            return np.array(cds_arr).squeeze()

        if (self.cds_act_dict is None) or (self.cds_ref_dict is None):
            _log.error('Dictionaries of CDS noise need generating: See `get_cds_dict()`')
            return

        # Temperature array
        temp_arr = np.array(self.temperature_dict[temp_key])

        if temperature is not None:
            if (temperature<temp_arr.min()) or (temperature>temp_arr.max()):
                tbounds = 'T=[{:.2f}, {:.2f}]K'.format(temp_arr.min(), temp_arr.max())
                _log.warning('Requested temperature is outside of bounds: {}.'.format(tbounds))
                _log.warning('Extrapolation may be inaccurate.')

        # CDS dictionary arrays
        d_act = self.cds_act_dict
        d_ref = self.cds_ref_dict

        if 'spat' in cds_type:
            cds_type_list = ['spat']
        elif 'temp' in cds_type:
            cds_type_list = ['temp']
        else:
            cds_type_list = ['spat', 'temp']

        cds_tot = cds_white = 0
        cds_pink_uncorr = cds_pink_corr = 0
        ref_ratio_all = 0
        for ct in cds_type_list:

            # Total noise per channel
            cds_key = f'{ct}_tot'
            if temperature is None:
                cds_tot += np.median(d_act[cds_key], axis=0)
            else:
                cds_tot += cds_fit(temperature, temp_arr, d_act[cds_key])

            # White noise per channel
            cds_key = f'{ct}_det'
            if temperature is None:
                cds_white += np.median(d_act[cds_key], axis=0)
            else:
                cds_white += cds_fit(temperature, temp_arr, d_act[cds_key])

            # 1/f noise per channel
            cds_key = f'{ct}_pink_uncorr'
            cds_pink_uncorr += np.median(d_act[cds_key], axis=0)
            # Correlated noise
            cds_key = f'{ct}_pink_corr'
            cds_pink_corr += np.median(d_act[cds_key])

            # Reference pixel noise ratio
            cds_key = f'{ct}_det' # or f'{cds_type}_tot'?
            ref_ratio_all += (d_ref[cds_key] / d_act[cds_key])

        ref_ratio = np.mean(ref_ratio_all)

        # Scale by number of modes included
        ntype = len(cds_type_list)
        cds_dict = {
            'tot'   : cds_tot / ntype,
            'white' : cds_white / ntype,
            'pink_uncorr' : cds_pink_uncorr / ntype,
            'pink_corr'   : cds_pink_corr / ntype,
            'ref_ratio'   : ref_ratio / ntype
        }

        return cds_dict

    def get_column_variations(self, force=False, **kwargs):
        """ Get column offset variations
        
        Create a series of column offset models.
        These are likely FETS in the ASIC preamp or ADC 
        causing entire columns within a ramp to jump around.
        """

        _log.info("Determining column variations (RTN)")
        allfiles = self.allfiles

        outname = self.paths_dict['column_variations']
        file_exists = os.path.isfile(outname)

        if file_exists and (not force):
            ramp_column_varations, header = get_fits_data(outname, return_header=True)
            prob_bad = header['PROB_VAR']
        else:
            kwargs_def = {
                'nchans': self.nchan, 'altcol': True, 'in_place': True,    
                'fixcol': True, 'avg_type': 'pixel', 'savgol': True, 'perint': False    
            }
            for k in kwargs_def.keys():
                if k not in kwargs:
                    kwargs[k] = kwargs_def[k]

            # Generate a compilation of column variations
            res = gen_col_variations(allfiles, DMS=self.DMS, super_bias=self.super_bias, 
                                     super_dark_ramp=self.super_dark_ramp, **kwargs)
            ramp_column_varations, prob_bad = res
            
            # Save column ramp variations
            hdu = fits.PrimaryHDU(ramp_column_varations)
            hdu.header['PROB_VAR'] = prob_bad
            hdu.writeto(outname, overwrite=True)

        self._column_variations = ramp_column_varations
        self._column_prob_bad   = prob_bad

    def get_ref_pixel_noise(self, force=False, **kwargs):
        """ Generate Dictionary of Reference Pixel behavior info"""

        _log.info("Determining reference pixel behavior")

        allfiles = self.allfiles

        outname = self.paths_dict['ref_pix_variations']
        file_exists = os.path.isfile(outname)

        if (not file_exists) or force:
            kwargs_def = {
                'nchans': self.nchan, 'altcol': True, 'in_place': True,    
                'fixcol': True, 'avg_type': 'pixel', 'savgol': True, 'perint': False    
            }
            for k in kwargs_def.keys():
                if k not in kwargs:
                    kwargs[k] = kwargs_def[k]

            ref_dict = gen_ref_dict(allfiles, self.super_bias, DMS=self.DMS, **kwargs)
            
            # Save to JSON file
            self._dict_to_json(ref_dict, outname)

        # Load from JSON file
        self._ref_pixel_dict = self._json_to_dict(outname)

    def get_power_spectrum(self, include_oh=False, calc_cds=True, per_pixel=False, 
                           return_corr=False, return_ucorr=False, mn_func=np.mean,
                           force=False, save=True):
        """
        Keyword Args
        ============
        include_oh : bool
            Zero-pad the data to insert line and frame overhead pixels? 
        calc_cds : bool
            Power spectrum of CDS pairs or individual frames?
        return_corr : bool
            Return power spectrum of channel correlated 1/f noise?
        return_ucorr : bool
            Return power spectra of channel-dependent (uncorrelated) 1/f noise?
        per_pixel : bool
            Calculate average power spectrum of each pixel along ramp (frame timescales)?
            If False, samples pixels within a frame (pixel read timescales).
        """

        _log.info("Building noise power spectrum dictionary...")

        # Get file name to save results
        if per_pixel:
            outname = self.paths_dict['power_spec_cds_pix'] if calc_cds else self.paths_dict['power_spec_full_pix']
        else:
            if include_oh:
                outname = self.paths_dict['power_spec_cds_oh'] if calc_cds else self.paths_dict['power_spec_full_oh']
            else:
                outname = self.paths_dict['power_spec_cds'] if calc_cds else self.paths_dict['power_spec_full']
        file_exists = os.path.isfile(outname)

        if file_exists and (not force):
            with open(outname, 'rb') as f:
                ps_all = np.load(f)
                ps_corr = np.load(f)
                ps_ucorr = np.load(f)

        else:
            super_bias = self.super_bias
            if super_bias is None:
                raise AttributeError('Super bias (`self.super_bias = None`) file has not been loaded.')

            ssd = self.det.same_scan_direction
            rsd = self.det.reverse_scan_direction

            res = get_power_spec_all(self.allfiles, super_bias=super_bias, det=self.det,
                                     DMS=self.DMS, include_oh=include_oh, calc_cds=calc_cds,
                                     return_corr=return_corr, return_ucorr=return_ucorr, mn_func=mn_func, 
                                     per_pixel=per_pixel, same_scan_direction=ssd, reverse_scan_direction=rsd)
            ps_all, ps_corr, ps_ucorr = res

            # Set as an arrays of 0s if not calculated for saving purposes
            ps_corr  = np.zeros_like(ps_all[0]).astype('bool') if ps_corr  is None else ps_corr
            ps_ucorr = np.zeros_like(ps_all).astype('bool')    if ps_ucorr is None else ps_ucorr

            # Save arrays to disk            
            if save:
                with open(outname, 'wb') as f:
                    np.save(f, ps_all)
                    np.save(f, ps_corr)
                    np.save(f, ps_ucorr)

        # If corr or ucorr were saved as 0s, set to None
        ps_corr = None if np.allclose(ps_corr, 0) else ps_corr
        ps_ucorr = None if np.allclose(ps_ucorr, 0) else ps_ucorr

        # Get corrsponding frequency arrays
        freq = get_freq_array(ps_all, dt=1/self.det._pixel_rate)
        self._pow_spec_dict = {
            'freq'     : freq,
            'ps_all'   : ps_all,
            'ps_corr'  : ps_corr,
            'ps_ucorr' : ps_ucorr,
        }

        # TODO: Check if something similar for per_pixel
        if not per_pixel:
            # Estimate 1/f scale factors for broken correlated power spectrum
            freq = self.pow_spec_dict['freq']
            ps_all = self.pow_spec_dict['ps_all']

            # Noise values
            cds_dict = self.cds_act_dict
            keys = ['spat_det', 'temp_det', 'spat_pink_uncorr', 'temp_pink_uncorr']
            cds_vals = np.array([np.sqrt(np.mean(cds_dict[k]**2, axis=0)) for k in keys])
            rd_noise = np.sqrt(np.mean(cds_vals[:2]**2))
            u_pink = np.sqrt(np.mean(cds_vals[2:]**2))

            # White Noise
            yf = freq**(0)
            variance = np.mean(rd_noise**2)
            yf1 = len(yf) * variance * yf / yf.sum()

            # Uncorrelated Pink Noise
            yf = freq**(-1); yf[0]=0
            variance = np.mean(u_pink**2) / np.sqrt(2)
            yf2 = len(yf) * variance * yf / yf.sum() 

            # Get residual, to calculate scale factors for correlated noise model
            yresid = ps_all.mean(axis=0) - yf2 - yf1
            scales = fit_corr_powspec(freq, yresid)

            self._pow_spec_dict['ps_corr_scale'] = scales

    def get_super_flats(self, split_low_high=False, smth_sig=10, force=False, **kwargs):
        """Get flat field information
        
        Splits flat field into to lflats and pflats (low and high frequency).
        """

        savename = self.paths_dict['super_flats']
        file_exists = os.path.isfile(savename)


        if file_exists and (not force):
            _log.info("Loading flat field information...")
            # Grab Super Dark Ramp
            super_flats = get_fits_data(savename)
        else:
            # Default ref pixel correction kw args
            kwargs_def = {
                'nchans': self.nchan, 'in_place': True, 'altcol': True,
                'fixcol': True, 'avg_type': 'pixel', 'savgol': True, 'perint': False,
            }
            for k in kwargs_def.keys():
                if k not in kwargs:
                    kwargs[k] = kwargs_def[k]

            # Get nominal non-linear coefficients
            _log.info("Calculating flat field information...")
            allfiles = self.linfiles
            data, _ = gen_super_ramp(allfiles, super_bias=self.super_bias, **kwargs)

            if self.linear_dict is None:
                self.get_linear_coeffs()

            # IPC and PPC kernels
            kppc = self.kernel_ppc
            kipc = self.kernel_ipc
            # PPC corrections
            if (kppc is not None) and kppc[1,2]>0:
                data = ppc_deconvolve(data, kppc)
            # IPC correction
            if kipc is not None:
                data = ipc_deconvolve(data, kipc)

            # Linearity correction
            hdr = fits.getheader(allfiles[0])
            det = create_detops(hdr, DMS=self.DMS)
            data = apply_linearity(data, det, self.linear_dict)

            # Perform fit to data (e-/group)
            tarr = np.arange(1,len(data)+1)
            cf_arr = cube_fit(tarr, data, deg=1, sat_vals=det.well_level, sat_frac=0.8, fit_zero=False)
            im_slope = cf_arr[1]

            del data

            super_flats = get_flat_fields(im_slope, split_low_high=split_low_high, 
                                          smth_sig=smth_sig, ref_info=det.ref_info)
            super_flats = np.asarray(super_flats)

            # Save super flats to directory
            hdu = fits.PrimaryHDU(super_flats)
            if split_low_high:
                hdu.header['SMTH_SIG'] = smth_sig
            hdu.writeto(savename, overwrite=True)

        sh = super_flats.shape
        if len(sh)==3:
            nz, ny, nx = sh
            if nz==2:
                lflats, pflats = super_flats
            else:
                pflats = super_flats
                lflats = np.ones_like(pflats)
        else:
            pflats = super_flats
            lflats = np.ones_like(pflats)

        self.lflats = lflats
        self.pflats = pflats

    def _get_linear_coeffs(self, deg=8, use_legendre=True, lxmap=[0,1e5], counts_cut=None, sat_calc=0.98,
                           nonlin=False, force=False, DMS=None, super_bias=None, **kwargs):
        """ Determine non-linear coefficents

        These coefficients allow us to go from an ideal linear ramp to 
        some observed (simulated) non-linear ramp.

        Parameters
        ==========
        force : bool
            Force calculation of coefficients.
        DMS : None or bool
            Option to specifiy if linearity files are DMS format.
            If set to None, then uses self.DMS.
        super_bias: None or ndarray
            Option to specify an input super bias image. If not specified,
            then defaults to self.super_bias.
        counts_cut : None or float
            Option to fit two sets of polynomial coefficients to lower and uppper
            values. 'counts_cut' specifies the division in values of electrons.
            Useful for pixels with different non-linear behavior at low flux levels.
            Recommended values of 15000 e-.
        deg : int
            Degree of polynomial to fit. Default=8.
        use_legendre : bool
            Fit with Legendre polynomial, an orthonormal basis set.
            Default=True.
        lxmap : ndarray or None
            Legendre polynomials are normaly mapped to xvals of [-1,+1].
            `lxmap` gives the option to supply the values for xval that
            should get mapped to [-1,+1]. If set to None, then assumes 
            [xvals.min(),xvals.max()]. 
        """

        if nonlin:
            savename = self.paths_dict['nonlinear_coeffs']
        else:
            savename = self.paths_dict['linear_coeffs']
            
        file_exists = os.path.isfile(savename)
        if file_exists and (not force):
            if nonlin:
                _log.info("Loading non-linearity coefficents")
            else:
                _log.info("Loading linearity coefficents")

            out = np.load(savename)
            cf_nonlin      = out.get('cf_nonlin')
            counts_cut     = out.get('counts_cut').tolist()
            if counts_cut == 0:
                counts_cut = None
                cf_nonlin_low = None
            else:
                cf_nonlin_low  = out.get('cf_nonlin_low')
            use_legendre   = out.get('use_legendre').tolist()
            lxmap          = out.get('lxmap').tolist()
            deg            = out.get('deg').tolist()
            if nonlin:
                cflin0_mean    = out.get('cflin0_mean')
                cflin0_std     = out.get('cflin0_std')
                corr_slope     = out.get('corr_slope')
                corr_intercept = out.get('corr_intercept')
                sat_vals       = out.get('sat_vals')
        else:
            if nonlin:
                _log.info("Generating non-linearity coefficents")
            else:
                _log.info("Generating linearity coefficents")
            allfiles = self.linfiles

            # Check if super bias exists
            if (self._super_bias is None) and (super_bias is None):
                _log.warning('Super bias not loaded or specified. Proceeding without bias correction.')
            elif super_bias is None:
                super_bias = self.super_bias

            if DMS is None:
                DMS = self.DMS

            # Set logging to WARNING to suppress messages
            log_prev = conf.logging_level
            setup_logging('WARN', verbose=False)

            f = allfiles[-1]
            hdr = fits.getheader(f)
            det = create_detops(hdr, DMS=DMS)

            setup_logging(log_prev, verbose=False)

            grp_max = find_group_sat(f, DMS=DMS, bias=super_bias, sat_calc=0.998)
            grp_max = grp_max + 10

            if grp_max > det.multiaccum.ngroup:
                grp_max = det.multiaccum.ngroup

            # Default ref pixel correction kw args
            kwargs_def = {
                'nchans': self.nchan, 'in_place': True, 'altcol': True,
                'fixcol': True, 'avg_type': 'pixel', 'savgol': True, 'perint': False,
            }
            for k in kwargs_def.keys():
                if k not in kwargs:
                    kwargs[k] = kwargs_def[k]

            # Get nominal non-linear coefficients
            _log.info("  Calculating average coefficients...")
            kppc = self.kernel_ppc
            kipc = self.kernel_ipc
            res, sat_vals = get_linear_coeffs(allfiles, super_bias=super_bias, DMS=DMS, grp_max=grp_max, deg=deg, 
                                              use_legendre=use_legendre, lxmap=lxmap, counts_cut=counts_cut, 
                                              return_satvals=True, kppc=kppc, kipc=kipc, nonlin=nonlin, sat_calc=sat_calc, 
                                              **kwargs)
            # Two separate fits for low and high pixel values
            if counts_cut is None:
                cf_nonlin = res
                cf_nonlin_low = None
            else:
                cf_nonlin, cf_nonlin_low = res

            # Obtain coefficient variations
            if nonlin:
                _log.info("  Calculating coefficient variations...")

                # Solve for coefficients for all data sets
                # Probes random variations
                cf_all = []
                for file in tqdm(allfiles, desc='Variance', leave=False):
                    res = get_linear_coeffs([file], super_bias=super_bias, DMS=DMS, counts_cut=counts_cut, 
                                            deg=deg, use_legendre=use_legendre, lxmap=lxmap, grp_max=grp_max, 
                                            sat_vals=sat_vals, nonlin=True, sat_calc=sat_calc, **kwargs)
                    if counts_cut is None:
                        cf = res
                    else: # Ignore variations to lower fits
                        cf, _ = res
                    cf_all.append(cf)
                cf_all = np.array(cf_all)

                # Coefficients are related to each
                # Save the linear correlation for each pixel
                cf_all_min = np.min(cf_all, axis=0)
                cf_all_max = np.max(cf_all, axis=0)
                cf_all_mean = np.mean(cf_all, axis=0)

                corr_slope1 = (cf_all_max[1:] - cf_all_mean[1:]) / (cf_all_max[0] - cf_all_mean[0])
                corr_slope2 = (cf_all_mean[1:] - cf_all_min[1:]) / (cf_all_mean[0] - cf_all_min[0])
                corr_slope = 0.5 * (corr_slope1 + corr_slope2)
                corr_intercept = cf_all_mean[1:] - corr_slope*cf_all_mean[0]
                corr_slope[:, self.mask_ref] = 0
                corr_intercept[:, self.mask_ref] = 0

                cflin0_mean = cf_nonlin[0]
                cflin0_std  = np.std(cf_all[:,0,:,:], axis=0)

                if counts_cut is None:
                    counts_cut = 0
                    cf_nonlin_low = 0
                np.savez(savename, cf_nonlin=cf_nonlin, cflin0_mean=cflin0_mean, cflin0_std=cflin0_std, 
                        corr_slope=corr_slope, corr_intercept=corr_intercept, deg=deg, 
                        use_legendre=use_legendre, lxmap=lxmap, sat_vals=sat_vals,
                        counts_cut=counts_cut, cf_nonlin_low=cf_nonlin_low)
            else:
                if counts_cut is None:
                    counts_cut = 0
                    cf_nonlin_low = 0
                np.savez(savename, cf_nonlin=cf_nonlin, cf_nonlin_low=cf_nonlin_low, counts_cut=counts_cut,
                        deg=deg, use_legendre=use_legendre, lxmap=lxmap, sat_vals=sat_vals)

        # Additional check on fitting of lower values 
        if (counts_cut==0) or (counts_cut is None):
            counts_cut = None
            cf_nonlin_low = None

        # Store everything in dictionary
        if nonlin:
            self.nonlinear_dict = {
                'cf_nonlin'      : cf_nonlin,
                'cflin0_mean'    : cflin0_mean,
                'cflin0_std'     : cflin0_std,
                'corr_slope'     : corr_slope,
                'corr_intercept' : corr_intercept,
                'use_legendre'   : use_legendre,
                'lxmap'          : lxmap,
                'deg'            : deg,
                'counts_cut'     : counts_cut,
                'cf_nonlin_low'  : cf_nonlin_low,
                'sat_vals'       : sat_vals,
            }
        else:
            self.linear_dict = {
                'cf_nonlin'      : cf_nonlin,
                'cf_nonlin_low'  : cf_nonlin_low,
                'counts_cut'     : counts_cut,
                'use_legendre'   : use_legendre,
                'lxmap'          : lxmap,
                'deg'            : deg,
            }


    def get_nonlinear_coeffs(self, deg=8, use_legendre=True, lxmap=[0,1e5], counts_cut=15000, 
                             sat_calc=0.998, force=False, DMS=None, super_bias=None, **kwargs):
        """ Determine non-linear coefficents

        These coefficients allow us to go from an ideal linear ramp to 
        some observed (simulated) non-linear ramp. Value are store in the 
        self.nonlinear_dict dictionary.

        Parameters
        ==========
        force : bool
            Force calculation of coefficients.
        DMS : None or bool
            Option to specifiy if linearity files are DMS format.
            If set to None, then uses self.DMS.
        super_bias: None or ndarray
            Option to specify an input super bias image. If not specified,
            then defaults to self.super_bias.
        counts_cut : None or float
            Option to fit two sets of polynomial coefficients to lower and uppper
            values. 'counts_cut' specifies the division in values of electrons.
            Useful for pixels with different non-linear behavior at low flux levels.
            Recommended values of 15000 e-.
        deg : int
            Degree of polynomial to fit. Default=8.
        use_legendre : bool
            Fit with Legendre polynomial, an orthonormal basis set.
            Default=True.
        lxmap : ndarray or None
            Legendre polynomials are normaly mapped to xvals of [-1,+1].
            `lxmap` gives the option to supply the values for xval that
            should get mapped to [-1,+1]. If set to None, then assumes 
            [xvals.min(),xvals.max()]. 
        """
        self._get_linear_coeffs(deg=deg, use_legendre=use_legendre, lxmap=lxmap, 
                                counts_cut=counts_cut, sat_calc=sat_calc, nonlin=True, 
                                force=force, DMS=DMS, super_bias=super_bias, **kwargs)

    def get_linear_coeffs(self, deg=8, use_legendre=True, lxmap=[0,1e5], counts_cut=None, 
                          sat_calc=0.98, force=False, DMS=None, super_bias=None, **kwargs):
        """ Determine linearity coefficents

        These coefficients allow us to convert from an observed ramp (DN) to
        an idealized linear ramp (in e-). Values are stored in the dictionary
        self.linear_dict.

        Parameters
        ==========
        force : bool
            Force calculation of coefficients.
        DMS : None or bool
            Option to specifiy if linearity files are DMS format.
            If set to None, then uses self.DMS.
        super_bias: None or ndarray
            Option to specify an input super bias image. If not specified,
            then defaults to self.super_bias.
        counts_cut : None or float
            Option to fit two sets of polynomial coefficients to lower and uppper
            values. 'counts_cut' specifies the division in values of electrons.
            Useful for pixels with different non-linear behavior at low flux levels.
            Recommended values of 15000 e-. 
        deg : int
            Degree of polynomial to fit. Default=8.
        use_legendre : bool
            Fit with Legendre polynomial, an orthonormal basis set.
            Default=True.
        lxmap : ndarray or None
            Legendre polynomials are normaly mapped to xvals of [-1,+1].
            `lxmap` gives the option to supply the values for xval that
            should get mapped to [-1,+1]. If set to None, then assumes 
            [xvals.min(),xvals.max()]. 
        """
        if counts_cut is not None:
            counts_cut = counts_cut / self.det.gain
        self._get_linear_coeffs(deg=deg, use_legendre=use_legendre, lxmap=lxmap, 
                                counts_cut=counts_cut, sat_calc=sat_calc, nonlin=False, 
                                force=force, DMS=DMS, super_bias=super_bias, **kwargs)


    def deconvolve_supers(self):
        """
        Deconvolve the super dark and super bias images
        """

        k_ppc = self.kernel_ppc
        k_ipc = self.kernel_ipc
        if (k_ppc is None) and (k_ipc is None):
            _log.error("Neither IPC or PPC kernels are defined")
            return

        _log.info("Deconvolving super dark and super bias images...")

        # PPC Deconvolution
        if k_ppc is not None:
            ssd = self.det.same_scan_direction
            rsd = self.det.reverse_scan_direction
            super_dark_deconv = ppc_deconvolve(self.super_dark, k_ppc,
                                               same_scan_direction=ssd, 
                                               reverse_scan_direction=rsd)
            super_bias_deconv = ppc_deconvolve(self.super_bias, k_ppc,
                                               same_scan_direction=ssd, 
                                               reverse_scan_direction=rsd)

        # IPC Deconvolution
        if k_ipc is not None:
            super_dark_deconv = ipc_deconvolve(super_dark_deconv, k_ipc)
            super_bias_deconv = ipc_deconvolve(super_bias_deconv, k_ipc)

        self._super_dark_deconv = super_dark_deconv
        self._super_bias_deconv = super_bias_deconv

    def plot_bias_darks(self, save=False, return_figax=False, deconvolve=False):
        """
        """
        if self.super_bias is None:
            _log.error("Super bias image has not yet been generated.")
            return
        if self.super_dark is None:
            _log.error("Super dark image has not yet been generated.")
            return

        scaid = self.scaid

        if deconvolve:
            super_bias, super_dark = self.super_bias_deconv, self.super_dark_deconv
        else:
            super_bias, super_dark = self.super_bias, self.super_dark


        fig, axes = plt.subplots(1,2,figsize=(14,8.5), sharey=True)
        cbar_labels = ['Relative Offset (DN)', 'Dark Current (DN/sec)']
        for i, im in enumerate([super_bias, super_dark]):

            mn = np.median(im)
            std = robust.medabsdev(im)

            vmin = mn - 3*std
            vmax = mn + 3*std
            ax = axes[i]
            image = ax.imshow(im, vmin=vmin, vmax=vmax)

            # Add colorbar
            cbar = fig.colorbar(image, ax=ax, orientation='horizontal', 
                                pad=0.05, fraction=0.1, aspect=30, shrink=1)
            cbar.set_label(cbar_labels[i])


        # Add titles and labels
        titles = ['Super Bias Image', 'Super Dark Current Image']
        for i, ax in enumerate(axes):
            ax.set_title(titles[i])

        fig.suptitle(f'SCA {scaid}', fontsize=16)

        fig.tight_layout()
        fig.subplots_adjust(top=0.92, wspace=0.02, bottom=0.01)

        if save:
            fname = f'{scaid}_bias_dark_images.pdf'
            save_name = os.path.join(self.paths_dict['figdir'], fname)
            _log.info(f"Saving to {save_name}")
            fig.savefig(save_name)

        if return_figax:
            return fig, axes


    def plot_dark_ramps(self, save=True, time_cut=None, return_figax=False):
        """ Plot average dark current ramps
        
        time_cut : float
            Some darks show distinct slopes before and after a 
            characteristic time. Setting this keyword will fit
            separate slopes before and after the specified time.
            A time of 200 sec is used for SCA 485.
        """

        # Make sure dictionary is not empty
        if self._dark_ramp_dict is None:
            self.get_pixel_slope_averages()
        scaid = self.scaid

        fig, axes = plt.subplots(1,2, figsize=(14,5))
        axes = axes.flatten()

        # Plot average of all pixel
        ax = axes[0]
        ax.set_title('Average Ramp of All Pixels')
        tarr = self.time_arr
        y = self.dark_ramp_dict['ramp_avg_all']
        ax.plot(tarr, y, marker='.', label='Median Pixel Values')

        if time_cut is None:
            cf = jl_poly_fit(tarr, y, deg=1)
            ax.plot(tarr, jl_poly(tarr,cf), label='Slope Fit = {:.4f} DN/sec'.format(cf[1]))
        else:
            for ind in [tarr<time_cut, tarr>time_cut, tarr>0]:
                cf = jl_poly_fit(tarr[ind], y[ind], deg=1)
                ax.plot(tarr, jl_poly(tarr,cf), label='Slope Fit = {:.4f} DN/sec'.format(cf[1]))

        # Plot each channel separately
        ax = axes[1]
        ax.set_title('Channel Ramps')
        for i in range(self.nchan):
            y = self.dark_ramp_dict['ramp_avg_ch'][i]
            cf = jl_poly_fit(tarr, y, deg=1)
            label = 'Ch{} = {:.4f} DN/sec'.format(i, cf[1])
            ax.plot(tarr, y, marker='.', label=label)
            
        ylim1 = ylim2 = 0
        for ax in axes:
            ax.set_xlabel('Time (sec)')
            ax.set_ylabel('Signal (DN)')
            ax_yl = ax.get_ylim()
            ylim1 = np.min([ylim1, ax_yl[0]])
            ylim2 = np.max([ylim2, ax_yl[1]])
            ax.legend()

        for ax in axes:
            ax.set_ylim([ylim1,ylim2])
            # Plot baseline at y=0
            xlim = ax.get_xlim()
            ax.plot(xlim, [0,0], color='k', ls='--', lw=1, alpha=0.25)
            ax.set_xlim(xlim)

        fig.suptitle(f'Dark Current (SCA {scaid})', fontsize=16)
            
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        if save:
            fname = f'{scaid}_dark_ramp_avg.pdf'
            save_name = os.path.join(self.paths_dict['figdir'], fname)
            _log.info(f"Saving to {save_name}")
            fig.savefig(save_name)

        if return_figax:
            return fig, axes


    def plot_dark_ramps_ch(self, save=True, time_cut=None, return_figax=False):
        """ Plot fits to each channel dark current ramp
        
        time_cut : float
            Some darks show distinct slopes before and after a 
            characteristic time. Setting this keyword will fit
            separate slopes before and after the specified time.
            A time of 200 sec is used for SCA 485.
        """

        # Make sure dictionary is not empty
        if self._dark_ramp_dict is None:
            self.get_pixel_slope_averages()
        scaid = self.scaid

        fig, axes = plt.subplots(2,2, figsize=(14,9))
        axes = axes.flatten()
            
        # Plot Individual Channels
        tarr = self.time_arr
        for i in range(self.nchan):
            ax = axes[i]
            y = self.dark_ramp_dict['ramp_avg_ch'][i]
            ax.plot(tarr, y, marker='.', label='Pixel Averages')

            if time_cut is None:
                cf = jl_poly_fit(tarr, y, deg=1)
                ax.plot(tarr, jl_poly(tarr,cf), label='Slope = {:.4f} DN/sec'.format(cf[1]))
            else:
                for ind in [tarr<time_cut, tarr>time_cut, tarr>0]:
                    cf = jl_poly_fit(tarr[ind], y[ind], deg=1)
                    ax.plot(tarr, jl_poly(tarr,cf), label='Slope = {:.4f} DN/sec'.format(cf[1]))

            ax.set_title(f'Amplifier Channel {i}')

        ylim1 = ylim2 = 0
        for ax in axes:
            ax.set_xlabel('Time (sec)')
            ax.set_ylabel('Dark Value (DN)')
            ax_yl = ax.get_ylim()
            ylim1 = np.min([ylim1, ax_yl[0]])
            ylim2 = np.max([ylim2, ax_yl[1]])
            ax.legend()

        for ax in axes:
            ax.set_ylim([ylim1,ylim2])
            # Plot baseline at y=0
            xlim = ax.get_xlim()
            ax.plot(xlim, [0,0], color='k', ls='--', lw=1, alpha=0.25)
            ax.set_xlim(xlim)
            
        fig.suptitle(f'Dark Current (SCA {scaid})', fontsize=16)
            
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        if save:
            fname = f'{scaid}_dark_ramp_chans.pdf'
            save_name = os.path.join(self.paths_dict['figdir'], fname)
            _log.info(f"Saving to {save_name}")
            fig.savefig(save_name)

        if return_figax:
            return fig, axes


    def plot_dark_distribution(self, save=False, xlim=None, return_figax=False):
        """Plot histogram of dark slope"""

        act_mask = self.mask_act
        ch_mask = self.mask_channels
        nchan = self.nchan
        scaid = self.scaid

        # Histogram of Dark Slope
        if self.super_dark is None:
            _log.error("Super dark image has not yet been generated.")
            return

        fig, axes = plt.subplots(1,2, figsize=(14,5), sharey=True)

        # Full image
        ax = axes[0]
        im = self.super_dark[act_mask]
        plot_dark_histogram(im, ax)

        # Individual Amplifiers
        ax = axes[1]
        carr = ['C0', 'C1', 'C2', 'C3']
        for ch in np.arange(nchan):
            ind = (ch_mask==ch) & act_mask
            im = self.super_dark[ind]
            label = f'Ch{ch}'
            plot_dark_histogram(im, ax, label=label, color=carr[ch], 
                                 plot_fit=False, plot_cumsum=False)
        ax.set_ylabel('')
        ax.set_title('Active Pixels per Amplifier')

        # Plot baseline at y=0
        for ax in axes:
            if xlim is None:
                xlim = ax.get_xlim()
            ax.plot(xlim, [0,0], color='k', ls='--', lw=1, alpha=0.25)
            ax.set_xlim(xlim)

        fig.suptitle(f'Dark Current Distriutions (SCA {self.scaid})', fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85, wspace=0.025)

        if save:
            fname = f'{self.scaid}_dark_histogram.pdf'
            save_name = os.path.join(self.paths_dict['figdir'], fname)
            _log.info(f"Saving to {save_name}")
            fig.savefig(save_name)

        if return_figax:
            return fig, axes


    def plot_dark_overview(self, save=False, xlim_hist=None, return_figax=False):
        """Plot Overview of Dark Current Characteristics"""
        
        if self.super_dark is None:
            _log.error("Super dark image has not yet been generated.")
            return
        
        scaid = self.scaid
        fig, axes = plt.subplots(1,3,figsize=(14,5))

        #########################################
        # Dark Current slope image
        ax = axes[0]

        im = self.super_dark
        mn = np.median(im)
        std = robust.medabsdev(im)

        vmin = mn - 3*std
        vmax = mn + 3*std
        image = ax.imshow(im, vmin=vmin, vmax=vmax)

        # Add colorbar
        cbar = fig.colorbar(image, ax=ax, orientation='horizontal',
                            pad=0.08, fraction=0.05, aspect=30, shrink=0.9)
        ax.set_title('Dark Current Image')
        cbar.set_label('Dark Current (DN/sec)')

        #########################################
        # Average pixel slope over time
        ax = axes[1]

        ax.set_title('Average Ramp of All Pixels')
        tarr = self.time_arr
        y = self.dark_ramp_dict['ramp_avg_all']
        ax.plot(tarr, y, marker='.', label='Median Pixel Values')

        cf = jl_poly_fit(tarr, y, deg=1)
        ax.plot(tarr, jl_poly(tarr,cf), label='Slope Fit = {:.4f} DN/sec'.format(cf[1]))

        # Plot baseline at y=0
        xlim = ax.get_xlim()
        ax.plot(xlim, [0,0], color='k', ls='--', lw=1, alpha=0.25)
        ax.set_xlim(xlim)

        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Signal (DN)')
        ax.legend()

        #########################################
        # Dark current histogram
        ax = axes[2]

        act_mask = self.mask_act
        scaid = self.scaid

        # Histogram of Dark Slope
        im = self.super_dark[act_mask]

        ax = plot_dark_histogram(im, ax, return_ax=True, plot_fit=False)
        ax.set_title('Slope Distribution')

        # Plot baseline at y=0
        if xlim_hist is None:
            xlim_hist = ax.get_xlim()
        ax.plot(xlim_hist, [0,0], color='k', ls='--', lw=1, alpha=0.25)
        ax.set_xlim(xlim_hist)

        fig.suptitle(f'Dark Current Overview (SCA {scaid})', fontsize=16)

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.1, top=0.85)

        if save:
            fname = f'{self.scaid}_dark_overview.pdf'
            save_name = os.path.join(self.paths_dict['figdir'], fname)
            _log.info(f"Saving to {save_name}")
            fig.savefig(save_name)

        if return_figax:
            return fig, axes


    def plot_ipc_ppc(self, k_ipc=None, k_ppc=None, save=False, return_figax=False):

        k_ipc = self.kernel_ipc if k_ipc is None else k_ipc
        k_ppc = self.kernel_ppc if k_ppc is None else k_ppc
        scaid = self.scaid
        
        if k_ipc is None:
            _log.info("IPC Kernel does not exist.")
            return

        if k_ipc is None:
            # Plot only IPC kernel
            fig, axes = plt.subplots(1,1, figsize=(5,5))
            plot_kernel(k_ipc, ax=axes)
            axes.set_title('IPC Kernel', fontsize=16)
            fig.tight_layout()
        else:
            # Plot both IPC and PPC
            fig, axes = plt.subplots(1,2, figsize=(10,5.5), sharey=True)

            ax = axes[0]
            plot_kernel(k_ipc, ax=ax)
            ax.set_title('IPC Kernel')

            ax = axes[1]
            plot_kernel(k_ppc, ax=ax)
            ax.set_title('PPC Kernel')

            fig.suptitle(f"Pixel Deconvolution Kernels (SCA {scaid})", fontsize=16)

            fig.tight_layout()
            fig.subplots_adjust(wspace=0.075, top=0.9)
        
        if save:
            fname = f'{self.scaid}_pixel_kernels.pdf'
            save_name = os.path.join(self.paths_dict['figdir'], fname)
            _log.info(f"Saving to {save_name}")
            fig.savefig(save_name)

        if return_figax:
            return fig, axes

    def plot_reset_overview(self, save=False, binsize=0.25, xlim_hist=None,
                            return_figax=False):
        """ Overview Plots of Bias and kTC Noise"""

        if self._super_bias is None:
            _log.error('Super bias image does not exist.')
            return

        if self._super_bias_sig is None:
            _log.error('Sigma image for super bias does not exist.')
            return

        scaid = self.scaid

        # Histogram of Bias kTC
        im = self._super_bias_sig[self.mask_act]
        binsize = binsize
        bins = np.arange(im.min(), im.max() + binsize, binsize)
        ig, vg, cv = hist_indices(im, bins=bins, return_more=True)

        nvals = np.array([len(i) for i in ig])
        nvals_rel = nvals / nvals.max()

        # Peak of distribution
        if self.ktc_noise is None:
            self.get_ktc_noise(binsize=binsize)
        peak = self.ktc_noise

        fig, axes = plt.subplots(1,3,figsize=(14,5))

        #####################################
        # Plot super bias image
        ax = axes[0]

        im = self._super_bias
        mn = np.median(im)
        std = robust.std(im)

        ax.imshow(im, vmin=mn-3*std, vmax=mn+3*std)
        ax.set_title('Super Bias Image')

        #####################################
        # Plot kTC noise image
        ax = axes[1]

        im = self._super_bias_sig
        mn = np.median(im)
        std = robust.std(im)

        ax.imshow(im, vmin=mn-3*std, vmax=mn+3*std)
        ax.set_title('kTC Noise = {:.1f} DN'.format(peak))

        #####################################
        # Plot kTC noise histogram
        ax = axes[2]
        ax.plot(cv, nvals_rel, label='Measured Noise')

        label = 'Peak ({:.1f} DN)'.format(peak)
        ax.plot(np.array([1,1])*peak, [0,1], ls='--', lw=1, label=label)
        ncum = np.cumsum(nvals) 
        ax.plot(cv, ncum / ncum.max(), color='C3', lw=1, label='Cumulative Sum')

        ax.set_title('kTC Noise Distribution')
        ax.set_xlabel('Bias Noise (DN)')
        ax.set_ylabel('Relative Number of Pixels')
        ax.legend()

        ax.set_xlim([0,3*peak])
        #ax.xaxis.get_major_locator().set_params(nbins=9, steps=[1, 2, 5, 10])

        # Plot baseline at y=0
        if xlim_hist is None:
            xlim_hist = ax.get_xlim()
        ax.plot(xlim_hist, [0,0], color='k', ls='--', lw=1, alpha=0.25)
        ax.set_xlim(xlim_hist)


        fig.suptitle(f'Reset Bias Overview (SCA {scaid})', fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.1, top=0.85)

        if save:
            fname = f'{self.scaid}_bias_overview.pdf'
            save_name = os.path.join(self.paths_dict['figdir'], fname)
            _log.info(f"Saving to {save_name}")
            fig.savefig(save_name)

        if return_figax:
            return fig, axes

    def plot_cds_noise(self, tkey='T_FPA1', save=False, return_figax=False,
        xlim=[36.1,40.1]):

        fig, axes = plt.subplots(2,3, figsize=(14,8), sharey=True)

        temp_arr = np.array(self.temperature_dict[tkey])

        d = self.cds_act_dict
        d2 = self.cds_ref_dict

        # 1. Total Noise
        k1, k2 = ('spat_tot', 'temp_tot')
        for k, ax in zip([k1,k2], axes[:,0]):
            
            cds_arr = d[k]
            for ch in np.arange(self.nchan):
                ax.plot(temp_arr, cds_arr[:,ch], marker='o', ls='none', label=f'Ch{ch}')
                    
            type_str = "Spatial" if 'spat' in k else "Temporal"
            title_str = f"{type_str} Total Noise"
            ax.set_title(title_str)
        
        # 2. White Noise
        k1, k2 = ('spat_det', 'temp_det')
        cmap = plt.get_cmap('tab20')
        tplot = np.array([temp_arr.min(), temp_arr.max()])
        for k, ax in zip([k1,k2], axes[:,1]):
            
            cds_arr = d[k]
            pix_type = ['Active', 'Ref']
            for j, cds_arr in enumerate([d[k], d2[k]]):
                marker = 'o' if j==0 else '.'
                for ch in np.arange(self.nchan):
                    label = f'Ch{ch} ({pix_type[j]})'
                    y = cds_arr[:,ch]
                    ax.plot(temp_arr, y, marker=marker, ls='none', label=label, color=cmap(ch*2+j))
                    cf = jl_poly_fit(temp_arr, y)
                    ax.plot(tplot, jl_poly(tplot, cf), lw=1, ls='--', color=cmap(ch*2+j))
                        
            type_str = "Spatial" if 'spat' in k else "Temporal"
            title_str = f"{type_str} White Noise"
            ax.set_title(title_str)
    
        # 3. Pink Noise
        k1, k2 = ('spat_pink_uncorr', 'temp_pink_uncorr')
        for k, ax in zip([k1,k2], axes[:,2]):
            
            cds_arr = d[k]
            for ch in np.arange(self.nchan):
                ax.plot(temp_arr, cds_arr[:,ch], marker='o', ls='none', label=f'Ch{ch}')
            
            k_corr = k.split('_')
            k_corr[-1] = 'corr'
            k_corr = '_'.join(k_corr)
            
            ax.plot(temp_arr, d[k_corr], marker='o', ls='none', label='Correlated')

            type_str = "Spatial" if 'spat' in k else "Temporal"
            title_str = f"{type_str} 1/f Noise"
            ax.set_title(title_str)

        for ax in axes:
            ax[0].set_ylabel('CDS Noise (DN)')
        for ax in axes[-1,:]:
            ax.set_xlabel('FPA Temperature (K)')
        for ax in axes.flatten():
            ax.set_xlim(xlim)
            handles, labels = ax.get_legend_handles_labels()
            ncol = 2 if len(handles)>5 else 1
            ax.legend(ncol=ncol)
        
        fig.suptitle(f'CDS Noise Overview (SCA {self.scaid})', fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.01, top=0.9)

        if save:
            fname = f'{self.scaid}_cds_noise.pdf'
            save_name = os.path.join(self.paths_dict['figdir'], fname)
            _log.info(f"Saving to {save_name}")
            fig.savefig(save_name)

        if return_figax:
            return fig, axes

    def plot_eff_noise(self, ideal_Poisson=False, save=False, return_figax=False):
        """Plot effective noise of slope fits"""

        det = self.det
        gain = det.gain
        nchan = det.nout

        # Average dark current (e-/sec)
        if self.dark_ramp_dict is None:
            idark = np.ones(nchan) * det.dark_current   # e-/sec
        else:
            idark = []
            tarr = self.time_arr
            for ch in np.arange(nchan):
                y = self.dark_ramp_dict['ramp_avg_ch'][ch]
                cf = jl_poly_fit(tarr, y, deg=1)
                idark.append(cf[1])
            idark = np.array(idark) * gain   # e-/sec

        eff_noise_dnsec = self.eff_noise_dict['en_spat_list'][0]
        # Average read noise per frame (e-)
        cds_var = (eff_noise_dnsec[0:nchan,0] * det.time_group * gain)**2 - (idark * det.time_group)
        read_noise = np.sqrt(cds_var / 2) # e-
        read_noise_ref = eff_noise_dnsec[-1,0] * det.time_group * gain / np.sqrt(2)

        ng_all = self.eff_noise_dict['ng_all_list'][0]
        tvals = self.eff_noise_dict['tarr_all_list'][0]
        p_excess = self.eff_noise_dict['p_excess']

        colarr = ['C0', 'C1', 'C2', 'C3', 'C4']
        fig, axes = plt.subplots(1,2, figsize=(14,4.5))

        ax = axes[0]

        # Measured Values
        xvals = tvals
        yvals = eff_noise_dnsec
        for ch in range(nchan):
            axes[0].plot(xvals, yvals[ch]*tvals, marker='o', label=f'Ch{ch} - Meas', color=colarr[ch])
            axes[1].semilogy(xvals, yvals[ch], marker='o', label=f'Ch{ch} - Meas', color=colarr[ch])
        ch = -1
        axes[0].plot(xvals, yvals[ch]*tvals, marker='o', label='Ref - Meas', color=colarr[ch])
        axes[1].plot(xvals, yvals[ch], marker='o', label='Ref - Meas', color=colarr[ch])

        # Theoretical Values
        xvals = tvals
        for ch in range(nchan):
            thr_e = det.pixel_noise(ng=ng_all, rn=read_noise[ch], idark=idark[ch], 
                                    ideal_Poisson=ideal_Poisson, p_excess=p_excess, scale_ints=False)
            yvals2 = (thr_e * tvals) / gain
            axes[0].plot(xvals, yvals2,  color=colarr[ch], lw=10, alpha=0.3, label=f'Ch{ch} - Theory')
            axes[1].plot(xvals, yvals2/tvals,  color=colarr[ch], lw=10, alpha=0.3, label=f'Ch{ch} - Theory')
        ch = -1
        thr_e = det.pixel_noise(ng=ng_all, rn=read_noise_ref, idark=0, p_excess=[0,0], scale_ints=False)
        yvals2 = (thr_e * tvals) / gain
        axes[0].plot(xvals, yvals2,  color=colarr[ch], lw=10, alpha=0.3, label=f'Ref - Theory')
        axes[1].plot(xvals, yvals2/tvals,  color=colarr[ch], lw=10, alpha=0.3, label=f'Ref - Theory')

        ax = axes[0]
        ax.set_ylim([0,ax.get_ylim()[1]])
        axes[0].set_ylabel('Effective Noise (DN)')
        axes[1].set_ylabel('Slope Noise (DN/sec)')
        for ax in axes:
            ax.set_xlabel('Time (sec)')
        #ax.set_title(f'Effective Noise (SCA {self.scaid})')

        axes[0].legend(ncol=2)

        fig.suptitle(f"Noise of Slope Fits (SCA {self.scaid})", fontsize=16)

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        if save:
            fname = f'{self.scaid}_eff_noise.pdf'
            save_name = os.path.join(self.paths_dict['figdir'], fname)
            _log.info(f"Saving to {save_name}")
            fig.savefig(save_name)

        if return_figax:
            return fig, axes


    def plot_eff_noise_patterns(self, ideal_Poisson=False, save=False, 
        ylim=None, return_figax=False):
        """Plot effective noise of slope fits for variety of read patterns"""

        det = self.det
        gain = det.gain
        nchan = det.nout
        patterns = list(det.multiaccum._pattern_settings.keys())

        en_spat_list = self.eff_noise_dict['en_spat_list']
        en_dn_list = []
        for i in range(len(patterns)):
            # Average spatial and temporal values
            var_avg_ch = en_spat_list[i]**2
            en_dn_list.append(np.sqrt(var_avg_ch[0:nchan].mean(axis=0)))

        tarr_all = self.eff_noise_dict['tarr_all_list']
        ng_all_list = self.eff_noise_dict['ng_all_list']
        p_excess = self.eff_noise_dict['p_excess']

        # Average dark current (e-/sec)
        if self.dark_ramp_dict is None:
            idark_avg = det.dark_current
        else:
            idark = []
            tarr = self.time_arr
            for ch in np.arange(nchan):
                y = self.dark_ramp_dict['ramp_avg_ch'][ch]
                cf = jl_poly_fit(tarr, y, deg=1)
                idark.append(cf[1])
            idark = np.array(idark) * gain
            idark_avg = np.mean(idark)

        # Average read noise per frame (e-)
        cds_var = (en_dn_list[0][0] * det.time_group * gain)**2 - (idark_avg * det.time_group)
        read_noise = np.sqrt(cds_var / 2)

        fig, axes = plt.subplots(3,3, figsize=(14,9), sharey=True)
        axes = axes.flatten()

        log_prev = conf.logging_level
        setup_logging('WARN', verbose=False)
        for i, ax in enumerate(axes):
            tvals = tarr_all[i]
            yvals = (en_dn_list[i] * tvals)

            xvals = tvals
            ax.plot(xvals, yvals, marker='o', label='Measured')
            
            det_new = deepcopy(det)
            ma_new = det_new.multiaccum
            ma_new.read_mode = patterns[i]

            ng_all = ng_all_list[i]
            thr_e = det_new.pixel_noise(ng=ng_all, rn=read_noise, idark=idark_avg, 
                                        ideal_Poisson=ideal_Poisson, p_excess=[0,0], scale_ints=False)
            
            yvals = (thr_e * tvals) / gain
            ax.plot(xvals, yvals, color='C1', label='Theory')

            tvals = tarr_all[i]
            ng_all = ng_all_list[i]
            thr_e = det_new.pixel_noise(ng=ng_all, rn=read_noise, idark=idark_avg, 
                                        ideal_Poisson=ideal_Poisson, p_excess=p_excess, scale_ints=False)
            
            yvals = (thr_e * tvals) / gain
            ax.plot(xvals, yvals, marker='.', color='C1', ls='--', label='Theory + Excess')

        for i, ax in enumerate(axes):
            if i==0:
                xr = [ax.get_xlim()[0],1200]
                ymax = 5*(int(ax.get_ylim()[1] / 5) + 1)
                yr = [0,ymax] if ylim is None else ylim
                
            ax.set_xlim(xr)
            ax.set_ylim(yr)
            ax.set_title(patterns[i])
            
            if i>5:
                ax.set_xlabel('Time (sec)')
            if np.mod(i,3) == 0:
                ax.set_ylabel('Noise (DN)')
            
        # Legend on first plot
        axes[0].legend()

        fig.suptitle(f'Noise of Slope Fits (SCA {self.scaid})', fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, wspace=0.03)

        setup_logging(log_prev, verbose=False)

        if save:
            fname = f'{self.scaid}_eff_noise_patterns.pdf'
            save_name = os.path.join(self.paths_dict['figdir'], fname)
            _log.info(f"Saving to {save_name}")
            fig.savefig(save_name)

        if return_figax:
            return fig, axes


    def plot_power_spectrum(self, save=False, cds=True, return_figax=False):

        scaid = self.scaid

        cds_dict = self.cds_act_dict
        keys = ['spat_det', 'temp_pink_corr', 'temp_pink_uncorr']
        cds_vals = [np.sqrt(np.mean(cds_dict[k]**2, axis=0)) for k in keys]
        rd_noise_cds, c_pink_cds, u_pink_cds = cds_vals

        nchan = self.nchan

        freq = self.pow_spec_dict['freq']
        ps_all = self.pow_spec_dict['ps_all']

        fig, axes = plt.subplots(1,2, figsize=(14,5))

        ax = axes[0]
        
        # Amplifier averages
        x = freq
        y = np.mean(ps_all, axis=0)
        label='Amplifier Averaged'
        ax.loglog(x[1:], y[1:], marker='o', ms=0.25, ls='none', color='grey', 
                  label=label, rasterized=True)

        # White Noise
        yf = x**(0)
        cds_var = np.mean(rd_noise_cds**2)
        yf1 = len(yf) * cds_var * yf / yf.sum() 
        ax.plot(x[1:], yf1[1:], ls='--', lw=1, label='White Noise')

        # Pink Noise per Channel
        yf = x**(-1); yf[0]=0
        cds_var = np.mean(u_pink_cds**2) / np.sqrt(2)
        yf2 = len(yf) * cds_var * yf / yf.sum()
        ax.plot(x[1:], yf2[1:], ls='--', lw=1, label='Uncorr Pink Noise')

        # Correlated Pink Noise
        yresid = y - yf2 - yf1
        scales = fit_corr_powspec(x, yresid)
        yf = broken_pink_powspec(x, scales)
        cds_var = c_pink_cds**2 / np.sqrt(2)
        yf3 = len(yf) * cds_var * yf / yf.sum() 
        ax.plot(x[1:], yf3[1:], ls='--', lw=1, label='Corr Pink Noise')

        # Total of the three components
        yf_sum = (yf1 + yf2 + yf3) 
        ax.plot(x[1:], yf_sum[1:], ls='--', lw=2, label='Sum')

        ax.set_ylabel('CDS Power (DN$^2$)')

        ax = axes[1]

        x = freq
        for ch in range(nchan):
            y = ps_all[ch]
            ax.loglog(x[1:], y[1:], marker='o', ms=0.25, ls='none', 
                      label=f'Ch{ch}', rasterized=True)
            
        for ax in axes:
            ax.set_xlim([5e-2, 7e4])
            xloc = np.array(ax.get_xticks())
            xlim = ax.get_xlim()
            xind = (xloc>=xlim[0]) & (xloc<=xlim[1])
            ax.set_xlabel('Frequency (Hz)')
            
            ax.set_xlim(xlim)
            ax.set_ylim([10,1e7])
            ax.legend(numpoints=3, markerscale=10)

            ax2 = ax.twiny()
            ax2.set_xlim(1/np.array(xlim))
            ax2.set_xscale('log')
            ax2.set_xlabel('Time (sec)')
            # new_tick_locations = xloc[xind]
            # ax2.set_xticks(new_tick_locations)
            # ax2.set_xticklabels(tick_function(new_tick_locations))

            ax.minorticks_on()


        fig.suptitle(f'Noise Power Spectrum (SCA {scaid})', fontsize=16)

        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        if save:
            fname = f'{scaid}_power_spectra.pdf'
            save_name = os.path.join(self.paths_dict['figdir'], fname)
            _log.info(f"Saving to {save_name}")
            fig.savefig(save_name, dpi=150)

        if return_figax:
            return fig, axes

class nircam_cal(nircam_dark):
    
    """ NIRCam Calibration class
    
    Assumes that all cal files exist in the calibration directory in PYNRC_PATH.
    """

    def __init__(self, scaid, same_scan_direction=False, reverse_scan_direction=False,
                 DMS=False, verbose=True):
                        

        self.DMS = DMS

        # Directory information
        # In case detname (A1...B5) is specified rather than scaid (481...490)
        self.scaid = detname_to_scaid(scaid)
        caldir = os.path.join(conf.PYNRC_PATH, 'calib') + '/'
        self._create_dir_structure(None, caldir)

        prev_log = conf.logging_level
        if verbose:
            setup_logging('INFO', verbose=False)
        else:
            setup_logging('WARN', verbose=False)

        # Set up detector information
        self.det = DetectorOps(detector=scaid)
        self.det.same_scan_direction = same_scan_direction
        self.det.reverse_scan_direction = reverse_scan_direction

        hdr = self._grab_single_header()

        # Detector size
        try:
            nx, ny, nz = (hdr['SUBSIZE1'], hdr['SUBSIZE2'], hdr['NGROUPS'])
        except:
            nx = hdr['NAXIS1']
            ny = hdr['NAXIS2']
            nz = hdr['NGROUP']

        self.det.multiaccum.ngroup = nz
        self.det.ypix = ny
        self.det.xpix = nx

        # Create masks for ref pixels, active pixels, and channels
        self._create_pixel_masks()

        self._init_attributes()
        
        # Dark ramp/slope info
        
        # Calculate dark slope image
        self.get_dark_slope_image()

        # Calculate pixel slope averages
        self.get_pixel_slope_averages()

        # Calculate CDS Noise for various component 
        # white noise, 1/f noise (correlated and independent), temporal and spatial
        self.get_cds_dict()

        # Effective Noise
        self.get_effective_noise()

        # Get kTC reset noise, IPC, and PPC values
        self.get_ktc_noise()

        # Get the power spectrum information
        # Saved to pow_spec_dict['freq', 'ps_all', 'ps_corr', 'ps_ucorr']
        self.get_power_spectrum(include_oh=False, calc_cds=True, mn_func=np.median, per_pixel=False)
        
        # Calculate IPC/PPC kernels
        self.get_ipc(calc_ppc=True)
        # Deconvolve the super dark and super bias images
        self.deconvolve_supers()
        
        # Get column variations
        self.get_column_variations()
        # Create dictionary of reference pixel behavior
        self.get_ref_pixel_noise()

        self.get_nonlinear_coeffs()
        try:
            self.get_linear_coeffs()
        except:
            _log.info('Skipping linearity coefficients. Not needed for simulations...')
        self.get_super_flats()
        
        setup_logging(prev_log, verbose=False)


#######################################
# Open and return FITS info
#######################################

def get_fits_data(fits_file, return_header=False, bias=None, reffix=False, 
                  DMS=False, int_ind=0, grp_ind=None, **kwargs):
    
    """ Read in FITS file data

    Parameters
    ==========
    fname : str
        FITS file (including path) to open.
    return_header : bool
        Return header as well as data?
    bias : ndarray
        If specified, will subtract bias image from ramp.
    reffix : bool
        Perform reference correction? 
    DMS : bool
        Is the FITS file DMS format?
    int_ind : int
        If DMS format, select integration index to extract.
        DMS FITS files usually have all integrations within
        a given exposure in a single FITS extension, which
        can be quite large.
    grp_ind : 2-element array
        Option to index specific groups from the data.
        For instance `grp_ind=[0:10]` will select only
        the first 10 groups from the FITS cube.
    
    Keyword Args
    ============
    altcol : bool
        Calculate separate reference values for even/odd columns. (default: True)
    supermean : bool
        Add back the overall mean of the reference pixels. (default: False)
    top_ref : bool
        Include top reference rows when correcting channel offsets. (default: True)
    bot_ref : bool
        Include bottom reference rows when correcting channel offsets. (default: True)
    ntop : int
        Specify the number of top reference rows. (default: 4)
    nbot : int
        Specify the number of bottom reference rows. (default: 4)
    mean_func : func
        Function used to calculate averages. (default: `robust.mean`)

    left_ref : bool
        Include left reference cols when correcting 1/f noise. (default: True)
    right_ref : bool
        Include right reference cols when correcting 1/f noise. (default: True)
    nleft : int
        Specify the number of left reference columns. (default: 4)
    nright : int
        Specify the number of right reference columns. (default: 4)
    perint : bool
        Smooth side reference pixel per integration, otherwise do frame-by-frame.
        (default: False)
    avg_type :str
        Type of side column averaging to perform to determine ref pixel drift. 
        Allowed values are 'pixel', 'frame', or 'int' (default: 'frame'):
        
            * 'int'   : Subtract the avg value of all side ref pixels in ramp.
            * 'frame' : For each frame, get avg of side ref pixels and subtract framewise.
            * 'pixel' : For each ref pixel, subtract its avg value from all frames.

    savgol : bool
        Use Savitsky-Golay filter method rather than FFT. (default: True)
    winsize : int
        Size of the window filter. (default: 31)
    order : int
        Order of the polynomial used to fit the samples. (default: 3)
    """
    
    # Want to automatically determine if FITS files have DMS structure
    hdul = fits.open(fits_file)
    hdr = hdul[0].header

    if DMS:
        if int_ind > hdr['NINTS']-1:
            hdul.close()
            nint = hdr['NINTS']
            raise ValueError(f'int_num must be less than {nint}.')

        data = hdul[1].data[int_ind]
    else:
        data = hdul[0].data

    # Select group indices
    if grp_ind is not None:
        data = data[grp_ind[0]:grp_ind[1]]
    # Convert to float
    data = data.astype(float)
    hdul.close()

    if bias is not None:
        data -= bias
    
    if reffix:
        data = reffix_hxrg(data, **kwargs)

    if return_header:
        return data, hdr
    else:
        return data

def ramp_resample(data, det_new, return_zero_frame=False):
    """ Resample a RAPID dataset into new detector format"""
    
    nz, ny, nx = data.shape
    
    # x1, y1 = (det_new.x0, det_new.y0)
    xpix, ypix = (det_new.xpix, det_new.ypix)
    # x2 = x1 + xpix
    # y2 = y1 + ypix

    # Do we need to crop out subarray?
    if ny==ypix:
        y1, y2 = (0, ny)
    else: # Will crop a subarray out of data 
        y1 = det_new.y0
        y2 = int(y1 + ypix)
    if nx==xpix:
        x1, x2 = (0, nx)
    else: # Will crop a subarray out of data 
        x1 = det_new.x0
        x2 = int(x1 + xpix)

    ma  = det_new.multiaccum
    nd1     = ma.nd1
    nd2     = ma.nd2
    nf      = ma.nf
    ngroup  = ma.ngroup        

    # Number of total frames up the ramp (including drops)
    # Keep last nd2 for reshaping
    nread_tot = nd1 + ngroup*nf + (ngroup-1)*nd2
    
    assert nread_tot <= nz, f"Output ramp has more total read frames ({nread_tot}) than input ({nz})."

    # Crop dataset
    data_out = data[0:nread_tot, y1:y2, x1:x2]

    # Save the first frame (so-called ZERO frame) for the zero frame extension
    if return_zero_frame:
        zeroData = deepcopy(data_out[0])
        
    # Remove drops and average grouped data
    if nf>1 or nd2>0:
        # Trailing drop frames were already excluded, so need to pull off last group of avg'ed frames
        data_end = data_out[-nf:,:,:].mean(axis=0) if nf>1 else data[-1:,:,:]
        data_end = data_end.reshape([1,ypix,xpix])
        
        # Only care about first (n-1) groups for now
        # Last group is handled separately
        data_out = data_out[:-nf,:,:]

        # Reshape for easy group manipulation
        data_out = data_out.reshape([-1,nf+nd2,ypix,xpix])
        
        # Trim off the dropped frames (nd2)
        if nd2>0: 
            data_out = data_out[:,:nf,:,:]

        # Average the frames within groups
        # In reality, the 16-bit data is bit-shifted
        data_out = data_out.reshape([-1,ypix,xpix]) if nf==1 else data_out.mean(axis=1)

        # Add back the last group (already averaged)
        data_out = np.append(data_out, data_end, axis=0)

    if return_zero_frame:
        return data_out, zeroData
    else:
        return data_out


#######################################
# Initial super bias function
#######################################

def _wrap_super_bias_for_mp(arg):
    args, kwargs = arg

    fname = args[0]
    data, hdr = get_fits_data(fname, return_header=True, reffix=True, **kwargs)

    # Get header information and create a NIRCam detector timing instance
    det = create_detops(hdr, DMS=kwargs['DMS'])

    # Time array
    tarr = det.times_group_avg

    deg = kwargs['deg']
    cf = jl_poly_fit(tarr, data, deg=deg)

    return cf[0]

def gen_super_bias(allfiles, DMS=False, mn_func=np.median, std_func=robust.std, 
                   return_std=False, deg=1, nsplit=3, return_all=False, **kwargs):
    """ Generate a Super Bias Image

    Read in a number of dark ramps, perform a polynomial fit to the data,
    and return the average of all bias offsets. This a very simple
    procedure that is useful for estimating an initial bias image. 
    Will not work well for weird pixels. 
    """
    
    # Set logging to WARNING to suppress messages
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)

    kw = kwargs.copy()
    kw['deg'] = deg
    kw['DMS'] = DMS

    if DMS:
        worker_args = []
        for f in allfiles:
            hdr = fits.getheader(f)
            # Account for multiple ints in each file
            for i in range(hdr['NINTS']):
                kw_i = kw.copy()
                kw_i['int_ind'] = i
                worker_args.append(([f],kw_i))
    else:
        worker_args = [([f],kw) for f in allfiles]
    
    nfiles = len(allfiles)
    
    if nsplit is None or nsplit<=1:
        bias_all = np.asarray([_wrap_super_bias_for_mp(wa) for wa in tqdm(worker_args)])
    else:
        bias_all = []
        # pool = mp.Pool(nsplit)
        try:
            with mp.Pool(nsplit) as pool:
                for res in tqdm(pool.imap_unordered(_wrap_super_bias_for_mp, worker_args), total=nfiles):
                    bias_all.append(res)
                pool.close()  # TODO: not sure if this is necessary?

            # bias_all = pool.map(_wrap_super_bias_for_mp, worker_args)
            if bias_all[0] is None:
                raise RuntimeError('Returned None values. Issue with multiprocess??')
        except Exception as e:
            _log.error('Caught an exception during multiprocess.')
            _log.error('Closing multiprocess pool.')
            pool.terminate()
            pool.close()
            raise e
        else:
            # Set back to previous logging level
            setup_logging(log_prev, verbose=False)
            _log.info('Closing multiprocess pool.')
            # pool.close()

        bias_all = np.asarray(bias_all)

    if return_all:
        return bias_all

    # Set back to previous logging level
    setup_logging(log_prev, verbose=False)

    super_bias = mn_func(bias_all, axis=0)
    if return_std:
        _super_bias = std_func(bias_all,axis=0)
        return super_bias, _super_bias
    else:
        return super_bias


def chisqr_red(yvals, yfit=None, err=None, dof=None,
               err_func=np.std):
    """ Calculate reduced chi square metric
    
    If yfit is None, then yvals assumed to be residuals.
    In this case, `err` should be specified.
    
    Parameters
    ==========
    yvals : ndarray
        Sampled values.
    yfit : ndarray
        Model fit corresponding to `yvals`.
    dof : int
        Number of degrees of freedom (nvals - nparams - 1).
    err : ndarray or float
        Uncertainties associated with `yvals`. If not specified,
        then use yvals point-to-point differences to estimate
        a single value for the uncertainty.
    err_func : func
        Error function uses to estimate `err`.
    """
    
    if (yfit is None) and (err is None):
        print("Both yfit and err cannot be set to None.")
        return
    
    diff = yvals if yfit is None else yvals - yfit
    
    sh_orig = diff.shape
    ndim = len(sh_orig)
    if ndim==1:
        if err is None:
            err = err_func(yvals[1:] - yvals[0:-1]) / np.sqrt(2)
        dev = diff / err
        chi_tot = np.sum(dev**2)
        dof = len(chi_tot) if dof is None else dof
        chi_red = chi_tot / dof
        return chi_red
    
    # Convert to 2D array
    if ndim==3:
        sh_new = [sh_orig[0], -1]
        diff = diff.reshape(sh_new)
        yvals = yvals.reshape(sh_new)
        
    # Calculate errors for each element
    if err is None:
        err_arr = np.array([yvals[i+1] - yvals[i] for i in range(sh_orig[0]-1)])
        err = err_func(err_arr, axis=0) / np.sqrt(2)
        del err_arr
    else:
        err = err.reshape(diff.shape)
    # Get reduced chi sqr for each element
    dof = sh_orig[0] if dof is None else dof
    chi_red = np.sum((diff / err)**2, axis=0) / dof
    
    if ndim==3:
        chi_red = chi_red.reshape(sh_orig[-2:])
        
    return chi_red

#######################################
# Super dark with more advanced bias
#######################################

def ramp_derivative(y, dx=None, fit0=True, deg=2, ifit=[0,10]):
    """
    Get the frame-by-frame derivative of a ramp.

    Parameters
    ==========
    y : ndarray
        Array of values (1D, 2D or 3D)
    dx : float 
        If dx is supplied, divide by value to get dy/dx.
    fit0 : bool
        In order to find slope of element 0, we have the option
        to fit some number of values to extrapolate this value.
        If not set, then dy0 = 2*dy[0] - dy[1].
    ifit : 2-element array
        Indices to fit in order to extrapolate dy0. Don't
        necessarily want to fit the entire dataset.
    deg : int
        Polynomial degree to use for extrapolation fit.
    """

    sh_orig = y.shape
    ndim = len(sh_orig)

    if ndim==1:
        dy = y[1:] - y[:-1]
        
        if fit0:
            xtemp = np.arange(len(dy))+1
            lxmap = [np.min(xtemp), np.max(xtemp)]
            i1, i2 = ifit
            
            xfit = xtemp[i1:i2+1]
            dyfit = dy[i1:i2+1]
            
            # First try to fit log/log
            xfit_log = np.log10(xfit+1)
            dyfit_log = np.log10(dyfit)
            
            # if there are no NaNs, then fit to log scale
            if not np.isnan(dyfit_log.sum()):
                lxmap_log = np.log10(lxmap)
                cf = jl_poly_fit(xfit_log, dyfit_log, deg=deg, use_legendre=True, lxmap=lxmap_log)
                dy0_log = jl_poly(0, cf, use_legendre=True, lxmap=lxmap_log)
                dy0 = 10**dy0_log
            else:            
                cf = jl_poly_fit(xtemp[i1:i2+1], dy[i1:i2+1], deg=deg, use_legendre=True, lxmap=lxmap)
                dy0 = jl_poly(0, cf, use_legendre=True, lxmap=lxmap)
        else:
            dy0 = 2*dy[0] - dy[1]

        dy = np.insert(dy, 0, dy0)

        if dx is not None:
            dy /= dx
            
        return dy
    
    # If fitting multiple pixels simultaneously
    # Convert to 2D array
    elif ndim==3:
        sh_new = [sh_orig[0], -1]
        y = y.reshape(sh_new)

    # Get differential
    dy = y[1:] - y[:-1]
    
    # Fit to slope to determine derivative of first element
    if fit0:
        xtemp = np.arange(len(dy))+1
        lxmap = [np.min(xtemp), np.max(xtemp)]
        i1, i2 = ifit

        # Value on which to perform fit
        xfit = xtemp[i1:i2+1]
        dyfit = dy[i1:i2+1]

        # First try to fit in log/log space
        xfit_log = np.log10(xfit+1)
        dyfit_log = np.log10(dyfit)

        # Variable to hold first element of differential
        dy0 = np.zeros([dy.shape[-1]])

        # Filter pixels that have valid values in logspace
        indnan = np.isnan(dyfit_log.sum(axis=0))
        # Fit invalid values in linear space
        cf = jl_poly_fit(xfit, dyfit[:,indnan], deg=deg, use_legendre=True, lxmap=lxmap)
        dy0[indnan] = jl_poly(0, cf, use_legendre=True, lxmap=lxmap)
        # Fit non-NaN'ed data in logspace
        if len(indnan[~indnan])>0:
            lxmap_log = np.log10(lxmap)
            cf = jl_poly_fit(xfit_log, dyfit_log[:,~indnan], deg=deg, use_legendre=True, lxmap=lxmap_log)
            dy0_log = jl_poly(0, cf, use_legendre=True, lxmap=lxmap_log)
            dy0[~indnan] = 10**dy0_log
    else:
        dy0 = 2*dy[0] - dy[1]
        
    dy = np.insert(dy, 0, dy0, axis=0)

    if ndim==3:
        dy = dy.reshape(sh_orig)

    if dx is not None:
        dy /= dx

    return dy

def gen_super_dark(allfiles, super_bias=None, DMS=False, **kwargs):
    """
    Average together all dark ramps to create a super dark ramp.
    First subtracts a bias frame. Tries to decipher t=0 intercept
    for odd behaving pixels.
    """
    
    # Set logging to WARNING to suppress messages
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)

    if super_bias is None:
        super_bias = 0
        
    # Header info from first file
    hdr = fits.getheader(allfiles[0])
    det = create_detops(hdr, DMS=DMS)

    # nchan = det.nout
    nx = det.xpix
    ny = det.ypix
    nz = det.multiaccum.ngroup
    # chsize = det.chsize

    # tarr = np.arange(1, nz+1) * det.time_group
    tarr = det.times_group_avg

    # Active and reference pixel masks
    mask_ref = det.mask_ref
    mask_act = ~mask_ref
    
    # TODO: Better algorithms to find bad pixels
    # See Bad_pixel_changes.pdf from Karl
    masks_dict = {
        'mask_ref': [],
        'mask_poly': [],
        'mask_deviant': [],
        'mask_negative': [],
        'mask_others': []
    }
    bias_off_all = []
    
    # Create a super dark ramp
    ramp_sum = np.zeros([nz,ny,nx])
    ramp_sum2 = np.zeros([nz,ny,nx])
    nsum = np.zeros([ny,nx])
    nint_tot = 0
    nfiles = len(allfiles)
    iter_files = tqdm(allfiles, desc='Files', leave=False) if nfiles>1 else allfiles
    for fname in iter_files:

        # If DMS, then might be multiple integrations per FITS file
        nint = fits.getheader(fname)['NINTS'] if DMS else 1
        nint_tot += nint  # Accounts for multiple ints FITS

        iter_range = trange(nint, desc='Ramps', leave=False) if nint>1 else range(nint)
        for i in iter_range:
            data = get_fits_data(fname, return_header=False, bias=super_bias,
                                 reffix=True, DMS=DMS, int_ind=i, **kwargs)

            # Fit everything with linear first
            deg = 1
            cf_all = np.zeros([3,ny,nx])
            cf_all[:2] = jl_poly_fit(tarr[1:], data[1:,:,:], deg=deg)
            yfit = jl_poly(tarr, cf_all)
            dof = data.shape[0] - deg
            # Get reduced chi-sqr metric
            chired_poly = chisqr_red(data, yfit=yfit, dof=dof)

            # Fit polynomial to those not well fit by linear func
            chi_cutoff = 2
            ibad = ((chired_poly > chi_cutoff) | np.isnan(chired_poly)) & mask_act
            deg = 2
            cf_all[:,ibad] = jl_poly_fit(tarr[1:], data[1:,ibad], deg=deg)
            yfit[:,ibad] = jl_poly(tarr, cf_all[:,ibad])
            dof = data.shape[0] - deg
            # Get reduced chi-sqr metric for poorly fit data
            chired_poly[ibad] = chisqr_red(data[:,ibad], yfit=yfit[:,ibad], dof=dof)
            
            del yfit

            # Find pixels poorly fit by any polynomial
            ibad = ((chired_poly > chi_cutoff) | np.isnan(chired_poly)) & mask_act
            bias_off = cf_all[0]
            bias_off[ibad] = 0

            # Those active pixels well fit by a polynomial
            mask_poly = (chired_poly <= chi_cutoff) & mask_act

            # Pixels with large deviations (5-sigma outliers)
            med_diff = np.median(cf_all[1])*tarr.max()
            std_diff = robust.std(cf_all[1])*tarr.max()
            mask_deviant = (data[-1] - data[1]) > (med_diff + std_diff*5)

            # Pixels with negative slopes
            mask_negative = (data[-1] - data[1]) < -(med_diff + std_diff*5)

            # Others
            mask_others = (~mask_poly) & (~mask_ref) & (~mask_deviant) & (~mask_negative)
            
            # Save to masks lists
            masks_dict['mask_poly'].append(mask_poly)
            # masks_dict['mask_ref'].append(mask_ref)
            masks_dict['mask_deviant'].append(mask_deviant)
            masks_dict['mask_negative'].append(mask_negative)
            masks_dict['mask_others'].append(mask_others)

            # Fit slopes of weird pixels to get their y=0 (bias) offset
            # ifit_others = mask_deviant | mask_others | mask_negative
            ifit_others = (~mask_poly) & mask_act
            yvals_fit = data[0:15,ifit_others]
            dy = ramp_derivative(yvals_fit, fit0=True, deg=1, ifit=[0,10])
            yfit = np.cumsum(dy, axis=0)

            bias_off[ifit_others] = (yvals_fit[0] - yfit[0])
            bias_off_all.append(bias_off)

            # Subtact bias
            data -= bias_off
            
            igood = mask_poly | mask_ref
            nsum[igood] += 1
            for j, im in enumerate(data):
                ramp_sum[j,igood] += im[igood]
                ramp_sum2[j] += im

            del data, yfit

    # Take averages
    igood = (nsum >= 0.75*nint_tot)
    for im in ramp_sum:
        im[igood] /= nsum[igood]
    ramp_sum2 /= nint_tot
    
    # Replace empty ramp_sum pixels with ramp_sum2
    # izero = np.sum(ramp_sum, axis=0) == 0
    ramp_sum[:,~igood] = ramp_sum2[:,~igood]
    ramp_avg = ramp_sum
    
    # del ramp_sum2
    
    # Get average of bias offsets
    bias_off_all = np.array(bias_off_all)
    bias_off_avg = robust.mean(bias_off_all, axis=0)
    
    # Convert masks to arrays
    for k in masks_dict.keys():
        masks_dict[k] = np.array(masks_dict[k])

    # Pixels with negative values
    mask_neg = (ramp_avg[0] < 0) & mask_act
    bias_off = np.zeros_like(bias_off_avg)

    yvals_fit = ramp_avg[:,mask_neg]
    dy = ramp_derivative(yvals_fit[0:15], fit0=True, deg=1, ifit=[0,10])
    yfit = np.cumsum(dy, axis=0)
    bias_off[mask_neg] = (yvals_fit[0] - yfit[0])

    # Pixels with largish positive values (indicative of RC pixels)
    mask_large = (ramp_avg[0] > 1000) | (ramp_avg[-1] > 50000)
    yvals_fit = ramp_avg[:,mask_large]
    dy = ramp_derivative(yvals_fit[0:15], fit0=True, deg=2, ifit=[0,10])
    yfit = np.cumsum(dy, axis=0)
    bias_off[mask_large] = (yvals_fit[0] - yfit[0])

    # Remove from ramp_avg and add into bias_off_avg
    ramp_avg -= bias_off
    bias_off_avg += bias_off

    # Pixels continuing to have largish positive values (indicative of RC pixels)
    bias_off = np.zeros_like(bias_off_avg)
    mask_large = ramp_avg[0] > 10000
    yvals_fit = ramp_avg[:,mask_large]
    dy = ramp_derivative(yvals_fit[0:15], fit0=False)
    yfit = np.cumsum(dy, axis=0)
    bias_off[mask_large] += (yvals_fit[0] - yfit[0])

    # Remove from ramp_avg and add into bias_off_avg
    ramp_avg -= bias_off
    bias_off_avg += bias_off

    setup_logging(log_prev, verbose=False)

    return ramp_avg, bias_off_avg, masks_dict
    

def gen_super_ramp(allfiles, super_bias=None, DMS=False, grp_max=None, sat_vals=None, **kwargs):
    """
    Average together all linearity ramps to create a super ramp.
    Subtracts a bias frame to determine more appropriate pixel
    by pixel average. Tries to decipher t=0 intercept for odd 
    behaving pixels. Also returns bias offsets.
    """

    # Set logging to WARNING to suppress messages
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)

    if super_bias is None:
        super_bias = 0
            
    # Header info from first file
    hdr = fits.getheader(allfiles[0])
    det = create_detops(hdr, DMS=DMS)

    # nchan = det.nout
    nx = det.xpix
    ny = det.ypix
    nz = det.multiaccum.ngroup
    # chsize = det.chsize

    tarr = det.times_group_avg

    # Active and reference pixel masks
    mask_ref = det.mask_ref
    mask_act = ~mask_ref

    # TODO: Algorithms to find bad pixels
    # See Bad_pixel_changes.pdf from Karl

    if grp_max is None:
        grp_max = find_group_sat(allfiles[-1], DMS=DMS, bias=super_bias, sat_vals=None, sat_calc=0.998)
        grp_max = grp_max + 10
    grp_ind = [0,nz] if grp_max>nz else [0,grp_max]

    # Update number of read frames
    nz = grp_ind[1]
    det.multiaccum.ngroup = nz
    tarr = det.times_group_avg
    
    # Create a super dark ramp
    ramp_sum = np.zeros([nz,ny,nx])
    bias_off_all = []
    nint_tot = np.zeros([nz])
    nfiles = len(allfiles)
    iter_files = tqdm(allfiles, desc='Super Ramp', leave=False) if nfiles>1 else allfiles
    for fname in iter_files:

        # If DMS, then might be multiple integrations per FITS file
        nint = fits.getheader(fname)['NINTS'] if DMS else 1
        # nint_tot += nint  # Accounts for multiple ints FITS

        iter_range = trange(nint, desc='Ramps', leave=False) if nint>1 else range(nint)
        for i in iter_range:
            data = get_fits_data(fname, DMS=DMS, return_header=False, bias=super_bias,
                                 reffix=True, int_ind=i, grp_ind=grp_ind, **kwargs)

            # Saturation levels
            svals = find_sat(data, ref_info=det.ref_info) if sat_vals is None else sat_vals

            # Fit polynomial data at <50% well to find bias offset
            deg = 2
            cf_all = cube_fit(tarr, data, sat_vals=svals, sat_frac=0.50, deg=deg, ref_info=det.ref_info)
            bias_off = cf_all[0]
            bias_off_all.append(bias_off)
            data -= bias_off

            for j, im in enumerate(data):
                ramp_sum[j] += im
                # Increment total frames here
                # Catches data where ramp is truncated (incomplete data)
                nint_tot[j] += 1

            del data

    # Take averages
    ramp_sum /= nint_tot.reshape([-1,1,1])
    ramp_avg = ramp_sum
    
    # Get average of bias offsets
    bias_off_all = np.array(bias_off_all)
    bias_off_avg = robust.mean(bias_off_all, axis=0)
    
    # Pixels with negative values
    mask_neg = (ramp_avg[10] < 0) & mask_act
    bias_off = np.zeros_like(bias_off_avg)

    yvals_fit = ramp_avg[:,mask_neg]
    dy = ramp_derivative(yvals_fit[0:15], fit0=True, deg=1, ifit=[0,10])
    yfit = np.cumsum(dy, axis=0)
    bias_off[mask_neg] = (yvals_fit[0] - yfit[0])

    # Remove from ramp_avg and add into bias_off_avg
    ramp_avg -= bias_off
    bias_off_avg += bias_off

    setup_logging(log_prev, verbose=False)

    return ramp_avg, bias_off_avg
    

def plot_dark_histogram(im, ax, binsize=0.0001, return_ax=False, label='Active Pixels', 
                         plot_fit=True, plot_cumsum=True, color='C1', xlim=None, xlim_std=7):
    
    from astropy.modeling import models, fitting
    
    bins = np.arange(im.min(), im.max() + binsize, binsize)
    ig, vg, cv = hist_indices(im, bins=bins, return_more=True)
    # Number of pixels in each bin
    nvals = np.array([len(i) for i in ig])

    # Fit a Gaussian to get peak of dark current
    ind_nvals_max = np.where(nvals==nvals.max())[0][0]
    mn_init = cv[ind_nvals_max]
    std_init = robust.std(im)
    g_init = models.Gaussian1D(amplitude=nvals.max(), mean=mn_init, stddev=std_init)

    fit_g = fitting.LevMarLSQFitter()
    nvals_norm = nvals / nvals.max()
    ind_fit = (cv>mn_init-1*std_init) & (cv<mn_init+1*std_init)
    g_res = fit_g(g_init, cv[ind_fit], nvals_norm[ind_fit])

    bg_max_dn = g_res.mean.value
    bg_max_npix = g_res.amplitude.value

    ax.plot(cv, nvals_norm, label=label, lw=2)
    if plot_fit:
        ax.plot(cv, g_res(cv), label='Gaussian Fit', lw=1.5, color=color)
    label = 'Peak = {:.4f} DN/sec'.format(bg_max_dn)
    ax.plot(2*[bg_max_dn], [0,bg_max_npix], label=label, ls='--', lw=1, color=color)
    if plot_cumsum:
        ax.plot(cv, np.cumsum(nvals) / im.size, color='C3', lw=1, label='Cumulative Sum')
    
    ax.set_ylabel('Relative Number of Pixels')
    ax.set_title('All Active Pixels')

    if xlim is None:
        xlim = np.array([-1,1]) * xlim_std * g_res.stddev.value + bg_max_dn
        xlim[0] = np.min([0,xlim[0]])

    ax.set_xlabel('Dark Rate (DN/sec)')

    ax.set_xlim(xlim)#[0,2*bg_max_dn])
    ax.legend(loc='upper left')

    if return_ax:
        return ax


#######################################
# Column variations
#######################################

def gen_col_variations(allfiles, super_bias=None, super_dark_ramp=None, 
    DMS=False, **kwargs):
    """ Create a series of column offset models 

    Returns a series of ramp variations to add to entire columns
    as well as the probability a given column will be affected.

    Likely due to FETS in the ASIC preamp or ADC or detector
    column buffer jumping around and causing entire columns 
    within a ramp to transition between two states.

    """

    # Set logging to WARNING to suppress messages
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)

    hdr = fits.getheader(allfiles[0])
    det = create_detops(hdr, DMS=DMS)

    nchan = det.nout
    nx = det.xpix
    # ny = det.ypix
    # nz = det.multiaccum.ngroup
    chsize = det.chsize
        
    if super_dark_ramp is None: 
        super_dark_ramp = 0
    if super_bias is None: 
        super_bias = 0
    
    ramp_column_varations = []
    nbad = []
    for f in tqdm(allfiles, desc='Files'):

        # If DMS, then might be multiple integrations per FITS file
        nint = fits.getheader(f)['NINTS'] if DMS else 1

        iter_range = trange(nint, desc='Ramps', leave=False) if nint>1 else range(nint)
        for i in iter_range:
            # Subtract bias, but don't yet perform reffix
            data = get_fits_data(f, bias=super_bias, DMS=DMS, int_ind=i)

            # Subtract super_dark_ramp to get residuals
            data -= super_dark_ramp

            data = reffix_hxrg(data, **kwargs)

            # Take the median of each column
            data_ymed = np.median(data, axis=1)
            # Set each ramp residual to 0 offset
            data_ymed -= np.median(data_ymed, axis=0)
            # Get rid of residual channel offsets
            for ch in range(nchan):
                x1 = ch*chsize
                x2 = x1 + chsize
                data_ymed[:,x1:x2] -= np.median(data_ymed[:,x1:x2], axis=1).reshape([-1,1])

            del data

            # Get derivatives
            # dymed = ramp_derivative(data_ymed, fit0=False)

            # Determine which columns have large excursions
            ymed_avg = np.mean(data_ymed, axis=0)
            ymed_std = np.std(data_ymed, axis=0)

            # dymed_avg = np.mean(dymed, axis=0)
            # dymed_std = np.std(dymed, axis=0)
            # print(np.median(ymed_avg), np.median(ymed_std))
            # print(robust.std(ymed_avg), robust.std(ymed_std))

            # Mask of outliers
            mask_outliers1 = np.abs(ymed_avg) > np.median(ymed_avg) + 1*robust.std(ymed_avg)
            mask_outliers2 = ymed_std > np.median(ymed_std) + 1*robust.std(ymed_std)
            mask_outliers = mask_outliers1 | mask_outliers2
            mask_outliers[:4] = False
            mask_outliers[-4:] = False

            data_ymed_outliers = data_ymed[:,mask_outliers]
            # dymed_outliers = dymed[:,mask_outliers]

            # data_ymed_good = data_ymed[:,~mask_outliers]
            # dymed_good = dymed[:,~mask_outliers]

            ramp_column_varations.append(data_ymed_outliers)
            nbad.append(data_ymed_outliers.shape[1])

    ramp_column_varations = np.hstack(ramp_column_varations)

    nbad = np.array(nbad)
    prob_bad = np.mean(nbad/nx)

    setup_logging(log_prev, verbose=False)

    return ramp_column_varations, prob_bad


#######################################
# Reference pixel information
#######################################

# Main reference bias offsets
# Amplifier bias offsets
def get_bias_offsets(data, nchan=4, ref_bot=True, ref_top=True, npix_ref=4):
    """ Get Reference Bias Characteristics

    Given some ramp data, determine the average master bias offset
    as well as the relative individual amplifier offsets. Also
    return the frame-to-frame variations caused by the preamp
    resets.
    """
    
    if ref_bot==False and ref_top==False:
        print('Need top and/or bottom refernece to be True')
        return

    nz, ny, nx = data.shape
    chsize = int(nx/nchan)
    
    # Mask of top and/and bottom reference pixels
    mask_ref = np.zeros([ny,nx]).astype('bool')
    mask_ref[0:npix_ref,:] = ref_bot
    mask_ref[-npix_ref:,:] = ref_top

    # Reference offsets for each frame
    bias_off_frame = np.median(data[:,mask_ref], axis=1)
    bias_mn = np.mean(bias_off_frame)
    bias_std_f2f = robust.std(bias_off_frame)
    
    # Remove average bias offsets from each frame
    for i, im in enumerate(data):
        im -= bias_off_frame[i]
    
    # Determine amplifier offsets
    amp_mn_all = []
    amp_std_f2f_all = []
    for ch in range(nchan):
        mask_ch = np.zeros([ny,nx]).astype('bool')
        mask_ch[:,ch*chsize:(ch+1)*chsize] = True
        
        mask_ch_pix = mask_ref & mask_ch
        
        # Reference pixel offsets for this amplifier
        data_ch = data[:,mask_ch_pix]

        amp_off_frame = np.median(data_ch, axis=1)
        amp_mn = np.mean(amp_off_frame)
        amp_std_f2f = robust.std(amp_off_frame)
        
        amp_mn_all.append(amp_mn)
        amp_std_f2f_all.append(amp_std_f2f)
        
    amp_mn_all = np.array(amp_mn_all)
    amp_std_f2f_all = np.array(amp_std_f2f_all)

    return bias_mn, bias_std_f2f, amp_mn_all, amp_std_f2f_all

def get_oddeven_offsets(data, nchan=4, ref_bot=True, ref_top=True, bias_off=None, amp_off=None):
    """ Even/Odd Column Offsets

    Return the per-amplifier offsets of the even and odd
    columns relative after subtraction of the matster and
    amplifier bias offsets.
    """
    
    if bias_off is None:
        bias_off = 0
    if amp_off is None:
        amp_off = np.zeros(nchan)
    
    nz, ny, nx = data.shape
    chsize = int(nx / nchan)
    
    mask_ref_even = np.zeros([ny,nx]).astype('bool')
    mask_ref_even[0:4,0::2] = ref_bot
    mask_ref_even[-4:,0::2] = ref_top

    mask_ref_odd = np.zeros([ny,nx]).astype('bool')
    mask_ref_odd[0:4,1::2] = ref_bot
    mask_ref_odd[-4:,1::2] = ref_top

    ch_odd_vals_ref = []
    ch_even_vals_ref = []
    for ch in range(nchan):

        # Reference pixels
        mask_ch = np.zeros([ny,nx]).astype('bool')
        mask_ch[:,ch*chsize:(ch+1)*chsize] = True
        mask_even_ch = mask_ch & mask_ref_even
        mask_odd_ch = mask_ch & mask_ref_odd

        data_ref_even = data[:,mask_even_ch]
        data_ref_odd = data[:,mask_odd_ch]
        
        data_ref_even_offset = np.mean(data_ref_even) - bias_off - amp_off[ch]
        data_ref_odd_offset = np.mean(data_ref_odd) - bias_off - amp_off[ch]

        ch_odd_vals_ref.append(data_ref_odd_offset)
        ch_even_vals_ref.append(data_ref_even_offset)

    ch_odd_vals_ref = np.array(ch_odd_vals_ref)
    ch_even_vals_ref = np.array(ch_even_vals_ref)
    
    return ch_even_vals_ref, ch_odd_vals_ref

def get_ref_instability(data, nchan=4, ref_bot=True, ref_top=True, mn_func=np.median):
    """ Reference Pixel Instability

    Determine the instability of the average reference pixel
    values relative to the active pixels on a frame-to-frame
    basis. The procedure is to compute a series of CDS frames,
    then look at the peak distributions of the active pixels
    relative to the reference pixels.
    """
    
    cds = data[1:] - data[:-1]
    nz, ny, nx = data.shape
    
    chsize = int(nx / nchan)

    # Mask of active pixels
    mask_act = np.zeros([ny,nx]).astype('bool')
    mask_act[4:-4,4:-4] = True

    # Mask of top and bottom reference pixels
    mask_ref = np.zeros([ny,nx]).astype('bool')
    mask_ref[0:4,:] = ref_bot
    mask_ref[-4:,:] = ref_top
    
    ref_inst = []
    for ch in range(nchan):
        mask_ch = np.zeros([ny,nx]).astype('bool')
        mask_ch[:,ch*chsize:(ch+1)*chsize] = True

        cds_ref = cds[:,mask_ref & mask_ch]
        cds_act = cds[:,mask_act & mask_ch]

        cds_ref_mn = mn_func(cds_ref, axis=1)
        cds_act_mn = mn_func(cds_act, axis=1)

        # Relative to Reference
        cds_act_mn -= cds_ref_mn

        ref_inst.append(np.std(cds_act_mn) / np.sqrt(2))

    ref_inst = np.array(ref_inst)
    
    return ref_inst

def gen_ref_dict(allfiles, super_bias, super_dark_ramp=None, DMS=False, **kwargs):
    """ Generate Reference Pixel Behavior Dictionary
    """

    if super_dark_ramp is None:
        super_dark_ramp = 0

    # Header info from first file
    hdr = fits.getheader(allfiles[0])
    det = create_detops(hdr, DMS=DMS)
    nchan = det.nout
    
    bias_mn_ref_all = []      # Main bias average offset
    bias_std_f2f_ref_all = [] # Main bias standard deviation per int
    amp_mn_ref_all = []       # Amplifier ref offset per integration
    amp_std_f2f_ref_all = []  # Ampl Ref frame-to-frame variations

    # Even/Odd Column Offsets
    col_even_offset_ref = []
    col_odd_offset_ref = []

    # Ref Instability frame-to-frame
    amp_std_ref_act_all = [] 
    for fname in tqdm(allfiles, desc='Files'):

        # If DMS, then might be multiple integrations per FITS file
        nint = fits.getheader(fname)['NINTS'] if DMS else 1

        iter_range = trange(nint, desc='Ramps', leave=False) if nint>1 else range(nint)
        for i in iter_range:
            # Relative to super bias and super dark ramp
            data = get_fits_data(fname, bias=super_bias, DMS=DMS, int_ind=i)

            data -= super_dark_ramp

            # Get master and amplifer offsets
            res = get_bias_offsets(data, nchan=nchan)

            bias_mn_ref_all.append(res[0])
            bias_std_f2f_ref_all.append(res[1])
            amp_mn_ref_all.append(res[2])
            amp_std_f2f_ref_all.append(res[3])

            # bias_off was subtracted in-place from data within get_bias_offsets()
            res_col = get_oddeven_offsets(data, nchan=nchan, bias_off=0, amp_off=res[2])
            col_even_offset_ref.append(res_col[0])
            col_odd_offset_ref.append(res_col[1])

            # Reference pixel instabilities
            data = reffix_hxrg(data, **kwargs)
            ref_inst = get_ref_instability(data, nchan=nchan)
            amp_std_ref_act_all.append(ref_inst)

            del data

    bias_mn_ref_all      = np.array(bias_mn_ref_all)
    bias_std_f2f_ref_all = np.array(bias_std_f2f_ref_all)
    amp_mn_ref_all       = np.array(amp_mn_ref_all)
    amp_std_f2f_ref_all  = np.array(amp_std_f2f_ref_all)

    col_even_offset_ref = np.array(col_even_offset_ref)
    col_odd_offset_ref  = np.array(col_odd_offset_ref)

    amp_std_ref_act_all = np.array(amp_std_ref_act_all)
    
    
    ref_dict = {}

    # Master bias offsets
    ref_dict['master_bias_mean'] = bias_mn_ref_all.mean()
    ref_dict['master_bias_std'] = robust.medabsdev(bias_mn_ref_all)
    ref_dict['master_bias_f2f'] = np.sqrt(np.mean(bias_std_f2f_ref_all**2))

    # Amplifier Offsets
    ref_dict['amp_offset_mean'] = amp_mn_ref_all.mean(axis=0)
    # There can be correlations between offsets that depend on temperature
    # Let's remove those to get the true standard deviation
    cf = jl_poly_fit(bias_mn_ref_all, amp_mn_ref_all)
    amp_sub = amp_mn_ref_all - jl_poly(bias_mn_ref_all, cf)
    ref_dict['amp_offset_std'] = robust.std(amp_sub, axis=0)
    ref_dict['amp_offset_f2f'] = np.sqrt(np.mean(amp_std_f2f_ref_all**2, axis=0))

    # Correlation between master_bias_mean and amp_offset_mean
    ref_dict['master_amp_cf'] = cf

    # Even/Odd Column offsets
    ref_dict['amp_even_col_offset'] = (np.mean(col_even_offset_ref, axis=0))
    ref_dict['amp_odd_col_offset']  = (np.mean(col_odd_offset_ref, axis=0))

    # Reference instability relative active pixels
    ref_dict['amp_ref_inst_f2f'] = np.sqrt(np.mean(amp_std_ref_act_all**2, axis=0))

    _log.info("Reference Pixels")

    _log.info('')
    _log.info("Master Bias Mean")
    _log.info(ref_dict['master_bias_mean'])
    _log.info("Master Bias StDev")
    _log.info(ref_dict['master_bias_std'])
    _log.info("Master Bias Frame-to-Frame StDev")
    _log.info(ref_dict['master_bias_f2f'])

    _log.info('')
    _log.info("Amp Offset Mean")
    _log.info(ref_dict['amp_offset_mean'])
    _log.info("Amp Offset StDev")
    _log.info(ref_dict['amp_offset_std'])
    _log.info("Amp Offset Frame-to-Frame StDev")
    _log.info(ref_dict['amp_offset_f2f'])

    _log.info("")
    _log.info("Even Columns Offset")
    _log.info(ref_dict['amp_even_col_offset'])
    _log.info("Odd Columns Offset")
    _log.info(ref_dict['amp_odd_col_offset'])

    _log.info("")
    _log.info("Reference Instability")
    _log.info(ref_dict['amp_ref_inst_f2f'])
    
    return ref_dict


#######################################
# Detector Noise
#######################################


def calc_ktc(bias_sigma_arr, binsize=0.25, return_std=False):
    """ Calculate kTC (Reset) Noise

    Use the uncertainty image from super bias to calculate
    the kTC noise. This function generates a histogram of
    the pixel uncertainties and takes the peak of the 
    distribution as the pixel reset noise.

    Parameters
    ----------
    bias_sigma_arr : ndarray
        Image of the pixel uncertainties.
    binsize : float
        Size of the histogram bins.
    return_std : bool
        Also return the standard deviation of the 
        distribution?
    
    """

    im = bias_sigma_arr
    binsize = binsize
    bins = np.arange(im.min(), im.max() + binsize, binsize)
    ig, vg, cv = hist_indices(im, bins=bins, return_more=True)

    nvals = np.array([len(i) for i in ig])
    # nvals_rel = nvals / nvals.max()

    # Peak of distribution
    ind_peak = np.where(nvals==nvals.max())[0][0]
    peak = cv[ind_peak]

    if return_std:
        return peak, robust.medabsdev(im)
    else:
        return peak


def calc_cdsnoise(data, temporal=True, spatial=True, std_func=np.std):
    """ Calculate CDS noise from input image cube"""

    if (temporal==False) and (spatial==False):
        _log.warning("Must select one or both of `temporal` or `spatial`")
        return
    
    # Make sure we select same number of even/odd frame
    vals1 = data[0::2]
    vals2 = data[1::2]
    nz1 = vals1.shape[0]
    nz2 = vals2.shape[0]
    nz = np.min([nz1,nz2])
    
    # Calculate CDS image pairs
    cds_arr = vals2[:nz] - vals1[:nz]
    
    # CDS noise per pixel (temporal)
    if temporal:
        cds_temp = std_func(cds_arr, axis=0)
        # Take median of the variance
        cds_temp_med = np.sqrt(np.median(cds_temp**2))
        
    # CDS noise per frame (spatial)
    if spatial:
        sh = cds_arr.shape
        cds_spat = std_func(cds_arr.reshape([sh[0],-1]), axis=1)
        # Take median of the variance
        cds_spat_med = np.sqrt(np.median(cds_spat**2))
        
    if temporal and spatial:
        res = cds_temp_med, cds_spat_med
    elif temporal:
        res = cds_temp_med
    elif spatial:
        res = cds_spat_med
    
    return res


def gen_cds_dict(allfiles, DMS=False, superbias=None,
                 mask_good_arr=None, same_scan_direction=False):
    """ Generate dictionary of CDS noise info
    
    Calculate read noise for:
      1. Total noise (no column correcton)
      2. 1/f noise (no column correcton)
      3. Intrinsic read noise (w/ column correcton)
      4. Both temporal and spatial
    
    """

    # Header info from first file
    hdr = fits.getheader(allfiles[0])
    det = create_detops(hdr, DMS=DMS)
    
    nchan = det.nout
    nx = det.xpix
    ny = det.ypix
    nz = det.multiaccum.ngroup
    chsize = det.chsize

    cds_act_dict = {
        'spat_tot': [], 'spat_det': [], 
        'temp_tot': [], 'temp_det': [], 
        'spat_pink_corr': [], 'spat_pink_uncorr': [],
        'temp_pink_corr': [], 'temp_pink_uncorr': [],
    }

    cds_ref_dict = {
        'spat_tot': [], 'spat_det': [],
        'temp_tot': [], 'temp_det': [],
    }
    
    # Active and reference pixel masks
    lower, upper, left, right = det.ref_info

    # Reference pixel mask
    # Just use top and bottom ref pixel
    mask_ref = np.zeros([ny,nx], dtype='bool')
    if lower>0: mask_ref[0:lower,:] = True
    if upper>0: mask_ref[-upper:,:] = True
    
    # Active pixels mask
    mask_act = np.zeros([ny,nx], dtype='bool')
    mask_act[lower:-upper,left:-right] = True
    
    # Channel mask
    mask_channels = det.mask_channels
    # mask_channels = np.zeros([ny,nx])
    # for ch in range(nchan):
    #     mask_channels[:,ch*chsize:(ch+1)*chsize] = ch
        
    # Mask of good pixels
    if mask_good_arr is None:
        mask_good_arr = np.ones([nz,ny,nx], dtype='bool')

    kwargs = {
        'nchans': nchan, 'altcol': True, 'in_place': True,    
        'fixcol': False, 'avg_type': 'pixel', 'savgol': True, 'perint': False    
    }

    for fname in tqdm(allfiles, desc='Files'):

        # If DMS, then might be multiple integrations per FITS file
        nint = fits.getheader(fname)['NINTS'] if DMS else 1

        iter_range = trange(nint, desc='Ramps', leave=False) if nint>1 else range(nint)
        for i in iter_range:
            # Relative to super bias and super dark ramp
            data = get_fits_data(fname, bias=superbias, DMS=DMS, int_ind=i)

            ##################################
            # 1. Full noise (det + 1/f)
            kwargs['fixcol'] = False
            data = get_fits_data(fname, bias=superbias, reffix=True, 
                                 DMS=DMS, int_ind=i, **kwargs)

            # Active pixels in each channel
            cds_temp_arr = []
            cds_spat_arr = []
            indgood = (mask_good_arr[i]) & mask_act
            for ch in np.arange(nchan):
                ind = indgood & (mask_channels == ch)
                cds_temp, cds_spat = calc_cdsnoise(data[:,ind])
                cds_temp_arr.append(cds_temp)
                cds_spat_arr.append(cds_spat)
            cds_act_dict['temp_tot'].append(cds_temp_arr)
            cds_act_dict['spat_tot'].append(cds_spat_arr)

            # Reference pixels in each channel
            cds_temp_arr = []
            cds_spat_arr = []
            indgood = mask_ref
            for ch in np.arange(nchan):
                ind = indgood & (mask_channels == ch)
                cds_temp, cds_spat = calc_cdsnoise(data[:,ind])
                cds_temp_arr.append(cds_temp)
                cds_spat_arr.append(cds_spat)
            cds_ref_dict['temp_tot'].append(cds_temp_arr)
            cds_ref_dict['spat_tot'].append(cds_spat_arr)

            ##################################
            # 2. 1/f noise contributions

            # Create array of extracted 1/f noise
            # Work on CDS pairs
            fn_data = []
            cds_data = data[1:20:2] - data[0:20:2]
            for im in cds_data:
                ch_arr = im.reshape([ny,-1,chsize]).transpose([1,0,2])
                mask = np.abs(im - np.median(im)) > 10*robust.medabsdev(im)
                mask = mask.reshape([ny,-1,chsize]).transpose([1,0,2])
                fnoise = channel_smooth_savgol(ch_arr, mask=mask)
                fnoise = fnoise.transpose([1,0,2]).reshape([ny,nx])
                fn_data.append(fnoise)
            fn_data = np.array(fn_data)
            # Divide by sqrt(2) since we've already performed a CDS difference
            fn_data /= np.sqrt(2)

            # Split into correlated and uncorrelated components
            fn_data_corr = []
            for j, im in enumerate(fn_data):
                fn_corr = channel_averaging(im, nchans=nchan,  off_chans=False,
                    same_scan_direction=same_scan_direction, mn_func=np.mean)
                # Subtract from fn_data
                fn_data[j] -= fn_corr
                # Only append first channel since the rest are the same data
                fn_data_corr.append(fn_corr[:,0:chsize])
            fn_data_corr = np.array(fn_data_corr)
            
            # Active pixels noise in each channel for uncorrelated data
            cds_temp_arr = []
            cds_spat_arr = []
            indgood = (mask_good_arr[i]) & mask_act
            for ch in np.arange(nchan):
                ind = indgood & (mask_channels == ch)
                cds_temp, cds_spat = calc_cdsnoise(fn_data[:,ind])
                cds_temp_arr.append(cds_temp)
                cds_spat_arr.append(cds_spat)
            cds_act_dict['temp_pink_uncorr'].append(cds_temp_arr)
            cds_act_dict['spat_pink_uncorr'].append(cds_spat_arr)

            del fn_data

            # Active pixels noise in correlated channel data
            indgood = (mask_good_arr[i]) & mask_act
            ind = indgood[:,0:chsize]
            cds_temp, cds_spat = calc_cdsnoise(fn_data_corr[:,ind])
            cds_act_dict['temp_pink_corr'].append(cds_temp)
            cds_act_dict['spat_pink_corr'].append(cds_spat)

            del fn_data_corr

            ##################################
            # 3. Detector contributions
            kwargs['fixcol'] = True
            data = reffix_hxrg(data, **kwargs)

            # New 1/f noise array
            for j, im in enumerate(data):
                ch_arr = im.reshape([ny,-1,chsize]).transpose([1,0,2])
                fnoise = channel_smooth_savgol(ch_arr)
                fnoise = fnoise.transpose([1,0,2]).reshape([ny,nx])
                # Remove 1/f noise contributions
                data[j] -= fnoise

            # Active pixels in each channel
            cds_temp_arr = []
            cds_spat_arr = []
            indgood = (mask_good_arr[i]) & mask_act
            for ch in np.arange(nchan):
                ind = indgood & (mask_channels == ch)
                cds_temp, cds_spat = calc_cdsnoise(data[:,ind])
                cds_temp_arr.append(cds_temp)
                cds_spat_arr.append(cds_spat)
            cds_act_dict['temp_det'].append(cds_temp_arr)
            cds_act_dict['spat_det'].append(cds_spat_arr)

            # Reference pixels in each channel
            cds_temp_arr = []
            cds_spat_arr = []
            indgood = mask_ref
            for ch in np.arange(nchan):
                ind = indgood & (mask_channels == ch)
                cds_temp, cds_spat = calc_cdsnoise(data[:,ind])
                cds_temp_arr.append(cds_temp)
                cds_spat_arr.append(cds_spat)
            cds_ref_dict['temp_det'].append(cds_temp_arr)
            cds_ref_dict['spat_det'].append(cds_spat_arr)

            # Done with data
            del data

    # Convert lists to np.array
    dlist = [cds_act_dict, cds_ref_dict]
    for d in dlist:
        for k in d.keys():
            if isinstance(d[k], (list)):
                d[k] = np.array(d[k])

    return cds_act_dict, cds_ref_dict


def calc_eff_noise(allfiles, superbias=None, temporal=True, spatial=True, 
                   ng_all=None, DMS=False, kw_ref=None, std_func=robust.medabsdev,
                   kernel_ipc=None, kernel_ppc=None, read_pattern='RAPID'):
    """ Determine Effective Noise

    Calculates the slope noise (in DN/sec) assuming a linear fits to a variety
    number of groups. The idea is to visualize the reduction in noise as you
    increase the number of groups in the fit and compare it to theoretical
    predictions (ie., slope noise formula).
    
    Parameters
    ----------
    allfiles : list
        List of input file names.
    DMS : bool
        Are files DMS formatted?
    superbias : ndarray
        Super bias to subtract from each dataset.
    temporal : bool
        Calculate slope noise using pixels' temporal distribution?
    spatial : bool
        Calcualte slope noise using pixel spatial distribution?
    ng_all : array-like
        Array of group to perform linear fits for slope calculations.
    kw_ref : dict
        Dictionary of keywords to pass to reference correction routine.
    std_func : func
        Function for calculating spatial distribution.
    kernel_ipc : ndarray
        IPC kernel to perform deconvolution on slope images.
    kernel_ppc : ndarray
        Similar to `kernel_ipc` except for PPC.
    read_pattern : string
        Reformulate data as if it were acquired using a read pattern
        other than RAPID.
    """

    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)

    hdr = fits.getheader(allfiles[0])
    det = create_detops(hdr, DMS=DMS)
    
    nchan = det.nout
    nx = det.xpix
    ny = det.ypix
    chsize = det.chsize

    # Masks for active, reference, and amplifiers
    ref_mask = det.mask_ref
    act_mask = ~ref_mask
    ch_mask = det.mask_channels
        
    if 'RAPID' not in read_pattern:
        det_new = deepcopy(det)
        ma_new = det_new.multiaccum
        # Change read mode and determine max number of allowed groups
        ma_new.read_mode = read_pattern
        ma_new.ngroup = int((det.multiaccum.ngroup - ma_new.nd1 + ma_new.nd2) / (ma_new.nf + ma_new.nd2))

        nz = ma_new.ngroup
        # Group time
        # tarr = np.arange(1, nz+1) * det_new.time_group + (ma_new.nd1 - ma_new.nd2 - ma_new.nf/2)*det_new.time_frame
        tarr = det_new.times_group_avg
        # Select number of groups to perform linear fits
        if ng_all is None:
            if nz<20:
                ng_all = np.arange(2,nz+1).astype('int')
            else:
                ng_all = np.append([2,3], np.linspace(5,nz,num=16).astype('int'))
    else:
        nz = det.multiaccum.ngroup
        # Group time
        tarr = np.arange(1, nz+1) * det.time_group
        # Select number of groups to perform linear fits
        if ng_all is None:
            ng_all = np.append([2,3,5], np.linspace(10,nz,num=15).astype('int'))

    # Make sure ng_all is unique
    ng_all = np.unique(ng_all)

    # Do not remove 1/f noise via ref column
    if kw_ref is None:
        kw_ref = {
            'nchans': nchan, 'altcol': True, 'in_place': True,    
            'fixcol': False, 'avg_type': 'pixel', 'savgol': True, 'perint': False    
        }
        
    # IPC and PPC kernels
    if kernel_ipc is not None:
        ipc_big = pad_or_cut_to_size(kernel_ipc, (ny,nx))
        kipc_fft = np.fft.fft2(ipc_big)
    else:
        kipc_fft = None
    if kernel_ppc is not None:
        ppc_big = pad_or_cut_to_size(kernel_ppc, (ny,chsize))
        kppc_fft = np.fft.fft2(ppc_big)
    else:
        kppc_fft = None

    # Calculate effective noise temporally
    if temporal:

        eff_noise_temp = []
        # Work with one channel at a time for better memory management
        for ch in trange(nchan, desc="Temporal", leave=False):
            ind_ch = act_mask & (ch_mask==ch)

            slope_chan_allfiles = []
            slope_ref_allfiles = []
            for fname in tqdm(allfiles, leave=False, desc="Files"):
                # If DMS, then might be multiple integrations per FITS file
                nint = fits.getheader(fname)['NINTS'] if DMS else 1
                iter_range = trange(nint, desc='Ramps', leave=False) if nint>1 else range(nint)
                for i in iter_range:
                    data = get_fits_data(fname, bias=superbias, reffix=True, 
                                         DMS=DMS, int_ind=i, **kw_ref)

                    # Reformat data?
                    if 'RAPID' not in read_pattern:
                        data = ramp_resample(data, det_new)
                    
                    slope_chan = []
                    slope_ref = []
                    for fnum in tqdm(ng_all, leave=False, desc="Group Fit"):
                        bias, slope = jl_poly_fit(tarr[0:fnum], data[0:fnum])
                        
                        # Deconvolve fits to remove IPC and PPC
                        if kipc_fft is not None:
                            slope = ipc_deconvolve(slope, None, kfft=kipc_fft)
                        if kppc_fft is not None:
                            slope = ppc_deconvolve(slope, None, kfft=kppc_fft, in_place=True)

                        slope_chan.append(slope[ind_ch])

                        # Do reference pixels
                        if ch==nchan-1:
                            slope_ref.append(slope[ref_mask])

                    slope_chan_allfiles.append(np.array(slope_chan))
                    if ch==nchan-1:
                        slope_ref_allfiles.append(np.array(slope_ref))

                    del data

            slope_chan_allfiles = np.array(slope_chan_allfiles)
            # Reference pixels
            if ch==nchan-1:
                slope_ref_allfiles = np.array(slope_ref_allfiles)

            # Calculate std dev for each pixels
            std_pix = np.std(slope_chan_allfiles, axis=0)
            # Get the median of the variance distribution
            eff_noise = np.sqrt(np.median(std_pix**2, axis=1))
            eff_noise_temp.append(eff_noise)
            if ch==nchan-1:
                std_pix = np.std(slope_ref_allfiles, axis=0)
                eff_noise_ref = np.sqrt(np.median(std_pix**2, axis=1))
                eff_noise_temp.append(eff_noise_ref)
            del slope_chan, slope_chan_allfiles, std_pix

        eff_noise_temp = np.array(eff_noise_temp)

    # Calculate effective noise spatially
    if spatial:
        eff_noise_all = []
        for fname in tqdm(allfiles, desc="Spatial", leave=False):
            # If DMS, then might be multiple integrations per FITS file
            nint = fits.getheader(fname)['NINTS'] if DMS else 1
            iter_range = trange(nint, desc='Ramps', leave=False) if nint>1 else range(nint)
            for i in iter_range:
                data = get_fits_data(fname, bias=superbias, reffix=True, 
                                     DMS=DMS, ind_int=i, **kw_ref)

                # Reformat data?
                if 'RAPID' not in read_pattern:
                    data = ramp_resample(data, det_new)

                eff_noise_chans = []
                # Spatial standard deviation
                for fnum in tqdm(ng_all, leave=False, desc="Group Fit"):
                    bias, slope = jl_poly_fit(tarr[0:fnum], data[0:fnum])

                    # Deconvolve fits to remove IPC and PPC
                    if kipc_fft is not None:
                        slope = ipc_deconvolve(slope, None, kfft=kipc_fft)
                    if kppc_fft is not None:
                        slope = ppc_deconvolve(slope, None, kfft=kppc_fft, in_place=True)
                    
                    eff_noise = []
                    # Each channel
                    for ch in np.arange(nchan):
                        ind_ch = act_mask & (ch_mask==ch)
                        eff_noise.append(std_func(slope[ind_ch]))
                    # Add reference pixels
                    eff_noise.append(std_func(slope[ref_mask]))

                    # Append to final array
                    eff_noise_chans.append(np.array(eff_noise))

                eff_noise_chans = np.array(eff_noise_chans).transpose()
                eff_noise_all.append(eff_noise_chans)

                del data

        eff_noise_all = np.array(eff_noise_all)
        eff_noise_spat = np.median(eff_noise_all, axis=0)

    setup_logging(log_prev, verbose=False)
        
    if temporal and spatial:
        res = ng_all, eff_noise_temp, eff_noise_spat
    elif temporal:
        res = ng_all, eff_noise_temp
    elif spatial:
        res = ng_all, eff_noise_spat
    
    return res

def fit_func_var_ex(params, det, patterns, ng_all_list, en_dn_list, 
    read_noise=None, idark=None, ideal_Poisson=False):
    """Function for lsq fit to get excess variance"""
    
    gain = det.gain
    if idark is None:
        idark = det.dark_current
    
    # Read noise per frame
    if read_noise is None:
        cds_var = (en_dn_list[0][0] * det.time_group * gain)**2 - (idark * det.time_frame)
        read_noise = np.sqrt(cds_var / 2)
    
    diff_all = []
    for i, patt in enumerate(patterns):

        det_new = deepcopy(det)
        ma_new = det_new.multiaccum
        ma_new.read_mode = patt
        ma_new.ngroup = int((det.multiaccum.ngroup - ma_new.nd1 + ma_new.nd2) / (ma_new.nf + ma_new.nd2))

        ng_all = ng_all_list[i]
        thr_e = det_new.pixel_noise(ng=ng_all, rn=read_noise, idark=idark, 
                                    ideal_Poisson=ideal_Poisson, p_excess=[0,0], scale_ints=False)
        
        tvals = (ng_all - 1) * det_new.time_group
        var_ex_obs = (en_dn_list[i] * gain * tvals)**2 - (thr_e * tvals)**2

        nf = ma_new.nf
        var_ex_fit = var_ex_model(ng_all, nf, params)

        diff_all.append(var_ex_obs - var_ex_fit)
        
    return np.concatenate(diff_all)


#######################################
# IPC and PPC Deconvolution
#######################################

def deconv_single_image(im, kfft):
    """Image deconvolution for a kernel
    
    Perform deconvolution of an image using a kernel. This function
    calculates the FFT of the input image, divides by the kernel's
    pre-calculated FFT, then performs an inverse FFT to obtain the
    deconvolved image.

    Parameters
    ----------
    im : ndarray
        Input image to deconvolve.
    kfft : Complex ndarray
        FFT of the deconvolution kernel.
    """

    # bias the image to avoid negative pixel values in image
    min_im = np.min(im)
    im = im - min_im

    # FFT of input image
    imfft = np.fft.fft2(im)
    im_final = np.fft.fftshift(np.fft.ifft2(imfft/kfft).real, axes=(-2,-1)) 
    im_final += min_im

    return im_final

def ipc_deconvolve(imarr, kernel, kfft=None, **kwargs):
    """Simple IPC image deconvolution
    
    Given an image (or image cube), apply an IPC deconvolution kernel
    to obtain the intrinsic flux distribution. Should also work for 
    PPC kernels. This simply calculates the FFT of the image(s) and
    kernel, divides them, then applies an iFFT to determine the
    deconvolved image.
    
    If performing PPC deconvolution, make sure to perform channel-by-channel
    with the kernel in the appropriate scan direction. IPC is usually symmetric,
    so this restriction may not apply. See `ppc_deconvolve` function. Calls 
    `ppc_deconvolve` for asymmetric (left-right) IPC kernels.
 
    Parameters
    ==========
    im : ndarray
        Image or array of images. 
    kernel : ndarry
        Deconvolution kernel. Ignored if `kfft` is specified.
    kfft : Complex ndarray
        Option to directy supply the kernel's FFT rather than
        calculating it within the function. The supplied ndarray
        should have shape (ny,nx) equal to the input `im`. Useful
        if calling ``ipc_deconvolve`` multiple times.
    symmetric : bool
        Is the input IPC kernel symmetric?
    
    Keyword Args
    ============
    in_place : bool
        Perform calculate in place (overwrites original image). 
    nchans : int
        Number of amplifier channels.
    same_scan_direction : bool
        Are all the output channels read in the same direction?
        By default fast-scan readout direction is ``[-->,<--,-->,<--]``
        If ``same_scan_direction``, then all ``-->``
    reverse_scan_direction : bool
        If ``reverse_scan_direction``, then ``[<--,-->,<--,-->]`` or all ``<--``
    """

    # Image cube shape
    sh = imarr.shape
    ndim = len(sh)
    if ndim==2:
        ny, nx = sh
        nz = 1
        imarr = imarr.reshape([nz,ny,nx])
    else:
        nz, ny, nx = sh

    # FFT of kernel
    if kfft is None:
        ipc_big = pad_or_cut_to_size(kernel, (ny,nx))
        kfft = np.fft.fft2(ipc_big)

    im_final = np.zeros_like(imarr)
    for i in trange(nz, leave=False, desc='Frames'):
        im_final[i] = deconv_single_image(imarr[i], kfft)

    return im_final.reshape(sh)

def ppc_deconvolve(im, kernel, kfft=None, nchans=4, in_place=False,
    same_scan_direction=False, reverse_scan_direction=False, **kwargs):
    """PPC image deconvolution
    
    Given an image (or image cube), apply PPC deconvolution kernel
    to obtain the intrinsic flux distribution. This performs channel-by-channel
    deconvolution, taking into account the specific readout directly.
    This function can also be used for asymmetric IPC kernels.
 
    Parameters
    ==========
    im : ndarray
        Image or array of images. Assumes detector coordinates where
        (0,0) is in bottom left.
    kernel : ndarry
        Deconvolution kernel. Ignored if `kfft` is specified.
    kfft : Complex ndarray
        Option to directy supply the kernel's FFT rather than
        calculating it within the function. The supplied ndarray
        should have shape (ny,nx) equal to the input `im`. Useful
        if calling ``ppc_deconvolve`` multiple times.
    in_place : bool
        Perform calculate in place (overwrites original image). 
    nchans : int
        Number of amplifier channels.
    same_scan_direction : bool
        Are all the output channels read in the same direction?
        By default fast-scan readout direction is ``[-->,<--,-->,<--]``
        If ``same_scan_direction``, then all ``-->``
    reverse_scan_direction : bool
        If ``reverse_scan_direction``, then ``[<--,-->,<--,-->]`` or all ``<--``
    """

    # Need copy, otherwise will overwrite input data 
    if not in_place:
        im = im.copy()

    # Image cube shape
    sh = im.shape
    ndim = len(sh)
    if ndim==2:
        ny, nx = sh
        nz = 1
    else:
        nz, ny, nx = sh
    chsize = int(nx / nchans)
    im = im.reshape([nz,ny,nchans,-1])

    # FFT of kernel
    if kfft is None:
        k_big = pad_or_cut_to_size(kernel, (ny,chsize))
        kfft = np.fft.fft2(k_big)

    # Channel-by-channel deconvolution
    for ch in trange(nchans, leave=False, desc='PPC Amps'):
        sub = im[:,:,ch,:]
        if same_scan_direction:
            flip = True if reverse_scan_direction else False
        elif np.mod(ch,2)==0:
            flip = True if reverse_scan_direction else False
        else:
            flip = False if reverse_scan_direction else True

        
        if flip:  # Orient to left->right readout direction
            sub = sub[:,:,::-1]
        # Call IPC function
        sub = ipc_deconvolve(sub, kernel, kfft=kfft)
        if flip:  # Orient back
            sub = sub[:,:,::-1]
        im[:,:,ch,:] = sub

    im = im.reshape(sh)

    return im


def get_ipc_kernel(imdark, tint=None, boxsize=5, nchans=4, bg_remove=True,
                   hotcut=[5000,50000], calc_ppc=False,
                   same_scan_direction=False, reverse_scan_direction=False,
                   ref_info=[4,4,4,4], suppress_error_msg=False):
    """ Derive IPC/PPC Convolution Kernels
    
    Find the IPC and PPC kernels used to convolve detector pixel data.
    Finds all hot pixels within hotcut parameters and measures the
    average relative flux within adjacent pixels.

    Parameters
    ==========
    imdark : ndarray
        Image to search for hot pixels in units of DN or DN/sec. 
        If in terms of DN/sec, make sure to set `tint` to convert to raw DN.

    Keyword Parameters
    ==================
    tint : float or None
        Integration time to convert dark current rate into raw pixel values (DN).
        If None, then input image is assumed to be in units of DN.
    boxsize : int
        Size of the box. Should be odd. If even, will increment by 1.
    nchans : int
        Number of amplifier channels; necessary for PPC measurements. 
    bg_remove : bool
        Remove the average dark current values for each hot pixel cut-out.
        Only works if boxsize>3.
    hotcut : array-like
        Min and max values of hot pixels (above bg and bias) to consider.
    calc_ppc : bool
        Calculate and return post-pixel coupling?
    same_scan_direction : bool
        Are all the output channels read in the same direction?
        By default fast-scan readout direction is ``[-->,<--,-->,<--]``
        If ``same_scan_direction``, then all ``-->``
    reverse_scan_direction : bool
        If ``reverse_scan_direction``, then ``[<--,-->,<--,-->]`` or all ``<--``

    """
    
    ny, nx = imdark.shape
    chsize = int(nx / nchans)

    imtemp = imdark.copy() if tint is None else imdark * tint

    boxhalf = int(boxsize/2)
    boxsize = int(2*boxhalf + 1)
    distmin = np.ceil(np.sqrt(2.0) * boxhalf)

    pixmask = ((imtemp>hotcut[0]) & (imtemp<hotcut[1]))
    # Get rid of pixels around border
    lower, upper, left, right = ref_info
    pixmask[0:lower+boxhalf, :] = False
    pixmask[-upper-boxhalf:, :] = False
    pixmask[:, 0:left+boxhalf] = False
    pixmask[:, -right-boxhalf:] = False

    # Ignore borders between amplifiers
    for ch in range(1, nchans):
        x1 = ch*chsize - boxhalf
        x2 = x1 + 2*boxhalf
        pixmask[:, x1:x2] = False
    indy, indx = np.where(pixmask)
    nhot = len(indy)
    if nhot < 2:
        if not suppress_error_msg:
            _log.warning("No hot pixels found!")
        return None

    # Only want isolated pixels
    # Get distances for every pixel
    # If too close, then set equal to 0
    for i in range(nhot):
        d = np.sqrt((indx-indx[i])**2 + (indy-indy[i])**2)
        ind_close = np.where((d>0) & (d<distmin))[0]
        if len(ind_close)>0: pixmask[indy[i], indx[i]] = 0
    indy, indx = np.where(pixmask)
    nhot = len(indy)
    if nhot < 2:
        if not suppress_error_msg:
            _log.warning("No hot pixels found!")
        return None
    else:
        _log.info(f'Number of hot pixels: {nhot}')

    # Stack all hot pixels in a cube
    hot_all = []
    for iy, ix in zip(indy, indx):
        x1, y1 = np.array([ix,iy]) - boxhalf
        x2, y2 = np.array([x1,y1]) + boxsize
        sub = imtemp[y1:y2, x1:x2]

        # Flip channels along x-axis for PPC
        if calc_ppc:
            # Check if an even or odd channel (index 0)
            for ch in np.arange(0,nchans,2):
                even = True if (ix > ch*chsize) and (ix < (ch+1)*chsize-1) else False
        
            if same_scan_direction:
                flip = True if reverse_scan_direction else False
            elif even:
                flip = True if reverse_scan_direction else False
            else:
                flip = False if reverse_scan_direction else True

            if flip: sub = sub[:,::-1]

        hot_all.append(sub)
    hot_all = np.array(hot_all)

    # Remove average dark current values
    if boxsize>3 and bg_remove==True:
        for im in hot_all:
            im -= np.median([im[0,:], im[:,0], im[-1,:], im[:,-1]])

    # Normalize by sum in 3x3 region
    norm_all = hot_all.copy()
    for im in norm_all:
        im /= im[boxhalf-1:boxhalf+2, boxhalf-1:boxhalf+2].sum()

    # Take average of normalized stack
    ipc_im_avg = np.median(norm_all, axis=0)
    # ipc_im_sig = robust.medabsdev(norm_all, axis=0)

    corner_val = (ipc_im_avg[boxhalf-1,boxhalf-1] + 
                 ipc_im_avg[boxhalf+1,boxhalf+1] + 
                 ipc_im_avg[boxhalf+1,boxhalf-1] + 
                 ipc_im_avg[boxhalf-1,boxhalf+1]) / 4
    if corner_val<0: corner_val = 0

    # Determine post-pixel coupling value?
    if calc_ppc:
        ipc_val = (ipc_im_avg[boxhalf-1,boxhalf] + \
                  ipc_im_avg[boxhalf,boxhalf-1] + \
                  ipc_im_avg[boxhalf+1,boxhalf]) / 3
        if ipc_val<0: ipc_val = 0
            
        ppc_val = ipc_im_avg[boxhalf,boxhalf+1] - ipc_val
        if ppc_val<0: ppc_val = 0

        k_ipc = np.array([[corner_val, ipc_val, corner_val],
                         [ipc_val, 1-4*ipc_val, ipc_val],
                         [corner_val, ipc_val, corner_val]])
        k_ppc = np.zeros([3,3])
        k_ppc[1,1] = 1 - ppc_val
        k_ppc[1,2] = ppc_val
        
        return (k_ipc / k_ipc.sum(), k_ppc / k_ppc.sum())
        
    # Just determine IPC
    else:
        ipc_val = (ipc_im_avg[boxhalf-1,boxhalf] + 
                  ipc_im_avg[boxhalf,boxhalf-1] + 
                  ipc_im_avg[boxhalf,boxhalf+1] + 
                  ipc_im_avg[boxhalf+1,boxhalf]) / 4
        if ipc_val<0: ipc_val = 0

        kernel = np.array([[corner_val, ipc_val, corner_val],
                           [ipc_val, 1-4*ipc_val, ipc_val],
                           [corner_val, ipc_val, corner_val]])
        
        return kernel / kernel.sum()

def plot_kernel(kern, ax=None, return_figax=False):
    """ Plot image of IPC or PPC kernel

    Parameters
    ----------
    kern : ndarray
        Kernel image (3x3 or 5x5, etc) to plot.
    ax : axes
        Axes to plot kernel on. If None, will create new
        figure and axes subplot.
    return_figax : bool
        Return the (figure, axes) for user manipulations?
    """

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(5,5))
    else:
        fig = None

    # Convert to log scale for better contrast between pixels
    kern = kern.copy()
    kern[kern==0] = 1e-7
    ny, nx = kern.shape
    extent = np.array([-nx/2,nx/2,-ny/2,ny/2]) 
    ax.imshow(np.log(kern), extent=extent, vmax=np.log(1), vmin=np.log(1e-5))

    # Add text to each pixel position
    for i in range(ny):
        ii = i + int(-ny/2)
        for j in range(nx):
            jj = j + int(-nx/2)
            if (ii==0) and (jj==0): # Different text format at center position
                ax.text(jj,ii, '{:.2f}%'.format(kern[i,j]*100), color='black', fontsize=16,
                    horizontalalignment='center', verticalalignment='center')
            else:
                ax.text(jj,ii, '{:.3f}%'.format(kern[i,j]*100), color='white', fontsize=16,
                    horizontalalignment='center', verticalalignment='center')


    ax.tick_params(axis='both', color='white', which='both')
    for k in ax.spines.keys():
        ax.spines[k].set_color('white')

    if fig is not None:
        ax.set_title('IPC Kernel', fontsize=16)
        fig.tight_layout()

    if return_figax:
        return fig, ax


#######################################
# Power spectrum information
#######################################

def pow_spec_ramp(data, nchan, nroh=0, nfoh=0, nframes=1, expand_npix=False,
                  same_scan_direction=False, reverse_scan_direction=False,
                  mn_func=np.mean, return_freq=False, dt=1, **kwargs):
    """ Get power spectrum within frames of input ramp
    
    Takes an input cube, splits it into output channels, and
    finds the power spectrum of each frame. Then, calculate 
    the average power spectrum for each channel.
    
    Use `nroh` and `nfoh` to expand the frame size to encapsulate
    the row and frame overheads not included in the science data.
    These just zero-pad the array.
    
    Parameters
    ==========
    data : ndarray
        Input Image cube.
    nchan : int
        Number of amplifier channels.
    nroh : int
        Number of pixel overheads per row.
    nfoh : int
        Number of row overheads per frame.
    nframes : int
        Number of frames to use to calculate an power spectrum.
        Normally we just use 1 frame time
    expand_npix : bool
        Should we zero-pad the array to a power of two factor
        for incresed speed?
    same_scan_direction : bool
        Are all the output channels read in the same direction?
        By default fast-scan readout direction is ``[-->,<--,-->,<--]``
        If ``same_scan_direction``, then all ``-->``
    reverse_scan_direction : bool
        If ``reverse_scan_direction``, then ``[<--,-->,<--,-->]`` or all ``<--``
    """
    
    nz, ny, nx = data.shape
    chsize = int(nx / nchan)

    # Channel size and ny plus pixel and row overheads
    ch_poh = chsize + nroh
    ny_poh = ny + nfoh
    
    ps_data = [] # Hold channel data
    for ch in range(nchan):
        # Array of pixel values
        if (nroh>0) or (nfoh>0):
            sig = np.zeros([nz,ny_poh,ch_poh])
            sig[:,0:ny,0:chsize] += data[:,:,ch*chsize:(ch+1)*chsize]
        else:
            sig = data[:,:,ch*chsize:(ch+1)*chsize]
            
        # Flip x-axis for odd channels
        if same_scan_direction:
            flip = True if reverse_scan_direction else False
        elif np.mod(ch,2)==0:
            flip = True if reverse_scan_direction else False
        else:
            flip = False if reverse_scan_direction else True
        sig = sig[:,:,::-1] if flip else sig
        
        if nframes==1:
        
            sig = sig.reshape([sig.shape[0],-1])
            npix = sig.shape[1]

            # Pad nsteps to a power of 2, which can be faster
            npix2 = int(2**np.ceil(np.log2(npix))) if expand_npix else npix

            # Power spectrum of each frame
            ps = np.abs(np.fft.rfft(sig, n=npix2))**2 / npix2
            
        else:
            
            sh = sig.shape
            npix = nframes * sh[-2] * sh[-1]
            # Pad nsteps to a power of 2, which can be faster
            npix2 = int(2**np.ceil(np.log2(npix))) if expand_npix else npix
            
            # Power spectrum for each set of frames
            niter = nz - nframes + 1
            ps = []
            for i in range(niter):
                sig2 = sig[i:i+nframes].ravel()

                # Power spectrum
                ps.append(np.abs(np.fft.rfft(sig2, n=npix2))**2 / npix2)
            ps = np.array(ps)
                
        # Average of all power spectra
        ps_data.append(mn_func(ps, axis=0))

    # Power spectrum of each output channel
    ps_data = np.array(ps_data)
    
    if return_freq:
        freq = get_freq_array(ps_data, dt=dt)
        return ps_data, freq
    else:
        return ps_data 


def pow_spec_ramp_pix(data, nchan, expand_nstep=False,
                      mn_func=np.mean, return_freq=False, dt=1, **kwargs):
    """ Get power spectrum of pixels within ramp
    
    Takes an input cube, splits it into output channels, and
    finds the power spectrum of each pixel. Return the average 
    power spectrum for each channel.
    
    Parameters
    ==========
    data : ndarray
        Input Image cube.
    nchan : int
        Number of amplifier channels.
    expand_nstep : bool
        Should we zero-pad the array to a power of two factor
        for incresed speed?
    """
    
    nz, ny, nx = data.shape
    chsize = int(nx / nchan)
    
    ps_data = [] # Hold channel data
    for ch in range(nchan):
        # Array of pixel values
        sig = data[:,:,ch*chsize:(ch+1)*chsize]

        sig = sig.reshape([sig.shape[0],-1])
        nstep = sig.shape[0]

        # Pad nsteps to a power of 2, which can be faster
        nstep2 = int(2**np.ceil(np.log2(nstep))) if expand_nstep else nstep

        # Power spectrum of each pixel
        ps = np.abs(np.fft.rfft(sig, n=nstep2, axis=0))**2 / nstep2

        # Average of all power spectra
        ps_data.append(mn_func(ps, axis=1))

    # Power spectrum of each output channel
    ps_data = np.array(ps_data)
    
    if return_freq:
        freq = get_freq_array(ps_data, dt=dt)
        return ps_data, freq
    else:
        return ps_data 


def fit_corr_powspec(freq, ps, flim1=[0,1], flim2=[10,100], alpha=-1, **kwargs):
    """ Fit Correlated Noise Power Spectrum

    Fit the scaling factors of the 1/f power law components
    observed in the correlated noise power spectra. This
    function separately calculates the high-freq and low-
    freq scale factor components defined by the fcut params.
    The mid-frequency ranges are interpolated in log space.

    Parameters
    ==========
    freq : ndarray
        Input frequencies corresponding to power spectrum.
    ps : ndarray
        Input power spectrum to fit.
    flim1 : float
        Fit frequencies within this range to get scaling
        for low frequency 1/f noise.
    flim2 : float
        Fit frequencies within this range to get scaling
        for high frequency 1/f noise.
    alpha : float
        Noise power spectrum scaling
    """
    
    yf = freq**alpha
    yf[0] = 0
    
    # Low frequency fit
    ind = (freq >= flim1[0]) & (freq <= flim1[1]) & (yf > 0)
    scl1 = np.median(ps[ind] / yf[ind])

    # High frequency fit
    ind = (freq >= flim2[0]) & (freq <= flim2[1]) & (yf > 0)
    scl2 = np.median(ps[ind] / yf[ind])

    return np.array([scl1, scl2])

def broken_pink_powspec(freq, scales, fcut1=1, fcut2=10, alpha=-1, **kwargs):

    scl1, scl2 = scales
    yf = freq**alpha
    yf[0] = 0

    # Output array
    res = np.zeros(len(yf))

    # Low frequency component
    ind = (freq <= fcut1)
    res[ind] = scl1*yf[ind]

    # High frequency componet
    ind = (freq >= fcut2)
    res[ind] = scl2*yf[ind]

    # Mid frequency interpolation, log space
    ind = (freq > fcut1) & (freq < fcut2)
    xlog = np.log10(freq)
    ylog = np.log10(res)
    ylog[ind] = np.interp(xlog[ind], xlog[~ind], ylog[~ind])
    res[ind] = 10**ylog[ind]

    return res

def get_power_spec(data, nchan=4, calc_cds=True, kw_powspec=None, per_pixel=False,
                   return_corr=False, return_ucorr=False, mn_func=np.mean):
    """
    Calculate the power spectrum of an input data ramp in a variety of ways.

    If return_corr and return_ucorr are both False, then will return (ps_all, None, None).

    Parameters
    ==========
    calc_cds : bool
        Power spectrum of CDS pairs or individual frames?
    per_pixel : bool
        Calculate average power spectrum of each pixel along ramp (frame timescales)?
        If False, samples pixels within a frame (pixel read timescales)
    return_corr : bool
        Return power spectrum of channel correlated 1/f noise?
    return_ucorr : bool
        Return power spectra of channel-dependent (uncorrelated) 1/f noise?
    kw_powspec : dict
        Keyword arguments to pass to `pow_spec_ramp` function.
    mn_func : func
        Function to use to perform averaging of individual power spectra.
    """

    nz, ny, nx = data.shape
    chsize = int(nx/nchan)

    # CDS or just subtract first frame
    if calc_cds:
        cds = data[1::2] - data[0::2]
    else:
        cds = data[1:] - data[0]

    # Remove averages from each frame
    cds_mn = np.median(cds.reshape([cds.shape[0], -1]), axis=1)
    cds -= cds_mn.reshape([-1,1,1])

    # Remove averages from each pixel
    cds_mn = np.median(cds, axis=0)
    cds -= cds_mn

    # Keywords for power spectrum
    # Only used for pow_spec_ramp, not pow_spec_ramp_pix
    if kw_powspec is None:
        kw_powspec = {
            'nroh': 0, 'nfoh': 0, 'nframes': 1,
            'same_scan_direction': False, 'reverse_scan_direction': False
        }
        same_scan_direction = kw_powspec['same_scan_direction']

    # Power spectrum of all frames data
    if per_pixel:
        ps_full = pow_spec_ramp_pix(cds, nchan, mn_func=mn_func)
    else:
        ps_full = pow_spec_ramp(cds, nchan, mn_func=mn_func, **kw_powspec)

    # Extract 1/f noise from data
    ps_corr, ps_ucorr = (None, None)
    if return_ucorr or return_corr:
        fn_data = []
        for im in cds:
            ch_arr = im.reshape([ny,-1,chsize]).transpose([1,0,2])
            mask = np.abs(im - np.median(im)) > 10*robust.medabsdev(im)
            mask = mask.reshape([ny,-1,chsize]).transpose([1,0,2])
            fnoise = channel_smooth_savgol(ch_arr, mask=mask)
            fnoise = fnoise.transpose([1,0,2]).reshape([ny,nx])
            fn_data.append(fnoise)
        fn_data = np.array(fn_data)

        # Delete data and cds arrays to free up memory
        del cds

        # Split into correlated and uncorrelated components
        fn_data_corr = []
        for j, im in enumerate(fn_data):
            # Extract correlated 1/f noise data        
            fn_corr = channel_averaging(im, nchans=nchan,  off_chans=False,
                same_scan_direction=same_scan_direction, mn_func=np.mean)
            # Subtract correlated noise from fn_data
            if return_ucorr: 
                fn_data[j] -= fn_corr
            # Only append first channel since the rest are the same data
            fn_data_corr.append(fn_corr[:,0:chsize])
        fn_data_corr = np.array(fn_data_corr)
        
        # Power spectrum of uncorrelated 1/f noise
        if return_ucorr:
            if per_pixel:
                ps_ucorr = pow_spec_ramp_pix(fn_data, nchan, mn_func=mn_func) 
            else:
                ps_ucorr = pow_spec_ramp(fn_data, nchan, mn_func=mn_func, **kw_powspec)
        del fn_data

        # Power spectrum of correlated 1/f noise
        if return_corr:
            if per_pixel:
                ps_corr = pow_spec_ramp_pix(fn_data_corr, 1, mn_func=mn_func)
            else:
                ps_corr = pow_spec_ramp(fn_data_corr, 1, mn_func=mn_func, **kw_powspec)
        del fn_data_corr

    return ps_full, ps_ucorr, ps_corr



def get_power_spec_all(allfiles, super_bias=None, det=None, DMS=False, include_oh=False, 
                       same_scan_direction=False, reverse_scan_direction=False,
                       calc_cds=True, return_corr=False, return_ucorr=False, 
                       per_pixel=False, mn_func=np.mean, kw_reffix=None):
    
    """
    Return the average power spectra (white, 1/f noise correlated and uncorrelated) of
    all FITS files. 

    Parameters
    ==========
    allfiles : array-like
        List of FITS files to operate on.
    super_bias : ndarray
        Option to subtract a super bias image from all frames in a ramp.
        Provides slightly better statistical averaging for reference pixel
        correction routines.
    det : Detector class
        Option to pass known NIRCam detector class. This will get generated 
        from a FITS header if not specified.
    DMS : bool
        Are the files DMS formatted or FITSWriter?
    include_oh : bool
        Zero-pad the data to insert line and frame overhead pixels? 
    same_scan_direction : bool
        Are all the output channels read in the same direction?
        By default fast-scan readout direction is ``[-->,<--,-->,<--]``
        If ``same_scan_direction``, then all ``-->``
    reverse_scan_direction : bool
        If ``reverse_scan_direction``, then ``[<--,-->,<--,-->]`` or all ``<--``
    calc_cds : bool
        Power spectrum of CDS pairs or individual frames?
    per_pixel : bool
        Calculate average power spectrum of each pixel along ramp (frame timescales)?
        If False, samples pixels within a frame (pixel read timescales).
    return_corr : bool
        Return power spectrum of channel correlated 1/f noise?
    return_ucorr : bool
        Return power spectra of channel-dependent (uncorrelated) 1/f noise?
    kw_powspec : dict
        Keyword arguments to pass to `pow_spec_ramp` function.
    mn_func : func
        Function to use to perform averaging of individual power spectra.
    """


    # Set logging to WARNING to suppress messages
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)

    if super_bias is None:
        super_bias = 0
            
    # Header info from first file
    if det is None:
        hdr = fits.getheader(allfiles[0])
        det = create_detops(hdr, DMS=DMS)

    # Overhead information
    nchan = det.nout

    # Row and frame overheads
    if include_oh:
        nroh = det._line_overhead
        nfoh = det._extra_lines
    else:
        nroh = nfoh = 0

    # Keywords for reffix
    if kw_reffix is None:
        kw_reffix = {
            'nchans': nchan, 'altcol': True, 'in_place': True,
            'fixcol': False, 'avg_type': 'pixel', 'savgol': True, 'perint': False
        }

    # Keywords for power spectrum
    kw_powspec = {
        'nroh': nroh, 'nfoh': nfoh, 'nframes': 1,
        'same_scan_direction': same_scan_direction,
        'reverse_scan_direction': reverse_scan_direction
    }

    pow_spec_all = []
    if return_corr: pow_spec_corr = []
    if return_ucorr: pow_spec_ucorr = []

    for fname in tqdm(allfiles, desc='Files'):

        # If DMS, then might be multiple integrations per FITS file
        nint = fits.getheader(fname)['NINTS'] if DMS else 1

        iter_range = trange(nint, desc='Ramps', leave=False) if nint>1 else range(nint)
        for i in iter_range:
            data = get_fits_data(fname, bias=super_bias, reffix=True, 
                                DMS=DMS, int_ind=i, **kw_reffix)

            ps_full, ps_ucorr, ps_corr = get_power_spec(data, nchan=nchan, 
                calc_cds=calc_cds, return_corr=return_corr, return_ucorr=return_ucorr, 
                per_pixel=per_pixel, mn_func=mn_func, kw_powspec=kw_powspec)

            pow_spec_all.append(ps_full)
            if return_corr: 
                pow_spec_corr.append(ps_corr)
            if return_ucorr: 
                pow_spec_ucorr.append(ps_ucorr)
            
            del data

    # Full spectra
    pow_spec_all = np.array(pow_spec_all)
    ps_all = np.mean(pow_spec_all, axis=0)
    # Correlated Noise
    if return_corr: 
        pow_spec_corr = np.array(pow_spec_corr)
        ps_corr = np.mean(pow_spec_corr, axis=0).squeeze()
    else:
        ps_corr = None

    # Uncorrelated Noise per amplifier channel
    if return_ucorr:
        pow_spec_ucorr = np.array(pow_spec_ucorr)
        ps_ucorr = np.mean(pow_spec_all, axis=0)
    else:
        ps_ucorr = None

    # Set back to previous logging level
    setup_logging(log_prev, verbose=False)

    return ps_all, ps_corr, ps_ucorr

def get_freq_array(pow_spec, dt=1, nozero=False, npix_odd=False):
    """ Return frequencies associated with power spectrum
    
    Parameters
    ==========
    pow_spec : ndarray
        Power spectrum to obtain associated frequency array.
    dt : float
        Delta time between corresponding elements in time domain.
    nozero : bool
        Set freq[0] = freq[1] to remove zeros? This is mainly so
        we don't obtain NaN's later when calculating 1/f noise.
    npix_odd : bool
        We normally assume that the original time-domain data
        was comprised of an even number of pixels. However, if it
        were actually odd, the frequency array will be slightly
        shifted. Set this to True if the intrinsic data that was
        used to generate the pow_spec had an odd number of elements.
    """

    # This assumes an even input array
    npix = 2 * (pow_spec.shape[-1] - 1)
    # Off by 1 if initial npix was odd
    if npix_odd:
        npix += 1
    freq = np.fft.rfftfreq(npix, d=dt)

    if nozero:
        # First element should not be 0
        freq[0] = freq[1]

    return freq


#######################################
# Linearity and Gain
#######################################

# Determine saturation level in ADU (relative to bias)
def find_sat(data, bias=None, ref_info=[4,4,4,4], bit_depth=16):
    """
    Given a data cube, find the values in ADU in which data
    reaches hard saturation.
    """

    # Maximum possible value corresponds to bit depth
    sat_max = 2**bit_depth-1
    sat_min = 0

    # Subtract bias?
    nz, ny, nx = data.shape
    imarr = data if bias is None else data - bias

    # Data can be characterized as large differences at start,
    # followed by decline and then difference of 0 at hard saturation

    # Determine difference between samples
    diff_arr = imarr[1:] - imarr[0:-1]

    # Select pixels to determine individual saturation values
    diff_max = np.median(diff_arr[0]) / 10
    diff_min = 100

    # Ensure a high rate at the beginning and a flat rate at the end
    sat_mask = (diff_arr[0]>diff_max) & (np.abs(diff_arr[-1]) < diff_min)

    # Median value to use for pixels that didn't reach saturation
    # sat_med = np.median(imarr[-1, sat_mask])
    
    # Initialize saturation array with median
    # sat_arr = np.ones([ny,nx]) * sat_med

    # Initialize saturation as max-min
    sat_arr = imarr[-1] - imarr[0]
    sat_arr[sat_mask] = imarr[-1, sat_mask]

    # Bound between 0 and bit depth
    sat_arr[sat_arr<sat_min] = sat_min
    sat_arr[sat_arr>sat_max] = sat_max

    # Reference pixels don't saturate
    # [bottom, upper, left, right]
    br, ur, lr, rr = ref_info
    ref_mask = np.zeros([ny,nx], dtype=bool)
    if br>0: ref_mask[0:br,:] = True
    if ur>0: ref_mask[-ur:,:] = True
    if lr>0: ref_mask[:,0:lr] = True
    if rr>0: ref_mask[:,-rr:] = True
    sat_arr[ref_mask] = sat_max
    
    return sat_arr

# Fit unsaturated data and return coefficients
def cube_fit(tarr, data, bias=None, sat_vals=None, sat_frac=0.95, 
             deg=1, fit_zero=False, verbose=False, ref_info=[4,4,4,4],
             use_legendre=False, lxmap=None, return_lxmap=False,
             return_chired=False):
        
    nz, ny, nx = data.shape
    
    # Subtract bias?
    imarr = data if bias is None else data - bias
    
    # Get saturation levels
    if sat_vals is None:
        sat_vals = find_sat(imarr, ref_info=ref_info)
        
    # Array of masked pixels (saturated)
    mask_good = imarr < sat_frac*sat_vals
    
    # Reshape for all pixels in single dimension
    imarr = imarr.reshape([nz, -1])
    mask_good = mask_good.reshape([nz, -1])

    # Initial 
    cf = np.zeros([deg+1, nx*ny])
    if return_lxmap:
        lx_min = np.zeros([nx*ny])
        lx_max = np.zeros([nx*ny])
    if return_chired:
        chired = np.zeros([nx*ny])

    # For each 
    npix_sum = 0
    i0 = 0 if fit_zero else 1
    for i in np.arange(i0,nz)[::-1]:
        ind = (cf[1] == 0) & (mask_good[i])
        npix = np.sum(ind)
        npix_sum += npix
        
        if verbose:
            print(i+1,npix,npix_sum, 'Remaining: {}'.format(nx*ny-npix_sum))
            
        if npix>0:
            if fit_zero:
                x = np.concatenate(([0], tarr[0:i+1]))
                y = np.concatenate((np.zeros([1, np.sum(ind)]), imarr[0:i+1,ind]), axis=0)
            else:
                x, y = (tarr[0:i+1], imarr[0:i+1,ind])

            if return_lxmap:
                lx_min[ind] = np.min(x) if lxmap is None else lxmap[0]
                lx_max[ind] = np.max(x) if lxmap is None else lxmap[1]
                
            # Fit line if too few points relative to polynomial degree
            if len(x) <= deg+1:
                cf[0:2,ind] = jl_poly_fit(x,y, deg=1, use_legendre=use_legendre, lxmap=lxmap)
            else:
                cf[:,ind] = jl_poly_fit(x,y, deg=deg, use_legendre=use_legendre, lxmap=lxmap)

            # Get reduced chi-sqr metric for poorly fit data
            if return_chired:
                yfit = jl_poly(x, cf[:,ind])
                deg_chi = 1 if len(x)<=deg+1 else deg
                dof = y.shape[0] - deg_chi
                chired[ind] = chisqr_red(y, yfit=yfit, dof=dof)

    imarr = imarr.reshape([nz,ny,nx])
    mask_good = mask_good.reshape([nz,ny,nx])
    
    cf = cf.reshape([deg+1,ny,nx])
    if return_lxmap:
        lxmap_arr = np.array([lx_min, lx_max]).reshape([2,ny,nx])
        if return_chired:
            chired = chired.reshape([ny,nx])
            return cf, lxmap_arr, chired
        else:
            return cf, lxmap_arr
    else:
        if return_chired:
            chired = chired.reshape([ny,nx])
            return cf, chired
        else:
            return cf


def time_to_sat(data, sat_vals, dt=1, sat_calc=0.998, ref_info=[4,4,4,4]):
    """ Determine time of saturation"""

    nz, ny, nx = data.shape

    # Active and reference pixel masks
    lower, upper, left, right = ref_info
    mask_ref = np.zeros([ny,nx], dtype='bool')
    if lower>0: mask_ref[0:lower,:] = True
    if upper>0: mask_ref[-upper:,:] = True
    if left>0:  mask_ref[:,0:left]  = True
    if right>0: mask_ref[:,-right:] = True

    # Time array
    tarr = np.arange(1,nz+1) * dt
    
    pvals = data
    svals = sat_vals

    # Find time where data reaches 99% of saturation
    mask99 = pvals < sat_calc*sat_vals
    
    # Linear interpolate to find time we reach full well
    pvals1 = np.max(pvals * mask99, axis=0)
    pvals2 = np.max(pvals * np.roll(mask99,1,axis=0), axis=0)

    tvals1 = np.max(tarr.reshape([-1,1,1]) * mask99, axis=0)
    tvals2 = np.max(tarr.reshape([-1,1,1]) * np.roll(mask99,1,axis=0), axis=0)

    # Time at which we reach 100% saturation
    tfin = tvals1 + (tvals2 - tvals1) * (sat_calc - pvals1 / svals) / ((pvals2 - pvals1) / svals)
    del pvals1, pvals2, tvals1, tvals2

    tfin[mask_ref] = 0
    tfin[~np.isfinite(tfin)] = 0

    return tfin

def find_group_sat(file, DMS=False, bias=None, sat_vals=None, sat_calc=0.998):
    """Group at which 98% of pixels are saturated"""

    # Set logging to WARNING to suppress messages
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)

    hdr = fits.getheader(file)
    det = create_detops(hdr, DMS=DMS)

    setup_logging(log_prev, verbose=False)

    nz, ny, nx = (det.multiaccum.ngroup, det.ypix, det.xpix)
    nchan = det.nout
    
    # Active and reference pixel masks
    # Masks for active, reference, and amplifiers
    mask_ref = det.mask_ref
    mask_act = ~mask_ref

    # Read in data
    kwargs_ref = {'nchans': nchan, 'in_place': True, 'altcol': True, 'fixcol': False}
    data = get_fits_data(file, DMS=DMS, bias=bias, reffix=True, **kwargs_ref)
    
    if sat_vals is None:
        sat_vals = find_sat(data, ref_info=det.ref_info)

    # Get saturation times for each 
    tsat = time_to_sat(data, sat_vals, dt=1, sat_calc=sat_calc, ref_info=det.ref_info)
    
    vals = tsat[mask_act]
    bins = np.arange(vals.min(), vals.max()+1, 1)
    ig, vg, cv = hist_indices(vals, bins=bins, return_more=True)
    nvals = np.array([len(i) for i in ig])
    nsum = np.cumsum(nvals) / nvals.sum()
    
    # Index containing 98% of pixels 
    imax = np.min(np.where(nsum>0.98)[0])
    
    return imax

def calc_nonlin_coeff(data, sat_vals, well_depth, sat_calc=0.98, ref_info=[4,4,4,4],
                      counts_cut=None, deg=8, use_legendre=True, lxmap=[0,1e5], **kwargs):
    
    """
    
    counts_cut : None or float
        Option to fit two sets of polynomial coefficients to lower and uppper
        values. 'counts_cut' specifies the division in values of electrons.
        Useful for pixels with different non-linear behavior at low flux levels.
        Recommended values of 15000 e-.
    """

    nz, ny, nx = data.shape

    # Time array
    tarr = np.arange(1,nz+1)
    
    # Active and reference pixel masks
    lower, upper, left, right = ref_info
    mask_ref = np.zeros([ny,nx], dtype='bool')
    if lower>0: mask_ref[0:lower,:] = True
    if upper>0: mask_ref[-upper:,:] = True
    if left>0:  mask_ref[:,0:left] = True
    if right>0: mask_ref[:,-right:] = True
    
    # Find time where data reaches 99% of saturation
    mask99 = data < sat_calc*sat_vals
    tfin = time_to_sat(data, sat_vals, sat_calc=0.998, ref_info=ref_info)

    # Get rid of 0s and NaN's
    ind_bad = (np.isnan(tfin)) | (tfin==0)
    tfin[ind_bad] = np.median(tfin[~ind_bad])
    
    # Create ideal pixel ramps in e-
    ramp = well_depth * tarr.reshape([-1,1,1]) / tfin.reshape([1,ny,nx])
    ramp[ramp>well_depth] = well_depth
    
    # Simultaneously fit pixels that have the same ideal ramps 
    bsize = 0.05
    bins = np.arange(tfin.min(), tfin.max()+bsize, bsize)
    ig, vg, cv = hist_indices(tfin, bins=bins, return_more=True)

    # Select only indices with len>0
    nvals = np.array([len(i) for i in ig])
    ig_nozero = np.array(ig)[nvals>0]
    
    # Reshape to put all pixels in single dimension
    data_flat   = data.reshape([data.shape[0], -1])
    ramp_flat   = ramp.reshape([ramp.shape[0], -1])

    mask100 = mask99 #data < sat_calc
    mask100_flat = mask100.reshape([mask100.shape[0], -1])

    if counts_cut is None:
        cf_arr = np.zeros([deg+1,nx*ny])
    else:
        cf_arr1 = np.zeros([deg+1,nx*ny])
        cf_arr2 = np.zeros([deg+1,nx*ny])

    for ii in trange(len(ig_nozero), leave=False, desc='Linearity Fitting'):
        ig_sub = ig_nozero[ii]

        # Grab values less than well depth
        ind = mask100_flat[:,ig_sub[0]]
        indz = np.where(ind==False)[0]
        if len(indz)>0:
            ind[indz[0]] = True  # Set next element true
        pix_dn = data_flat[:,ig_sub][ind] # DN Values
        pix_el = ramp_flat[:,ig_sub][ind]  # electron values
        pix_el_mn = np.mean(pix_el, axis=1)
        # Gain function
        gain = pix_el_mn.reshape([-1,1]) / pix_dn

        if counts_cut is None:
            cf_arr[:,ig_sub] = jl_poly_fit(pix_el_mn, gain, deg=deg, 
                                           use_legendre=use_legendre, lxmap=lxmap)
        else:
            # Fit high pixel values
            ifit1 = (pix_el_mn >= counts_cut)
            if ifit1.sum() > 0:
                cf_arr1[:,ig_sub] = jl_poly_fit(pix_el_mn[ifit1], gain[ifit1], deg=deg, 
                                                use_legendre=use_legendre, lxmap=lxmap)

            # Fit low pixel values
            ifit2 = ~ifit1
            if ifit2.sum() > 0:
                cf_arr2[:,ig_sub] = jl_poly_fit(pix_el_mn[ifit2], gain[ifit2], deg=deg, 
                                                use_legendre=use_legendre, lxmap=lxmap)

    # Reshape and set reference masks to 0
    if counts_cut is None:
        cf_arr = cf_arr.reshape([deg+1,ny,nx])
        cf_arr[:,mask_ref] = 0
        return cf_arr
    else:
        cf_arr1 = cf_arr1.reshape([deg+1,ny,nx])
        cf_arr1[:,mask_ref] = 0
        cf_arr2 = cf_arr2.reshape([deg+1,ny,nx])
        cf_arr2[:,mask_ref] = 0
        return cf_arr1, cf_arr2


def calc_linearity_coeff(data, sat_vals, well_depth, sat_calc=0.98, ref_info=[4,4,4,4],
                         counts_cut=None, deg=8, use_legendre=True, lxmap=[0,1e5], 
                         nonlin=False, **kwargs):
    
    """
    
    counts_cut : None or float
        Option to fit two sets of polynomial coefficients to lower and uppper
        values. 'counts_cut' specifies the division in values of electrons.
        Useful for pixels with different non-linear behavior at low flux levels.
        Recommended values of 15000 e-.
    """

    if nonlin:
        return calc_nonlin_coeff(data, sat_vals, well_depth, sat_calc=sat_calc, ref_info=ref_info,
                                 counts_cut=counts_cut, deg=deg, use_legendre=use_legendre, lxmap=lxmap)

    nz, ny, nx = data.shape

    # Time array
    tarr = np.arange(1,nz+1)
    
    # Active and reference pixel masks
    lower, upper, left, right = ref_info
    mask_ref = np.zeros([ny,nx], dtype='bool')
    if lower>0: mask_ref[0:lower,:] = True
    if upper>0: mask_ref[-upper:,:] = True
    if left>0:  mask_ref[:,0:left] = True
    if right>0: mask_ref[:,-right:] = True
    
    # Find time where data reaches 99% of saturation
    mask99 = data < sat_calc*sat_vals
    tfin = time_to_sat(data, sat_vals, sat_calc=0.998, ref_info=ref_info)

    # Get rid of 0s and NaN's
    ind_bad = (np.isnan(tfin)) | (tfin==0)
    tfin[ind_bad] = np.median(tfin[~ind_bad])
    
    # Create ideal pixel ramps in e-
    ramp = well_depth * tarr.reshape([-1,1,1]) / tfin.reshape([1,ny,nx])
    ramp[ramp>well_depth] = well_depth
    
    # Reshape to put all pixels in single dimension
    data_flat = data.reshape([data.shape[0], -1])
    ramp_flat = ramp.reshape([ramp.shape[0], -1])

    gain_flat = ramp_flat / data_flat
    
    mask99_flat = mask99.reshape([mask99.shape[0], -1])
    
    if counts_cut is None:
        cf_arr = np.zeros([deg+1,nx*ny])
    else:
        cf_arr1 = np.zeros([deg+1,nx*ny])
        cf_arr2 = np.zeros([deg+1,nx*ny])
        
        
    for i in trange(nx*ny, leave=False, desc='Linearity Fitting'):
        # Grab values less than well depth
        ind = mask99_flat[:,i]
        indz = np.where(ind==False)[0]
        if len(indz)>0:
            ind[indz[0]] = True  # Set next element true
            
        pix_dn = data_flat[:,i]
        pix_e  = ramp_flat[:,i]
        gain   = gain_flat[:,i]
        
        # Linearity or non-linearity coefficients
        vals = pix_e if nonlin else pix_dn
        
        if counts_cut is None:
            cf_arr[:,i] = jl_poly_fit(vals[ind], gain[ind], deg=deg, 
                                      use_legendre=use_legendre, lxmap=lxmap)
        else:
            # Fit high pixel values
            ifit1 = (pix_dn >= counts_cut)
            if ifit1.sum() > 0:
                cf_arr1[:,i] = jl_poly_fit(vals[ifit1], gain[ifit1], deg=deg, 
                                                use_legendre=use_legendre, lxmap=lxmap)

            # Fit low pixel values
            ifit2 = ~ifit1
            if ifit2.sum() > 0:
                cf_arr2[:,i] = jl_poly_fit(vals[ifit2], gain[ifit2], deg=deg, 
                                                use_legendre=use_legendre, lxmap=lxmap)

    # Reshape and set reference masks to 0
    if counts_cut is None:
        cf_arr = cf_arr.reshape([deg+1,ny,nx])
        cf_arr[:,mask_ref] = 0
        return cf_arr
    else:
        cf_arr1 = cf_arr1.reshape([deg+1,ny,nx])
        cf_arr1[:,mask_ref] = 0
        cf_arr2 = cf_arr2.reshape([deg+1,ny,nx])
        cf_arr2[:,mask_ref] = 0
        return cf_arr1, cf_arr2


def get_linear_coeffs(allfiles, super_bias=None, DMS=False, kppc=None, kipc=None,
    counts_cut=None, deg=8, use_legendre=True, lxmap=[0,1e5], return_satvals=False, 
    nonlin=False, sat_calc=0.98, **kwargs):


    if super_bias is None:
        super_bias = 0
        
    # Set logging to WARNING to suppress messages
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)

    # Header info from first file
    hdr = fits.getheader(allfiles[0])
    det = create_detops(hdr, DMS=DMS)

    # Set back to previous logging level
    setup_logging(log_prev, verbose=False)

    # Well level in electrons
    well_depth = det.well_level

    data_mn, _ = gen_super_ramp(allfiles, super_bias=super_bias, DMS=DMS, **kwargs)

    # Update number of read frames
    # nz, ny, nx = data_mn.shape
    # det.multiaccum.ngroup = nz
    # tarr = det.times_group_avg

    # IPC and PPC kernels
    # PPC corrections
    if (kppc is not None) and (kppc[1,2]>0):
        data_mn = ppc_deconvolve(data_mn, kppc)
    # IPC correction
    if kipc is not None:
        data_mn = ipc_deconvolve(data_mn, kipc)

    # Get saturation levels
    sat_vals = find_sat(data_mn)

    # Get coefficients to obtain non-linear ramp
    res = calc_linearity_coeff(data_mn, sat_vals, well_depth, deg=deg, counts_cut=counts_cut, 
                               use_legendre=use_legendre, lxmap=lxmap, nonlin=nonlin, sat_calc=sat_calc)

    if return_satvals:
        return res, sat_vals
    else:
        return res

def pixel_linearity_gains(frame, coeff_arr, use_legendre=True, lxmap=[0,1e5]):
    """
    Given some image data and coefficients, determine effective 
    gain value to use to go from DN to electrons.
    """

    # from numpy.polynomial import legendre
    from scipy.special import eval_legendre

    ncf = coeff_arr.shape[0]
    xvals = frame.reshape([1,-1])
    if use_legendre:
        # Values to map to [-1,+1]
        if lxmap is None:
            lxmap = [np.min(xvals), np.max(xvals)]

        # Remap xvals -> lxvals
        dx = lxmap[1] - lxmap[0]
        lxvals = 2 * (xvals - (lxmap[0] + dx/2)) / dx
        xfan = np.array([eval_legendre(n, lxvals) for n in range(ncf)])
    else:
        # Create an array of exponent values
        parr = np.arange(ncf, dtype='float')
        xfan = xvals**parr.reshape([-1,1]) # Array broadcasting

    gain = np.sum(xfan.reshape([ncf,-1]) * coeff_arr.reshape([ncf,-1]), axis=0)
    return gain 

def apply_linearity(cube, det, coeff_dict):
    """Apply pixel linearity corrections to ramp

    Linearize a bias-subtracted, ref-pixel-corrected ramp and convert
    from units of DN to electrons.

    Parameters
    ----------
    cube : ndarray
        Ramp data in DN of size (nz,ny,nx). Should be bias-subtracted and
        ref-pixel-corrected. Should match det subarray shape.
    det : Detector Class
        NIRCam detector class.
    coeff_dict : ndarray
        Dictionary holding coefficient information:

            - 'cf_nonlin'    : Set of polynomial coefficients of size (ncf,ny,nx).
            - 'use_legendre' : Coefficients use Legendre polynomials?
            - 'lxmap'        : Legendre polynomial normalization range, usually [0,1e5]

        Possible to separately fit lower flux values:

             - 'counts_cut'    : Flux cut-off value in electrons
             - 'cf_nonlin_low' : Coefficients for flux values below counts_cut

    """

    nz, _, _ = cube.shape
    nx, ny = (det.xpix, det.ypix)

    # Need to crop input coefficients in the event of subarrays
    x1, x2 = (det.x0, det.x0 + nx)
    y1, y2 = (det.y0, det.y0 + ny)

    if cube.shape[-2]!=ny or cube.shape[-1]!=nx:
        # Assume full frame cube needs to be cropped
        cube = cube[:,y1:y2,x1:x2]

    # Nominal coefficient array
    cf_arr         = coeff_dict.get('cf_nonlin')[:,y1:y2,x1:x2]
    use_legendre   = coeff_dict.get('use_legendre', False)
    lxmap          = coeff_dict.get('lxmap')

    # Information for lower flux values
    counts_cut = coeff_dict.get('counts_cut')
    if counts_cut is None:
        cf_low = None
    else:
        cf_low = coeff_dict.get('cf_nonlin_low')[:,y1:y2,x1:x2]

    res = np.zeros_like(cube)
    for i in trange(nz, desc='Linearity', leave=False):
        frame = cube[i]

        if counts_cut is None:
            gain = pixel_linearity_gains(frame, cf_arr, use_legendre=use_legendre, lxmap=lxmap)
        else:
            ind1 = (frame >= counts_cut)
            ind2 = ~ind1

            gain = np.zeros_like(frame)
            if ind1.sum()>0: # Upper values
                gain[ind1] = pixel_linearity_gains(frame[ind1], cf_arr[:,ind1], use_legendre=use_legendre, lxmap=lxmap)
            if ind2.sum()>0: # Lower values
                gain[ind2] = pixel_linearity_gains(frame[ind2], cf_low[:,ind2], use_legendre=use_legendre, lxmap=lxmap)

        gain = gain.reshape([ny,nx])
        # Convert from DN to electrons
        res[i,:] = frame * gain
        del gain

        # For reference pixels, copy frame data and multiple by detector gain
        mask_ref = det.mask_ref
        res[i,mask_ref] = frame[mask_ref] * det.gain

    return res

def apply_nonlin(cube, det, coeff_dict, randomize=True, rand_seed=None):
    """Apply pixel non-linearity to ideal ramp

    Given a simulated cube of data in electrons, apply non-linearity 
    coefficients to obtain values in DN (ADU). This 

    Parameters
    ----------
    cube : ndarray
        Simulated ramp data in e-. These should be intrinsic
        flux values with Poisson noise, but prior to read noise,
        kTC, IPC, etc. Size (nz,ny,nx). Should match det subarray shape.
    det : Detector Class
        Desired detector class output
    coeff_dict : ndarray
        Dictionary holding coefficient information:

            - 'cf_nonlin'    : Set of polynomial coefficients of size (ncf,ny,nx).
            - 'use_legendre' : Coefficients use Legendre polynomials?
            - 'lxmap'        : Legendre polynomial normalization range, usually [0,1e5]
            - 'sat_vals'     : An image indicating what saturation levels in DN for each pixel

        Possible to separately fit lower flux values:

             - 'counts_cut'    : Flux cut-off value in electrons
             - 'cf_nonlin_low' : Coefficients for flux values below counts_cut

        To include randomization in line with observed variation:

            - 'cflin0_mean'    : Average 0th-order coefficient
            - 'cflin0_std'     : Measured standard deviation of 0th-order coefficent
            - 'corr_slope'     : Slope of linear correlation between 0th-order and higher orders
            - 'corr_intercept' : Intercept of linear Correaltion between 0th-order and higher orders
    
    Keyword Args
    ------------
    randomize : bool
        Add variation to the non-linearity coefficients  
    """

    rng = np.random.default_rng(rand_seed)

    nz, _, _ = cube.shape
    nx, ny = (det.xpix, det.ypix)

    # Need to crop input coefficients in the event of subarrays
    x1, x2 = (det.x0, det.x0 + nx)
    y1, y2 = (det.y0, det.y0 + ny)

    if cube.shape[-2]!=ny or cube.shape[-1]!=nx:
        # Assume full frame cube needs to be cropped
        cube = cube[:,y1:y2,x1:x2]

    # Nominal coefficient array
    cf_arr         = coeff_dict.get('cf_nonlin')[:,y1:y2,x1:x2]
    use_legendre   = coeff_dict.get('use_legendre', False)
    lxmap          = coeff_dict.get('lxmap')

    # Mean and standard deviation of first coefficients
    cflin0_mean    = coeff_dict.get('cflin0_mean', cf_arr[0])[y1:y2,x1:x2]
    cflin0_std     = coeff_dict.get('cflin0_std')[y1:y2,x1:x2]
    # The rest of the coefficents have a direct correlation to the first
    corr_slope     = coeff_dict.get('corr_slope')[:,y1:y2,x1:x2]
    corr_intercept = coeff_dict.get('corr_intercept')[:,y1:y2,x1:x2]

    # Information for lower flux values
    counts_cut = coeff_dict.get('counts_cut')
    if counts_cut is None:
        cf_low = None
    else:
        cf_low = coeff_dict.get('cf_nonlin_low')[:,y1:y2,x1:x2]

    sat_vals = coeff_dict.get('sat_vals')[y1:y2,x1:x2] # Saturation in DN
    well_depth = det.well_level # Full well in e- corresponding to sat in DN

    if randomize:
        cf0_rand = rng.normal(loc=cflin0_mean, scale=cflin0_std)
        cf_arr = np.concatenate(([cf0_rand], corr_slope * cf0_rand + corr_intercept))

    res = np.zeros_like(cube)
    for i in trange(nz, desc='Non-Linearity', leave=False):
        frame = cube[i]

        # Values higher than well depth
        ind_high = frame > well_depth

        if counts_cut is None:
            gain = pixel_linearity_gains(frame, cf_arr, use_legendre=use_legendre, lxmap=lxmap)
        else:
            ind1 = (frame >= counts_cut)
            ind2 = ~ind1

            gain = np.zeros_like(frame)
            if ind1.sum()>0: # Upper values
                gain[ind1] = pixel_linearity_gains(frame[ind1], cf_arr[:,ind1], 
                                                   use_legendre=use_legendre, lxmap=lxmap)
            if ind2.sum()>0: # Lower values
                gain[ind2] = pixel_linearity_gains(frame[ind2], cf_low[:,ind2], 
                                                   use_legendre=use_legendre, lxmap=lxmap)

        gain = gain.reshape([ny,nx])
        # Avoid NaNs
        igood = gain!=0
        # Convert from electrons to ADU
        res[i,igood] = frame[igood] / gain[igood]
        del gain

        # Correct any pixels that are above saturation DN
        ind_over = (res[i]>sat_vals) | ind_high
        res[i,ind_over] = sat_vals[ind_over]

        # For reference pixels, copy frame data and divide by detector gain
        # Normally reference pixels should start as 0s, but just in case...
        mask_ref = det.mask_ref
        res[i,mask_ref] = frame[mask_ref] / det.gain


    return res

def get_flat_fields(im_slope, split_low_high=False, smth_sig=10, ref_info=[4,4,4,4]):
    """ Calculate QE variations in flat field"""

    from astropy.convolution import convolve_fft, Gaussian2DKernel

    ny, nx = im_slope.shape

    # Crop out active pixel region
    lower, upper, left, right = ref_info
    iy1, iy2 = (lower, ny - upper)
    ix1, ix2 = (left, nx - right)
    im_act = im_slope[iy1:iy2,ix1:ix2]

    # Assuming a uniformly illuminated field, get fractional QE variations
    qe_frac = im_act / np.median(im_act)

    ### Outlier removal

    # Perform a quick median filter
    imarr = []
    xysh = 2
    for xsh in np.arange(-xysh, xysh):
        for ysh in np.arange(-xysh, xysh):
            if not xsh==ysh==0:
                im_shift = fshift(qe_frac, delx=xsh, dely=ysh, pad=True, cval=1)
                imarr.append(im_shift)
                
    imarr = np.asarray(imarr)
    im_med = np.median(imarr, axis=0)

    del imarr

    # Replace outliers with the median of their surrounding values
    diff = qe_frac - im_med
    mask_good = robust.mean(diff, Cut=20, return_mask=True)
    mask_bad = ~mask_good
    qe_frac[mask_bad] = im_med[mask_bad]

    if split_low_high:

        # Perform a Gaussian smooth to get low frequency flat field info
        kernel = Gaussian2DKernel(smth_sig)
        qe_frac_pad = np.pad(qe_frac, pad_width=100, mode='reflect')
        im_smth = convolve_fft(qe_frac_pad, kernel, allow_huge=True, boundary='fill')

        lflats = pad_or_cut_to_size(im_smth, qe_frac.shape)
        pflats = qe_frac / lflats
        # Set QE variations of ref pixels to 1 (fill_val=1)
        lflats = pad_or_cut_to_size(lflats, (ny,nx), fill_val=1)
        pflats = pad_or_cut_to_size(pflats, (ny,nx), fill_val=1)

        return lflats, pflats
    else:
        return pad_or_cut_to_size(qe_frac, (ny,nx), fill_val=1)

