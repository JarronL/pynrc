from __future__ import division, print_function, unicode_literals

from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel
from astropy.convolution import convolve_fft
from scipy import fftpack
from copy import deepcopy

from webbpsf_ext.image_manip import convolve_image, _convolve_psfs_for_mp
from webbpsf_ext.image_manip import convolve_image, make_disk_image, distort_image
from webbpsf_ext.bandpasses import nircam_com_th

# Import libraries
from .pynrc_core import NIRCam, DetectorOps, merge_dicts
from .nrc_utils import *
from .maths.coords import gen_sgd_offsets

from tqdm.auto import tqdm, trange

import logging
_log = logging.getLogger('pynrc')

import numpy as np
eps = np.finfo(float).eps

class nrc_hci(NIRCam):
    """NIRCam coronagraphy (and direct imaging)

    Subclass of the :mod:`~pynrc.NIRCam` instrument class with updates for PSF
    generation of off-axis PSFs. If a coronagraph is not present,
    then this is effectively the same as the :mod:`~pynrc.NIRCam` class.

    Parameters
    ----------
    wind_mode : str
        'FULL', 'STRIPE', or 'WINDOW'
    xpix : int
        Size of the detector readout along the x-axis. The detector is
        assumed to be in window mode  unless the user explicitly
        sets wind_mode='FULL'.
    ypix : int
        Size of the detector readout along the y-axis. The detector is
        assumed to be in window mode  unless the user explicitly
        sets wind_mode='FULL'.
    large_grid : bool
        Use a large number (high-density) of grid points to create coefficients.
        If True, then produces a higher fidelity PSF variations across the FoV, 
        but will take much longer to genrate on the first pass and requires more
        disk space and memory while running.
    bar_offset : float
        Custom offset position along bar mask (-10 to +10 arcsec).
    use_ap_info : bool   
        For subarray observations, the mask reference points are not
        actually in the center of the array. Set this to true to 
        shift the sources to actual aperture reference location. 
        Default is to place in center of array.
    autogen_coeffs : bool
        Automatically generate base PSF coefficients. Equivalent to performing
        `self.gen_psf_coeff()`. `gen_wfedrift_coeff`, and `gen_wfemask_coeff`.
        Default: True.
    sgd_type : str or None
        Small grid dither pattern. Valid types are
        '9circle', '5box', '5diamond', '3bar', or '5bar'. If 'auto', 
        then defaults are '5diamond' for round masks, '5bar' for bar masks, 
        and '5diamond' for direct imaging. If None, then no FSM pointings,
        but there will be a single slew.
    fsm_std : float
        One-sigma accuracy per axis of fine steering mirror positions.
        This provides randomness to each position relative to the nominal 
        central position. Ignored for central position. 
        Values should be in units of mas. 
    slew_std : float
        One-sigma accuracy per axis of the initial slew. This is applied
        to all positions and gives a baseline offset relative to the
        desired mask center. ***Values should be in units of mas***
    """

    def __init__(self, wind_mode='WINDOW', xpix=320, ypix=320, large_grid=False, bar_offset=None, 
                 use_ap_info=False, autogen_coeffs=True, sgd_type=None, slew_std=5, fsm_std=2.5, **kwargs):

        super().__init__(wind_mode=wind_mode, xpix=xpix, ypix=ypix, fov_bg_match=True, 
                         autogen_coeffs=autogen_coeffs, **kwargs)

        if autogen_coeffs:
            # Enable WFE drift
            self.gen_wfedrift_coeff()
            # Enable mask-dependent
            self.gen_wfemask_coeff(large_grid=large_grid)

        log_prev = conf.logging_level
        setup_logging('WARN', verbose=False)

        self._use_ap_info = use_ap_info

        # Set locations based on detector
        self._set_xypos()
        # Create mask throughput images seen by detector
        self._gen_cmask()

        # Create default SGD offsets
        self.sgd_type = sgd_type
        self.slew_std = slew_std
        self.fsm_std  = fsm_std
        self.gen_pointing_offsets()

        setup_logging(log_prev, verbose=False)

        self._bar_offset = bar_offset

    @property
    def bar_offset(self):
        """Offset position along bar mask (arcsec)."""
        if 'TA' in self.siaf_ap.AperName:
            bar_offset = 0
        elif self._bar_offset is None:
            # bar_offset, _ = offset_bar(self._filter, self.image_mask)
            narrow = ('NARROW' in self.siaf_ap.AperName)
            bar_offset = self.get_bar_offset(narrow=narrow)
            bar_offset = 0 if bar_offset is None else bar_offset # Circular masks return None
        else:
            bar_offset = self._bar_offset
        return bar_offset
    @bar_offset.setter
    def bar_offset(self, value):
        """Set the bar offset position. None to auto-determine"""
        # Only update if the value changes
        if self.image_mask is None:
            self._bar_offset = 0 #None
        elif self.image_mask[-2:]=='WB':
            # Value limits between -10 and 10
            if (value is not None) and np.abs(value)>10:
                value = 10 if value>0 else -10
                msg1 = 'bar_offset value must be between -10 and 10 arcsec.'
                msg2 = '  Setting to {}.'.format(value)
                _log.warning('{}\n{}'.format(msg1,msg2))
            
            self._bar_offset = value
        else:
            self._bar_offset = 0


    def gen_offset_psf(self, offset_r, offset_theta, sp=None, return_oversample=False, 
        wfe_drift=None, use_coeff=True, coron_rescale=False, use_cmask=False, **kwargs):
        """Create a PSF offset from center FoV

        Generate some off-axis PSF at a given (r,theta) offset from center of mask.
        The `offset_r` and `offset_theta` parameters are assumed to be in 'idl' frame.
        This function is mainly for coronagraphic observations where the
        off-axis PSF varies w.r.t. position. The PSF is centered in the
        resulting image. The offset values are assumed to be in 'idl' coordinate
        frame relative to the mask center.

        Parameters
        ----------
        offset_r : float
            Radial offset of the target from center in arcsec.
        offset_theta : float
            Position angle for that offset, in degrees CCW (+Y).

        Keyword Args
        ------------
        sp : None, :mod:`pysynphot.spectrum`
            If not specified, the default is flat in phot lam
            (equal number of photons per spectral bin).
        return_oversample : bool
            Return either the pixel-sampled or oversampled PSF.
        use_coeff : bool
            If True, uses `calc_psf_from_coeff`, other WebbPSF's built-in `calc_psf`.
        coron_rescale : bool
            Rescale off-axis coronagraphic PSF to better match analytic prediction
            when source overlaps coronagraphic occulting mask.
            Primarily used for planetary companion PSFs.
        """

        coords = rtheta_to_xy(offset_r, offset_theta)

        # FULL TA coronagraphic masks require siaf_ap to be passed
        # since we want offset relative to TA position rather than
        # center of the coronagraphic 20x20" aperture
        apname = self.siaf_ap.AperName
        if self.is_coron and ('TAMASK' in apname) and ('FULL' in apname):
            kwargs['siaf_ap'] = self.siaf_ap
            use_cmask = True

        if use_coeff:
            psf = self.calc_psf_from_coeff(sp=sp, return_oversample=return_oversample, 
                wfe_drift=wfe_drift, coord_vals=coords, coord_frame='idl', 
                coron_rescale=coron_rescale, return_hdul=False, **kwargs)
        else:
            psf = self.calc_psf(sp=sp, return_hdul=False, return_oversample=return_oversample, 
                wfe_drift=wfe_drift, coord_vals=coords, coord_frame='idl', **kwargs)

        # Being coronagraphic mask attenuation
        if not self.is_coron:
            return psf


        # Determine if any throughput loss due to coronagraphic mask
        # artifacts, such as the mask holder or ND squares.
        # If ND_acq=True, then PSFs already included ND throughput in bandpass.
        if use_cmask and self.is_coron and (not self.ND_acq):
            # First, anything in a rectangular region around the
            # mask has already been correctly accounted for
            delx_asec, dely_asec = coords
            if (( np.abs(delx_asec)<10 and np.abs(dely_asec)<4.5 ) and
                ('TAMASK' not in apname)):
                # Set transmission to 1 for coronagraphic observations
                # within occulting mask region
                trans = 1
            else:
                cmask = self.mask_images['DETSAMP']
                # 1. For the FULL TA masks, we want offsets relative
                #    to the mask reference point
                # 2. For all others, we want reference points relative to
                #    the 20"x20" coronagraphic field 
                if 'TAMASK' in apname:
                    siaf_ap_relative = self.siaf_ap 
                else:
                    si_mask_apname = self._psf_coeff_mod.get('si_mask_apname')
                    siaf_ap_relative = self.siaf[si_mask_apname]

                # Convert offsets w.r.t. to mask center to xsci/ysci for current siaf_ap
                # First get tel (V2/V3) coordinates, then 'sci' coords
                xtel, ytel = siaf_ap_relative.convert(delx_asec, dely_asec, 'idl', 'tel')
                xy_sci = self.siaf_ap.convert(xtel, ytel, 'tel', 'sci')
                xsci, ysci = np.array(xy_sci).astype('int')
                
                # Extract a 3x3 region to average
                cmask_sub = cmask[ysci-3:ysci+3,xsci-3:xsci+3]
                trans = np.mean(cmask_sub)
                # COM substrate already accounted for in filter throughput curve,
                # so it will be double-counted here. Need to divide it out.
                w_um = self.bandpass.avgwave() / 1e4
                com_th = nircam_com_th(wave_out=w_um)
                trans /= com_th
        else:
            trans = 1

        psf *= trans

        return psf

    def _set_xypos(self, xy=None):
        """
        Set x0 and y0 subarray positions.
        """
        wind_mode = self.det_info['wind_mode']
        if xy is not None:
            self.update_detectors(x0=xy[0], y0=xy[1])
        elif self._use_ap_info:
            xvert, yvert = self.siaf_ap.closed_polygon_points('det')
            xy = (np.min(xvert).astype('int'), np.min(yvert).astype('int'))
            self.update_detectors(x0=xy[0], y0=xy[1])
        elif self.is_coron:
            full = True if 'FULL' in wind_mode else False
            if full:
                xcen, ycen = self.siaf_ap.reference_point('det')
            else:
                cdict = coron_ap_locs(self.module, self.channel, self.image_mask, full=full)
                xcen, ycen = cdict['cen']

            xpix, ypix = (self.det_info['xpix'], self.det_info['ypix'])
            if full: 
                x0, y0 = (0,0)
            else: 
                x0, y0 = (int(xcen-xpix/2), int(ycen-ypix/2))

            # Make sure subarray sizes don't push out of bounds
            if (y0 + ypix) > 2048: 
                y0 = 2048 - ypix
            if (x0 + xpix) > 2048: 
                x0 = 2048 - xpix
                
            self.update_detectors(x0=x0, y0=y0)

    def _gen_cmask(self):
        """ Generate coronagraphic mask transmission images.

        Output images are in 'sci' coordinates.
        """
        self.mask_images = gen_coron_mask(self)

    def attenuate_with_mask(self, image_oversampled, cmask=None):
        """ Image attenuation from coronagraph mask features

        Multiply image data by coronagraphic mask.
        Excludes region already affected by observed occulting mask.
        Involves mainly ND Squares and opaque COM holder.
        Appropriately accounts for COM substrate wavelength-dep throughput.
        
        WARNING: If self.ND_acq=True, then bandpass already includes
        ND throughput, so be careful not to double count.
        """

        cmask = self.mask_images.get('OVERSAMP') if cmask is None else cmask

        # In case of imaging
        if cmask is None:
            return image_oversampled

        return attenuate_with_coron_mask(self, image_oversampled, cmask)

    def gen_pointing_offsets(self, rand_seed=None):
        """
        Create a series of x and y position offsets for a SGD pattern.
        This includes the central position as the first in the series.
        By default, will also add random movement errors using the
        `slew_std` and `fsm_std` keywords. Returned values are in arcsec.

        Uses the attributes `sgd_type`, `slew_std`, and `fsm_std`.
        
        Attributes
        ==========
        sgd_type : str or None
            Small grid dither pattern. Valid types are
            '9circle', '5box', '5diamond', '3bar', or '5bar'. If 'auto', 
            then defaults are '5diamond' for round masks, '5bar' for bar masks, 
            and '5diamond' for direct imaging. If None, then no FSM pointings,
            but there will be a single slew.
        fsm_std : float
            One-sigma accuracy per axis of fine steering mirror positions.
            This provides randomness to each position relative to the nominal 
            central position. Ignored for central position. 
            Values should be in units of mas. 
        slew_std : float
            One-sigma accuracy per axis of the initial slew. This is applied
            to all positions and gives a baseline offset relative to the
            desired mask center. 
            Values should be in units of mas.

        Parameters
        ==========
        rand_seed : int
            Input a random seed in order to make reproduceable pseudo-random
            numbers.
        """
        sgd_type = self.sgd_type
        slew_std = self.slew_std
        fsm_std  = self.fsm_std

        if sgd_type == 'auto':
            if self.is_coron and self.image_mask[-1]=='R':
                sgd_type = '5diamond'
            elif self.is_coron and self.image_mask[-1]=='B':
                sgd_type = '5bar'
            else:
                sgd_type = '5diamond'

        if sgd_type is None:
            rng = np.random.default_rng(seed=rand_seed)
            xoff, yoff = rng.normal(scale=slew_std, size=2) / 1000
            fsm_std = None
        else:
            xoff, yoff = gen_sgd_offsets(sgd_type, slew_std=slew_std, fsm_std=fsm_std, rand_seed=rand_seed)

        _log.info("Saving SGD position info to `self.pointing_info` dictionary attribute")
        self.pointing_info = {'xoff' : xoff, 'yoff' : yoff,}


class obs_hci(nrc_hci):
    """NIRCam coronagraphic observations

    Subclass of the :mod:`~pynrc.nrc_hci` instrument class used to observe
    stars (plus exoplanets and disks) with either a coronagraph or direct
    imaging.

    The main concept is to generate a science target of the primary
    source along with a simulated disk structure. Planets are further
    added to the astronomical scene. A separate reference source is
    also defined for PSF subtraction, which contains a specified WFE.
    A variety of methods exist to generate slope images and analyze
    the PSF-subtracted results via images and contrast curves.

    Parameters
    ----------
    sp_sci : :mod:`pysynphot.spectrum`
        A pysynphot spectrum of science target (e.g., central star).
        Should already be normalized to the apparent flux.
    sp_ref : :mod:`pysynphot.spectrum` or None
        A pysynphot spectrum of reference target.
        Should already be normalized to the apparent flux.
    distance : float
        Distance in parsecs to the science target. This is used for
        flux normalization of the planets and disk.
    wfe_ref_drift: float
        WFE drift in nm RMS between the science and reference targets.
        Expected values are between ~3-10 nm.
    wfe_roll_drift: float
        WFE drift in nm RMS between science roll angles. Default=0.
    wind_mode : str
        'FULL', 'STRIPE', or 'WINDOW'
    xpix : int
        Size of the detector readout along the x-axis. The detector is
        assumed to be in window mode  unless the user explicitly
        sets wind_mode='FULL'.
    ypix : int
        Size of the detector readout along the y-axis. The detector is
        assumed to be in window mode  unless the user explicitly
        sets wind_mode='FULL'.
    disk_params : dict
        Arguments describing disk model information for a given FITS file:
            - 'file'       : Path to model file or an HDUList.
            - 'pixscale'   : Pixel scale for model image (arcsec/pixel).
            - 'dist'       : Assumed model distance in parsecs.
            - 'wavelength' : Wavelength of observation in microns.
            - 'units'      : String of assumed flux units (ie., MJy/arcsec^2 or muJy/pixel)
            - 'cen_star'   : True/False. Is a star already placed in the central pixel? 
    autogen_coeffs : bool
        Automatically generate base PSF coefficients. Equivalent to performing
        `self.gen_psf_coeff()`. `gen_wfedrift_coeff`, and `gen_wfemask_coeff`.
        Default: True.
    use_ap_info : bool   
        For subarray observations, the mask reference points are not actually in the 
        center of the array. Set this to True to shift the sources to actual 
        aperture reference location. Otherwise, default will place in center of array.
    sgd_type : str or None
        Small grid dither pattern. Valid types are
        '9circle', '5box', '5diamond', '3bar', or '5bar'. If 'auto', 
        then defaults are '5diamond' for round masks, '5bar' for bar masks, 
        and '5diamond' for direct imaging. If None, then no FSM pointings,
        but there will be a single slew.
    fsm_std : float
        One-sigma accuracy per axis of fine steering mirror positions.
        This provides randomness to each position relative to the nominal 
        central position. Ignored for central position. 
        Values should be in units of mas. 
    slew_std : float
        One-sigma accuracy per axis of the initial slew. This is applied
        to all positions and gives a baseline offset relative to the
        desired mask center. ***Values should be in units of mas***
    """

    def __init__(self, sp_sci, distance, sp_ref=None, wfe_ref_drift=5, wfe_roll_drift=2,
        wind_mode='WINDOW', xpix=320, ypix=320, disk_params=None, autogen_coeffs=True,
        sgd_type=None, slew_std=5, fsm_std=2.5, **kwargs):

        if 'FULL'   in wind_mode: xpix = ypix = 2048
        if 'STRIPE' in wind_mode: xpix = 2048

        super().__init__(wind_mode=wind_mode, xpix=xpix, ypix=ypix, autogen_coeffs=autogen_coeffs, 
                         sgd_type=sgd_type, slew_std=slew_std, fsm_std=fsm_std, **kwargs)

        # wind_mode = self.Detector.wind_mode
        # if (wind_mode=='FULL') and (self.channel=='short' or self.channel=='SW'):
        #     raise NotImplementedError('SW Full Frame not yet implemented.')

        # Spectral models
        self.sp_sci = sp_sci
        self.sp_ref = sp_sci if sp_ref is None else sp_ref
        self.wfe_ref_drift = wfe_ref_drift
        self.wfe_roll_drift = wfe_roll_drift

        # Distance to source in pc
        self.distance = distance
        self._planets = []

        # Open and rescale input disk image to observation parameters
        self._disk_params = disk_params
        self.gen_disk_hdulist()
        if autogen_coeffs:
            self.gen_disk_psfs()

    @property
    def wfe_ref_drift(self):
        """WFE drift (nm) of ref obs relative to sci obs"""
        return self._wfe_ref_drift
    @wfe_ref_drift.setter
    def wfe_ref_drift(self, value):
        """Set the WFE drift value between sci and ref observations"""
        self._wfe_ref_drift = value

    @property
    def wfe_roll_drift(self):
        """WFE drift (nm) of Roll2 obs relative to Roll1 obs"""
        return self._wfe_roll_drift
    @wfe_roll_drift.setter
    def wfe_roll_drift(self, value):
        """Set the WFE drift value between roll observations"""
        self._wfe_roll_drift = value

    def gen_pointing_offsets(self, sgd_type=None, slew_std=5, fsm_std=2.5, 
        rand_seed=None, verbose=False):
        """
        Create a series of x and y position offsets for a SGD pattern.
        This includes the central position as the first in the series.
        By default, will also add random position errors using the
        `slew_std` and `fsm_std` keywords. Returned values are in arcsec.

        This initializes a set of target acquisition offsets
        for each roll position and reference observation.  
        
        Parameters
        ==========
        sgd_type : str or None
            Small grid dither pattern. Valid types are
            '9circle', '5box', '5diamond', '3bar', '5bar', '5miri', and '9miri'
            where the first four refer to NIRCam coronagraphic dither
            positions and the last two are for MIRI coronagraphy. If 'auto', 
            then defaults are '5diamond' for round masks, '5bar' for bar masks, 
            and '5diamond' for direct imaging. If None, then no FSM pointings,
            but there will be a single slew.
        fsm_std : float
            One-sigma accuracy per axis of fine steering mirror positions.
            This provides randomness to each position relative to the nominal 
            central position. Ignored for central position. 
            Values should be in units of mas. 
        slew_std : float
            One-sigma accuracy per axis of the initial slew. This is applied
            to all positions and gives a baseline offset relative to the
            desired mask center. 
            Values should be in units of mas.
        """
        sgd_type = self.sgd_type
        slew_std = self.slew_std
        fsm_std  = self.fsm_std

        if sgd_type == 'auto':
            if self.is_coron and self.image_mask[-1]=='R':
                sgd_type = '5diamond'
            elif self.is_coron and self.image_mask[-1]=='B':
                sgd_type = '5bar'
            else:
                sgd_type = '5diamond'

        if sgd_type is None:
            rng = np.random.default_rng(seed=rand_seed)
            xyoff_ref = rng.normal(scale=slew_std, size=2) / 1000
            fsm_std = None
        else:
            xoff_ref, yoff_ref = gen_sgd_offsets(sgd_type, slew_std=slew_std, 
                                                 fsm_std=fsm_std, rand_seed=rand_seed)
            xyoff_ref = np.array([xoff_ref,yoff_ref]).transpose()

        xyoff_roll1 = rng.normal(scale=slew_std, size=2) / 1000
        xyoff_roll2 = rng.normal(scale=slew_std, size=2) / 1000

        _log.info("Saving SGD position info to `self.pointing_info` dictionary attribute")
        self.pointing_info = {
            'sgd_type': sgd_type, 'slew_std': slew_std, 'fsm_std': fsm_std,
            'roll1': xyoff_roll1, 'roll2': xyoff_roll2, 'ref': xyoff_ref,
        }

        if verbose:
            print('Pointing Info')
            for k in self.pointing_info.keys():
                v = self.pointing_info[k]
                print("  {:<10} :".format(k), v)

    @property
    def disk_params(self):
        return self._disk_params
    def gen_disk_hdulist(self, file=None, pixscale=None, dist=None, 
                         wavelength=None, units=None, cen_star=None, shape_out=None):
        """Create a correctly scaled disk model image.

        Rescale disk model flux to current pixel scale and distance.
        If instrument bandpass is different from disk model, scales 
        flux assuming a grey scattering model. 

        Result (in photons/sec) is saved in self.disk_hdulist attribute.
        """
        if self._disk_params is None:
            self.disk_hdulist = None
        else:

            if file is not None:
                self._disk_params['file'] = file
            if pixscale is not None:
                self._disk_params['pixscale'] = pixscale
            if dist is not None:
                self._disk_params['dist'] = dist
            if wavelength is not None:
                self._disk_params['wavelength'] = wavelength
            if units is not None:
                self._disk_params['units'] = units
            if cen_star is not None:
                self._disk_params['cen_star'] = cen_star

            file       = self._disk_params['file']
            pixscale   = self._disk_params['pixscale']
            dist       = self._disk_params['dist']
            wavelength = self._disk_params['wavelength']
            units      = self._disk_params['units']
            cen_star   = self._disk_params['cen_star']

            # TODO: Double-check 'APERNAME', 'DET_X', 'DET_Y', 'DET_V2', 'DET_V3'
            # correspond to the desired observation
            disk_hdul = _gen_disk_hdulist(self, file, pixscale, dist, wavelength, units, cen_star, 
                                          sp_star=self.sp_sci, dist_out=self.distance, shape_out=shape_out)
            self.disk_hdulist = disk_hdul


    # Any time update_detectors is called, also call gen_ref_det()
    def update_detectors(self, verbose=False, **kwargs):
        super().update_detectors(verbose=verbose, **kwargs)

        # Updates ref detector window size
        self.gen_ref_det()

    def gen_ref_det(self, **kwargs):
        """
        Function to generate and update Reference Detector class.
        Used to keep track of detector and multiaccum config,
        which can differ between sci and ref observations.
        """

        # Check if kwargs is empty
        if not kwargs:
            try: 
                kwargs = self._det_info_ref
            except AttributeError: 
                kwargs = {}
        else:
            try: 
                self._det_info_ref.update(kwargs)
            except AttributeError: 
                self._det_info_ref = kwargs
            kwargs = self._det_info_ref

        # These should always be the same between sci and ref
        kw_copy = ['wind_mode', 'xpix', 'ypix', 'x0', 'y0']
        for kw in kw_copy:
            kwargs[kw] = self.det_info[kw]

        # Update reference detector class
        try:
            del self.Detector_ref
        except AttributeError: 
            pass
        self.Detector_ref = DetectorOps(detector=self.detector, **kwargs)

        # Update stored kwargs
        kw1 = self.Detector_ref.to_dict()
        _ = kw1.pop('detector', None)
        kw2 = self.Detector_ref.multiaccum.to_dict()
        self._det_info_ref = merge_dicts(kw1,kw2)


    def gen_disk_psfs(self, wfe_drift=0, force=False, **kwargs):
        """
        Save instances of NIRCam PSFs that are incrementally offset
        from coronagraph center to convolve with a disk image.
        """
        from webbpsf_ext.webbpsf_ext_core import _transmission_map

        if (self._disk_params is None) and (force==False):
            # No need to generate if no disk
            self.psf_list = None
        elif self.image_mask is None:
            # If no mask, then assume PSF looks the same at all radii
            # This return a single ndarray
            kwargs['return_oversample'] = True
            kwargs['return_hdul'] = True
            kwargs['wfe_drift'] = wfe_drift
            self.psf_list = self.calc_psf_from_coeff(**kwargs)
        elif 'WB' in self.image_mask:
            # Bar mask
            kwargs['ysci_vals'] = 0
            kwargs['xsci_vals'] = np.linspace(-8,8,9) # np.linspace(-9,9,19)
            kwargs['wfe_drift'] = wfe_drift
            self.psf_list = self.calc_psfs_grid(osamp=self.oversample, **kwargs)
        else:
            # Circular masks, just need single on-axis PSF
            kwargs['return_oversample'] = True
            kwargs['return_hdul'] = True
            kwargs['wfe_drift'] = wfe_drift
            self.psf_list = self.calc_psf_from_coeff(**kwargs)

        # Generate oversampled mask transmission for mask-dependent PSFs
        if (self.image_mask is not None) and (self.mask_images.get('OVERMASK') is None):
            nx, ny = (self.det_info['xpix'], self.det_info['ypix'])
            if 'FULL' in self.det_info['wind_mode']:
                trans = build_mask_detid(self.Detector.detid, oversample=self.oversample,
                                         pupil=self.pupil_mask, nd_squares=False, mask_holder=False)
            elif self._use_ap_info:
                siaf_ap = self.siaf_ap
                xv = np.arange(nx)
                yv = np.arange(ny)
                xg, yg = np.meshgrid(xv,yv)
                res = _transmission_map(self, (xg,yg), 'sci', siaf_ap=siaf_ap)
                trans = frebin(res[0]**2, scale=self.oversample)
            else:
                siaf_ap = self.siaf[self.aperturename]
                xr, yr = siaf_ap.reference_point('sci')
                xv = np.arange(nx) - nx/2 + xr
                yv = np.arange(ny) - ny/2 + yr
                xg, yg = np.meshgrid(xv,yv)
                xidl, yidl = siaf_ap.convert(xg,yg,'sci','idl')
                res = _transmission_map(self, (xidl,yidl), 'idl', siaf_ap=siaf_ap)
                trans = frebin(res[0]**2, scale=self.oversample)
            self.mask_images['OVERMASK'] = trans

            # renormalize all PSFs for disk convolution to 1
            # for hdu in self.psf_list:
            #     hdu.data /= hdu.data.sum()

    def planet_spec(self, **kwargs):
        """Exoplanet spectrum

        Return the planet spectrum from Spiegel & Burrows (2012) normalized
        to distance of current target. Output is a :mod:`pysynphot.spectrum`.

        See `spectra.companion_spec()` function for more details.
        """

        sp = companion_spec(self.bandpass, dist=self.distance, **kwargs)
        return sp

    @property
    def planets(self):
        """Planet info (if any exists)"""
        return self._planets

    def delete_planets(self):
        """Remove planet info"""
        try: del self._planets
        except: pass
        self._planets = []

    def add_planet(self, model='SB12', atmo='hy3s', mass=10, age=100, entropy=10,
        xy=None, rtheta=None, runits='AU', Av=0, renorm_args=None, sptype=None,
        accr=False, mmdot=None, mdot=None, accr_rin=2, truncated=False, **kwargs):
        """Insert a planet into observation.

        Add exoplanet information that will be used to generate a point
        source image using a spectrum from Spiegel & Burrows (2012).
        Use self.delete_planets() to delete them.

        Coordinate convention is for +N up and +E to left.

        Parameters
        ==========
        model : str
            Exoplanet model to use ('sb12', 'bex', 'cond') or
            stellar spectrum model ('bosz', 'ck04models', 'phoenix').
        atmo : str
            A string consisting of one of four atmosphere types:
            ['hy1s', 'hy3s', 'cf1s', 'cf3s'].
        mass: int
            Number 1 to 15 Jupiter masses.
        age: float
            Age in millions of years (1-1000).
        entropy: float
            Initial entropy (8.0-13.0) in increments of 0.25

        sptype : str
            Instead of using a exoplanet spectrum, specify a stellar type.
        renorm_args : dict
            Pysynphot renormalization arguments in case you want
            very specific luminosity in some bandpass.
            Includes (value, units, bandpass).

        Av : float
            Extinction magnitude (assumes Rv=4.0) of the exoplanet
            due to being embedded in a disk.

        xy : tuple, None
            (X,Y) position in sky coordinates of companion (N up, E left).
        rtheta : tuple, None
            Radius and position angle relative to stellar position.
            Alternative to xy keyword.
        runits : str
            What are the spatial units? Valid values are 'AU', 'asec', or 'pix'.

        accr : bool
            Include accretion? default: False
        mmdot : float
            From Zhu et al. (2015), the Mjup^2/yr value.
            If set to None then calculated from age and mass.
        mdot : float
            Or use mdot (Mjup/yr) instead of mmdot.
        accr_rin : float
            Inner radius of accretion disk (units of RJup; default: 2)
        truncated: bool
             Full disk or truncated (ie., MRI; default: False)?

         """

        if (xy is None) and (rtheta is None):
            _log.error('Either xy or rtheta must be specified.')

        if (xy is not None) and (rtheta is not None):
            _log.warning('Both xy and rtheta are specified. ' + \
                         'The xy keyword shall take priority.')

        # Size of subarray image in terms of pixels
        image_shape = (self.det_info['ypix'], self.det_info['xpix'])

        # XY location of planets within subarray with units from runits keyword
        loc = rtheta_to_xy(rtheta[0], rtheta[1]) if xy is None else xy

        # Define pixel location relative to the center of the subarray
        if 'AU' in runits:
            au_per_pixel = self.distance*self.pixelscale
            xoff, yoff = np.array(loc) / au_per_pixel
        elif ('asec' in runits) or ('arcsec' in runits):
            xoff, yoff = np.array(loc) / self.pixelscale
        elif ('pix' in runits):
            xoff, yoff = loc
        else:
            errstr = "Do not recognize runits='{}'".format(runits)
            raise ValueError(errstr)

        # Offset in terms of arcsec
        xoff_asec, yoff_asec = np.array([xoff, yoff]) * self.pixelscale
        _log.debug('(xoff,yoff) = {} pixels'.format((xoff,yoff)))
        _log.debug('(xoff_asec,yoff_asec) = {} arcsec'.format((xoff_asec,yoff_asec)))

        # Make sure planet is within image bounds
        sh_diff = np.abs(np.array([yoff,xoff])) - np.array(image_shape)/2
        if np.any(sh_diff>=0):
            _log.warning('xoff,yoff = {} is beyond image boundaries.'.format((xoff,yoff)))

        # X and Y pixel offsets from center of image
        # Dictionary of planet info
        if sptype is None:
            d = {'model':model, 'atmo':atmo, 'mass':mass, 'age':age,
                 'entropy':entropy, 'Av':Av, 'renorm_args':renorm_args,
                 'accr':accr, 'mmdot':mmdot, 'mdot':mdot, 'accr_rin':accr_rin,
                 'truncated':truncated, 'xyoff_asec':(xoff_asec, yoff_asec)}
        else:
            d = {'model':model, 'sptype':sptype, 'Av':Av,
                 'renorm_args':renorm_args, 'xyoff_asec':(xoff_asec, yoff_asec)}
        self._planets.append(d)


    def gen_planets_image(self, PA_offset=0, xyoff_asec=(0,0), use_cmask=True, 
        wfe_drift=None, use_coeff=True, return_oversample=False, coron_rescale=True, 
        **kwargs):
        """Create image of just planets.

        Use info stored in self.planets to create a noiseless slope image
        of just the exoplanets (no star). 

        Coordinate convention is for +N up and +E to left.

        Parameters
        ----------
        PA_offset : float
            Rotate entire scene by some position angle.
            Positive values are counter-clockwise from +Y direction.
            This should be -1 times telescope V3 PA.
        xyoff_asec : tuple
            Offsets (dx,dy) specified in arcsec. These are meant to be
            for minor shifts, use as SGD. Bar offsets are accounted
            for automatically. 
        use_cmask : bool
            Use the coronagraphic mask image to determine if any planet is
            getting obscurred by a corongraphic mask feature
        wfe_drift : float
            WFE drift value (in nm RMS). Not usually a concern for companion
            PSFs, so default is 0.
        use_coeff : bool
            If True, uses `calc_psf_from_coeff`, other WebbPSF's built-in `calc_psf`
            for stellar sources.
        return_oversample : bool
            Return either the detector pixel-sampled or oversampled image.
        coron_rescale : bool
            Rescale off-axis coronagraphic PSF to better match analytic prediction
            when source overlaps coronagraphic occulting mask. Default: True.

        """
        if len(self.planets)==0:
            _log.info("No planet info at self.planets")
            return 0.0

        PA_offset=0 if PA_offset is None else PA_offset

        # Additional field offsets
        offx_asec, offy_asec = xyoff_asec

        # Size of final image
        ypix, xpix = (self.det_info['ypix'], self.det_info['xpix'])
        xpix_over = xpix * self.oversample
        ypix_over = ypix * self.oversample

        # Oversampled pixel scale
        pixscale_over = self.pixelscale / self.oversample

        image_over = np.zeros([ypix_over, xpix_over])
        bar_offset = self.bar_offset
        bar_offpix = bar_offset / self.pixelscale

        for pl in tqdm(self.planets, desc='Companions', leave=False):

            ##################################
            # Generate Image

            # Create slope image (postage stamp) of planet
            sp = self.planet_spec(**pl)

            # Location relative to star
            plx_asec, ply_asec = pl['xyoff_asec']

            # Add in PA offset
            if PA_offset!=0:
                plx_asec, ply_asec = xy_rot(plx_asec, ply_asec, PA_offset)

            # print(f'Planet offset ({plx_asec:.4f}, {ply_asec:.4f}) asec')

            # Add in bar offset for PSF generation
            xoff_idl, yoff_idl = (plx_asec + offx_asec + bar_offset, ply_asec + offy_asec)
            r, th = xy_to_rtheta(xoff_idl, yoff_idl)
            psf_planet = self.gen_offset_psf(r, th, sp=sp, return_oversample=True, 
                                             use_coeff=use_coeff, wfe_drift=wfe_drift, 
                                             coron_rescale=coron_rescale, use_cmask=use_cmask,
                                             **kwargs)

            # Expand to full size
            psf_planet = pad_or_cut_to_size(psf_planet, (ypix_over, xpix_over))

            ##################################
            # Shift image

            # Determine final shift amounts to mask location
            # Shift to position relative to center of image
            if (('FULL' in self.det_info['wind_mode']) and (self.image_mask is not None)) or self._use_ap_info:
                xcen, ycen = self.siaf_ap.reference_point('sci')
                delx_pix = (xcen - (xpix/2 + 0.5))  # 'sci' pixel shifts
                dely_pix = (ycen - (ypix/2 + 0.5))  # 'sci' pixel shifts
                delx_asec, dely_asec = np.array([delx_pix, dely_pix]) * self.pixelscale
            else:
                # Otherwise assumed mask is already in center of subarray
                delx_pix, dely_pix = (bar_offpix, 0)
                delx_asec, dely_asec = np.array([delx_pix, dely_pix]) * self.pixelscale

            # Add dither offsets
            delx_asec += offx_asec 
            dely_asec += offy_asec 

            # PSF shifting
            delx_over, dely_over = np.array([delx_asec, dely_asec]) / pixscale_over
            delx_det, dely_det = np.array([delx_asec, dely_asec]) / self.pixelscale

            # print(f'delx, dely = ({delx_asec:.4f}, {dely_asec:.4f}) asec')
            # print(f'delx, dely = ({delx_det:.2f}, {dely_det:.2f}) det pixels')

            # Determine planet PSF shift in pixels compared to center of mask
            # Convert delx and dely from 'idl' to 'sci' coords
            xsci_ref, ysci_ref = self.siaf_ap.reference_point('sci')
            xsci_pl, ysci_pl = self.siaf_ap.idl_to_sci(plx_asec, ply_asec)
            delx_sci, dely_sci = (xsci_pl - xsci_ref, ysci_pl - ysci_ref)

            delx_over += delx_sci * self.oversample
            dely_over += dely_sci * self.oversample

            interp = 'linear' if ('FULL' in self.det_info['wind_mode']) else 'cubic'
            psf_planet = fshift(psf_planet, delx=delx_over, dely=dely_over, pad=True, interp=interp)

            # Add to image
            image_over += psf_planet

        if return_oversample:
            return image_over
        else:
            return frebin(image_over, scale=1/self.oversample)


    def gen_disk_image(self, PA_offset=0, xyoff_asec=(0,0), use_cmask=True, 
        return_oversample=False, **kwargs):
        """Create image of just disk.

        Generate a (noiseless) convolved image of the disk at some PA offset.
        The PA offset value will rotate the image CCW.
        Image units of e-/sec.

        Coordinate convention is for N up and E to left.

        For now, we don't perform any changes to the WFE.

        Parameters
        ----------
        PA_offset : float
            Rotate entire scene by some position angle.
            Positive values are counter-clockwise from +Y direction.
            This should be -1 times telescope V3 PA.
        xyoff_asec : tuple
            Offsets (dx,dy) specified in arcsec. These are meant to be
            for minor shifts, use as SGD. Bar offsets are accounted
            for automatically. 
        use_cmask : bool
            Use the coronagraphic mask image to attenuate disk regions
            getting obscurred by a corongraphic mask feature
        return_oversample : bool
            Return either the detector pixel-sampled or oversampled image.
        """

        if self.disk_hdulist is None:
            return 0.0

        # Final image shape
        det = self.Detector
        ypix, xpix = (det.ypix, det.xpix) #(self.det_info['ypix'], self.det_info['xpix'])

        oversample = self.oversample
        pixscale_over = self.pixelscale / oversample

        # Bar offset in arcsec
        bar_offset = self.bar_offset
        # In detector pixels
        bar_offpix = bar_offset  / self.pixelscale

        # Determine final shift amounts to location along bar
        # Shift to position relative to center of image
        if (('FULL' in self.det_info['wind_mode']) and (self.image_mask is not None)) or self._use_ap_info:
            xcen, ycen = self.siaf_ap.reference_point('sci')
            delx_pix = (xcen - (xpix/2 + 0.5))  # 'sci' pixel shifts
            dely_pix = (ycen - (ypix/2 + 0.5))  # 'sci' pixel shifts
            # Convert to 'idl' offsets
            # delx_asec, dely_asec = self.siaf_ap.convert(xcen+delx_pix, ycen+dely_pix, 'sci', 'idl')
            delx_asec, dely_asec = np.array([delx_pix, dely_pix]) * self.pixelscale
        else:
            # Otherwise assumed mask is in center of subarray for simplicity
            # The 0.5 offset indicates PSF is centered in middle of pixel
            # TODO: How does this interact w/ odd/even PSFs??
            xcen, ycen = (xpix/2 + 0.5, ypix/2 + 0.5)
            # Add bar offset
            xcen += bar_offpix  # Add bar offset
            delx_pix, dely_pix = (bar_offpix, 0)
            delx_asec = delx_pix * self.pixelscale
            dely_asec = dely_pix * self.pixelscale

        # Add dither offsets
        offx_asec, offy_asec = xyoff_asec  # 'idl' dither offsets
        delx_asec += offx_asec 
        dely_asec += offy_asec 

        # Expand to oversampled array
        hdul_disk = deepcopy(self.disk_hdulist)
        # Final size of subarray
        out_shape = np.array([ypix*oversample, xpix*oversample], dtype='int')
        # Extend to accommodate rotations and shifts
        extend = int(np.sqrt(2)*np.max(self.disk_hdulist[0].data.shape))
        oversized_shape = out_shape + extend
        # But don't make smaller than current size
        orig_shape = hdul_disk[0].data.shape
        new_shape = np.array([oversized_shape,orig_shape]).max(axis=0)
        hdul_disk[0].data = pad_or_cut_to_size(hdul_disk[0].data, new_shape)

        ##################################
        # Shift/Rotate image
        ##################################
        # Rotate and shift oversampled disk image
        # Positive PA_offset will rotate image clockwise (PA_offset = -1*PA_V3)
        hdul_rot_shift = rotate_shift_image(hdul_disk, angle=PA_offset, order=1,
                                            delx_asec=delx_asec, dely_asec=dely_asec)
        hdul_rot_shift[0].data = pad_or_cut_to_size(hdul_rot_shift[0].data, out_shape)
        xref, yref = xcen, ycen
        hdul_rot_shift[0].header['XIND_REF'] = (xref*oversample, "x index of aperture reference")
        hdul_rot_shift[0].header['YIND_REF'] = (yref*oversample, "y index of aperture reference")
        hdul_rot_shift[0].header['CFRAME'] = 'sci'

        hdul_disk.close()
        del hdul_disk

        ##################################
        # Image distortion
        ##################################
        # Crop to reasonably small size (get rid of 0s)
        res = crop_zero_rows_cols(hdul_rot_shift[0].data, symmetric=False, return_indices=True)
        im_crop, ixy_vals = res
        ix1, ix2, iy1, iy2 = ixy_vals
        # Get 'sci' pixel values that correspond to center of input image
        xarr = (np.arange(hdul_rot_shift[0].data.shape[1]) + 1) / oversample
        yarr = (np.arange(hdul_rot_shift[0].data.shape[0]) + 1) / oversample
        cen_sci = (np.mean(xarr[ix1:ix2]), np.mean(yarr[iy1:iy2]))

        im_orig = hdul_rot_shift[0].data
        hdul_rot_shift[0].data = im_crop
        im_distort = distort_image(hdul_rot_shift, aper=self.siaf_ap, sci_cen=cen_sci)
        im_orig[iy1:iy2, ix1:ix2] = im_distort
        hdul_rot_shift[0].data = im_orig

        ##################################
        # Mask attenuation
        ##################################
        # Multiply raw disk data by coronagraphic mask.
        # Exclude region already affected by observed mask.
        # Mostly ND Squares and opaque COM holder.
        # If ND_acq, then disk already multipled by ND throughput in bandpass.
        cmask = self.mask_images['OVERSAMP']
        if use_cmask and (cmask is not None) and (not self.ND_acq):
            hdul_rot_shift[0].data = self.attenuate_with_mask(hdul_rot_shift[0].data, cmask=cmask)

        ##################################
        # Image convolution
        ##################################

        hdul_psfs = self.psf_list
        if not self.is_coron: # Single PSF
            image_conv = convolve_image(hdul_rot_shift, hdul_psfs)
        else:
            # For coronagraphy, assume position-dependent PSFs are a function
            # of coronagraphic mask transmission, off-axis, and on-axis PSFs.
            # PSF(x,y) = trans(x,y)*psf_off + (1-trans(x,y))*psf_on
            trans = self.mask_images['OVERMASK']
            
            # Off-axis component
            hdul_rot_shift_off = deepcopy(hdul_rot_shift)
            hdul_rot_shift_off[0].data = hdul_rot_shift_off[0].data * trans
            psf_off = self.calc_psf_from_coeff(return_oversample=True, return_hdul=True,  
                                               coord_vals=(10,10), coord_frame='idl')
            image_conv_off = convolve_image(hdul_rot_shift_off, psf_off)

            # On-axis component (closest PSF convolution)
            hdul_rot_shift_on = deepcopy(hdul_rot_shift)
            hdul_rot_shift_on[0].data = hdul_rot_shift_on[0].data * (1 - trans)
            image_conv_on = convolve_image(hdul_rot_shift_on, hdul_psfs)
            
            image_conv = image_conv_on + image_conv_off

        if return_oversample:
            return image_conv
        else:
            return frebin(image_conv, scale=1/self.oversample)

    def gen_slope_image(self, PA=0, xyoff_asec=(0,0), return_oversample=False,
        exclude_disk=False, exclude_planets=False, exclude_noise=False, 
        zfact=None, do_ref=False, do_roll2=False, im_star=None, sat_val=0.9,
        wfe_drift0=0, wfe_ref_drift=None, wfe_roll_drift=None, **kwargs):
        """Create slope image of observation
        
        Beware that stellar position (centered on a pixel) will likely not
        fall in the exact center of the slope image (between pixel borders)
        because images are generally even while psf_fovs may be odd.

        Parameters
        ----------
        PA : float
            Position angle of roll position (counter-clockwise, from West to East).
            Scence will rotate in opposite direction.
        xyoff_asec : 
            Positional offset of scene from reference location in 'idl' coords.
            Shift occurs after PA rotation.
        do_ref : bool
            Slope image for reference star observation using `self.wfe_ref_drift`.
        do_roll2 : bool
            Slope image for observation during Roll2 using `self.wfe_roll_drift`.
            Will compound with `do_ref` if both are set.
        exclude_disk : bool
            Do not include disk in final image (for radial contrast),
            but still add Poisson noise from disk.
        exclude_planets : bool
            Do not include planets in final image (for radial contrast),
            but still add Poisson noise from disk.
        exclude_noise : bool
            Don't add random Gaussian noise (detector+photon)
        zfact : float
            Zodiacal background factor (default=2.5)
        im_star : ndarray or None
            Pass a precomputed slope image of the stellar source already
            positioned at it's correct location.
        wfe_drift0 : float
            Initial RMS WFE drift value of First PSF. Roll2 and Ref observations
            will be incremented from this. Default is 0.
        wfe_ref_drift : float
            WFE drift between sci and ref observations (nm RMS).
        wfe_roll_drift : float
            WFE drift between Roll1 and Roll2 observations (nm RMS).
        return_oversample : bool
            Return either the detector pixel-sampled or oversampled image.

        Keyword Args
        ------------
        use_cmask : bool
            Use the coronagraphic mask image to attenuate planet or disk that
            is obscurred by a corongraphic mask feature. (Default=False)
        ra : float
            Right ascension in decimal degrees
        dec : float
            Declination in decimal degrees
        thisday : int
            Calendar day to use for background calculation.  If not given, will use the
            average of visible calendar days.
        ideal_Poisson : bool
            If set to True, use total signal for noise estimate,
            otherwise MULTIACCUM equation is used.
        use_coeff : bool
            If True, uses `calc_psf_from_coeff`, other WebbPSF's built-in `calc_psf`
            for stellar sources.
        """

        # Initial WFE drift offset value
        wfe_drift = wfe_drift0
        # Option to override wfe_ref_drift and wfe_roll_drift
        wfe_ref_drift  = self.wfe_ref_drift  if wfe_ref_drift  is None else wfe_ref_drift
        wfe_roll_drift = self.wfe_roll_drift if wfe_roll_drift is None else wfe_roll_drift

        if do_ref: 
            wfe_drift = wfe_drift + wfe_ref_drift
            det = self.Detector_ref
            sp = self.sp_ref
        else:
            det = self.Detector
            sp = self.sp_sci

        # Add additional WFE drift for 2nd roll position
        if do_roll2: 
            wfe_drift = wfe_drift + wfe_roll_drift

        oversample = 1 if not return_oversample else self.oversample 
        pixscale_over = self.pixelscale / oversample

        # Final detector image shape
        ypix, xpix = (det.ypix, det.xpix)
        # Oversampled size
        xpix_over = xpix * oversample
        ypix_over = ypix * oversample

        # Bar offset in arcsec
        bar_offset = self.bar_offset
        # In detector pixels
        bar_offpix = bar_offset  / self.pixelscale

        # Additional field offsets
        if xyoff_asec is None:
            if do_ref:
                xyoff_asec = self.pointing_info['ref']
            elif do_roll2:
                xyoff_asec = self.pointing_info['roll2']
            else:
                xyoff_asec = self.pointing_info['roll1']

        offx_asec, offy_asec = xyoff_asec  # 'idl' dither offsets

        ##################################
        # Generate Image
        ##################################

        # Default to use coronagraph mask ND square and holder attenuation
        use_cmask = kwargs.get('use_cmask')
        if use_cmask is None:
            kwargs['use_cmask'] = True

        # Get (r,th) in idl coordinates relative to mask center (arcsec) for PSF creation
        xoff_idl, yoff_idl = (offx_asec + bar_offset, offy_asec)
        r, th = xy_to_rtheta(xoff_idl, yoff_idl)

        # Stellar PSF doesn't rotate
        if im_star is None:
            _log.info('  gen_slope_image: Creating stellar PSF...')
            im_star = self.gen_offset_psf(r, th, sp=sp, wfe_drift=wfe_drift,
                                          return_oversample=return_oversample, **kwargs)
        elif isinstance(im_star, (int,float)) and (im_star==0):
            im_star = np.zeros([ypix_over, xpix_over])

        # Expand to full size
        im_star = pad_or_cut_to_size(im_star, (ypix_over, xpix_over))

        ##################################
        # Shift image
        ##################################

        # Determine final shift amounts to mask location
        # Shift to position relative to center of image
        if (('FULL' in self.det_info['wind_mode']) and (self.image_mask is not None)) or self._use_ap_info:
            xcen, ycen = self.siaf_ap.reference_point('sci')
            delx_pix = (xcen - (xpix/2 + 0.5))  # 'sci' pixel shifts
            dely_pix = (ycen - (ypix/2 + 0.5))  # 'sci' pixel shifts
            delx_asec, dely_asec = np.array([delx_pix, dely_pix]) * self.pixelscale
        else:
            # Otherwise assumed mask is already in center of subarray
            delx_pix, dely_pix = (bar_offpix, 0)
            delx_asec, dely_asec = np.array([delx_pix, dely_pix]) * self.pixelscale

        # Add dither offsets
        delx_asec += offx_asec 
        dely_asec += offy_asec 

        # PSF shifting
        delx_over, dely_over = np.array([delx_asec, dely_asec]) / pixscale_over
        delx_det, dely_det = np.array([delx_asec, dely_asec]) / self.pixelscale

        _log.debug(f'r, th = ({r:.4f} asec, {th:.1f} deg)')
        _log.debug(f'delx, dely = ({delx_asec:.4f}, {dely_asec:.4f}) asec')
        _log.debug(f'delx, dely = ({delx_det:.2f}, {dely_det:.2f}) det pixels')

        # print(f'r, th = ({r:.4f} asec, {th:.1f} deg)')
        # print(f'delx, dely = ({delx_asec:.4f}, {dely_asec:.4f}) asec')
        # print(f'delx, dely = ({delx_det:.2f}, {dely_det:.2f}) det pixels')
        
        # Stellar PSF doesn't rotate
        interp = 'linear' if ('FULL' in self.det_info['wind_mode']) else 'cubic'
        im_star = fshift(im_star, delx=delx_over, dely=dely_over, pad=True, interp=interp)
        im_star[im_star<0] = 0

        ##################################
        # Disk and Planet images
        ##################################

        if do_ref:
            no_disk = no_planets = True
        else:
            no_disk    = exclude_disk    and exclude_noise
            no_planets = exclude_planets and exclude_noise

        # Make sure to include planets and disks for Poisson noise calculations
        # Telescope PA is counter-clockwise, therefore image rotation is opposite direction
        kwargs2 = kwargs.copy()
        kwargs2['PA_offset']  = -1*PA
        kwargs2['xyoff_asec'] = xyoff_asec
        kwargs2['return_oversample'] = return_oversample
        kwargs2['use_coeff'] = True
        # Companions
        if no_planets:
            im_pl = 0
        else:
            _log.info('  gen_slope_image: Creating companion image...')
            im_pl = self.gen_planets_image(**kwargs2)

        # Extended disk structures
        if no_disk:
            im_disk = 0
        else:
            _log.info('  gen_slope_image: Creating disk image...')
            im_disk = self.gen_disk_image(**kwargs2)

        # Zodiacal bg levels
        _log.info('  gen_slope_image: Creating zodiacal background image...')
        fzodi = self.bg_zodi_image(zfact=zfact, **kwargs)
        if oversample!=1:
            fzodi = frebin(fzodi, scale=oversample)

        # Combine components
        im_final_over = im_star + im_disk + im_pl + fzodi

        # Noise per detector pixel
        if not exclude_noise:
            _log.info('  gen_slope_image: Adding noise...')

            # Rebin to detector sampled
            im_final = frebin(im_final_over, scale=1/oversample)

            # For each pixel, how many groups until saturation?
            ng_sat = sat_val * self.well_level / (im_final * det.time_group)
            # Cap ng_sat to ngroup
            ngroup = det.multiaccum.ngroup
            ng_sat[ng_sat > ngroup] = ngroup
            ng_sat = ng_sat.astype('int')

            im_noise = det.pixel_noise(fsrc=im_final, ng=ng_sat, **kwargs)
            # Fix any values due to ng<1
            ind_fix = (np.isnan(im_noise)) | (ng_sat < 1)
            if np.sum(ind_fix)>0:
                im_noise[ind_fix] = det.pixel_noise(fsrc=im_final[ind_fix], ng=1, nf=1, **kwargs)

            im_noise_over = np.sqrt(frebin(im_noise**2, scale=oversample))

            # Add random Gaussian noise
            im_final_over += np.random.normal(scale=im_noise_over)

        # Get rid of disk and planet emission
        # while keeping their noise contributions
        if exclude_disk:    im_final_over -= im_disk
        if exclude_planets: im_final_over -= im_pl

        return im_final_over


    def star_flux(self, fluxunit='counts', do_ref=False, sp=None):
        """ Stellar flux

        Return the stellar flux in pysynphot-supported units, such as
        vegamag, counts, or Jy.

        Parameters
        ----------
        fluxunits : str
            Desired output units, such as counts, vegamag, Jy, etc.
            Must be a Pysynphot supported unit string.
        sp : :mod:`pysynphot.spectrum`
            Normalized Pysynphot spectrum.
        """

        if sp is None:
            sp = self.sp_ref if do_ref else self.sp_sci

        # Create pysynphot observation
        bp = self.bandpass
        obs = S.Observation(sp, bp, binset=bp.wave)

        return obs.effstim(fluxunit)

    def _fix_sat_im(self, image, sat_val=0.9, oversample=1, **kwargs):
        """Fix saturated region of an image


        Parameters
        ----------
        image : ndarray
            Image to clean.
        sav_val : float
            Well level fraction to considered saturated.

        Keyword Args
        ------------
        full_size : bool
            Expand (or contract) to size of detector array?
            If False, returned image is fov_pix size.
        ngroup : int
            How many group times to determine saturation level?
            If this number is higher than the total groups in ramp,
            then a warning is produced. The default is ngroup=2,
            A value of 0 corresponds to the so-called "zero-frame,"
            which is the very first frame that is read-out and saved
            separately. This is the equivalent to ngroup=1 for RAPID
            and BRIGHT1 observations.
        do_ref : bool
            Get saturation levels assuming reference observation.
        niter_max : int
            Number of iterations for fixing NaNs. Default=5.
        """

        # Account for possible oversampling
        if oversample>1:
            image_det = frebin(image, scale=1/oversample)
            sat_level = self.saturation_levels(image=image_det, **kwargs)
            sat_level = frebin(sat_level, scale=oversample, total=False)
        else:
            sat_level = self.saturation_levels(image=image, **kwargs)
        sat_mask = sat_level > sat_val
        image[sat_mask] = np.nan
        image = fix_nans_with_med(image, **kwargs)

        # If there are any leftover NaNs, make them 0.
        nan_mask = np.isnan(image)
        image[nan_mask] = 0

        return image

    def gen_roll_image(self, PA1=0, PA2=10, return_oversample=False,
        no_ref=False, opt_diff=True, fix_sat=False, ref_scale_all=False, 
        wfe_drift0=0, wfe_ref_drift=None, wfe_roll_drift=None, 
        xyoff_roll1=None, xyoff_roll2=None, xyoff_ref=None, **kwargs):
        """Make roll-subtracted image.

        Create a final roll-subtracted slope image based on current observation
        settings. Coordinate convention is for N up and E to left.

        Procedure:

        - Create Roll 1 and Roll 2 slope images (star+exoplanets+disk)
        - Create Reference Star slope image
        - Add noise to all images
        - Scale ref image
        - Subtract ref image from both rolls
        - De-rotate Roll 1 and Roll 2 to common coordinates
        - Average Roll 1 and Roll 2

        Returns an HDUList of final image (N rotated upwards).

        Parameters
        ----------
        PA1 : float
            Position angle of first telescope roll position.
            This is counter-clockwise (West to East).
        PA2 : float, None
            Position angle of second roll position. If set equal to PA1
            (or to None), then only one roll will be performed.
            Otherwise, two rolls are performed, each using the specified
            MULTIACCUM settings (doubling the effective exposure time).
        no_ref : bool
            Exclude reference observation. Subtraction is then Roll1-Roll2.
        opt_diff : bool
            Optimal reference differencing (scaling only on the inner regions)
        fix_sat : bool
            Calculate saturated regions and fix with median of nearby data.
        ref_scale_all : bool
            Normally we just use the science and reference PSFs to calculate
            scaling. However, if there is an unresolved companion or disk
            emission close to the star, then we won't get the correct scale
            factor for optimal reference subtraction. Instead, this option
            inludes disk and companions for calculating the reference scale
            factor.
        wfe_drift0 : float
            Initial RMS WFE drift value of First PSF. Roll2 and Ref observations
            will be incremented from this. Default is 0.
        wfe_ref_drift : float
            WFE drift between sci and ref observations (nm RMS).
        wfe_roll_drift : float
            WFE drift between Roll1 and Roll2 observations (nm RMS).
        return_oversample : bool
            Return either the detector pixel-sampled or oversampled image.
        xyoff_roll1 : tuple or None
            Explicitly set pointing offset for Roll 1 (arcsec)
        xyoff_roll2 : tuple or None
            Explicitly set pointing offset for Roll 2 (arcsec)
        xyoff_ref : tuple or None
            Explicitly set pointing offset for Reference (arcsec)

        Keyword Args
        ------------
        exclude_disk : bool
            Do not include disk in final image (for radial contrast),
            but still add Poisson noise from disk.
        exclude_planets : bool
            Do not include planets in final image (for radial contrast),
            but still add Poisson noise from disk.
        exclude_noise : bool
            Don't add random Gaussian noise (detector+photon)
        use_coeff : bool
            If True, uses `calc_psf_from_coeff`, other WebbPSF's built-in `calc_psf`
            for stellar sources.
        ideal_Poisson : bool
            If set to True, use total signal for noise estimate,
            otherwise MULTIACCUM equation is used.
        use_cmask : bool
            Use the coronagraphic mask image to attenuate planets or disk
            obscurred by a corongraphic mask feature. Default is True.
        zfact : float
            Zodiacal background factor (default=2.5)
        ra : float
            Right ascension in decimal degrees
        dec : float
            Declination in decimal degrees
        thisday : int
            Calendar day to use for background calculation.  If not given, will use the
            average of visible calendar days.

        """

        # Final image shape
        xpix, ypix = (self.det_info['xpix'], self.det_info['ypix'])
        oversample = self.oversample
        pixscale_over = self.pixelscale / oversample

        if return_oversample:
            pixscale_out = pixscale_over
            osamp_out = oversample
        else:
            pixscale_out = self.pixelscale
            osamp_out = 1

        # Default to use coronagraph mask ND square and holder attenuation
        use_cmask = kwargs.get('use_cmask')
        if use_cmask is None:
            kwargs['use_cmask'] = True

        # Sub-image for determining ref star scale factor
        subsize = 50 * oversample
        xsub = np.min([subsize,xpix])
        ysub = np.min([subsize,ypix])
        sub_shape = (ysub, xsub)

        # Option to override wfe_ref_drift and wfe_roll_drift
        wfe_ref_drift  = self.wfe_ref_drift  if wfe_ref_drift  is None else wfe_ref_drift
        wfe_roll_drift = self.wfe_roll_drift if wfe_roll_drift is None else wfe_roll_drift

        # Position angle decisions
        if PA2 is None:
            roll_angle = 0
        else:
            roll_angle = PA2 - PA1
        xpix_over = xpix * oversample
        ypix_over = ypix * oversample

        # Bar offset in arcsec
        bar_offset = self.bar_offset
        bar_offpix = bar_offset  / self.pixelscale

        # Additional field offsets
        xyoff_asec1    = self.pointing_info.get('roll1', (0,0)) if xyoff_roll1 is None else xyoff_roll1
        xyoff_asec2    = self.pointing_info.get('roll2', (0,0)) if xyoff_roll2 is None else xyoff_roll2
        xyoff_asec_ref = self.pointing_info.get('ref',   (0,0)) if xyoff_ref   is None else xyoff_ref
        xyoff_asec1    = np.asarray(xyoff_asec1)
        xyoff_asec2    = np.asarray(xyoff_asec2)
        xyoff_asec_ref = np.asarray(xyoff_asec_ref)

        ##################################
        # Generate Roll1 Image
        ##################################

        # Add in offsets for PSF generation
        offx_asec, offy_asec = xyoff_asec1
        r, th = xy_to_rtheta(offx_asec + bar_offset, offy_asec)

        # Create stellar PSF centered in image
        im_star = self.gen_offset_psf(r, th, sp=self.sp_sci, return_oversample=True, 
                                      wfe_drift=wfe_drift0, **kwargs)
        # Stellar cut-out for reference scaling
        im_star_sub = pad_or_cut_to_size(im_star, sub_shape)
        # Expand to full size
        im_star = pad_or_cut_to_size(im_star, (ypix_over, xpix_over))

        ###################################
        # Get image shifts to mask location
        ###################################

        # Shift to position relative to center of image
        if (('FULL' in self.det_info['wind_mode']) and (self.image_mask is not None)) or self._use_ap_info:
            xcen, ycen = self.siaf_ap.reference_point('sci')
            delx_pix = (xcen - (xpix/2 + 0.5))  # 'sci' pixel shifts
            dely_pix = (ycen - (ypix/2 + 0.5))  # 'sci' pixel shifts
            delx_asec, dely_asec = np.array([delx_pix, dely_pix]) * self.pixelscale
            xcen_baroff = xcen  # Use SIAF aperture location
        else:
            # Otherwise assumed mask is in center of subarray for simplicity
            # The 0.5 offset indicates PSF is centered in middle of pixel
            # TODO: How does this interact w/ odd/even PSFs??
            xcen, ycen = (xpix/2 + 0.5, ypix/2 + 0.5)
            xcen_baroff = xcen + bar_offpix  # Include bar offset position
            # Add bar offset
            xcen += bar_offpix
            delx_pix, dely_pix = (bar_offpix, 0)
            delx_asec, dely_asec = np.array([delx_pix, dely_pix]) * self.pixelscale

        # Create cen_over parameter to pass to de-rotate function
        xcen_over, ycen_over = np.array([xcen_baroff, ycen]) * oversample
        cen_over = (xcen_over, ycen_over)

        # Perform shift and create slope image
        interp = 'linear' if ('FULL' in self.det_info['wind_mode']) else 'cubic'
        #im_star = fshift(im_star, delx=delx_over, dely=dely_over, pad=True, interp=interp)
        im_roll1 = self.gen_slope_image(PA=PA1, xyoff_asec=xyoff_asec1, im_star=im_star, 
                                        return_oversample=True, **kwargs)

        if no_ref and (roll_angle==0):
            _log.warning('If no_ref=True, then PA1 must not equal PA2. Setting no_ref=False')
            no_ref = False

        # Include disk and companion flux for calculating the reference scale factor
        # Use summed roll image; shift star to center of array and crop
        if ref_scale_all:
            # Get dither offsets
            offx_asec, offy_asec = xyoff_asec1
            delx_asec_dith = delx_asec + offx_asec 
            dely_asec_dith = dely_asec + offy_asec
            delx_over, dely_over = np.array([delx_asec_dith, dely_asec_dith]) / pixscale_over
            # Shift and crop star+disk+planets image
            im_star_sub = pad_or_cut_to_size(im_roll1, sub_shape, interp=interp,
                                             offset_vals=(-1*dely_over,-1*delx_over))

        # Fix saturated pixels
        if fix_sat:
            im_roll1 = self._fix_sat_im(im_roll1, oversample=oversample, **kwargs)

        ##################################################
        # Pure roll subtraction (no reference PSF)
        ##################################################
        if no_ref:
            # Create Roll2 image
            if (np.abs(wfe_roll_drift) < eps) and np.allclose(xyoff_asec1, xyoff_asec2):
                im_roll2 = self.gen_slope_image(PA=PA2, im_star=im_star, do_roll2=True, 
                                                return_oversample=True, **kwargs)
            else:
                im_roll2 = self.gen_slope_image(PA=PA2, xyoff_asec=xyoff_asec2, do_roll2=True, 
                                                wfe_drift0=wfe_drift0, wfe_roll_drift=wfe_roll_drift, 
                                                return_oversample=True, **kwargs)

            # Fix saturated pixels
            if fix_sat:
                im_roll2 = self._fix_sat_im(im_roll2, oversample=oversample, **kwargs)

            # if oversample>1:
            #     kernel = Gaussian2DKernel(0.5*oversample)
            #     im_roll1 = convolve_fft(im_roll1, kernel, allow_huge=True)
            #     im_roll2 = convolve_fft(im_roll2, kernel, allow_huge=True)

            # Shift roll images by dither/pointing offsets to align and subtract
            dx, dy = -1 * xyoff_asec1 / pixscale_over
            im_roll1_sh = fshift(im_roll1, delx=dx, dely=dy, interp=interp)
            dx, dy = -1 * xyoff_asec2 / pixscale_over
            im_roll2_sh = fshift(im_roll2, delx=dx, dely=dy, interp=interp)
            # Difference the two rolls
            diff_r1 = im_roll1_sh - im_roll2_sh
            diff_r2 = -1 * diff_r1

            # De-rotate each image around center of mask
            diff_r1_rot = rotate_offset(diff_r1, -PA1, cen=cen_over, reshape=True, cval=np.nan)
            diff_r2_rot = rotate_offset(diff_r2, -PA2, cen=cen_over, reshape=True, cval=np.nan)

            # Expand to the same size
            new_shape = tuple(np.max(np.array([diff_r1_rot.shape, diff_r2_rot.shape]), axis=0))
            diff_r1_rot = pad_or_cut_to_size(diff_r1_rot, new_shape, fill_val=np.nan)
            diff_r2_rot = pad_or_cut_to_size(diff_r2_rot, new_shape, fill_val=np.nan)

            # Replace NaNs with values from other differenced mask
            nan_mask = np.isnan(diff_r1_rot)
            diff_r1_rot[nan_mask] = diff_r2_rot[nan_mask]
            nan_mask = np.isnan(diff_r2_rot)
            diff_r2_rot[nan_mask] = diff_r1_rot[nan_mask]

            final = (diff_r1_rot + diff_r2_rot) / 2

            # Rebin if requesting detector sampled images 
            if not return_oversample:
                final = frebin(final, scale=1/oversample)
                im_roll1 = frebin(im_roll1, scale=1/oversample)
                im_roll2 = frebin(im_roll2, scale=1/oversample)

            hdu = fits.PrimaryHDU(final)
            hdu.header['EXTNAME'] = ('ROLL_SUB')
            hdu.header['OVERSAMP'] = (osamp_out, 'Oversample compared to detector pixels')
            hdu.header['OSAMP'] =    (osamp_out, 'Oversample compared to detector pixels')
            hdu.header['PIXELSCL'] = (pixscale_out, 'Image pixel scale (asec/pix)')
            hdu.header['FILTER']   = (self.filter, 'Filter name')
            if self.is_lyot:  
                hdu.header['PUPIL']    = (self.pupil_mask, 'Pupil plane mask')
            if self.is_coron: 
                hdu.header['CORONMSK'] = (self.image_mask, 'Image plane mask')
            hdu.header['TEXP_SCI'] = (2*self.multiaccum_times['t_exp'], 'Total science exposure time (sec)')
            hdu.header['TEXP_REF'] = (0, 'Total reference exposure time (sec)')
            hdu.header['ROLL_ANG'] = (roll_angle, 'Delta roll angle (deg)')

            hdulist = fits.HDUList([hdu])

            roll_names = ['ROLL1', 'ROLL2']
            pa_vals = [PA1, PA2]
            for ii, im in enumerate([im_roll1, im_roll2]):
                hdu = fits.ImageHDU(im)
                hdu.header['EXTNAME'] = (roll_names[ii])
                hdu.header['OVERSAMP'] = (osamp_out, 'Oversample compared to detector pixels')
                hdu.header['OSAMP'] =    (osamp_out, 'Oversample compared to detector pixels')
                hdu.header['PIXELSCL'] = (pixscale_out, 'Image pixel scale (asec/pix)')
                hdu.header['FILTER']   = (self.filter, 'Filter name')
                if self.is_lyot:  
                    hdu.header['PUPIL']    = (self.pupil_mask, 'Pupil plane mask')
                if self.is_coron: 
                    hdu.header['CORONMSK'] = (self.image_mask, 'Image plane mask')
                hdu.header['TEXP']     = (self.Detector.time_exp, 'Total exposure time (sec)')
                hdu.header['PA']       = (pa_vals[ii], "Position angle (deg)")
                dx, dy = xyoff_asec1 if ii==0 else xyoff_asec2
                hdu.header['DX_ASEC']  = (dx, 'Pointing offset in x-ideal (asec)')
                hdu.header['DY_ASEC']  = (dy, 'Pointing offset in y-ideal (asec)')

                hdulist.append(hdu)

            return hdulist


        ##################################################
        # Continuing with a ref PSF subtraction algorithm
        ##################################################

        # Reference star slope simulation
        wfe_drift_ref = wfe_drift0 + wfe_ref_drift

        # Add in offsets for PSF generation
        offx_asec, offy_asec = xyoff_asec_ref
        r, th = xy_to_rtheta(offx_asec + bar_offset, offy_asec)

        # Create stellar PSF centered in image
        im_ref = self.gen_offset_psf(r, th, sp=self.sp_ref, return_oversample=True, 
                                     wfe_drift=wfe_drift_ref, **kwargs)
        # Stellar cut-out for reference scaling
        im_ref_sub = pad_or_cut_to_size(im_ref, sub_shape)
        # Expand to full size
        im_ref = pad_or_cut_to_size(im_ref, (ypix_over, xpix_over))

        # Create Reference slope image
        # Essentially just adds image shifts and noise
        im_ref = self.gen_slope_image(PA=0, xyoff_asec=xyoff_asec_ref, im_star=im_ref, 
                                      do_ref=True, return_oversample=True, **kwargs)

        # Fix saturated pixels
        if fix_sat:
            im_ref = self._fix_sat_im(im_ref, do_ref=True, oversample=oversample, **kwargs)

        # Determine reference star scale factor
        scale1 = scale_ref_image(im_star_sub, im_ref_sub)
        _log.debug('scale1: {0:.3f}'.format(scale1))
        # print('scale1: {0:.3f}'.format(scale1), im_star_sub.sum(), im_ref_sub.sum())
        # if oversample>1:
        #     kernel = Gaussian2DKernel(0.5*oversample)
        #     im_ref = convolve_fft(im_ref, kernel, allow_huge=True)
        #     im_roll1 = convolve_fft(im_roll1, kernel, allow_huge=True)

        # Shift roll images by pointing offsets
        dx, dy = -1 * xyoff_asec1 / pixscale_over
        im_roll1_sh = fshift(im_roll1, delx=dx, dely=dy, interp=interp)
        dx, dy = -1 * xyoff_asec_ref / pixscale_over
        im_ref_sh = fshift(im_ref, delx=dx, dely=dy, interp=interp)
                    
        # Telescope Roll 2 with reference subtraction
        if (abs(roll_angle) > eps):
            # Subtraction with and without scaling
            im_diff1_r1 = im_roll1_sh - im_ref_sh
            im_diff2_r1 = im_roll1_sh - im_ref_sh * scale1

            # WFE drift difference between rolls
            wfe_drift1 = wfe_drift0
            wfe_drift2 = wfe_drift1 + wfe_roll_drift

            # Create roll2 image
            if (np.abs(wfe_roll_drift) < eps) and np.allclose(xyoff_asec1, xyoff_asec2):
                # Assume Roll2 and Roll1 have exactly the same position offset and WFE drift
                im_star2     = im_star
                im_star2_sub = im_star_sub
            else:

                ##################################
                # Generate Roll2 Image

                # Add in offsets for PSF generation
                offx_asec, offy_asec = xyoff_asec2
                r, th = xy_to_rtheta(offx_asec + bar_offset, offy_asec)

                # Create stellar PSF (Roll 2) centered in image
                im_star2 = self.gen_offset_psf(r, th, sp=self.sp_sci, return_oversample=True, 
                                               wfe_drift=wfe_drift2, **kwargs)
                im_star2_sub = pad_or_cut_to_size(im_star2, sub_shape)
                im_star2     = pad_or_cut_to_size(im_star2, (ypix_over, xpix_over))


            # Create Roll2 slope image
            im_roll2 = self.gen_slope_image(PA=PA2, xyoff_asec=xyoff_asec2, im_star=im_star2, 
                                            do_roll2=True, return_oversample=True, **kwargs)

            # Include disk and companion flux for calculating the reference scale factor
            if ref_scale_all:
                # Get dither offsets
                offx_asec, offy_asec = xyoff_asec2
                delx_asec_dith = delx_asec + offx_asec 
                dely_asec_dith = dely_asec + offy_asec
                delx_over, dely_over = np.array([delx_asec_dith, dely_asec_dith]) / pixscale_over
                # Shift and crop star+disk+planets image
                im_star2_sub = pad_or_cut_to_size(im_roll2, sub_shape, interp=interp,
                                                  offset_vals=(-1*dely_over,-1*delx_over))

            # Fix saturated pixels
            if fix_sat:
                im_roll2 = self._fix_sat_im(im_roll2, oversample=oversample, **kwargs)

            # Subtract reference star from Roll 2
            scale2 = scale_ref_image(im_star2_sub, im_ref_sub)
            _log.debug('scale2: {0:.3f}'.format(scale2))
            # print('scale2: {0:.3f}'.format(scale2), im_star2_sub.sum(), im_ref_sub.sum())
            # if oversample>1:
            #     kernel = Gaussian2DKernel(0.5*oversample)
            #     im_roll2 = convolve_fft(im_roll2, kernel, allow_huge=True)

            # Shift roll images by pointing offsets
            dx, dy = -1 * xyoff_asec2 / pixscale_over
            im_roll2_sh = fshift(im_roll2, delx=dx, dely=dy, interp=interp)
                
            # Subtraction with and without scaling
            im_diff1_r2 = im_roll2_sh - im_ref_sh
            im_diff2_r2 = im_roll2_sh - im_ref_sh * scale2
            #im_diff_r2 = optimal_difference(im_roll2, im_ref, scale2)

            # De-rotate each image
            diff1_r1_rot = rotate_offset(im_diff1_r1, -PA1, cen=cen_over, reshape=True, cval=np.nan)
            diff2_r1_rot = rotate_offset(im_diff2_r1, -PA1, cen=cen_over, reshape=True, cval=np.nan)
            diff1_r2_rot = rotate_offset(im_diff1_r2, -PA2, cen=cen_over, reshape=True, cval=np.nan)
            diff2_r2_rot = rotate_offset(im_diff2_r2, -PA2, cen=cen_over, reshape=True, cval=np.nan)

            # Expand all images to the same size
            new_shape = tuple(np.max(np.array([diff1_r1_rot.shape, diff1_r2_rot.shape]), axis=0))
            diff1_r1_rot = pad_or_cut_to_size(diff1_r1_rot, new_shape, np.nan)
            diff2_r1_rot = pad_or_cut_to_size(diff2_r1_rot, new_shape, np.nan)
            diff1_r2_rot = pad_or_cut_to_size(diff1_r2_rot, new_shape, np.nan)
            diff2_r2_rot = pad_or_cut_to_size(diff2_r2_rot, new_shape, np.nan)

            # Replace NaNs with values from other differenced mask
            nan_mask = np.isnan(diff1_r1_rot)
            diff1_r1_rot[nan_mask] = diff1_r2_rot[nan_mask]
            diff2_r1_rot[nan_mask] = diff2_r2_rot[nan_mask]
            nan_mask = np.isnan(diff1_r2_rot)
            diff1_r2_rot[nan_mask] = diff1_r1_rot[nan_mask]
            diff2_r2_rot[nan_mask] = diff2_r1_rot[nan_mask]

            # final1 has better noise in outer regions (background)
            # final2 has better noise in inner regions (PSF removal)
            final1 = (diff1_r1_rot + diff1_r2_rot) / 2
            final2 = (diff2_r1_rot + diff2_r2_rot) / 2

            if opt_diff:
                rho = dist_image(final1)
                binsize = 1
                bins = np.arange(rho.min(), rho.max() + binsize, binsize)

                nan_mask = np.isnan(final1) | np.isnan(final2)
                igroups, _, rr = hist_indices(rho[~nan_mask], bins, True)

                func_std = np.std #robust.medabsdev
                std1 = binned_statistic(igroups, final1[~nan_mask], func=func_std)
                std2 = binned_statistic(igroups, final2[~nan_mask], func=func_std)

                ibin_better1 = np.where(std1 < std2)[0]
                ibin_better2 = np.where(std2 < std1)[0]
                if len(ibin_better1) < len(ibin_better2):
                    # Get all pixel indices
                    if len(ibin_better1)>0:
                        ind_all = np.array([item for ibin in ibin_better1 for item in igroups[ibin]])
                        ind_all.sort()
                        temp = final2[~nan_mask]
                        temp[ind_all] = final1[~nan_mask][ind_all]
                        final2[~nan_mask] = temp
                    final = final2
                else:
                    if len(ibin_better2)>0:
                        ind_all = np.array([item for ibin in ibin_better2 for item in igroups[ibin]])
                        ind_all.sort()
                        temp = final1[~nan_mask]
                        temp[ind_all] = final2[~nan_mask][ind_all]
                        final1[~nan_mask] = temp
                    final = final1
            else:
                # Choose version that optmizes PSF subtraction
                final = final2

            texp_sci = 2 * self.multiaccum_times['t_exp']

        # For only a single roll
        else:
            # Optimal differencing (with scaling only on the inner regions)
            if opt_diff:
                final = optimal_difference(im_roll1_sh, im_ref_sh, scale1)
            else:
                final = im_roll1_sh - im_ref_sh * scale1

            final = rotate_offset(final, -PA1, cen=cen_over, reshape=True, cval=np.nan)
            texp_sci = self.multiaccum_times['t_exp']

        # Rebin if requesting detector sampled images
        if not return_oversample:
            final = frebin(final, scale=1/oversample)
            im_roll1 = frebin(im_roll1, scale=1/oversample)

        hdu = fits.PrimaryHDU(final)
        hdu.header['EXTNAME'] = ('REF_SUB')
        hdu.header['OVERSAMP'] = (osamp_out, 'Oversample compared to detector pixels')
        hdu.header['OSAMP'] =    (osamp_out, 'Oversample compared to detector pixels')
        hdu.header['PIXELSCL'] = (pixscale_out, 'Image pixel scale (asec/pix)')
        hdu.header['FILTER']   = (self.filter, 'Filter name')
        if self.is_lyot: 
            hdu.header['PUPIL'] = (self.pupil_mask, 'Pupil plane mask')
        if self.is_coron: 
            hdu.header['CORONMSK'] = (self.image_mask, 'Image plane mask')
        hdu.header['TEXP_SCI'] = (texp_sci, 'Total science exposure time (sec)')
        hdu.header['TEXP_REF'] = (self.Detector_ref.time_exp, 'Total reference exposure time (sec)')
        hdu.header['ROLL_ANG'] = (roll_angle, 'Delta roll angle (deg)')

        hdulist = fits.HDUList([hdu])

        # Add Roll1
        hdu = fits.ImageHDU(im_roll1)
        hdu.header['EXTNAME'] = ('ROLL1')
        hdu.header['OVERSAMP'] = (osamp_out, 'Oversample compared to detector pixels')
        hdu.header['OSAMP'] =    (osamp_out, 'Oversample compared to detector pixels')
        hdu.header['PIXELSCL'] = (pixscale_out, 'Image pixel scale (asec/pix)')
        hdu.header['FILTER']   = (self.filter, 'Filter name')
        if self.is_lyot:  
            hdu.header['PUPIL']    = (self.pupil_mask, 'Pupil plane mask')
        if self.is_coron: 
            hdu.header['CORONMSK'] = (self.image_mask, 'Image plane mask')
        hdu.header['TEXP']     = (self.Detector.time_exp, 'Total exposure time (sec)')
        hdu.header['PA']       = (PA1, "Position angle (deg)")
        hdu.header['DX_ASEC']  = (xyoff_asec1[0], 'Pointing offset in x-ideal (asec)')
        hdu.header['DY_ASEC']  = (xyoff_asec1[1], 'Pointing offset in y-ideal (asec)')
        hdulist.append(hdu)

        # Add Roll2
        try:
            if not return_oversample:
                im_roll2 = frebin(im_roll2, scale=1/oversample)
            hdu = fits.ImageHDU(im_roll2)
            hdu.header['EXTNAME'] = ('ROLL2')
            hdu.header['OVERSAMP'] = (osamp_out, 'Oversample compared to detector pixels')
            hdu.header['OSAMP']    = (osamp_out, 'Oversample compared to detector pixels')
            hdu.header['PIXELSCL'] = (pixscale_out, 'Image pixel scale (asec/pix)')
            hdu.header['FILTER']   = (self.filter, 'Filter name')
            if self.is_lyot:  
                hdu.header['PUPIL']    = (self.pupil_mask, 'Pupil plane mask')
            if self.is_coron: 
                hdu.header['CORONMSK'] = (self.image_mask, 'Image plane mask')
            hdu.header['TEXP']     = (self.Detector.time_exp, 'Total exposure time (sec)')
            hdu.header['PA']       = (PA2, "Position angle (deg)")
            hdu.header['DX_ASEC']  = (xyoff_asec2[0], 'Pointing offset in x-ideal (asec)')
            hdu.header['DY_ASEC']  = (xyoff_asec2[1], 'Pointing offset in y-ideal (asec)')
            hdulist.append(hdu)
        except:
            pass

        # Add Ref image
        try:
            if not return_oversample:
                im_ref = frebin(im_ref, scale=1/oversample)
            hdu = fits.ImageHDU(im_ref)
            hdu.header['EXTNAME'] = ('REF')
            hdu.header['OVERSAMP'] = (osamp_out, 'Oversample compared to detector pixels')
            hdu.header['OSAMP']    = (osamp_out, 'Oversample compared to detector pixels')
            hdu.header['PIXELSCL'] = (pixscale_out, 'Image pixel scale (asec/pix)')
            hdu.header['FILTER']   = (self.filter, 'Filter name')
            if self.is_lyot:  
                hdu.header['PUPIL']    = (self.pupil_mask, 'Pupil plane mask')
            if self.is_coron: 
                hdu.header['CORONMSK'] = (self.image_mask, 'Image plane mask')
            hdu.header['TEXP']     = (self.Detector_ref.time_exp, 'Total exposure time (sec)')
            hdu.header['PA']       = (0, "Position angle (deg)")
            hdu.header['DX_ASEC']  = (xyoff_asec_ref[0], 'Pointing offset in x-ideal (asec)')
            hdu.header['DY_ASEC']  = (xyoff_asec_ref[1], 'Pointing offset in y-ideal (asec)')
            hdulist.append(hdu)
        except:
            pass

        return hdulist

    def calc_contrast(self, hdu_diff=None, roll_angle=10, nsig=1,
        exclude_disk=True, exclude_planets=True, no_ref=False,
        wfe_drift0=0, wfe_ref_drift=None, wfe_roll_drift=None,
        func_std=np.std, **kwargs):
        """Create contrast curve.

        Generate n-sigma contrast curve for the current observation settings.
        Make sure that MULTIACCUM parameters are set for both the main
        class (``self.update_detectors()``) as well as the reference target
        class (``self.nrc_ref.update_detectors()``).

        Parameters
        ----------
        hdu_diff : HDUList
            Option to pass an already pre-made differenced image.

        roll_angle : float
            Telescope roll angle (deg) between two observations.
            If set to 0 or None, then only one roll will be performed.
            If value is >0, then two rolls are performed, each using the
            specified MULTIACCUM settings (doubling the effective exposure
            time).
        nsig  : float
            n-sigma contrast curve.
        exclude_disk : bool
            Ignore disk when generating image?
        exclude_planets : bool
            Ignore planets when generating image?
        no_ref : bool
            Exclude reference observation. Subtraction is then Roll1-Roll2.
        func_std : func
            The function to use for calculating the radial standard deviation.

        Keyword Args
        ------------
        zfact : float
            Zodiacal background factor (default=2.5)
        exclude_noise : bool
            Don't add random Gaussian noise (detector+photon)?
        ideal_Poisson : bool
            If set to True, use total signal for noise estimate,
            otherwise MULTIACCUM equation is used.
        opt_diff : bool
            Optimal reference differencing (scaling only on the inner regions)

        Returns
        -------
        tuple
            Three arrays in a tuple: the radius in arcsec, n-sigma contrast,
            and n-sigma magnitude sensitivity limit (vega mag).
        """
        from webbpsf_ext.webbpsf_ext_core import _nrc_coron_psf_sums, nrc_mask_trans

        if no_ref and (roll_angle==0):
            _log.warning('If no_ref=True, roll_angle must not equal 0. Setting no_ref=False')
            no_ref = False

        # If no HDUList is passed, then create one
        if hdu_diff is None:
            roll_angle = 0 if roll_angle is None else roll_angle
            PA1 = 0
            PA2 = None if abs(roll_angle) < eps else roll_angle
            hdu_diff = self.gen_roll_image(PA1=PA1, PA2=PA2, exclude_disk=exclude_disk,
                                           exclude_planets=exclude_planets, no_ref=no_ref,
                                           wfe_drift0=wfe_drift0, wfe_ref_drift=wfe_ref_drift,
                                           wfe_roll_drift=wfe_roll_drift, **kwargs)

        data = hdu_diff[0].data
        header = hdu_diff[0].header

        # Bin to detector-sampled data
        xpix, ypix = (self.det_info['xpix'], self.det_info['ypix'])
        pixscale = self.pixelscale

        # Radial noise binned to detector pixels
        rr, stds = radial_std(data, pixscale=header['PIXELSCL'], oversample=header['OSAMP'], 
                              supersample=False, func=func_std)

        interp = 'linear' if ('FULL' in self.det_info['wind_mode']) else 'cubic'
        # Normalize by psf max value
        if no_ref:
            # No reference image subtraction; pure roll subtraction
            # Generate 2 PSFs separated by roll angle to find self-subtracted PSF peak

            off_vals = []
            max_vals = []
            rvals_pix = np.insert(np.arange(1,xpix/2,5), 0, 0.1)
            for roff_pix in rvals_pix:
                roff_asec = roff_pix * pixscale
                psf1 = self.gen_offset_psf(roff_asec, 0, return_oversample=False, 
                                           coron_rescale=True)
                psf2 = self.gen_offset_psf(roff_asec, roll_angle, return_oversample=False, 
                                           coron_rescale=True)

                psf1 = fshift(psf1, delx=0, dely=roff_pix, pad=False, interp=interp)
                xoff, yoff = xy_rot(0, roff_pix, 10)
                psf2 = fshift(psf2, delx=xoff, dely=yoff, pad=False, interp=interp)

                diff = psf1 - psf2
                maxv = np.max(diff)

                off_vals.append(roff_pix)
                max_vals.append(maxv)
                if maxv >= 0.95*psf1.max():
                    off_vals = off_vals + [roff_pix+5, xpix/2]
                    max_vals = max_vals + [psf1.max(), psf1.max()]
                    break

            max_vals = np.array(max_vals)
            off_asec = np.array(off_vals) * pixscale

            # Interpolate in log space
            psf_max_log = np.interp(rr, off_asec, np.log10(max_vals))
            psf_max = 10**psf_max_log

        elif not self.is_coron: # Direct imaging
            psf = self.calc_psf_from_coeff(return_oversample=False, return_hdul=False)
            psf_max = psf.max()

        elif self.image_mask[-1]=='R': # Round masks
            ny, nx = (ypix, xpix)
            yv = (np.arange(ny) - ny/2) * pixscale
            xv = np.zeros_like(yv)

            # Get mask transmission at selected points
            trans = nrc_mask_trans(self.image_mask, xv, yv)
            # Linear combination of min/max to determine PSF max value at given distance
            # Get a and b values for each position
            avals = trans**2
            bvals = 1 - avals

            # Init if _psf_sums dict doesn't exist
            try:
                _ = self._psf_sums['psf_off_max']
            except:
                _ = _nrc_coron_psf_sums(self, (0,0), 'idl')
            psf_off_max = self._psf_sums['psf_off_max']
            psf_cen_max = self._psf_sums['psf_cen_max']

            # Linear combination
            psf_max = avals * psf_off_max + bvals * psf_cen_max
            # Interpolate values at rr locations
            psf_max = 10**np.interp(rr, yv, np.log10(psf_max))
            # Fix anything outside of bounds
            if rr.max()>10:
                psf_max[rr>10] = psf_max[(rr>4.5) & (rr<10)].max()

        elif self.image_mask[-1]=='B': # Bar masks
            # For off-axis PSF max values, use fiducial at bar_offset location
            bar_offset = self.bar_offset
            ny, nx = (ypix, xpix)

            # Get mask transmission for grid of points
            yv = (np.arange(ny) - ny/2) * pixscale
            xv = np.ones_like(yv) * bar_offset
            trans = nrc_mask_trans(self.image_mask, xv, yv)
            # Linear combination of min/max to determine PSF max value at given distance
            # Get a and b values for each position
            avals = trans**2
            bvals = 1 - avals

            # Init if _psf_sums dict doesn't exist
            try:
                _ = self._psf_sums['psf_off_max']
            except:
                _ = _nrc_coron_psf_sums(self, (0,0), 'idl')

            psf_off_max = self._psf_sums['psf_off_max']
            # Linear interpolate psf center max value
            xvals = self._psf_sums['psf_cen_xvals']
            psf_cen_max_arr = self._psf_sums['psf_cen_max_arr']
            finterp = interp1d(xvals, psf_cen_max_arr, kind='linear', fill_value='extrapolate')
            psf_cen_max = finterp(bar_offset)

            # Linear combination
            psf_max = avals * psf_off_max + bvals * psf_cen_max
            # Interpolate values at rr locations
            psf_max = 10**np.interp(rr, yv, np.log10(psf_max))
            # Fix anything outside of bounds
            if rr.max()>10:
                psf_max[rr>10] = np.max(psf_max[(rr>4.5) & (rr<10)])

        # We also want to know the Poisson noise for the PSF values.
        # For instance, even if psf_max is significantly above the
        # background noise, Poisson noise could conspire to reduce
        # the peak PSF value below the noise level.

        # Count rate necessary to obtain some nsig
        texp  = self.multiaccum_times['t_exp']
        p     = 1 / texp
        crate = (p*nsig**2 + nsig * np.sqrt((p*nsig)**2 + 4*stds**2)) / 2
        # Get total count rate
        crate /= psf_max

        # Compute contrast
        contrast = crate / self.star_flux()

        # Magnitude sensitivity
        star_mag = self.star_flux('vegamag')
        sen_mag = star_mag - 2.5*np.log10(contrast)

        return (rr, contrast, sen_mag)

    def saturation_levels(self, ngroup=2, do_ref=False, image=None, **kwargs):
        """Saturation levels

        Create image showing level of saturation for each pixel.
        Saturation at different number of groups is possible with
        ngroup keyword. Returns an array the same shape as `det_info`
        [ypix,xpix] properties.

        Parameters
        ----------
        ngroup : int
            How many group times to determine saturation level?
            If this number is higher than the total groups in ramp,
            then a warning is produced. The default is ngroup=2,
            A value of 0 corresponds to the so-called "zero-frame,"
            which is the very first frame that is read-out and saved
            separately. This is the equivalent to ngroup=1 for RAPID
            and BRIGHT1 observations.
        do_ref : bool
            Get saturation levels for reference soure instead of science
        image : ndarray
            Rather than generating an image on the fly, pass a pre-computed
            slope image.

        Keyword Args
        ------------
        exclude_disk : bool
            Do not include disk in final image (for radial contrast),
            but still add Poisson noise from disk.
        exclude_planets : bool
            Do not include planets in final image (for radial contrast),
            but still add Poisson noise from disk.
        use_cmask : bool
            Use the coronagraphic mask image to attenuate planet or disk that
            is obscurred by a corongraphic mask feature.
        zfact : float
            Zodiacal background factor (default=2.5)
        ra : float
            Right ascension in decimal degrees
        dec : float
            Declination in decimal degrees
        thisday : int
            Calendar day to use for background calculation.  If not given, will use the
            average of visible calendar days.

        """

        assert ngroup >= 0

        if do_ref: 
            det = self.Detector_ref
        else:
            det = self.Detector
        ma = det.multiaccum
        multiaccum_times = det.times_to_dict()

        if ngroup > ma.ngroup:
            _log.warning("Specified ngroup is greater than self.det_info['ngroup'].")

        t_frame = multiaccum_times['t_frame']
        if ngroup==0:
            t_sat = t_frame
        else:
            nf = ma.nf; nd1 = ma.nd1; nd2 = ma.nd2
            t_sat = (nd1 + ngroup*nf + (ngroup-1)*nd2) * t_frame

        # Slope image of input source
        if image is None:
            image = self.gen_slope_image(do_ref=do_ref, exclude_noise=True, **kwargs)

        # Well levels after "saturation time"
        sat_level = image * t_sat / self.well_level

        return sat_level


def _gen_disk_hdulist(inst, file, pixscale, dist, wavelength, units, cen_star, 
    sp_star=None, dist_out=None, shape_out=None):
    """Create a correctly scaled disk model image.

    Rescale disk model flux to current pixel scale and distance.
    If instrument bandpass is different from disk model, scales 
    flux assuming a grey scattering model. 

    Result (in photons/sec) is saved in self.disk_hdulist attribute.
    """

    disk_params = {
        'file'       : file,
        'pixscale'   : pixscale,
        'dist'       : dist,
        'wavelength' : wavelength,
        'units'      : units,
        'cen_star'   : cen_star,
    }

    xpix = inst.det_info['xpix']
    ypix = inst.det_info['ypix']
    oversample = inst.oversample
    pixscale_out = inst.pixelscale / oversample

    # TODO: Double-check 'APERNAME', 'DET_X', 'DET_Y', 'DET_V2', 'DET_V3'
    # correspond to the desired observation
    disk_hdul = make_disk_image(inst, disk_params, sp_star=sp_star, pixscale_out=pixscale_out,
                                dist_out=dist_out, shape_out=shape_out)

    # Get rid of the central star flux
    # and anything interior to a few pixels
    image = disk_hdul[0].data
    image_rho = dist_image(image)
    ind_max = (image == image.max())
    inner_pix = 3 * oversample
    if (image[image_rho<inner_pix].max() == image.max()) and (image.max()>1000*image[~ind_max].max()):
        image[image_rho < inner_pix] = 0

    # Crop image to minimum size plus some border
    if shape_out is None:
        im_disk = crop_zero_rows_cols(disk_hdul[0].data, symmetric=True)
        sh_new = np.array(im_disk.shape) + 20
        disk_hdul[0].data = pad_or_cut_to_size(im_disk, sh_new)
        # ydata, xdata = disk_hdul[0].data.shape
        # ynew = np.max([ypix, ydata])
        # xnew = np.max([xpix, xdata])
        # disk_hdul[0].data = pad_or_cut_to_size(disk_hdul[0].data, (ynew,xnew))

    return disk_hdul


def get_cen_offsets(self, idl_offset=(0,0), PA_offset=0):
    """ Determine pixel offsets relative to center of subarray
    
    Given the 'idl' offset of some object relative to an observation's
    siaf_ap reference position (e.g., mask center), determine the 
    relative position from the center of the subarray image.
    
    Parameters
    ==========
    idl_offset : tuple
        Source location in arcsec from SIAF aperture position.
    PA_offset : float
        Rotation of scene by some position angle.
        Positive values are counter-clockwise from +Y direction.
        This should be -1 times telescope V3 PA.
    """
    
    # Location relative to star
    idl_offset = np.array(idl_offset)
    delx_asec, dely_asec = idl_offset

    # Add in PA offset
    if PA_offset!=0:
        delx_asec, dely_asec = xy_rot(delx_asec, dely_asec, PA_offset)

    # Determine planet PSF shift in pixels compared to center of mask
    # Convert delx and dely from 'idl' to 'sci' coords
    xsci_ref, ysci_ref = self.siaf_ap.reference_point('sci')
    xsci_pl, ysci_pl = self.siaf_ap.idl_to_sci(delx_asec, dely_asec)
    delx_sci, dely_sci = (xsci_pl - xsci_ref, ysci_pl - ysci_ref)

    # Add in bar offset
    delx_asec = delx_sci * self.pixelscale + self.bar_offset
    dely_asec = dely_sci * self.pixelscale
    
    return (delx_asec, dely_asec)


def gen_coron_mask(self):
    """
    Generate coronagraphic mask transmission images.

    Output images are in 'sci' coordinates.
    """
    mask = self.image_mask
    pupil = self.pupil_mask
    oversample = self.oversample

    mask_dict = {}
    if (not self.is_coron) and (not self.is_lyot):
        mask_dict['DETSAMP'] = None
        mask_dict['OVERSAMP'] = None
    else:
        detid = self.Detector.detid
        x0, y0 = (self.det_info['x0'], self.det_info['y0'])
        xpix, ypix = (self.det_info['xpix'], self.det_info['ypix'])

        # im_det  = build_mask_detid(detid, oversample=1, pupil=pupil)
        im_over = build_mask_detid(detid, oversample=oversample, 
                                    pupil=pupil, filter=self.filter)
        # Convert to det coords and crop
        # im_det  = sci_to_det(im_det, detid)
        im_over = sci_to_det(im_over, detid)
        im_det = frebin(im_over, scale=1/oversample, total=False)

        im_det = im_det[y0:y0+ypix, x0:x0+xpix]
        # Crop oversampled image
        ix1, iy1 = np.array([x0, y0]) * oversample
        ix2 = ix1 + xpix * oversample
        iy2 = iy1 + ypix * oversample
        im_over = im_over[iy1:iy2, ix1:ix2]

        # Revert to sci coords
        mask_dict['DETSAMP'] = det_to_sci(im_det, detid)
        mask_dict['OVERSAMP'] = det_to_sci(im_over, detid)

    return mask_dict
    

def attenuate_with_coron_mask(self, image_oversampled, cmask):
    """ Image attenuation from coronagraph mask features

    Multiply image data by coronagraphic mask.
    Excludes region already affected by observed occulting mask.
    Involves mainly ND Squares and opaque COM holder.
    Appropriately accounts for COM substrate wavelength-dep throughput.
    
    WARNING: If self.ND_acq=True, then bandpass already includes
    ND throughput, so be careful not to double count.
    """

    # In case of imaging
    if cmask is None:
        return image_oversampled

    w_um = self.bandpass.avgwave() / 1e4
    com_th = nircam_com_th(wave_out=w_um)
    pixscale_over = self.pixelscale / self.oversample

    # Exclude actual coronagraphic mask since this will be
    # taken into account during PSF convolution. Not true for
    # all other elements within FOV, ND squares, and mask holder.
    if 'FULL' in self.det_info['wind_mode']:
        cmask_temp = cmask.copy()                
        # center = cdict['cen_sci']
        cdict = coron_ap_locs(self.module, self.channel, self.image_mask, pupil=self.pupil_mask)
        center = np.array(cdict['cen_sci']) * self.oversample
        # center = np.array(self.siaf_ap.reference_point('sci'))*self.oversample
        r, th = dist_image(cmask_temp, pixscale=pixscale_over, center=center, return_theta=True)
        x_asec, y_asec = rtheta_to_xy(r, th)
        ind = (np.abs(y_asec)<4.5) & (cmask_temp>0)
        cmask_temp[ind] = com_th
    elif self.ND_acq:
        # No modifications if ND_acq
        cmask_temp = cmask
    else:
        cmask_temp = cmask.copy()
        r, th = dist_image(cmask_temp, pixscale=pixscale_over, return_theta=True)
        x_asec, y_asec = rtheta_to_xy(r, th)
        # ind = (np.abs(x_asec)<10.05) & (np.abs(y_asec)<4.5)
        ind = np.abs(y_asec)<4.5
        cmask_temp[ind] = com_th
    
    # COM throughput taken into account in bandpass throughput curve
    # so divide out
    cmask_temp = cmask_temp / com_th

    return image_oversampled * cmask_temp
