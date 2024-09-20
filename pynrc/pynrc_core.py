"""pyNRC - Python ETC and Simulator for JWST NIRCam"""

# Import libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.io import ascii
from webbpsf_ext.webbpsf_ext_core import NIRCam_ext
from .nrc_utils import *

from .detops import det_timing, multiaccum, nrc_header
from webbpsf_ext.webbpsf_ext_core import _check_list
from webbpsf_ext.synphot_ext import Observation, ArraySpectrum

from tqdm.auto import trange, tqdm

import pysiaf
from pysiaf import rotations

from . import conf
from .logging_utils import setup_logging

import logging
_log = logging.getLogger('pynrc')

__epsilon = np.finfo(float).eps

class DetectorOps(det_timing):
    """ 
    Class to hold detector operations information. Includes SCA attributes such as
    detector names and IDs as well as :class:`multiaccum` class for ramp settings.

    Parameters
    ----------------
    detector : int, str
        NIRCam detector ID (481-490) or SCA ID (A1-B5).
    wind_mode : str
        Window mode type 'FULL', 'STRIPE', 'WINDOW'.
    xpix : int
        Size of window in x-pixels for frame time calculation.
    ypix : int
        Size of window in y-pixels for frame time calculation.
    x0 : int
        Lower-left x-coord position of detector window.
    y0 : int
        Lower-left y-coord position of detector window.
    nff : int
        Number of fast row resets.

    Keyword Args
    ------------
    read_mode : str
        NIRCam Ramp Readout mode such as 'RAPID', 'BRIGHT1', etc.
    nint : int
        Number of integrations (ramps).
    ngroup : int
        Number of groups in a integration.
    nf : int
        Number of frames per group.
    nd1 : int
        Number of drop frame after reset (before first group read). 
    nd2 : int
        Number of drop frames within a group (ie., groupgap). 
    nd3 : int
        Number of drop frames after final read frame in ramp. 


    Examples
    --------
    Use kwargs functionality to pass keywords to the multiaccum class.
    
    Send via a dictionary of keywords and values:
        >>> kwargs = {'read_mode':'RAPID', 'nint':5, 'ngroup':10}
        >>> d = DetectorOps(**kwargs)
    
    Set the keywords directly:   
        >>> d = DetectorOps(read_mode='RAPID', nint=5, ngroup=10)
    """

    def __init__(self, detector=481, wind_mode='FULL', xpix=2048, ypix=2048, 
                 x0=0, y0=0, nff=None, **kwargs):
                 
        super().__init__(wind_mode=wind_mode, xpix=xpix, ypix=ypix, 
                         x0=x0, y0=y0, mode='JWST', nff=nff, **kwargs)

        # Typical values for SW/LW detectors that get saved based on SCA ID.
        # After setting the SCA ID, these various parameters can be updated,
        # however they will be reset whenever the SCA ID is modified.
        #   - Pixel Scales in arcsec/pix
        #   - Well saturation level in e-
        #   - Typical dark current values in e-/sec (ISIM CV3)
        #   - Read Noise in e-
        #   - IPC and PPC in %
        #   - p_excess: Parameters that describe the excess variance observed in
        #     effective noise plots.
        self._properties_SW = {'pixel_scale':pixscale_SW, 'dark_current':0.002, 'read_noise':11.5, 
                               'IPC':0.54, 'PPC':0.09, 'p_excess':(1.0,5.0), 'ktc':37.6,
                               'well_level':105e3, 'well_level_old':81e3}
        self._properties_LW = {'pixel_scale':pixscale_LW, 'dark_current':0.034, 'read_noise':10.0, 
                               'IPC':0.60, 'PPC':0.19, 'p_excess':(1.5,10.0), 'ktc':36.8,
                               'well_level':83e3, 'well_level_old':75e3}
        # Automatically set the pixel scale based on detector selection
        self.auto_pixscale = True  

        # Pre-flight estimates
        # self._gain_list = {481:2.07, 482:2.01, 483:2.16, 484:2.01, 485:1.83, 
        #                    486:2.00, 487:2.42, 488:1.93, 489:2.30, 490:1.85}
        ## Updated flight estimates from P330E (PID 1538)
        self._gain_list = {481:2.13, 482:2.17, 483:2.25, 484:2.08, 485:1.88, 
                           486:2.00, 487:2.24, 488:2.09, 489:2.18, 490:1.90}

        self._scaids = {481:'A1', 482:'A2', 483:'A3', 484:'A4', 485:'A5',
                        486:'B1', 487:'B2', 488:'B3', 489:'B4', 490:'B5'}
        # Allow user to specify name using either SCA ID or Detector ID (ie., 481 or 'A1')
        try: # First, attempt to set SCA ID
            self.scaid = detector
        except ValueError: 
            try: # If that doesn't work, then try to set Detector ID
                self.detid = get_detname(detector)[3:]
            except ValueError: # If neither work, raise ValueError exception
                raise ValueError("Invalid detector: {0} \n\tValid names are: {1},\n\t{2}" \
                      .format(detector, ', '.join(self.detid_list), \
                      ', '.join(str(e) for e in self.scaid_list)))

        # For full arrays number of resets in first integration is 0
        # self.wind_mode = wind_mode

        _log.info('Initializing SCA {}/{}'.format(self.scaid,self.detid))

    @property
    def wind_mode(self):
        """Window mode attribute"""
        return self._wind_mode
    @wind_mode.setter
    def wind_mode(self, value):
        """Set Window mode attribute"""
        self._wind_mode = value
        self.multiaccum.nr1 = 0 if value=='FULL' else 1

    @property
    def scaid(self):
        """Selected SCA ID from detectors in the `scaid_list` attribute. 481, 482, etc."""
        return self._scaid

    @property
    def detid(self):
        """Selected Detector ID from detectors in the `detid_list` attribute. A1, A2, etc."""
        return self._detid

    @property
    def detname(self):
        """Selected Detector ID from detectors in the `scaid_list` attribute. NRCA1, NRCA2, etc."""
        return self._detname

    # Used for setting the SCA ID then updating all the other detector properties
    @scaid.setter
    def scaid(self, value):
        """Set SCA ID (481, 482, ..., 489, 490). Automatically updates other relevant attributes."""
        _check_list(value, self.scaid_list, var_name='scaid')

        self._scaid = value
        self._detid = self._scaids.get(self._scaid)

        # Detector Name (as stored in FITS headers): NRCA1, NRCALONG, etc.
        if self.channel=='LW': self._detname = 'NRC' + self.module + 'LONG'
        else:  self._detname = 'NRC' + self._detid

        # Select various detector properties (pixel scale, dark current, read noise, etc)
        # depending on LW or SW detector
        dtemp = self._properties_LW if self.channel=='LW' else self._properties_SW
        if self.auto_pixscale: 
            self.pixelscale = dtemp['pixel_scale']
        self.ktc          = dtemp['ktc']
        self.dark_current = dtemp['dark_current']
        self.read_noise   = dtemp['read_noise']
        self.IPC          = dtemp['IPC']
        self.PPC          = dtemp['PPC']
        self.p_excess     = dtemp['p_excess']
        self.well_level   = dtemp['well_level']

        self.gain = self._gain_list.get(self._scaid, 2.0)

    # Similar to scaid.setter, except if detector ID is specified.
    @detid.setter
    def detid(self, value):
        """Set detector ID (A1, A2, ..., B4, B5). Automatically updates other relevant attributes."""
        if 'NRC' in value:
            value = value[3:]
        _check_list(value, self.detid_list, var_name='detid')

        # Switch dictionary keys and values, grab the corresponding SCA ID,
        # and then call scaid.setter
        newdict = {y:x for x,y in self._scaids.items()}
        self.scaid = newdict.get(value)

    @property
    def scaid_list(self):
        """Allowed SCA IDs"""
        return sorted(list(self._scaids.keys()))

    @property
    def detid_list(self):
        """Allowed Detector IDs"""
        return sorted(list(self._scaids.values()))

    @property
    def module(self):
        """NIRCam modules A or B (inferred from detector ID)"""
        return self._detid[0]

    @property
    def channel(self):
        """Detector channel 'SW' or 'LW' (inferred from detector ID)"""
        return 'LW' if self.detid.endswith('5') else 'SW'

    def xtalk(self, file_path=None):
        """Detector cross talk information"""

        if file_path is None:
            file = 'xtalk20150303g0.errorcut.txt'
            file_path = os.path.join(conf.PYNRC_PATH, 'sim_params', file)

        xt_coeffs = ascii.read(file_path, header_start=0)
        ind = xt_coeffs['Det'] == self.detid
        return xt_coeffs[ind]

    def pixel_noise(self, fsrc=0.0, fzodi=0.0, fbg=0.0, rn=None, ktc=None, idark=None,
        p_excess=None, ng=None, nf=None, scale_ints=True, verbose=False, **kwargs):
        """Noise values per pixel.
        
        Return theoretical noise calculation for the specified MULTIACCUM exposure 
        in terms of e-/sec. This uses the pre-defined detector-specific noise 
        properties. Can specify flux of a source as well as background and 
        zodiacal light (in e-/sec/pix). After getting the noise per pixel per
        ramp (integration), value(s) are divided by the sqrt(NINT) to return
        the final noise

        Parameters
        ----------
        fsrc : float or image
            Flux of source in e-/sec/pix
        fzodi : float or image
            Flux of the zodiacal background in e-/sec/pix
        fbg : float or image
            Flux of telescope background in e-/sec/pix
        idark : float or image
            Option to specify dark current in e-/sec/pix.
        rn : float
            Option to specify Read Noise per pixel (e-).
        ktc : float
            Option to specify kTC noise (in e-). Only valid for single frame (n=1)
        p_excess : array-like
            Optional. An array or list of two elements that holds the parameters
            describing the excess variance observed in effective noise plots.
            By default these are both 0. For NIRCam detectors, recommended
            values are [1.0,5.0] for SW and [1.5,10.0] for LW.
        ng : None or int or image
            Option to explicitly states number of groups. This is specifically
            used to enable the ability of only calculating pixel noise for
            unsaturated groups for each pixel. If a numpy array, then it should
            be the same shape as `fsrc` image. By default will use `self.ngroup`.
        scale_ints : bool
            Scale pixel noise by by sqrt(nint)?
        verbose : bool
            Print out results at the end.

        Keyword Arguments
        -----------------
        ideal_Poisson : bool
            If set to True, use total signal for noise estimate,
            otherwise MULTIACCUM equation is used.

        Notes
        -----
        fsrc, fzodi, and fbg are functionally the same as they are immediately summed.
        They can also be single values or multiple elements (list, array, tuple, etc.).
        If multiple inputs are arrays, make sure their array sizes match.
        
        """
        ma = self.multiaccum
        if ng is None:
            ng = ma.ngroup
        if nf is None:
            nf = ma.nf
        if rn is None:
            rn = self.read_noise
        if ktc is None:
            ktc = self.ktc
        if p_excess is None:
            p_excess = self.p_excess
        if idark is None:
            idark = self.dark_current

        # Pixel noise per ramp (e-/sec/pix)
        pn = pix_noise(ngroup=ng, nf=nf, nd2=ma.nd2, tf=self.time_frame, 
                       rn=rn, ktc=ktc, p_excess=p_excess, 
                       idark=idark, fsrc=fsrc, fzodi=fzodi, fbg=fbg, **kwargs)
    
        # Divide by sqrt(Total Integrations)
        final = pn / np.sqrt(ma.nint) if scale_ints else pn

        if verbose:
            print('Noise (e-/sec/pix): {}'.format(final))
            print('Total Noise (e-/pix): {}'.format(final*self.time_exp))

        return final

    @property
    def fastaxis(self):
        """Fast readout direction in sci coords"""
        # https://jwst-pipeline.readthedocs.io/en/latest/jwst/references_general/references_general.html#orientation-of-detector-image
        # 481, 3, 5, 7, 9 have fastaxis equal -1
        # Others have fastaxis equal +1
        fastaxis = -1 if np.mod(self.scaid,2)==1 else +1
        return fastaxis
    @property
    def slowaxis(self):
        """Slow readout direction in sci coords"""
        # https://jwst-pipeline.readthedocs.io/en/latest/jwst/references_general/references_general.html#orientation-of-detector-image
        # 481, 3, 5, 7, 9 have slowaxis equal +2
        # Others have slowaxis equal -2
        slowaxis = +2 if np.mod(self.scaid,2)==1 else -2
        return slowaxis

    def make_header(self, filter=None, pupil_mask=None, obs_time=None, **kwargs):
        """
        Create a generic NIRCam FITS header.

        Parameters
        ----------
        filter :str
            Name of filter element.
        pupil_mask : str
            Name of pupil element.
        obs_time : datetime 
            Specifies when the observation was considered to be executed.
            If not specified, then it will choose the current time.
            This must be a datetime object:
            
            >>> datetime.datetime(2016, 5, 9, 11, 57, 5, 796686)
        """
        return nrc_header(self, filter=filter, pupil=pupil_mask, obs_time=obs_time, **kwargs)


class NIRCam(NIRCam_ext):
    """NIRCam base instrument class
    
    Creates a NIRCam instrument class that holds all the information pertinent to
    an observation using a given observation. This class extends the NIRCam subclass
    :class:`webbpsf_ext.NIRCam_ext`, to generate PSF coefficients to calculate an arbitrary
    PSF based on wavelength, field position, and WFE drift.

    In addition to PSF generation, includes ability to estimate detector saturation 
    limits, sensitivities, and perform ramp optimizations.
    """

    def __init__(self, filter=None, pupil_mask=None, image_mask=None, 
                 ND_acq=False, detector=None, apname=None, autogen_coeffs=True, **kwargs):

        """ Init Function

        Parameters
        ==========
        filter : str
            Name of input filter.
        pupil_mask : str, None
            Pupil elements such as grisms or lyot stops (default: None).
        image_mask : str, None
            Specify which coronagraphic occulter (default: None).
        ND_acq : bool
            Add in neutral density attenuation in throughput and PSF creation?
            Used primarily for sensitivity and saturation calculations.
            Not recommended for simulations (TBI). 
        detector : int or str
            NRC[A-B][1-5] or 481-490
        apname : str
            Pass specific SIAF aperture name, which will update pupil mask, image mask,
            and detector subarray information.
        autogen_coeffs : bool
            Automatically generate base PSF coefficients. Equivalent to performing
            ``self.gen_psf_coeff()``. Default: True
            WFE drift and field-dependent coefficients should be run manually via
            ``gen_wfedrift_coeff``, ``gen_wfefield_coeff``, and ``gen_wfemask_coeff``.

        Keyword Args
        ============

        wind_mode : str
            Window mode type 'FULL', 'STRIPE', 'WINDOW'.
        xpix : int
            Size of window in x-pixels for frame time calculation.
        ypix : int
            Size of window in y-pixels for frame time calculation.
        x0 : int
            Lower-left x-coord position of detector window.
        y0 : int
            Lower-left y-coord position of detector window.
        read_mode : str
            NIRCam Ramp Readout mode such as 'RAPID', 'BRIGHT1', etc.
        nint : int
            Number of integrations (ramps).
        ngroup : int
            Number of groups in a integration.
        nf : int
            Number of frames per group.
        nd1 : int
            Number of drop frame after reset (before first group read). 
        nd2 : int
            Number of drop frames within a group (ie., groupgap). 
        nd3 : int
            Number of drop frames after final read frame in ramp. 
        nr1 : int
            Number of reset frames within first ramp.
        nr2 : int
            Number of reset frames for subsequent ramps.

        PSF Keywords
        ============

        fov_pix : int
            Size of the PSF FoV in pixels (real SW or LW pixels).
            The defaults depend on the type of observation.
            Odd number place the PSF on the center of the pixel,
            whereas an even number centers it on the "crosshairs."
        oversample : int
            Factor to oversample during WebbPSF calculations.
            Default 2 for coronagraphy and 4 otherwise.
        include_si_wfe : bool
            Include SI WFE measurements? Default=True.
        include_ote_field_dependence : bool
            Include OTE field-dependent WFE measurements? Default=True.
        include_distortions : bool
            If True, will include a distorted version of the PSF.
        pupil : str
            File name or HDUList specifying telescope entrance pupil.
            Can also be an OTE_Linear_Model.
        pupilopd : tuple or HDUList
            Tuple (file, index) or filename or HDUList specifying OPD.
            Can also be an OTE_Linear_Model.
        wfe_drift : float
            Wavefront error drift amplitude in nm.
        offset_r : float
            Radial offset from the center in arcsec.
        offset_theta :float
            Position angle for radial offset, in degrees CCW.
        bar_offset : float
            For wedge masks, option to set the PSF position across the bar.
        jitter : str or None
            Currently either 'gaussian' or None.
        jitter_sigma : float
            If ``jitter = 'gaussian'``, then this is the size of the blurring effect.
        npsf : int
            Number of wavelengths/PSFs to fit.
        ndeg : int
            Degree of polynomial fit.
        nproc : int
            Manual setting of number of processor cores to break up PSF calculation.
            If set to None, this is determined based on the requested PSF size,
            number of available memory, and hardware processor cores. The automatic
            calculation endeavors to leave a number of resources available to the
            user so as to not crash the user's machine. 
        save : bool
            Save the resulting PSF coefficients to a file? (default: True)
        force : bool
            Forces a recalculation of PSF even if saved PSF exists. (default: False)
        quick : bool
            Only perform a fit over the filter bandpass with a lower default polynomial degree fit.
            (default: True)
        use_legendre : bool
            Fit with Legendre polynomials, an orthonormal basis set. (default: True)
        """

        if detector is not None:
            detector = get_detname(detector)

        # Available Filters
        # Note: Certain narrowband filters reside in the pupil wheel and cannot be paired
        # with pupil elements. This will be checked for later.
        self._filters_sw = ['F070W', 'F090W', 'F115W', 'F150W', 'F150W2', 'F200W',
             'F140M', 'F162M', 'F182M', 'F210M', 'F164N', 'F187N', 'F212N']
        self._filters_lw = ['F277W', 'F322W2', 'F356W', 'F444W', 'F323N', 'F405N', 'F466N', 'F470N',
             'F250M', 'F300M', 'F335M', 'F360M', 'F410M', 'F430M', 'F460M', 'F480M']
     
        # Coronagraphic Masks
        self._coron_masks = [None, 'MASK210R', 'MASK335R', 'MASK430R', 'MASKSWB', 'MASKLWB']
        # self.offset_bar = offset_bar

        # Pupil Wheel elements
        self._lyot_masks = ['CIRCLYOT', 'WEDGELYOT']
        # DHS in SW and Grisms in LW
        self._dhs = ['DHS0', 'DHS60']
        # Grism0/90 => GrismR/C
        self._grism = ['GRISMR', 'GRISMC']
        # Weak lens are only in SW pupil wheel (+4 in filter wheel)
        self._weak_lens = ['WLP4', 'WLPM4', 'WLP8', 'WLM8', 'WLP12']

        # Check alternate inputs
        if pupil_mask is not None:
            pupil_mask = pupil_mask.upper()
            # If alternate Weak Lens values are specified
            if 'WL' in pupil_mask:
                wl_alt = {
                    'WEAK LENS +4': 'WLP4',
                    'WEAK LENS +8': 'WLP8', 
                    'WEAK LENS -8': 'WLM8',
                    'WEAK LENS +12 (=4+8)': 'WLP12', 
                    'WEAK LENS -4 (=4-8)': 'WLM4',
                    }
                pupil_mask = wl_alt.get(pupil_mask, pupil_mask)
            # Pair F200W throughput with WL+4
            # The F212N2 throughput is then handled in read_filter() function 
            wl_list = ['WLP12', 'WLM4', 'WLP4']
            if (pupil_mask in wl_list) and ((filter is None) or (filter!='F200W')):
                filter = 'F200W'

            # Check Grism alternate inputs
            if 'GRISM0' in pupil_mask:
                pupil_mask = 'GRISMR'
            elif 'GRISM90' in pupil_mask: 
                pupil_mask = 'GRISMC'

            # Cannot be set to clear
            if pupil_mask=='CLEAR':
                _log.warning('CLEAR is not a valid pupil mask element. Setting to None.')
                pupil_mask = None

        super().__init__(filter=filter, pupil_mask=pupil_mask, image_mask=image_mask, **kwargs)

        if apname is None:
            if detector is not None:
                self.detector = detector
            self._ND_acq = ND_acq

            self._validate_wheels()
            self.update_detectors(**kwargs)
            ap_name_rec = self.get_siaf_apname()
            self.update_from_SIAF(ap_name_rec, image_mask=image_mask,
                                  pupil_mask=pupil_mask)
        else:
            self.update_from_SIAF(apname, image_mask=image_mask, 
                                  pupil_mask=pupil_mask, **kwargs)

            # Default to no jitter for coronagraphy
            # self.options['jitter'] = None if self.is_coron else 'gaussian'

        # Generate PSF coefficients
        if autogen_coeffs:
            self.gen_psf_coeff(**kwargs)

        # Background fov pix is only for coronagraphic masks
        # Create a background reference class
        self._fov_pix_bg = 33
        self._fov_bg_match = False
        # if autogen_coeffs:
        self._update_bg_class(**kwargs)

        # Initialize PSF offset to center of image
        # Calculate PSF offset from center if _nrc_bg coefficients are available
        if self._nrc_bg.psf_coeff is not None:
            self.calc_psf_offset_from_center(use_coeff=True)
        else:
            self.calc_psf_offset_from_center(use_coeff=False)
            # self.psf_offset_to_center = np.array([0,0])

        # Check aperture info is consistent if not explicitly specified
        # TODO: This might fail because self.Detector has not yet been initialized??
        try:
            ap_name_rec = self.get_siaf_apname()
        except AttributeError:
            raise AttributeError(f'Detector might not be initialized bececause {apname} is not valid.')

        if ((apname is None) and (ap_name_rec != self.aperturename) and
            not (('FULL' in self.aperturename) and ('TAMASK' in self.aperturename))):
            # Warning strings
            out_str1 = f'Current aperture {self.aperturename} does not match recommendation ({ap_name_rec}).'
            out_str2 = f'Perhaps try self.aperturename = self.get_siaf_apname()'
            _log.info(out_str1)
            _log.info(out_str2)
        

    def _update_bg_class(self, fov_bg_match=None, **kwargs):
        """
        If there is a coronagraphic spot or bar, then we may need to
        generate another background PSF for sensitivity information.
        It's easiest just to ALWAYS do a small footprint without the
        coronagraphic mask and save the PSF coefficients. 

        WARNING: This assumes throughput of the coronagraphic substrate
        for observations with a Lyot pupil mask.

        Parameters
        ==========
        fov_bg_match : bool or None
            Determines whether or not to match bg FoV to sci FoV for
            coronagraphic observations. If set to None, default to 
            `self._fov_bg_match` property. If a boolean value is
            provided, then `self._fov_bg_match` is updated.
        """

        try:
            # Make sure we don't inadvertently delete base object
            if self._nrc_bg is not self: 
                del self._nrc_bg
        except AttributeError: 
            pass

        # Update background PSF size if fov_bg_match is True
        if fov_bg_match is not None:
            self._fov_bg_match = fov_bg_match


        self._fov_pix_bg = self.fov_pix if self._fov_bg_match else self._fov_pix_bg
        if self._image_mask is None:
            self._nrc_bg = self
        else:
            log_prev = conf.logging_level
            setup_logging('WARN', verbose=False)

            nrc_bg = NIRCam_ext(filter=self.filter, pupil_mask=self.pupil_mask,
                                fov_pix=self._fov_pix_bg, oversample=self._oversample)

            # Generate coefficients
            nrc_bg.gen_psf_coeff(**kwargs)
            setup_logging(log_prev, verbose=False)

            # Match detector positions for WFE calculations
            nrc_bg.detector_position = self.detector_position

            # Save as attribute
            self._nrc_bg = nrc_bg

    # Allowed values for filters, coronagraphic masks, and pupils
    # @property
    # def filter_list(self):
    #     """List of allowable filters."""
    #     return self._filters_sw + self._filters_lw
    # @property
    # def image_mask_list(self):
    #     """List of allowable coronagraphic mask values."""
    #     return self._coron_masks
    # @property
    # def pupil_mask_list(self):
    #     """List of allowable pupil mask values."""
    #     return ['CLEAR','FLAT'] + self._lyot_masks + self._grism + self._dhs + self._weak_lens

    # Check consistencies
    def _validate_wheels(self):
        """ 
        Validation to make sure the selected filters and pupils are allowed to be in parallel.
        """
        
        def do_warn(wstr):
            _log.warning(wstr)
            _log.warning('Proceed at your own risk!')

        filter     = self._filter
        pupil_mask = self._pupil_mask
        image_mask = self._image_mask
        if self.channel=='long' or self.channel=='LW':
            channel = 'LW'
        else:
            channel = 'SW'

        if image_mask is None: 
            image_mask = ''
        if pupil_mask is None: 
            pupil_mask = ''

        # Weak lenses can only occur in SW modules
        if ('WEAK LENS' in pupil_mask) and (channel=='LW'):
            wstr = '{} in pupil is not valid with filter {}.'.format(pupil_mask,filter)
            wstr = wstr + '\nWeak lens only in SW module.'
            do_warn(wstr)

        # DHS in SW modules
        if ('DHS' in pupil_mask) and (channel=='LW'):
            wstr = '{} in pupil is not valid with filter {}.'.format(pupil_mask,filter)
            wstr = wstr + '\nDHS only in SW module.'
            do_warn(wstr)
            
        # DHS cannot be paired with F164N or F162M
        flist = ['F164N', 'F162M']
        if ('DHS' in pupil_mask) and (filter in flist):
            wstr = 'Both {} and filter {} exist in same pupil wheel.'.format(pupil_mask,filter)
            do_warn(wstr)

        # Grisms in LW modules
        if ('GRISM' in pupil_mask) and (channel=='SW'):
            wstr = '{} in pupil is not valid with filter {}.'.format(pupil_mask,filter)
            wstr = wstr + '\nGrisms only in LW module.'
            do_warn(wstr)
            
        # Grisms cannot be paired with any Narrowband filters
        flist = ['F323N', 'F405N', 'F466N', 'F470N']
        if ('GRISM' in pupil_mask) and (filter in flist):
            wstr = 'Both {} and filter {} exist in same pupil wheel.'.format(pupil_mask,filter)
            do_warn(wstr)

        # MASK430R falls in SW SCA gap and cannot be seen by SW module
        if ('MASK430R' in image_mask) and (channel=='SW'):
            wstr = '{} mask is not visible in SW module (filter is {})'.format(image_mask,filter)
            do_warn(wstr)

        # Need F200W paired with WEAK LENS +4
        # The F212N2 filter is handled in the read_filter function
        wl_list = ['WEAK LENS +12 (=4+8)', 'WEAK LENS -4 (=4-8)', 'WEAK LENS +4']
        if (pupil_mask in wl_list) and (filter!='F200W'):
            wstr = '{} is only valid with filter F200W.'.format(pupil_mask)
            do_warn(wstr)

        # Items in the same SW pupil wheel
        sw2 = ['WEAK LENS +8', 'WEAK LENS -8', 'F162M', 'F164N', 'CIRCLYOT', 'WEDGELYOT']
        if (filter in sw2) and (pupil_mask in sw2):
            wstr = '{} and {} are both in the SW Pupil wheel.'.format(filter,pupil_mask)
            do_warn(wstr)

        # Items in the same LW pupil wheel
        lw2 = ['F323N', 'F405N', 'F466N', 'F470N', 'CIRCLYOT', 'WEDGELYOT']
        if (filter in lw2) and (pupil_mask in lw2):
            wstr = '{} and {} are both in the LW Pupil wheel.'.format(filter,pupil_mask)
            do_warn(wstr)
    
        # ND_acq must have a LYOT stop, otherwise coronagraphic mask is not in FoV
        if self.ND_acq and ('LYOT' not in pupil_mask):
            wstr = 'CIRCLYOT or WEDGELYOT must be in pupil wheel if ND_acq=True.'
            do_warn(wstr)

        # ND_acq and coronagraphic mask are mutually exclusive
        if self.ND_acq and (image_mask != ''):
            wstr = 'If ND_acq is set, then mask must be None.'
            do_warn(wstr)

    def update_detectors(self, verbose=False, **kwargs):
        """ Update detector operation parameters

        Creates detector object based on :attr:`detector` attribute.
        This function should be called any time a filter, pupil, mask, or
        module is modified by the user.

        If the user wishes to change any properties of the multiaccum ramp
        or detector readout mode, pass those arguments through this function
        rather than creating a whole new NIRCam() instance. For example:
        
            >>> nrc = pynrc.NIRCam('F430M', ngroup=10, nint=5)
            >>> nrc.update_detectors(ngroup=2, nint=10, wind_mode='STRIPE', ypix=64)
    
        A dictionary of the keyword settings can be referenced in :attr:`det_info`.
        This dictionary cannot be modified directly.
        
        Parameters
        ----------
        verbose : bool
            Print out ramp and detector settings.
        
        Keyword Args
        ------------
        wind_mode : str
            Window mode type 'FULL', 'STRIPE', 'WINDOW'.
        xpix : int
            Size of window in x-pixels for frame time calculation.
        ypix : int
            Size of window in y-pixels for frame time calculation.
        x0 : int
            Lower-left x-coord position of detector window.
        y0 : int
            Lower-left y-coord position of detector window.
        read_mode : str
            NIRCam Ramp Readout mode such as 'RAPID', 'BRIGHT1', etc.
        nint : int
            Number of integrations (ramps).
        ngroup : int
            Number of groups in a integration.
        nf : int
            Number of frames per group.
        nd1 : int
            Number of drop frame after reset (before first group read). 
        nd2 : int
            Number of drop frames within a group (ie., groupgap). 
        nd3 : int
            Number of drop frames after final read frame in ramp. 
        nr1 : int
            Number of reset frames within first ramp.
        nr2 : int
            Number of reset frames for subsequent ramps.
        """
        # Check if kwargs is empty
        if not kwargs:
            try: 
                kwargs = self.det_info
            except AttributeError: 
                kwargs = {}
        else:
            try: 
                self._det_info.update(kwargs)
            except AttributeError: 
                self._det_info = kwargs
            kwargs = self.det_info

        # Update detector class
        # For now, it's just easier to delete old instances and start from scratch
        # rather than tracking changes and updating only the changes. That could 
        # get complicated, and I don't think there is a memory leak from deleting
        # the Detector instances.
        try:
            del self.Detector
        except AttributeError: 
            pass
        self.Detector = DetectorOps(detector=self.detector, **kwargs)

        # Update stored kwargs
        kw1 = self.Detector.to_dict()
        _ = kw1.pop('detector', None)
        kw2 = self.multiaccum.to_dict()
        self._det_info = merge_dicts(kw1,kw2)

        if verbose:
            print('New Ramp Settings')
            keys = ['read_mode', 'nf', 'nd2', 'ngroup', 'nint']
            for k in keys:
                v = self.det_info[k]
                if isinstance(v,float): print("{:<9} : {:>8.0f}".format(k, v))
                else: print("  {:<10} : {:>8}".format(k, v))

            print('New Detector Settings')
            keys = ['wind_mode', 'xpix', 'ypix', 'x0', 'y0']
            for k in keys:
                v = self.det_info[k]
                if isinstance(v,float): print("{:<9} : {:>8.0f}".format(k, v))
                else: print("  {:<10} : {:>8}".format(k, v))
    
            print('New Ramp Times')
            ma = self.multiaccum_times
            keys = ['t_group', 't_frame', 't_int', 't_int_tot1', 't_int_tot2', 't_exp', 't_acq']
            for k in keys:
                print('  {:<10} : {:>8.3f}'.format(k, ma[k]))

    def update_psf_coeff(self, filter=None, pupil_mask=None, image_mask=None, detector=None, 
        fov_pix=None, oversample=None, include_ote_field_dependence=None, 
        include_si_wfe=None, include_distortions=None, 
        pupil=None, pupilopd=None, offset_r=None, offset_theta=None, bar_offset=None, 
        jitter=None, jitter_sigma=None, npsf=None, ndeg=None, nproc=None, quick=None,
        save=None, force=False, use_legendre=None, **kwargs):

        """ Update properties and create new set of PSF coefficients

        Parameters
        ----------
        filter : str
            Name of NIRCam filter.
        pupil_mask : str, None
            NIRCam pupil elements such as grisms or lyot stops (default: None).
        image_mask : str, None
            Specify which coronagraphic occulter (default: None).
        detector : str
            Name of detector (e.g., "NRCA5")
        fov_pix : int
            Size of the PSF FoV in pixels (real SW or LW pixels).
            The defaults depend on the type of observation.
            Odd number place the PSF on the center of the pixel,
            whereas an even number centers it on the "crosshairs."
        oversample : int
            Factor to oversample during WebbPSF calculations.
            Default 2 for coronagraphy and 4 otherwise.
        include_si_wfe : bool
            Include SI WFE measurements? Default=True.
        include_ote_field_dependence : bool
            Include OTE field-dependent WFE measurements? Default=True.
        include_distortions : bool
            If True, will include a distorted version of the PSF.
        pupil : str
            File name or HDUList specifying telescope entrance pupil.
            Can also be an OTE_Linear_Model.
        pupilopd : tuple or HDUList
            Tuple (file, index) or filename or HDUList specifying OPD.
            Can also be an OTE_Linear_Model.
        wfe_drift : float
            Wavefront error drift amplitude in nm.
        offset_r : float
            Radial offset from the center in arcsec.
        offset_theta :float
            Position angle for radial offset, in degrees CCW.
        bar_offset : float
            For wedge masks, option to set the PSF position across the bar.
        jitter : str or None
            Currently either 'gaussian' or None.
        jitter_sigma : float
            If ``jitter = 'gaussian'``, then this is the size of the blurring effect.
        npsf : int
            Number of wavelengths/PSFs to fit.
        ndeg : int
            Degree of polynomial fit.
        nproc : int
            Manual setting of number of processor cores to break up PSF calculation.
            If set to None, this is determined based on the requested PSF size,
            number of available memory, and hardware processor cores. The automatic
            calculation endeavors to leave a number of resources available to the
            user so as to not crash the user's machine. 
        save : bool
            Save the resulting PSF coefficients to a file? (default: True)
        force : bool
            Forces a recalcuation of PSF even if saved PSF exists. (default: False)
        quick : bool
            Only perform a fit over the filter bandpass with a lower default polynomial degree fit.
            (default: True)
        use_legendre : bool
            Fit with Legendre polynomials, an orthonormal basis set. (default: True)
        """

        update_coeffs = False
        update_bg_coeffs = False

        # filter, pupil mask, and image mask
        if (filter is not None) and (filter != self.filter):
            update_coeffs = True
            update_bg_coeffs = True
            self.filter = filter
        if (pupil_mask is not None) and (pupil_mask != self.pupil_mask):
            update_coeffs = True
            update_bg_coeffs = True
            if (pupil_mask.upper()=="CLEAR") or (pupil_mask.upper()=="NONE"):
                pupil_mask = None
            self.pupil_mask = pupil_mask
        if (image_mask is not None) and (image_mask != self.image_mask):
            update_coeffs = True
            update_bg_coeffs = True
            if (image_mask.upper()=="CLEAR") or (image_mask.upper()=="NONE"):
                image_mask = None
            self.image_mask = image_mask
        if (fov_pix is not None) and (fov_pix != self.fov_pix):
            update_coeffs = True
            self.fov_pix = fov_pix
        if (oversample is not None) and (oversample != self.oversample):
            update_coeffs = True
            self.oversample = oversample

        # SI WFE and distortions
        if (include_si_wfe is not None) and (include_si_wfe != self.include_si_wfe):
            update_coeffs = True
            self.include_si_wfe = include_si_wfe
        if (include_ote_field_dependence is not None) and (include_ote_field_dependence != self.include_ote_field_dependence):
            update_coeffs = True
            self.include_ote_field_dependence = include_ote_field_dependence
        if (include_distortions is not None) and (include_distortions != self.include_distortions):
            update_coeffs = True
            self.include_distortions = include_distortions

        # Pupil OPD information
        if (pupil is not None) and (self.pupil != pupil):
            update_coeffs = True
            self.pupil = pupil
        if (pupilopd is not None) and (self.pupilopd != pupilopd):
            update_coeffs = True
            self.pupilopd = pupilopd

        # Source and mask offsetting
        if (offset_r is not None) and (self.options.get('source_offset_r') != offset_r):
            update_coeffs = True
            self.options['source_offset_r'] = offset_r
        if (offset_theta is not None) and (self.options.get('source_offset_theta') != offset_theta):
            update_coeffs = True
            self.options['source_offset_theta'] = offset_theta
        if (bar_offset is not None) and (self.options.get('bar_offset') != bar_offset):
            update_coeffs = True
            self.options['bar_offset'] = bar_offset

        # Jitter
        if (jitter is not None) and (self.options.get('jitter') != jitter):
            update_coeffs = True
            self.options['jitter'] = jitter
        if (jitter_sigma is not None) and (self.options.get('jitter_sigma') != jitter_sigma):
            update_coeffs = True
            self.options['jitter_sigma'] = jitter_sigma

        # Misecellaneous
        if (npsf is not None) and (self.npsf != npsf):
            update_coeffs = True
            self.npsf = npsf
        if (ndeg is not None) and (self.ndeg != ndeg):
            update_coeffs = True
            self.ndeg = ndeg
        if (quick is not None) and (self.quick != quick):
            update_coeffs = True
            self.quick = quick
        if (use_legendre is not None) and (self.use_legendre != use_legendre):
            update_coeffs = True
            self.use_legendre = use_legendre

        # Detector update
        if detector is not None:
            update_coeffs = True
            self.detector = get_detname(detector)
            self.update_detectors()

        # Regenerate PSF coefficients
        if update_coeffs:
            try:
                del self.psf_coeff, self.psf_coeff_header
            except AttributeError:
                pass
            save = True if save is None else save
            self.gen_psf_coeff(save=save, force=force, nproc=nproc, **kwargs)

            # Update drift, field, and mask-dependent coefficients
            if self._psf_coeff_mod['wfe_drift'] is not None:
                self.gen_wfedrift_coeff()
            if self._psf_coeff_mod['si_field'] is not None:
                self.gen_wfefield_coeff()
            if self._psf_coeff_mod['si_mask'] is not None:
                self.gen_wfemask_coeff()

        # Update bg class if filter or pupil mask is changed
        if update_bg_coeffs:
            self._update_bg_class()

    @property
    def psf_info(self):
        """PSF parameters"""
        d_options = self.options

        d = {
            'fov_pix': self.fov_pix, 'oversample': self.oversample,
            'npsf': self.npsf, 'ndeg': self.ndeg, 
            'include_si_wfe': self.include_si_wfe,
            'include_ote_field_dependence': self.include_ote_field_dependence, 
            'include_distortions': self.include_distortions,
            'jitter': d_options.get('jitter'), 'jitter_sigma': d_options.get('jitter_sigma'), 
            'offset_r': d_options.get('source_offset_r', 0), 'offset_theta': d_options.get('source_offset_theta', 0),
            'bar_offset': d_options.get('bar_offset', None), 
            'pupil': self.pupil, 'pupilopd': self.pupilopd, 
        }

        return d

    @property
    def multiaccum(self):
        """:class:`multiaccum` object"""
        return self.Detector.multiaccum
    @property
    def multiaccum_times(self):
        """Exposure timings in dictionary
        
        t_frame   : Time of a single frame.
        t_group   : Time of a single group (read frames + drop frames).
        t_int     : Photon collection time for a single ramp/integration.
        t_int_tot1: Total time for all frames (reset+read+drop) in a first ramp.
        t_int_tot2: Total time for all frames (reset+read+drop) in a subsequent ramp.
        t_exp     : Total photon collection time for all ramps.
        t_acq     : Total acquisition time to complete exposure with all overheads.
        """
        return self.Detector.times_to_dict()

    @property
    def det_info(self):
        """Dictionary housing detector info parameters and keywords."""
        return self._det_info
    @property
    def well_level(self):
        """Detector well level in units of electrons"""
        return self.Detector.well_level

    @property
    def siaf_ap_names(self):
        """Give all possible SIAF aperture names"""
        return list(self.siaf.apernames)

    def get_siaf_apname(self):
        """Get SIAF aperture based on instrument settings"""

        # Return already defined ap name
        # if (self.siaf_ap is not None) and (not override):
        #     return self.siaf_ap.AperName
        # else:
        detid = self.Detector.detid
        wind_mode = self.Detector.wind_mode

        is_lyot = self.is_lyot
        is_coron = self.is_coron
        is_grism = self.is_grism

        pupil_mask = self.pupil_mask
        if self.channel=='long' or self.channel=='LW':
            channel = 'LW'
        else:
            channel = 'SW'

        # Time series filters
        ts_filters = ['F277W','F356W','F444W','F322W2']
        # Coronagraphic bar filters
        swb_filters = ['F182M','F187N','F210M','F212N','F200W']
        lwb_filters = [
            'F250M','F300M','F277W','F335M','F360M',
            'F356W','F410M','F430M','F460M','F480M','F444W'
        ]

        # Coronagraphy
        if is_coron:
            wstr = 'FULL_' if wind_mode=='FULL' else ''
            key = 'NRC{}_{}{}'.format(detid,wstr,self.image_mask)
            if ('LWB' in self.image_mask) and (self.module=='A') and (self.filter in lwb_filters):
                key = key + '_{}'.format(self.filter)
            elif ('SWB' in self.image_mask) and (self.module=='A') and (self.filter in swb_filters):
                key = key + '_{}'.format(self.filter)
            if wind_mode=='STRIPE':
                key = None
        # Just Lyot stop without masks, assuming TA aperture
        elif is_lyot: #and self.ND_acq:
            tastr = 'TA' if self.ND_acq else 'FSTA'
            key = 'NRC{}_{}'.format(detid,tastr)
            if ('CIRC' in pupil_mask) and ('SW' in channel):
                key = key + 'MASK210R'
            elif ('CIRC' in pupil_mask) and ('LW' in channel):
                key = key + 'MASK430R' if ('F4' in self.filter) else key + 'MASK335R'
            elif ('WEDGE' in pupil_mask) and ('SW' in channel):
                key = key + 'MASKSWB'
            elif ('WEDGE' in pupil_mask) and ('LW' in channel):
                key = key + 'MASKLWB'
        # Time series grisms
        elif is_grism and ('GRISMR' in pupil_mask) and (self.filter in ts_filters):
            if wind_mode=='FULL':
                key = f'NRC{detid}_GRISM_{self.filter}'
            elif wind_mode=='STRIPE':
                key = 'NRC{}_GRISM{}_{}'.format(detid,self.det_info['ypix'],self.filter)
            else:
                key = None
        # SW Time Series with LW grism
        elif wind_mode=='STRIPE':
            key = 'NRC{}_GRISMTS{:.0f}'.format(detid,self.det_info['ypix'])
        # WFSS
        # TODO: WFSS SIAF apertures no longer support 'sci' and 'det' coordinates
        # These apertures are not useful
        elif is_grism and (wind_mode=='FULL'):
            key = 'NRC{}_FULL_{}'.format(detid, pupil_mask)
            _log.warning('WFSS SIAF apertures are currently unsupported')
        # Subarrays
        elif wind_mode=='WINDOW':
            key = 'NRC{}_SUB{}P'.format(detid,self.det_info['xpix'])
            if key not in self.siaf_ap_names:
                key = 'NRC{}_TAPSIMG{}'.format(detid,self.det_info['xpix'])
            if key not in self.siaf_ap_names:
                key = 'NRC{}_TAGRISMTS{}'.format(detid,self.det_info['xpix'])
            if key not in self.siaf_ap_names:
                key = 'NRC{}_TAGRISMTS_SCI_{}'.format(detid,self.filter)
            if key not in self.siaf_ap_names:
                key = 'NRC{}_SUB{}'.format(detid,self.det_info['xpix'])
        # Full frame generic
        elif wind_mode=='FULL':
            key = 'NRC{}_FULL'.format(detid)
        else:
            key = None

        # Check if key exists
        if key in self.siaf_ap_names:
            _log.info('Suggested SIAF aperture name: {}'.format(key))
            return key
        else:
            _log.warning("Suggested SIAF aperture name '{}' is not defined".format(key))
            return None

    def get_subarray_name(self, apname=None):
        """Get JWST NIRCam subarray name"""

        if apname is None:
            apname = self.get_siaf_apname()

        pupil_mask = self.pupil_mask
        image_mask = self.image_mask 
        module = self.module

        detid = self.Detector.detid
        wind_mode = self.Detector.wind_mode
        ypix = self.det_info['ypix']

        is_lyot = self.is_lyot
        is_coron = self.is_coron
        is_grism = self.is_grism
        is_ndacq = self.ND_acq

        if 'FULL' in wind_mode:
            subarray_name = 'FULLP' if apname[-1] == 'P' else 'FULL'
        elif 'STRIPE' in wind_mode:
            subarray_name = f'SUBGRISM{ypix}'
        elif is_coron:
            sub_str = f'SUB{ypix}'
            mask_str = image_mask[4:]
            if ('335R' in image_mask) and (module == 'A'):
                subarray_name = sub_str + module
            else:
                subarray_name = sub_str + module + mask_str
        # Just Lyot stop without masks, assuming TA aperture
        elif is_lyot:
            mask_str = image_mask[4:]
            # Faint source TA
            if not is_ndacq:
                subarray_name = 'SUBFS' + module + mask_str
            elif 'LWB' in image_mask: # ND TA
                if 'LWBL' in apname:
                    subarray_name = 'SUBND' + module + 'LWBL'
                else:
                    subarray_name = 'SUBND' + module + 'LWBS'
            elif 'SWB' in image_mask: # ND TA
                if 'SWBS' in apname:
                    subarray_name = 'SUBND' + module + 'LWBS'
                else:
                    subarray_name = 'SUBND' + module + 'LWBL'
            else:
                subarray_name = 'SUBND' + module + mask_str
        else:
            subarray_name = f'SUB{ypix}P' if apname[-1] == 'P' else f'SUB{ypix}'
        # TODO: Grism TS TA, Fine phasing (FP), and DHS
        
        return subarray_name
        
    def update_from_SIAF(self, apname, image_mask=None, pupil_mask=None, **kwargs):
        """Update detector properties based on SIAF aperture"""

        if apname is None:
            _log.warning('update_from_SIAF: Input apname was None. Returning...')
            return

        if not (apname in self.siaf_ap_names):
            # raise ValueError(f'Cannot find {apname} in siaf.apernames list.')
            _log.warning(f'update_from_SIAF: Cannot find {apname} in siaf.apernames list. Returning...')
            return
            
        if ('NRCALL' in apname) or ('NRCAS' in apname) or ('NRCBS' in apname):
            raise ValueError(f'{apname} is not valid. Single detector apertures only.')
            
        # Convert SCA name to detector ID
        scaname = apname[0:5]
        module = scaname[3]
        channel = 'LW' if scaname[-1]=='5' else 'SW'
        detid = 480 + int(scaname[4]) if module=='A' else 485 + int(scaname[4]) 
        
        siaf_ap = self.siaf[apname]
        xpix = int(siaf_ap.XSciSize)
        ypix = int(siaf_ap.YSciSize)
        if (xpix >= 2048) and (ypix>=2048):
            wind_mode = 'FULL'
        elif (xpix >= 2048):
            wind_mode = 'STRIPE'
        else:
            wind_mode = 'WINDOW'
        
        # Get lower left corner from siaf info
        # This is in full frame detector coordinates
        x0, y0 = np.array(siaf_ap.dms_corner()) - 1
              
        # Update pupil and mask info
        ND_acq = False
        filter = None
        # Coronagraphic mask observations
        if ('MASK' in apname) or ('FULL_WEDGE' in apname):
            # Set default pupil
            if pupil_mask is None:
                if ('WB' in apname) or ('BAR' in apname):
                    pupil_mask = 'WEDGELYOT'
                elif ('210R' in apname) or ('335R' in apname) or ('430R' in apname) or ('RND' in apname):
                    pupil_mask = 'CIRCLYOT'
                else:
                    _log.warning(f'No Lyot pupil setting for {apname}')

            # Set mask occulter for all full arrays (incl. TAs) and science subarrays
            # Treats full array TAs like a full coronagraphic observation
            if image_mask is not None:
                pass
            elif ('FULL' in apname) or ('_MASK' in apname):
                if ('MASKSWB' in apname):
                    image_mask  = 'MASKSWB'
                elif ('MASKLWB' in apname):
                    image_mask  = 'MASKLWB'            
                elif ('MASK210R' in apname):
                    image_mask  = 'MASK210R'
                elif ('MASK335R' in apname):
                    image_mask  = 'MASK335R'
                elif ('MASK430R' in apname):
                    image_mask  = 'MASK430R'
                if 'TA' in apname:
                    _log.warning('Full TA apertures are treated similar to coronagraphic observations.')
                    _log.warning("To calculate SNR, self.update_psf_coeff(image_mask='CLEAR') and set self.ND_acq.")
            elif '_TAMASK' in apname:
                # For small TA subarray, turn off mask and enable ND square
                image_mask = None
                ND_acq = True
            elif '_FSTAMASK in apname':
                # Not really anything to do here
                image_mask = None
            else:
                _log.warning(f'No mask setting for {apname}')

        # Grism observations
        elif 'GRISM' in apname:
            if ('_GRISMC' in apname): # GRISMC WFSS
                pupil_mask = 'GRISMC' if pupil_mask is None else pupil_mask
            elif ('_GRISMR' in apname): # GRISMR WFSS
                pupil_mask = 'GRISMR' if pupil_mask is None else pupil_mask
            elif ('_GRISMTS' in apname): # SW apertures in parallel w/ LW GRISMTS
                pupil_mask = 'WLP8' if pupil_mask is None else pupil_mask
            elif ('_TAGRISMTS' in apname): # GRISM TA have no pupil
                pupil_mask = None
            elif ('_GRISM' in apname): # Everything else is GRISMR
                pupil_mask = 'GRISMR' if pupil_mask is None else pupil_mask
            else:
                _log.warning(f'No grism setting for {apname}')


        # Look for filter specified in aperture name
        if ('_F1' in apname) or ('_F2' in apname) or ('_F3' in apname) or ('_F4' in apname):
            # Find all instances of "_"
            inds = [pos for pos, char in enumerate(apname) if char == '_']
            # Filter is always appended to end, but can have different string sizes (F322W2)
            filter = apname[inds[-1]+1:]
            # If filter doesn't make sense with channel
            if channel=='SW' and filter not in self._filters_sw:
                filter = None
            if channel=='LW' and filter not in self._filters_lw:
                filter = None

        # Save to internal variables
        self.pupil_mask = pupil_mask
        self.image_mask = image_mask
        self._ND_acq = ND_acq

        # Filter stuff
        # Defaults
        fsw_def, flw_def = ('F210M', 'F335M') 
        if filter is not None: 
            self.filter = filter
        try:
            if self._filter is None:
                self._filter = fsw_def if 'SW' in channel else flw_def
        except AttributeError:
            self._filter = fsw_def if 'SW' in channel else flw_def
        # If filter doesn't make sense with channel
        if channel=='SW' and self._filter not in self._filters_sw:
            self._filter = fsw_def
        if channel=='LW' and self._filter not in self._filters_lw:
            self._filter = flw_def
	
        # For NIRCam, update detector depending mask and filter
        self._update_coron_detector()
        self.detector = get_detname(scaname)
        self._update_coron_detector()

        self._validate_wheels()

        # Update detector settings
        det_kwargs = {'xpix': xpix, 'ypix': ypix, 'x0': x0, 'y0': y0, 'wind_mode':wind_mode}
        kwargs = merge_dicts(kwargs, det_kwargs)
        self.update_detectors(**kwargs)

        # Update aperture
        self.siaf_ap = siaf_ap

        # Update detector position to default of aperture
        ap_webbpsf = self.siaf[self.aperturename]
        self.detector_position = ap_webbpsf.det_to_sci(siaf_ap.XDetRef, siaf_ap.YDetRef)


    def calc_psf_from_coeff(self, sp=None, return_oversample=True, return_hdul=True,
        wfe_drift=None, coord_vals=None, coord_frame='tel', use_bg_psf=False, **kwargs):

        kwargs['sp'] = sp
        kwargs['return_oversample'] = return_oversample
        kwargs['return_hdul'] = return_hdul
        kwargs['wfe_drift'] = wfe_drift
        kwargs['coord_vals'] = coord_vals
        kwargs['coord_frame'] = coord_frame

        if use_bg_psf:
            return self._nrc_bg.calc_psf_from_coeff(**kwargs)
        else:
            return super().calc_psf_from_coeff(**kwargs)

    def calc_psf(self, sp=None, return_oversample=True, return_hdul=True,
        wfe_drift=None, coord_vals=None, coord_frame='tel', use_bg_psf=False, 
        **kwargs):

        kwargs['sp'] = sp
        kwargs['return_oversample'] = return_oversample
        kwargs['return_hdul'] = return_hdul
        kwargs['wfe_drift'] = wfe_drift
        kwargs['coord_vals'] = coord_vals
        kwargs['coord_frame'] = coord_frame

        # # Print coordinates in 'sci' frame
        # if (coord_vals is None) or (coord_frame is None):
        #     print(f'sci coords: {(self.siaf_ap.XSciRef, self.siaf_ap.YSciRef)}')
        # elif coord_frame=='sci':
        #     print(f'sci coords: {coord_vals}')
        # else:
        #     cvals_sci = self.siaf_ap.convert(coord_vals[0], coord_vals[1], coord_frame, 'sci')
        #     print(f'sci coords: {cvals_sci} (convert from {coord_frame})')

        _log.info("Calculating PSF from WebbPSF parent function")
        log_prev = conf.logging_level
        setup_logging('WARN', verbose=False)

        if use_bg_psf:
            res = self._nrc_bg.calc_psf(**kwargs)
        else:
            res = super().calc_psf(**kwargs)

        setup_logging(log_prev, verbose=False)
        return res
    

    def calc_psf_offset_from_center(self, use_coeff=True, halfwidth=None):
        """Calculate the offset necessary to shift PSF to array center
        
        Returns values in detector-sampled pixels.

        The array center is the middle of a pixel for odd images, 
        and at pixel boundaries for even images.
        """

        from webbpsf_ext.imreg_tools import recenter_psf

        _log.info("Calculating PSF offset to center of array...")

        calc_psf_func = self.calc_psf_from_coeff if use_coeff else self.calc_psf
        psf_over = calc_psf_func(return_oversample=True, return_hdul=False, use_bg_psf=True,
                                 coord_vals=(0,0), coord_frame='idl')

        oversample = self.oversample

        # Determine shift amount to place PSF in center of array
        if halfwidth is None:
            if self.is_lyot:
                if oversample==1:
                    halfwidth=1
                elif oversample<=3:
                    # Prevent special case COM algorithm from not converging
                    if ('LWB' in self.aperturename) and 'F4' in self.filter:
                        halfwidth=5
                    else:
                        halfwidth=3
                elif oversample<=5:
                    halfwidth=7
            else:
                halfwidth = 15
        _, xyoff_psf_over = recenter_psf(psf_over, niter=3, halfwidth=halfwidth)

        # Convert to detector pixels
        xyoff_psf = np.array(xyoff_psf_over) / oversample

        self.psf_offset_to_center = xyoff_psf
        # print(f"PSF offset to center: {xyoff_psf[0]:.3f}, {xyoff_psf[1]:.3f}")

    def recenter_psf(self, psf, sampling=1, 
                     shift_func=fourier_imshift, interp='cubic', **kwargs):
        """Recenter PSF to array center"""

        xsh_to_cen, ysh_to_cen = self.psf_offset_to_center * sampling
        kwargs['interp'] = interp

        # print(f"Recentering PSF: ({xsh_to_cen/sampling:.3f}, {ysh_to_cen/sampling:.3f}) pixels")

        return shift_func(psf, xsh_to_cen, ysh_to_cen, **kwargs)


    def sat_limits(self, sp=None, bp_lim=None, units='vegamag', well_frac=0.8,
        ngroup=None, trim_psf=33, verbose=False, **kwargs):
        """Saturation limits.        
        
        Generate the limiting magnitude (80% saturation) with the current instrument
        parameters (filter and ramp settings) assuming some spectrum. If no spectrum
        is defined, then a G2V star is assumed.

        The user can also define a separate bandpass in which to determine the
        limiting magnitude that will cause the current NIRCam bandpass to saturate.

        Parameters
        ----------
        sp : :class:`webbpsf_ext.synphot_ext.Spectrum`
            Spectrum to determine saturation limit.
        bp_lim : :class:`webbpsf_ext.synphot_ext.Bandpass`
            Bandpass to report limiting magnitude.
        units : str
            Output units (defaults to vegamag).
        well_frac : float
            Fraction of full well to consider 'saturated'.
        ngroup : int, None
            Option to specify the number of groups to determine
            integration time. If not set, then the default is to
            use those specified in the Detectors class. Can set
            ngroup=0 for the so-called Zero Frame in the event
            there are multiple reads per group.
        trim_psf : int, None
            Option to crop the PSF coefficient around the brightest pixel.
            For PSFs with large `fov_pix` values, this option helps speed
            up the saturation limit calculation. Afterall, we're usually
            only interested in the brightest pixel when calculating
            saturation limits. Set to `None` to use the 'fov_pix' value.
            Default = 33 (detector pixels).
        verbose : bool
            Print result details.

        Example
        -------
        >>> nrc = pynrc.NIRCam('F430M') # Initiate NIRCam observation
        >>> sp_A0V = pynrc.stellar_spectrum('A0V') # Define stellar spectral type
        >>> bp_k = pynrc.bp_2mass('k') # synphot K-Band bandpass
        >>> mag_lim = nrc.sat_limits(sp_A0V, bp_k, verbose=True)
        
        Returns K-Band Limiting Magnitude for F430M assuming A0V source.
        """	

        from webbpsf_ext.psfs import gen_image_from_coeff
        from copy import deepcopy

        bp_lim = self.bandpass if bp_lim is None else bp_lim
        quiet = False if verbose else True

        # Total time spent integrating minus the reset frame
        if ngroup is None:
            t_sat = self.multiaccum_times['t_int']
        else:
            t_frame = self.multiaccum_times['t_frame']
            if ngroup==0:
                t_sat = t_frame
            else:
                ma = self.multiaccum
                nf = ma.nf; nd1 = ma.nd1; nd2 = ma.nd2
                t_sat = (nd1 + ngroup*nf + (ngroup-1)*nd2) * t_frame

        # Full well level
        well_level = self.well_level

        # kwargs = merge_dicts(kwargs, self._psf_info)

        # We don't necessarily need the entire image, so cut down to size
        # 1. Create a temporary image at bp avg wavelength (monochromatic)
        # 2. Find x,y position of max PSF
        # 3. Cut out postage stamp region around that PSF coeff
        psf_coeff = self.psf_coeff
        psf_coeff_hdr = deepcopy(self.psf_coeff_header)
        fov_pix, osamp = (psf_coeff_hdr['FOVPIX'], psf_coeff_hdr['OSAMP'])
        if (trim_psf is not None) and (trim_psf < fov_pix):
            
            # Quickly create a temporary PSF to find max value location
            # wtemp = np.array([bp_lim.wave[0], bp_lim.avgwave(), bp_lim.wave[-1]])
            # ttemp = np.array([bp_lim.sample(w) for w in wtemp])
            # bptemp = ArrayBandpass(wave=wtemp, throughput=ttemp)
            # psf_temp, psf_temp_over = gen_image_coeff(bptemp, coeff=psf_coeff, coeff_hdr=psf_coeff_hdr, \
            #     fov_pix=fov_pix, oversample=osamp, return_oversample=True)

            res = gen_image_from_coeff(self, psf_coeff, psf_coeff_hdr, nwaves=3, return_oversample=True)
            if self.is_grism:
                _, psf_temp_over = res
            else:
                psf_temp_over = res

            # Amount to shift PSF
            yind, xind = np.argwhere(psf_temp_over==psf_temp_over.max())[0]
            ypix, xpix = psf_temp_over.shape
            ysh = int(yind - ypix/2)
            xsh = int(xind - xpix/2)

            fov_pix_over = trim_psf * osamp
            coeff = []
            for im in psf_coeff:
                im = fshift(im, -xsh, -ysh, interp='cubic')
                im = pad_or_cut_to_size(im, (fov_pix_over,fov_pix_over))
                coeff.append(im)
            psf_coeff = np.array(coeff)
            psf_coeff_hdr['FOVPIX'] = trim_psf

        satlim = saturation_limits(self, psf_coeff=psf_coeff, psf_coeff_hdr=psf_coeff_hdr, sp=sp, units=units, 
                                   bp_lim=bp_lim, int_time=t_sat, full_well=well_level, well_frac=well_frac,
                                   verbose=verbose, **kwargs)

        return satlim


    def saturation_levels(self, sp=None, full_size=True, ngroup=2, image=None, charge_migration=True, **kwargs):
        """ Saturation levels
        
        Create image showing level of saturation for each pixel.
        
        Parameters
        ----------
        sp : :class:`webbpsf_ext.synphot_ext.Spectrum`
            A synphot spectral object (normalized).
        full_size : bool
            Expand (or contract) to size of detector array?
            If False, use fov_pix size.
        ngroup : int
            How many group times to determine saturation level?
            If this number is higher than the total groups in ramp, 
            then a warning is produced. The default is ngroup=2, 
            A value of 0 corresponds to the so-called "zero-frame," 
            which is the very first frame that is read-out and saved 
            separately. This is the equivalent to ngroup=1 for RAPID
            and BRIGHT1 observations.
        image : ndarray
            Rather than generating an image on the fly, pass a pre-computed
            slope image. Overrides `sp` and `full_size`
        charge_migration : bool
            Include charge migration effects?

        Keyword Args
        ------------
        satmax : float
            Saturation value to limit charge migration. Default is 1.5.
        niter : int
            Number of iterations for charge migration. Default is 5.
        corners : bool
            Include corner pixels in charge migration? Default is True.

        """
        
        assert ngroup >= 0
        
        is_grism = self.is_grism

        t_frame = self.multiaccum_times['t_frame']
        t_int = self.multiaccum_times['t_int']
        if ngroup==0:
            t_sat = t_frame
        else:
            ma = self.multiaccum
            nf = ma.nf; nd1 = ma.nd1; nd2 = ma.nd2
            t_sat = (nd1 + ngroup*nf + (ngroup-1)*nd2) * t_frame
        
        if t_sat>t_int:
            _log.warning('ngroup*t_group is greater than t_int.')
    
        # Slope image of input 
        if image is not None:
            sat_level = image * t_sat / self.well_level
        else:

            image = self.calc_psf_from_coeff(sp=sp, return_oversample=False, return_hdul=False)
            if is_grism: 
                wave, image = image
            else: 
                wave = None
            
            if full_size:
                shape = (self.det_info['ypix'], self.det_info['xpix'])
                image = pad_or_cut_to_size(image, shape)
            
            # Add in zodi background to full image
            image += self.bg_zodi(**kwargs)

            # Well levels after "saturation time"
            sat_level = image * t_sat / self.well_level

        # Add in charge migration effects
        if charge_migration:
            sat_level = do_charge_migration(sat_level, **kwargs)

        if wave is None:
            return sat_level
        else:
            return (wave, sat_level)

    def sensitivity(self, nsig=10, units=None, sp=None, verbose=False, **kwargs):
        """Sensitivity limits.
        
        Convenience function for returning the point source (and surface brightness)
        sensitivity for the given instrument setup. See `sensitivities` function 
        for more details.

        Parameters
        ----------
        nsig : int, float
            Desired nsigma sensitivity (default 10).
        units : str
            Output units (defaults to uJy for grisms, nJy for imaging).
        sp : :class:`webbpsf_ext.synphot_ext.Spectrum`
            Input spectrum to use for determining sensitivity.
            Only the spectral shape matters, unless ``forwardSNR=True``.
        verbose : bool
            Print result details.
            
        Keyword Args
        ------------
        forwardSNR : bool
            Find the SNR of the input spectrum instead of sensitivity.
        zfact : float
            Factor to scale Zodiacal spectrum (default 2.5)
        ideal_Poisson : bool
            If set to True, use total signal for noise estimate,
            otherwise MULTIACCUM equation is used.
        rad_EE : float
            Extraction aperture radius (in pixels) for imaging mode.
        dw_bin : float
            Delta wavelength for spectral sensitivities (grisms & DHS).
        ap_spec : int, float
            Instead of dw_bin, specify the spectral extraction aperture in pixels.
            Takes priority over dw_bin. Value will get rounded up to nearest int.
        """	

        tf = self.multiaccum_times['t_frame']

        det = self.Detector
        ktc = det.ktc
        rn = det.read_noise
        idark = det.dark_current
        p_excess = det.p_excess

        pupil_mask = '' if self.pupil_mask is None else self.pupil_mask

        kw1 = self.multiaccum.to_dict()
        kw2 = {'rn':rn, 'ktc':ktc, 'idark':idark, 'p_excess':p_excess}
        kwargs = merge_dicts(kwargs,kw1,kw2)
        if 'ideal_Poisson' not in kwargs.keys():
            kwargs['ideal_Poisson'] = True
            
        # Always use the bg coeff
        psf_coeff = self._nrc_bg.psf_coeff
        psf_coeff_hdr = self._nrc_bg.psf_coeff_header.copy()
        fov_pix, osamp = (psf_coeff_hdr['FOVPIX'], psf_coeff_hdr['OSAMP'])
        # We don't necessarily need the entire image, so cut down to size for speed
        if (not ('WEAK LENS' in pupil_mask)) and (fov_pix > 33):
            fov_pix = 33
            fov_pix_over = fov_pix * osamp
            psf_coeff = np.array([pad_or_cut_to_size(im, (fov_pix_over,fov_pix_over)) for im in psf_coeff])
            kwargs['fov_pix'] = fov_pix
            psf_coeff_hdr['FOVPIX'] = fov_pix

        bglim = sensitivities(self, psf_coeff=psf_coeff, psf_coeff_hdr=psf_coeff_hdr,
            sp=sp, units=units, nsig=nsig, tf=tf, verbose=verbose, **kwargs)
    
        return bglim

    def bg_zodi(self, zfact=None, **kwargs):
        """Zodiacal background flux.
        
        There are options to call `jwst_backgrounds` to obtain better 
        predictions of the background. Specify keywords `ra`, `dec`, 
        and `thisday` to use `jwst_backgrounds`.
        
        Returned values are in units of e-/sec/pixel

        Parameters
        ----------
        zfact : float
            Factor to scale Zodiacal spectrum (default 2.5)

        Keyword Args
        ------------
        ra : float
            Right ascension in decimal degrees
        dec : float
            Declination in decimal degrees
        thisday : int
            Calendar day to use for background calculation.  
            If not given, will use the average of visible calendar days.

        Notes
        -----
        Representative values for zfact:

            * 0.0 - No zodiacal emission
            * 1.0 - Minimum zodiacal emission from JWST-CALC-003894
            * 1.2 - Required NIRCam performance
            * 2.5 - Average (default)
            * 5.0 - High
            * 10.0 - Maximum

        """
    
        # Dark image
        if self.is_dark:
            return 0

        bp = self.bandpass
        sp_zodi   = zodi_spec(zfact, **kwargs)
        obs_zodi  = Observation(sp_zodi, bp, bp.waveset)
        fzodi_pix = obs_zodi.countrate() * (self.pixelscale/206265.0)**2
        
        # Recommend a zfact value if ra, dec, and thisday specified
        if 'ra' in kwargs.keys():
            sp_zodi_temp   = zodi_spec(zfact=1)
            obs_zodi_temp  = Observation(sp_zodi_temp, bp, bp.waveset)
            fzodi_pix_temp = obs_zodi_temp.countrate() * (self.pixelscale/206265.0)**2
            zf_rec = fzodi_pix / fzodi_pix_temp
            str1 = 'Using ra,dec,thisday keywords can be relatively slow. \n'
            str2 = '\tFor your specified loc and date, we recommend using zfact={:.1f}'.format(zf_rec)
            _log.warning(str1 + str2)

        # Don't forget about Lyot mask attenuation (not in bandpass throughput)
        if self.is_lyot:
            fzodi_pix *= 0.19

        return fzodi_pix

    def bg_zodi_image(self, zfact=None, frame='sci', **kwargs):
        """Zodiacal light image
        
        Returns an image of background Zodiacal light emission
        in e-/sec in specified coordinate frame.

        Parameters
        ----------
        zfact : float
            Factor to scale Zodiacal spectrum (default 2.5)
        frame : str
            Return in 'sci' or 'det' coordinates?

        Keyword Args
        ------------
        ra : float
            Right ascension in decimal degrees
        dec : float
            Declination in decimal degrees
        thisday : int
            Calendar day to use for background calculation.  
            If not given, will use the average of visible calendar days.

        Notes
        -----
        Representative values for zfact:

            * 0.0 - No zodiacal emission
            * 1.0 - Minimum zodiacal emission from JWST-CALC-003894
            * 1.2 - Required NIRCam performance
            * 2.5 - Average (default)
            * 5.0 - High
            * 10.0 - Maximum
        """

        detid = self.Detector.detid
        x0, y0 = (self.det_info['x0'], self.det_info['y0'])
        xpix, ypix = (self.det_info['xpix'], self.det_info['ypix'])

        # Dark image
        if self.is_dark:
            return np.zeros([ypix,xpix])

        bp = self.bandpass
        sp_zodi   = zodi_spec(zfact, **kwargs)
        obs_zodi  = Observation(sp_zodi, bp, bp.waveset)
        fzodi_pix = obs_zodi.countrate() * (self.pixelscale/206265.0)**2

        # Get equivalent 
        if 'ra' in kwargs.keys():
            sp_zodi_temp   = zodi_spec(zfact=1)
            obs_zodi_temp  = Observation(sp_zodi_temp, bp, bp.waveset)
            fzodi_pix_temp = obs_zodi_temp.countrate() * (self.pixelscale/206265.0)**2
            zfact = fzodi_pix / fzodi_pix_temp
            _ = kwargs.pop('ra')
            _ = kwargs.pop('dec')
            _ = kwargs.pop('thisday')

        filter = self.filter
        pupil_mask = self.pupil_mask

        if self.is_grism:
            # sci coords
            im_bg = grism_background_image(filter, pupil=pupil_mask, module=self.module, sp_bg=sp_zodi, **kwargs)
            # Convert to det coords and crop
            im_bg = sci_to_det(im_bg, detid)
            im_bg = im_bg[y0:y0+ypix, x0:x0+xpix]
            # Back to sci coords
            im_bg = det_to_sci(im_bg, detid)
        elif self.is_coron or self.coron_substrate:
            # Create full image, then crop based on detector configuration
            im_bg = build_mask_detid(detid, oversample=1, pupil=pupil_mask, filter=self.filter)
            if im_bg is None:
                # In the event the specified detid has no coronagraphic mask
                # This includes ['A1', 'A3', 'B2', 'B4']
                im_bg = np.ones([ypix,xpix])
            else:
                # Convert to det coords and crop
                im_bg = sci_to_det(im_bg, detid)
                im_bg = im_bg[y0:y0+ypix, x0:x0+xpix]
                # Back to sci coords and multiply by e-/sec/pix
                im_bg = det_to_sci(im_bg, detid)

            # Multiply by e-/sec/pix
            im_bg *= self.bg_zodi(zfact, **kwargs)
        else:
            # No spatial structures for direct imaging an certain Lyot masks.
            im_bg = np.ones([ypix,xpix]) * self.bg_zodi(zfact, **kwargs)

        # Clear reference pixels
        # im_bg = sci_to_det(im_bg, detid)
        # mask_ref = self.Detector.mask_ref
        # im_bg[mask_ref] = 0
        # im_bg = det_to_sci(im_bg, detid)

        if frame=='det':
            return sci_to_det(im_bg, detid)
        elif frame=='sci':
            return im_bg
        else:
            raise ValueError(f"frame {frame} not recognized. Use either 'sci' or 'det'.")

        

    def ramp_optimize(self, sp, sp_bright=None, is_extended=False, patterns=None,
        snr_goal=None, snr_frac=0.02, tacq_max=None, tacq_frac=0.1,
        well_frac_max=0.8, nint_min=1, nint_max=5000, ng_min=2, ng_max=None,
        return_full_table=False, even_nints=False, verbose=False, **kwargs):
        """Optimize ramp settings.
        
        Find the optimal ramp settings to observe a spectrum based on input constraints.
        This function quickly runs through each detector readout pattern and 
        calculates the acquisition time and SNR for all possible settings of NINT
        and NGROUP that fulfill the SNR requirement (and other constraints). 

        The final output table is then filtered, removing those exposure settings
        that have the same exact acquisition times but worse SNR. Further "obvious"
        comparisons are done that exclude settings where there is another setting
        that has both better SNR and less acquisition time. The best results are
        then sorted by an efficiency metric (SNR / sqrt(acq_time)). To skip filtering
        of results, set return_full_table=True.

        The result is an AstroPy Table.

        Parameters
        ----------
        sp : :class:`webbpsf_ext.synphot_ext.Spectrum`
            A synphot spectral object to calculate SNR.
        sp_bright : :class:`webbpsf_ext.synphot_ext.Spectrum`, None
            Same as sp, but optionally used to calculate the saturation limit
            (treated as brightest source in field). If a coronagraphic mask 
            observation, then this source is assumed to be occulted and 
            sp is fully unocculted.
        is_extended : bool
            Treat sp source as extended object, then in units/arcsec^2

        snr_goal : float
            Minimum required SNR for source. For grism, this is the average
            SNR for all wavelength.
        snr_frac : float
            Give fractional buffer room rather than strict SNR cut-off.
        tacq_max : float
            Maximum amount of acquisition time in seconds to consider.
        tacq_frac : float
            Fractional amount of time to consider exceeding tacq_max.

        patterns : numpy array
            Subset of MULTIACCUM patterns to check, otherwise check all.
        nint_min/max  : int
            Min/max number of desired integrations.
        ng_min/max : int
            Min/max number of desired groups in a ramp.
        well_frac_max : float
            Maximum level that the pixel well is allowed to be filled. 
            Fractions greater than 1 imply hard saturation, but the reported 
            SNR will not be aware of any saturation that may occur to sp.

        even_nints  : bool
            Return only the even NINTS
        return_full_table : bool
            Don't filter or sort the final results (ingores event_ints).
        verbose : bool
            Prints out top 10 results.

        Keyword Args
        ------------
        zfact : float
            Factor to scale Zodiacal spectrum (default 2.5)
        ra : float
            Right ascension in decimal degrees
        dec : float
            Declination in decimal degrees
        thisday : int
            Calendar day to use for background calculation.  
            If not given, will use the average of visible calendar days.

        ideal_Poisson : bool
            Use total signal for noise estimate?
            Otherwise MULTIACCUM equation is used. 
            Default = True
            
        rad_EE : int
            Extraction aperture radius (in pixels) for imaging mode.
        dw_bin : float
            Delta wavelength to calculate spectral sensitivities for
            grisms and DHS.
        ap_spec : float, int
            Instead of dw_bin, specify the spectral extraction aperture 
            in pixels. Takes priority over dw_bin. Value will get rounded 
            up to nearest int.
        
        Note
        ----
        The keyword arguments ra, dec, thisday are not recommended for use 
        given the amount of time it takes to query the web server. 
        Instead, use :meth:`bg_zodi` to match a zfact estimate.
                  
        Returns
        -------
        astropy table
            A sorted and filtered table of ramp options.

        """

        def parse_snr(snr, grism_obs, ind_snr):
            if grism_obs:
                res = snr['snr']
                return np.median(res)
            else:
                return snr[ind_snr]['snr']            


        pupil_mask = self.pupil_mask

        grism_obs = self.is_grism
        dhs_obs   = (pupil_mask is not None) and ('DHS' in pupil_mask)

        det_params_orig = self.det_info.copy()

        if dhs_obs:
            raise NotImplementedError('DHS has yet to be fully included.')
        if grism_obs and is_extended:
            raise NotImplementedError('Extended objects not implemented for grism observations.')
    
        if (snr_goal is not None) and (tacq_max is not None):
            raise ValueError('Keywords snr_goal and tacq_max are mutually exclusive.')
        if (snr_goal is None) and (tacq_max is None):
            raise ValueError('Must set either snr_goal or tacq_max.')

        # Brightest source in field
        if sp_bright is None:
            sp_bright = sp

        gen_psf = self.calc_psf_from_coeff
        kw_gen_psf = {'return_oversample': False,'return_hdul': False}

        # Generate PSFs for faint and bright objects and get max pixel flux
        # Only necessary for point sources
        if is_extended:
            ind_snr = 1
            obs = Observation(sp, self.bandpass, binset=self.bandpass.waveset)
            psf_faint = obs.countrate() * self.pixelscale**2
            psf_bright = gen_psf(sp=sp_bright, use_bg_psf=False, **kw_gen_psf)
            pix_count_rate = np.max([psf_bright.max(), psf_faint])
        else:
            ind_snr = 0

            if grism_obs:
                _, psf_bright = gen_psf(sp=sp_bright, use_bg_psf=False, **kw_gen_psf)
                _, psf_faint  = gen_psf(sp=sp, use_bg_psf=True, **kw_gen_psf)
            else:
                psf_bright = gen_psf(sp=sp_bright, use_bg_psf=False, **kw_gen_psf)
                psf_faint  = gen_psf(sp=sp, use_bg_psf=True, **kw_gen_psf)
            pix_count_rate = np.max([psf_bright.max(), psf_faint.max()])

        image = self.sensitivity(sp=sp, forwardSNR=True, return_image=True, **kwargs)

        # Correctly format patterns
        pattern_settings = self.multiaccum._pattern_settings
        if patterns is None:
            patterns = list(pattern_settings.keys())
        if not isinstance(patterns, list):
            patterns = [patterns]
    
        m = np.zeros(len(patterns))
        s = np.zeros(len(patterns))
        for i,patt in enumerate(patterns):
            v1,v2,v3 = pattern_settings.get(patt)
            m[i] = v1
            s[i] = v2
        # Sort by nf (m+s) then by m
        isort = np.lexsort((m,m+s))
        patterns = list(np.array(patterns)[isort])
    
        patterns.sort()

        log_prev = conf.logging_level
        setup_logging("WARN", verbose=False)

        rows = []
        if tacq_max is not None:
            # Cycle through each readout pattern
            for read_mode in patterns:
                if verbose: print(read_mode)

                # Maximum allowed groups for given readout pattern
                _,_,ngroup_max = pattern_settings.get(read_mode)
                if ng_max is not None:
                    ngroup_max = ng_max
                nng = ngroup_max - ng_min + 1
                if nng>30:
                    _log.warning(f'Cycling through {nng} NGROUPs. This may take a while!')
                for ng in range(ng_min,ngroup_max+1):
                    self.update_detectors(read_mode=read_mode, ngroup=ng, nint=1)
                    mtimes = self.multiaccum_times

                    # Get saturation level of observation
                    # Total time spent integrating minus the reset frame
                    int_time = mtimes['t_int']

                    well_frac = pix_count_rate * int_time / self.well_level
                    # If above well_frac_max, then this setting is invalid
                    # Also, all subsequent values of ng will be too high
                    # so just break out of for loop.
                    if well_frac > well_frac_max:
                        break
            
                    # Approximate integrations needed to obtain required t_acq
                    nint1 = int(((1-tacq_frac)*tacq_max) / mtimes['t_acq'])
                    nint2 = int(((1+tacq_frac)*tacq_max) / mtimes['t_acq'] + 0.5)
            
                    nint1 = np.max([nint1,nint_min])
                    nint2 = np.min([nint2,nint_max])
            
                    nint_all = np.arange(nint1, nint2+1)
                    
                    narr = len(nint_all)
                    # Sometimes there are a lot of nint values to check
                    # Let's pair down to <5 per ng
                    if narr>5:
                        i1 = int(narr/2-2)
                        i2 = i1 + 5
                        nint_all = nint_all[i1:i2]
                
                    #print(len(nint_all))
                    for nint in nint_all:
                        if nint > nint_max: 
                            break
                        self.update_detectors(nint=nint)
                        mtimes = self.multiaccum_times
                        sen = self.sensitivity(sp=sp, forwardSNR=True, image=image, **kwargs)
                        snr = parse_snr(sen, grism_obs, ind_snr)
                
                        rows.append((read_mode, ng, nint, mtimes['t_int'], mtimes['t_exp'], \
                            mtimes['t_acq'], snr, well_frac))

        elif snr_goal is not None:
            for i,read_mode in enumerate(patterns):
                if verbose: print(read_mode)

                # Maximum allowed groups for given readout pattern
                _,_,ngroup_max = pattern_settings.get(read_mode)
                if ng_max is not None:
                    ngroup_max = ng_max #np.min([ng_max,ngroup_max])
                nng = ngroup_max - ng_min + 1
                if nng>20:
                    _log.warning(f'Cycling through {nng} NGROUPs. This may take a while!')
                        
                ng_saved = False
                for ng in range(ng_min,ngroup_max+1):
                    self.update_detectors(read_mode=read_mode, ngroup=ng, nint=1)
                    mtimes = self.multiaccum_times

                    # Get saturation level of observation
                    int_time = mtimes['t_int']
                    well_frac = pix_count_rate * int_time / self.well_level
                    # If above well_frac_max, then this setting is invalid
                    if well_frac > well_frac_max:
                        continue

                    # Get SNR (assumes no saturation)
                    sen = self.sensitivity(sp=sp, forwardSNR=True, image=image, **kwargs)
                    snr = parse_snr(sen, grism_obs, ind_snr)

                    # Approximate integrations needed to get to required SNR
                    nint = int((snr_goal / snr)**2)
                    nint = np.max([nint_min,nint])
                    if nint>nint_max:
                        continue
    
                    # Find NINT with SNR > 0.95 snr_goal
                    self.update_detectors(nint=nint)
                    mtimes = self.multiaccum_times
                    sen = self.sensitivity(sp=sp, forwardSNR=True, image=image, **kwargs)
                    snr = parse_snr(sen, grism_obs, ind_snr)
                    while (snr<((1-snr_frac)*snr_goal)) and (nint<=nint_max):
                        nint += 1
                        self.update_detectors(nint=nint)
                        mtimes = self.multiaccum_times
                        sen = self.sensitivity(sp=sp, forwardSNR=True, image=image, **kwargs)
                        snr = parse_snr(sen, grism_obs, ind_snr)
                
                    # Skip if NINT
                    if (nint > nint_max):# or :
                        continue
                        
                    # We want to make sure that at least one NINT setting is saved
                    # if the resulting SNR is higher than our stated goal.
                    if (snr > ((1+snr_frac)*snr_goal)) and ng_saved:
                        continue

                    rows.append((read_mode, ng, nint, mtimes['t_int'], mtimes['t_exp'], \
                        mtimes['t_acq'], snr, well_frac))
                    ng_saved = True

                    # Increment NINT until SNR > 1.05 snr_goal
                    # Add each NINT to table output
                    while (snr < ((1+snr_frac)*snr_goal)) and (nint<=nint_max):
                        nint += 1
                        if (nint > nint_max): break # double-check
                        self.update_detectors(nint=nint)
                        sen = self.sensitivity(sp=sp, forwardSNR=True, image=image, **kwargs)
                        snr = parse_snr(sen, grism_obs, ind_snr)
                        mtimes = self.multiaccum_times
                        rows.append((read_mode, ng, nint, mtimes['t_int'], mtimes['t_exp'], \
                            mtimes['t_acq'], snr, well_frac))
                
        # Return to detector mode to original parameters
        self.update_detectors(**det_params_orig)
        setup_logging(log_prev, verbose=False)

        names = ('Pattern', 'NGRP', 'NINT', 't_int', 't_exp', 't_acq', 'SNR', 'Well')
        if len(rows)==0:
            _log.warning('No ramp settings allowed within constraints! Reduce constraints.')
            return Table(names=names)

        # Place rows into a AstroPy Table
        t_all = Table(rows=rows, names=names)
        t_all['Pattern'].format = '<10'
        t_all['t_int'].format = '9.2f'
        t_all['t_exp'].format = '9.2f'
        t_all['t_acq'].format = '9.2f'
        t_all['SNR'].format   = '8.1f'
        t_all['Well'].format  = '8.3f'

        t_all['eff'] = t_all['SNR'] / np.sqrt(t_all['t_acq'])
        # Round to 3 sig digits
        t_all['eff'] = (1000*t_all['eff']).astype(int) / 1000.
        t_all['eff'].format = '8.3f'

        # Filter table?
        if return_full_table:
            # Sort by efficiency, then acq time
            ind_sort = np.lexsort((t_all['t_acq'],1/t_all['eff']))
            t_all = t_all[ind_sort]
            if verbose: 
                print("Top 10 results sorted by 'efficiency' [SNR/sqrt(t_acq)]:")
                print(t_all[0:10])
        else:
            t_all = table_filter(t_all, **kwargs)
            ind_sort = np.lexsort((t_all['t_acq'],1/t_all['eff']))
            t_all = t_all[ind_sort]
            # Select only even integrations
            if even_nints:
                ind = (t_all['NINT'] % 2 == 0)
                t_all = t_all[ind]

            if verbose: print(t_all)

        return t_all

    def gen_psfs_over_fov(self, sptype='G0V', wfe_drift=0, osamp=1, npsf_per_full_fov=15,
                          return_coords=None, use_coeff=True, **kwargs):
        """Create PSF grid over full field of view
        
        Wrapper around `calc_psfs_grid` that returns normalized PSFs across 
        the field of view. 
        
        Create a grid of PSFs across instrument aperture FoV. By default,
        imaging observations will be for full detector FoV with regularly
        spaced grid. Coronagraphic observations will cover nominal 
        coronagraphic mask region (usually 10s of arcsec) and will have
        logarithmically spaced values where appropriate.

        Parameters
        ==========
        sptype : str
            Spectral type, such as 'A0V' or 'K2III'.
        wfe_drift : float
            Desired WFE drift value relative to default OPD.
        osamp : int
            Sampling of output PSF relative to detector sampling.
        npsf_per_full_fov : int
            Number of PSFs across one dimension of the instrument's field of 
            view. If a coronagraphic observation, then this is for the nominal
            coronagrahic field of view.
        return_coords : None or str
            Option to also return coordinate values in desired frame 
            ('det', 'sci', 'tel', 'idl'). Output is then xvals, yvals, hdul_psfs.
        use_coeff : bool
            If True, uses `calc_psf_from_coeff`, other WebbPSF's built-in `calc_psf`.

        Keyword Args
        ============
        xsci_vals: None or ndarray
            Option to pass a custom grid values along x-axis in 'sci' coords.
            If coronagraph, this instead corresponds to coronagraphic mask axis, 
            which has a slight rotation in MIRI.
        ysci_vals: None or ndarray
            Option to pass a custom grid values along y-axis in 'sci' coords.
            If coronagraph, this instead corresponds to coronagraphic mask axis, 
            which has a slight rotation in MIRI.
        """

        # Create input spectrum that is star normalized by unit response
        bp = self.bandpass
        sp = stellar_spectrum(sptype, bp.unit_response(), 'flam', bp)

        return self.calc_psfs_grid(sp=sp, wfe_drift=wfe_drift, osamp=osamp,
                                   return_coords=return_coords, use_coeff=use_coeff,
                                   npsf_per_full_fov=npsf_per_full_fov, **kwargs)

    def _gen_obs_params(self, target_name, ra, dec, date_obs, time_obs, pa_v3=0, 
        siaf_ap_ref=None, xyoff_idl=(0,0), visit_type='SCIENCE', time_series=False,
        time_exp_offset=0, segNum=None, segTot=None, int_range=None, filename=None, **kwargs):

        """ Generate a simple obs_params dictionary

        An obs_params dictionary is used to create a jwst data model (e.g., Level1bModel).
        Additional ``**kwargs`` will add/update elements to the final output dictionary.

        Parameters
        ==========
        ra : float
            RA in degrees associated with observation pointing
        dec : float
            RA in degrees associated with observation pointing
        data_obs : str
            YYYY-MM-DD
        time_obs : str
            HH:MM:SS

        Keyword Arg
        ===========
        pa_v3 : float
            Telescope V3 position angle.
        siaf_ap_ref : pysiaf Aperture
            SIAF aperture class used for telescope pointing (if different than self.siaf_ap)
        xyoff_idl : tuple, list
            (x,y) offset in arcsec ('idl' coords) to dither observation
        visit_type : str
            'T_ACQ', 'CONFIRM', or 'SCIENCE'
        time_series : bool
            Is this a time series observation?
        time_exp_offset : float
            Exposure start time (in seconds) relative to beginning of observation execution. 
        segNum : int
            The segment number of the current product. Only for TSO.
        segTot : int
            The total number of segments. Only for TSO.
        int_range : list
            Integration indices to use 
        filename : str or None  
            Name of output filename. If set to None, then auto generates a dummy name.
        """
        from .simul.apt import create_obs_params
        from .simul.dms import DMS_filename

        filt = self.filter
        pupil = 'CLEAR' if self.pupil_mask is None else self.pupil_mask
        mask = 'None' if self.image_mask is None else self.image_mask
        det = self.Detector
        siaf_ap_obs = self.siaf_ap
        if siaf_ap_ref is None:
            siaf_ap_ref = self.siaf_ap
        ra_dec = (ra, dec)

        kwargs['target_name'] = target_name
        kwargs['nexposures'] = 1

        obs_params = create_obs_params(filt, pupil, mask, det, siaf_ap_ref, ra_dec, date_obs, time_obs,
            pa_v3=pa_v3, siaf_ap_obs=siaf_ap_obs, xyoff_idl=xyoff_idl, time_exp_offset=time_exp_offset, 
            visit_type=visit_type, time_series=time_series, segNum=segNum, segTot=segTot, int_range=int_range,
            filename=filename, **kwargs)

        if filename is None:
            obs_id_info = obs_params['obs_id_info']
            detname = det.detid
            filename = DMS_filename(obs_id_info, detname, segNum=segNum, prodType='uncal')
            obs_params['filename'] = filename

        return obs_params


    def simulate_ramps(self, sp=None, im_slope=None, cframe='sci', nint=None, 
        do_dark=False, rand_seed=None, **kwargs):
        """ Simulate Ramp Data

        Create a series of ramp data based on the current NIRCam settings. 
        This method calls the :func:`gen_ramp` function, which in turn calls 
        the detector noise generator :func:`~pynrc.simul.simulate_detector_ramp`.

        Parameters
        ----------
        im_slope : numpy array, None
            Pass the slope image directly. If not set, then a slope
            image will be created from the input spectrum keyword. This
            should include zodiacal light emission, but not dark current.
            Make sure this array is in detector coordinates.
        sp : :class:`webbpsf_ext.synphot_ext.Spectrum`, None
            A synphot spectral object. If not specified, then it is
            assumed that we're looking at blank sky.
        cframe : str
            Output coordinate frame, 'sci' or 'det'.
        nint : None or int
            Options to specify arbitrary number of integrations. 
        do_dark : bool
            Make a dark ramp (ie., pupil_mask='FLAT'), no external flux.

        Keyword Args
        ------------
        zfact : float
            Factor to scale Zodiacal spectrum (default 2.5)
        ra : float
            Right ascension in decimal degrees
        dec : float
            Declination in decimal degrees
        thisday : int
            Calendar day to use for background calculation. If not given, 
            will use the average of visible calendar days.

        return_full_ramp : bool
            By default, we average groups and drop frames as specified in the
            `det` input. If this keyword is set to True, then return all raw
            frames within the ramp. The last set of `nd2` frames will be omitted.
        out_ADU : bool
            If True, divide by gain and convert to 16-bit UINT.
        super_bias : ndarray or None
            Option to include a custom super bias image. If set to None, then
            grabs from ``cal_obj``. Should be the same shape as ``im_slope``.
        super_dark : ndarray or None
            Option to include a custom super dark image. If set to None, then
            grabs from ``cal_obj``. Should be the same shape as ``im_slope``.
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
        include_refoffsets : bool
            Include reference offsts between amplifiers and odd/even columns?
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
            Add cosmic ray events? See Robberto et al 2010 (JWST-STScI-001928).
        cr_model: str
            Cosmic ray model to use: 'SUNMAX', 'SUNMIN', or 'FLARES'.
        cr_scale: float
            Scale factor for probabilities.
        apply_nonlinearity : bool
            Apply non-linearity?
        random_nonlin : bool
            Add randomness to the linearity coefficients?
        apply_flats: bool
            Apply sub-pixel QE variations (crosshatching)?
        latents : None or ndarray
            (TODO) Apply persistence from previous integration.

        """
        from .reduce.calib import nircam_cal

        rng = np.random.default_rng(rand_seed)

        det = self.Detector
        nint = det.multiaccum.nint if nint is None else nint

        pupil_mask = 'FLAT' if do_dark else self.pupil_mask
        xpix = self.det_info['xpix']
        ypix = self.det_info['ypix']

        # Set logging to WARNING to suppress messages
        log_prev = conf.logging_level
        setup_logging('WARN', verbose=False)

        det_cal_obj = nircam_cal(self.scaid, verbose=False)

        # If requesting dark images
        if do_dark:
            im_slope = np.zeros([ypix,xpix])
        # If slope image is not specified
        elif im_slope is None:
            # Detector sampled images
            gen_psf = self.calc_psf_from_coeff
            kw_gen_psf = {'return_oversample': False,'return_hdul': False}

            # Imaging+Coronagraphy
            if pupil_mask is None:
                im_slope = gen_psf(sp=sp, **kw_gen_psf)
            # No visible source
            elif ('FLAT' in pupil_mask) or (sp is None):
                im_slope = np.zeros([ypix,xpix])
            # Grism spec
            elif ('GRISM' in pupil_mask):
                w, im_slope = gen_psf(sp=sp, **kw_gen_psf)
            # DHS spectroscopy
            elif ('DHS' in pupil_mask):
                raise NotImplementedError('DHS has yet to be fully included')
            # Imaging+Coronagraphy
            else:
                im_slope = gen_psf(sp=sp, **kw_gen_psf)

            # Expand or cut to detector size
            im_slope = pad_or_cut_to_size(im_slope, (ypix,xpix))

            # Add in Zodi emission
            # Returns 0 if self.pupil_mask='FLAT'
            im_slope += self.bg_zodi_image(**kwargs)
            
        # Minimum value of slope
        im_min = im_slope[im_slope>=0].min()
        # Expand or cut to detector size
        im_slope = pad_or_cut_to_size(im_slope, (ypix,xpix))
        # Make sure there are no negative numbers
        im_slope[im_slope<=0] = im_min

        # Create a list of arguments to pass
        worker_arguments = []
        for i in range(nint):
            rseed_i = rng.integers(0,2**32-1)
            kw = {'im_slope': im_slope, 'cframe': cframe, 
                  'return_zero_frame': True, 'rand_seed': rseed_i}
            kws = merge_dicts(kw, kwargs)

            args = (det, det_cal_obj)
            worker_arguments.append((args, kws))

        res_zeros = []
        res_ramps = []
        for wa in tqdm(worker_arguments, desc='Ramps', leave=False):
            out = gen_ramps(wa)
            res_ramps.append(out[0])
            res_zeros.append(out[1])
        
        setup_logging(log_prev, verbose=False)

        return np.asarray(res_ramps), np.asarray(res_zeros)

    def simulate_level1b(self, target_name, ra, dec, date_obs, time_obs, 
        sp=None, im_slope=None, cframe='sci', nint=None, do_dark=False, 
        save_dir=None, return_model=False, return_hdul=False, **kwargs):

        """ Simulate DMS Level 1b data model """

        from .simul.dms import level1b_data_model, save_level1b_fits
        from stdatamodels import fits_support

        # Update total number of integrations
        if nint is not None:
            nint_orig = self.Detector.multiaccum.nint
            self.update_detectors(nint=nint)

        kwargs['out_ADU'] = True
        sci_data, zero_data = self.simulate_ramps(sp=sp, im_slope=im_slope, cframe=cframe, nint=nint, 
            do_dark=do_dark, **kwargs)

        obs_params = self._gen_obs_params(target_name, ra, dec, date_obs, time_obs, **kwargs)
        obs_params['save_dir'] = save_dir

        outModel = level1b_data_model(obs_params, sci_data=sci_data, zero_data=zero_data)
        if save_dir:
            save_level1b_fits(outModel, obs_params, save_dir=save_dir)

        # Return number of integrations
        if nint is not None:
            self.update_detectors(nint=nint_orig)

        if return_hdul:
            out_hdul, out_asdf = fits_support.to_fits(outModel._instance, outModel._schema)

        if return_model and return_hdul:
            return outModel, out_hdul
        elif return_model:
            return outModel
        elif return_hdul:
            return out_hdul


def table_filter(t, topn=None, **kwargs):
    """Filter and sort table.
    
    Filter a resulting ramp table to exclude those with worse SNR for the same
    or larger tacq. This is performed on a pattern-specific basis and returns
    the Top N rows for each readout patten. The rows are ranked by an efficiency
    metric, which is simply SNR / sqrt(tacq). If topn is set to None, then all
    values that make the cut are returned (sorted by the efficiency metric).
    
    Args
    ----
    topn : int, None
        Maximum number of rows to keep.
    """

    if topn is None: topn = len(t)

    temp = multiaccum()
    pattern_settings = temp._pattern_settings

    patterns = np.unique(t['Pattern'])

    m = np.zeros(len(patterns))
    s = np.zeros(len(patterns))
    for i,patt in enumerate(patterns):
        v1,v2,v3 = pattern_settings.get(patt)
        m[i] = v1
        s[i] = v2
    # Sort by nf (m+s) then by m
    isort = np.lexsort((m,m+s))
    patterns = list(np.array(patterns)[isort])

    tnew = t.copy()
    tnew.remove_rows(np.arange(len(t)))

    for pattern in patterns:
        rows = t[t['Pattern']==pattern]

        # For equivalent acquisition times, remove worse SNR
        t_uniq = np.unique(rows['t_acq'])
        ind_good = []
        for tacq in t_uniq:
            ind = np.where(rows['t_acq']==tacq)[0]
            ind_snr_best = rows['SNR'][ind]==rows['SNR'][ind].max()
            ind_good.append(ind[ind_snr_best][0])
        rows = rows[ind_good]

        # For each remaining row, exlude those that take longer with worse SNR than any other row
        ind_bad = []
        ind_bad_comp = []
        for i,row in enumerate(rows):
            for j,row_compare in enumerate(rows):
                if i==j: continue
                if (row['t_acq']>row_compare['t_acq']) and (row['SNR']<=(row_compare['SNR'])):
                    ind_bad.append(i)
                    ind_bad_comp.append(j)
                    break
        rows.remove_rows(ind_bad)

        isort = np.lexsort((rows['t_acq'],1/rows['eff']))
        for row in rows[isort][0:topn]:
            tnew.add_row(row)

    return tnew

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict.
    If the same key appars multiple times, priority goes to key/value 
    pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def gen_ramps(args):
    """
    Helper function for generating FITs integrations from a slope image
    """
    from .simul.ngNRC import simulate_detector_ramp

    args_orig, kwargs = args
    try:
        res = simulate_detector_ramp(*args_orig, **kwargs)
    except Exception as e:
        print('Caught exception in worker thread:')
        # This prints the type, value, and stack trace of the
        # current exception being handled.
        traceback.print_exc()

        print()
        raise e

    return res

def nproc_use_ng(det, nint=None):
    """Optimize processor usage.
    
    Attempt to estimate a reasonable number of processes to use for multiple 
    simultaneous slope_to_ramp() calculations. We attempt to estimate how many 
    calculations can happen in parallel without swapping to disk.

    NOTE: Requires psutil package. Otherwise defaults to mp.cpu_count() / 2

    Parameters
    -----------
    det : :class:`DetectorOps`
        Input detector class
    """
    import multiprocessing as mp
    try:
        import psutil
    except ImportError:
        nproc = int(mp.cpu_count() // 2)
        if nproc < 1: nproc = 1

        _log.info("No psutil package available, cannot estimate optimal nprocesses.")
        _log.info("Returning nproc=ncpu/2={}.".format(nproc))
        return nproc

    ma      = det.multiaccum
    nd1     = ma.nd1
    nd2     = ma.nd2
    nf      = ma.nf
    ngroup  = ma.ngroup
    nint    = ma.nint if nint is None else nint
    naxis3  = nd1 + ngroup*nf + (ngroup-1)*nd2

    # Compute the number of time steps per integration, per output
    nstep_frame = (det.chsize+12) * (det.ypix+1)
    nstep = nstep_frame * naxis3
    # Pad nsteps to a power of 2, which is much faster
    nstep2 = int(2**np.ceil(np.log2(nstep)))

    # Memory formulas are based on fits to memory usage
    # In GBytes
    cf = np.array([1.48561822e-15, 7.02203657e-08, 2.52022191e-01])
    mem_total = np.polynomial.polynomial.polyval(nstep2, cf[::-1])

    # Available memory
    mem = psutil.virtual_memory()
    avail_GB = mem.available / (1024**3) - 1.0 # Leave 1 GB

    # How many processors to split into?
    nproc = avail_GB // mem_total
    nproc = np.min([nproc, mp.cpu_count(), poppy.conf.n_processes])
    if nint is not None:
        nproc = np.min([nproc, nint])
        # Resource optimization:
        # Split iterations evenly over processors to free up minimally used processors.
        # For example, if there are 5 processors only doing 1 iteration and a single
        #   processor doing 2 iterations, those 5 processors (and their memory) will not
        #   get freed until the final processor is finished. So, to minimize the number
        #   of idle resources, take the total iterations and divide by two (round up),
        #   and that should be the final number of processors to use.
        np_max = np.ceil(nint / nproc)
        nproc = int(np.ceil(nint / np_max))

    if nproc < 1: nproc = 1

    return int(nproc)


def saturation_limits(inst, psf_coeff=None, psf_coeff_hdr=None, sp=None, bp_lim=None, 
              int_time=21.47354, full_well=None, well_frac=0.8, units='vegamag', 
              verbose=False, **kwargs):
    """Saturation limits

    Estimate the saturation limit of a point source for some bandpass.
    By default, it outputs the max K-Band magnitude assuming a G2V star,
    following the convention on the UA NIRCam webpage. This can be useful if
    one doesn't know how bright a source is in the selected NIRCam filter
    bandpass. Returns the saturation limit in Vega magnitudes by default,
    however, any flux unit supported by Pysynphot is possible via the 'units'
    keyword.

    Parameters
    ==========

    inst : NIRCam class
        pynrc or webbpsf_ext or webbpsf
    psf_coeff : ndarray
        A cube of polynomial coefficients for generating PSFs. This is generally 
        oversampled with a shape (fov_pix*oversamp, fov_pix*oversamp, deg).
        If not set, defaults to `inst.psf_coeff`.
    psf_coeff_hdr : FITS header
        Header information saved while generating coefficients.
    sp : Pysynphot spectrum
        Spectrum to calculate saturation (default: G2V star).
    bp_lim : Pysynphot bandpass
        The bandpass at which we report the magnitude that will saturate the NIRCam 
        band assuming some spectrum sp (default: 2MASS K-Band).
    int_time : float
        Integration time in seconds (default corresponds to 2 full frames).
    full_well : float
        Detector full well level in electrons. If not set, defaults to `inst.well_level`.
    well_frac : float
        Fraction of full well to consider "saturated." 0.8 by default.
    units : str
        Output units for saturation limit.
    """

    from webbpsf_ext.psfs import gen_image_from_coeff

    # Instrument bandpass
    bp = inst.bandpass

    # bandpass at which we report the magnitude that will saturate the NIRCam band assuming some spectrum sp
    if bp_lim is None:
        bp_lim = bp_2mass('k')
        bp_lim.name = 'K-Band'

    # Spectrum and bandpass to report magnitude that saturates NIRCam band
    if sp is None: 
        sp = stellar_spectrum('G2V')

    # Just for good measure, make sure we're all in the same wave units
    bp_lim.convert(bp.waveunits)
    sp.convert(bp.waveunits)

    # Renormalize to 10th magnitude star (Vega mags)
    mag_norm = 10.0
    sp_norm = sp.renorm(mag_norm, 'vegamag', bp_lim)
    sp_norm.name = sp.name

    # Set up an observation of the spectrum using the specified bandpass
    # Use the bandpass wavelengths to bin the fluxes
    obs = Observation(sp_norm, bp, binset=bp.waveset)
    # Convert observation to counts (e/sec)
    obs.convert('counts')

    # Zodiacal Light contributions to background
    sp_zodi = zodi_spec(**kwargs)
    pix_scale = inst.pixelscale
    obs_zodi = Observation(sp_zodi, bp, binset=bp.waveset)
    fzodi_pix = obs_zodi.countrate() * (pix_scale/206265.0)**2  # e-/sec/pixel
    # Collecting area gets reduced for coronagraphic (Lyot pupil) observations
    # This isn't accounted for later, because zodiacal light doesn't use PSF information
    if inst.is_lyot: 
        fzodi_pix *= 0.19

    # Total stellar flux and associated magnitude
    star_flux = obs.countrate() # e/sec
    mag_nrc = obs.effstim('vegamag')
    _log.debug('Total Source Count Rate for {0} = {1:0.1f} mags: {2:.0f} e-/sec'.\
        format(bp_lim.name, mag_norm, star_flux))
    _log.debug('Magnitude in {0} band: {1:.2f}'.format(bp.name, mag_nrc))

    # The number of pixels to span spatially
    if psf_coeff is None:
        psf_coeff = inst.psf_coeff
    if psf_coeff_hdr is None:
        psf_coeff_hdr = inst.psf_coeff_header
    fov_pix = psf_coeff_hdr['FOVPIX']

    # Generate the PSF image for analysis
    # Use gen_image_from_coeff() rather than inst.calc_psf_from_coeff() in case we
    # are supplying custom psf_coeff
    t0 = time.time()
    result = gen_image_from_coeff(inst, psf_coeff, psf_coeff_hdr, 
                                  sp_norm=sp_norm, return_oversample=False)
    t1 = time.time()
    _log.debug('Took %.2f seconds to generate images' % (t1-t0))

    # Saturation level (some fraction of full well) in electrons
    full_well = inst.well_level if full_well is None else full_well
    sat_level = well_frac * full_well

    # If grism spectroscopy
    pupil_mask = inst.pupil_mask
    if inst.is_grism:
        wspec, spec = result

        # Spectra are in 'sci' coords
        # If GRISMC (along columns) rotate image by 90 deg CW 
        if (pupil_mask=='GRISMC') or (pupil_mask=='GRISM90'):
            spec = np.rot90(spec, k=1)
        elif inst.module=='B':
            # Flip left to right so dispersion is in same direction as mod A
            spec = spec[:,::-1]
            wspec = wspec[::-1]

         # Time to saturation for 10-mag source
        sat_time = sat_level / spec
        _log.debug('Approximate Time to {1:.2f} of Saturation: {0:.1f} sec'.\
            format(sat_time.min(),well_frac))

        # Magnitude necessary to saturate a given pixel
        ratio = int_time / sat_time
        ratio[ratio < __epsilon] = __epsilon
        sat_mag = mag_norm + 2.5*np.log10(ratio)

        # Wavelengths to grab saturation values
        igood2 = bp.throughput > (bp.throughput.max()/4)
        wgood2 = bp.wave[igood2] / 1e4
        wsat_arr = np.unique((wgood2*10 + 0.5).astype('int')) / 10
        wdel = wsat_arr[1] - wsat_arr[0]
        msat_arr = []
        for w in wsat_arr:
            l1 = w - wdel / 4
            l2 = w + wdel / 4
            ind = ((wspec > l1) & (wspec <= l2))
            msat = sat_mag[fov_pix//2-1:fov_pix//2+2, ind].max()
            sp_temp = sp.renorm(msat, 'vegamag', bp_lim)
            obs_temp = Observation(sp_temp, bp_lim, binset=bp_lim.waveset)
            msat_arr.append(obs_temp.effstim(units))

        msat_arr = np.array(msat_arr)

        # Print verbose information
        if verbose:
            if bp_lim.name == bp.name:
                print('{0} Saturation Limit assuming {1} source:'.\
                    format(bp_lim.name,sp.name))
            else:
                print('{0} Saturation Limit for {1} assuming {2} source:'.\
                    format(bp_lim.name,bp.name,sp.name))

            names = ('Wave','Sat Limit ({})'.format(units))
            tbl = Table([wsat_arr,msat_arr], names=names)
            for k in tbl.keys():
                tbl[k].format = '9.2f'
            print(tbl)


        # Return saturation list along with corresponding wavelengths to dictionary
        return {'wave':wsat_arr.tolist(), 'satmag':msat_arr.tolist(),
            'units':units, 'Spectrum':sp_norm.name, 'bp_lim':bp_lim.name}

    # DHS spectroscopy
    elif (pupil_mask is not None) and ('DHS' in pupil_mask):
        raise NotImplementedError('DHS not implemented')

    # Imaging
    else:
        psf = result

         # Time to saturation for 10-mag source
         # Only need the maximum pixel value
        sat_time = sat_level / psf.max()
        _log.debug(f'Point source approximate Time to {well_frac:.2f} of Saturation: {sat_time:.2f} sec')

        # Magnitude necessary to saturate a given pixel
        ratio = int_time/sat_time
        sat_mag = mag_norm + 2.5*np.log10(ratio)

        # Convert to desired unit
        sp_temp = sp.renorm(sat_mag, 'vegamag', bp_lim)
        obs_temp = Observation(sp_temp, bp_lim, binset=bp_lim.waveset)
        res1 = obs_temp.effstim(units)
        
        out1 = {'satlim':res1, 'units':units, 'bp_lim':bp_lim.name, 'Spectrum':sp_norm.name}

        # For surface brightness saturation (extended object)
        # Assume the fiducial (sp_norm) to be in terms of mag/arcsec^2
        # Multiply countrate() by pix_scale^2 to get in terms of per pixel (area)
        # This is the count rate per pixel for the fiducial starting point
        image_ext = obs.countrate() * pix_scale**2 # e-/sec/pixel
        
        sat_time = sat_level / image_ext
        _log.debug(f'Extended object approximate Time to {well_frac:.2f} of Saturation: {sat_time:.2f} sec')
        
        # Magnitude necessary to saturate a given pixel
        ratio = int_time / sat_time
        sat_mag_ext = mag_norm + 2.5*np.log10(ratio)

        # Convert to desired units
        sp_temp = sp.renorm(sat_mag_ext, 'vegamag', bp_lim)
        obs_temp = Observation(sp_temp, bp_lim, binset=bp_lim.waveset)
        res2 = obs_temp.effstim(units)

        out2 = out1.copy()
        out2['satlim'] = res2
        out2['units']  = units + '/arcsec^2'

        # Print verbose information
        if verbose:
            if bp_lim.name == bp.name:
                print('{} Saturation Limit assuming {} source (point source): {:.2f} {}'.\
                    format(bp_lim.name, sp_norm.name, out1['satlim'], out1['units']) )
                print('{} Saturation Limit assuming {} source (extended): {:.2f} {}'.\
                    format(bp_lim.name, sp_norm.name, out2['satlim'], out2['units']) )
            else:
                print('{} Saturation Limit for {} assuming {} source (point source): {:.2f} {}'.\
                    format(bp_lim.name, bp.name, sp_norm.name, out1['satlim'], out1['units']) )
                print('{} Saturation Limit for {} assuming {} source (extended): {:.2f} {}'.\
                    format(bp_lim.name, bp.name, sp_norm.name, out2['satlim'], out2['units']) )

        return out1, out2

def _mlim_helper(sub_im, mag_norm=10, mag_arr=np.arange(5,35,1),
    nsig=5, nint=1, snr_fact=1, forwardSNR=False, **kwargs):
    """Helper function for determining grism sensitivities"""

    sub_im_sum = sub_im.sum()

    # Just return the SNR for the input sub image
    if forwardSNR:
        im_var = pix_noise(fsrc=sub_im, **kwargs)**2
        ns_sum = np.sqrt(np.sum(im_var) / nint)
        return snr_fact * sub_im_sum / ns_sum

    fact_arr = 10**((mag_arr-mag_norm)/2.5)
    snr_arr = []

    for f in fact_arr:
        im = sub_im / f
        im_var = pix_noise(fsrc=im, **kwargs)**2
        im_sum = sub_im_sum / f
        ns_sum = np.sqrt(np.sum(im_var) / nint)

        snr_arr.append(im_sum / ns_sum)
    snr_arr = snr_fact*np.asarray(snr_arr)
    return np.interp(nsig, snr_arr[::-1], mag_arr[::-1])


def sensitivities(inst, psf_coeff=None, psf_coeff_hdr=None, sp=None, units=None, 
                  forwardSNR=False, nsig=10, tf=10.737, ngroup=2, nf=1, nd2=0, nint=1,
                  return_image=False, image=None, cr_noise=True, 
                  dw_bin=None, ap_spec=None, rad_EE=None, verbose=False, **kwargs):
    """Sensitivity Estimates

    Estimates the sensitivity for a set of instrument parameters.
    By default, a flat spectrum is convolved with the specified bandpass.
    For imaging, this function also returns the surface brightness sensitivity.

    The number of photo-electrons are computed for a source at some magnitude
    as well as the noise from the detector readout and some average zodiacal
    background flux. Detector readout noise follows an analytical form that
    matches extensive long dark observations during cryo-vac testing.

    This function returns the n-sigma background limit in units of uJy (unless
    otherwise specified; valid units can be found on the synphot webpage at
    https://synphot.readthedocs.io/).

    For imaging, a single value is given assuming aperture photometry with a
    radius of ~1 FWHM rounded to the next highest integer pixel (or 2.5 pixels,
    whichever is larger). For spectral observations, this function returns an
    array of sensitivities at 0.1um intervals with apertures corresponding to
    2 spectral pixels and a number of spatial pixels equivalent to 1 FWHM rounded
    to the next highest integer (minimum of 5 spatial pixels).

    Parameters
    ==========

    Instrument Settings
    -------------------
    filter_or_bp : Either the name of the filter or pre-computed Pysynphot bandpass.
    pupil  : NIRCam pupil elements such as grisms or lyot stops
    mask   : Specify the coronagraphic occulter (spots or bar)
    module : 'A' or 'B'
    pix_scale : Pixel scale in arcsec/pixel

    Spectrum Settings
    -------------------
    sp         : A synphot spectral object to calculate sensitivity
                 (default: Flat spectrum in photlam)
    nsig       : Desired nsigma sensitivity
    units      : Output units (defaults to uJy for grisms, nJy for imaging)
    forwardSNR : Find the SNR of the input spectrum instead of determining sensitivity.

    Ramp Settings
    -------------------
    tf     : Time per frame
    ngroup : Number of groups per integration
    nf     : Number of averaged frames per group
    nd2    : Number of dropped frames per group
    nint   : Number of integrations/ramps to consider

    PSF Information
    -------------------
    coeff : A cube of polynomial coefficients for generating PSFs. This is
            generally oversampled with a shape (fov_pix*oversamp, fov_pix*oversamp, deg).
            If not set, this will be calculated using :func:`gen_psf_coeff`.
    coeff_hdr    : Header associated with coeff cube.
    fov_pix      : Number of detector pixels in the image coefficient and PSF.
    oversample   : Factor of oversampling of detector pixels.
    offset_r     : Radial offset of the target from center.
    offset_theta : Position angle for that offset, in degrees CCW (+Y).

    Misc.
    -------------------
    image        : Explicitly pass image data rather than calculating from coeff.
    return_image : Instead of calculating sensitivity, return the image calced from coeff.
                   Useful if needing to calculate sensitivities for many different settings.
    rad_EE       : Extraction aperture radius (in pixels) for imaging mode.
    dw_bin       : Delta wavelength to calculate spectral sensitivities (grisms & DHS).
    ap_spec      : Instead of dw_bin, specify the spectral extraction aperture in pixels.
                   Takes priority over dw_bin. Value will get rounded up to nearest int.
    cr_noise     : Include noise from cosmic ray hits?

    Keyword Args
    -------------------
    zodi_spec     - zfact, ra, dec, thisday, [locstr, year, day]
    pix_noise     - rn, ktc, idark, and p_excess
    gen_psf_coeff - npsf and ndeg
    read_filter   - ND_acq
    """

    # PSF coefficients
    from webbpsf_ext.psfs import gen_image_from_coeff
    from webbpsf_ext.bandpasses import bp_igood

    pupil_mask = inst.pupil_mask

    grism_obs = inst.is_grism
    dhs_obs   = (pupil_mask is not None) and ('DHS' in pupil_mask)
    lyot_obs  = inst.is_lyot
    coron_obs = inst.is_coron

    # Get filter throughput and create bandpass
    bp = inst.bandpass
    waveset = np.copy(bp.wave)

    # Pixel scale (arcsec/pixel)
    pix_scale = inst.pixelscale

    # Spectrum and bandpass to report magnitude that saturates NIRCam band
    if sp is None:
        flux = np.zeros_like(bp.throughput) + 10.
        sp = ArraySpectrum(bp.waveset, flux, fluxunits='photlam')
        sp.name = 'Flat spectrum in photlam'

    if forwardSNR:
        sp_norm = sp
    else:
        # Renormalize to 10th magnitude star
        mag_norm = 10
        sp_norm = sp.renorm(mag_norm, 'vegamag', bp)
        sp_norm.name = sp.name

    # Zodiacal Light Stuff
    sp_zodi = zodi_spec(**kwargs)
    obs_zodi = Observation(sp_zodi, bp, binset=bp.waveset)
    fzodi_pix = obs_zodi.countrate() * (pix_scale/206265.0)**2  # e-/sec/pixel
    # Collecting area gets reduced for coronagraphic observations
    # This isn't accounted for later, because zodiacal light doesn't use PSF information
    if coron_obs: 
        fzodi_pix *= 0.19

    # The number of pixels to span spatially for WebbPSF calculations
    fov_pix    = psf_coeff_hdr['FOVPIX']
    oversample = psf_coeff_hdr['OSAMP']

    # Generate the PSF image for analysis.
    # This process can take a while if being done over and over again.
    # Let's provide the option to skip this with a pre-generated image.
    # Skip image generation if `image` keyword is not None.
    # Remember, this is for a very specific NORMALIZED spectrum
    t0 = time.time()
    if image is None:
        image = gen_image_from_coeff(inst, psf_coeff, psf_coeff_hdr, 
                                     sp_norm=sp_norm, return_oversample=False)
    t1 = time.time()
    _log.debug(f'fov_pix={fov_pix}, oversample={oversample}')
    _log.debug('Took {:.2f} seconds to generate images'.format(t1-t0))
    if return_image:
        return image

    # Cosmic Ray Loss (JWST-STScI-001721)
    # SNR with cosmic ray events depends directly on ramp integration time
    if cr_noise:
        tint = (ngroup*nf + (ngroup-1)*nd2) * tf
        snr_fact = 1.0 - tint*6.7781e-5
    else:
        snr_fact = 1.0

    # If grism spectroscopy
    if grism_obs:

        if units is None: 
            units = 'uJy'
        wspec, spec = image

        # Spectra are in 'sci' coords
        # If GRISMC (along columns) rotate image by 90 deg CW 
        if (pupil_mask=='GRISMC') or (pupil_mask=='GRISM90'):
            spec = np.rot90(spec, k=1)
        elif inst.module=='B':
            # Flip left to right so dispersion is in same direction as mod A
            spec = spec[:,::-1]
            wspec = wspec[::-1]

        # Wavelengths to grab sensitivity values
        igood2 = bp_igood(bp, min_trans=bp.throughput.max()/3, fext=0)
        wgood2 = waveset[igood2] / 1e4
        wsen_arr = np.unique((wgood2*10 + 0.5).astype('int')) / 10

        # Add an addition 0.1 on either side
        dw = 0.1
        wsen_arr = np.concatenate(([wsen_arr.min()-dw],wsen_arr,[wsen_arr.max()+dw]))

        #wdel = wsen_arr[1] - wsen_arr[0]

        # FWHM at each pixel position
        #fwhm_pix_arr = np.ceil(wsen_arr * 0.206265 / 6.5 / pix_scale)
        # Make sure there's at least 5 total pixels in spatial dimension
        #temp = fwhm_pix_arr.repeat(2).reshape([fwhm_pix_arr.size,2])
        #temp[:,0] = 2
        #rad_arr = temp.max(axis=1)
        # Ignore the above, let's always do a 5pix spatial aperture
        rad_arr = np.zeros(wsen_arr.size) + 2 # (2*2+1)

        # Spatial aperture size at each wavelength
        ap_spat = (2*rad_arr+1).astype('int')
        # Indices with spectral image
        ispat1 = (fov_pix - ap_spat) // 2
        ispat2 = ispat1 + ap_spat

        # Get spectral indices on the spectral image
        if (dw_bin is None) and (ap_spec is None):
            ap_spec = 2
        elif (dw_bin is not None) and (ap_spec is None):
            ap_spec = wspec.size * dw_bin / (wspec.max() - wspec.min())
            ap_spec = int(ap_spec+0.5)
        else:
            ap_spec = int(ap_spec+0.5)
        diff = abs(wspec.reshape(wspec.size,1) - wsen_arr)
        ind_wave = []
        for i in np.arange(wsen_arr.size):
            ind = (np.where(diff[:,i]==min(diff[:,i])))[0]
            ind_wave.append(ind[0])
        ispec1 = np.asarray(ind_wave) - ap_spec // 2
        ispec2 = ispec1 + ap_spec

        # At each wavelength, grab a sub image and find the limiting magnitude
        bglim_arr = []
        for i in np.arange(wsen_arr.size):
            sub_im = spec[ispat1[i]:ispat2[i],ispec1[i]:ispec2[i]]

            if forwardSNR:
                snr = _mlim_helper(sub_im, nint=nint, forwardSNR=forwardSNR,
                    ngroup=ngroup, nf=nf, nd2=nd2, tf=tf, fzodi=fzodi_pix,
                    snr_fact=snr_fact, **kwargs)
                bglim_arr.append(snr)

            else:
                # Interpolate over a coarse magnitude grid
                mag_arr=np.arange(5,35,1)
                mag_lim = _mlim_helper(sub_im, mag_norm, mag_arr, nsig=nsig, nint=nint,
                    ngroup=ngroup, nf=nf, nd2=nd2, tf=tf, fzodi=fzodi_pix,
                    snr_fact=snr_fact, **kwargs)

                # Zoom in and interoplate over finer grid
                mag_arr = np.arange(mag_lim-1,mag_lim+1,0.05)
                mag_lim = _mlim_helper(sub_im, mag_norm, mag_arr, nsig=nsig, nint=nint,
                    ngroup=ngroup, nf=nf, nd2=nd2, tf=tf, fzodi=fzodi_pix,
                    snr_fact=snr_fact, **kwargs)

                # Renormalize spectrum to magnitude limit and convert to desired units
                sp_norm2 = sp.renorm(mag_lim, 'vegamag', bp)
                sp_norm2.convert(units)
                bglim = np.interp(wsen_arr[i],sp_norm2.wave/1e4, sp_norm2.flux)
                bglim_arr.append(bglim)

        bglim_arr = np.asarray(bglim_arr)

        # Return sensitivity list along with corresponding wavelengths to dictionary
        if forwardSNR:
            sp_norm.convert(units)
            fvals = np.interp(wsen_arr, sp_norm.wave/1e4, sp_norm.flux)
            out = {'wave':wsen_arr.tolist(), 'snr':bglim_arr.tolist(),
                   'flux_units':units, 'flux':fvals.tolist(), 'Spectrum':sp.name}

            if verbose:
                print('{0} SNR for {1} source'.format(bp.name,sp.name))
                names = ('Wave','SNR','Flux ({})'.format(units))
                tbl = Table([wsen_arr,bglim_arr, fvals], names=names)
                for k in tbl.keys():
                    tbl[k].format = '9.2f'
                print(tbl)

        else:
            out = {'wave':wsen_arr.tolist(), 'sensitivity':bglim_arr.tolist(),
                   'units':units, 'nsig':nsig, 'Spectrum':sp.name}

            if verbose:
                print('{} Background Sensitivity ({}-sigma) for {} source'.\
                    format(bp.name,nsig,sp.name))

                names = ('Wave','Limit ({})'.format(units))
                tbl = Table([wsen_arr,bglim_arr], names=names)
                for k in tbl.keys():
                    tbl[k].format = '9.2f'
                print(tbl)

        return out

    # DHS spectroscopy
    elif dhs_obs:
        raise NotImplementedError('DHS has yet to be fully included')

    # Imaging (includes coronagraphy)
    else:
        if units is None: 
            units = 'nJy'

        # Wavelength to grab sensitivity values
        obs = Observation(sp_norm, bp, binset=bp.waveset)
        efflam = obs.efflam()*1e-4 # microns

        # Encircled energy
        rho_pix = dist_image(image)
        bins = np.arange(rho_pix.min(), rho_pix.max() + 1, 1)
        # Groups indices for each radial bin
        igroups, _, rad_pix = hist_indices(rho_pix, bins, True)
        # Sum of each radial annulus
        sums = binned_statistic(igroups, image, func=np.sum)
        # Encircled energy within each radius
        EE_flux = np.cumsum(sums)

        # How many pixels do we want?
        fwhm_pix = 1.2 * efflam * 0.206265 / 6.5 / pix_scale
        if rad_EE is None:
            rad_EE = np.max([fwhm_pix,2.5])
        npix_EE = np.pi * rad_EE**2

        # For surface brightness sensitivity (extended object)
        # Assume the fiducial (sp_norm) to be in terms of mag/arcsec^2
        # Multiply countrate() by pix_scale^2 to get in terms of per pixel (area)
        # This is the count rate per pixel for the fiducial starting point
        image_ext = obs.countrate() * pix_scale**2 # e-/sec/pixel
        #print(image_ext)

        if forwardSNR:
            im_var = pix_noise(ngroup=ngroup, nf=nf, nd2=nd2, tf=tf,
                fzodi=fzodi_pix, fsrc=image, **kwargs)**2

            # root squared sum of noise within each radius
            sums = binned_statistic(igroups, im_var, func=np.sum)
            EE_var = np.cumsum(sums)
            EE_sig = np.sqrt(EE_var / nint)

            EE_snr = snr_fact * EE_flux / EE_sig
            snr_rad = np.interp(rad_EE, rad_pix, EE_snr)
            flux_val = obs.effstim(units)
            out1 = {'type':'Point Source', 'snr':snr_rad, 'Spectrum':sp.name,
                'flux':flux_val, 'flux_units':units}

            # Extended object surfrace brightness
            im_var = pix_noise(ngroup=ngroup, nf=nf, nd2=nd2, tf=tf,
                fzodi=fzodi_pix, fsrc=image_ext, **kwargs)**2
            im_sig = np.sqrt(im_var*npix_EE / nint)
            # Total number of pixels within r=fwhm or 2.5 pixels
            fsum2 = image_ext * npix_EE
            snr2 = snr_fact * fsum2 / im_sig # SNR per "resolution element"ish
            out2 = {'type':'Surface Brightness', 'snr':snr2, 'Spectrum':sp.name,
                'flux':flux_val, 'flux_units':units+'/arcsec^2'}

            if verbose:
                for out in [out1,out2]:
                    print('{} SNR ({:.2f} {}): {:.2f} sigma'.\
                        format(out['type'], out['flux'], out['flux_units'], out['snr']))

        else:
            # Interpolate over a coarse magnitude grid to get SNR
            # Then again over a finer grid
            for ii in np.arange(2):
                if ii==0: mag_arr = np.arange(5,35,1)
                else: mag_arr = np.arange(mag_lim-1,mag_lim+1,0.05)

                fact_arr = 10**((mag_arr-mag_norm)/2.5)
                snr_arr = []
                for f in fact_arr:
                    #im_var = image/f/tint + var_const

                    im_var = pix_noise(ngroup=ngroup, nf=nf, nd2=nd2, tf=tf,
                        fzodi=fzodi_pix, fsrc=image/f, **kwargs)**2

                    # root squared sum of noise within each radius
                    sums = binned_statistic(igroups, im_var, func=np.sum)
                    EE_var = np.cumsum(sums)
                    EE_sig = np.sqrt(EE_var / nint)

                    EE_snr = snr_fact * (EE_flux/f) / EE_sig
                    snr_rad = np.interp(rad_EE, rad_pix, EE_snr)
                    snr_arr.append(snr_rad)

                snr_arr = np.asarray(snr_arr)
                mag_lim = np.interp(nsig, snr_arr[::-1], mag_arr[::-1])

                _log.debug('Mag Limits [{0:.2f},{1:.2f}]; {2:.0f}-sig: {3:.2f}'.\
                    format(mag_arr.min(),mag_arr.max(),nsig,mag_lim))

            # Renormalize spectrum at given magnitude limit
            sp_norm2 = sp.renorm(mag_lim, 'vegamag', bp)
            # Determine effective stimulus
            obs2 = Observation(sp_norm2, bp, binset=bp.waveset)
            bglim = obs2.effstim(units)

            out1 = {'sensitivity':bglim, 'units':units, 'nsig':nsig, 'Spectrum':sp.name}

            # Same thing as above, but for surface brightness
            for ii in np.arange(2):
                if ii==0: mag_arr = np.arange(5,35,1)
                else: mag_arr = np.arange(mag_lim-1,mag_lim+1,0.05)

                fact_arr = 10**((mag_arr-mag_norm)/2.5)
                snr_arr = []
                for f in fact_arr:
                    im_var = pix_noise(ngroup=ngroup, nf=nf, nd2=nd2, tf=tf,
                        fzodi=fzodi_pix, fsrc=image_ext/f, **kwargs)**2

                    im_sig = np.sqrt(im_var*npix_EE / nint)
                    fsum2 = image_ext * npix_EE / f
                    snr2 = snr_fact * fsum2 / im_sig
                    #print('{:.5f} {:.5f} {:.2f}'.format(fsum2,im_sig,snr2))

                    snr_arr.append(snr2)

                snr_arr = np.asarray(snr_arr)
                mag_lim = np.interp(nsig, snr_arr[::-1], mag_arr[::-1])

                _log.debug('Mag Limits (mag/asec^2) [{0:.2f},{1:.2f}]; {2:.0f}-sig: {3:.2f}'.\
                    format(mag_arr.min(),mag_arr.max(),nsig,mag_lim))

            # mag_lim is in terms of mag/arcsec^2 (same as mag_norm)
            sp_norm2 = sp.renorm(mag_lim, 'vegamag', bp)
            obs2 = Observation(sp_norm2, bp, binset=bp.waveset)
            bglim2 = obs2.effstim(units) # units/arcsec**2

            out2 = out1.copy()
            out2['sensitivity'] = bglim2
            out2['units'] = units+'/arcsec^2'

            if verbose:
                print('{} Sensitivity ({}-sigma): {:.2f} {}'.\
                       format('Point Source', nsig, bglim, out1['units']))
                print('{} Sensitivity ({}-sigma): {:.2f} {}'.\
                       format('Surface Brightness', nsig, bglim2, out2['units']))

        return out1, out2

