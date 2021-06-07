from __future__ import division, print_function, unicode_literals

from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel
from astropy.convolution import convolve_fft
from scipy import fftpack
from copy import deepcopy

from webbpsf_ext.image_manip import convolve_image, make_disk_image

# Import libraries
from .pynrc_core import NIRCam, DetectorOps, merge_dicts
from .nrc_utils import *
from .maths.coords import gen_sgd_offsets

import logging
_log = logging.getLogger('pynrc')

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

    """

    def __init__(self, wind_mode='WINDOW', xpix=320, ypix=320, large_grid=False, **kwargs):

        super().__init__(wind_mode=wind_mode, xpix=xpix, ypix=ypix, fov_bg_match=True, **kwargs)

        log_prev = conf.logging_level
        setup_logging('WARN', verbose=False)

        # Enable WFE drift
        self.gen_wfedrift_coeff()
        # Enable mask-dependent
        self.gen_wfemask_coeff(large_grid=large_grid)
        # Set locations based on detector
        self._set_xypos()
        # Create mask throughput images seen by detector
        self._gen_cmask()
        # Create default SGD offsets
        self.gen_pointing_offsets(sgd_type=None, slew_std=5, fsm_std=2.5)

        setup_logging(log_prev, verbose=False)



    # def _gen_cached_psfs(self):
    #     """Generate a set of cached PSF for disk convolutions."""
    #     if self.image_mask is None:
    #         # If no mask center and bg and off-axis PSFs are all the same
    #         _, psf = self.gen_psf(return_oversample=True, use_bg_psf=False)
    #         self.psf_center_over = psf
    #         self.psf_offaxis_over = self.psf_center_over
    #     elif self.image_mask[-1]=='R':
    #         _, psf = self.gen_psf(return_oversample=True, use_bg_psf=False)
    #         self.psf_center_over = psf
    #         _, psf = self.gen_psf(return_oversample=True, use_bg_psf=True)
    #         self.psf_offaxis_over = psf
    #     elif self.image_mask[-1]=='B':
    #         # Bar mask "central" PSFs are a list of PSFs along the mask
    #         self._gen_psfbar_list()
    #         _, psf = self.gen_psf(return_oversample=True, use_bg_psf=True)
    #         self.psf_offaxis_over = psf


    def gen_offset_psf(self, offset_r, offset_theta, sp=None, return_oversample=False, wfe_drift=None):
        """Create a PSF offset from center FoV

        Generate some off-axis PSF at a given (r,theta) offset from center.
        This function is mainly for coronagraphic observations where the
        off-axis PSF varies w.r.t. position. The PSF is centered in the
        resulting image. The offset values are assumed to be in 'idl' coordinate
        frame.

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

        """

        coords = rtheta_to_xy(offset_r, offset_theta)

        psf = self.calc_psf_from_coeff(sp=sp, return_oversample=return_oversample, 
            wfe_drift=wfe_drift, coord_vals=coords, coord_frame='idl', return_hdul=False)
        
        return psf

        # if sp is None:
        #     # No spectral information, so use cached PSFs
        #     # Let _psf_lin_comb() handle things
        #     psf_center  = None 
        #     psf_offaxis = None 
        # else:
        #     if self.mask is None:
        #         # Direct imaging; both PSFs are the same
        #         _, psf_center = self.gen_psf(sp, return_oversample=True, use_bg_psf=False,
        #                                      wfe_drift=wfe_drift)
        #         psf_offaxis = None
        #     elif self.mask[-1]=='R': # Round masks
        #         _, psf_center  = self.gen_psf(sp, return_oversample=True, use_bg_psf=False)
        #         _, psf_offaxis = self.gen_psf(sp, return_oversample=True, use_bg_psf=True,
        #                                       wfe_drift=wfe_drift)
        #     elif self.mask[-1]=='B': # Bar masks
        #         offx_asec, offy_asec = rtheta_to_xy(offset_r, offset_theta)
        #         _, psf_center = self.gen_psf(sp, return_oversample=True, use_bg_psf=False, 
        #                                      bar_offset=offx_asec)
        #         _, psf_offaxis = self.gen_psf(sp, return_oversample=True, use_bg_psf=True,
        #                                       wfe_drift=wfe_drift)

        # psf = self._psf_lin_comb(offset_r, offset_theta, psf_center, psf_offaxis)
        # if return_oversample:
        #     return psf
        # else:
        #     fov_pix = self.fov_pix
        #     return krebin(psf, (fov_pix,fov_pix))

    def _set_xypos(self, xy=None):
        """
        Set x0 and y0 subarray positions.
        """
        wind_mode = self.det_info['wind_mode']
        if xy is not None:
            self.update_detectors(x0=xy[0], y0=xy[1])
        elif self.is_coron:
            full = True if 'FULL' in wind_mode else False
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
        """
        Generate coronagraphic mask transmission images.

        Output images are in V2/V3 coordinates.
        """
        mask = self.image_mask
        pupil = self.pupil_mask
        oversample = self.oversample

        mask_dict = {}
        if mask is None:
            mask_dict['DETSAMP'] = None
            mask_dict['OVERSAMP'] = None
        else:
            detid = self.Detector.detid
            x0, y0 = (self.det_info['x0'], self.det_info['y0'])
            xpix, ypix = (self.det_info['xpix'], self.det_info['ypix'])

            im_det  = build_mask_detid(detid, oversample=1, pupil=pupil)
            # im_over = build_mask_detid(detid, oversample=oversample, pupil=pupil)

            # Convert to det coords and crop
            im_det  = sci_to_det(im_det, detid)
            # im_over = sci_to_det(im_over, detid)
            im_over = frebin(im_det, scale=oversample)

            im_det = im_det[y0:y0+ypix, x0:x0+xpix]
            # Crop oversampled image
            ix1, iy1 = np.array([x0, y0]) * oversample
            ix2 = ix1 + xpix * oversample
            iy2 = iy1 + ypix * oversample
            im_over = im_over[iy1:iy2, ix1:ix2]

            # Revert to sci coords
            mask_dict['DETSAMP'] = det_to_sci(im_det, detid)
            mask_dict['OVERSAMP'] = det_to_sci(im_over, detid)

        self.mask_images = mask_dict

    def gen_pointing_offsets(self, sgd_type='auto', slew_std=5, fsm_std=2.5):
        """
        Create a series of x and y position offsets for a SGD pattern.
        This includes the central position as the first in the series.
        By default, will also add random movement errors using the
        `slew_std` and `fsm_std` keywords. Returned values are in arcsec.
        
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

        if sgd_type == 'auto':
            if self.is_coron and self.image_mask[-1]=='R':
                sgd_type = '5diamond'
            elif self.is_coron and self.image_mask[-1]=='B':
                sgd_type = '5bar'
            else:
                sgd_type = '5diamond'

        if sgd_type is None:
            xoff, yoff = np.random.normal(scale=slew_std, size=2) / 1000
            fsm_std = None
        else:
            xoff, yoff = gen_sgd_offsets(sgd_type, slew_std=slew_std, fsm_std=fsm_std)

        _log.info("Saving SGD position info to `self.pointing_info` dictionary attribute")
        self.pointing_info = {
            'sgd_type' : sgd_type,
            'slew_std' : slew_std,
            'fsm_std'  : fsm_std,
            'xoff' : xoff, 'yoff' : yoff,
        }



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
        WFE drift in nm between the science and reference targets.
        Expected values are between ~3-10 nm.
    wfe_roll_drift: float
        WFE drift in nm between science roll angles. Default=0.
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
    """

    def __init__(self, sp_sci, distance, sp_ref=None, wfe_ref_drift=5, wfe_roll_drift=2,
        wind_mode='WINDOW', xpix=320, ypix=320, disk_params=None, 
        sgd_type=None, slew_std=5, fsm_std=2.5, **kwargs):

        if 'FULL'   in wind_mode: xpix = ypix = 2048
        if 'STRIPE' in wind_mode: xpix = 2048

        #super(NIRCam,self).__init__(**kwargs)
        # Not sure if this works for both Python 2 and 3
        # nrc_hci.__init__(self, wind_mode=wind_mode, xpix=xpix, ypix=ypix, wfe_drift=False,
        #                  verbose=verbose, **kwargs)
        super().__init__(wind_mode=wind_mode, xpix=xpix, ypix=ypix, wfe_drift=False, **kwargs)

        if (wind_mode=='FULL') and (self.channel=='SW'):
            raise NotImplementedError('SW Full Frame not yet implemented.')

        self._bar_offset = None

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

    @property
    def bar_offset(self):
        """Offset position along bar mask (arcsec)."""
        if self._bar_offset is None:
            bar_offset, _ = offset_bar(self._filter, self.image_mask)
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
            xpix = self.det_info['xpix']
            ypix = self.det_info['ypix']
            oversample = self.psf_info['oversample']

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

            disk_hdul = make_disk_image(self, self.disk_params, sp_star=self.sp_sci, 
                                        dist_out=self.distance, shape_out=shape_out)

            # Get rid of the central star flux
            # and anything interior to a few pixels
            image = disk_hdul[0].data
            image_rho = dist_image(image)
            ind_max = image == image.max()
            inner_pix = 3 * oversample
            if (image[image_rho<inner_pix].max() == image.max()) and (image.max()>1000*image[~ind_max].max()):
                image[image_rho < inner_pix] = 0

            # Expand to full observation size
            if shape_out is None:
                ydata, xdata = disk_hdul[0].data.shape
                ynew = np.max([ypix, ydata])
                xnew = np.max([xpix, xdata])
                disk_hdul[0].data = pad_or_cut_to_size(disk_hdul[0].data, (ynew,xnew))
            self.disk_hdulist = disk_hdul

    # Any time update_detectors is called, also call _gen_ref
    def update_detectors(self, verbose=False, **kwargs):
        super().update_detectors(verbose=verbose, **kwargs)

        # Update ref detector window size
        self.gen_ref()

    def gen_ref(self, **kwargs):
        """
        Function to generate Reference Detector class.
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

        # Update detector class
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


    def _gen_psf_list(self):
        """
        Save instances of NIRCam PSFs that are incrementally offset
        from coronagraph center to convolve with a disk image.
        """

        # If no mask, then the PSF looks the same at all radii
        self.psf_list = []
        if self.mask is None:
            psf = self.gen_offset_psf(0, 0)
            self.psf_list = [psf]
        elif self.mask[-1]=='R': # Round masks
            self.psf_list = [self.gen_offset_psf(offset, 0) for offset in self.offset_list]
        elif self.mask[-1]=='B': # Bar masks
            # Set bar offset to 0, for this part, then return to original value
            # baroff_orig = self.bar_offset
            # self.bar_offset = 0
            self.psf_list = [self.gen_offset_psf(offset, 0) for offset in self.offset_list]
            # self.bar_offset = baroff_orig

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
        accr=False, mmdot=None, mdot=None, accr_rin=2, truncated=False):
        """Insert a planet into observation.

        Add exoplanet information that will be used to generate a point
        source image using a spectrum from Spiegel & Burrows (2012).
        Use self.delete_planets() to delete them.

        Coordinate convention is for +N up and +E to left.

        Parameters
        ----------
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
        au_per_pixel = self.distance*self.pixelscale
        if 'AU' in runits:
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


    def gen_planets_image(self, PA_offset=0, xyoff_asec=(0,0), use_cmask=False, 
        quick_PSF=True, wfe_drift=None, return_oversample=False, **kwargs):
        """Create image of just planets.

        Use info stored in self.planets to create a noiseless slope image
        of just the exoplanets (no star). 

        Coordinate convention is for +N up and +E to left.

        Parameters
        ----------
        PA_offset : float
            Rotate entire scene by some position angle.
            Positive values are counter-clockwise from +Y direction.
            Corresponds to instrument aperture PA.
        xyoff_asec : tuple
            Offsets (dx,dy) specified in arcsec. These are meant to be
            for minor shifts, use as SGD. Bar offsets are accounted
            for automatically. 
        quick_PSF : bool
            Rather than generate a spectrum-weighted PSF, use the
            cached PSF scaled by the photon count through the bandpass.
            Resulting PSFs are less accurate, but generation is much faster.
            Also ignores the WFE drift parameter. Default is True.
        use_cmask : bool
            Use the coronagraphic mask image to determine if any planet is
            getting obscurred by a corongraphic mask feature
        """
        if len(self.planets)==0:
            _log.info("No planet info at self.planets")
            return 0.0

        PA_offset=0 if PA_offset is None else PA_offset

        # Additional field offsets
        offx_asec, offy_asec = xyoff_asec

        ypix, xpix = (self.det_info['ypix'], self.det_info['xpix'])
        xpix_over = xpix * self.oversample
        ypix_over = ypix * self.oversample

        image_over = np.zeros([ypix_over, xpix_over])
        bar_offset = self.bar_offset
        for pl in self.planets:

            # Create slope image (postage stamp) of planet
            sp = self.planet_spec(**pl)

            # Location relative to star
            delx_asec, dely_asec = pl['xyoff_asec']

            # Add in PA offset
            if PA_offset!=0:
                delx_asec, dely_asec = xy_rot(delx_asec, dely_asec, PA_offset)

            # Add in bar offset for PSF generation
            delx_asec = delx_asec + offx_asec + bar_offset
            dely_asec = dely_asec + offy_asec
            r, th = xy_to_rtheta(delx_asec, dely_asec)
            if quick_PSF:
                psf = self.gen_offset_psf(r, th, return_oversample=True)
                obs = S.Observation(sp, self.bandpass, binset=self.bandpass.wave)
                psf_planet = psf*obs.effstim('counts')
            else:
                psf_planet = self.gen_offset_psf(r, th, sp=sp, return_oversample=True, wfe_drift=wfe_drift)

            # Expand to full size
            psf_planet = pad_or_cut_to_size(psf_planet, (ypix_over, xpix_over))

            # Shift to position relative to center of image
            delx, dely = np.array([delx_asec, dely_asec]) / self.pixelscale
            if ('FULL' in self.det_info['wind_mode']) and (self.image_mask is not None):
                cdict = coron_ap_locs(self.module, self.channel, self.image_mask, full=True)
                xcen, ycen = cdict['cen_sci']
                delx += (xcen - xpix/2)
                dely += (ycen - ypix/2)
            delx_over = delx * self.oversample
            dely_over = dely * self.oversample
            psf_planet = fshift(psf_planet, delx=delx_over, dely=dely_over, pad=True)

            # Determine if any throughput loss due to coronagraphic mask
            # artifacts, such as the mask holder or ND squares.
            # Planet positions are relative to the center of the mask,
            # which is not centered in a full detector.
            # All subarrays should have the mask placed at the center.
            if self.is_coron and use_cmask:
                cmask = self.mask_images['DETSAMP']
                # First, anything in a rectangular region around the
                # mask has already been correctly accounted for
                if (np.abs(delx_asec)<10) and (np.abs(dely_asec)<5):
                    trans = 1
                elif 'FULL' in self.det_info['wind_mode']:
                    # If a full detector observation, then adjust
                    # to be relative to mask location
                    xoff = delx_asec / self.pixelscale
                    yoff = dely_asec / self.pixelscale
                    xpos = int(xcen + xoff)
                    ypos = int(ycen + yoff)
                    cmask_sub = cmask[ypos-3:ypos+3,xpos-3:xpos+3]
                    trans = np.mean(cmask_sub)
                else:
                    xpos, ypos = (int(delx), int(dely))
                    cmask_sub = cmask[ypos-3:ypos+3,xpos-3:xpos+3]
                    trans = np.mean(cmask_sub)
            else:
                trans = 1

            psf_planet *= trans

            # Add to image
            image_over += psf_planet

        if return_oversample:
            return image_over
        else:
            return frebin(image_over, scale=1/self.oversample)


    def gen_disk_image(self, PA_offset=0, xyoff_asec=(0,0), use_cmask=False,
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
            Corresponds to instrument aperture PA.
        xyoff_asec : tuple
            Offsets (dx,dy) specified in arcsec. These are meant to be
            for minor shifts, use as SGD. Bar offsets are accounted
            for automatically. 
        use_cmask : bool
            Use the coronagraphic mask image to attenuate disk regions
            getting obscurred by a corongraphic mask feature
        """

        if self.disk_hdulist is None:
            return 0.0

        # Final image shape
        ypix, xpix = (self.det_info['ypix'], self.det_info['xpix'])
        bar_offset = self.bar_offset

        pixscale_over = self.pixelscale / self.oversample

        # Determine final shift amounts to location along bar
        # Shift to position relative to center of image
        if ('FULL' in self.det_info['wind_mode']) and (self.image_mask is not None):
            cdict = coron_ap_locs(self.module, self.channel, self.image_mask, full=True)
            xcen, ycen = cdict['cen_sci']
        else:
            xcen, ycen = (xpix/2, ypix/2)
        delx_pix = (xcen - xpix/2)
        dely_pix = (ycen - ypix/2)

        offx_asec, offy_asec = xyoff_asec
        delx_asec = offx_asec + delx_pix * self.pixelscale + bar_offset
        dely_asec = offy_asec + dely_pix * self.pixelscale

        # Rotate and shift oversampled disk image
        hdul_rot_shift = rotate_shift_image(self.disk_hdulist, PA_offset=PA_offset, 
                                            delx_asec=delx_asec, dely_asec=dely_asec)
        disk_image = hdul_rot_shift[0].data

        # Multiply raw disk data by coronagraphic mask.
        # Exclude region already affected by observed mask.
        cmask = self.mask_images['OVERSAMP']
        if use_cmask and (cmask is not None):
            # Exclude actual coronagraphic mask since this will be
            # taken into account during PSF convolution. Not true for
            # all other elements within FOV, ND squares, and mask holder.
            cmask_temp = cmask.copy()
            if 'FULL' in self.det_info['wind_mode']:
                r, th = dist_image(cmask, pixscale=pixscale_over,
                                   center=cdict['cen_sci'], return_theta=True)
                x_asec, y_asec = rtheta_to_xy(r, th)
                ind = (np.abs(x_asec)<10) & (np.abs(y_asec)<5)
                cmask_temp[ind] = 1
            else:
                r, th = dist_image(cmask, pixscale=pixscale_over, return_theta=True)
                x_asec, y_asec = rtheta_to_xy(r, th)
                ind = (np.abs(x_asec)<10) & (np.abs(y_asec)<5)
                cmask_temp[ind] = 1

            disk_image *= cmask_temp

        ##################################
        # Image convolution
        ##################################

        noff = len(self.offset_list)
        xypix = int(np.sqrt(ypix*xpix))
        nproc = nproc_use_convolve(xypix, 1, noff)

        if not self.is_coron: # Single PSF
            psf = self.calc_psf_from_coeff(return_oversample=True, return_hdul=False)
            # Normalize PSF sum to 1.0
            # Otherwise convolve_fft may throw an error if psf.sum() is too small
            norm = psf.sum()
            psf = psf / norm
            image_conv = convolve_fft(disk_image, psf, fftn=fftpack.fftn,
                                      ifftn=fftpack.ifftn, allow_huge=True)
            image_conv *= norm
        else:

            hdu = fits.PrimaryHDU(disk_image)
            hdu.header['PIXELSCL'] = (pixscale_over, 'arcsec/pixel')
            hdul_disk_image = fits.HDUList([hdu])

            # Nearest neighbor convolution
            image_conv = convolve_image(hdul_disk_image, self.psf_list)

        # In case there are any negative values
        image_conv[image_conv<0] = 0
        if return_oversample:
            return image_conv
        else:
            return frebin(image_conv, scale=1/self.oversample)


    def star_flux(self, fluxunit='counts', sp=None):
        """ Stellar flux.

        Return the stellar flux in whatever units, such as
        vegamag, counts, or Jy.

        Parameters
        ----------
        fluxunits : str
            Desired output units, such as counts, vegamag, Jy, etc.
            Must be a Pysynphot supported unit string.
        sp : :mod:`pysynphot.spectrum`
            Normalized Pysynphot spectrum.
        """

        sp = self.sp_sci if sp is None else sp

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
        wfe_drift0=0, wfe_ref_drift=None, wfe_roll_drift=None, **kwargs):
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
            Position angle of first roll position (clockwise, from East to West)
        PA2 : float, None
            Position angle of second roll position. If set equal to PA1
            (or to None), then only one roll will be performed.
            Otherwise, two rolls are performed, each using the specified
            MULTIACCUM settings (doubling the effective exposure time).
        oversample : float
            Set oversampling of final image.
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
        ideal_Poisson : bool
            If set to True, use total signal for noise estimate,
            otherwise MULTIACCUM equation is used.
        quick_PSF : bool
            Rather than generate a spectrum-weighted PSF for planets, use
            the cached PSF scaled by the photon count through the bandpass.
            Resulting PSFs are slightly less accurate, but much faster.
            Default is True.
        use_cmask : bool
            Use the coronagraphic mask image to attenuate planets or disk
            obscurred by a corongraphic mask feature.
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

        if return_oversample:
            pixscale_out = self.pixelscale / oversample
            osamp_out = oversample
        else:
            pixscale_out = self.pixelscale
            osamp_out = 1


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
        xyoff_asec1 = self.pointing_info.get('roll1', (0,0))
        offx_asec, offy_asec = xyoff_asec1

        # Add in bar offset for PSF generation
        delx_asec = offx_asec + bar_offset
        dely_asec = offy_asec
        r, th = xy_to_rtheta(delx_asec, dely_asec)

        # Create stellar PSF centered in image
        im_star = self.gen_offset_psf(r, th, sp=self.sp_sci, return_oversample=True, wfe_drift=wfe_drift0)
        # Stellar cut-out for reference scaling
        im_star_sub = pad_or_cut_to_size(im_star, sub_shape)
        # Expand to full size
        im_star = pad_or_cut_to_size(im_star, (ypix_over, xpix_over))

        # Shift to position relative to center of image
        delx, dely = np.array([delx_asec, dely_asec]) / self.pixelscale
        if ('FULL' in self.det_info['wind_mode']) and (self.image_mask is not None):
            cdict = coron_ap_locs(self.module, self.channel, self.image_mask, full=True)
            xcen, ycen = cdict['cen_sci']
            delx += (xcen - xpix/2)
            dely += (ycen - ypix/2)
        else:
            xcen, ycen = (xpix/2 + bar_offpix, ypix/2)
        # Account for possible oversampling
        xcen_over, ycen_over = np.array([xcen, ycen]) * oversample
        cen_over = (xcen_over, ycen_over)
        delx_over, dely_over = np.array([delx, dely]) * oversample

        # Perform shift and create slope image
        im_star = fshift(im_star, delx=delx_over, dely=dely_over, pad=True)
        im_roll1 = self.gen_slope_image(PA=PA1, xyoff_asec=xyoff_asec1, im_star=im_star, return_oversample=True, **kwargs)

        if no_ref and (roll_angle==0):
            _log.warning('If no_ref=True, then PA1 must not equal PA2. Setting no_ref=False')
            no_ref = False

        # Include disk and companion flux for calculating the reference scale factor
        if ref_scale_all:
            im_star_sub = pad_or_cut_to_size(im_star_sub, sub_shape, offset_vals=(-1*dely_over,-1*delx_over))

        # Fix saturated pixels
        if fix_sat:
            im_roll1 = self._fix_sat_im(im_roll1, oversample=oversample, **kwargs)

        # Pure roll subtraction (no reference PSF)
        ##################################################

        if no_ref:
            # Roll2
            wfe_drift1 = wfe_drift0
            wfe_drift2 = wfe_drift1 + wfe_roll_drift

            xyoff_asec2 = self.pointing_info.get('roll2', (0,0))

            # Create roll2 image
            if (np.abs(wfe_roll_drift) < eps) and (xyoff_asec1==xyoff_asec2):
                im_roll2 = self.gen_slope_image(PA=PA2, im_star=im_star, do_roll2=True, return_oversample=True, **kwargs)
            else:
                im_roll2 = self.gen_slope_image(PA=PA2, wfe_drift0=wfe_drift1, do_roll2=True, return_oversample=True, **kwargs)

            # Fix saturated pixels
            if fix_sat:
                im_roll2 = self._fix_sat_im(im_roll2, oversample=oversample, **kwargs)

            # if oversample != 1:
            #     im_roll1 = frebin(im_roll1, scale=oversample)
            #     im_roll2 = frebin(im_roll2, scale=oversample)
            if oversample>1:
                kernel = Gaussian2DKernel(0.5*oversample)
                im_roll1 = convolve_fft(im_roll1, kernel, allow_huge=True)
                im_roll2 = convolve_fft(im_roll2, kernel, allow_huge=True)

            # Shift roll images by pointing offsets
            dx, dy = -1*self.pointing_info['roll1'] * oversample / self.pixelscale
            im_roll1_sh = fshift(im_roll1, delx=dx, dely=dy)
            dx, dy = -1*self.pointing_info['roll2'] * oversample / self.pixelscale
            im_roll2_sh = fshift(im_roll2, delx=dx, dely=dy)
            # Difference the two rolls
            diff_r1 = im_roll1_sh - im_roll2_sh
            diff_r2 = -1 * diff_r1

            # De-rotate each image
            diff_r1_rot = rotate_offset(diff_r1, PA1, cen=cen_over, reshape=True, cval=np.nan)
            diff_r2_rot = rotate_offset(diff_r2, PA2, cen=cen_over, reshape=True, cval=np.nan)

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
            hdu.header['OVERSAMP'] = (oversample, 'Oversample compared to detector pixels')
            hdu.header['PIXELSCL'] = (self.pixelscale / oversample, 'Image pixel scale (asec/pix)')
            hdu.header['FILTER']   = (self.filter, 'Filter name')
            if self.is_lyot:  hdu.header['PUPIL']    = (self.pupil_mask, 'Pupil plane mask')
            if self.is_coron: hdu.header['CORONMSK'] = (self.image_mask, 'Image plane mask')
            hdu.header['TEXP_SCI'] = (2*self.multiaccum_times['t_exp'], 'Total science exposure time (sec)')
            hdu.header['TEXP_REF'] = (0, 'Total reference exposure time (sec)')
            hdu.header['ROLL_ANG'] = (roll_angle, 'Delta roll angle (deg)')

            hdulist = fits.HDUList([hdu])

            roll_names = ['ROLL1', 'ROLL2']
            keys = ['roll1', 'roll2']
            pa_vals = [PA1, PA2]
            for ii, im in enumerate([im_roll1, im_roll2]):
                hdu = fits.ImageHDU(im)
                hdu.header['EXTNAME'] = (roll_names[ii])
                hdu.header['OVERSAMP'] = (oversample, 'Oversample compared to detector pixels')
                hdu.header['PIXELSCL'] = (self.pixelscale / oversample, 'Image pixel scale (asec/pix)')
                hdu.header['FILTER']   = (self.filter, 'Filter name')
                if self.is_lyot:  hdu.header['PUPIL']    = (self.pupil_mask, 'Pupil plane mask')
                if self.is_coron: hdu.header['CORONMSK'] = (self.image_mask, 'Image plane mask')
                hdu.header['TEXP']     = (self.Detector.time_exp, 'Total exposure time (sec)')
                hdu.header['PA']       = (pa_vals[ii], "Position angle (deg)")
                hdu.header['DX_ASEC']  = (self.pointing_info[keys[ii]][0], 'Pointing offset in x-ideal (asec)')
                hdu.header['DY_ASEC']  = (self.pointing_info[keys[ii]][1], 'Pointing offset in y-ideal (asec)')

                hdulist.append(hdu)

            return hdulist

        # Continuing with a ref PSF subtraction algorithm
        ##################################################

        # Reference star slope simulation
        # Ideal slope
        wfe_drift_ref = wfe_drift0 + wfe_ref_drift

        # Additional field offsets
        xyoff_asec_ref = self.pointing_info.get('ref', (0,0))
        offx_asec, offy_asec = xyoff_asec_ref

        # Add in bar offset for PSF generation
        delx_asec = offx_asec + bar_offset
        dely_asec = offy_asec
        r, th = xy_to_rtheta(delx_asec, dely_asec)

        # Create stellar PSF centered in image
        im_ref = self.gen_offset_psf(r, th, sp=self.sp_ref, return_oversample=True, wfe_drift=wfe_drift_ref)
        # Stellar cut-out for reference scaling
        im_ref_sub = pad_or_cut_to_size(im_ref, sub_shape)
        # Expand to full size
        im_ref = pad_or_cut_to_size(im_ref, (ypix_over, xpix_over))

        # Shift to position relative to center of image
        delx, dely = np.array([delx_asec, dely_asec]) / self.pixelscale
        if ('FULL' in self.det_info['wind_mode']) and (self.image_mask is not None):
            delx += (xcen - xpix/2)
            dely += (ycen - ypix/2)
        delx_over, dely_over = np.array([delx, dely]) * oversample

        # Perform shift and create slope image
        im_ref = fshift(im_ref, delx=delx_over, dely=dely_over, pad=True)
        # With noise
        im_ref = self.gen_slope_image(im_star=im_ref, do_ref=True, return_oversample=True, **kwargs)

        # Fix saturated pixels
        if fix_sat:
            im_ref = self._fix_sat_im(im_ref, do_ref=True, oversample=oversample, **kwargs)

        # Determine reference star scale factor
        scale1 = scale_ref_image(im_star_sub, im_ref_sub)
        _log.debug('scale1: {0:.3f}'.format(scale1))
        # if oversample != 1:
        #     im_ref   = frebin(im_ref, scale=oversample)
        #     im_roll1 = frebin(im_roll1, scale=oversample)
        if oversample>1:
            kernel = Gaussian2DKernel(0.5*oversample)
            im_ref = convolve_fft(im_ref, kernel, allow_huge=True)
            im_roll1 = convolve_fft(im_roll1, kernel, allow_huge=True)

        # Shift roll images by pointing offsets
        dx, dy = -1*self.pointing_info['roll1'] * oversample / self.pixelscale
        im_roll1_sh = fshift(im_roll1, delx=dx, dely=dy)
        dx, dy = -1*self.pointing_info['ref'] * oversample / self.pixelscale
        im_ref_sh = fshift(im_ref, delx=dx, dely=dy)
                    
        # Telescope Roll 2 with reference subtraction
        if (abs(roll_angle) > eps):
            # Subtraction with and without scaling
            im_diff1_r1 = im_roll1_sh - im_ref_sh
            im_diff2_r1 = im_roll1_sh - im_ref_sh * scale1

            # WFE drift difference between rolls
            wfe_drift1 = wfe_drift0
            wfe_drift2 = wfe_drift1 + wfe_roll_drift

            # Create roll2 image
            xyoff_asec2 = (0,0)
            if (np.abs(wfe_roll_drift) < eps) and (xyoff_asec1==xyoff_asec2):
                # Assume Roll2 and Roll1 have exactly the same position offset and WFE drift
                im_star2     = im_star
                im_star2_sub = im_star_sub
            else:
                offx_asec, offy_asec = xyoff_asec2
                # Add in bar offset for PSF generation
                delx_asec = offx_asec + bar_offset
                dely_asec = offy_asec
                r, th = xy_to_rtheta(delx_asec, dely_asec)

                # Create stellar PSF centered in image
                im_star2     = self.gen_offset_psf(r, th, sp=self.sp_sci, return_oversample=True, wfe_drift=wfe_drift2)
                im_star2_sub = pad_or_cut_to_size(im_star2, sub_shape)
                im_star2     = pad_or_cut_to_size(im_star2, (ypix_over, xpix_over))

                # Shift to position relative to center of image
                delx, dely = np.array([delx_asec, dely_asec]) / self.pixelscale
                if ('FULL' in self.det_info['wind_mode']) and (self.image_mask is not None):
                    delx += (xcen - xpix/2)
                    dely += (ycen - ypix/2)
                delx_over, dely_over = np.array([delx, dely]) * oversample

                # Perform shift and create slope image
                im_star2 = fshift(im_star2, delx=delx_over, dely=dely_over, pad=True)
            # Create Roll2 slope image
            im_roll2 = self.gen_slope_image(PA=PA2, im_star=im_star2, do_roll2=True, return_oversample=True, **kwargs)

            # Include disk and companion flux for calculating the reference scale factor
            if ref_scale_all:
                im_star2_sub = pad_or_cut_to_size(im_star2_sub, sub_shape, offset_vals=(-1*dely_over,-1*delx_over))

            # Fix saturated pixels
            if fix_sat:
                im_roll2 = self._fix_sat_im(im_roll2, oversample=oversample, **kwargs)

            # Subtract reference star from Roll 2
            #im_star_sub = pad_or_cut_to_size(im_star+im_pl_r2, sub_shape)
            scale2 = scale_ref_image(im_star2_sub, im_ref_sub)
            _log.debug('scale2: {0:.3f}'.format(scale2))
            # if oversample != 1:
            #     im_roll2 = frebin(im_roll2, scale=oversample)
            if oversample>1:
                kernel = Gaussian2DKernel(0.5*oversample)
                im_roll2 = convolve_fft(im_roll2, kernel, allow_huge=True)

            # Shift roll images by pointing offsets
            dx, dy = -1*self.pointing_info['roll2'] * oversample / self.pixelscale
            im_roll2_sh = fshift(im_roll2, delx=dx, dely=dy)
                
            # Subtraction with and without scaling
            im_diff1_r2 = im_roll2_sh - im_ref_sh
            im_diff2_r2 = im_roll2_sh - im_ref_sh * scale2
            #im_diff_r2 = optimal_difference(im_roll2, im_ref, scale2)

            # De-rotate each image
            # Convention for rotate() is opposite PA_offset
            diff1_r1_rot = rotate_offset(im_diff1_r1, PA1, cen=cen_over, reshape=True, cval=np.nan)
            diff2_r1_rot = rotate_offset(im_diff2_r1, PA1, cen=cen_over, reshape=True, cval=np.nan)
            diff1_r2_rot = rotate_offset(im_diff1_r2, PA2, cen=cen_over, reshape=True, cval=np.nan)
            diff2_r2_rot = rotate_offset(im_diff2_r2, PA2, cen=cen_over, reshape=True, cval=np.nan)

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
                # Choose better PSF removal version
                final = final2

            texp_sci = 2 * self.multiaccum_times['t_exp']

        # For only a single roll
        else:
            # Optimal differencing (with scaling only on the inner regions)
            if opt_diff:
                final = optimal_difference(im_roll1_sh, im_ref_sh, scale1)
            else:
                final = im_roll1_sh - im_ref_sh * scale1

            final = rotate_offset(final, PA1, cen=cen_over, reshape=True, cval=np.nan)
            texp_sci = self.multiaccum_times['t_exp']

        # De-rotate PA1 to North
        #if abs(PA1) > eps:
        #    final = rotate(final, PA1, reshape=False)

        # Rebin if requesting detector sampled images 
        if not return_oversample:
            final = frebin(final, scale=1/oversample)
            im_roll1 = frebin(im_roll1, scale=1/oversample)

        hdu = fits.PrimaryHDU(final)
        hdu.header['EXTNAME'] = ('REF_SUB')
        hdu.header['OVERSAMP'] = (osamp_out, 'Oversample compared to detector pixels')
        hdu.header['PIXELSCL'] = (pixscale_out, 'Image pixel scale (asec/pix)')
        hdu.header['FILTER']   = (self.filter, 'Filter name')
        if self.is_lyot: hdu.header['PUPIL'] = (self.pupil_mask, 'Pupil plane mask')
        if self.is_coron: hdu.header['CORONMSK'] = (self.image_mask, 'Image plane mask')
        hdu.header['TEXP_SCI'] = (texp_sci, 'Total science exposure time (sec)')
        hdu.header['TEXP_REF'] = (self.Detector_ref.time_exp, 'Total reference exposure time (sec)')
        hdu.header['ROLL_ANG'] = (roll_angle, 'Delta roll angle (deg)')

        hdulist = fits.HDUList([hdu])

        # Add Roll1
        hdu = fits.ImageHDU(im_roll1)
        hdu.header['EXTNAME'] = ('ROLL1')
        hdu.header['OVERSAMP'] = (osamp_out, 'Oversample compared to detector pixels')
        hdu.header['PIXELSCL'] = (pixscale_out, 'Image pixel scale (asec/pix)')
        hdu.header['FILTER']   = (self.filter, 'Filter name')
        if self.is_lyot:  hdu.header['PUPIL']    = (self.pupil_mask, 'Pupil plane mask')
        if self.is_coron: hdu.header['CORONMSK'] = (self.image_mask, 'Image plane mask')
        hdu.header['TEXP']     = (self.Detector.time_exp, 'Total exposure time (sec)')
        hdu.header['PA']       = (PA1, "Position angle (deg)")
        hdu.header['DX_ASEC']  = (self.pointing_info['roll1'][0], 'Pointing offset in x-ideal (asec)')
        hdu.header['DY_ASEC']  = (self.pointing_info['roll1'][1], 'Pointing offset in y-ideal (asec)')
        hdulist.append(hdu)

        # Add Roll2
        try:
            if not return_oversample:
                im_roll2 = frebin(im_roll2, scale=1/oversample)
            hdu = fits.ImageHDU(im_roll2)
            hdu.header['EXTNAME'] = ('ROLL2')
            hdu.header['OVERSAMP'] = (osamp_out, 'Oversample compared to detector pixels')
            hdu.header['PIXELSCL'] = (pixscale_out, 'Image pixel scale (asec/pix)')
            hdu.header['FILTER']   = (self.filter, 'Filter name')
            if self.is_lyot:  hdu.header['PUPIL']    = (self.pupil_mask, 'Pupil plane mask')
            if self.is_coron: hdu.header['CORONMSK'] = (self.image_mask, 'Image plane mask')
            hdu.header['TEXP']     = (self.Detector.time_exp, 'Total exposure time (sec)')
            hdu.header['PA']       = (PA2, "Position angle (deg)")
            hdu.header['DX_ASEC']  = (self.pointing_info['roll2'][0], 'Pointing offset in x-ideal (asec)')
            hdu.header['DY_ASEC']  = (self.pointing_info['roll2'][1], 'Pointing offset in y-ideal (asec)')
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
            hdu.header['PIXELSCL'] = (pixscale_out, 'Image pixel scale (asec/pix)')
            hdu.header['FILTER']   = (self.filter, 'Filter name')
            if self.is_lyot:  hdu.header['PUPIL']    = (self.pupil_mask, 'Pupil plane mask')
            if self.is_coron: hdu.header['CORONMSK'] = (self.image_mask, 'Image plane mask')
            hdu.header['TEXP']     = (self.Detector_ref.time_exp, 'Total exposure time (sec)')
            hdu.header['PA']       = (0, "Position angle (deg)")
            hdu.header['DX_ASEC']  = (self.pointing_info['ref'][0], 'Pointing offset in x-ideal (asec)')
            hdu.header['DY_ASEC']  = (self.pointing_info['ref'][1], 'Pointing offset in y-ideal (asec)')
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
            Option to pass an already pre-made difference image.

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

        data_rebin = frebin(data, scale=1/header['OVERSAMP'])
        xpix, ypix = (self.det_info['xpix'], self.det_info['ypix'])
        pixscale = self.pixelscale

        # Radial noise
        rr, stds = radial_std(data, pixscale=pixscale, oversample=header['OVERSAMP'], 
                              supersample=False, func=func_std)

        # Normalize by psf max value
        if no_ref:
            # No reference image subtraction; pure roll subtraction
            # Generate 2 PSFs separated by roll angle to find self-subtracted PSF peak

            off_vals = []
            max_vals = []
            rvals_pix = np.insert(np.arange(1,xpix/2,5), 0, 0.1)
            for j, roff_pix in enumerate(rvals_pix):
                roff_asec = roff_pix * pixscale
                psf1 = self.gen_offset_psf(roff_asec, 0, return_oversample=False)
                psf2 = self.gen_offset_psf(roff_asec, roll_angle, return_oversample=False)

                psf1 = fshift(psf1, delx=0, dely=roff_pix, pad=False)
                xoff, yoff = xy_rot(0, roff_pix, 10)
                psf2 = fshift(psf2, delx=xoff, dely=yoff, pad=False)

                diff = psf1 - psf2
                maxv = diff.max()

                off_vals.append(roff_pix)
                max_vals.append(maxv)
                if maxv >= 0.95*psf1.max():
                    off_vals = off_vals + [roff_pix+5, xpix/2]
                    max_vals = max_vals + [psf1.max(), psf1.max()]
                    break

            max_vals = np.array(max_vals)
            off_asec = np.array(off_vals) * pixscale

            #psf_max = np.interp(rr, off_asec, max_vals)
            psf_max_log = np.interp(rr, off_asec, np.log10(max_vals))
            psf_max = 10**psf_max_log

        elif self.mask is None: # Direct imaging
            psf = self.gen_offset_psf(0, 0)
            psf_max = psf.max()
        elif self.mask[-1]=='R': # Round masks
            npix = np.max([xpix,ypix])

            # Image mask
            im_mask = coron_trans(self.mask, module=self.module, pixelscale=pixscale, npix=npix, nd_squares=False)
            im_mask = pad_or_cut_to_size(im_mask, data_rebin.shape)

            ny, nx = im_mask.shape
            xv = (np.arange(nx) - nx/2) * pixscale

            # a and b coefficients at each offset location
            avals = np.interp(rr, xv, im_mask[ny//2,:]**2)
            bvals = 1 - avals

            # Linearly combine PSFs
            fov_pix = self.psf_info['fov_pix']
            psf_center  = krebin(self.psf_center_over, (fov_pix,fov_pix))
            psf_offaxis = krebin(self.psf_offaxis_over, (fov_pix,fov_pix))
            psf_max = np.array([np.max(psf_offaxis*a + psf_center*b)
                                for a,b in zip(avals,bvals)])
            psf_max[rr>10] = psf_max[(rr>5) & (rr<10)].max()

        elif self.mask[-1]=='B': # Bar masks
            npix = np.max([xpix,ypix])

            # Image mask
            im_mask = coron_trans(self.mask, module=self.module, pixelscale=pixscale, npix=npix, nd_squares=False)
            im_mask = pad_or_cut_to_size(im_mask, data_rebin.shape)

            # For offaxis PSF max values, use fiducial at bar_offset location
            bar_offset = self.bar_offset
            ny, nx = im_mask.shape
            xloc = int(bar_offset / pixscale + nx/2)
            mask_cut = im_mask[:,xloc]

            # Interpolate along the horizontal cut
            yv = (np.arange(ny) - ny/2) * pixscale
            avals = np.interp(rr, yv, mask_cut**2)
            bvals = 1 - avals

            # Get PSF at middle of bar
            vals = self.psf_center_offsets
            arr = np.array(self.psf_center_over)
            func = interp1d(vals, arr, axis=0, kind='linear')
            psf_center = func(bar_offset)

            # Linearly combine PSFs
            fov_pix = self.psf_info['fov_pix']
            psf_center  = krebin(psf_center, (fov_pix,fov_pix))
            psf_offaxis = krebin(self.psf_offaxis_over, (fov_pix,fov_pix))
            psf_max = np.array([np.max(psf_offaxis*a + psf_center*b)
                                for a,b in zip(avals,bvals)])
            psf_max[rr>10] = psf_max[(rr>5) & (rr<10)].max()

        #plt.plot(rr[rr<3], psf_max[rr<3])


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
            Position angle of roll position (clockwise, from East to West).
        xyoff_asec : 
            Positional offset of scene from reference location in 'idl' coords.
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

        Keyword Args
        ------------
        quick_PSF : bool
            Rather than generate a spectrum-weighted PSF for planets, use
            the cached PSF scaled by the photon count through the bandpass.
            Resulting PSFs are slightly less accurate, but much faster.
            Default is True.
        use_cmask : bool
            Use the coronagraphic mask image to attenuate planet or disk that
            is obscurred by a corongraphic mask feature.
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
        else:
            offx_asec, offy_asec = xyoff_asec

        # Add in bar offset for PSF generation
        delx_asec = offx_asec + bar_offset
        dely_asec = offy_asec
        r, th = xy_to_rtheta(delx_asec, dely_asec)

        # Shift to position relative to center of image
        delx, dely = np.array([delx_asec, dely_asec]) / self.pixelscale
        if ('FULL' in self.det_info['wind_mode']) and (self.image_mask is not None):
            cdict = coron_ap_locs(self.module, self.channel, self.image_mask, full=True)
            xcen, ycen = cdict['cen_sci']
            delx += (xcen - xpix/2)
            dely += (ycen - ypix/2)
        else:
            xcen, ycen = (xpix/2 + bar_offpix, ypix/2)
        # Account for possible oversampling
        delx_over, dely_over = np.array([delx, dely]) * oversample

        # Stellar PSF doesn't rotate
        if im_star is None:
            _log.info('  gen_slope_image: Creating stellar PSF...')
            im_star = self.gen_offset_psf(r, th, sp=sp, return_oversample=return_oversample, wfe_drift=wfe_drift)
            im_star = pad_or_cut_to_size(im_star, (ypix_over, xpix_over), offset_vals=(dely_over,delx_over))

        # Disk and Planet images
        if do_ref:
            no_disk = no_planets = True
        else:
            no_disk    = exclude_disk    and exclude_noise
            no_planets = exclude_planets and exclude_noise

        # Make sure to include planets and disks for Poisson noise calculations
        kwargs2 = kwargs.copy()
        kwargs2['PA_offset']  = PA
        kwargs2['xyoff_asec'] = xyoff_asec
        kwargs2['return_oversample'] = return_oversample
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

    def gen_pointing_offsets(self, sgd_type=None, slew_std=5, fsm_std=2.5):
        """
        Create a series of x and y position offsets for a SGD pattern.
        This includes the central position as the first in the series.
        By default, will also add random movement errors using the
        `slew_std` and `fsm_std` keywords. Returned values are in arcsec.
        
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

        if sgd_type == 'auto':
            if self.is_coron and self.image_mask[-1]=='R':
                sgd_type = '5diamond'
            elif self.is_coron and self.image_mask[-1]=='B':
                sgd_type = '5bar'
            else:
                sgd_type = '5diamond'

        if sgd_type is None:
            xyoff_ref = np.random.normal(scale=slew_std, size=2) / 1000
            fsm_std = None
        else:
            xoff_ref, yoff_ref = gen_sgd_offsets(sgd_type, slew_std=slew_std, fsm_std=fsm_std)
            xyoff_ref = np.array([xoff_ref,yoff_ref]).transpose()

        xyoff_roll1 = np.random.normal(scale=slew_std, size=2) / 1000
        xyoff_roll2 = np.random.normal(scale=slew_std, size=2) / 1000

        _log.info("Saving SGD position info to `self.pointing_info` dictionary attribute")
        self.pointing_info = {
            'sgd_type': sgd_type, 'slew_std': slew_std, 'fsm_std': fsm_std,
            'roll1': xyoff_roll1, 'roll2': xyoff_roll2, 'ref': xyoff_ref,
        }

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
        quick_PSF : bool
            Rather than generate a spectrum-weighted PSF for planets, use
            the cached PSF scaled by the photon count through the bandpass.
            Resulting PSFs are slightly less accurate, but much faster.
            Default is True.
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


class obs_coronagraphy(obs_hci):
    """Deprecated class. Use :class:`obs_hci` instead."""
    def __init__(self, *args, **kwargs):

        _log.warning('obs_coronagraphy is depcrecated. Use obs_hci instead.')
        obs_hci.__init__(self, *args, **kwargs)



def model_to_hdulist(args_model, sp_star, filter_or_bp,
    pupil=None, mask=None, module=None):

    """HDUList from model FITS file.

    Convert disk model to an HDUList with units of photons/sec/pixel.
    If observed filter is different than input filter, we assume that
    the disk has a flat scattering, meaning it scales with stellar
    continuum. Pixel sizes and distances are left unchanged, and
    stored in header.

    Parameters
    ----------
    args_model - tuple
        Arguments describing the necessary model information:
            - fname   : Name of model file or an HDUList
            - scale0  : Pixel scale (in arcsec/pixel)
            - dist0   : Assumed model distance
            - wave_um : Wavelength of observation
            - units0  : Assumed flux units (ie., MJy/arcsec^2 or muJy/pixel)
    sp_star : :mod:`pysynphot.spectrum`
        A pysynphot spectrum of central star. Used to adjust observed
        photon flux if filter differs from model input
    filter_or_bp : str, :mod:`pysynphot.obsbandpass`
        Either the name of a filter or a Pysynphot bandpass.
    pupil : str, None
        Instrument pupil setting (Lyot mask, grism, DHS, etc.)
    mask : str, None
        Coronagraphic mask.
    module : str
        NIRCam module 'A' or 'B'.
    """

    #filt, mask, pupil = args_inst
    fname, scale0, dist0, wave_um, units0 = args_model
    wave0 = wave_um * 1e4

    # Get the bandpass for the filter, pupil, and mask
    # This corresponds to the flux at the entrance pupil
    # for the particular filter.
    #bp = read_filter(filter, pupil=pupil, mask=mask, module=module)
    # Get filter throughput and create bandpass
    if isinstance(filter_or_bp, six.string_types):
        filter = filter_or_bp
        bp = read_filter(filter, pupil=pupil, mask=mask, module=module)
    else:
        bp = filter_or_bp
        filter = bp.name



    # Detector pixel scale and PSF oversample
    #detscale = channel_select(bp)[0]
    #oversample = 4
    #pixscale_over = detscale / oversample

    #### Read in the image, then convert from mJy/arcsec^2 to photons/sec/pixel

    if isinstance(fname, fits.HDUList):
        hdulist = deepcopy(fname)
    else:
        # Open file
        hdulist = fits.open(fname)
        #data    = hdulist[0].data#.copy()
        #header  = hdulist[0].header
        #hdutemp.close()

    # Break apart units0
    units_list = units0.split('/')
    if 'mJy' in units_list[0]:
        units_pysyn = S.units.mJy()
    elif 'uJy' in units_list[0]:
        units_pysyn = S.units.muJy()
    elif 'nJy' in units_list[0]:
        units_pysyn = S.units.nJy()
    elif 'MJy' in units_list[0]:
        hdulist[0].data *= 1000 # Convert to Jy
        units_pysyn = S.units.Jy()
    elif 'Jy' in units_list[0]: # Jy should be last
        units_pysyn = S.units.Jy()
    else:
        errstr = "Do not recognize units0='{}'".format(units0)
        raise ValueError(errstr)

    # Convert from input units to photlam (photons/sec/cm^2/A/angular size)
    im = units_pysyn.ToPhotlam(wave0, hdulist[0].data)

    # We want to assume scattering is flat in photons/sec/A
    # This means everything scales with stellar continuum
    sp_star.convert('photlam')
    wstar, fstar = (sp_star.wave/1e4, sp_star.flux)

    # Compare observed wavelength to image wavelength
    wobs_um = bp.avgwave() / 1e4 # Current bandpass wavelength

    wdel = np.linspace(-0.1,0.1)
    f_obs = np.interp(wobs_um+wdel, wstar, fstar)
    f0    = np.interp(wave_um+wdel, wstar, fstar)
    im *= np.mean(f_obs / f0)

    # Convert to photons/sec/pixel
    im *= bp.equivwidth() * S.refs.PRIMARY_AREA
    # If input units are per arcsec^2 then scale by pixel scale
    # This will be ph/sec for each oversampled pixel
    if ('arcsec' in units_list[1]) or ('asec' in units_list[1]):
        im *= scale0**2
    elif 'mas' in units_list[1]:
        im *= (scale0*1000)**2

    # Save into HDUList
    hdulist[0].data = im

    hdulist[0].header['UNITS']    = 'photons/sec'
    hdulist[0].header['PIXELSCL'] = (scale0, 'arcsec/pixel')
    hdulist[0].header['DISTANCE'] = (dist0, 'parsecs')

    return hdulist


def _wrap_convolve_for_mp(args):
    """
    Internal helper routine for parallelizing computations across multiple processors.

    Create a list of arguments to pass to this function:
        worker_args = [(inst, image, rho, offset_list, i) for i,inst in enumerate(nrc_star_list)]

    Then create a theadpool:
        pool = mp.Pool(nproc)
        images = pool.map(_wrap_coeff_for_mp, worker_args)
        pool.close()
        images = np.array(images)

    For single processing, just use:
        images = [_wrap_convolve_for_mp(wa) for wa in worker_args]
        images = np.array(images)

    For multiprocessing:
    """

    psf, model, rho, offset_list, i = args

    noff = len(offset_list)
    if noff==1:
        r1 = 0
        r2 = rho.max()+1
    elif i==0:
        r1 = offset_list[i]
        r2 = (offset_list[i] + offset_list[i+1]) / 2.
    elif i==noff-1:
        r1 = (offset_list[i] + offset_list[i-1]) / 2.
        r2 = rho.max()+1.
    else:
        r1 = (offset_list[i] + offset_list[i-1]) / 2.
        r2 = (offset_list[i] + offset_list[i+1]) / 2.

    ind = (rho>=r1) & (rho<r2)
    im_temp = model.copy()
    im_temp[~ind] = 0

    # Normalize PSF sum to 1.0
    # Otherwise convolve_fft may throw an error if psf.sum() is too small
    norm = psf.sum()
    psf = psf / norm
    res = convolve_fft(im_temp, psf, fftn=fftpack.fftn, ifftn=fftpack.ifftn, allow_huge=True)
    res *= norm

    return res

def _wrap_conv_trans_for_mp(args):
    """
    Similar to `_wrap_convolve_for_mp` except bins data by mask
    transmission value.
    """

    psf, model, tvals_edges, cmask, i = args

    tvals1 = tvals_edges[i]
    tvals2 = tvals_edges[i+1]
    ind = (cmask>=tvals1) & (cmask<=tvals2)

    im_temp = model.copy()
    im_temp[~ind] = 0

    if np.allclose(im_temp,0):
        # No need to convolve anything if no flux!
        return im_temp
    else:
        # Normalize PSF sum to 1.0
        # Otherwise convolve_fft may throw an error if psf.sum() is too small
        norm = psf.sum()
        psf = psf / norm
        res = convolve_fft(im_temp, psf, fftn=fftpack.fftn, ifftn=fftpack.ifftn, allow_huge=True)
        res *= norm

        return res

