from __future__ import division, print_function, unicode_literals

from astropy.convolution import convolve_fft, Gaussian2DKernel
from scipy.ndimage.interpolation import rotate
from scipy import fftpack

# Import libraries
from . import *
from .nrc_utils import *

import logging
_log = logging.getLogger('pynrc')

eps = np.finfo(float).eps

class obs_coronagraphy(NIRCam):
    """NIRCam coronagraphy (and direct imaging)
    
    Subclass of the NIRCam instrument class used to observe stars 
    (plus exoplanets and disks) with either a coronagraph or direct 
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
    sp_ref : :mod:`pysynphot.spectrum`
        A pysynphot spectrum of reference target (e.g., central star).
        Should already be normalized to the apparent flux.
    distance : float
        Distance in parsecs to the science target. This is used for
        flux normalization of the planets and disk.
    wfe_drift: float
        WFE drift in nm between the science and reference targets.
        Expected values are between ~3-10 nm
    xpix : int
        Size of the detector readout along the x-axis. The detector is
        assumed to be in window mode  unless the user explicitly 
        sets wind_mode='FULL'.
    ypix : int
        Size of the detector readout along the y-axis. The detector is
        assumed to be in window mode  unless the user explicitly 
        sets wind_mode='FULL'.
    wind_mode : str
        'FULL', 'STRIPE', or 'WINDOW'
    disk_hdu : HDUList
        A model of the disk in photons/sec. This requires header
        keywords PIXSCALE (in arcsec/pixel) and DISTANCE (in pc).
        
    """
    
    def __init__(self, sp_sci, sp_ref, distance, wfe_drift=10, offset_list=None, 
                 wind_mode='WINDOW', xpix=320, ypix=320, disk_hdu=None, 
                 verbose=False, **kwargs):
                 
        if 'FULL'   in wind_mode: xpix = ypix = 2048
        if 'STRIPE' in wind_mode: xpix = 2048

        #super(NIRCam,self).__init__(**kwargs)
        # Not sure if this works for both Python 2 and 3
        NIRCam.__init__(self, wind_mode=wind_mode, xpix=xpix, ypix=ypix, **kwargs)
        
        # Spectral models
        self.sp_sci = sp_sci
        self.sp_ref = sp_ref
        self._wfe_drift = wfe_drift
        
        # Distance to source in pc
        self.distance = distance
        self._planets = []

        # Offsets positions to build PSFs
        if self.mask is None:
            # if no coronagraphic mask, then only 1 PSF
            self.offset_list = [0.0]
            if offset_list is not None:
                print('No coronagraph, so offset_list automatically set to [0.0].')
        elif offset_list is None:
            self.offset_list = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
        else:
            self.offset_list = offset_list
        
        if verbose: print("Generating list of PSFs...")
        # Faster once PSFs have already been previously generated 
        log_prev = conf.logging_level
        setup_logging('WARN', verbose=False)
        self._gen_psf_off()
        setup_logging(log_prev, verbose=False)
        
        self._gen_ref(verbose=verbose)
        self._set_xypos()
        
        # Rescale input disk image to observation parameters
        self._disk_hdulist_input = disk_hdu
        self._gen_disk_hdulist()
        
        if verbose: print("Finished.")
        
    @property
    def wfe_drift(self):
        """Assumed WFE drift"""
        return self._wfe_drift
    @wfe_drift.setter
    def wfe_drift(self, value):
        """Set the WFE drift value (updates self.nrc_ref)"""
        # Only update if the value changes
        vold = self._wfe_drift; self._wfe_drift = value
        if vold != self._wfe_drift: 
            self._gen_ref()

    def _gen_disk_hdulist(self):
        """Create a correctly scaled disk model image"""
        if self._disk_hdulist_input is None:
            self.disk_hdulist = None
        else:
            xpix = self.det_info['xpix']
            ypix = self.det_info['ypix']
            oversample = self.psf_info['oversample']
        
            disk_hdul = self._disk_hdulist_input
            hdr = disk_hdul[0].header
            
            # Get rid of the central star flux
            # and anything interior to 1.2 l/D
            image = disk_hdul[0].data
            image_rho = dist_image(image, pixscale=self.pix_scale) # Arcsec
            fwhm = 206265 * 1.22 * self.bandpass.avgwave() *1E-10 / 6.5
            image[image_rho < (2*fwhm)] = 0
            #mask  = (image==image.max())
            #indy,indx = np.where(mask == True)
            #image[indy,indx] = 0
                       
            # Resample disk to detector pixel scale
            # args_in  = (input pixelscale,  input distance)
            # args_out = (output pixelscale, output distance)
            args_in = (hdr['PIXELSCL'], hdr['DISTANCE'])
            args_out = (self.pix_scale, self.distance)            
            hdulist_out = image_rescale(disk_hdul, args_in, args_out, cen_star=False)
            
            # Expand to full observation size
            hdulist_out[0].data = pad_or_cut_to_size(hdulist_out[0].data, (ypix,xpix))
            self.disk_hdulist = hdulist_out
                        
        
    def _gen_ref(self, verbose=False):
        """Function to generate Reference observation class"""

        # PSF information
        opd = (self.psf_info['opd'][0], self.psf_info['opd'][1], self._wfe_drift)
        fov_pix = self.psf_info['fov_pix']
        oversample = self.psf_info['oversample']

        # Detector information
        wind_mode = self.det_info['wind_mode']
        xpix = self.det_info['xpix']
        ypix = self.det_info['ypix']
        
        offset_r = self.psf_info['offset_r']
        offset_theta = self.psf_info['offset_theta']
        
        # Create a NIRCam reference class
        # If it already exists, just update OPD info
        try:
            if verbose: print("Updating NIRCam reference OPD...")
            nrc = self.nrc_ref
            nrc.update_psf_coeff(opd=opd)
        except AttributeError:
            if verbose: print("Creating NIRCam reference class...")
            nrc = NIRCam(self.filter, self.pupil, self.mask, module=self.module, \
                         wind_mode=wind_mode, xpix=xpix, ypix=ypix, \
                         fov_pix=fov_pix, oversample=oversample, opd=opd,
                         offset_r=offset_r, offset_theta=offset_theta)        
            self.nrc_ref = nrc

        
    def _gen_psf_off(self):
        """
        Create instances of NIRCam observations that are incrementally offset 
        from coronagraph center to determine maximum value of the detector-
        sampled PSF for determination of contrast. Also saves the list of
        PSFs for later retrieval.
        """
        
        # If no mask, then the PSF looks the same at all radii
        if self.mask is None:
            psf = self.gen_psf()
            self.psf_max_vals = ([0,10], [psf.max(),psf.max()]) # radius and psf max
            self.psf_offsets = [self]
            self.psf_list = [psf]
        else:
            psf_off = []
            psf_max = []
            self.psf_list = []
            self.psf_offsets = []
            for offset in self.offset_list:
            
                # Full FoV
                fov_pix = 2 * np.max(self.offset_list) / self.pix_scale
                # Increase to the next power of 2 and make odd
                fov_pix = int(2**np.ceil(np.log2(fov_pix)+1))
                oversample = self.psf_info['oversample']
            
                nrc_inst = NIRCam(self.filter, self.pupil, self.mask, \
                                  fov_pix=fov_pix, oversample=oversample, \
                                  offset_r=offset)
                # Append offsets and PSF max values
                psf = nrc_inst.gen_psf()
                psf_off.append(offset)                
                psf_max.append(psf.max())
                self.psf_offsets.append(nrc_inst)

                # Shift to center
                offset_pix = -offset / nrc_inst.pix_scale
                psf = fshift(psf, dely=offset_pix, pad=True)
                self.psf_list.append(psf)

                
            # Add background PSF info (without mask) for large distance
            psf_off.append(np.max([np.max(self.offset_list)+1, 4]))
            psf_max.append(self.gen_psf(use_bg_psf=True).max())
            self.psf_max_vals = (psf_off, psf_max)

    def _set_xypos(self):
        """
        Set x0 and y0 subarray positions.
        Needs to be more specific for SW+335R and LW+210R as well as
        for different modules.
        """
        xpix = self.det_info['xpix']
        ypix = self.det_info['ypix']
        x0   = self.det_info['x0']
        y0   = self.det_info['y0']
        mask = self.mask
        wind_mode = self.det_info['wind_mode']
        
        # Coronagraphic Mask
        if ((x0==0) and (y0==0)) and ((mask is not None) and ('MASK' in mask)) \
            and ('FULL' not in wind_mode):
            # Default positions (really depends on xpix/ypix)
            if 'LWB' in mask:
                x0=275; y0=1508
            elif 'SWB' in mask:
                x0=171; y0=236
            elif '430R' in mask:
                x0=916; y0=1502
            elif '335R' in mask:
                x0=1238; y0=1502
            elif '210R' in mask:
                x0=392; y0=224
            
            # Make sure subarray sizes don't push out of bounds
            if (y0 + ypix) > 2048: y0 = 2048 - ypix
            if (x0 + xpix) > 2048: x0 = 2048 - xpix
        
            self.update_detectors(x0=x0, y0=y0)


    def planet_spec(self, Av=0, **kwargs):
        """Exoplanet spectrum.
        
        Return the planet spectrum from Spiegel & Burrows (2012) normalized
        to distance of current target. Output is a :mod:`pysynphot.spectrum`.
        
        Parameters
        ----------
        Av : float
            Extinction magnitude (assumes Rv=4.0).
            
        Keyword Args
        ------------
        atmo : str
            A string consisting of one of four atmosphere types:
            ['hy1s', 'hy3s', 'cf1s', 'cf3s'].
        mass: int
            Number 1 to 15 Jupiter masses.
        age: float
            Age in millions of years (1-1000).
        entropy: float
            Initial entropy (8.0-13.0) in increments of 0.25

        accr : bool
            Include accretion (default: False)?
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
        # Create planet class and convert to Pysynphot spectrum
        planet = planets_sb12(distance=self.distance, **kwargs)
        sp = planet.export_pysynphot()

        # Add extinction from the disk
        Rv = 4.0
        if Av>0: sp *= S.Extinction(Av/Rv,name='mwrv4')
        
        return sp
    

    @property
    def planets(self):
        """Planet info (if any exists)"""
        return self._planets

    def add_planet(self, atmo='hy3s', mass=10, age=100, entropy=10,
        rtheta=(0,0), runits='AU', Av=0, renorm_args=None, sptype=None,
        accr=False, mmdot=None, mdot=None, accr_rin=2, truncated=False):
        """Insert a planet into observation.
        
        Add exoplanet information that will be used to generate a point
        source image using a spectrum from Spiegel & Burrows (2012).
        Use self.kill_planets() to delete them.
        
        Coordinate convention is for +V3 up and +V2 to left.
        
        Parameters
        ----------
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

        rtheta : tuple
            Radius and position angle relative to stellar position.
        runits : str
            What units is radius? Valid values are 'AU', 'asec', or 'pix'.

        accr : bool
            Include accretion (default: False)?
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

        # Size of subarray image in terms of pixels
        image_shape = (self.det_info['ypix'], self.det_info['xpix'])
        
        # XY location of planets within subarray with units from runits keyword
        loc = rtheta_to_xy(rtheta[0], rtheta[1])

        # Define pixel location relative to the center of the subarray
        au_per_pixel = self.distance*self.pix_scale
        if 'AU' in runits:
            xoff, yoff = np.array(loc) / au_per_pixel
        elif ('asec' in runits) or ('arcsec' in runits):
            xoff, yoff = np.array(loc) / self.pix_scale
        elif ('pix' in runits):
            xoff, yoff = loc
        else:
            errstr = "Do not recognize runits='{}'".format(runits)
            raise ValueError(errstr)
            
        # Offset in terms of arcsec
        xoff_asec, yoff_asec = np.array([xoff, yoff]) * self.pix_scale
        _log.debug('(xoff,yoff) = {} pixels'.format((xoff,yoff)))
        _log.debug('(xoff_asec,yoff_asec) = {} arcsec'.format((xoff_asec,yoff_asec)))
    
        # Make sure planet is within image bounds
        sh_diff = np.abs(np.array([yoff,xoff])) - np.array(image_shape)/2
        if np.any(sh_diff>=0):
            _log.warning('xoff,yoff = {} is beyond image boundaries.'.format((xoff,yoff)))
            
        # X and Y pixel offsets from center of image
        # Dictionary of planet info
        if sptype is None:
            d = {'xyoff_pix':(xoff,yoff), 'atmo':atmo, 'mass':mass, 'age':age, 
                 'entropy':entropy, 'Av':Av, 'renorm_args':renorm_args, 
                 'accr':accr, 'mmdot':mmdot, 'mdot':mdot, 'accr_rin':accr_rin, 
                 'truncated':truncated}
        else:
            d = {'xyoff_pix':(xoff,yoff), 'sptype':sptype, 'Av':Av, 
                 'renorm_args':renorm_args}
        self._planets.append(d)
        
        
    def gen_planets_image(self, PA_offset=0):
        """Create image of just planets.
        
        Use info stored in self.planets to create a noiseless slope image 
        of just the exoplanets (no star).
        
        Coordinate convention is for +V3 up and +V2 to left.
        
        Parameters
        ----------        
        PA_offset : float
            Rotate entire scene by some position angle.
            Positive values are counter-clockwise from +Y direction.
            Corresponds to instrument aperture PA.
        """
        if len(self.planets)==0:
            _log.info("No planet info at self.planets")
            return 0
            
        if PA_offset is None: PA_offset=0
            
        image_shape = (self.det_info['ypix'], self.det_info['xpix'])
        image = np.zeros(image_shape)
        for pl in self.planets:
            # Choose the PSF closest to the planet position
            xoff, yoff = pl['xyoff_pix']
            
            # Add in PA offset
            if PA_offset!=0:
                #r, theta = xy_to_rtheta(xoff, yoff)
                #xoff, yoff = rtheta_to_xy(r, theta+PA_offset)
                xoff, yoff = xy_rot(xoff, yoff, PA_offset)
            
            xoff_asec, yoff_asec = np.array(pl['xyoff_pix']) * self.pix_scale
            if len(self.offset_list) > 1:
                if 'WB' in self.mask: # Bar mask
                    roff_asec = np.abs(yoff_asec)
                else: # Circular symmetric
                    roff_asec = np.sqrt(xoff_asec**2 + yoff_asec**2)

                roff_asec = np.sqrt(xoff_asec**2 + yoff_asec**2)
                abs_diff = np.abs(np.array(self.offset_list)-roff_asec)
                ind = np.where(abs_diff == abs_diff.min())[0][0]
            else:
                ind = 0

            # Create slope image (postage stamp) of planet
            if pl.get('sptype') is None:
                sp = self.planet_spec(**pl)
            else:
                sp = stellar_spectrum(pl['sptype'])
            renorm_args = pl['renorm_args']
            if (renorm_args is not None) and (len(renorm_args) > 0):
                sp_norm = sp.renorm(*renorm_args)
                sp_norm.name = sp.name
                sp = sp_norm
            
            psf_planet = self.psf_offsets[ind].gen_psf(sp)
        
            # This is offset according to offset_list
            # First, shift to center
            offset_pix = -self.offset_list[ind] / self.pix_scale
            psf_planet = fshift(psf_planet, dely=offset_pix, pad=True)
        
            # Expand to full size
            psf_planet = pad_or_cut_to_size(psf_planet, image_shape)
        
            # Shift to final position and add to image
            #psf_planet = fshift(psf_planet, delx=xpix-xcen, dely=ypix-ycen, pad=True)
            image += fshift(psf_planet, delx=xoff, dely=yoff, pad=True)
            
        return image
        
    def kill_planets(self):
        """Remove planet info"""
        self._planets = []
    
    
    def gen_disk_image(self, PA_offset=0):
        """Create image of just disk.
        
        Generate a (noiseless) convolved image of the disk at some PA offset. 
        The PA offset value will rotate the image CCW. Units of e-/sec.
        
        Coordinate convention is for +V3 up and +V2 to left.
        
        Parameters
        ----------        
        PA_offset : float
            Rotate entire scene by some position angle.
            Positive values are counter-clockwise from +Y direction.
            Corresponds to instrument aperture PA.
        """
            
        if self.disk_hdulist is None:
            return 0.0

        # Final image shape
        image_shape = ypix, xpix = (self.det_info['ypix'], self.det_info['xpix'])
        
        # The detector-sampled disk image
        disk_image  = self.disk_hdulist[0].data
        header = self.disk_hdulist[0].header
        if PA_offset!=0: 
            disk_image = rotate(disk_image, -PA_offset, reshape=False)
            
        if len(self.offset_list) == 1: # Direct imaging
            psf = self.psf_list[0]
            image_conv = convolve_fft(disk_image, psf, fftn=fftpack.fftn, 
                                      ifftn=fftpack.ifftn, allow_huge=True)
        else:
            noff = len(self.offset_list)

            if 'WB' in self.mask: # Bar mask
                ind1, ind2 = np.indices(image_shape)
                image_rho = np.abs(ind1 - ypix/2)
            else: # Circular symmetric
                image_rho = dist_image(disk_image, pixscale=header['PIXELSCL'])
            worker_arguments = [(psf, disk_image, image_rho, self.offset_list, i) 
                                for i,psf in enumerate(self.psf_list)]
                                
            npix = ypix*xpix
            nproc = nproc_use_convolve(npix, 1, noff)
            if nproc<=1:
                imconv_slices = map(_wrap_convolve_for_mp, worker_arguments)
            else:
                pool = mp.Pool(nproc)
                try:
                    imconv_slices = pool.map(_wrap_convolve_for_mp, worker_arguments)
                except Exception as e:
                    print('Caught an exception during multiprocess:')
                    raise e
                finally:
                    pool.close()
                
            # Turn into a numpy array of shape (noff,nx,ny)
            imconv_slices = np.array(imconv_slices)

            # Sum all images together
            image_conv = imconv_slices.sum(axis=0)
            
        image_conv[image_conv<0] = 0
        return image_conv



    def star_flux(self, fluxunit='counts'):
        """ Stellar flux.
        
        Return the stellar flux in whatever units, such as 
        vegamag, counts, or Jy.
        """

        # Create pysynphot observation
        bp = self.bandpass
        obs = S.Observation(self.sp_sci, bp, binset=bp.wave)
        
        return obs.effstim(fluxunit)


    def calc_contrast(self, hdu_diff=None, roll_angle=10, nsig=1, 
        exclude_disk=True, exclude_planets=True, **kwargs):
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
        
        
        Keyword Args
        ------------
        zfact : float
            Zodiacal background factor (default=2.5)
        exclude_noise : bool
            Don't add random Gaussian noise (detector+photon)?

        Returns
        -------
        tuple
            Three arrays in a tuple: the radius in arcsec, n-sigma contrast,
            and n-sigma magnitude limit (vega mag). 
        """
        from astropy.convolution import convolve, Gaussian1DKernel
        

        # If no HDUList is passed, then create one
        if hdu_diff is None:
            PA1 = 0
            PA2 = None if abs(roll_angle) < eps else roll_angle
            hdu_diff = self.gen_roll_image(PA1=PA1, PA2=PA2, 
                                           exclude_disk=exclude_disk, 
                                           exclude_planets=exclude_planets, **kwargs)
        
    
        # Radial noise
        data = hdu_diff[0].data
        header = hdu_diff[0].header
        rho = dist_image(data, pixscale=header['PIXELSCL'])

        # Get radial profiles
        binsize = header['OVERSAMP'] * header['PIXELSCL']
        bins = np.arange(rho.min(), rho.max() + binsize, binsize)
        igroups, _, rr = hist_indices(rho, bins, True)
        stds = binned_statistic(igroups, data, func=np.std)
        stds = convolve(stds, Gaussian1DKernel(1))

        # Ignore corner regions
        arr_size = np.min(data.shape) * header['PIXELSCL']
        mask = rr < (arr_size/2)
        rr = rr[mask]
        stds = stds[mask]

        # Normalized PSF radial standard deviation
        # Divide out count rate
        stds = stds / self.star_flux()
    
        # Grab the normalized PSF values generated on init
        psf_off_list, psf_max_list = self.psf_max_vals
        psf_off_list.append(rr.max())
        psf_max_list.append(psf_max_list[-1])
        # Interpolate at each radial position
        psf_max = np.interp(rr, psf_off_list, psf_max_list)
        # Normalize and multiply by psf max
        contrast = stds / psf_max
        # Sigma limit
        contrast *= nsig

        # Magnitude sensitivity
        star_mag = self.star_flux('vegamag')
        sen_mag = star_mag - 2.5*np.log10(contrast)

        return (rr, contrast, sen_mag)

    def gen_roll_image(self, PA1=0, PA2=10, zfact=None, oversample=None, 
        exclude_disk=False, exclude_planets=False, exclude_noise=False, 
        opt_diff=True):
        """Make roll-subtracted image.
        
        Create a final roll-subtracted slope image based on current observation
        settings. Coordinate convention is for +V3 up and +V2 to left.
        
        Procedure:
        
        - Create Roll 1 and Roll 2 slope images (star+exoplanets)
        - Create Reference Star slope image
        - Add random Gaussian noise to all images
        - Subtract ref image from both rolls
        - De-rotate Roll 2 by roll_angle amplitude
        - Average Roll 1 and de-rotated Roll 2

        Returns an HDUList of final image (North rotated upwards).

          
        Parameters
        ----------
        PA1 : float
            Position angle of first roll position (clockwise, from East to West)
        PA2 : float, None
            Position angle of second roll position. If set equal to PA1 
            (or to None), then only one roll will be performed.
            Otherwise, two rolls are performed, each using the specified 
            MULTIACCUM settings (doubling the effective exposure time).
        zfact : float
            Zodiacal background factor (default=2.5)
        oversample : float
            Set oversampling of final image.
        exclude_disk : bool
            Ignore disk when subtracted image (for radial contrast),
            but still add Poisson noise from disk.
        exclude_noise : bool
            Don't add random Gaussian noise (detector+photon)
        
        """
    
        # Final image shape
        xpix, ypix = (self.det_info['xpix'], self.det_info['ypix'])
        image_shape = (ypix, xpix)
        # Sub-image for determining ref star scale factor
        subsize = 50
        xsub = np.min([subsize,xpix])
        ysub = np.min([subsize,ypix])
        sub_shape = (ysub, xsub)
        
        # Position angle decisions
        if PA2 is None: 
            roll_angle = 0
        else:
            roll_angle = PA2 - PA1
        if oversample is None: oversample = 1
   
        sci = self
        ref = self.nrc_ref
        
        # Reference star slope simulation
        # Ideal slope
        im_ref = ref.gen_psf(sci.sp_ref, return_oversample=False)
        im_ref = pad_or_cut_to_size(im_ref, image_shape)
        im_ref_sub = pad_or_cut_to_size(im_ref, sub_shape)
        # Noise per pixel
        if not exclude_noise:
            det = ref.Detectors[0]
            fzodi = ref.bg_zodi(zfact)
            im_noise = det.pixel_noise(fsrc=im_ref, fzodi=fzodi)
            # Add random noise
            im_ref += np.random.normal(scale=im_noise)
        
        # Stellar PSF is fixed
        im_star = sci.gen_psf(sci.sp_sci, return_oversample=False)
        im_star = pad_or_cut_to_size(im_star, image_shape)
        
        # Disk and Planet images
        im_disk_r1 = sci.gen_disk_image(PA_offset=PA1)
        im_pl_r1   = sci.gen_planets_image(PA_offset=PA1)

        # Telescope Roll 1
        im_roll1 = im_star + im_disk_r1 + im_pl_r1
        # Noise per pixel
        if not exclude_noise:
            det = sci.Detectors[0]
            fzodi = sci.bg_zodi(zfact)
            im_noise1 = det.pixel_noise(fsrc=im_roll1, fzodi=fzodi)
            # Add random noise
            im_roll1 += np.random.normal(scale=im_noise1)
            
        if exclude_disk:
            im_roll1 -= im_disk_r1
        if exclude_planets:
            im_roll1 -= im_pl_r1
    
        # Subtract reference star from Roll 1
        #im_roll1_sub = pad_or_cut_to_size(im_roll1, sub_shape)
        #scale1 = scale_ref_image(im_roll1_sub, im_ref_sub)
        im_star_sub = pad_or_cut_to_size(im_star+im_pl_r1, sub_shape)
        scale1 = scale_ref_image(im_star_sub, im_ref_sub)
        _log.debug('scale1: {0:.3f}'.format(scale1))
        #scale1 = im_roll1.max() / im_ref.max()
        if oversample != 1:
            im_ref_rebin = frebin(im_ref, scale=oversample)
            im_roll1     = frebin(im_roll1, scale=oversample)
        else:
            im_ref_rebin = im_ref
        
        # Telescope Roll 2
        if abs(roll_angle) > eps:
            # Subtraction with and without scaling
            im_diff1_r1 = im_roll1 - im_ref_rebin
            im_diff2_r1 = im_roll1 - im_ref_rebin * scale1
            #im_diff_r1 = optimal_difference(im_roll1, im_ref_rebin, scale1)

            im_disk_r2 = sci.gen_disk_image(PA_offset=PA2)
            im_pl_r2   = sci.gen_planets_image(PA_offset=PA2)
            im_roll2   = im_star + im_disk_r2 + im_pl_r2
            # Noise per pixel
            if not exclude_noise:
                det = sci.Detectors[0]
                fzodi = sci.bg_zodi(zfact)
                im_noise2 = det.pixel_noise(fsrc=im_roll2, fzodi=fzodi)
                # Add random noise
                im_roll2 += np.random.normal(scale=im_noise2)

            if exclude_disk:
                im_roll2 -= im_disk_r2
            if exclude_planets:
                im_roll2 -= im_pl_r2

            # Subtract reference star from Roll 2
            #im_roll2_sub = pad_or_cut_to_size(im_roll2, sub_shape)
            #scale2 = scale_ref_image(im_roll2_sub, im_ref_sub)
            im_star_sub = pad_or_cut_to_size(im_star+im_pl_r2, sub_shape)
            scale2 = scale_ref_image(im_star_sub, im_ref_sub)
            _log.debug('scale2: {0:.3f}'.format(scale2))
            #scale2 = im_roll2.max() / im_ref.max()
            if oversample != 1:
                im_roll2 = frebin(im_roll2, scale=oversample)
            # Subtraction with and without scaling
            im_diff1_r2 = im_roll2 - im_ref_rebin
            im_diff2_r2 = im_roll2 - im_ref_rebin * scale2
            #im_diff_r2 = optimal_difference(im_roll2, im_ref_rebin, scale2)

            # De-rotate Roll 2 onto Roll 1
            # Convention for rotate() is opposite PA_offset
            im_diff1_r2_rot = rotate(im_diff1_r2, roll_angle, reshape=False, cval=np.nan)
            im_diff2_r2_rot = rotate(im_diff2_r2, roll_angle, reshape=False, cval=np.nan)
            final1 = (im_diff1_r1 + im_diff1_r2_rot) / 2
            final2 = (im_diff2_r1 + im_diff2_r2_rot) / 2
            
            # Replace NaNs with values from im_diff_r1
            nan_mask1 = np.isnan(final1)
            nan_mask2 = np.isnan(final2)
            final1[nan_mask1] = im_diff1_r1[nan_mask1]
            final2[nan_mask2] = im_diff2_r1[nan_mask2]
            
            # final1 has better noise in outer regions (background)
            # final2 has better noise in inner regions (PSF removal)
            if opt_diff:
                rho = dist_image(final1)
                binsize = 1
                bins = np.arange(rho.min(), rho.max() + binsize, binsize)
                igroups = hist_indices(rho, bins)

                std1 = binned_statistic(igroups, final1, func=np.std)
                std2 = binned_statistic(igroups, final2, func=np.std)
                
                ibin_better = np.where(std1 < std2)[0]
                for ibin in ibin_better:
                    final2.ravel()[igroups[ibin]] = final1.ravel()[igroups[ibin]]
                    
            final = final2

        else:
            # Optimal differencing (with scaling only on the inner regions)
            if opt_diff:
                final = optimal_difference(im_roll1, im_ref_rebin, scale1)
            else:
                final = im_roll1 - im_ref_rebin * scale1
            
        # De-rotate PA1 to North
        if abs(PA1) > eps:
            final = rotate(final, PA1, reshape=False)
        
        hdu = fits.PrimaryHDU(final)
        hdu.header['EXTNAME'] = ('ROLL_SUB')
        hdu.header['OVERSAMP'] = oversample
        hdu.header['PIXELSCL'] = sci.pix_scale / hdu.header['OVERSAMP']
        hdulist = fits.HDUList([hdu])

        return hdulist
        
    def saturation_levels(self, full_size=True, ngroup=0, **kwargs):
        """Saturation levels.
        
        Create image showing level of saturation for each pixel.
        Can either show the saturation after one frame (default)
        or after the ramp has finished integrating (ramp_sat=True).
        
        Parameters
        ----------
        full_size : bool
            Expand (or contract) to size of detector array?
            If False, use fov_pix size.
        ngroup : int
            How many group times to determine saturation level?
            The default is ngroup=0, which corresponds to the
            so-called "zero-frame." This is the very first frame
            that is read-out and saved separately. If this number
            is higher than the total groups in ramp, then a
            warning is produced.
        
        """
        
        assert ngroup >= 0

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
    
        # Slope image of input source
        im_star = self.gen_psf(self.sp_sci)
        im_disk = self.gen_disk_image()
        im_pl = self.gen_planets_image()
        image = im_star + im_disk + im_pl
        if full_size:
            shape = (self.det_info['ypix'], self.det_info['xpix'])
            image = pad_or_cut_to_size(image, shape)

        # Well levels after "saturation time"
        sat_level = image * t_sat / self.well_level
    
        return sat_level



def model_to_hdulist(args_model, sp_star, filter, 
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
            - fname   : Name of model file
            - scale0  : Pixel scale (in arcsec/pixel)
            - dist0   : Assumed model distance
            - wave_um : Wavelength of observation
            - units0  : Assumed flux units (ie., MJy/arcsec^2 or muJy/pixel)
    sp_star : :mod:`pysynphot.spectrum`
        A pysynphot spectrum of central star. Used to adjust observed
        photon flux if filter differs from model input
    filter : str
        NIRCam filter used in observation to determine final photon flux.
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
    bp = read_filter(filter, pupil=pupil, mask=mask, module=module)

    # Detector pixel scale and PSF oversample
    #detscale = channel_select(bp)[0]
    #oversample = 4
    #pixscale_over = detscale / oversample
    
    #### Read in the image, then convert from mJy/arcsec^2 to photons/sec/pixel

    # Open file
    hdulist = fits.open(fname)
    #data    = hdulist[0].data#.copy()
    #header  = hdulist[0].header
    #hdutemp.close()

    # Break apart units0
    units_list = units0.split('/')
    if 'Jy' in units_list[0]:
        units_pysyn = S.units.Jy()
    if 'mJy' in units_list[0]:
        units_pysyn = S.units.mJy()
    if 'muJy' in units_list[0]:
        units_pysyn = S.units.muJy()
    if 'nJy' in units_list[0]:
        units_pysyn = S.units.nJy()    
    if 'MJy' in units_list[0]:
        hdulist[0].data *= 1000 # Convert to Jy
        units_pysyn = S.units.Jy()

    # Convert from input units to photlam (photons/sec/cm^2/A/angular size)
    # Compare observed wavelength to image wavelength
    wave_obs = bp.avgwave() # Current bandpass wavelength
    im = units_pysyn.ToPhotlam(wave0, hdulist[0].data)

    # We want to assume scattering is flat in photons/sec/A
    # This means everything scales with stellar continuum
    sp_star.convert('photlam') 
    im *= sp_star.sample(wave_obs) / sp_star.sample(wave0)

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
        worker_arguments = [(inst, image, rho, offset_list, i) for i,inst in enumerate(nrc_star_list)]

    Then create a theadpool:
        pool = mp.Pool(nproc)
        images = pool.map(_wrap_coeff_for_mp, worker_arguments)
        pool.close()
        images = np.array(images)

    For single processing, just use:
        images = map(_wrap_convolve_for_mp, worker_arguments)
        images = np.array(images)

    For multiprocessing:
    """

    psf, model, rho, offset_list, i = args

    noff = len(offset_list)
    if noff==1:
        r1 = 0; r2 = rho.max()+1	
    elif i==0:
        r1 = offset_list[i]
        r2 = (offset_list[i] + offset_list[i+1])/2
    elif i==noff-1:
        r1 = (offset_list[i] + offset_list[i-1])/2
        r2 = rho.max()+1.
    else:
        r1 = (offset_list[i] + offset_list[i-1])/2
        r2 = (offset_list[i] + offset_list[i+1])/2

    #r1 = offset_list[i]
    #r2 = offset_list[i+1] if i<(noff-1) else rho.max()


    ind = (rho>=r1) & (rho<r2)
    im_temp = model.copy()
    im_temp[~ind] = 0

    # Generate psf and convolve with temp image
    #_, psf_over = nrc_object.gen_psf(return_oversample=True)
    #offset_pix = -offset_list[i] / pixscale_over
    #psf_over = fshift(psf_over, dely=offset_pix, pad=True)
    return convolve_fft(im_temp, psf, fftn=fftpack.fftn, ifftn=fftpack.ifftn, allow_huge=True)
