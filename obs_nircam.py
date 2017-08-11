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
    """
    Subclass of the NIRCam instrument class used to observe stars 
    (plus exoplanets and disks) with either a coronagraph or direct imaging.

    Parameters
    ==========
    sp_sci/sp_ref : Pysynphot spectra of science and reference sources
    distance      : Distance in parsecs
    wfe_drift     : WFE drift of OPDs in nm
    offset_list   : For coronagraph, incremental offset positions to build PSFs
                    for accurately determining contrast curves. A default is
                    applied if set to None.
    wind_mode     : 'FULL', 'STRIPE', or 'WINDOW'
    xpix, ypix    : Size of detector readout (assumes subarray).
    oversample    : PSF oversampling (default=2)
    disk_hdu      : A model of the disk in photons/sec. This requires header
                    keywords PIXSCALE (in arcsec/pixel) and DISTANCE (in pc).
    """
    
    def __init__(self, sp_sci, sp_ref, distance, wfe_drift=10, offset_list=None, 
                 wind_mode='WINDOW', xpix=320, ypix=320, oversample=2, 
                 disk_hdu=None, verbose=False, **kwargs):
                 
        if 'FULL'   in wind_mode: xpix = ypix = 2048
        if 'STRIPE' in wind_mode: xpix = 2048

        #super(NIRCam,self).__init__(**kwargs)
        # Not sure if this works for both Python 2 and 3
        NIRCam.__init__(self, wind_mode=wind_mode, xpix=xpix, ypix=ypix, 
                        oversample=2, **kwargs)
        
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
        
        # Create a NIRCam reference class
        # If it already exists, just update OPD info
        try:
            nrc = self.nrc_ref
            nrc.update_psf_coeff(opd=opd)
        except AttributeError:
            if verbose: print("Creating NIRCam reference class...")
            nrc = NIRCam(self.filter, self.pupil, self.mask, module=self.module, \
                         wind_mode=wind_mode, xpix=xpix, ypix=ypix, \
                         fov_pix=fov_pix, oversample=oversample, opd=opd)        
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
        """
        Return the planet spectrum from Spiegel & Burrows (2012) normalized
        to distance of current target.
        
        Parameters:
            Av : Extinction magnitude (assumes Rv=4.0)
            atmo: A string consisting of one of four atmosphere types:
                hy1s = hybrid clouds, solar abundances
                hy3s = hybrid clouds, 3x solar abundances
                cf1s = cloud-free, solar abundances
                cf3s = cloud-free, 3x solar abundances
            mass: Integer number 1 to 15 Jupiter masses.
            age: Age in millions of years (1-1000)
            entropy: Initial entropy (8.0-13.0) in increments of 0.25
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
        rtheta=(0,0), runits='AU', Av=0, renorm_args=None, sptype=None):
        """
        Add exoplanet information that will be used to generate a point
        source image using a spectrum from Spiegel & Burrows (2012).
        Use self.kill_planets() to delete them.
        
        Coordinate convention is for +V3 up and +V2 to left.
        
        Parameters:
            atmo: A string consisting of one of four atmosphere types:
                hy1s = hybrid clouds, solar abundances
                hy3s = hybrid clouds, 3x solar abundances
                cf1s = cloud-free, solar abundances
                cf3s = cloud-free, 3x solar abundances
            mass    : Integer number 1 to 15 Jupiter masses.
            age     : Age in millions of years (1-1000)
            entropy : Initial entropy (8.0-13.0) in increments of 0.25
            
            sptype : Instead of using a exoplanet spectrum, specify a stellar type.
            
            renorm_args : Pysynphot renormalization arguments in case you want
                very specific luminosity in some bandpass
                Includes (value, units, bandpass). 

            rtheta : Radius and position angle relative to star (center).
            runits : What units is radius? Valid values are 'AU', 'asec', or 'pix'.
            Av     : Extinction magnitude (assumes Rv=4.0).
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
                 'entropy':entropy, 'Av':Av, 'renorm_args':renorm_args}
        else:
            d = {'xyoff_pix':(xoff,yoff), 'sptype':sptype, 'Av':Av, 
                 'renorm_args':renorm_args}
        self._planets.append(d)
        
        
    def gen_planets_image(self, PA_offset=0):
        """
        Use info stored in self.planets to create a noiseless slope image 
        of just the exoplanets (no star).
        
        Coordinate convention is for +V3 up and +V2 to left.
        
        PA_offset (float) : Rotate entire scene by some position angle.
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
        self._planets = []
    
    
    def gen_disk_image(self, PA_offset=0):
        """
        Generate a (noiseless) convolved image of the disk at some PA offset. 
        The PA offset value will rotate the image CCW. Units of e-/sec.
        
        Coordinate convention is for +V3 up and +V2 to left.
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
        """
        Return the stellar flux in whatever units, such as 
        vegamag, counts, or Jy.
        """

        # Create pysynphot observation
        bp = self.bandpass
        obs = S.Observation(self.sp_sci, bp, binset=bp.wave)
        
        return obs.effstim(fluxunit)


    def calc_contrast(self, hdu_diff=None, roll_angle=10, nsig=1, 
        exclude_disk=True, exclude_planets=True, **kwargs):
        """
        Generate n-sigma contrast curve for the current observation settings.
        Make sure that MULTIACCUM parameters are set in both the main
        class (self.update_detectors()) as well as the reference target
        class (self.nrc_ref.update_detectors()).
        
        roll_angle : Telescope roll angle (deg) between two observations.
            If set to 0 or None, then only one roll will be performed.
            If value is >0, then two rolls are performed, each using the
            specified MULTIACCUM settings (doubling the effective exposure
            time).
        nsig  : n-sigma contrast curve
        
        
        **kwargs
        ==========
        zfact : Zodiacal background factor (default=2.5)
        exclude_disk  : Ignore disk when generating image
        exclude_noise : Don't add random Gaussian noise (detector+photon)


        Returns 3 arrays in a tuple:
            radius in arcsec
            n-sigma contrast
            n-sigma magnitude limit (vega mags)        
        """
        from astropy.convolution import convolve, Gaussian1DKernel
        

        # If no HDUList is passed, then create one
        if hdu_diff is None:
            PA1 = 0
            PA2 = None if abs(roll_angle) < eps else roll_angle
            hdu_diff = self.gen_roll_image(PA1=PA1, PA2=PA2, 
                                           exclude_disk=exclude_disk, **kwargs)
        
    
        # Radial noise
        data = hdu_diff[0].data
        header = hdu_diff[0].header
        rho = dist_image(data, pixscale=header['PIXELSCL'])

        # Get radial profiles
        binsize = header['OVERSAMP'] * header['PIXELSCL']
        bins = np.arange(rho.min(), rho.max() + binsize, binsize)
        igroups, _, rr = nrc_utils.hist_indices(rho, bins, True)
        stds = nrc_utils.binned_statistic(igroups, data, func=np.std)
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
        """
        Create a final roll-subtracted slope image based on current observation
        settings. Coordinate convention is for +V3 up and +V2 to left.
        
        Procedure:
          - Create Roll 1 and Roll 2 slope images (star+exoplanets)
          - Create Reference Star slope image
          - Add random Gaussian noise to all images
          - Subtract ref image from both rolls
          - De-rotate Roll 2 by roll_angle amplitude
          - Average Roll 1 and de-rotated Roll 2
          
        Parameters
        ==========
        PA1 : Position angle of first roll position (clockwise, from East to West)
        PA2 : Position angle of second roll position (optional)
              If set equal to PA1 (or to None), then only one roll will be performed.
              Otherwise, two rolls are performed, each using the specified 
              MULTIACCUM settings (doubling the effective exposure time).
        zfact      : Zodiacal background factor (default=2.5)
        oversample : Set oversampling of final image.
        
        exclude_disk  : Ignore disk when subtracted image (for radial contrast),
                        but still add Poisson noise from disk.
        exclude_noise : Don't add random Gaussian noise (detector+photon)
        
        Returns an HDUList of final image (North rotated upwards).
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
        """
        Create image showing level of saturation for each pixel.
        Can either show the saturation after one frame (default)
        or after the ramp has finished integrating (ramp_sat=True).
        
        Parameters
        ==========
        full_size : Expand (or contract) to size of detector array?
                    If False, use fov_pix size.
        ngroup    : How many group times to determine saturation level?
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



def model_to_hdulist(args_model, sp_star, filter, pupil=None, mask=None):

    """
    Convert disk model to an HDUList with units of photons/sec/pixel.
    If observed filter is different than input filter, we assume that
    the disk has a flat scattering, meaning it scales with stellar
    continuum. Pixel sizes and distances are left unchanged, and 
    stored in header.
    
    args_model - Arguments describing the necessary model information
        fname   : Name of model file
        scale0  : Pixel scale (in arcsec/pixel)
        dist0   : Assumed model distance
        wave_um : Wavelength of observation
        units0  : Assumed flux units (ie., Jy/arcsec^2, 
    """

    #filt, mask, pupil = args_inst
    fname, scale0, dist0, wave_um, units0 = args_model
    wave0 = wave_um * 1e4
    
    bp = read_filter(filter, pupil=pupil, mask=mask)

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


        

### 
### class nrc_diskobs(NIRCam):
###     """
###     Subclass of the NIRCam instrument class. This subclass is specifically used to
###     create observations of disks using either the coronagraph or direct imaging.
### 
###     Pass an HDUlist that contains the model image, which contains header information
###     on the model's PIXSCALE (arcsec/pixel) and DISTANCE (parsecs). 
### 
###     Parameters
###     ==========
###     offset_list : For coronagraph, incremental offset positions to build PSFs
###         for convolving disk images.
###     xpix, ypix  : Size of detector readout (assumes subarray).
###     """
### 
###     def __init__(self, hdulist, offset_list=None, xpix=320, ypix=320, save_psf=True, **kwargs):
###     
###         # If __init__ is called manually and self.hdulist already exists from
###         # a previous initialization, then do not bother rerunning the parent
###         # class __init__.
###         try:
###             self.hdulist
###         except AttributeError:
###             #super(NIRCam,self).__init__(**kwargs)
###             # Not sure if this works for both Python 2 and 3
###             NIRCam.__init__(self, wind_mode='WINDOW', xpix=xpix, ypix=ypix, save_psf=save_psf, **kwargs)
### 
###         # Model image and header information
###         self.hdulist = hdulist
### 
###         # Offsets positions to build PSFs
###         if self.mask is None:
###             # if no coronagraphic mask, then only 1 PSF
###             self.offset_list = [0.0]
###             if offset_list is not None:
###                 print('No coronagraph, so offset_list automatically set to [0.0].')
###         elif offset_list is None:
###             self.offset_list = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
###         else:
###             self.offset_list = offset_list
###     
###         self.psf_pix_det = int(np.ceil(self.hdu_image.shape[0] / self.image_oversample))
###         # Pixel distances of the oversampled images
###         ycen, xcen = np.where(self.hdu_image == self.hdu_image.max())
###         cen = (ycen[0], xcen[0])
###         self.image_rho = dist_image(self.hdu_image, pixscale=self.image_scale, center=cen)
### 
###         print("Generating list of PSFs...")
###         # This will get skipped if the PSFs have already been generated
###         log_prev = conf.logging_level
###         setup_logging('WARN', verbose=False)
###         self._gen_psfs()
###         setup_logging(log_prev, verbose=False)
### 
###         print("Convolving image slices...")
###         self._convolve_image()
### 
###         print("Making reference PSF image...")
###         self._make_ref_image()
###     
###         # Create the coronagraphic mask element image
###         if self.mask is not None:
###             fov = self.image_scale * self.hdu_image.shape[0]
###             self.image_coron = coron_trans(self.mask, self.module, self.image_scale, fov=fov)
###             #self.image_coron = pad_or_cut_to_size(image_coron, self.hdu_image.shape)
###     
###         print("Finished.")
###     
###     @property
###     def image_scale(self):
###         """Pixel scale of image (usually oversampled)"""
###         return self.hdulist[0].header['PIXELSCL']
###     @property
###     def image_oversample(self):
###         """Image oversampling amount"""
###         return np.int(np.round(self.pix_scale / self.image_scale))
###     @property
###     def image_dist(self):
###         """Object distance (pc)"""
###         return self.hdulist[0].header['DISTANCE']
###     @property
###     def hdu_image(self):
###         """Model image data in hdulist"""
###         return self.hdulist[0].data
###     @property
###     def hdu_header(self):
###         """Header info in hdulist"""
###         return self.hdulist[0].header
### 
###     @property
###     def image_planet(self):
###         """Planet image (if it exists)"""
###         try: im = self._image_planet
###         except: im = 0.0
###         return im
### 
###     def _gen_psfs(self):
###         """
###         Create instances of NIRCam observations that are incrementally offset 
###         from coronagraph center.
### 
###         Generate a list of PSFs (oversampled) for each offset position.
###         """
###     
###         # Create NIRcam objects
###         #self.psf_over_list = []
###         self.psf_over_list = []
###     
###         # if no coronagraphic mask, then only 1 PSF
###         #offset_list = [0.0] if self.mask is None else self.offset_list
###         psf_dir = conf.PYNRC_PATH + 'disk_psfs/'
###     
###         # Generate PSFs from coefficients and shift to center
###         for offset in self.offset_list:		
###             # Generate filename and check if it exists
###             # Saving the padded PSFs is much faster than generating new ones each time
###             f = self.filter
###             m = 'none' if self.mask is None else self.mask
###             p = 'none' if self.pupil is None else self.pupil
###             fov_pix = 2*self.psf_pix_det
###             oversample = self.image_oversample
###             outfile = '{}_{}_{}_{:04.0f}_{:1.0f}_{:4.2f}.npy'.\
###                         format(f,m,p,fov_pix,oversample,offset)
###                     
###             psf_path = psf_dir+outfile
###             if os.path.exists(psf_path):
###                 psf_over = np.load(psf_path)
###             else:
###                 print('  Offset: {} asecs'.format(offset))
###                 nrc_inst = NIRCam(self.filter, mask=self.mask, pupil=self.pupil, \
###                                     fov_pix=fov_pix, oversample=oversample, \
###                                     offset_r=offset)
###                 _, psf_over = nrc_inst.gen_psf(return_oversample=True)
###                 np.save(psf_path, psf_over)
### 
###             # Shift to center
###             offset_pix = -offset / self.image_scale
###             #psf_over = fshift(psf_over, dely=offset_pix, pad=True)            
###             self.psf_over_list.append(fshift(psf_over, dely=offset_pix, pad=True))
### 
###     def _convolve_image(self):
### 
###         noff = len(self.offset_list)
###         if self.mask is None: # Direct imaging
###             worker_arguments = [(self.psf_over_list[0], self.hdu_image, self.image_rho, 
###                                 self.offset_list, i) for i in np.arange(noff)]
###         else: # Coronagraphic imaging
###             worker_arguments = [(psf, self.hdu_image, self.image_rho, self.offset_list, i) 
###                                 for i,psf in enumerate(self.psf_over_list)]
### 
###         nproc = nproc_use_convolve(self.psf_pix_det, self.image_oversample, noff)
###         if nproc<=1:
###             imconv_slices = map(_wrap_convolve_for_mp, worker_arguments)
###         else:
###             pool = mp.Pool(nproc)
###             imconv_slices = pool.map(_wrap_convolve_for_mp, worker_arguments)
###             pool.close()
### 
###         # Turn into a numpy array of shape (noff,nx,ny)
###         imconv_slices = np.array(imconv_slices)
### 
###         # Sum all images together
###         self.image_conv = imconv_slices.sum(axis=0)
### 
###         # Sum images that exclude flux from the inner 5 AU
###         darr = np.array(self.offset_list) * self.image_dist
###         ind = np.where(darr>5)
###         self.image_conv2 = imconv_slices[ind[0],:,:].sum(axis=0)
### 
### 
###     def _make_ref_image(self):
### 
###         psf_ref = self.psf_over_list[0]
###     
###         image = self.hdu_image
###         ycen, xcen = np.where(image == image.max())
###         cen = (ycen[0], xcen[0])
###     
###         mask = (image==image.max())
###         indx,indy = np.where(mask == True)
###         arr = np.arange(-3,3,1)
###         for ix in arr:
###             for iy in arr:
###                 mask[indx[0]+ix,indy[0]+iy] = True
###     
###         im_temp = np.zeros(image.shape)
###         im_temp[mask] = image[mask]
###         self.im_ref = convolve_fft(im_temp, psf_ref, fftn=fftpack.fftn, ifftn=fftpack.ifftn, allow_huge=True)
###     
###         resid = self.image_conv - self.image_conv2
###         scale = np.mean(resid / self.im_ref)
### 
###         self.im_ref *= scale
###     
###     def star_flux(self, fluxunit):
###         """
###         Return the stellar flux in whatever units, such as 
###         vegamag, counts, or Jy.
###         """
### 
###         star_count = self.hdu_image.max()
###         if 'counts' in fluxunit: 
###             return star_count
### 
###         # Create pysynphot observation
###         bp = self.bandpass
###         waveset = bp.wave
###     
###         #sp_flat = S.ArraySpectrum(waveset, 0*waveset + 10.)
###         # Normalize to spectrum to return stellar count rate
###         #sp_norm = sp_flat.renorm(bp.unit_response()*star_count, 'flam', bp)
###         # Bandpass unit response is the flux (in flam) of a star that 
###         # produces a response of one count per second in that bandpass
###         norm = bp.unit_response()*star_count
###         sp_norm = stellar_spectrum('flat', norm, 'flam', bp)
###         # Convert to an observation
###         obs = S.Observation(sp_norm, bp, binset=waveset)
### 
###         return obs.effstim(fluxunit)
###     
###     def planet_flux(self, fluxunit='counts', Av=0, **kwargs):
###         """
###         Return the planet flux rate for spectrum from Spiegel & Burrows (2011). 
###         Parameters:
###             Av : Extinction magnitude (assumes Rv=4.0)
###             atmo: A string consisting of one of four atmosphere types:
###                 hy1s = hybrid clouds, solar abundances
###                 hy3s = hybrid clouds, 3x solar abundances
###                 cf1s = cloud-free, solar abundances
###                 cf3s = cloud-free, 3x solar abundances
###             mass: Integer number 1 to 15 Jupiter masses.
###             age: Age in millions of years (1-1000)
###             entropy: Initial entropy (8.0-13.0) in increments of 0.25
###             base_dir: Location of atmospheric model sub-directories.
###         """
###         # Create planet class and convert to Pysynphot spectrum
###         planet = planets_sb11(distance=self.image_dist, **kwargs)
###         sp = planet.export_pysynphot()
### 
###         # Add extinction from the disk
###         Rv = 4.0
###         sp *= S.Extinction(Av/Rv,name='mwrv4')
###         obs = S.Observation(sp, self.bandpass, binset=self.bandpass.wave)
###     
###         return obs.effstim(fluxunit)
###     
###     
###     def add_planet(self, loc=(100,100), loc_units='AU', Av=0, **kwargs):
###         """
###         Add an exoplanet point source using spectrum from Spiegel & Burrows (2011).
###         Doing this multiple times will and multiple planets.
###         Used self.kill_planets() to delete them.
### 
###         Parameters:
###             loc = (x,y) : Position to place point source relative to center
###             loc_units : What units are loc? Valid are values are 'AU' or 'asec'
###             Av : Extinction magnitude (assumes Rv=4.0)
### 
###         **kwargs:
###             atmo: A string consisting of one of four atmosphere types:
###                 hy1s = hybrid clouds, solar abundances
###                 hy3s = hybrid clouds, 3x solar abundances
###                 cf1s = cloud-free, solar abundances
###                 cf3s = cloud-free, 3x solar abundances
###             mass: Integer number 1 to 15 Jupiter masses.
###             age: Age in millions of years (1-1000)
###             entropy: Initial entropy (8.0-13.0) in increments of 0.25
###             base_dir: Location of atmospheric model sub-directories.
###         """
### 
###         image = self.hdu_image
### 
###         # Create point image at loc position to convolve with PSF
###         au_per_pixel = self.image_dist*self.image_scale
###         if 'AU' in loc_units:
###             xoff, yoff = np.array(loc) / au_per_pixel
###             xoff_asec, yoff_asec = np.array(loc) / self.image_dist
###         elif ('asec' in loc_units) or ('arcsec' in loc_units):
###             xoff, yoff = np.array(loc) / self.image_scale
###             xoff_asec, yoff_asec = np.array(loc)
###         else:
###             _log.warning("Do not recognize loc_units = '{}'. Assuming 'AU'".format(loc_units))
###             xoff, yoff = np.array(loc) / au_per_pixel
###             xoff_asec, yoff_asec = np.array(loc) / self.image_dist
###         
###         ycen, xcen = np.where(self.hdu_image == self.hdu_image.max())
###         xpix, ypix = int(round(xoff+xcen[0])), int(round(yoff+ycen[0]))
###     
###         #x_det, y_det = (self.kwargs['xpix'], self.kwargs['ypix'])
###         # Make sure planet is within image bounds
###         sh_diff = np.abs(np.array([ypix,xpix]))-np.array(image.shape)
###         if np.any(sh_diff>=0):
###             _log.warning('ypix,xpix = {} is beyond image boundaries. Planet not placed.'\
###                          .format((ypix,xpix)))
###             return
###     
###         im_temp = np.zeros(image.shape)
###         # Counts/second to convolve with planet PSF
###         #print(ypix,xpix,obs.countrate())
###         im_temp[ypix,xpix] = self.planet_flux(Av, **kwargs) #obs.countrate()
###     
###         # Planet PSF and convolution (oversampled)
###         # Choose the PSF closest to the planet position
###         if len(self.offset_list) > 1:
###             roff_asec = np.sqrt(xoff_asec**2 + yoff_asec**2)
###             abs_diff = np.abs(np.array(self.offset_list)-roff_asec)
###             ind = np.where(abs_diff == abs_diff.min())[0][0]
###         else:
###             ind = 0
###         psf_planet = self.psf_over_list[ind]
###         self._image_planet = self.image_planet + \
###             convolve_fft(im_temp, psf_planet, fftn=fftpack.fftn, ifftn=fftpack.ifftn, allow_huge=True)
### 
###     def kill_planets(self):
###         """Delete any planet data"""
###         try: del self._image_planet
###         except: print('No planets to delete!')
###     
###     def get_contrast(self, wfe_drift=5, zfact=2.5, cen=None, nsig=1, maglim=False):
###         """
###         Return radially averaged contrast curve of the current observation.
### 
###         Parameters
###         ============
###         wfe_drift : WFE drift (nm) to determine speckle noise contributions (default 5nm)
###         zfact     : Zodiacal flux multiplication factor (default 2.5)
###         cen       : Location of star relative to center position (for BAR MASKs)
###         maglim    : Output the magnitude limits at each radial position if True
###             rather than the relative contrast limits
### 
###         """
### 
###         # Location and flux of star
###         image = self.hdu_image
###         star_flux = image.max()
###     
###         oversample = self.psf_info['oversample']
### 
###         if cen is None:
###             ycen, xcen = np.where(image == image.max())
###             ycen = int(round(ycen[0]/oversample))
###             xcen = int(round(xcen[0]/oversample))
###             cen = (ycen, xcen)
###     
###         # Create HDUList of observations and grab radial stdev
###         hdulist = self.obs_images(wfe_drift=wfe_drift, zfact=zfact)
###         rr, stds = radial_profile(hdulist, ext='NOISE', center=cen)
###         xsize = hdulist['NOISE'].header['PIXELSCL'] * hdulist['NOISE'].data.shape[0] / 2
###         mask = rr<xsize
###         rr = rr[mask]
###         stds = stds[mask]
###     
###         # Normalized PSF radial standard deviation
###         stds = nsig * stds / star_flux 
###     
###         # PSF representing planet PSF
###         offset_list = self.offset_list
###         noff = len(offset_list)
###         if noff > 1:
###             # Array of PSF maxes for each offset position
###             psf_max = []
###             for i in range(noff):
###                 psf_ref = self.psf_over_list[i]
###                 psf_ref = pad_or_cut_to_size(psf_ref, (20,20))
###                 psf_ref = frebin(psf_ref, scale=1/oversample)
###                 psf_max.append(psf_ref.max())
###             # Interpolate PSF max at each pixel radius offset
###             psf_max = np.interp(rr, offset_list, psf_max)
###             contrast = stds / psf_max
###                     
###         else:
###             psf_ref = self.psf_over_list[-1]
###             psf_ref = frebin(psf_ref, scale=1/oversample)
### 
###             # 1-sigma contrast
###             contrast = stds / psf_ref.max()
### 
###         # Sigma limit
###         star_mag = self.star_flux('vegamag')
###         sig_lim = star_mag - 2.5*np.log10(contrast)
###     
###         # Make sure n-sigma limit at furthest radius matches bg_sensitivity
###         # This is really only valid when we've hit the bg limit
###         # Test bg limit by checking if last two elements are equal
###         # Only currently only valid for direct imaging
###         #if self.mask is None:
###         sens_mag, _ = self.sensitivity(units='vegamag', nsig=nsig, ideal_Poisson=True)
###         sig_lim_true = sens_mag['sensitivity']
###         #diff = sig_lim_true - sig_lim.max()
###         #print(diff.min(), sig_lim_true, sig_lim.max())
###     
###         #if diff < 0:
###             #sig_lim += diff
###             #contrast = 10**((star_mag-sig_lim)/2.5)
###     
###         if maglim:
###             return rr, sig_lim
###         else:
###             return rr, contrast
### 
###     
###     def obs_noise(self, zfact=2.5, wfe_drift=5, return_oversample=False):
###         """
###         Calculate the noise per pixel (detector sampled) for the current
###         observational parameters given some zodi level (default=2.5).
###     
###         The noise takes into account subtraction of a reference PSF.
###         Speckle noise is also returned (TBD)
###     
###         Parameters
###         ============
###         zfact     : Zodiacal flux multiplication factor (default 2.5)
###         wfe_drift : WFE drift (nm) to determine speckle noise contributions (default 5nm)
###         """
###     
###         oversample  = self.image_oversample
###         image_total = self.image_conv + self.image_planet
###         im_rebin    = frebin(image_total, scale=1/oversample)
###         imref_rebin = frebin(self.im_ref, scale=1/oversample)
###     
###         fzodi = self.bg_zodi(zfact)
###     
###         det = self.Detectors[0]
###         im_noise1 = det.pixel_noise(fsrc=im_rebin, fzodi=fzodi)
###         #im_noise2 = det.pixel_noise(fsrc=imref_rebin, fzodi=fzodi)
###     
###         # Traditional readout, dark current, and photon noise values
###         im_noise = im_noise1
###         #im_noise = np.sqrt(im_noise1**2 + im_noise2**2)
###     
###         # Speckle noise values
###         speckle_dir = conf.PYNRC_PATH + 'speckle_maps/'
###         if self.mask is None: speckle_dir += 'direct/'
###         else: speckle_dir += self.mask + '/'
###     
###         # Grab the speckle noise map
###         speckle_file = self.filter+'_speckle_noise_1nm.fits'
###         speckle_path =speckle_dir + speckle_file
###         if os.path.exists(speckle_path):
###             speckle_noise = fits.getdata(speckle_path) * self.hdu_image.max() * wfe_drift
###             speckle_noise = pad_or_cut_to_size(speckle_noise, im_noise.shape)
###         else:
###             _log.warning('Speckle noise map not found at {}'.format(speckle_path))
###             speckle_noise = 0.0 #np.zeros_like(im_noise)
### 
###         # Grab the residual speckle map
###         resid_file = self.filter+'_residual_speckles_1nm.fits'
###         resid_path = speckle_dir + resid_file
###         if os.path.exists(resid_path):
###             speckle_resid = fits.getdata(resid_path)[np.random.randint(0,9),:,:]
###             speckle_resid *= self.hdu_image.max() * wfe_drift
###             speckle_resid = pad_or_cut_to_size(speckle_resid, im_noise.shape)
###         else:
###             _log.warning('Speckle noise map not found at {}'.format(resid_path))
###             speckle_resid = 0.0 #np.zeros_like(im_noise)
###     
###         if return_oversample:
###             im_noise = np.sqrt(frebin(im_noise**2, scale=oversample))
###             speckle_noise = np.sqrt(frebin(speckle_noise**2, scale=oversample))
###             speckle_resid = frebin(speckle_resid, scale=oversample)
###     
###     
###         return im_noise, speckle_noise, speckle_resid
### 
###     def obs_images(self, scaled=False, return_oversample=False, verbose=False, **kwargs):
###         """
###         Output HDUlists of the model, convolved images, noise images, etc.
###     
###         kwargs:
###             zfact
###             wfe_drift
###         """
###         oversample  = self.image_oversample
###         pix_scale   = self.pix_scale
###         image_scale = self.image_scale
###         dist_out    = self.image_dist
###     
### 
###         # These images are oversampled
###         immodel   = self.hdu_image.copy()	
###         # Mask out the model's central stellar pixel
###         immodel[immodel==immodel.max()] = 0
### 
###         im_planet = self.image_planet
###         image     = self.image_conv + im_planet  # Image of star+disk+planet
###         imdisk    = self.image_conv2 # Image of only outer disk
###         imref     = self.im_ref      # Reference star
###         imsub     = image - imref
###     
###         # Some other random stuff
###         imcoron = np.zeros_like(image) if self.mask is None else self.image_coron
###         rho = self.image_rho
### 
###         ramp_time = self.multiaccum_times['t_int']
###         pscale = image_scale
###     
###         imtot_rebin = frebin(image, scale=1/oversample)
###         rho_rebin   = frebin(rho,   scale=1/oversample, total=False)
###         imcor_rebin = frebin(imcoron, scale=1/oversample, total=False)
### 
###         if return_oversample:			
###             # Noise values for oversampled images
###             im_noise, speckle_noise, speckle_resid = self.obs_noise(return_oversample=True, **kwargs)
###             imvar = im_noise**2 + speckle_noise**2
###             tot_noise = np.sqrt(imvar)
### 
###             # Final simulated image with added noise
###             tot_noise_rebin = np.sqrt(frebin(im_noise**2, scale=1/oversample))
###             noise_random_rebin = np.random.normal(scale=tot_noise_rebin)
###             noise_random = frebin(noise_random_rebin, scale=oversample)
###             im_final = imsub + noise_random + speckle_resid
###         
###             sat_level = imtot_rebin * ramp_time / self.well_level
###             sat_level = frebin(sat_level, scale=oversample, total=False)
###         else:
###             scale = 1 / oversample
###             immodel   = frebin(immodel,   scale=scale)
###             image     = imtot_rebin
###             imdisk    = frebin(imdisk,    scale=scale)
###             imref     = frebin(imref,     scale=scale)
###             imsub     = frebin(imsub,     scale=scale)
###             rho       = rho_rebin
###             imcoron   = imcor_rebin
###     
###             im_noise, speckle_noise, speckle_resid = self.obs_noise(return_oversample=False, **kwargs)
###             imvar = im_noise**2 + speckle_noise**2
###             tot_noise = np.sqrt(imvar)
### 
###             # Final simulated image with added noise
###             im_final = imsub + np.random.normal(scale=im_noise) + speckle_resid
### 
###             # Add a saturation level
###             sat_level = image * ramp_time / self.well_level
###             pscale = pix_scale
### 
###         # Mask some things internal to 6lambda/D for occulter or 2lambda/D for others
###         res_el = self.bandpass.avgwave()*2.06265e-05/6.5
###         cen_mask = (rho<2*res_el) if self.mask is None else (imcoron<0.75)
### 
###         rho_sqr = rho**2 if scaled else np.ones(immodel.shape)
### 
###         # SNR Image
###         #tot_noise = convolve_fft(tot_noise, Gaussian2DKernel(3), 
###         #	fftn=fftpack.fftn, ifftn=fftpack.ifftn, allow_huge=True)
###         snr = imsub / tot_noise
### 
###         # If saturated, then set SNR=0, im_final=0
###         sat_mask = sat_level > 0.8
###         # Mask everything internal to the largest saturated radius
###         if sat_mask.any(): 
###             sat_mask = (rho <= rho[sat_mask].max())
###         snr[sat_mask] = 0
###         im_final[sat_mask] = 0
### 
###         if verbose:
###             # Outside of occulting mask region
###             cen_mask_rebin = (rho_rebin<2*res_el) if self.mask is None else (imcor_rebin<0.75)
###             mask_good_rebin = ~cen_mask_rebin
###             v1 = imtot_rebin.max(); v2 = imtot_rebin[mask_good_rebin].max()
###             print('Max Flux: {:.1f} [{:.1f}] counts/sec'.format(v1,v2))
### 
###             mask_good = ~cen_mask
###             v1 = sat_level.max(); v2 = sat_level[mask_good].max()
###             print('Well Fill: {:.1f} [{:.1f}] times full well'.format(v1,v2))
###             print('SNR Max: {:.1f}'.format(snr[mask_good].max()))
### 
###         # Add a list of images to an HDUList	
###         im_list1 = [immodel, image, imdisk, imsub, im_final]
###         names1   = ['MODEL', 'CONV', 'DISK', 'REFSUB', 'FINAL']
###         for i,im in enumerate(im_list1):
###             hdu = fits.PrimaryHDU(im * rho_sqr)
###             hdu.header['PIXELSCL'] = pscale
###             hdu.header['EXTNAME']  = (names1[i])
###             if i==0: hdulist = fits.HDUList([hdu])
###             else: hdulist.append(hdu)
###         
###         im_list2 = [snr, tot_noise, speckle_noise, speckle_resid, rho]
###         names2   = ['SNR', 'NOISE', 'SPNOISE', 'SPRESID', 'RHO']
###         for i,im in enumerate(im_list2):
###             hdu = fits.PrimaryHDU(im)
###             hdu.header['PIXELSCL'] = pscale
###             hdu.header['EXTNAME']  = (names2[i])
###             hdulist.append(hdu)
### 
###         sat_mask = np.uint8(sat_mask)
###         cen_mask = np.uint8(cen_mask)
###         im_list3 = [sat_level, sat_mask, cen_mask]
###         names3   = ['SAT_LEVEL', 'SAT_MASK', 'CEN_MASK']
###         for i,im in enumerate(im_list3):
###             hdu = fits.PrimaryHDU(im)
###             hdu.header['PIXELSCL'] = pscale
###             hdu.header['EXTNAME']  = (names3[i])
###             hdulist.append(hdu)
### 
###         if self.mask is not None:
###             hdu = fits.PrimaryHDU(imcoron)
###             hdu.header['PIXELSCL'] = pscale
###             hdu.header['EXTNAME']  = ('CORON')
###             hdulist.append(hdu)
###     
###         for hdu in hdulist:
###             hdu.header['INSTRUME'] = 'NIRCam'
###             hdu.header['FILTER'] = self.filter
### 
###         return hdulist	





### def observe_disk(args_inst, args_model, dist_out=140, subsize=None, 
###              sptype='G2V', star_kmag=None, **kwargs):
### 
###     """
###     Perform observation of a disk model.
### 
###     Parameters
###     ============
###     args_inst  : Argument tuple consisting of (filter, mask, pupil)
###     args_model : Argument tuple defining the model parameters including
###         (filename, pixel scale, distance, wavelength, units)
###         pixel scale is assumed to be arcsec/pixel
###         distance should be pc
###         wavelength should be um
###         units are the model surface brightness (ie., mJy/pixel, Jy/arcsec, etc.)
###     dist_out   : Desired distance to place the model object.
###     sptype     : Spectral type of central source, which is use to determine
###         the scaling of the scattered light component in the observed band.
###         The stellar flux is also scaled accordingly.
###     star_kmag  : K-Band magnitude of central source to scale photon flux in
###         the observed filter bandwidth, otherwise just used the scaled flux.
###         This can be useful if the star is extincted in some way.
###     subsize : Pixel size of the subarray (typically, 160, 320, or 640)
###         This is fixed for coronagraphic imaging
###     """
###     filt, mask, pupil = args_inst
###     fname, scale0, dist0, wave_um, units0 = args_model
###     wave0 = wave_um * 1e4
### 
###     # If coronagraphic imaging, then override window size and position
###     # May want to put this check inside Class
###     bp = read_filter(filt, pupil=pupil, mask=mask)
###     if (mask is not None) and ('MASK' in mask):
###         # If subarray size is specified
###         if subsize is None:
###             subsize = 640 if bp.avgwave()<2.4 else 320
###         if 'LWB' in mask:
###             x0=275; y0=1508
###         elif 'SWB' in mask:
###             x0=171; y0=236
###         elif '430R' in mask:
###             x0=916; y0=1502
###         elif '335R' in mask:
###             x0=1238; y0=1502
###         elif '210R' in mask:
###             x0=392; y0=224
###         # Make sure subarray sizes don't push out of bounds
###         if (y0 + subsize) > 2048: y0 = 2048 - subsize
###         if (x0 + subsize) > 2048: x0 = 2048 - subsize
###     else:
###         x0 = 0; y0 = 0
### 
###     if 'x0' not in kwargs: kwargs['x0']=x0
###     if 'y0' not in kwargs: kwargs['y0']=y0
###     
###     # Default subarray size of 160
###     if subsize is None: subsize = 160
### 
###     # Detector pixel scale and PSF oversample
###     detscale = channel_select(bp)[0]
###     oversample = 2
###     pixscale_over = detscale / oversample
### 
###     #### Read in the image, then convert from mJy/arcsec^2 to photons/sec/pixel
### 
###     # if no FITs file is specified, then we only want a star
###     # with its flux placed in the central pixel
###     if fname is None:
###         im = np.zeros([3,3])
###         im[1,1] = 1.0
###         if star_kmag is None:
###             print('If no file name, then a K-Band stellar magnitude is required.')
###             return
###     else :# Open file
###         hdutemp = fits.open(fname)
###         im = hdutemp[0].data.copy()
###         hdutemp.close()
### 
###     # Break apart units0
###     units_list = units0.split('/')
###     if 'Jy' in units_list[0]:
###         units_pysyn = S.units.Jy()
###     if 'mJy' in units_list[0]:
###         units_pysyn = S.units.mJy()
###     if 'umJy' in units_list[0]:
###         units_pysyn = S.units.umJy()
###     if 'nJy' in units_list[0]:
###         units_pysyn = S.units.nJy()    
### 
###     # Convert from mJy to photlam (photons/sec/cm^2/A/angular size)
###     # Compare observed wavelength to image wavelength
###     wave_obs = bp.avgwave() # Current bandpass wavelength
###     im = units_pysyn.ToPhotlam(wave0, im)
### 
###     # We want to assume scattering is flat in photons/sec/A
###     # This means everything scales with stellar continuum
###     sp = stellar_spectrum(sptype)
###     sp.convert('photlam') 
###     im *= sp.sample(wave_obs) / sp.sample(wave0)
### 
###     # Convert to photons/sec/pixel
###     im *= bp.equivwidth() * S.refs.PRIMARY_AREA
###     # If input units are per arcsec^2 then scale by pixel scale
###     # This will be ph/sec for each oversampled pixel
###     if ('arcsec' in units_list[1]) or ('asec' in units_list[1]):
###         im *= scale0**2
###     elif 'mas' in units_list[1]:
###         im *= (scale0*1000)**2
### 
###     # Place image into an HDUlist
###     hdu = fits.PrimaryHDU(im)
###     hdutemp = fits.HDUList([hdu])
### 
###     # Rescale
###     args_in = (scale0, dist0)
###     args_out = (pixscale_over, dist_out)
###     hdulist = image_rescale(hdutemp, args_in, args_out)
###     hdulist[0].header['UNITS']    = 'photons/sec'
###     hdulist[0].header['DETSAMP']  = detscale
###     hdulist[0].header['OVERSAMP'] = oversample
###     hdulist[0].header['PIXELSCL'] = pixscale_over
###     hdulist[0].header['DISTANCE'] = dist_out
### 
###     # If the stellar magnitude is specified then determine flux
###     # and replace central pixel after scaling by distance
###     if star_kmag is not None:
###         im = hdulist[0].data
###         sp_norm = sp.renorm(star_kmag, 'vegamag', S.ObsBandpass('k'))
###         obs_temp = S.Observation(sp_norm, bp, binset=bp.wave)
###         _log.info('Modifying stellar flux from {:.2e} to {:.2e}'.\
###                   format(im[im==im.max()][0], obs_temp.countrate()))
###         im[im==im.max()] = obs_temp.countrate()
### 
###     hdutemp.close()
### 
###     _log.info('Model Oversample Shape: {}'.format(hdulist[0].data.shape))
###     _log.info('Model Max Flux: {:.2e} {}'.format(hdulist[0].data.max(), hdulist[0].header['UNITS']))
### 
###     # image needs to be expanded (or contracted) to subarray size
###     ypix = xpix = subsize
###     new_shape = (ypix*oversample, xpix*oversample)
###     hdulist[0].data = pad_or_cut_to_size(hdulist[0].data, new_shape)
###     _log.info('FoV Oversample Shape: {}'.format(hdulist[0].data.shape))
### 
###     #im_shape = hdulist[0].data.shape
###     ##det_shape = tuple(s//oversample for s in im_shape)
###     #ypix2, xpix2 = (ypix*oversample, xpix*oversample)
###     #if im_shape[0]<ypix2:
###     #    ypad, xpad = (np.array([ypix2,xpix2]) - np.array(im_shape)) // 2
###     #    pad_vals = ([ypad]*2, [xpad]*2)
###     #    hdulist[0].data = np.pad(hdulist[0].data, pad_vals, 'constant')
###     #elif im_shape[0]>ypix2:
###     #    y0, x0 = (np.array(im_shape) - np.array([ypix2,xpix2])) // 2
###     #    im = hdulist[0].data
###     #    hdulist[0].data = im[y0:y0+ypix2, x0:x0+xpix2]
###     #print('Oversample Shape: {}'.format(hdulist[0].data.shape))
###    
###     # Generate the model disk observation
###     return nrc_diskobs(hdulist, filter=filt, mask=mask, pupil=pupil, oversample=oversample, 
###                         xpix=xpix, ypix=ypix, **kwargs)
###               
### 
### def observe_star(args_inst, dist_out=140, subsize=None, 
###              sptype='G2V', star_kmag=None, **kwargs):
### 
###     """
###     Perform observation of a single star (no disk).
### 
###     Parameters
###     ============
###     args_inst  : Argument tuple consisting of (filter, mask, pupil)
###     dist_out   : Desired distance to place the model object.
###     sptype     : Spectral type of central source, which is use to determine
###         the scaling of the scattered light component in the observed band.
###         The stellar flux is also scaled accordingly.
###     star_kmag  : K-Band magnitude of central source to scale photon flux in
###         the observed filter bandwidth, otherwise just used the scaled flux.
###         This can be useful if the star is extincted in some way.
###     subsize : Pixel size of the subarray (typically, 160, 320, or 640)
###         This is fixed for coronagraphic imaging
### 
###     """
###     filt, mask, pupil = args_inst
###     #fname, scale0, dist0, wave_um, units0 = args_model
###     #wave0 = wave_um * 1e4
### 
###     # If coronagraphic imaging, then override window size and position
###     # May want to put this check inside Class
###     bp = read_filter(filt, pupil=pupil, mask=mask)
###     if (mask is not None) and ('MASK' in mask):
###         # If subarray size is specified
###         if subsize is None:
###             subsize = 640 if bp.avgwave()<2.4 else 320
###         if 'LWB' in mask:
###             x0=275; y0=1508
###         elif 'SWB' in mask:
###             x0=171; y0=236
###         elif '430R' in mask:
###             x0=916; y0=1502
###         elif '335R' in mask:
###             x0=1238; y0=1502
###         elif '210R' in mask:
###             x0=392; y0=224
###         # Make sure subarray sizes don't push out of bounds
###         if (y0 + subsize) > 2048: y0 = 2048 - subsize
###         if (x0 + subsize) > 2048: x0 = 2048 - subsize
###     else:
###         x0 = 0; y0 = 0
### 
###     if 'x0' not in kwargs: kwargs['x0']=x0
###     if 'y0' not in kwargs: kwargs['y0']=y0
###     
###     # Default subarray size of 160
###     if subsize is None: subsize = 160
### 
###     # Detector pixel scale and PSF oversample
###     detscale = channel_select(bp)[0]
###     oversample = 2
###     pixscale_over = detscale / oversample
### 
###     #### Read in the image, then convert from mJy/arcsec^2 to photons/sec/pixel
### 
###     # if no FITs file is specified, then we only want a star
###     # with its flux placed in the central pixel
###     im = np.zeros([3,3])
###     im[1,1] = 1.0
### 
###     # Place image into an HDUlist
###     hdu = fits.PrimaryHDU(im)
###     hdulist = fits.HDUList([hdu])
### 
###     # Rescale
###     hdulist[0].header['UNITS']    = 'photons/sec'
###     hdulist[0].header['DETSAMP']  = detscale
###     hdulist[0].header['OVERSAMP'] = oversample
###     hdulist[0].header['PIXELSCL'] = pixscale_over
###     hdulist[0].header['DISTANCE'] = dist_out
### 
###     im = hdulist[0].data
### 
###     sp = stellar_spectrum(sptype)
###     sp_norm = sp.renorm(star_kmag, 'vegamag', S.ObsBandpass('k'))
###     obs_temp = S.Observation(sp_norm, bp, binset=bp.wave)
###     _log.info('Modifying stellar flux from {:.2e} to {:.2e}'.\
###               format(im[im==im.max()][0], obs_temp.countrate()))
###     im[im==im.max()] = obs_temp.countrate()
### 
###     _log.info('Model Oversample Shape: {}'.format(hdulist[0].data.shape))
###     _log.info('Model Max Flux: {:.2e} {}'.format(hdulist[0].data.max(), hdulist[0].header['UNITS']))
### 
###     # image needs to be expanded (or contracted) to subarray size
###     ypix = xpix = subsize
###     new_shape = (ypix*oversample, xpix*oversample)
###     hdulist[0].data = pad_or_cut_to_size(hdulist[0].data, new_shape)
###     _log.info('FoV Oversample Shape: {}'.format(hdulist[0].data.shape))
### 
###     # Generate the model disk observation
###     return nrc_diskobs(hdulist, filter=filt, mask=mask, pupil=pupil, oversample=oversample, 
###                         xpix=xpix, ypix=ypix, **kwargs)


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
