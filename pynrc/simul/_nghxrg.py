"""
NGHXRG - Teledyne HxRG Noise Generator

**DEPRECATED FOR ngNRC.py**

Modification History:

8-15 April 2015, B.J. Rauscher, NASA/GSFC

- Implement Pierre Ferruit's (ESA/ESTEC)recommendation to use
  numpy.fft.rfft and numpy.fft.irfft for faster execution. This saves
  about 30% in execution time.
- Clean up setting default values per several suggestions from
  Chaz Shapiro (NASA/JPL).
- Change from setting the shell variable H2RG_PCA0 to point to the PCA-zero
  file to having a shell variable point to the NG home directory. This is per
  a suggestion from Chaz Shapiro (NASA/JPL) that would allow seamlessly adding  
  more PCA-zero templates.
- Implement a request form Chaz Shapiro (NASA/JPL) for status
  reporting. This was done by adding the "verbose" arguement.
- Implement a request from Pierre Ferruit (ESA/ESTEC) to generate
  3-dimensional data cubes.
- Implement a request from Pierre Ferruit to treat ACN as different 1/f
  noise in even/odd columns. Previously ACN was treated purely as a feature
  in Fourier space.
- Version 2(Beta)

16 April 2015, B.J. Rauscher

- Fixed a bug in the pinkening filter definitions. Abs() was used where
  sqrt() was intended. The bug caused power spectra to have the wrong shape at
  low frequency.
- Version 2.1(Beta)

17 April 2015, B.J. Rauscher

- Implement a request from Chaz Shapiro for HXRGNoise() to exit gracefully if
  the bias_file is not found.
- Version 2.2 (Beta)

8 July 2015, B.J. Rauscher

- Address PASP referee comments
    * Fast scan direction is now reversible. To reverse the slow scan
      direction use the numpy flipud() function.
    * Modifications to support subarrays. Specifically,
        > Setting reference_pixel_border_width=0 (integer zero);
            + (1) eliminates the reference pixel border and
            + (2) turns off adding in a bias pattern
                  when simulating data cubes. (Turned on in v2.5)
- Version 2.4

12 Oct 2015, J.M. Leisenring, UA/Steward

- Make compatible with Python 2.x
    * from __future__ import division
- Allow pca0 & kTC noise to show up for a single frame
- Included options for subarray modes (FULL, WINDOW, and STRIPE)
    * Keywords x0 and y0 define a subarray position (lower left corner)
    * Selects correct pca0 region for subarray underlay
    * Adds reference pixels if they exist within subarray window
- Tie negative values to 0 and anything >=2^16 to 2^16-1
- Version 2.5

20 Oct 2015, J.M. Leisenring, UA/Steward

- Padded nstep to the next power of 2 in order to improve FFT runtime
    * nstep2 = int(2**np.ceil(np.log2(nstep)))
    * Speeds up FFT calculations by ~5x
- Don't generate noise elements if their magnitudes are equal to 0.
- Returns a copy of final HDU result for easy retrieval
- Version 2.6

17 Nov 2015, J.M. Leisenring, UA/Steward

- Read in bias_file rather than pca0_file
    * Calculate pca0 element by normalizing bias file
- Include alternating column offsets (aco_a & aco_b)
    * These can be defined as a list or np array
- Version 2.7

15 Feb 2016, J.M. Leisenring, UA/Steward

- Add a reference instability
    * ref pixels don't track active pixels perfectly
- Version 2.8

9 May 2016, J.M. Leisenring, UA/Steward

- Each channel can have it's own read noise value
- Version 2.9

21 July 2016, J.M. Leisenring, UA/Steward

- Add option to use FFTW
    * This can be faster for some processors/OS architectures
    * The decision to use this is highly dependent on the computer
      and should be tested beforehand.
    * For more info: https://pypi.python.org/pypi/pyFFTW
- Version 3.0

"""
# Necessary for Python 2.6 and later
#from __future__ import division, print_function
from __future__ import absolute_import, division, print_function, unicode_literals

__version__ = "3.0"

import os
import warnings
from astropy.io import fits
import numpy as np
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import convolve
from astropy.stats.funcs import median_absolute_deviation as mad
import datetime
# import matplotlib.pyplot as plt # Handy for debugging

try:
    import pyfftw
    pyfftw_available = True
except ImportError:
    pyfftw_available = False
import multiprocessing as mp
#import time


warnings.filterwarnings('ignore')
import logging
_log = logging.getLogger('nghxrg')

class HXRGNoise:
    """Simulate Teledyne HxRG + SIDECAR ASIC noise
    
    HXRGNoise is a class for making realistic Teledyne HxRG system
    noise. The noise model includes correlated, uncorrelated,
    stationary, and non-stationary components. The default parameters
    make noise that resembles Channel 1 of JWST NIRSpec. NIRSpec uses
    H2RG detectors. They are read out using four video outputs at
    1.e+5 pix/s/output.
    
    Parameters
    ----------
    naxis1 : int
        X-dimension of the FITS cube.
    naxis2 : int
        Y-dimension of the FITS cube.
    naxis3 : int
        Z-dimension of the FITS cube (number of up-the-ramp samples).
    n_out : int
        Number of detector amplifiers/channels/outputs.
    nroh : int
        New row overhead in pixels. This allows for a short
        wait at the end of a row before starting the next one.
    nfoh : int
        New frame overhead in rows. This allows for a short
        wait at the end of a frame before starting the next one.
    nfoh_pix(TBD) : int
        New frame overhead in pixels. This allows for a short
        wait at the end of a frame before starting the next one.
        Generally a single pix offset for full frame and stripe
        for JWST ASIC systems.
    dt : float
        Pixel dwell time in seconds (10e-6 sec, for instance).
    bias_file : str
        Name of a FITS file that contains bias pattern, also used for PCA-zero.
    dark_file : str
        Name of a FITS file that contains dark current values per pixel.
    verbose : bool
        Enable this to provide status reporting.
    wind_mode : str
        'FULL', 'STRIPE', or 'WINDOW'.
    x0/y0 : int
        Pixel positions of subarray mode.
    det_size : int
        Pixel dimension of full detector (square).
    reference_pixel_border_width : int
        Width of reference pixel border around image area.
    same_scan_direction : bool
        Are all the output channels read in the same direction?
        By default fast-scan readout direction is ``[-->,<--,-->,<--]``
        If ``same_scan_direction``, then all ``-->``
    reverse_scan_direction : bool
        Enable this to reverse the fast scanner readout directions. 
        This capability was added to support Teledyne's programmable 
        fast scan readout directions. The default setting of False 
        corresponds to what HxRG detectors default to upon power up.
        If ``reverse_scan_direction``, then ``[<--,-->,<--,-->]`` or all ``<--``
    use_fftw : bool
        If pyFFTW is installed, you can use this in place of np.fft.
    ncores : int
        Specify number of cores (threads, actually) to use for pyFFTW.
    
    """

    # These class variables are common to all HxRG detectors
    nghxrg_version = float(__version__) # Sofware version

    def __init__(self, naxis1=None, naxis2=None, naxis3=None, n_out=None,
                 dt=None, nroh=None, nfoh=None, nfoh_pix=None,
                 dark_file=None, bias_file=None, verbose=False,
                 reverse_scan_direction=False, same_scan_direction=False,
                 reference_pixel_border_width=None,
                 wind_mode='FULL', x0=0, y0=0, det_size=None, 
                 use_fftw=False, ncores=None):

        # pyFFTW usage
        self.use_fftw = True if (use_fftw and pyfftw_available) else False
        # By default, use 50% of available cores for FFTW parallelization
        self.ncores = mp.cpu_count() // 2 if ncores is None else int(ncores)	

        # ======================================================================
        #
        # DEFAULT CLOCKING PARAMETERS
        #
        # The following parameters define the default HxRG clocking pattern. The
        # parameters that define the default noise model are defined in the
        # mknoise() method.
        #
        # ======================================================================

        # Subarray?
        if wind_mode is None:
            wind_mode = 'FULL'
        if det_size is None:
            det_size = 2048
        wind_mode = wind_mode.upper()
        modes = ['FULL', 'STRIPE', 'WINDOW']
        if wind_mode not in modes:
            _log.warning(f'{wind_mode} not a valid window readout mode! Returning...')
            os.sys.exit()
        if wind_mode == 'WINDOW':
            n_out = 1
        if wind_mode == 'FULL':
            x0 = 0; y0 = 0
        if wind_mode == 'STRIPE':
            x0 = 0

        # Default clocking pattern is JWST NIRSpec
        self.naxis1    = 2048  if naxis1   is None else int(naxis1)
        self.naxis2    = 2048  if naxis2   is None else int(naxis2)
        self.naxis3    = 1     if naxis3   is None else int(naxis3)
        self.n_out     = 4     if n_out    is None else int(n_out)
        self.dt        = 10e-6 if dt       is None else dt
        self.nroh      = 12    if nroh     is None else int(nroh)
        self.nfoh      = 1     if nfoh     is None else int(nfoh)
        self.nfoh_pix  = 0     #if nfoh_pix is None else int(nfoh_pix)
        self.reference_pixel_border_width = 4 if reference_pixel_border_width is None \
                                              else int(reference_pixel_border_width)
                                  
        # Check that det_size is greater than self.naxis1 and self.naxis2 in WINDOW mode (JML)
        if wind_mode == 'WINDOW':
            if (self.naxis1 > det_size):
                _log.warning('NAXIS1 %s greater than det_size %s! Returning...' % (self.naxis1,det_size))
                os.sys.exit()
            if (self.naxis2 > det_size):
                _log.warning('NAXIS2 %s greater than det_size %s! Returning...' % (self.naxis1,det_size))
                os.sys.exit()

        # Initialize PCA-zero file and make sure that it exists and is a file
        #self.bias_file = os.getenv('NGHXRG_HOME')+'/sca_images/nirspec_pca0.fits' if \
        #				 bias_file is None else bias_file
        #self.bias_file = 'nirspec_pca0.fits' if bias_file is None else bias_file
        self.bias_file = bias_file
        if bias_file is not None:
            if os.path.isfile(self.bias_file) is False:
                raise ValueError('There was an error finding bias_file {}'.format(bias_file))
                # print('There was an error finding bias_file!')
                # print(bias_file)
                #os.sys.exit()

#             print('There was an error finding bias_file! Check to be')
#             print('sure that the NGHXRG_HOME shell environment')
#             print('variable is set correctly and that the')
#             print('$NGHXRG_HOME/ directory contains the desired PCA0')
#             print('file. The default is nirspec_pca0.fits.')
#             os.sys.exit()


        # Add in dark current file (JML)
        self.dark_file = dark_file
        if dark_file is not None:
            if os.path.isfile(self.dark_file) is False:
                raise ValueError('There was an error finding dark_file {}'.format(dark_file))
                #print('There was an error finding dark_file!')
                #print(dark_file)
                #os.sys.exit()


        # ======================================================================

        # Configure Subarray
        self.wind_mode = wind_mode
        self.det_size  = det_size
        self.x0 = x0
        self.y0 = y0

        # Configure status reporting
        self.verbose = verbose

        # Configure readout direction
        self.reverse_scan_direction = reverse_scan_direction
        self.same_scan_direction = same_scan_direction


        # Compute the number of pixels in the fast-scan direction per output
        self.xsize = self.naxis1 // self.n_out
    
        # Compute the number of time steps per integration, per output
        self.nstep_frame = (self.xsize+self.nroh) * (self.naxis2+self.nfoh) + self.nfoh_pix
        self.nstep =  self.nstep_frame * self.naxis3
        # Pad nsteps to a power of 2, which is much faster
        self.nstep2 = int(2**np.ceil(np.log2(self.nstep)))

        # Compute frame time and ramp time
        self.tframe = self.nstep_frame * self.dt
        self.inttime = self.tframe * self.naxis3

        # For adding in ACN, it is handy to have masks of the even
        # and odd pixels on one output neglecting any gaps
        self.m_even = np.zeros((self.naxis3,self.naxis2,self.xsize))
        self.m_odd = np.zeros_like(self.m_even)
        for x in np.arange(0,self.xsize,2):
            self.m_even[:,:self.naxis2,x] = 1
            self.m_odd[:,:self.naxis2,x+1] = 1
        self.m_even = np.reshape(self.m_even, np.size(self.m_even))
        self.m_odd = np.reshape(self.m_odd, np.size(self.m_odd))

        # Also for adding in ACN, we need a mask that point to just
        # the real pixels in ordered vectors of just the even or odd
        # pixels
        self.m_short = np.zeros((self.naxis3, self.naxis2+self.nfoh, \
                                      (self.xsize+self.nroh)//2))
        self.m_short[:,:self.naxis2,:self.xsize//2] = 1
        self.m_short = np.reshape(self.m_short, np.size(self.m_short))

        # Define frequency arrays       
        self.f1 = np.fft.rfftfreq(self.nstep2) # Frequencies for nstep elements
        self.f2 = np.fft.rfftfreq(2*self.nstep2) # ... for 2*nstep elements
        self.f3 = np.fft.rfftfreq(2*self.naxis3)

        # First element should not be 0
        self.f1[0] = self.f1[1]
        self.f2[0] = self.f2[1]
        self.f3[0] = self.f3[1]

        # Define pinkening filters. F1 and p_filter1 are used to
        # generate ACN. F2 and p_filter2 are used to generate 1/f noise.
        # F2 and p_filter2 are used to generate reference instabilities.
        self.alpha = -1 # Hard code for 1/f noise until proven otherwise
        self.p_filter1 = np.sqrt(self.f1**self.alpha)
        self.p_filter2 = np.sqrt(self.f2**self.alpha)
        self.p_filter3 = np.sqrt(self.f3**self.alpha)
        self.p_filter1[0] = 0.
        self.p_filter2[0] = 0.
        self.p_filter3[0] = 0.


        # Initialize pca0. This includes scaling to the correct size,
        # zero offsetting, and renormalization. We use robust statistics
        # because pca0 is real data
        if self.bias_file is None:
            h = fits.PrimaryHDU(np.zeros([det_size, det_size]))
            hdu = fits.HDUList([h])
        else:
            hdu = fits.open(self.bias_file)
        nx_pca0 = hdu[0].header['naxis1']
        ny_pca0 = hdu[0].header['naxis2']
        data = hdu[0].data     

        # Make sure the real PCA image is correctly scaled to size of fake data (JML)
        # Depends if we're FULL, STRIPE, or WINDOW
        if wind_mode == 'FULL':
            scale1 = self.naxis1 / nx_pca0
            scale2 = self.naxis2 / ny_pca0
            zoom_factor = np.max([scale1, scale2])
        if wind_mode == 'STRIPE':
            zoom_factor = self.naxis1 / nx_pca0
        if wind_mode == 'WINDOW':
            # Scale based on det_size
            scale1 = self.det_size / nx_pca0
            scale2 = self.det_size / ny_pca0
            zoom_factor = np.max([scale1, scale2])

        # Resize PCA0 data
        #print(zoom_factor)
        if zoom_factor != 1:
            data = zoom(data, zoom_factor, order=1, mode='wrap')
    
        # Copy data to save as bias pattern
        bias_image = data.copy()

        # Renormalize for PCA0 noise stuff
        data -= np.median(data) # Zero offset
        data /= (1.4826 * mad(data)) # Renormalize

        # Select region of pca0 associated with window position
        if self.wind_mode == 'WINDOW':
            x1 = self.x0; y1 = self.y0
        elif self.wind_mode == 'STRIPE':
            x1 = 0; y1 = self.y0
        else:
            x1 = 0; y1 = 0

        x2 = x1 + self.naxis1
        y2 = y1 + self.naxis2
        # Make sure x2 and y2 are valid
        if (x2 > data.shape[0] or y2 > data.shape[1]):
            _log.warning('Specified window size does not fit within detector array!')
            _log.warning('X indices: [%s,%s]; Y indices: [%s,%s]; XY Size: [%s, %s]' % 
                        (x1,x2,y1,y2,data.shape[0],data.shape[1]))
            os.sys.exit()
    
        # Save as properties
        self.pca0 = data[y1:y2,x1:x2]
        self.bias_image = bias_image[y1:y2,x1:x2]

        # Open dark current file (ADU/sec/pixel)
        if self.dark_file is not None:
            dark_hdu = fits.open(self.dark_file)
            dark_image = dark_hdu[0].data
            self.dark_image = dark_image[y1:y2,x1:x2]
            # Dark current distributions are very wide because of uncertainties
            # This causes certain pixels to fall below 0.
            # We can assume all pixels within 5-sigma have the same dark current
            #   as well as those with negative values.
            # Those with large dark currents are likely real.
            sig = 1.4826 * mad(dark_image)
            med = np.median(dark_image)
            l1 = med - 5*sig; l2 = med + 5*sig
            self.dark_image[(self.dark_image > l1) & (self.dark_image < l2)] = med
            # Set negative values to median
            self.dark_image[self.dark_image<0] = med
    
            # Set negative values to median
            #self.dark_image[self.dark_image<0] = np.median(self.dark_image)
            #self.dark_image[self.dark_image<0.005] = 0.001
        else:
            self.dark_image = None

        # How many reference pixels on each border?
        w = self.reference_pixel_border_width # Easier to work with
        lower = w-y1; upper = w-(det_size-y2)
        left = w-x1; right = w-(det_size-x2)
        ref_all = np.array([lower,upper,left,right])
        ref_all[ref_all<0] = 0
        self.ref_all = ref_all


    def message(self, message_text):
        """Used for status reporting"""
        if self.verbose is True:
            print('NG: ' + message_text + ' at DATETIME = ', (datetime.datetime.now().time()))

    def white_noise(self, nstep=None):
        """Gaussian noise
        
        Generate white noise for an HxRG including all time steps
        (actual pixels and overheads).

        Parameters
        ----------
        nstep : int
            Length of vector returned
        """
        return(np.random.standard_normal(nstep))    

    def pink_noise(self, mode, fmin=None):
        """Generate a vector of non-periodic pink noise.

        Parameters
        ----------
        mode : str
            Selected from 'pink', 'acn', or 'ref_inst'.
        fmin : float, optional
            Low-frequency cutoff. A value of 0 means no cut-off.
        """

        # Configure depending on mode setting
        if 'pink' in mode:
            nstep  = 2*self.nstep
            nstep2 = 2*self.nstep2
            f = self.f2
            p_filter = self.p_filter2
        elif 'acn' in mode:
            nstep  = self.nstep
            nstep2 = self.nstep2
            f = self.f1
            p_filter = self.p_filter1
        elif 'ref_inst' in mode:
            nstep  = 2*self.naxis3
            nstep2 = 2*self.naxis3
            f = self.f3
            p_filter = self.p_filter3

        # Build scaling factors for all frequencies
        fmin = 1./nstep2 if fmin is None else np.max([fmin, 1./nstep2])
        ix  = np.sum(f < fmin)   # Index of the cutoff
        if ix > 1 and ix < len(f):
            f = f.copy()
            p_filter = p_filter.copy()
            f[:ix] = f[ix]
            p_filter[:ix] = p_filter[ix]

        # Calculate theoretical output standard deviation from scaling
        w = p_filter[1:-1]
        w_last = p_filter[-1] * (1 + (nstep2 % 2)) / 2. # correct f = +-0.5
        the_std = 2 * np.sqrt(np.sum(w**2) + w_last**2) / nstep2

        # Generate scaled random power + phase
        sr = np.random.normal(scale=p_filter)
        si = np.random.normal(scale=p_filter)

        # If the signal length is even, frequencies +/- 0.5 are equal
        # so the coefficient must be real.
        if (nstep2 % 2) == 0: 
            si[-1] = 0

        # Regardless of signal length, the DC component must be real
        si[0] = 0

        # Combine power + corrected phase to Fourier components
        thefft  = sr + 1J * si

        #p0 = time.time()
        # Apply the pinkening filter.
        if self.use_fftw:
            result = pyfftw.interfaces.numpy_fft.irfft(thefft, overwrite_input=True,\
                planner_effort='FFTW_ESTIMATE', threads=self.ncores)
        else:
            result = np.fft.irfft(thefft)

        #p1 = time.time()
        #print("FFT and IFFT took",p1-p0," seconds")

        # Keep 1st half of nstep and scale to unit variance
        result = result[:nstep//2] / the_std
  
        return(result)



    def mknoise(self, o_file=None, gain=None,
                rd_noise=None, c_pink=None, u_pink=None, 
                acn=None, aco_a=None, aco_b=None, pca0_amp=None,
                reference_pixel_noise_ratio=None, ktc_noise=None,
                bias_off_avg=None, bias_off_sig=None, bias_amp=None,
                ch_off=None, ref_f2f_corr=None, ref_f2f_ucorr=None, ref_inst=None,
                out_ADU=True):
        """Create FITS cube containing only noise

        Parameters
        ----------
        o_file : str, None
            Output filename. If None, then no output.
        gain : float
            Gain in e/ADU. Defaults to 1.0.
        ktc_noise : float
            kTC noise in electrons. Set this equal to 
            sqrt(k*T*C_pixel)/q_e, where k is Boltzmann's constant, 
            T is detector temperature, and C_pixel is pixel capacitance. 
            For an H2RG, the pixel capacitance is typically about 40 fF.
        rd_noise : float
            Standard deviation of read noise in electrons. 
            Can be an array for individual amplifiers. 
        c_pink :float
            Standard deviation of correlated pink noise in electrons.
        u_pink : float
            Standard deviation of uncorrelated pink noise in electrons.
            Can be an array for individual amplifiers. 
        acn : float
            Standard deviation of alterating column noise in electrons
        pca0_amp : float
            Standard deviation of pca0 in electrons
        reference_pixel_noise_ratio : float
            Ratio of the standard deviation of the reference pixels to 
            the science pixels. Reference pixels are usually a little 
            lower noise.                                          
        bias_off_avg : float
            On average, integrations start here in electrons. 
            Set this so that all pixels are in range.
        bias_off_sig : float
            bias_off_avg has some variation. This is its std dev.
        bias_amp : float
            A multiplicative factor that we multiply bias_image by
            to simulate a bias pattern. This is completely
            independent from adding in "picture frame" noise. Set to
            0.0 remove bias pattern. For NIRCam, default is 1.0.
        ch_off : float
            Offset of each channel relative to bias_off_avg.
            Can be an array for individual amplifiers. 
        ref_f2f_corr : float
            Random frame-to-frame reference offsets due to PA reset,
            correlated between channels.
        ref_f2f_ucorr : float
            Random frame-to-frame reference offsets due to PA reset,
            per channel. Can be an array for individual amplifiers.
        aco_a : float
            Relative offsets of altnernating columns "a".
            Can be an array for individual amplifiers.
        aco_b : float
            Relative offsets of altnernating columns "b".
            Can be an array for individual amplifiers.
        ref_inst : float
            Reference instability relative to active pixels.
        out_ADU : bool
            Return as converted to ADU (True) or raw electrons?
              
        Notes
        -----
        Because of the noise correlations, there is no simple way to
        predict the noise of the simulated images. However, to a
        crude first approximation, these components add in
        quadrature.

        The units in the above are mostly "electrons". This follows convention
        in the astronomical community. From a physics perspective, holes are
        actually the physical entity that is collected in Teledyne's p-on-n
        (p-type implants in n-type bulk) HgCdTe architecture.
        """

        self.message('Starting mknoise()')

        # ======================================================================
        #
        # DEFAULT NOISE PARAMETERS
        #
        # These defaults create noise similar to that seen in the JWST NIRSpec.
        #
        # ======================================================================

        #self.gain      = 1.0      if gain         is None else gain
        #self.rd_noise  = 5.2      if rd_noise     is None else rd_noise
        #self.c_pink    = 3.0      if c_pink       is None else c_pink
        #self.u_pink    = 1.0      if u_pink       is None else u_pink
        #self.acn       = 0.5      if acn          is None else acn
        #self.pca0_amp  = 0.2      if pca0_amp     is None else pca0_amp

        self.gain     = 1.0  if gain     is None else gain
        # Set certain values equal to None if they are set to 0.
        #self.rd_noise = None if rd_noise == 0.0  else rd_noise
        self.rd_noise = rd_noise
        self.c_pink   = None if c_pink   == 0.0  else c_pink
        self.u_pink   = u_pink
        self.acn      = None if acn      == 0.0  else acn
        self.pca0_amp = None if pca0_amp == 0.0  else pca0_amp


        # Change this only if you know that your detector is different from a
        # typical H2RG.
        self.reference_pixel_noise_ratio = 0.8 if \
            reference_pixel_noise_ratio is None else reference_pixel_noise_ratio

        # These are used only when generating cubes. They are
        # completely removed when the data are calibrated to
        # correlated double sampling or slope images. We include
        # them in here to make more realistic looking raw cubes.
        self.ktc_noise     = 29.    if ktc_noise     is None else ktc_noise 
        self.bias_off_avg  = 5000.  if bias_off_avg  is None else bias_off_avg
        self.bias_off_sig  = 0.     if bias_off_sig  is None else bias_off_sig
        self.bias_amp      = 1.     if bias_amp      is None else bias_amp
        self.ch_off        = 0.     if ch_off        is None else ch_off
        self.aco_a         = 0.     if aco_a         is None else aco_a
        self.aco_b         = 0.     if aco_b         is None else aco_b
        self.ref_f2f_corr  = None   if ref_f2f_corr  is None else ref_f2f_corr
        self.ref_f2f_ucorr = None   if ref_f2f_ucorr is None else ref_f2f_ucorr
        self.ref_inst      = None   if ref_inst      is None else ref_inst

        # ======================================================================

        # Initialize the result cube and add a bias pattern.
        self.message('Initializing results cube')
        result = np.zeros((self.naxis3, self.naxis2, self.naxis1), dtype=np.float32)
                  
        # Inject a bias pattern.
        bias_pattern = self.bias_image*self.bias_amp

        # Add overall bias offset plus random component
        bias_pattern += self.bias_off_avg + self.bias_off_sig * np.random.randn()	
            
        # Add in some kTC noise. Since this should always come out
        # in calibration, we do not attempt to model it in detail.
        if self.ktc_noise > 0:
            bias_pattern += self.ktc_noise * np.random.standard_normal((self.naxis2, self.naxis1))

        # Add pedestal offset to each output channel
        # Check if self.ch_off is a numpy array or list
        if isinstance(self.ch_off, (np.ndarray,list)):
            temp = np.asarray(self.ch_off)
            if temp.size != self.n_out:
                _log.warning('Number of elements in ch_off not equal to n_out')
                os.sys.exit()
            for ch in range(self.n_out):
                bias_pattern[:,self.xsize*ch:self.xsize*(ch+1)] += temp[ch]
        else:
            bias_pattern += self.ch_off

        # Add in alternating column offsets to bias pattern
        inda = np.arange(0,self.xsize,2)
        indb = inda+1		
        # Check if self.aco_a/b are numpy arrays or lists
        if isinstance(self.aco_a, (np.ndarray,list)):
            temp = np.asarray(self.aco_a) # Always set to a numpy array
            if temp.size != self.n_out:
                _log.warning('Number of elements in aco_a not equal to n_out')
                os.sys.exit()
        else: # Assumes aco_a is a single value as opposed to an array or list
            temp = np.ones(self.n_out) * self.aco_a
        # Add alternating column offsets for each channel
        for ch in range(self.n_out):
            chan = bias_pattern[:,self.xsize*ch:self.xsize*(ch+1)]
            chan[:,inda] += temp[ch]
        
        # Do the same, but with column b
        if isinstance(self.aco_b, (np.ndarray,list)):
            temp = np.asarray(self.aco_b) # Always set to a numpy array
            if temp.size != self.n_out:
                _log.warning('Number of elements in aco_b not equal to n_out')
                os.sys.exit()
        else: # Assumes aco_b is a single value as opposed to an array or list
            temp = np.ones(self.n_out) * self.aco_b
        # Add alternating column offsets for each channel
        for ch in range(self.n_out):
            chan = bias_pattern[:,self.xsize*ch:self.xsize*(ch+1)]
            chan[:,indb] += temp[ch]
    
        # Add in the bias pattern
        for z in np.arange(self.naxis3):
            result[z,:,:] += bias_pattern

        # Add in random frame-to-frame bias offsets
        # First, correlated bias between channels
        if self.ref_f2f_corr is not None:
            for z in np.arange(self.naxis3):
                result[z,:,:] += self.ref_f2f_corr * np.random.randn()
        # Next, channel-specific bias offsets
        if self.ref_f2f_ucorr is not None:
            if isinstance(self.ref_f2f_ucorr, (np.ndarray,list)):
                temp = np.asarray(self.ref_f2f_ucorr)
                if temp.size != self.n_out:
                    _log.warning('Number of elements in ref_f2f_ucorr not equal to n_out')
                    os.sys.exit()
            else: # Single value as opposed to an array or list
                temp = np.ones(self.n_out) * self.ref_f2f_ucorr
            for z in np.arange(self.naxis3):
                for ch in range(self.n_out):
                    result[z,:,self.xsize*ch:self.xsize*(ch+1)] += temp[ch] * np.random.randn()
        # Reference instability (frame-to-frame reference offset not recorded in active pixels)
        if self.ref_inst is not None:
            ref_noise = self.ref_inst * self.pink_noise('ref_inst')
            w = self.ref_all
            for z in np.arange(self.naxis3):
                if w[0] > 0:
                    result[z, :w[0], :] += ref_noise[z]
                if w[1] > 0:
                    result[z, -w[1]:,:] += ref_noise[z]
                if w[2] > 0:
                    result[z, :, :w[2]] += ref_noise[z]
                if w[3] > 0:
                    result[z, :,-w[3]:] += ref_noise[z]
            

        # Make white read noise. This is the same for all pixels.
        if self.rd_noise is not None:
            self.message('Generating rd_noise')

            # We want self.rd_noise to be an array or list
            if isinstance(self.rd_noise, (np.ndarray,list)):
                temp = np.asarray(self.rd_noise)
                if temp.size != self.n_out:
                    _log.warning('Number of elements in rd_noise not equal to n_out')
                    os.sys.exit()
            else: # Single value as opposed to an array or list
                self.rd_noise = np.ones(self.n_out) * self.rd_noise
    
            w = self.ref_all
            r = self.reference_pixel_noise_ratio  # Easier to work with
            for z in np.arange(self.naxis3):
                here = np.zeros((self.naxis2, self.naxis1))
        
                # First assume no ref pixels and just add in random noise
                for op in np.arange(self.n_out):
                    x0 = op * self.xsize
                    x1 = x0 + self.xsize
                    here[:,x0:x1] = self.rd_noise[op] * np.random.standard_normal((self.naxis2,self.xsize))
        
                # If there are reference pixels, overwrite with appropriate noise values
                # Noisy reference pixels for each side of detector
                rd_ref = r * np.mean(self.rd_noise)
                if w[0] > 0: # lower
                    here[:w[0],:] = rd_ref * np.random.standard_normal((w[0],self.naxis1))
                if w[1] > 0: # upper
                    here[-w[1]:,:] = rd_ref * np.random.standard_normal((w[1],self.naxis1))
                if w[2] > 0: # left
                    here[:,:w[2]] = rd_ref * np.random.standard_normal((self.naxis2,w[2]))
                if w[3] > 0: # right
                    here[:,-w[3]:] = rd_ref * np.random.standard_normal((self.naxis2,w[3]))
        
                # Add the noise in to the result
                result[z,:,:] += here
                

        # Add correlated pink noise.
        if self.c_pink is not None:
            # c_pink_map was used to hold the entire correlated pink noise cube
            # c_pink_map was useful for debugging, but eats up a lot of space
            #self.c_pink_map = np.zeros((self.naxis3,self.naxis2,self.naxis1))
            self.message('Adding c_pink noise')
            tt = self.c_pink * self.pink_noise('pink') # tt is a temp. variable
            #self.c_pink_full = tt.copy()
    
            tt = np.reshape(tt, (self.naxis3, self.naxis2+self.nfoh, \
                                 self.xsize+self.nroh))[:,:self.naxis2,:self.xsize]
            for op in np.arange(self.n_out):
                x1 = op * self.xsize
                x2 = x1 + self.xsize

                # By default fast-scan readout direction is [-->,<--,-->,<--]
                # If reverse_scan_direction is True, then [<--,-->,<--,-->]
                # same_scan_direction: all --> or all <--
                if self.same_scan_direction:
                    flip = True if self.reverse_scan_direction else False
                elif np.mod(ch,2)==0:
                    flip = True if self.reverse_scan_direction else False
                else:
                    flip = False if self.reverse_scan_direction else True

                if flip: 
                    #self.c_pink_map[:,:,x1:x2] = tt[:,:,::-1]
                    result[:,:,x1:x2] += tt[:,:,::-1]
                else:
                    #self.c_pink_map[:,:,x1:x2] = tt
                    result[:,:,x1:x2] += tt

            del tt
            #result += self.c_pink_map
            #del self.c_pink_map


        # Add uncorrelated pink noise. Because this pink noise is stationary and
        # different for each output, we don't need to flip it (but why not?)
        if self.u_pink is not None:
            # We want self.u_pink to be an array or list
            if isinstance(self.u_pink, (np.ndarray,list)):
                temp = np.asarray(self.u_pink)
                if temp.size != self.n_out:
                    _log.warning('Number of elements in u_pink not equal to n_out')
                    os.sys.exit()
            else: # Single value as opposed to an array or list
                self.u_pink = np.ones(self.n_out) * self.u_pink
        
            # Only do the rest if any values are not 0
            if self.u_pink.any():
                # u_pink_map was used to hold the entire correlated pink noise cube
                # u_pink_map was useful for debugging, but eats up a lot of space
                #self.u_pink_map = np.zeros((self.naxis3,self.naxis2,self.naxis1))
                self.message('Adding u_pink noise')
                for op in np.arange(self.n_out):
                    x1 = op * self.xsize
                    x2 = x1 + self.xsize
                    tt = self.u_pink[op] * self.pink_noise('pink')
                    tt = np.reshape(tt, (self.naxis3, self.naxis2+self.nfoh, \
                                     self.xsize+self.nroh))[:,:self.naxis2,:self.xsize]

                    if self.same_scan_direction:
                        flip = True if self.reverse_scan_direction else False
                    elif np.mod(ch,2)==0:
                        flip = True if self.reverse_scan_direction else False
                    else:
                        flip = False if self.reverse_scan_direction else True

                    if flip: 
                        #self.u_pink_map[:,:,x1:x2] = tt[:,:,::-1]
                        result[:,:,x1:x2] += tt[:,:,::-1]
                    else:
                        #self.u_pink_map[:,:,x1:x2] = tt
                        result[:,:,x1:x2] += tt

                    del tt
                #result += self.u_pink_map
                #del self.u_pink_map


        # Add ACN
        if self.acn is not None:
            self.message('Adding acn noise')
            for op in np.arange(self.n_out):

                # Generate new pink noise for each even and odd vector.
                # We give these the abstract names 'a' and 'b' so that we
                # can use a previously worked out formula to turn them
                # back into an image section.
                a = self.acn * self.pink_noise('acn')
                b = self.acn * self.pink_noise('acn')

                # Pick out just the real pixels (i.e. ignore the gaps)
                a = a[np.where(self.m_short == 1)]
                b = b[np.where(self.m_short == 1)]

                # Reformat into an image section. This uses the formula
                # mentioned above.
                acn_cube = np.reshape(np.transpose(np.vstack((a,b))),
                                      (self.naxis3,self.naxis2,self.xsize))

                # Add in the ACN. Because pink noise is stationary, we can
                # ignore the readout directions. There is no need to flip
                # acn_cube before adding it in.
                x0 = op * self.xsize
                x1 = x0 + self.xsize
                result[:,:,x0:x1] += acn_cube
                del acn_cube



        # Add PCA-zero. The PCA-zero template is modulated by 1/f.
        if self.pca0_amp is not None:
            self.message('Adding PCA-zero "picture frame" noise')
            gamma = self.pink_noise(mode='pink')
            zoom_factor = self.naxis2 * self.naxis3 / np.size(gamma)
            gamma = zoom(gamma, zoom_factor, order=1, mode='mirror')
            gamma = np.reshape(gamma, (self.naxis3,self.naxis2))
            for z in np.arange(self.naxis3):
                for y in np.arange(self.naxis2):
                    result[z,y,:] += self.pca0_amp*self.pca0[y,:]*gamma[z,y]



        # Add in dark current for each frame
        #k = np.array([[0,0.01,0],[0.01,0.96,0.01],[0,0.01,0]])
        if self.dark_image is not None:
            self.message('Adding dark current')
            gain_temp = self.gain # 2.0 # Temporary gain (e-/ADU)
            dark_frame = self.dark_image * self.tframe * gain_temp # electrons
            # Set reference pixels' dark current equal to 0
            if w[0] > 0: # lower
                dark_frame[:w[0],:] = 0
            if w[1] > 0: # upper
                dark_frame[-w[1]:,:] = 0
            if w[2] > 0: # left
                dark_frame[:,:w[2]] = 0
            if w[3] > 0: # right
                dark_frame[:,-w[3]:] = 0
    
            # For each read frame, create random dark current instance based on Poisson
            dark_temp = np.zeros([self.naxis2,self.naxis1])		
            for z in np.arange(self.naxis3):
                dark_temp += np.random.poisson(dark_frame, size=None)
                #dark_ipc = convolve(dark_temp, k, mode='constant', cval=0.0)
                result[z,:,:] += dark_temp
        

        # If the data cube has only 1 frame, reformat into a 2-dimensional image.
        if self.naxis3 == 1:
            self.message('Reformatting cube into image')
            result = result[0,:,:]


        # Convert to ADU and unsigned int
        if out_ADU:
            # Apply Gain (e/ADU) to convert to ADU (DN)
            if self.gain != 1:
                self.message('Applying Gain')
                result /= self.gain

            # If the data cube has more than one frame, convert to unsigned int
            #if self.naxis3 > 1:
            self.message('Converting to 16-bit unsigned integer')
            # Ensure that there are no negative pixel values. 
            result[result < 0] = 0
            # And that anything higher than 65535 gets tacked to the top end
            result[result >= 2**16] = 2**16 - 1
            result = result.astype('uint16')

        # Create HDU
        # THIS NEEDS TO BE UPDATED WITH BETTER INFO/FORMAT
        hdu = fits.PrimaryHDU(result)
#         hdu.header.append()
#         hdu.header.append(('TFRAME', self.tframe, 'Time in seconds between frames'))
#         hdu.header.append(('INTTIME', self.inttime, 'Total integration time for one MULTIACCUM'))
#         hdu.header.append(('RD_NOISE', self.rd_noise, 'Read noise'))
#         #hdu.header.append(('PEDESTAL', self.pedestal, 'Pedestal drifts'))
#         hdu.header.append(('C_PINK', self.c_pink, 'Correlated pink'))
#         #hdu.header.append(('U_PINK', self.u_pink, 'Uncorrelated pink'))
#         hdu.header.append(('ACN', self.acn, 'Alternating column noise'))
#         hdu.header.append(('PCA0', self.pca0_amp, \
#                            'PCA zero, AKA picture frame'))
        hdu.header['HISTORY'] = 'Created by NGHXRG version ' \
                                + str(self.nghxrg_version)
                        
        # Write the result to a FITS file
        if o_file is not None:
            self.message('Writing FITS file')
            hdu.writeto(o_file, clobber='True')
    
        self.message('Exiting mknoise()')

        return hdu