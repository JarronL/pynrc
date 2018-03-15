"""pyNRC - Python ETC and Simulator for JWST NIRCam"""

from __future__ import division, print_function, unicode_literals

#from astropy.convolution import convolve_fft

#from scipy import fftpack
#import pdb
# Import libraries
from astropy.table import Table
from .nrc_utils import *

import logging
_log = logging.getLogger('pynrc')

class multiaccum(object):
    """
    A class for defining MULTIACCUM ramp settings.
    See `NIRCam MULTIACCUM documentation
    <https://jwst-docs.stsci.edu/display/JTI/NIRCam+Detector+Readout+Patterns>`_
    for more details.

    Parameters
    ----------------
    read_mode : str
        NIRCam Ramp Readout mode such as 'RAPID', 'BRIGHT1', DEEP8, etc.
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

    Notes
    -----

    **NIRCam-specific readout modes**
    
    ========  ===  ===
    Pattern    NF  ND2
    ========  ===  ===
    RAPID      1    0
    BRIGHT1    1    1
    BRIGHT2    2    0
    SHALLOW2   2    3
    SHALLOW4   4    1
    MEDIUM2    2    8
    MEDIUM8    8    2
    DEEP2      2   18
    DEEP8      8   12
    ========  ===  ===
    """

    def __init__(self, read_mode='RAPID', nint=1, ngroup=1, nf=1, nd1=0, nd2=0, nd3=0, 
                 **kwargs):


        # Pre-defined patterns
        patterns = ['RAPID', 'BRIGHT1', 'BRIGHT2', 'SHALLOW2', 'SHALLOW4', 'MEDIUM2', 'MEDIUM8', 'DEEP2', 'DEEP8']
        nf_arr   = [1,1,2,2,4,2,8, 2, 8]
        nd2_arr  = [0,1,0,3,1,8,2,18,12]
        ng_max   = [10,10,10,10,10,10,10,20,20]  # Currently ignored, because not valid for TSO
        self._pattern_settings = dict(zip(patterns, zip(nf_arr, nd2_arr, ng_max)))

        #self.nexp = nexp # Don't need multiple exposures. Just increase nint
        self.nint = nint
        self._ngroup_max = 10000
        self.ngroup = ngroup

        # Modify these directly rather via the @property
        self._nf = self._check_int(nf,1)
        self._nd1 = self._check_int(nd1,0)
        self._nd2 = self._check_int(nd2,0)
        self._nd3 = self._check_int(nd3,0)
        # Now set read mode to specified mode, which may modify nf, nd1, nd2, and nd3
        self.read_mode = read_mode

#     @property
#     def nexp(self):
#         """Number of exposures in an obervation."""
#         return self._nexp
#     @nexp.setter
#     def nexp(self, value):
#         self._nexp = self._check_int(value)

    @property
    def nint(self):
        """Number of ramps (integrations) in an exposure."""
        return self._nint
    @nint.setter
    def nint(self, value):
        self._nint = self._check_int(value)

    @property
    def ngroup(self):
        """Number of groups in a ramp (integration)."""
        return self._ngroup
    @ngroup.setter
    def ngroup(self, value):
        value = self._check_int(value,1)
        if value > self._ngroup_max:
            _log.warning('Specified ngroup (%s) greater than allowed value (%s)' \
                         % (value, self._ngroup_max))
            _log.warning('Setting ngroup = %s' % self._ngroup_max)
            value = self._ngroup_max
        self._ngroup = value

    @property
    def nf(self):
        """Number of frames per group."""
        return self._nf
    @nf.setter
    def nf(self, value):
        value = self._check_int(value,1)
        self._nf = self._check_custom(value, self._nf)

    @property
    def nd1(self):
        """Number of drop frame after reset (before first group read)."""
        return self._nd1
    @nd1.setter
    def nd1(self, value):
        value = self._check_int(value, minval=0)
        self._nd1 = self._check_custom(value, self._nd1)

    @property
    def nd2(self):
        """Number of drop frames within a group (aka, groupgap)."""
        return self._nd2
    @nd2.setter
    def nd2(self, value):
        value = self._check_int(value, minval=0)
        self._nd2 = self._check_custom(value, self._nd2)

    @property
    def nd3(self):
        """Number of drop frames after final read frame in ramp."""
        return self._nd3
    @nd3.setter
    def nd3(self, value):
        value = self._check_int(value, minval=0)
        self._nd3 = self._check_custom(value, self._nd3)

    @property
    def read_mode(self):
        """Selected Read Mode in the `patterns_list` attribute."""
        return self._read_mode
    @read_mode.setter
    def read_mode(self, value):
        """Set MULTIACCUM Readout. Automatically updates other relevant attributes."""
        if value is None:
            _log.info('Readout pattern has None value. Setting to CUSTOM.')
            value = 'CUSTOM'

        value = value.upper()
        _check_list(value, self.patterns_list, var_name='read_mode')

        self._read_mode = value
        self._validate_readout()

    @property
    def patterns_list(self):
        """Allowed NIRCam MULTIACCUM patterns"""
        plist = sorted(list(self._pattern_settings.keys()))
        return ['CUSTOM'] + plist


    def to_dict(self, verbose=False):
        """Export ramp settings to a dictionary."""
        p = [('read_mode',self.read_mode), ('nint',self.nint), ('ngroup',self.ngroup), \
             ('nf',self.nf), ('nd1',self.nd1), ('nd2',self.nd2), ('nd3',self.nd3)]
        return tuples_to_dict(p, verbose)


    def _validate_readout(self):
        """ 
        Validation to make sure the defined ngroups, nf, etc. are consistent with
        the selected MULTIACCUM readout pattern.
        """

        if self.read_mode not in self.patterns_list:
            _log.warning('Readout %s not a valid NIRCam readout mode. Setting to CUSTOM.' % self.read_mode)
            self._read_mode = 'CUSTOM'
            _log.warning('Using explicit settings: ngroup=%s, nf=%s, nd1=%s, nd2=%s, nd3=%s' \
                         % (self.ngroup, self.nf, self.nd1, self.nd2, self.nd3))
        elif self.read_mode == 'CUSTOM':
            _log.info('%s readout mode selected.' % self.read_mode)
            _log.info('Using explicit settings: ngroup=%s, nf=%s, nd1=%s, nd2=%s, nd3=%s' \
                         % (self.ngroup, self.nf, self.nd1, self.nd2, self.nd3))
        else:
            _log.info('%s readout mode selected.' % self.read_mode)
            nf, nd2, _ = self._pattern_settings.get(self.read_mode)
            self._nf  = nf
            self._nd1 = 0
            self._nd2 = nd2
            self._nd3 = 0
            _log.info('Setting nf=%s, nd1=%s, nd2=%s, nd3=%s.' % (self.nf, self.nd1, self.nd2, self.nd3))


    def _check_custom(self, val_new, val_orig):
        """Check if read_mode='CUSTOM' before changing variable."""
        if self.read_mode == 'CUSTOM': 
            return val_new
        else: 
            print("Can only modify parameter if read_mode='CUSTOM'.")
            return val_orig

    def _check_int(self, val, minval=1):
        """Check if a value is a positive integer, otherwise throw exception."""
        val = float(val)
        if (val.is_integer()) and (val>=minval): 
            return int(val)
        else:
            #_log.error("Value {} should be a positive integer.".format(val))
            raise ValueError("Value {} must be an integer >={}.".format(val,minval))

    
class DetectorOps(object):
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

        # Typical values for SW/LW detectors that get saved based on SCA ID
        # After setting the SCA ID, these various parameters can be updated,
        # however they will be reset whenever the SCA ID is modified.
        #   - Pixel Scales in arcsec/pix (SIAF PRDDEVSOC-D-012, 2016 April)
        #   - Well saturation level in e-
        #   - Typical dark current values in e-/sec (ISIM CV3)
        #   - Read Noise in e-
        #   - IPC and PPC in %
        #   - p_excess: Parameters that describe the excess variance observed in
        #     effective noise plots.
        self._properties_SW = {'pixel_scale':0.0311, 'dark_current':0.002, 'read_noise':11.5, 
                               'IPC':0.54, 'PPC':0.09, 'p_excess':(1.0,5.0), 'ktc':37.6,
                               'well_level':100e3, 'well_level_old':81e3}
        self._properties_LW = {'pixel_scale':0.0630, 'dark_current':0.034, 'read_noise':10.0, 
                               'IPC':0.60, 'PPC':0.19, 'p_excess':(1.5,10.0), 'ktc':36.8,
                               'well_level':80e3, 'well_level_old':75e3}
        # Automatically set the pixel scale based on detector selection
        self.auto_pixscale = True  

        self._gain_list = {481:2.07, 482:2.01, 483:2.16, 484:2.01, 485:1.83, 
                           486:2.00, 487:2.42, 488:1.93, 489:2.30, 490:1.85}

        self._scaids = {481:'A1', 482:'A2', 483:'A3', 484:'A4', 485:'A5',
                        486:'B1', 487:'B2', 488:'B3', 489:'B4', 490:'B5'}
        # Allow user to specify name using either SCA ID or Detector ID (ie., 481 or 'A1')
        try: # First, attempt to set SCA ID
            self.scaid = detector 
        except ValueError: 
            try: # If that doesn't work, then try to set Detector ID
                self.detid = detector
            except ValueError: # If neither work, raise ValueError exception
                raise ValueError("Invalid detector: {0} \n\tValid names are: {1},\n\t{2}" \
                      .format(detector, ', '.join(self.detid_list), \
                      ', '.join(str(e) for e in self.scaid_list)))

        self._nff = nff

        self._detector_pixels = 2048
        self.wind_mode = wind_mode.upper()
        self.xpix = xpix; self.x0 = x0
        self.ypix = ypix; self.y0 = y0
        self._validate_pixel_settings()

        # Pixel Rate in Hz
        self._pixel_rate = 1e5
        # Number of extra clock ticks per line
        self._line_overhead = 12

        _log.info('Initializing SCA {}/{}'.format(self.scaid,self.detid))
        self.multiaccum = multiaccum(**kwargs)

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
        if self.auto_pixscale: self.pixelscale = dtemp['pixel_scale']
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
        _check_list(value, self.detid_list, var_name='detid')

        # Switch dictioary keys and values, grab the corresponding SCA ID,
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

    @property
    def nout(self):
        """Number of simultaenous detector output channels stripes"""
        return 1 if self.wind_mode == 'WINDOW' else 4

    @property
    def chsize(self):
        """Width of a single amplifier output channel"""
        return self.xpix / self.nout

    @property
    def ref_info(self):
        """Array of reference pixel borders being read out [lower, upper, left, right]."""
        det_size = self._detector_pixels
        x1 = self.x0; x2 = x1 + self.xpix
        y1 = self.y0; y2 = y1 + self.ypix

        w = 4 # Width of ref pixel border
        lower = w-y1; upper = w-(det_size-y2)
        left  = w-x1; right = w-(det_size-x2)
        ref_all = np.array([lower,upper,left,right])
        ref_all[ref_all<0] = 0
        return ref_all
        
    @property
    def nff(self):
        """Number of fast row resets that occur before Reset Frame"""
        if self._nff is None:
            if self.wind_mode=='WINDOW': 
                if   self.ypix>256: nff = 2048
                elif self.ypix>64:  nff = 512
                elif self.ypix>16:  nff = 256
                elif self.ypix>8:   nff = 64
                else:               nff = 16
            elif self.wind_mode=='STRIPE': 
                if   self.ypix==2048: nff = 0
                elif self.ypix>=256:  nff = 2048
                else:                 nff = 512
            elif self.wind_mode=='FULL':
                nff = 0
        else:
            nff = self._nff
        
        return nff
    @nff.setter
    def nff(self, val):
        self._nff = val


    def _validate_pixel_settings(self):
        """ 
        Validation to make sure the defined pixel sizes are consistent with
        detector readout mode (FULL, STRIPE, or WINDOW)
        """

        wind_mode = self.wind_mode

        modes = ['FULL', 'STRIPE', 'WINDOW']
        if wind_mode not in modes:
            _log.warning('%s not a valid window readout mode! Returning...' % wind_mode)
            return

        detpix = self._detector_pixels
        xpix = self.xpix; x0 = self.x0
        ypix = self.ypix; y0 = self.y0

        # Check some consistencies with frame sizes
        if wind_mode == 'FULL':
            if ypix != detpix:
                _log.warning('In {0} mode, but ypix not {1}. Setting ypix={1}.'.format(wind_mode,detpix))
                ypix = detpix
            if y0 != 0:
                _log.warning('In {0} mode, but x0 not 0. Setting y0=0.'.format(wind_mode))
                y0 = 0

        if (wind_mode == 'STRIPE') or (wind_mode == 'FULL'):
            if xpix != detpix:
                _log.warning('In {0} mode, but xpix not {1}. Setting xpix={1}.'.format(wind_mode,detpix))
                xpix = detpix
            if x0 != 0:
                _log.warning('In {0} mode, but x0 not 0. Setting x0=0.'.format(wind_mode))
                x0 = 0
    
        if (x0+xpix) > detpix:
            raise ValueError("x0+xpix ({}+{}) is larger than detector size ({})!".format(x0,xpix,detpix))
        if (y0+ypix) > detpix:
            raise ValueError("y0+ypix ({}+{}) is larger than detector size ({})!".format(y0,ypix,detpix))

        # Update values if no errors were thrown
        self.xpix = xpix; self.x0 = x0
        self.ypix = ypix; self.y0 = y0

    @property
    def _extra_lines(self):
        """Determine how many extra lines/rows are added to a to a given frame"""
        if self.nout == 1:
            xtra_lines = 2 if self.xpix>10 else 3
        elif self.nout == 4:
            xtra_lines = 1
            
        return xtra_lines
        
    @property
    def _exp_delay(self):
        """
        Additional overhead time at the end of an exposure.
        This does not add any more photon flux to a pixel.
        Due to transition to idle.
        """

        # Window Mode        
        if self.nout == 1:
            if   self.xpix>150: xtra_lines = 0
            elif self.xpix>64:  xtra_lines = 1
            elif self.xpix>16:  xtra_lines = 2
            elif self.xpix>8:   xtra_lines = 4
            else:               xtra_lines = 5
        # Full and Stripe
        else: xtra_lines = 1
        
        # Clock ticks per line
        xticks = self.chsize + self._line_overhead  
        return xticks * xtra_lines / self._pixel_rate


    @property
    def _frame_overhead_pix(self):
        """
        Full and Stripe mode frames have an additional pixel at the end.
        """
        
        pix_offset = 0 if self.nout==1 else 1
        return pix_offset
        
    @property
    def time_row_reset(self):
        """NFF Row Resets time per INT"""
        
        nff = self.nff
           
        xtra_lines = int(nff / (self.chsize))
        
        # Clock ticks per line
        xticks = self.chsize + self._line_overhead  
        return xticks * xtra_lines / self._pixel_rate
        
    @property
    def time_frame(self):
        """Determine frame times based on xpix, ypix, and wind_mode."""

        chsize = self.chsize                        # Number of x-pixels within a channel
        xticks = self.chsize + self._line_overhead  # Clock ticks per line
        flines = self.ypix + self._extra_lines      # Lines per frame

        # Add a single pix offset for full frame and stripe.
        pix_offset = self._frame_overhead_pix
        end_delay = 0 # Used for syncing each frame w/ FPE bg activity. Not currently used.

        # Total number of clock ticks per frame (reset, read, and drops)
        fticks = xticks*flines + pix_offset + end_delay

        # Return frame time
        return fticks / self._pixel_rate

    @property
    def time_group(self):
        """Time per group based on time_frame, nf, and nd2."""
        return self.time_frame * (self.multiaccum.nf + self.multiaccum.nd2)

    @property
    def time_ramp(self):
        """Photon collection time for a single ramp (integration)."""

        # How many total frames (incl. dropped and all) per ramp?
        # Exclude nd3 (drops that add nothing)
        ma = self.multiaccum
        nf = ma.nf; nd1 = ma.nd1; nd2 = ma.nd2
        ngroup = ma.ngroup

        tint = (nd1 + ngroup*nf + (ngroup-1)*nd2) * self.time_frame
        #if tint > 1200:
        #    _log.warning('Ramp time of %.2f is long. Is this intentional?' % tint)

        return tint

    @property
    def time_int(self):
        """Photon collection time for a single integration (ramp)."""
        return self.time_ramp

    @property
    def time_exp(self):
        """Total photon collection time for all ramps."""
        return self.multiaccum.nint * self.time_ramp

    @property
    def time_total_int(self):
        """Total time for all frames in a ramp.
        
        Includes resets and excess drops, as well as NFF Rows Reset.
        """

        ma = self.multiaccum
        nf = ma.nf; nd1 = ma.nd1; nd2 = ma.nd2; nd3 = ma.nd3
        ngroup = ma.ngroup
        nr = 1

        nframes = nr + nd1 + ngroup*nf + (ngroup-1)*nd2 + nd3        
        return nframes * self.time_frame + self.time_row_reset

    @property
    def time_total(self):
        """Total exposure acquisition time"""
        return self.multiaccum.nint * self.time_total_int + self._exp_delay

    def to_dict(self, verbose=False):
        """Export detector settings to a dictionary."""

        p = [('detector',self.scaid), ('wind_mode',self.wind_mode), \
             ('xpix',self.xpix), ('ypix',self.ypix), ('x0',self.x0), ('y0',self.y0)]
        return tuples_to_dict(p, verbose)

    def times_to_dict(self, verbose=False):
        """Export ramp times as dictionary."""

        times = [('t_frame',self.time_frame), ('t_group',self.time_group), \
                 ('t_int',self.time_int), ('t_exp',self.time_exp), \
                 ('t_acq',self.time_total), ('t_int_tot',self.time_total_int)]
        return tuples_to_dict(times, verbose)

    def pixel_noise(self, fsrc=0.0, fzodi=0.0, fbg=0.0, verbose=False, **kwargs):
        """Noise values per pixel.
        
        Return theoretical noise calculation for the specified MULTIACCUM exposure 
        in terms of e-/sec. This uses the pre-defined detector-specific noise 
        properties. Can specify flux of a source as well as background and 
        zodiacal light (in e-/sec/pix). After getting the noise per pixel per
        ramp (integration), value(s) are divided by the sqrt(NINT) to return
        the final noise

        Parameters
        ----------
        fsrc : float
            Flux of source in e-/sec/pix
        fzodi : float
            Flux of the zodiacal background in e-/sec/pix
        fbg : float
            Flux of telescope background in e-/sec/pix
        ideal_Poisson : bool
            If set to True, use total signal for noise estimate,
            otherwise MULTIACCUM equation is used.
        verbose : bool
            Print out results at the end.

        Notes
        -----
        fsrc, fzodi, and fbg are functionally the same as they are immediately summed.
        They can also be single values or multiple elements (list, array, tuple, etc.).
        If multiple inputs are arrays, make sure their array sizes match.
        
        """

        ma = self.multiaccum
        # Pixel noise per ramp (e-/sec/pix)
        pn = pix_noise(ma.ngroup, ma.nf, ma.nd2, tf=self.time_frame, \
                       rn=self.read_noise, ktc=self.ktc, p_excess=self.p_excess, \
                       idark=self.dark_current, fsrc=fsrc, fzodi=fzodi, fbg=fbg, **kwargs)

        # Divide by sqrt(Total Integrations)
        final = pn / np.sqrt(ma.nint)
        if verbose:
            print('Noise (e-/sec/pix): {}'.format(final))
            print('Total Noise (e-/pix): {}'.format(final*self.time_exp))

        return final

    def make_header(self, filter=None, pupil=None, obs_time=None, **kwargs):
        """
        Create a generic NIRCam FITS header.

        Parameters
        ----------
        filter :str
            Name of filter element.
        pupil : str
            Name of pupil element.
        obs_time : datetime 
            Specifies when the observation was considered to be executed.
            If not specified, then it will choose the current time.
            This must be a datetime object:
            
            >>> datetime.datetime(2016, 5, 9, 11, 57, 5, 796686)
        """
        return nrc_header(self, filter=filter, pupil=pupil, obs_time=obs_time, **kwargs)

    def pixel_timing_map(self, same_scan_direction=False, reverse_scan_direction=False,
        avg_groups=True, return_flat=True):
        """
        Create array of pixel times for a single ramp. 
        
        Each pixel value corresponds to the precise time at which
        that pixel was read out during the ramp acquisiton. The first
        pixel(s) have t=0.
        
        Parameters
        ----------
        same_scan_direction : bool
            Are all the output channels read in the same direction?
            By default fast-scan readout direction is ``[-->,<--,-->,<--]``
            If ``same_scan_direction``, then all ``-->``
        reverse_scan_direction : bool
            If ``reverse_scan_direction``, then ``[<--,-->,<--,-->]`` or all ``<--``
        avg_groups : bool
            For groups where nf>1, the telescope data gets averaged via a 
            bit-shifter. Setting ``avg_groups=True`` also averages the
            pixel times in a similar manner. Default is True.
        return_flat : bool
            
            
        Returns
        -------
        array
            If ``return_flat=True`` then the data is a flattened array for a
            single channel output. Otherwise, the output is a data cube of the
            same size and shape of the raw data with these detector settings.
            
        Example
        -------
        Assume you have a cube of raw full frame data (RAPID, ngroup=5).
        Create a Detector instance:
        
        >>> d = DetectorOps(ngroup=5)
        >>> tarr = d.pixel_timing_map(return_flat=True, avg_groups=True)
        
        >>> nx = ny = 2048
        >>> nout = 4            # Number of amplifier output channels
        >>> chsize = nx // nout # Channel size (x-direction)
        >>> # Reshape into (nz, ny, nout, chsize)
        >>> data = data.reshape([-1,ny,nout,chsize])
        >>> # Reverse odd channels in x-direction to match even
        >>> for ch in range(nout):
        >>>     if np.mod(ch,2)==1:
        >>>         data[:,:,ch,:] = data[:,:,ch,::-1]
        >>> # Final data reshaped into 4 flattened output channels
        >>> data = data.transpose([0,1,3,2]).reshape([-1,nout])
        >>> # Can plot this like plt.plot(tarr, data) to make 4 line plots

        """
        xpix = self.xpix
        ypix = self.ypix
        nout = self.nout

        chsize = self.chsize                   # Number of x-pixels within a channel
        xticks = chsize + self._line_overhead  # Clock ticks per line
        flines = ypix + self._extra_lines # Lines per frame

        # Add a single pix offset for full frame and stripe.
        pix_offset = self._frame_overhead_pix

        # Total number of clock ticks per frame (reset, read, and drops)
        fticks = xticks*flines + pix_offset

        # Total ticks in a ramp (exclude nd3 drop frames)
        ma = self.multiaccum
        nf = ma.nf; nd1 = ma.nd1; nd2 = ma.nd2
        ngroup = ma.ngroup

        nframes = nd1 + ngroup*nf + (ngroup-1)*nd2
        nticks = fticks * nframes
        
        # Make large array of pixel timing
        arr = np.arange(nticks).reshape([nframes,-1])
        
        # Chop off single pix overhead
        if self._frame_overhead_pix>0:
            arr = arr[:,:-self._frame_overhead_pix]
            
        # Reshape to 3D array
        arr = arr.reshape([nframes, flines, xticks])
        
        # Chop off x & y overheads
        arr = arr[:,:ypix,:chsize]

        # By default fast-scan readout direction is [-->,<--,-->,<--]
        # If same_scan_direction, then all --> 
        # If reverse_scan_direction, then [<--,-->,<--,-->] or all <--
        if reverse_scan_direction:
            arr = arr[::-1]
        
        if same_scan_direction: # Everything in same direction 
            if nout>1:
                data = arr.repeat(nout,2)
            else:
                data = arr
        else: # Consecutive outputs reversed
            if nout>1:
                arr_list = []
                for ch in range(nout):
                    if np.mod(ch,2) == 0: arr_list.append(arr)
                    else: arr_list.append(arr[:,:,::-1])
                data = np.concatenate(arr_list, axis=2)
            else:
                data = arr
                
        del arr, arr_list

        # Timing for averaged (bit-shifted) frames
        # Remove drops and average grouped data
        if avg_groups and (nf>1 or nd2>0):
            # Trailing drop frames already excluded
            # so need to pull off last group of avg'ed frames
            data_end = data[-nf:,:,:].mean(axis=0) if nf>1 else data[-1:,:,:]
            data_end = data_end.reshape([1,ypix,xpix])

            # Only care about first (n-1) groups
            # Last group is handled separately
            # Cut off last set of nf frames
            data = data[:-nf,:,:]

            # Reshape for easy group manipulation
            data = data.reshape([-1,nf+nd2,ypix,xpix])

            # Trim off the dropped frames (nd2)
            if nd2>0: data = data[:,:nf,:,:]

            # Average the frames within groups
            # In reality, the 16-bit data is bit-shifted
            data = data.reshape([-1,ypix,xpix]) if nf==1 else data.mean(axis=1)

            # Add back the last group (already averaged)
            data = np.append(data,data_end,axis=0)
            
        # Put into time
        data /= self._pixel_rate
            
        # Return timing info
        if return_flat: # Flatten array
            return data.ravel()
        else: # Get rid of dimensions of length 1
            return data.squeeze()

class NIRCam(object):

    """NIRCam base instrument class
    
    Creates a NIRCam instrument class that holds all the information pertinent to
    an observation using a given channel/module (SWA, SWB, LWA, LWB). 

    The user merely inputs the filter name, pupil element, coronagraphic mask,
    and module A or B. Based on these settings, a sequence of detector objects
    will be created that spans the correct channel. For instance, a filter value
    of 'F210M' paired with module='A' will generate four detector objects, one
    for each of the four SWA SCAs 481-484 (A1-A4, if you prefer).

    A number of other options are passed via the kwargs parameter that allow
    setting of the detector readout mode, MULTIACCUM settings, and settings
    for the PSF calculation to be passed to WebbPSF.

    Parameters
    ----------
    filter : str
        Name of NIRCam filter.
    pupil : str, None
        NIRCam pupil elements such as grisms or lyot stops (default: None).
    mask : str, None
        Specify which coronagraphic occulter (default: None).
    module : str
        NIRCam Module 'A' or 'B' (default: 'A').
    ND_acq : bool
        ND square acquisition (default: False).
        
    Notes
    -----
    **Detector Settings**
    
    The following keyword arguments will be passed to both the :class:`DetectorOps` and
    :class:`multiaccum` instances. These can be set directly upon initialization of the
    of the NIRCam instances or updated later using the :meth:`update_detectors`
    method. All detector instances will be considered to operate in this mode.
    
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
        
    Notes
    -----
    **PSF Settings**

    The following keyword arguments will be passed to the PSF generation function
    :func:`~pynrc.nrc_utils.psf_coeff` which calls :mod:`webbpsf`.  These can be set directly 
    upon initialization of the of the NIRCam instances or updated later using the 
    :meth:`update_psf_coeff` method. 

    Keyword Args
    ------------
    fov_pix : int
        Size of the FoV in pixels (real SW or LW pixels).
        The defaults depend on the type of observation.
    oversample : int
        Factor to oversample during WebbPSF calculations.
        Default 2 for coronagraphy and 4 otherwise.
    offset_r : float
        Radial offset from the center in arcsec.
    offset_theta :float
        Position angle for radial offset, in degrees CCW.
    bar_offset : float or None
        For bar masks, the position along the bar to place the PSF (arcsec).
        Use :func:`~pynrc.nrc_utils.offset_bar` for filter-dependent locations.
        If not set, default is to decide location based on selected filter.
        Updates :attr:`bar_offset` attribute.
    wfe_drift : float
        Wavefront error drift amplitude in nm.
        Updates :attr:`wfe_drift` attribute.
    opd : tuple or HDUList
        Tuple (file, slice) or filename or HDUList specifying OPD.
    tel_pupil : str
        File name or HDUList specifying telescope entrance pupil.
    jitter : str or None
        Currently either 'gaussian' or None.
    jitter_sigma : float
        If ``jitter = 'gaussian'``, then this is the size of the blurring effect.
    save : bool
        Save the resulting PSF coefficients to a file? (default: True)
    force : bool
        Forces a recalcuation of PSF even if saved PSF exists. (default: False)
    quick : bool
        Only perform a fit over the filter bandpass with a smaller default
        polynomial degree fit. Not compatible with save.
        
    Examples
    --------
    Basic example of generating a full frame NIRCam observation:
    
    >>> inst = pynrc.NIRCam('F210M', module='A', read_mode='DEEP8', nint=10, ngroup=5)

    is the same as:

    >>> inst = pynrc.NIRCam('F210M', module='A')
    >>> inst.update_detectors(read_mode='DEEP8', nint=10, ngroup=5)


    """
    
    # Variable indicating whether or not to warn about 
    _fov_pix_warn = True

    def __init__(self, filter='F210M', pupil=None, mask=None, module='A', ND_acq=False,
        **kwargs):
                 
        # Available Filters
        # Note: Certain narrowband filters reside in the pupil wheel and cannot be paired
        # with pupil elements. This will be check for later.
        self._filters_sw = ['F070W', 'F090W', 'F115W', 'F150W', 'F150W2', 'F200W',
             'F140M', 'F162M', 'F182M', 'F210M', 'F164N', 'F187N', 'F212N']
        self._filters_lw = ['F277W', 'F322W2', 'F356W', 'F444W', 'F323N', 'F405N', 'F466N', 'F470N',
             'F250M', 'F300M', 'F335M', 'F360M', 'F410M', 'F430M', 'F460M', 'F480M']
     
        # Coronagraphic Masks
        self._coron_masks = [None, 'MASK210R', 'MASK335R', 'MASK430R', 'MASKSWB', 'MASKLWB']

        # Pupil Wheel elements
        self._lyot_masks = ['CIRCLYOT', 'WEDGELYOT']
        # DHS in SW and Grisms in LW
        self._dhs = ['DHS0', 'DHS60']
        self._grism = ['GRISM0', 'GRISM90']
        # Weak lens are only in SW pupil wheel (+4 in filter wheel)
        weak_lens = ['+4', '+8', '-8', '+12 (=4+8)', '-4 (=4-8)']
        self._weak_lens = ['WEAK LENS '+s for s in weak_lens]	

        # Let's figure out what keywords the user has set and try to 
        # interpret what he/she actually wants. If certain values have
        # been set to None or don't exist, then populate with defaults
        # and continue onward.

        # Set everything to upper case first
        filter = filter.upper()
        pupil = 'CLEAR' if pupil is None else pupil.upper()
        if mask is not None: mask = mask.upper()
        module = 'A' if module is None else module.upper()
        
        self._ice_scale = kwargs['ice_scale'] if 'ice_scale' in kwargs.keys() else None
        self._nvr_scale = kwargs['nvr_scale'] if 'nvr_scale' in kwargs.keys() else None

        # If alternate Weak Lens values are specified
        if 'WL' in pupil:
            wl_alt = {'WLP4' :'WEAK LENS +4', 
                      'WLP8' :'WEAK LENS +8', 
                      'WLP12':'WEAK LENS +12 (=4+8)', 
                      'WLM4' :'WEAK LENS -4 (=4-8)',
                      'WLM8' :'WEAK LENS -8'}
            pupil = wl_alt.get(pupil, pupil)


        # Validate all values, set values, and update bandpass
        # Test and set the intrinsic/hidden variables directly rather than through setters
        _check_list(filter, self.filter_list, 'filter')
        _check_list(module, ['A','B'], 'module')
        _check_list(pupil, self.pupil_list, 'pupil')
        _check_list(mask, self.mask_list, 'mask')
        self._filter = filter
        self._module = module
        self._pupil = pupil
        self._mask = mask
        self._ND_acq = ND_acq
        
        self._wfe_drift = 0
        self._bar_offset = None
        self._fov_pix_bg = 33

        self._update_bp()		
        self._validate_wheels()
        self.update_detectors(**kwargs)
        self.update_psf_coeff(**kwargs)


    # Allowed values for filters, coronagraphic masks, and pupils
    @property
    def filter_list(self):
        """List of allowable filters."""
        return self._filters_sw + self._filters_lw
    @property
    def mask_list(self):
        """List of allowable coronagraphic mask values."""
        return self._coron_masks
    @property
    def pupil_list(self):
        """List of allowable pupil mask values."""
        return ['CLEAR','FLAT'] + self._lyot_masks + self._grism + self._dhs + self._weak_lens

    @property
    def filter(self):
        """Name of filter bandpass"""
        return self._filter
    @filter.setter
    def filter(self, value):
        """Set the filter name"""
        value = value.upper()
        _check_list(value, self.filter_list, 'filter')

        # Store original settings of filter name and SW or LW channel
        vold = self._filter; ch_old = self.channel
        # Changes to the new filter and update filter curve and bandpass
        self._filter = value
        if vold != self._filter: 
            self._update_bp()
            self.update_psf_coeff()
        if ch_old != self.channel: self.update_detectors()

        self._validate_wheels()

    @property
    def pupil(self):
        """Name of pupil element"""
        return self._pupil
    @pupil.setter
    def pupil(self, value):
        """Set the pupil name"""
        value = 'CLEAR' if value is None else value.upper()
        
        # If alternate Weak Lens values are specified
        if 'WL' in value:
            wl_alt = {'WLP4' :'WEAK LENS +4', 
                      'WLP8' :'WEAK LENS +8', 
                      'WLP12':'WEAK LENS +12 (=4+8)', 
                      'WLM4' :'WEAK LENS -4 (=4-8)',
                      'WLM8' :'WEAK LENS -8'}
            value = wl_alt.get(value, value)
        
        _check_list(value, self.pupil_list, 'pupil')
        vold = self._pupil; self._pupil = value
        if vold != self._pupil: 
            self._update_bp()
            self.update_psf_coeff()
        self._validate_wheels()

    @property
    def mask(self):
        """Name of coronagraphic mask element"""
        return self._mask
    @mask.setter
    def mask(self, value):
        """Set the coronagraphic mask"""
        if value is not None: value = value.upper()
        _check_list(value, self.mask_list, 'mask')
        vold = self._mask; self._mask = value
        if vold != self._mask: 
            self._update_bp()
            self.update_psf_coeff()
            self.update_detectors()
        self._validate_wheels()

    @property
    def module(self):
        """NIRCam modules A or B"""
        return self._module
    @module.setter
    def module(self, value):
        """Set the which NIRCam module (A or B)"""
        _check_list(value, ['A','B'], 'module')
        vold = self._module; self._module = value
        if vold != self._module:
            self._update_bp()
            self.update_detectors()
            self.update_psf_coeff()
    
    @property
    def ND_acq(self):
        """Use Coronagraphic ND acquisition square?"""
        return self._ND_acq
    @ND_acq.setter
    def ND_acq(self, value):
        """Set whether or not we're placed on an ND acquisition square."""
        _check_list(value, [True, False], 'ND_acq')
        vold = self._ND_acq; self._ND_acq = value
        if vold != self._ND_acq:
            self._update_bp()
        self._validate_wheels()

    @property
    def channel(self):
        """NIRCam wavelength channel ('SW' or 'LW')."""
        if self.filter in self._filters_sw: return 'SW'
        if self.filter in self._filters_lw: return 'LW'

        # If we got this far, then something went wrong.
        err_str = 'Something went wrong. Do not recognize filter {}'.format(self.filter)
        _log.error(err_str)
        raise ValueError(err_str)

    @property
    def bandpass(self):
        """The wavelength dependent bandpass."""
        return self._bandpass
    def _update_bp(self):
        """Update bandpass based on filter, pupil, and module, etc."""
        self._bandpass = read_filter(self.filter, self.pupil, self.mask, 
                                     self.module, self.ND_acq,
                                     ice_scale=self._ice_scale, nvr_scale=self._nvr_scale)

    def plot_bandpass(self, ax=None, color=None, title=None, **kwargs):
        """
        Plot the instrument bandpass on a selected axis.
        Can pass various keywords to ``matplotlib.plot`` function.
        
        Parameters
        ----------
        ax : matplotlib.axes, optional
            Axes on which to plot bandpass.
        color : 
            Color of bandpass curve.
        title : str
            Update plot title.
        
        Returns
        -------
        matplotlib.axes
            Updated axes
        """

        if ax is None:
            f, ax = plt.subplots(**kwargs)
        color='indianred' if color is None else color

        bp = self.bandpass
        w = bp.wave / 1e4; f = bp.throughput
        ax.plot(w, f, color=color, label=bp.name+' Filter', **kwargs)
        ax.set_xlabel('Wavelength ($\mu m$)')
        ax.set_ylabel('Throughput')

        if title is None:
            title = bp.name + ' - Mod' + self.module
        ax.set_title(title)
    
        return ax

    
    @property
    def multiaccum(self):
        """:class:`multiaccum` object"""
        return self.Detectors[0].multiaccum
    @property
    def multiaccum_times(self):
        """Exposure timings in dictionary
        
        t_frame   : Time of a single frame.
        t_group   : Time of a single group (read frames + drop frames).
        t_int     : Photon collection time for a single ramp/integration.
        t_int_tot : Total time for all frames (reset+read+drop) in a ramp/integration.
        t_exp     : Total photon collection time for all ramps.
        t_acq     : Total acquisition time to complete exposure with all overheads.
        """
        return self.Detectors[0].times_to_dict()

    @property
    def det_info(self):
        """Dictionary housing detector info parameters and keywords."""
        return self._det_info

    @property
    def pix_scale(self):
        """Pixel scale in arcsec/pixel (deprecated)"""
        return self.pixelscale
    @property
    def pixelscale(self):
        """Pixel scale in arcsec/pixel"""
        return self.Detectors[0].pixelscale
    @property
    def well_level(self):
        """Detector well level in units of electrons"""
        return self.Detectors[0].well_level


    def update_detectors(self, verbose=False, det_list=None, **kwargs):
        """
        Generates a list of detector objects depending on module and channel.
        This function will get called any time a filter, pupil, mask, or
        module is modified by the user directly:
        
        >>> nrc = pynrc.NIRCam('F430M', ngroup=10, nint=5)
        >>> nrc.filter = 'F444W'

        If the user wishes to change any properties of the multiaccum ramp
        or detector readout mode, pass those arguments through this function
        rather than creating a whole new NIRCam() instance. For example:
        
        >>> nrc = pynrc.NIRCam('F430M', ngroup=10, nint=5)
        >>> nrc.update_detectors(ngroup=2, nint=10, wind_mode='STRIPE', ypix=64)
    
        A dictionary of the keyword settings can be referenced in :attr:`det_info`.
        This dictionary cannot be modified directly.
        
        Parameters
        ----------
        det_list : list, None
            List of detector names (481, 482, etc.) to consider.
            If not set, then defaults are chosen based on observing mode.
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

        """

        if self.module=='A':
            if self.channel=='LW':
                det_list = [485]
            elif self.mask is None:
                det_list = [481,482,483,484]
            elif ('WB' in self.mask) or ('WEDGELYOT' in self.pupil):
                det_list = [484]
            elif ('210R' in self.mask) or ('335R' in self.mask) or ('CIRCLYOT' in self.pupil):
                det_list = [482]
            else:
                errmsg = 'No detector makes sense here ({} {} {}).'\
                    .format(self.filter, self.pupil, self.mask)
                raise ValueError(errmsg)
                
            #det_list = [481,482,483,484] if self.channel=='SW' else [485]
            #if ('CIRCLYOT'  in self.pupil) and (self.channel=='SW'): det_list = [482]
            #if ('WEDGELYOT' in self.pupil) and (self.channel=='SW'): det_list = [484]
        if self.module=='B':
            if self.channel=='LW':
                det_list = [490]
            elif self.mask is None:
                det_list = [486,487,488,489]
            elif ('WB' in self.mask) or ('WEDGELYOT' in self.pupil):
                det_list = [488]
            elif ('210R' in self.mask) or ('335R' in self.mask) or ('CIRCLYOT' in self.pupil):
                det_list = [486]
            else:
                errmsg = 'No detector makes sense here ({} {} {}).'\
                    .format(self.filter, self.pupil, self.mask)
                raise ValueError(errmsg)

            #det_list = [486,487,488,489] if self.channel=='SW' else [490]
            #if ('CIRCLYOT'  in self.pupil) and (self.channel=='SW'): det_list = [486]
            #if ('WEDGELYOT' in self.pupil) and (self.channel=='SW'): det_list = [488]

        # Check if kwargs is empty
        if not kwargs:
            try: kwargs = self.det_info
            except AttributeError: kwargs = {}
        else:
            try: self._det_info.update(kwargs)
            except AttributeError: self._det_info = kwargs
            kwargs = self.det_info

        # Update detector class
        # For now, it's just easier to delete old instances and start from scratch
        # rather than tracking changes and updating only the changes. That could 
        # get complicated, and I don't think there is a memory leak from deleting
        # the Detectors instances?
        try: del self.Detectors
        except AttributeError: pass
        self.Detectors = [DetectorOps(det, **kwargs) for det in det_list]

        # Update stored kwargs
        kw1 = self.Detectors[0].to_dict()
        _ = kw1.pop('detector',None)
        kw2 = self.multiaccum.to_dict()
        self._det_info = merge_dicts(kw1,kw2)

        if verbose:
            print('New Ramp Settings:')
            keys = ['read_mode', 'nf', 'nd2', 'ngroup', 'nint']
            for k in keys:
                v = self.det_info[k]
                if isinstance(v,float): print("{:<9} : {:>8.0f}".format(k, v))
                else: print("  {:<9} : {:>8}".format(k, v))

            print('New Detector Settings')
            keys = ['wind_mode', 'xpix', 'ypix', 'x0', 'y0']
            for k in keys:
                v = self.det_info[k]
                if isinstance(v,float): print("{:<9} : {:>8.0f}".format(k, v))
                else: print("  {:<9} : {:>8}".format(k, v))
    
            print('New Ramp Times')
            ma = self.multiaccum_times
            keys = ['t_group', 't_frame', 't_int', 't_int_tot', 't_exp', 't_acq']
            for k in keys:
                print('  {:<9} : {:>8.3f}'.format(k, ma[k]))


    # Check consistencies with pre-defined readout patterns
    def _validate_wheels(self):
        """ 
        Validation to make sure the selected filters and pupils are allowed to be in parallel.
        """
        
        def do_warn(wstr):
            _log.warning(wstr)
            _log.warning('Proceed at your own risk!')

        filter  = self.filter
        pupil   = self.pupil
        mask    = self.mask
        channel = self.channel

        if mask is None: mask = ''

        warn_flag = False
        # Weak lenses can only occur in SW modules
        if ('WEAK LENS' in pupil) and (channel=='LW'):
            wstr = '{} in pupil is not valid with filter {}.'.format(pupil,filter)
            wstr = wstr + '\nWeak lens only in SW module.'
            do_warn(wstr)

        # DHS in SW modules
        if ('DHS' in pupil) and (channel=='LW'):
            wstr = '{} in pupil is not valid with filter {}.'.format(pupil,filter)
            wstr = wstr + '\nDHS only in SW module.'
            do_warn(wstr)
            
        # DHS cannot be paird with F164N or F162M
        flist = ['F164N', 'F162M']
        if ('DHS' in pupil) and (filter in flist):
            wstr = 'Both {} and filter {} exist in same pupil wheel.'.format(pupil,filter)
            do_warn(wstr)

        # Grisms in LW modules
        if ('GRISM' in pupil) and (channel=='SW'):
            wstr = '{} in pupil is not valid with filter {}.'.format(pupil,filter)
            wstr = wstr + '\nGrisms only in LW module.'
            do_warn(wstr)
            
        # Grisms cannot be paired with any Narrowband filters
        flist = ['F323N', 'F405N', 'F466N', 'F470N']
        if ('GRISM' in pupil) and (filter in flist):
            wstr = 'Both {} and filter {} exist in same pupil wheel.'.format(pupil,filter)
            do_warn(wstr)

        # MASK430R falls in SW SCA gap and cannot be seen by SW module
        if ('MASK430R' in mask) and (channel=='SW'):
            wstr = '{} mask is no visible in SW module (filter is {})'.format(mask,filter)
            do_warn(wstr)

        # WEAK LENS +4 only valid with F212N
        # Or is is F210M???
        wl_list = ['WEAK LENS +12 (=4+8)', 'WEAK LENS -4 (=4-8)', 'WEAK LENS +4']
        if (pupil in wl_list) and (filter!='F212N'):
            wstr = '{} is only valid with filter F212N.'.format(pupil)
            do_warn(wstr)

        # Items in the same SW pupil wheel
        sw2 = ['WEAK LENS +8', 'WEAK LENS -8', 'F162M', 'F164N', 'CIRCLYOT', 'WEDGELYOT']
        if (filter in sw2) and (pupil in sw2):
            wstr = '{} and {} are both in the SW Pupil wheel.'.format(filter,pupil)
            do_warn(wstr)

        # Items in the same LW pupil wheel
        lw2 = ['F323N', 'F405N', 'F466N', 'F470N', 'CIRCLYOT', 'WEDGELYOT']
        if (filter in lw2) and (pupil in lw2):
            wstr = '{} and {} are both in the LW Pupil wheel.'.format(filter,pupil)
            do_warn(wstr)
    
        # ND_acq must have a LYOT stop, otherwise coronagraphic mask is not in FoV
        if self.ND_acq and ('LYOT' not in pupil):
            wstr = 'CIRCLYOT or WEDGELYOT must be in pupil wheel if ND_acq=True.'
            do_warn(wstr)

        # ND_acq and coronagraphic mask are mutually exclusive
        if self.ND_acq and (mask != ''):
            wstr = 'If ND_acq is set, then mask must be None.'
            do_warn(wstr)


    @property
    def psf_coeff(self):
        """Cube of polynomial coefficients used to generate a PSF."""
        return self._psf_coeff
    @property
    def psf_info(self):
        """Info used to create psf_coeff."""
        return self._psf_info

    @property
    def psf_coeff_bg(self):
        """Cube of polynomial coefficients used to generate a PSF."""
        return self._psf_coeff_bg
    @property
    def psf_info_bg(self):
        """Info used to create psf_coeff for faint background sources."""
        return self._psf_info_bg

    @property
    def wfe_drift(self):
        """WFE drift relative to nominal PSF (nm)"""
        return self._wfe_drift
    @wfe_drift.setter
    def wfe_drift(self, value):
        """Set the WFE drift value and update coefficients"""
        # Only update if the value changes
        vold = self._wfe_drift; self._wfe_drift = value
        if vold != self._wfe_drift: 
            self.update_psf_coeff(wfe_drift=self._wfe_drift)

    @property
    def bar_offset(self):
        """Offset position along bar mask (arcsec)."""
        return self._bar_offset
    @bar_offset.setter
    def bar_offset(self, value):
        """Set the bar offset position and update coefficients"""
        # Only update if the value changes
        if self.mask is None:
            self._bar_offset = 0 #None
        elif self.mask[-1]=='B':
            vold = self._bar_offset
            # Value limits between -10 and 10
            if np.abs(value)>10:
                value = 10 if value>0 else -10
                msg1 = 'bar_offset value must be between -10 and 10.'
                msg2 = 'Setting to {}.'.format(value)
                _log.warning('{} {}'.format(msg1,msg2))
            
            self._bar_offset = value
            if vold != self._bar_offset: 
                self.update_psf_coeff(bar_offset=self._bar_offset)
        else:
            self._bar_offset = 0
            
    def update_psf_coeff(self, fov_pix=None, oversample=None, 
        offset_r=None, offset_theta=None, tel_pupil=None, opd=None,
        wfe_drift=None, jitter='gaussian', jitter_sigma=0.007, 
        bar_offset=None, save=None, force=False, **kwargs):
        """Create new PSF coefficients.
        
        Generates a set of PSF coefficients from a sequence of WebbPSF images.
        These coefficients can then be used to generate a sequence of
        monochromatic PSFs (useful if you need to make hundreds of PSFs
        for slitless grism or DHS observations) that are subsequenty
        convolved with the the instrument throughput curves and stellar
        spectrum. The coefficients are stored in :attr:`psf_coeff` attribute.

        A corresponding dictionary :attr:`psf_info` is also saved that contains
        the size of the FoV (in detector pixels) and oversampling factor 
        that was used to generate the coefficients.

        While originally created for coronagraphy, grism, and DHS observations,
        this method is actually pretty quick, so it has become the default
        for imaging as well.

        Parameters
        ----------
        fov_pix : int
            Size of the FoV in pixels (real SW or LW pixels).
            The defaults depend on the type of observation.
            Odd number place the PSF on the center of the pixel,
            whereas an even number centers it on the "crosshairs."
        oversample : int
            Factor to oversample during WebbPSF calculations.
            Default 2 for coronagraphy and 4 otherwise.
        offset_r : float
            Radial offset from the center in arcsec.
        offset_theta : float
            Position angle for radial offset, in degrees CCW.
        bar_offset : float or None
            For bar masks, the position along the bar to place the PSF (arcsec).
            Use :func:`~pynrc.nrc_utils.offset_bar` for filter-dependent locations.
            If both :attr:`bar_offset` attribute and `bar_offset` keyword are None,
            then decide location based on selected filter. A positive value will
            move the source to the right when viewing V2 to the left and V3 up.
            Updates :attr:`bar_offset` attribute and coefficients appropriately.
        wfe_drift : float
            Wavefront error drift amplitude in nm.
            Updates :attr:`wfe_drift` attribute and coefficients appropriately.
        opd : tuple or HDUList
            Tuple (file, slice) or filename or HDUList specifying OPD.
        tel_pupil : str
            File name or HDUList specifying telescope entrance pupil.
        jitter : str or None
            Currently either 'gaussian' or None.
        jitter_sigma : float
            If ``jitter = 'gaussian'``, then this is the size of the blurring effect.
        save : bool
            Save the resulting PSF coefficients to a file? (default: True)
        force : bool
            Forces a recalcuation of PSF even if saved PSF exists. (default: False)

        """

        if oversample is None: 
            # Check if oversample has already been saved
            try: oversample = self._psf_info['oversample']
            except:
                oversample = 2 if 'LYOT' in self.pupil else 4

        # Default size is 33 pixels
        fov_default = 33
        # Use dictionary as case/switch statement
        pup_switch = {
            'WEAK LENS +4': 101,
            'WEAK LENS +8': 161,
            'WEAK LENS -8': 161,
            'WEAK LENS +12 (=4+8)': 221,
            'WEAK LENS -4 (=4-8)': 101,
            'GRISM0': 33,
            'GRISM90': 33,
            'CIRCLYOT': 33,
            'WEDGELYOT': 33,
        }
        # If fov_pix was not set, then choose here
        if fov_pix is None:
            try: 
                fov_pix = self._psf_info['fov_pix']
            except (AttributeError, KeyError): 
                fov_pix = pup_switch.get(self.pupil, fov_default)

        oversample = int(oversample)
        fov_pix = int(fov_pix)
        # Only want to throw this warning once based on self._fov_pix_warn status
        # Skip if weak lens or coronagraphic mask
        do_warn = True
        do_warn = False if  'WEAK' in self.pupil else do_warn
        do_warn = False if self.mask is not None else do_warn
        
        if (np.mod(fov_pix,2)==0) and (self._fov_pix_warn==True) and (do_warn): 
            _log.warning('fov_pix specified as even; PSF is centered at pixel corners.')
            self._fov_pix_warn = False
            
        _log.info('Updating PSF coeff with fov_pix={} and oversample={}'.\
            format(fov_pix,oversample))
    
        if offset_r is None:
            try: offset_r = self._psf_info['offset_r']
            except (AttributeError, KeyError): offset_r = 0
        if offset_theta is None:
            try: offset_theta = self._psf_info['offset_theta']
            except (AttributeError, KeyError): offset_theta = 0
        if tel_pupil is None:
            try: tel_pupil = self._psf_info['tel_pupil']
            except (AttributeError, KeyError): tel_pupil = None
        if jitter is None:
            try: jitter = self._psf_info['jitter']
            except (AttributeError, KeyError): jitter = None
        if jitter_sigma is None:
            try: jitter_sigma = self._psf_info['jitter_sigma']
            except (AttributeError, KeyError): jitter_sigma = 0.007
        if opd is None:
            try: opd = self._psf_info['opd']
            except (AttributeError, KeyError): opd = opd_default
        if save is None:
            try: save = self._psf_info['save']
            except (AttributeError, KeyError): save = True

        #print(opd)
        self._psf_info={'fov_pix':fov_pix, 'oversample':oversample, 
            'offset_r':offset_r, 'offset_theta':offset_theta, 
            'tel_pupil':tel_pupil, 'save':save, 'force':force,
            'opd':opd, 'jitter':jitter, 'jitter_sigma':jitter_sigma}
        self._psf_coeff = psf_coeff(self.bandpass, self.pupil, self.mask, self.module, 
            **self._psf_info)

        # WFE Drift is handled differently than the rest of the parameters
        # This is because we use wfed_coeff() to determine the resid values
        # for the PSF coefficients to generate a drifted PSF.
        if wfe_drift is not None:
            self._wfe_drift = wfe_drift
            
        wfe_drift = self._wfe_drift
        if wfe_drift>0:
            _log.info('Updating WFE drift ({}nm) for fov_pix={} and oversample={}'.\
                      format(wfe_drift,fov_pix,oversample))
            wfe_kwargs = dict(self._psf_info)
            wfe_kwargs['pupil']  = self.pupil
            wfe_kwargs['mask']   = self.mask
            wfe_kwargs['module'] = self.module
            #del wfe_kwargs['save'], wfe_kwargs['force']

            wfe_cf = wfed_coeff(self.bandpass, **wfe_kwargs)
            cf_fit = wfe_cf.reshape([wfe_cf.shape[0], -1])
            cf_mod = jl_poly(np.array([wfe_drift]), cf_fit)
            cf_mod = cf_mod.reshape(self._psf_coeff.shape)
            self._psf_coeff += cf_mod
            
        # Bar masks can have offsets
        if (self.mask is not None) and ('WB' in self.mask):
            r_bar, th_bar = offset_bar(self.filter, self.mask)
            # Want th_bar to be -90 so that r_bar matches webbpsf
            if th_bar>0: 
                r_bar  = -1 * r_bar
                th_bar = -1 * th_bar

            # Specifying bar_offset keyword overrides everything
            if bar_offset is not None:
                self._bar_offset = bar_offset
            # If _bar_offset attribute unspecified, then based on filter
            if self._bar_offset is None:
                self._bar_offset = r_bar

            bar_offset = self._bar_offset
            
            wedge_kwargs = dict(self._psf_info)
            wedge_kwargs['module'] = self.module
            
            wedge_cf = wedge_coeff(self.filter, self.pupil, self.mask, **wedge_kwargs)
            cf_fit = wedge_cf.reshape([wedge_cf.shape[0], -1])
            cf_mod = jl_poly(np.array([bar_offset]), cf_fit)
            cf_mod = cf_mod.reshape(self._psf_coeff.shape)
            self._psf_coeff += cf_mod
        else:
            self._bar_offset = 0
            
    
        # If there is a coronagraphic spot or bar, then we may need to
        # generate another background PSF for sensitivity information.
        # It's easiest just to ALWAYS do a small footprint without the
        # coronagraphic mask and save the PSF coefficients. 
        if self.mask is not None:
            self._psf_info_bg = {'fov_pix':self._fov_pix_bg, 'oversample':oversample, 
                'offset_r':0, 'offset_theta':0, 'tel_pupil':tel_pupil, 
                'opd':opd, 'jitter':None, 'save':True, 'force':False}
            self._psf_coeff_bg = psf_coeff(self.bandpass, self.pupil, None, self.module, 
                **self._psf_info_bg)

            # Update off-axis WFE drift
            if wfe_drift>0:
                wfe_kwargs = dict(self._psf_info_bg)
                wfe_kwargs['pupil']  = self.pupil
                wfe_kwargs['mask']   = None
                wfe_kwargs['module'] = self.module
                #del wfe_kwargs['save'], wfe_kwargs['force']

                wfe_cf = wfed_coeff(self.bandpass, **wfe_kwargs)
                cf_fit = wfe_cf.reshape([wfe_cf.shape[0], -1])
                cf_mod = jl_poly(np.array([wfe_drift]), cf_fit)
                cf_mod = cf_mod.reshape(self._psf_coeff_bg.shape)
                self._psf_coeff_bg += cf_mod

        else:
            self._psf_info_bg  = self._psf_info
            self._psf_coeff_bg = self._psf_coeff


    def sat_limits(self, sp=None, bp_lim=None, units='vegamag', well_frac=0.8,
        ngroup=None, verbose=False, **kwargs):
        """Saturation limits.        
        
        Generate the limiting magnitude (80% saturation) with the current instrument
        parameters (filter and ramp settings) assuming some spectrum. If no spectrum
        is defined, then a G2V star is assumed.

        The user can also define a separate bandpass in which to determine the
        limiting magnitude that will cause the current NIRCam bandpass to saturate.

        Parameters
        ----------
        sp : :mod:`pysynphot.spectrum`
            Spectrum to determine saturation limit.
        bp_lim : :mod:`pysynphot.obsbandpass`
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
        verbose : bool
            Print result details.

        Example
        -------
        >>> nrc = pynrc.NIRCam('F430M') # Initiate NIRCam observation
        >>> sp_A0V = pynrc.stellar_spectrum('A0V') # Define stellar spectral type
        >>> bp_k = S.ObsBandpass('steward,k') # Pysynphot K-Band bandpass
        >>> bp_k.name = 'K-Band'
        >>> mag_lim = nrc.sat_limits(sp_A0V, bp_k, verbose=True)
        
        Returns K-Band Limiting Magnitude for F430M assuming A0V source.
        """	

        if bp_lim is None: bp_lim = self.bandpass
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

        well_level = self.Detectors[0].well_level

        kwargs = merge_dicts(kwargs, self._psf_info)

        # We don't necessarily need the entire image, so cut down to size
        psf_coeff = self._psf_coeff
        if not ('WEAK LENS' in self.pupil):
            fov_pix = 51
            fov_pix_over = fov_pix * kwargs['oversample']
            coeff = []
            for im in psf_coeff:
                coeff.append(pad_or_cut_to_size(im, (fov_pix_over,fov_pix_over)))
            psf_coeff = np.array(coeff)
            kwargs['fov_pix'] = fov_pix

        satlim = sat_limit_webbpsf(self.bandpass, pupil=self.pupil, mask=self.mask,
            module=self.module, full_well=well_level, well_frac=well_frac,
            sp=sp, bp_lim=bp_lim, int_time=t_sat, quiet=quiet, units=units, 
            coeff=psf_coeff, **kwargs)

        return satlim

    def sensitivity(self, nsig=10, units=None, sp=None, verbose=False, **kwargs):
        """Sensitivity limits.
        
        Convenience function for returning the point source (and surface brightness)
        sensitivity for the given instrument setup. See bg_sensitivity() for more
        details.

        Parameters
        ----------
        sp : :mod:`pysynphot.spectrum`
            Input spectrum to use for determining sensitivity.
            Only the spectral shape matters, unless ``forwardSNR=True``.
        nsig : int, float
            Desired nsigma sensitivity (default 10).
        units : str
            Output units (defaults to uJy for grisms, nJy for imaging).
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

        quiet = False if verbose else True

        pix_scale = self.pix_scale
        well_level = self.well_level
        tf = self.multiaccum_times['t_frame']

        ktc = self.Detectors[0].ktc
        rn = self.Detectors[0].read_noise
        idark = self.Detectors[0].dark_current
        p_excess = self.Detectors[0].p_excess


        kw1 = self.multiaccum.to_dict()
        kw2 = self._psf_info_bg
        kw3 = {'rn':rn, 'ktc':ktc, 'idark':idark, 'p_excess':p_excess}
        kwargs = merge_dicts(kwargs,kw1,kw2,kw3)
        if 'ideal_Poisson' not in kwargs.keys():
            kwargs['ideal_Poisson'] = True
            
        # Always use the bg coeff
        psf_coeff = self._psf_coeff_bg
        # We don't necessarily need the entire image, so cut down to size
        if not ('WEAK LENS' in self.pupil):
            fov_pix = 33
            fov_pix_over = fov_pix * kwargs['oversample']
            coeff = []
            for im in psf_coeff:
                coeff.append(pad_or_cut_to_size(im, (fov_pix_over,fov_pix_over)))
            psf_coeff = np.array(coeff)
            kwargs['fov_pix'] = fov_pix

        bglim = bg_sensitivity(self.bandpass, self.pupil, self.mask, self.module,
            pix_scale=pix_scale, sp=sp, units=units, nsig=nsig, tf=tf, quiet=quiet, 
            coeff=psf_coeff, **kwargs)
    
        return bglim

    def bg_zodi(self, zfact=None, **kwargs):
        """Zodiacal background flux.
        
        There are options to query the IPAC Euclid Background Model
        (http://irsa.ipac.caltech.edu/applications/BackgroundModel/),
        but this method takes a while. Instead, if the keywords
        locstr, year, day are specified, this function will print
        information on an equivalent zfact value that produces the
        same flux within the bandpass. Values may be filter dependent.
        
        Returned values are in units of e-/sec/pixel

        Parameters
        ----------
        zfact : float
            Factor to scale Zodiacal spectrum (default 2.5)

        Keyword Args
        ------------
        locstr : 
            Object name or RA/DEC (decimal degrees or sexigesimal)
        year : int
            Year of observation
        day : float
            Day of observation

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
        if ('FLAT' in self.pupil):
            return 0

        bp = self.bandpass
        waveset   = bp.wave
        sp_zodi   = zodi_spec(zfact, **kwargs)
        obs_zodi  = S.Observation(sp_zodi, bp, waveset)
        fzodi_pix = obs_zodi.countrate() * (self.pix_scale/206265.0)**2
        
        # Recommend a zfact value if locstr, year, and day specified
        if 'locstr' in kwargs.keys():
            sp_zodi_temp   = zodi_spec(zfact=1)
            obs_zodi_temp  = S.Observation(sp_zodi_temp, bp, waveset)
            fzodi_pix_temp = obs_zodi_temp.countrate() * (self.pix_scale/206265.0)**2
            zf_rec = fzodi_pix / fzodi_pix_temp
            str1 = 'Using locstr,year,day keywords can be very slow. \n'
            str2 = 'For your specified loc and date, we recommend using zfact={:.1f}'\
                .format(zf_rec)
            _log.warning(str1)
            _log.warning(str2)

        # Don't forget about Lyot mask attenuation (not in bandpass throughput)
        if ('LYOT' in self.pupil):
            fzodi_pix *= 0.19

        return fzodi_pix
        
    def saturation_levels(self, sp, full_size=True, ngroup=2, **kwargs):
        """Saturation levels.
        
        Create image showing level of saturation for each pixel.
        Can either show the saturation after one frame (default)
        or after the ramp has finished integrating (ramp_sat=True).
        
        Parameters
        ----------
        sp : :mod:`pysynphot.spectrum`
            A pysynphot spectral object (normalized).
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
        
        """
        
        assert ngroup >= 0
        
        if (self.pupil is not None) and ('GRISM' in self.pupil): 
            is_grism = True
        else:
            is_grism = False

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
        image = self.gen_psf(sp)
        if is_grism: 
            wave, image = image
            
        if full_size:
            shape = (self.det_info['ypix'], self.det_info['xpix'])
            image = pad_or_cut_to_size(image, shape)
            
        # Add in zodi background to full image
        image += self.bg_zodi(**kwargs)

        # Well levels after "saturation time"
        sat_level = image * t_sat / self.well_level
    
        if is_grism:
            return (wave, sat_level)
        else:
            return sat_level


    def gen_psf(self, sp=None, return_oversample=False, use_bg_psf=False, **kwargs):
        """PSF image.
        
        Create a PSF image from instrument settings. The image is noiseless and
        doesn't take into account any non-linearity or saturation effects, but is
        convolved with the instrument throughput. Pixel values are in counts/sec.
        The result is effectively an idealized slope image (no background).

        If no spectral dispersers (grisms or DHS), then this returns a single
        image or list of images if sp is a list of spectra. By default, it returns
        only the detector-sampled PSF, but setting return_oversample=True will
        also return a set of oversampled images as a second output.

        Parameters
        ----------
        sp : :mod:`pysynphot.spectrum`
            If not specified, the default is flat in phot lam 
            (equal number of photons per spectral bin).
            The default is normalized to produce 1 count/sec within that bandpass,
            assuming the telescope collecting area and instrument bandpass. 
            Coronagraphic PSFs will further decrease this due to the smaller pupil
            size and coronagraphic spot. 
        return_oversample : bool
            If True, then also returns the oversampled version of the PSF
        use_bg_psf : bool
            If a coronagraphic observation, off-center PSF is different.

        """

        if use_bg_psf:
            psf_coeff = self._psf_coeff_bg
            psf_info  = self._psf_info_bg
        else:
            psf_coeff = self._psf_coeff
            psf_info  = self._psf_info

        return gen_image_coeff(self.bandpass, sp_norm=sp, 
            pupil=self.pupil, mask=self.mask, module=self.module, 
            coeff=psf_coeff, fov_pix=psf_info['fov_pix'], 
            oversample=psf_info['oversample'], 
            return_oversample=return_oversample, **kwargs)
    

    def gen_exposures(self, sp=None, im_slope=None, file_out=None, return_results=None,
                      targ_name=None, timeFileNames=False, DMS=True,
                      dark=True, bias=True, nproc=None, **kwargs):
        """Generate raw mock data.
        
        Create a series of ramp integration saved to FITS files based on
        the current NIRCam settings. This method calls the :func:`gen_fits`
        function, which in turn calls the detector noise generator 
        :mod:`~pynrc.simul.ngNRC`

        Currently, this image simulator does NOT take into account:
        
            - QE variations across a pixel's surface
            - Intrapixel Capacitance (IPC)
            - Post-pixel Coupling (PPC) due to ADC "smearing"
            - Pixel non-linearity
            - Persistence/latent image
            - Optical distortions
            - Zodiacal background roll off for grism edges
            - Cosmic Rays
            
        To Do: Double-check the output for grism data w.r.t V2V3_to_det().


        Parameters
        ----------
        im_slope : numpy array, None
            Pass the slope image directly. If not set, then a slope
            image will be created from the input spectrum keyword. This
            should include zodiacal light emission, but not dark current.
            Make sure this array is in detector coordinates.
        sp : :mod:`pysynphot.spectrum`, None
            A pysynphot spectral object. If not specified, then it is
            assumed that we're looking at blank sky.
        targ_name : str, None
            A target name for the exposure file's header.
        file_out : str, None
            Path and name of output FITs files. Time stamps will
            be automatically inserted for unique file names if
            ``timeFileNames=True``.
        timeFileNames : bool
            Save the exposure times in the file name? This is useful to see 
            the timing, but also makes it a little harder to combine INTs 
            later for DMS simulations.
        DMS : bool
            Create DMS data file and header format? Otherwise, the output is
            more similar to ISIM CV2/3 and OTIS campaigns.
        return_results : bool, None
            By default, we return results if file_out is not set and
            the results are not returned if file_out is set. This decision
            is based on the large amount of data and memory usage this 
            method may incur if the results were always returned by default.
            Instead, it's better to save the FITs to disk, especially if 
            NINTs is large. We include the return_results keyword if the 
            user would like to do both (or neither).
        dark : bool
            Include the dark current?
        bias : bool
            Include the bias frame?

        Keyword Args
        ------------
        zfact : float
            Factor to scale Zodiacal spectrum (default 2.5)
        locstr : 
            Object name or RA/DEC (decimal degrees or sexigesimal)
        year : int
            Year of observation
        day : float
            Day of observation

        """

        det = self.Detectors[0]
        filter = self.filter
        pupil = self.pupil
        xpix = self.det_info['xpix']
        ypix = self.det_info['ypix']

        # If slope image is not specified
        if im_slope is None:
            # No visible source
            if ('FLAT' in pupil) or (sp is None):
                im_slope = np.zeros([ypix,xpix])
            # Grism spec
            elif ('GRISM' in pupil):
                w, im_slope = self.gen_psf(sp)
            # DHS spectroscopy
            elif ('DHS' in pupil):
                raise NotImplementedError('DHS has yet to be fully included')
            # Imaging+Coronagraphy
            else:
                im_slope = self.gen_psf(sp)
    
            # Add in Zodi emission
            # Returns 0 if self.pupil='FLAT'
            im_slope += self.bg_zodi(**kwargs)
            
            targ_name = sp.name if targ_name is None else targ_name
            
            # Image coordinates have +V3 up and +V2 to left
            # Want to convert to detector coordinates
            # Need to double-check the output for grism data.
            #im_slope = V2V3_to_det(im_slope, det.detid)

        # Minimum value of slope
        im_min = im_slope[im_slope>=0].min()
        # Expand or cut to detector size
        im_slope = pad_or_cut_to_size(im_slope, (ypix,xpix))
        # Make sure there are no negative numbers
        im_slope[im_slope<=0] = im_min

        # Create times indicating start of new ramp
        t0 = datetime.datetime.now()
        if timeFileNames == True:
            dt = self.multiaccum_times['t_int_tot']
        else:
            dt = 0.
        nint = self.det_info['nint']
        time_list = [t0 + datetime.timedelta(seconds=i*dt) for i in range(nint)]

        # Create list of file names for each INT
        if file_out is None:
            if return_results is None: return_results=True
            file_list = [None]*nint
        else:
            if return_results is None: return_results=False
            file_list = []
            #file_out = '/Volumes/NIRData/grism_sim/grism_sim.fits'
            if file_out.lower()[-5:] == '.fits':
                file_out = file_out[:-5]
            if file_out[-1:] == '_':
                file_out = file_out[:-1]

            for fileInd, t in enumerate(time_list):
                file_time = t.isoformat()[:-7]
                file_time = file_time.replace(':', 'h', 1)
                file_time = file_time.replace(':', 'm', 1)
                file_list.append(file_out + '_' + file_time + "_{0:04d}".format(fileInd) + '.fits')

        # Create a list of arguments to pass
        # For now, we're only doing the first detector. This will need to get more
        # sophisticated for SW FPAs
        worker_arguments = [(det, im_slope, True, fout, filter, pupil, otime, \
                             targ_name, DMS, dark, bias, return_results) \
                            for fout,otime in zip(file_list, time_list)]

        nproc = nproc_use_ng(det) if nproc is None else nproc
        if nproc<=1:
            #map(gen_fits, worker_arguments)
            res = [gen_fits(wa) for wa in worker_arguments]
        else:
            pool = mp.Pool(nproc)
            try:
                res = pool.map(gen_fits, worker_arguments)
            except Exception as e:
                print('Caught an exception during multiprocess:')
                raise e
            finally:
                pool.close()
        
        if return_results: return res


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
        sp : :mod:`pysynphot.spectrum`
            A pysynphot spectral object to calculate SNR.
        sp_bright : :mod:`pysynphot.spectrum`, None
            Same as sp, but optionally used to calculate the saturation limit
            (treated as brightest source in field). If a coronagraphic mask 
            observation, then this source is assumed to be occulted and 
            sp is fully unocculted.
        is_extended : bool
            Treat source(s) as extended objects, then in units/arcsec^2

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
        locstr : 
            Object name or RA/DEC (decimal degrees or sexigesimal)
        year : int
            Year of observation
        day : float
            Day of observation

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
        The keyword arguments locstr, year, day are not recommended for use 
        given the amount of time it takes to query the Euclid web server. 
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


        pupil = self.pupil
        grism_obs = (pupil is not None) and ('GRISM' in pupil)
        dhs_obs   = (pupil is not None) and ('DHS'   in pupil)
        coron_obs = (pupil is not None) and ('LYOT'  in pupil)

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

        # Generate PSFs for faint and bright objects and get max pixel flux
        # Only necessary for point sources
        if is_extended:
            ind_snr = 1
            obs = S.Observation(sp, self.bandpass, binset=self.bandpass.wave)
            pix_count_rate = obs.countrate() * self.pix_scale**2
        else:
            ind_snr = 0

            if grism_obs:
                _, psf_bright = self.gen_psf(sp_bright, use_bg_psf=False)
                _, psf_faint  = self.gen_psf(sp, use_bg_psf=True)
            else:
                psf_bright = self.gen_psf(sp_bright, use_bg_psf=False)
                psf_faint  = self.gen_psf(sp, use_bg_psf=True)
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

        rows = []
        if tacq_max is not None:
            snr_goal_list = []
            # Cycle through each readout pattern
            for read_mode in patterns:
                if verbose: print(read_mode)

                # Maximum allowed groups for given readout pattern
                _,_,ngroup_max = pattern_settings.get(read_mode)
                if ng_max is not None:
                    ngroup_max = ng_max
                nng = ngroup_max - ng_min + 1
                if nng>30:
                    _log.warning('Cycling through {} NGRPs. This may take a while!'\
                        .format(nng))
                for ng in range(ng_min,ngroup_max+1):
                    self.update_detectors(read_mode=read_mode, ngroup=ng, nint=1)
                    mtimes = self.multiaccum_times

                    # Get saturation level of observation
                    # Total time spent integrating minus the reset frame
                    int_time = mtimes['t_int']

                    well_frac = pix_count_rate * int_time / self.well_level
                    # If above well_frac_max, then this setting is invalid
                    if well_frac > well_frac_max:
                        continue
            
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
                        if nint > nint_max: break
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
                    _log.warning('Cycling through {} NGRPs. This may take a while!'\
                        .format(nng))
                        
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
        t_all['SNR'].format = '8.1f'
        t_all['Well'].format = '8.3f'

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



def _check_list(value, temp_list, var_name=None):
    """
    Helper function to test if a value exists within a list. 
    If not, then raise ValueError exception.
    This is mainly used for limiting the allowed values of some variable.
    """
    if value not in temp_list:
        # Replace None value with string for printing
        if None in temp_list: temp_list[temp_list.index(None)] = 'None'
        var_name = '' if var_name is None else var_name + ' '
        err_str = "Invalid {}setting: {} \n\tValid values are: {}" \
                         .format(var_name, value, ', '.join(temp_list))
        raise ValueError(err_str)

def tuples_to_dict(pairs, verbose=False):
    """
    Take a list of paired tuples and convert to a dictionary
    where the first element of each tuple is the key and the 
    second element is the value.
    
    Args
    ----
    pairs : list
        List of tuples - [(a1,a2), (b1,b2), (c1,c2), ...]
    
    Returns
    -------
    dict
    """
    d={}
    for (k, v) in pairs:
        d[k] = v
        if verbose:
            if isinstance(v,float): print("{:<10} {:>10.4f}".format(k, v))
            else: print("{:<10} {:>10}".format(k, v))
    return d

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


def gen_fits(args):
    """
    Helper function for generating FITs integrations from a slope image
    """
    from .simul.ngNRC import slope_to_ramp

    # Must call np.random.seed() for multiprocessing, otherwise 
    # random numbers for parallel processes start in the same seed state!
    np.random.seed()
    try:
        res = slope_to_ramp(*args)
    except Exception as e:
        print('Caught exception in worker thread:')
        # This prints the type, value, and stack trace of the
        # current exception being handled.
        traceback.print_exc()

        print()
        raise e

    return res

def nproc_use_ng(det):
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
    nint    = ma.nint
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
