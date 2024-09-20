from __future__ import print_function, division
import numpy as np

from astropy.io import fits
from astropy.time import Time
import datetime, time

import logging
_log = logging.getLogger('pynrc')

from . import conf
from .logging_utils import setup_logging


from webbpsf_ext.webbpsf_ext_core import _check_list
from .nrc_utils import pix_noise



class multiaccum(object):
    """
    A class for defining MULTIACCUM ramp settings.
    See `NIRCam MULTIACCUM documentation
    <https://jwst-docs.stsci.edu/display/JTI/NIRCam+Detector+Readout+Patterns>`_
    for more details.

    Parameters
    ----------------
    read_mode : str
        NIRCam Ramp Readout mode such as 'RAPID', 'BRIGHT1', 'DEEP8', etc., or 'CUSTOM'
    nint : int
        Number of integrations (ramps).
    ngroup : int
        Number of groups in a integration.
    nf : int
        Number of frames per group.
    nd1 : int
        Number of drop frame after reset (before first group read). Default=0.
    nd2 : int
        Number of drop frames within a group (ie., groupgap). 
    nd3 : int
        Number of drop frames after final read frame in ramp. Default=1.
    nr1 : int
        Number of reset frames within first ramp. Default=0.
    nr2 : int
        Number of reset frames for subsequent ramps. Default=1.
    wind_mode : str
        Set to determine maximum number of allowed groups.

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
                 nr1=1, nr2=1, wind_mode='FULL', **kwargs):

        self.nint = nint
        self._ngroup_max = 10000
        self.ngroup = ngroup

        # Modify these directly rather via the @property
        self._nr1 = self._check_int('nr1',nr1,0)
        self._nr2 = self._check_int('nr2',nr2,0)
        self._nf = self._check_int('nf',nf,1)
        self._nd1 = self._check_int('nd1',nd1,0)
        self._nd2 = self._check_int('nd2',nd2,0)
        self._nd3 = self._check_int('nd3',nd3,0)
        # Now set read mode to specified mode, which may modify nf, nd1, nd2, and nd3
        self.read_mode = read_mode

    @property
    def nint(self):
        """Number of ramps (integrations) in an exposure."""
        return self._nint
    @nint.setter
    def nint(self, value):
        self._nint = self._check_int('nint',value,1)

    @property
    def ngroup(self):
        """Number of groups in a ramp (integration)."""
        return self._ngroup
    @ngroup.setter
    def ngroup(self, value):
        value = self._check_int('ngroup',value,1)
        if value > self._ngroup_max:
            _log.warning('Specified ngroup ({}) greater than allowed value ({})'\
                         .format(value, self._ngroup_max))
            _log.warning('Setting ngroup = {}'.format(self._ngroup_max))
            value = self._ngroup_max
        self._ngroup = value

    @property
    def nf(self):
        """Number of frames per group."""
        return self._nf
    @nf.setter
    def nf(self, value):
        value = self._check_int('nf',value,1)
        self._nf = self._check_custom(value, self._nf)

    @property
    def nd1(self):
        """Number of drop frame after reset (before first group read)."""
        return self._nd1
    @nd1.setter
    def nd1(self, value):
        value = self._check_int('nd1',value,0)
        self._nd1 = self._check_custom(value, self._nd1)

    @property
    def nd2(self):
        """Number of drop frames within a group (aka, groupgap)."""
        return self._nd2
    @nd2.setter
    def nd2(self, value):
        value = self._check_int('nd2',value,0)
        self._nd2 = self._check_custom(value, self._nd2)

    @property
    def nd3(self):
        """Number of drop frames after final read frame in ramp."""
        return self._nd3
    @nd3.setter
    def nd3(self, value):
        value = self._check_int('nd3',value,0)
        self._nd3 = self._check_custom(value, self._nd3)

    @property
    def nr1(self):
        """Number of reset frames before first integration."""
        return self._nr1
    @nr1.setter
    def nr1(self, value):
        self._nr1 = self._check_int('nr1', value, minval=0)

    @property
    def nr2(self):
        """Number of reset frames for subsequent integrations."""
        return self._nr2
    @nr2.setter
    def nr2(self, value):
        self._nr2 = self._check_int('nr2', value, minval=0)

    @property
    def nread_tot(self):
        """Total number of read frames in a ramp, including drops"""
        nf = self.nf; nd1 = self.nd1; nd2 = self.nd2; nd3 = self.nd3
        ngroup = self.ngroup

        nframes = nd1 + ngroup*nf + (ngroup-1)*nd2 + nd3
        return nframes

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

    @property
    def _pattern_settings(self):
        patterns = ['RAPID', 'BRIGHT1', 'BRIGHT2', 'SHALLOW2', 'SHALLOW4', 'MEDIUM2', 'MEDIUM8', 'DEEP2', 'DEEP8']
        nf_arr   = [1,1,2,2,4,2,8, 2, 8]
        nd2_arr  = [0,1,0,3,1,8,2,18,12]
        ng_max   = [10,10,10,10,10,10,10,20,20]

        return dict(zip(patterns, zip(nf_arr, nd2_arr, ng_max)))

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
            _log.warning('Readout {} not a valid NIRCam readout mode. Setting to CUSTOM.'\
                         .format(self.read_mode))
            self._read_mode = 'CUSTOM'
            _log.warning('Using explicit settings: ngroup={}, nf={}, nd1={}, nd2={}, nd3={}'\
                         .format(self.ngroup, self.nf, self.nd1, self.nd2, self.nd3))
        elif self.read_mode == 'CUSTOM':
            _log.info('{} readout mode selected.'.format(self.read_mode))
            _log.info('Using explicit settings: ngroup={}, nf={}, nd1={}, nd2={}, nd3={}'\
                      .format(self.ngroup, self.nf, self.nd1, self.nd2, self.nd3))
        else:
            _log.info('{} readout mode selected.'.format(self.read_mode))
            nf, nd2, _ = self._pattern_settings.get(self.read_mode)
            self._nf  = nf
            self._nd1 = 0
            self._nd2 = nd2
            self._nd3 = 0
            _log.info('Setting ngroup={}, nf={}, nd1={}, nd2={}, nd3={}.'\
                     .format(self.ngroup, self.nf, self.nd1, self.nd2, self.nd3))


    def _check_custom(self, val_new, val_orig):
        """Check if read_mode='CUSTOM' before changing variable."""
        if self.read_mode == 'CUSTOM': 
            return val_new
        else: 
            _log.warning("Can only modify parameter if read_mode='CUSTOM'.")
            return val_orig

    def _check_int(self, pstr, val, minval=1):
        """Check if a value is a positive integer, otherwise throw exception."""
        val = float(val)
        if (val.is_integer()) and (val>=minval): 
            return int(val)
        else:
            raise ValueError("{}={} must be an integer >={}.".format(pstr,val,minval))


    
class det_timing(object):
    """ 
    Class to hold detector operations information. Includes SCA attributes such as
    detector names and IDs as well as :class:`multiaccum` class for ramp settings.

    Parameters
    ----------------
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
        >>> d = det_timing(**kwargs)
    
    Set the keywords directly:   
        >>> d = det_timing(read_mode='RAPID', nint=5, ngroup=10)
    """

    def __init__(self, wind_mode='FULL', xpix=2048, ypix=2048, x0=0, y0=0, 
                 mode='SLOW', nff=None, **kwargs):

        self._opmode = mode.upper()
        
        if 'JWST' in self._opmode:
            pixrate = 1e5
            loh = 12
            nchans = 4
            reset_type = 'pixel'
        elif 'SLOW' in self._opmode:
            pixrate = 1e5
            loh = 8
            nchans = 32
            reset_type = 'pixel'
            if nff is None: nff = 0 
        elif 'MED' in self._opmode:
            pixrate = 3e5
            loh = 8
            nchans = 32
            reset_type = 'pixel'
            if nff is None: nff = 0 
        elif 'FAST' in self._opmode:
            pixrate = 5e6
            loh = 3
            nchans = 32
            reset_type = 'line'
            if nff is None: nff = 0 

        self._detector_pixels = 2048
        self._nchans = nchans
        self._nff = nff

        self.multiaccum = multiaccum(wind_mode=wind_mode, **kwargs)
        self.wind_mode = wind_mode.upper()
        self._xpix = xpix; self._x0 = x0
        self._ypix = ypix; self._y0 = y0

        self._validate_pixel_settings()

        # By default fast-scan readout direction is [-->,<--,-->,<--]
        # If same_scan_direction, then all --> 
        # If reverse_scan_direction, then [<--,-->,<--,-->] or all <--
        self.same_scan_direction = False
        self.reverse_scan_direction = False

        # Pixel Rate in Hz
        self._pixel_rate = pixrate
        # Number of extra clock ticks per line
        self._line_overhead = loh
        # Pixel or line resets
        self._reset_type = reset_type

    @property
    def wind_mode(self):
        """Window mode attribute"""
        return self._wind_mode
    @wind_mode.setter
    def wind_mode(self, value):
        """Set Window mode attribute"""
        self._wind_mode = value

    @property
    def y0(self):
        return int(self._y0)
    @y0.setter
    def y0(self, val):
        self._y0 = val
    @property
    def x0(self):
        return int(self._x0)
    @x0.setter
    def x0(self, val):
        self._x0 = val
    @property
    def ypix(self):
        return int(self._ypix)
    @ypix.setter
    def ypix(self, val):
        self._ypix = val
    @property
    def xpix(self):
        return int(self._xpix)
    @xpix.setter
    def xpix(self, val):
        self._xpix = val

    @property
    def nout(self):
        """Number of simultaneous detector output channels stripes"""
        return 1 if self.wind_mode == 'WINDOW' else self._nchans

    @property
    def chsize(self):
        """Size of Amplifier Channel"""
        return int(self.xpix // self.nout)

    @property
    def ref_info(self):
        """Array of reference pixel borders [lower, upper, left, right]."""
        det_size = self._detector_pixels
        x1 = self.x0; x2 = x1 + self.xpix
        y1 = self.y0; y2 = y1 + self.ypix

        w = 4 # Width of ref pixel border
        lower = int(w-y1)
        upper = int(w-(det_size-y2))
        left  = int(w-x1)
        right = int(w-(det_size-x2))
        ref_all = np.array([lower,upper,left,right], dtype='int')
        ref_all[ref_all<0] = 0
        return ref_all

    @property
    def mask_act(self):
        """Active pixel mask for det coordinates"""
        # mask_act = np.zeros([self.ypix,self.xpix]).astype('bool')
        # rb, rt, rl, rr = self.ref_info
        # mask_act[rb:-rt,rl:-rr] = True  # This doesn't work if rr or rt are 0!!
        return ~self.mask_ref
    @property
    def mask_ref(self):
        """Reference pixel mask for det coordinates"""
        # [bottom, upper, left, right]
        rb, rt, rl, rr = self.ref_info
        ref_mask = np.zeros([self.ypix,self.xpix], dtype=bool)
        if rb>0: ref_mask[0:rb,:] = True
        if rt>0: ref_mask[-rt:,:] = True
        if rl>0: ref_mask[:,0:rl] = True
        if rr>0: ref_mask[:,-rr:] = True
        return ref_mask
    @property
    def mask_channels(self):
        """Channel masks for det coordinates"""
        ch_mask = np.zeros([self.ypix,self.xpix])
        for ch in np.arange(self.nout):
            ch_mask[:,ch*self.chsize:(ch+1)*self.chsize] = ch
        return ch_mask

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
            wstr = str.join(', ', modes)
            raise ValueError("{} not a valid readout mode. Acceptable values: {}".format(wind_mode, wstr))

        detpix = self._detector_pixels
        xpix = self.xpix; x0 = self.x0
        ypix = self.ypix; y0 = self.y0

        # Check some consistencies with frame sizes
        if wind_mode == 'FULL':
            if ypix != detpix:
                _log.warning(f'In {wind_mode} mode, but ypix not {detpix}. Setting ypix={detpix}.')
                ypix = detpix
            if y0 != 0:
                _log.warning(f'In {wind_mode} mode, but x0 not 0. Setting y0=0.')
                y0 = 0

        if (wind_mode == 'STRIPE') or (wind_mode == 'FULL'):
            if xpix != detpix:
                _log.warning(f'In {wind_mode} mode, but xpix not {detpix}. Setting xpix={detpix}.')
                xpix = detpix
            if x0 != 0:
                _log.warning(f'In {wind_mode} mode, but x0 not 0. Setting x0=0.')
                x0 = 0
    
        if (x0+xpix) > detpix:
            raise ValueError("x0+xpix ({}+{}) is larger than detector size ({})!"\
                             .format(x0,xpix,detpix))
        if (y0+ypix) > detpix:
            raise ValueError("y0+ypix ({}+{}) is larger than detector size ({})!"\
                             .format(y0,ypix,detpix))

        # Update values if no errors were thrown
        self._xpix = xpix; self._x0 = x0
        self._ypix = ypix; self._y0 = y0

    def _fix_precision(self, input):
        """
        Many timing calculations result from minor precision issues with very
        small numbers (1e-16) added to the real result. This function attempts
        to truncate these small innaccuracies by dividing by the clock sample
        time to get the total integer number of clock cycles.
        """
        return int(input * self._pixel_rate + 0.5) / self._pixel_rate

    @property
    def _extra_lines(self):
        """Number of extra lines per frame.
        
        Determine how many extra lines/rows are added to a to a given frame.
        Based on ASIC microcode 10 and NIRCam arrays. Other instruments might
        operate differently.
        """
        if self.nout == 1:
            xtra_lines = 2 if self.xpix>10 else 3
        else:
            xtra_lines = 1
            
        return xtra_lines
        
    @property
    def _exp_delay(self):
        """Transition to idle delay (sec)
        
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
        else: 
            xtra_lines = 1
        
        # Clock ticks per line
        xticks = self.chsize + self._line_overhead
        
        # Return number of seconds
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
        """NFF Row Resets time per integration"""
        
        nff = self.nff
            
        xtra_lines = int(nff / (self.chsize))
        
        # Clock ticks per line
        xticks = self.chsize + self._line_overhead  
        return xticks * xtra_lines / self._pixel_rate
        
    @property
    def time_frame(self):
        """Determine frame time (sec) based on xpix, ypix, and wind_mode."""

        chsize = self.chsize                   # Number of x-pixels within a channel
        xticks = chsize + self._line_overhead  # Clock ticks per line
        flines = self.ypix + self._extra_lines # Lines per frame

        # Add a single pix offset for full frame and stripe.
        pix_offset = self._frame_overhead_pix
        #end_delay = 0 # Used for syncing each frame w/ FPE bg activity. Not currently used.

        # Total number of clock ticks per frame (reset, read, and drops)
        fticks = xticks*flines + pix_offset

        # Return frame time
        return fticks / self._pixel_rate

    @property
    def time_group(self):
        """Time per group based on time_frame, nf, and nd2."""
        return self.time_frame * (self.multiaccum.nf + self.multiaccum.nd2)

    @property
    def time_ramp(self):
        """Photon collection time for a single ramp."""

        # How many total frames (incl. dropped and all) per ramp?
        # Exclude nd3 (drops that add nothing)
        ma = self.multiaccum
        nf = ma.nf; nd1 = ma.nd1; nd2 = ma.nd2
        ngroup = ma.ngroup

        tint = (nd1 + ngroup*nf + (ngroup-1)*nd2) * self.time_frame
        return self._fix_precision(tint)

    @property
    def time_int(self):
        """Same as time_ramp, except that 'int' follows the JWST nomenclature"""
        return self.time_ramp

    @property
    def time_ramp_eff(self):
        """Effective ramp time for slope fit tf*(ng-1)"""
        
        ma = self.multiaccum
        if ma.ngroup<=1:
            res = self.time_frame * (ma.nd1 + (ma.nf + 1) / 2)
        else:
            res = self.time_group * (ma.ngroup - 1)

        return self._fix_precision(res)

    @property
    def time_int_eff(self):
        """Same as time_ramp_eff, except that 'int' follows the JWST nomenclature"""
        return self.time_ramp_eff

    @property
    def time_exp(self):
        """Total photon collection time for all ramps."""
        res = self.multiaccum.nint * self.time_ramp
        return self._fix_precision(res)

    # @property
    # def time_total_int(self):
    #     """Total time for all frames in a ramp.
        
    #     Includes resets and excess drops, as well as NFF Rows Reset.
    #     """

    #     ma = self.multiaccum
    #     nf = ma.nf; nd1 = ma.nd1; nd2 = ma.nd2; nd3 = ma.nd3
    #     ngroup = ma.ngroup
    #     nr = 1

    #     nframes = nr + nd1 + ngroup*nf + (ngroup-1)*nd2 + nd3        
    #     return nframes * self.time_frame + self.time_row_reset

    @property
    def time_total_int1(self):
        """Total time for all frames in first ramp of exposure.
        
        Includes resets and excess drops, as well as NFF Rows Reset.
        """

        ma = self.multiaccum
        nf = ma.nf; nd1 = ma.nd1; nd2 = ma.nd2; nd3 = ma.nd3
        ngroup = ma.ngroup
        nr = ma.nr1

        nframes = nr + nd1 + ngroup*nf + (ngroup-1)*nd2 + nd3        
        res = nframes * self.time_frame + self.time_row_reset
        return self._fix_precision(res)

    @property
    def time_total_int2(self):
        """Total time for all frames in a subsequent ramp.
        
        Includes resets and excess drops, as well as NFF Rows Reset.
        Only differs from time_total_int1 in case nr1 != nr2
        """

        ma = self.multiaccum
        nf = ma.nf; nd1 = ma.nd1; nd2 = ma.nd2; nd3 = ma.nd3
        ngroup = ma.ngroup
        nr = ma.nr2
        
        # Set to 0 if only a single integration
        if ma.nint <= 1:
            return 0.
        else:
            nframes = nr + nd1 + ngroup*nf + (ngroup-1)*nd2 + nd3        
            res = nframes * self.time_frame + self.time_row_reset
            return self._fix_precision(res)

    @property
    def time_total(self):
        """Total exposure acquisition time"""
        # exp1 = 0 if self.multiaccum.nint == 0 else self.time_total_int1
        # exp2 = 0 if self.multiaccum.nint <= 1 else self.time_total_int2 * (self.multiaccum.nint-1)
        exp1 = self.time_total_int1
        exp2 = self.time_total_int2 * (self.multiaccum.nint-1)
        res = exp1 + exp2 + self._exp_delay
        return self._fix_precision(res)

    @property
    def times_group_avg(self):
        """Times at each averaged group since reset"""
        ma = self.multiaccum
        nf_avg = np.arange(ma.nf+1).sum() / ma.nf
        return np.arange(ma.ngroup) * self.time_group + (ma.nd1 + nf_avg) * self.time_frame

    def to_dict(self, verbose=False):
        """Export detector settings to a dictionary."""

        p = [('wind_mode',self.wind_mode), ('nout',self.nout), 
             ('xpix',self.xpix), ('ypix',self.ypix), ('x0',self.x0), ('y0',self.y0)]
        return tuples_to_dict(p, verbose)

    def times_to_dict(self, verbose=False):
        """Export ramp times as dictionary with option to print output to terminal."""

        times = [('t_frame', self.time_frame), ('t_group', self.time_group), 
                 ('t_int',   self.time_int),   ('t_exp',   self.time_exp), 
                 ('t_acq',   self.time_total), 
                 ('t_int_tot1', self.time_total_int1), 
                 ('t_int_tot2', self.time_total_int2)]
        return tuples_to_dict(times, verbose)

    def int_times_table(self, date_start, time_start, offset_seconds=None):
        """Create and populate the INT_TIMES table, which is saved as a
        separate extension in the output data file.

        Parameters
        ----------
        date_start : str
            Date string of observation ('2020-02-28')
        time_start : str
            Time string of observation ('12:24:56')
        offset_seconds : None or float
            Time from beginning of observation until start of integration.

        Returns
        -------
        int_times_tab : astropy.table.Table
            Table of starting, mid, and end times for each integration
        """
        
        from astropy.table import Table
        from astropy.time import Time, TimeDelta
        from astropy import units as u

        if offset_seconds is None:
            offset_seconds = 0

        integration_numbers = np.arange(self.multiaccum.nint)

        start_time_string = date_start + 'T' + time_start
        start_time = Time(start_time_string) + offset_seconds * u.second

        integration_time = self.time_total_int2
        integ_time_delta = TimeDelta(integration_time * u.second)
        start_times = start_time + (integ_time_delta * integration_numbers)

        reset_time = self.multiaccum.nr2 * self.time_frame
        integration_time_exclude_reset = TimeDelta((integration_time - reset_time) * u.second)
        end_times = start_times + integration_time_exclude_reset

        mid_times = start_times + integration_time_exclude_reset / 2.

        # For now, let's keep the BJD (Barycentric?) times identical
        # to the MJD times.
        start_times_bjd = start_times
        mid_times_bjd = mid_times
        end_times_bjd = end_times

        # Create table
        nrows = len(integration_numbers)
        data_list = [(integration_numbers[i] + 1, 
                      start_times.mjd[i], mid_times.mjd[i], end_times.mjd[i],
                      start_times_bjd.mjd[i], mid_times_bjd.mjd[i], end_times_bjd.mjd[i]) 
                     for i in range(nrows)]

        int_times_tab = np.array(data_list,
                                 dtype=[('integration_number','<i2'),
                                        ('int_start_MJD_UTC','<f8'),
                                        ('int_mid_MJD_UTC', '<f8'),
                                        ('int_end_MJD_UTC','<f8'),
                                        ('int_start_BJD_TDB','<f8'),
                                        ('int_mid_BJD_TDB','<f8'),
                                        ('int_end_BJD_TDB','<f8')])

        return int_times_tab
        
    def pixel_noise(self, ng=None, nf=None, verbose=False, **kwargs):
        """Noise values per pixel.
        
        Return theoretical noise calculation for the specified MULTIACCUM exposure 
        in terms of e-/sec. This uses the pre-defined detector-specific noise 
        properties. Can specify flux of a source as well as background and 
        zodiacal light (in e-/sec/pix). After getting the noise per pixel per
        ramp (integration), value(s) are divided by the sqrt(NINT) to return
        the final noise

        Parameters
        ----------
        ng : None or int or image
            Option to explicitly state number of groups. This is specifically
            used to enable the ability of only calculating pixel noise for
            unsaturated groups for each pixel. If a numpy array, then it should
            be the same shape as `fsrc` image. By default will use `self.multiaccum.ngroup`.
        nf : int
            Option to explicitly states number of frames in each group.
            By default will use `self.multiaccum.nf`.
        verbose : bool
            Print out results at the end.

        Keyword Arguments
        -----------------
        rn : float
            Read Noise per pixel (e-).
        ktc : float
            kTC noise (in e-). Only valid for single frame (n=1)
        p_excess : array-like
            An array or list of two elements that holds the parameters
            describing the excess variance observed in effective noise plots.
            By default these are both 0. For NIRCam detectors, recommended
            values are [1.0,5.0] for SW and [1.5,10.0] for LW.
        idark : float
            Dark current in e-/sec/pix.
        fsrc : float
            Flux of source in e-/sec/pix.
        fzodi : float
            Zodiacal light emission in e-/sec/pix.
        fbg : float
            Any additional background (telescope emission or scattered light?)
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

        # Pixel noise per ramp (e-/sec/pix)
        pn = pix_noise(ngroup=ng, nf=nf, nd2=ma.nd2, tf=self.time_frame, **kwargs)
    
        # Divide by sqrt(Total Integrations)
        final = pn / np.sqrt(ma.nint)
        if verbose:
            print('Noise (e-/sec/pix): {}'.format(final))
            print('Total Noise (e-/pix): {}'.format(final*self.time_exp))

        return final

    def pix_timing_map(self, same_scan_direction=None, reverse_scan_direction=None,
                       avg_groups=False, reset_zero=False, return_flat=False):
        """Create array of pixel times for a single ramp. 
        
        Each pixel value corresponds to the precise time at which
        that pixel was read out during the ramp acquisition. The first
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
            pixel times in a similar manner. Default is False.
        return_flat : bool
            Return a single array rather than image array.
        
        Keyword Args
        ------------
        reset_zero : bool
            Return timing relative to when reset to get photon-collection time of each pixel.
            Otherwise, t=0 corresponds to very first pixel(s) read in the ramp.
            
        Returns
        -------
        ndarray
            If ``return_flat=True`` then the data is a flattened array for a
            single channel output. Otherwise, the output is a data cube of the
            same size and shape of the raw data with these detector settings.

        Example
        -------
        Assume you have a cube of raw full frame data (RAPID, ngroup=5).
        Create a det_timing instance and get channel:
        
        >>> d = det_timing(ngroup=5)
        >>> tarr = d.pixel_timing_map(return_flat=True, avg_groups=True)
        
        >>> nx, ny = (d.xpix, d.ypix)
        >>> nout   = d.nout      # Number of amplifier output channels
        >>> chsize = d.chsize    # Channel size (x-direction)
        >>> # Reshape into (nz, ny, nout, chsize)
        >>> data = data.reshape([-1,ny,nout,chsize])
        >>> # Reverse odd channels in x-direction to match even chans
        >>> for ch in range(nout):
        >>>     if np.mod(ch,2)==1:
        >>>         data[:,:,ch,:] = data[:,:,ch,::-1]
        >>> # Final data reshaped into 4 flattened output channels
        >>> data = data.transpose([0,1,3,2]).reshape([-1,nout])
        >>> # Can plot this like plt.plot(tarr, data) to make nout line plots
        
        """
        
        xpix = self.xpix
        ypix = self.ypix
        nout = self.nout

        chsize = self.chsize                   # Number of x-pixels within a channel
        xticks = chsize + self._line_overhead  # Clock ticks per line
        flines = ypix + self._extra_lines      # Lines per frame

        if same_scan_direction is None:
            same_scan_direction = self.same_scan_direction
        if reverse_scan_direction is None:
            reverse_scan_direction = self.reverse_scan_direction
        
        # Pixel-by-pixel or line-by-line reset?
        line_reset = True if 'line' in self._reset_type.lower() else False

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
        arr = np.arange(nticks, dtype=float).reshape([nframes,-1])
        arr_reset = np.arange(fticks, dtype=float)
        if reset_zero:
            arr += arr_reset.max() + 1
        
        # Chop off single pix overhead
        if pix_offset>0:
            arr = arr[:,:-pix_offset]
            arr_reset = arr_reset[:-pix_offset]
            
        # Reshape to 3D array
        arr = arr.reshape([nframes, flines, xticks])
        arr_reset = arr_reset.reshape([flines, xticks])

        # If performing line-by-line reset, all pixels in a line 
        # were reset at the same time.
        if line_reset:
            for i in range(flines):
                arr_reset[i,:] = np.max(arr_reset[i,:])
        
        # Chop off x pixel & y line overheads
        arr = arr[:,:ypix,:chsize]
        arr_reset = arr_reset[:ypix,:chsize]
        
        # By default fast-scan readout direction is [-->,<--,-->,<--]
        # If same_scan_direction, then all --> 
        # If reverse_scan_direction, then [<--,-->,<--,-->] or all <--
        if reverse_scan_direction:
            arr = arr[::-1]
            arr_reset = arr_reset[:,::-1]

        arr_list = []
        arr2_list = []
        if nout>1:
            # Consecutive outputs reversed?
            for ch in range(nout):
                if (np.mod(ch,2) == 0) or (same_scan_direction == True): 
                    arr_list.append(arr)
                    arr2_list.append(arr_reset)
                else: 
                    arr_list.append(arr[:,:,::-1])
                    arr2_list.append(arr_reset[:,::-1])
            data = np.concatenate(arr_list, axis=2)
            data_reset = np.concatenate(arr2_list, axis=1)
        else:
            data = arr
            data_reset = arr_reset
            
        del arr, arr_list, arr2_list, arr_reset   

        # Timing for averaged (bit-shifted) frames
        # Remove drops and average grouped data
        if (avg_groups and nf>1) or (nd2>0):
            # Trailing drop frames already excluded
            # so need to pull off last group of frames
            # in order to properly reshape things.
            if avg_groups and (nf>1):
                data_end = data[-nf:,:,:].mean(axis=0)
            else:
                data_end = data[-nf:,:,:]
            data_end = data_end.reshape([-1,ypix,xpix])

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
            if avg_groups and (nf>1):
                data = data.mean(axis=1)
            else:
                data = data.reshape([-1,ypix,xpix])

            # Add back the last group (already averaged)
            data = np.append(data,data_end,axis=0)

        if reset_zero:
            for im in data:
                im -= data_reset
            
        # Put into time
        # print(data.dtype)
        data /= self._pixel_rate

        # Return timing info
        if return_flat: # Flatten array
            return data.ravel()
        else: # Get rid of dimensions of length 1
            return data.squeeze()


def tuples_to_dict(pairs, verbose=False):
    """
    Take a list of paired tuples and convert to a dictionary
    where the first element of each tuple is the key and the 
    second element is the value.
    """
    d={}
    for (k, v) in pairs:
        d[k] = v
        if verbose:
            if isinstance(v,float): print("{:<10} {:>10.4f}".format(k, v))
            else: print("{:<10} {:>10}".format(k, v))
    return d


def nrc_header(det_class, filter=None, pupil=None, obs_time=None, header=None,
               DMS=True, targ_name=None):
    """Simulated header

    Create a generic NIRCam FITS header from a detector_ops class.

    Parameters
    ----------
    filter : str
        Name of filter element.
    pupil : str
        Name of pupil element.
    DMS : bool
        Make the header in a format used by Data Management Systems.
    obs_time : datetime
        Specifies when the observation was considered to be executed.
        If not specified, then it will choose the current time.
        This must be a datetime object:

            >>> datetime.datetime(2016, 5, 9, 11, 57, 5, 796686)

    header : obj
        Can pass an existing header that will be updated.
        This has not been fully tested.
    targ_name : str
        Standard astronomical catalog name for a target.
        Otherwise, it will be UNKNOWN.
    """

    from .version import __version__

    filter = 'UNKNOWN' if filter is None else filter
    pupil  = 'UNKNOWN' if pupil  is None else pupil
    targ_name = 'UNKNOWN' if targ_name is None else targ_name

    d = det_class
    # MULTIACCUM ramp information
    ma = d.multiaccum

    # How many axes?
    naxis = 2 if ma.ngroup == 1 else 3
    if naxis == 3:
        naxis3 = ma.ngroup
    naxis1 = d.xpix
    naxis2 = d.ypix

    # Select Detector ID based on SCA ID
    detector = d.detname

    # Are we in subarray?
    sub_bool = True if d.wind_mode != 'FULL' else False
    # Horizontal window mode?
    hwinmode = 'ENABLE' if d.wind_mode=='WINDOW' else 'DISABLE'

    # Window indices (0-indexed)
    x1 = d.x0; x2 = x1 + d.xpix
    y1 = d.y0; y2 = y1 + d.ypix

    # Ref pixel info
    ref_all = d.ref_info

    # Dates and times
    obs_time = datetime.datetime.utcnow() if obs_time is None else obs_time
    # Total time to complete obs including all overheads
    tdel = d.time_total 
    dtstart = obs_time.isoformat()
    aTstart = Time(dtstart)
    dtend = (obs_time + datetime.timedelta(seconds=tdel)).isoformat()
    aTend = Time(dtend)
    dstart = dtstart[:10]; dend = dtend[:10]
    tstart = dtstart[11:-3]; tend = dtend[11:-3]
    tsample = 1e6/d._pixel_rate

    ################################################################
    # Create blank header
    hdr_update = False  if header is None else True
    hdr = fits.Header() if header is None else header

    # Add in basic header info
    hdr['SIMPLE']  = (True,   'conforms to FITS standard')
    hdr['BITPIX']  = (16,     'array data type')
    if DMS == True:
        hdr['SUBSTRT1'] = (x1+1, 'Starting pixel in axis 1 direction')
        hdr['SUBSTRT2'] = (y1+1, 'Starting pixel in axis 2 direction')
        hdr['SUBSIZE1'] = naxis1
        hdr['SUBSIZE2'] = naxis2
        hdr['NAXIS'] = (naxis,  'number of array dimensions')
    else:
        hdr['NAXIS']   = (naxis,  'number of array dimensions')
        hdr['NAXIS1']  = naxis1
        hdr['NAXIS2']  = naxis2

        if hdr_update: hdr.pop('NAXIS3', None)
        if naxis == 3: hdr['NAXIS3']  = (naxis3, 'length of third data axis')
    hdr['EXTEND']  = True

    hdr['DATE']    = ('',   'date file created (yyyy-mm-ddThh:mm:ss,UTC)')
    hdr['BSCALE']  = (1,     'scale factor for array value to physical value')
    hdr['BZERO']   = (32768, 'physical value for an array value of zero')
    hdr['UNITS']   = ('',  'Units for the data type (ADU, e-, etc.)')
    hdr['ORIGIN']  = ('UAz',  'institution responsible for creating FITS file')
    hdr['FILENAME']= ('',   'name of file')
    hdr['FILETYPE']= ('raw', 'type of data found in data file')

    # Observation Description
    hdr['TELESCOP']= ('JWST',    'telescope used to acquire data')
    hdr['INSTRUME']= ('NIRCAM',  'instrument identifier used to acquire data')
    hdr['OBSERVER']= ('UNKNOWN', 'person responsible for acquiring data')
    hdr['DATE-OBS']= (dstart, 'UT date of observation (yyyy-mm-dd)')
    hdr['TIME-OBS']= (tstart, 'Approximate UT time of start of observation (hh:mm:ss.sss)')
    if DMS == True:
        if 'GRISM' in pupil:
            exp_type = 'NRC_GRISM'
        elif pupil == None:
            exp_type = 'UNKNOWN'
        else:
            exp_type = 'NRC_IMAGE'
        hdr['EXP_TYPE'] = (exp_type,'Type of data in the exposure')
    hdr['DATE-END']= (dend,   'UT date of end of observation(yyyy-mm-dd)')
    hdr['TIME-END']= (tend,   'UT time of end of observation (hh:mm:ss.sss)')
    hdr['SCA_ID']  = (d.scaid,   'Unique SCA identification in ISIM')
    hdr['DETECTOR']= (d.detname, 'ASCII Mnemonic corresponding to the SCA_ID')
    hdr['PIXELSCL']= (d.pixelscale, 'Detector Pixel Scale (arcsec/pixel)')

    nx_noref = naxis1 - ref_all[2] - ref_all[3]
    ny_noref = naxis2 - ref_all[0] - ref_all[1]
    fovx = nx_noref * d.pixelscale
    fovy = ny_noref * d.pixelscale
    hdr['FOV']     = ('{:.2f}x{:.2f}'.format(fovx,fovy), 'Field of view in arcsec')

    if DMS == True:
        hdr['TARG_RA']=  (0.0, 'Target RA at mid time of exposure') #arbitrary position
        hdr['TARG_DEC']= (0.0, 'Target Dec at mid time of exposure') #arbitrary position

        hdr['PROGRAM'] = ('12345', 'Program number')
        hdr['OBSERVTN']= ('001',   'Observation number')
        hdr['VISIT']   = ('001',   'Visit Number')
        hdr['VISITGRP']= ('01',  'Visit Group Identifier')

        hdr['SEQ_ID']  = ('1', 'Parallel sequence identifier')
        hdr['ACT_ID']  = ('1', 'Activity identifier')
        hdr['EXPOSURE']= ('1', 'Exposure request number')
        hdr['OBSLABEL']= ('Target 1 NIRCam Observation 1', 'Proposer label for the observation')
        hdr['EXPSTART']= (aTstart.mjd, 'UTC exposure start time')
        hdr['EXPEND']  = (aTend.mjd, 'UTC exposure end time')
        hdr['EFFEXPTM']= (d.time_int_eff*d.time_int, 'Effective exposure time (sec)')
        hdr['NUMDTHPT']= ('1','Total number of points in pattern')
        hdr['PATT_NUM']= (1,'Position number in primary pattern')

    hdr['TARGNAME'] = (targ_name, 'Standard astronomical catalog name for target')
    hdr['OBSMODE'] = ('UNKNOWN', 'Observation mode')

    if DMS == True:
        if d.channel == 'LW':
            headerChannel = 'LONG'
        elif d.channel == 'SW':
            headerChannel = 'SHORT'
        else:
            headerChannel = 'UNKNOWN'
        hdr['CHANNEL'] = headerChannel

        hdr['GRATING'] = ('N/A - NIRCam', 'Name of the grating element used')
        hdr['BAND']    = ('N/A - NIRCam', 'MRS wavelength band')
        hdr['LAMP']    = ('N/A - NIRCam', 'Internal lamp state')
        hdr['GWA_XTIL']= ('N/A - NIRCam', 'Grating X tilt angle relative to mirror')
        hdr['GWA_YTIL']= ('N/A - NIRCam', 'Grating Y tilt angle relative to mirror')
        hdr['GWA_TILT']= ('N/A - NIRCam', 'GWA TILT (avg/calib) temperature (K)')
        hdr['MSAMETFL']= ('N/A - NIRCam', 'MSA metadata file name')
        hdr['MSAMETID']= ('N/A - NIRCam', 'MSA metadata ID')

    # Positions of optical elements
    hdr['FILTER']  = (filter, 'Module ' + d.module + ' ' + d.channel + ' FW element')
    hdr['PUPIL']   = (pupil, 'Module ' + d.module + ' ' + d.channel + ' PW element')
    hdr['PILSTATE']= ('RETRACTED', 'Module ' + d.module + ' PIL deploy state')

    # Readout Mode
    hdr['NSAMPLE'] = (1,            'A/D samples per read of a pixel')
    if DMS == True:
        frmName = 'NFRAMES'
        grpName = 'NGROUPS'
        intName = 'NINTS'
    else:
        frmName = 'NFRAME'
        grpName = 'NGROUP'
        intName = 'NINT'
    hdr[frmName]   = (ma.nf,         'Number of frames in group')
    hdr[grpName]   = (ma.ngroup,     'Number groups in an integration')
    hdr[intName]   = (ma.nint,     'Number of integrations in an exposure')

    # Timing information
    hdr['TSAMPLE'] = (tsample,        'Delta time between samples in microsec')
    hdr['TFRAME']  = (d.time_frame,   'Time in seconds between frames')
    hdr['TGROUP']  = (d.time_group,   'Delta time between groups')
    hdr['DRPFRMS1']= (ma.nd1, 'Number of frame skipped prior to first integration')
    hdr['GROUPGAP']= (ma.nd2, 'Number of frames skipped')
    hdr['DRPFRMS3']= (ma.nd3, 'Number of frames skipped between integrations')
    hdr['FRMDIVSR']= (ma.nf,  'Divisor applied to each group image')
    hdr['INTAVG']  = (1, 'Number of integrations averaged in one image')
    hdr['NRESETS1']= (ma.nr1, 'Number of reset frames prior to first integration')
    hdr['NRESETS2']= (ma.nr2, 'Number of reset frames between each integration')
    hdr['INTTIME'] = (d.time_int, 'Total integration time for one MULTIACCUM')
    hdr['EXPTIME'] = (d.time_exp, 'Exposure duration (seconds) calculated')
    hdr['FASTAXIS']= (d.fastaxis, 'Fast readout direction relative to image axes for Amp1')
    hdr['SLOWAXIS']= (d.slowaxis, 'Slow readout direction relative to image axes')
    
    # Subarray names
    if DMS == True:
        if (d.xpix == 2048) & (d.ypix == 2048):
            subName = 'FULL'
        elif (d.xpix == 640) & (d.ypix == 640):
            subName = 'SUB640'
        elif (d.xpix == 320) & (d.ypix == 320):
            subName = 'SUB320'
        elif (d.xpix == 400) & (d.ypix == 400):
            subName = 'SUB400P'
        elif (d.xpix == 64) & (d.ypix == 64):
            subName = 'SUB64P'
        elif (d.xpix == 2048) & (d.ypix == 256):
            subName = 'SUBGRISM256'
        elif (d.xpix == 2048) & (d.ypix == 128):
            subName = 'SUBGRISM128'
        elif (d.xpix == 2048) & (d.ypix == 64):
            subName = 'SUBGRISM64'
        else:
            subName = 'UNKNOWN'
        hdr['SUBARRAY']= (subName, 'Detector subarray string')
    else:
        hdr['SUBARRAY']= (sub_bool, 'T if subarray used, F if not')
        hdr['HWINMODE']= (hwinmode, 'If enabled, single output mode used')

    # Readout Patterns
    if DMS == True:
        hdr['READPATT']= (ma.read_mode, 'Readout pattern name')
        hdr['ZROFRAME']= (True,       'T if zeroth frame present, F if not')
    else:
        hdr['READOUT'] = (ma.read_mode, 'Readout pattern name')
        hdr['ZROFRAME']= (False,       'T if zeroth frame present, F if not')

    #Reference Data
    hdr['TREFROW'] = (ref_all[1], 'top reference pixel rows')
    hdr['BREFROW'] = (ref_all[0], 'bottom reference pixel rows')
    hdr['LREFCOL'] = (ref_all[2], 'left col reference pixels')
    hdr['RREFCOL'] = (ref_all[3], 'right col reference pixels')
    hdr['NREFIMG'] = (0, 'number of reference rows added to end')
    hdr['NXREFIMG']= (0, 'reference image columns')
    hdr['NYREFIMG']= (0, 'reference image rows')
    hdr['COLCORNR']= (x1+1, 'The Starting Column for ' + detector)
    hdr['ROWCORNR']= (y1+1, 'The Starting Row for ' + detector)

    hdr.insert('EXTEND', '', after=True)
    hdr.insert('EXTEND', '', after=True)
    hdr.insert('EXTEND', '', after=True)

    hdr.insert('FILETYPE', '', after=True)
    hdr.insert('FILETYPE', ('','Observation Description'), after=True)
    hdr.insert('FILETYPE', '', after=True)

    hdr.insert('OBSMODE', '', after=True)
    hdr.insert('OBSMODE', ('','Optical Mechanisms'), after=True)
    hdr.insert('OBSMODE', '', after=True)

    hdr.insert('PILSTATE', '', after=True)
    hdr.insert('PILSTATE', ('','Readout Mode'), after=True)
    hdr.insert('PILSTATE', '', after=True)

    hdr.insert('ZROFRAME', '', after=True)
    hdr.insert('ZROFRAME', ('','Reference Data'), after=True)
    hdr.insert('ZROFRAME', '', after=True)

    hdr.insert('ROWCORNR', '', after=True)
    hdr.insert('ROWCORNR', '', after=True)

    hdr['comment'] = 'Simulated data generated by {} v{}'\
                      .format(__package__,__version__)

    return hdr

def config2(input, intype='int'):
    """NIRCam CONFIG2 (0x4011) Register

    Return a dictionary of configuration parameters depending on the
    value of CONFIG2 register (4011).

    Parameters
    ----------
    input : int, str
        Value of CONFIG2, nominally as an int. Binary and Hex values
        can also be passed as strings.
    intype: str
        Input type (int, hex, or bin) for integer, hex, string,
        or binary string.

    """
    if 'hex' in intype:
        if '0x' in input:
            input = int(input, 0)
        else:
            input = int(input, 16)
    if 'bin' in intype:
        if '0b' in input:
            input = int(input, 0)
        else:
            input = int(input, 2)

    # Convert to 16-bit binary string
    input = "{0:016b}".format(input)

    # Config2 Bits (Right to Left)
    # ----------------------------
    # 0 : Vertical Enable
    # 1 : Horizontal Enable
    # 2 : Global reset per integration
    # 3 : Enable Fast row-by-row reset (only in window/stripe)
    # 6-4 : Number of fast row resets per int
    # 7 : Window mode in Idle when window enabled?
    # 8 : 0 = Preamp reset per frame; 1 = reset per row
    # 9 : Permanent Reset
    # 10 : Single step mode
    # 11 : Test pattern
    # 12 : FGS window mode
    # 13 : Power down preamp, adc, and ap during Idle
    # 14 : Power down preamp, adc, and ap during Drop
    # 15 : 0 = Preamp reset per frame; 1 = reset per integration

    # NFF Rows Reset
    # --------------
    # 000 = 1
    # 001 = 4
    # 010 = 16
    # 011 = 64
    # 100 = 256
    # 101 = 512
    # 110 = 1024
    # 111 = 2048

    nff_dict = {'000':   1, '001':   4, '010':  16, '011':  64,
                '100': 256, '101': 512, '110':1024, '111':2048}

    # Reverse for easier indexing of single values
    input2 = input[::-1]

    d = {}
    d['00_window_vert']  = True if bool(int(input2[0])) else False
    d['01_window_horz']  = True if bool(int(input2[1])) else False
    d['02_global_reset'] = True if bool(int(input2[2])) else False
    d['03_rows_reset']   = True if bool(int(input2[3])) else False
    d['04_rows_nff']     = nff_dict.get(input2[4:7][::-1])
    d['07_idle_window']  = True if bool(int(input2[7])) else False
    d['08_pa_reset']     = 'row' if bool(int(input2[8])) else 'frame'
    d['09_perm_reset']   = True if bool(int(input2[9])) else False
    d['10_single_step']  = True if bool(int(input2[10])) else False
    d['11_test_patt']    = True if bool(int(input2[11])) else False
    d['12_fgs_wind']     = True if bool(int(input2[12])) else False
    d['13_power_idl']    = True if bool(int(input2[13])) else False
    d['14_power_drop']   = True if bool(int(input2[14])) else False
    d['15_pa_reset']     = 'int' if bool(int(input2[15])) else 'frame'

    return d

def create_detops(header, DMS=False, read_mode=None, nint=None, ngroup=None,
    detector=None, wind_mode=None, xpix=None, ypix=None, x0=None, y0=None,
    nff=None):
    """NIRCam Detector class from header

    Create a NIRCam detector class based on header settings.
    Can override settings with a variety of keyword arguments.

    Parameters
    ----------
    header : obj
        Header from NIRCam FITS file
    DMS : bool
        Is header format from Data Management Systems? Otherwises, ISIM-like.

    Keyword Args
    ------------
    read_mode : str
        NIRCam Ramp Readout mode such as 'RAPID', 'BRIGHT1', etc.
    nint : int
        Number of integrations (ramps).
    ngroup : int
        Number of groups in a integration.
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

    """
    # Create detector class
    from pynrc.pynrc_core import DetectorOps

    # Detector ID
    if detector is None:
        detector = header.get('SCA_ID')
        if detector is None:
            detector = header.get('DETECTOR')         

    # Detector size
    xpix = header['SUBSIZE1'] if DMS else header['NAXIS1'] if xpix is None else xpix
    ypix = header['SUBSIZE2'] if DMS else header['NAXIS2'] if ypix is None else ypix

    # Subarray position
    # Headers are 1-indexed, while detector class is 0-indexed
    if x0 is None:
        x1 = header['SUBSTRT1'] if DMS else header['COLCORNR']
        x0 = x1 - 1
    if y0 is None:
        y1 = header['SUBSTRT2'] if DMS else header['ROWCORNR']
        y0 = y1 - 1

    # Subarray setting: Full, Stripe, or Window
    if wind_mode is None:
        if xpix==ypix==2048:
            wind_mode = 'FULL'
        else:
            log_prev = conf.logging_level
            setup_logging('ERROR', verbose=False)

            # Test if STRIPE or WINDOW
            det_stripe = DetectorOps(detector, 'STRIPE', xpix, ypix, x0, y0)
            det_window = DetectorOps(detector, 'WINDOW', xpix, ypix, x0, y0)
            dt_stripe = np.abs(header['TFRAME'] - det_stripe.time_frame)
            dt_window = np.abs(header['TFRAME'] - det_window.time_frame)
            wind_mode = 'STRIPE' if dt_stripe<dt_window else 'WINDOW'

            setup_logging(log_prev, verbose=False)


    # Add MultiAccum info
    hnames = ['READPATT', 'NINTS', 'NGROUPS'] if DMS else ['READOUT',  'NINT',  'NGROUP']

    read_mode = header[hnames[0]] if read_mode is None else read_mode
    nint      = header[hnames[1]] if nint      is None else nint
    ngroup    = header[hnames[2]] if ngroup    is None else ngroup

    ma_args = {'read_mode':read_mode, 'nint':nint, 'ngroup':ngroup}

    return DetectorOps(detector, wind_mode, xpix, ypix, x0, y0, nff, **ma_args)

