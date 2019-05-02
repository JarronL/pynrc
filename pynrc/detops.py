from __future__ import print_function, division
import numpy as np

import logging
_log = logging.getLogger('detops')

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
                 nr1=1, nr2=1, **kwargs):


        # Pre-defined patterns
        patterns = ['RAPID', 'BRIGHT1', 'BRIGHT2', 'SHALLOW2', 'SHALLOW4', 'MEDIUM2', 'MEDIUM8', 'DEEP2', 'DEEP8']
        nf_arr   = [1,1,2,2,4,2,8, 2, 8]
        nd2_arr  = [0,1,0,3,1,8,2,18,12]
        # TODO: ng_max currently ignored, because not valid for TSO
        ng_max   = [10,10,10,10,10,10,10,20,20]
        self._pattern_settings = dict(zip(patterns, zip(nf_arr, nd2_arr, ng_max)))

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
        value = self._check_int('nd1',nd1,0)
        self._nd1 = self._check_custom(value, self._nd1)

    @property
    def nd2(self):
        """Number of drop frames within a group (aka, groupgap)."""
        return self._nd2
    @nd2.setter
    def nd2(self, value):
        value = self._check_int('nd2',nd2,0)
        self._nd2 = self._check_custom(value, self._nd2)

    @property
    def nd3(self):
        """Number of drop frames after final read frame in ramp."""
        return self._nd3
    @nd3.setter
    def nd3(self, value):
        value = self._check_int('nd3',nd3,0)
        self._nd3 = self._check_custom(value, self._nd3)

    @property
    def nr1(self):
        """Number of reset frames before first integration."""
        return self._nr1
    @nr1.setter
    def nr1(self, value):
        self._nr1 = self._check_int(value, minval=0)

    @property
    def nr2(self):
        """Number of reset frames for subsequent integrations."""
        return self._nr2
    @nr2.setter
    def nr2(self, value):
        self._nr2 = self._check_int(value, minval=0)

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
            _log.warning('Readout {} not a valid NIRCam readout mode. Setting to CUSTOM.'\
                         .format(self.read_mode))
            self._read_mode = 'CUSTOM'
            _log.warning('Using explicit settings: ngroup={}, nf={}, nd1={}, nd2={}, nd3={}'\
                         .format(self.ngroup, self.nf, self.nd1, self.nd2, self.nd3))
        elif self.read_mode == 'CUSTOM':
            _log.info('{} readout mode selected.'.format(read_mode))
            _log.info('Using explicit settings: ngroup={}, nf={}, nd1={}, nd2={}, nd3={}'\
                      .format(self.ngroup, self.nf, self.nd1, self.nd2, self.nd3))
        else:
            _log.info('{} readout mode selected.'.format(self.read_mode))
            nf, nd2, _ = self._pattern_settings.get(self.read_mode)
            self._nf  = nf
            self._nd1 = 0
            self._nd2 = nd2
            self._nd3 = 0
            _log.info('Setting nf={}, nd1={}, nd2={}, nd3={}.'\
                     .format(self.nf, self.nd1, self.nd2, self.nd3))


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

        self.wind_mode = wind_mode.upper()
        self.xpix = xpix; self.x0 = x0
        self.ypix = ypix; self.y0 = y0

        self._validate_pixel_settings()


        # Pixel Rate in Hz
        self._pixel_rate = pixrate
        # Number of extra clock ticks per line
        self._line_overhead = loh
        # Pixel or line resets
        self._reset_type = reset_type
        
        self.multiaccum = multiaccum(**kwargs)

    @property
    def nout(self):
        """Number of simultaenous detector output channels stripes"""
        return 1 if self.wind_mode == 'WINDOW' else self._nchans

    @property
    def chsize(self):
        """"""
        return int(self.xpix // self.nout)

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
            print('{} not a valid window readout mode! Returning...'.format(wind_mode))
            return

        detpix = self._detector_pixels
        xpix = self.xpix; x0 = self.x0
        ypix = self.ypix; y0 = self.y0

        # Check some consistencies with frame sizes
        if wind_mode == 'FULL':
            if ypix != detpix:
                print('In {0} mode, but ypix not {1}. Setting ypix={1}.'\
                       .format(wind_mode,detpix))
                ypix = detpix
            if y0 != 0:
                print('In {0} mode, but x0 not 0. Setting y0=0.'.format(wind_mode))
                y0 = 0

        if (wind_mode == 'STRIPE') or (wind_mode == 'FULL'):
            if xpix != detpix:
                print('In {0} mode, but xpix not {1}. Setting xpix={1}.'.format(wind_mode,detpix))
                xpix = detpix
            if x0 != 0:
                print('In {0} mode, but x0 not 0. Setting x0=0.'.format(wind_mode))
                x0 = 0
    
        if (x0+xpix) > detpix:
            raise ValueError("x0+xpix ({}+{}) is larger than detector size ({})!"\
                             .format(x0,xpix,detpix))
        if (y0+ypix) > detpix:
            raise ValueError("y0+ypix ({}+{}) is larger than detector size ({})!"\
                             .format(y0,ypix,detpix))

        # Update values if no errors were thrown
        self.xpix = xpix; self.x0 = x0
        self.ypix = ypix; self.y0 = y0

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
        """Determine frame times based on xpix, ypix, and wind_mode."""

        chsize = self.chsize                        # Number of x-pixels within a channel
        xticks = self.chsize + self._line_overhead  # Clock ticks per line
        flines = self.ypix + self._extra_lines      # Lines per frame

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
        return tint

    @property
    def time_int(self):
        """Same as time_ramp, except that 'int' follows the JWST nomenclature"""
        return self.time_ramp

    @property
    def time_ramp_eff(self):
        """Effective ramp time for slope fit tf*(ng-1)"""
        
        ma = self.multiaccum
        if ma.ngroup<=1:
            return self.time_frame * (ma.nd1 + (ma.nf + 1) / 2)
        else:
            return self.time_group * (ma.ngroup - 1)

    @property
    def time_int_eff(self):
        """Same as time_ramp_eff, except that 'int' follows the JWST nomenclature"""
        return self.time_ramp_eff

    @property
    def time_exp(self):
        """Total photon collection time for all ramps."""
        return self.multiaccum.nint * self.time_ramp

#     @property
#     def time_total_int(self):
#         """Total time for all frames in a ramp.
#         
#         Includes resets and excess drops, as well as NFF Rows Reset.
#         """
# 
#         ma = self.multiaccum
#         nf = ma.nf; nd1 = ma.nd1; nd2 = ma.nd2; nd3 = ma.nd3
#         ngroup = ma.ngroup
#         nr = 1
# 
#         nframes = nr + nd1 + ngroup*nf + (ngroup-1)*nd2 + nd3        
#         return nframes * self.time_frame + self.time_row_reset

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
        return nframes * self.time_frame + self.time_row_reset

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
            return nframes * self.time_frame + self.time_row_reset

    @property
    def time_total(self):
        """Total exposure acquisition time"""
#         exp1 = 0 if self.multiaccum.nint == 0 else self.time_total_int1
#         exp2 = 0 if self.multiaccum.nint <= 1 else self.time_total_int2 * (self.multiaccum.nint-1)
        exp1 = self.time_total_int1
        exp2 = self.time_total_int2 * (self.multiaccum.nint-1)
        return exp1 + exp2 + self._exp_delay

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
        
    def pix_timing_map(self, same_scan_direction=False, reverse_scan_direction=False,
                       avg_groups=False, reset_zero=False, return_flat=False):
        """Create array of pixel times for a single ramp. 
        
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
        arr = np.arange(nticks, dtype=np.float).reshape([nframes,-1])
        arr_reset = np.arange(fticks, dtype=np.float)
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
        print(data.dtype)
        data /= self._pixel_rate

        # Return timing info
        if return_flat: # Flatten array
            return data.ravel()
        else: # Get rid of dimensions of length 1
            return data.squeeze()
        
        
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
    """
    d={}
    for (k, v) in pairs:
        d[k] = v
        if verbose:
            if isinstance(v,float): print("{:<10} {:>10.4f}".format(k, v))
            else: print("{:<10} {:>10}".format(k, v))
    return d

