from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import logging
_log = logging.getLogger('pynrc')

# The six library is useful for Python 2 and 3 compatibility
#import six

# Import libraries
import numpy as np
import pyrnc
from pynrc.maths import robust
#from pynrc.maths import nrc_utils

#from . import *
#from .nrc_utils import *


### # import matplotlib
### # import matplotlib.pyplot as plt
### matplotlib.rcParams['image.origin'] = 'lower'
### matplotlib.rcParams['image.interpolation'] = 'none'
### #matplotlib.rcParams['image.cmap'] = 'gist_heat'
### 
### from astropy.io import fits
### 
### dir = '/Volumes/NIRData/NIRCam/Char_Darks/CV3/FITS/489/'
### f = dir + 'NRCNRCB4-DARK-60091316411_1_489_SE_2016-01-09T14h23m38.fits'
### hdulist = fits.open(f)
### cube = np.ndarray.astype(hdulist[0].data,np.float)
### hdulist.close()
### del hdulist[0].data
### 
### fbias = '/Volumes/NIRData/NIRCam/Char_Darks/CV3/489/SUPER_BIAS_489.FITS'
### hdulist = fits.open(fbias)
### bias = hdulist[0].data
### 
### # We need a better superbias frame.
### # These ones did not correctly determine the intrinsic values
### # of the reference pixels.
### for im in cube:
###     im -= bias
### 
### cube = reffix_amps(cube)

class references(object):

    def __init__(self, hdul, DMS=True):
    
        self.DMS = DMS

        self.hdulist = hdul
        self.nreft = self.header['TREFROW']
        self.nrefb = self.header['BREFROW']
        self.nrefl = self.header['LREFCOL']
        self.nrefr = self.header['RREFCOL']
        
        self._create_multiaccum()

        
    @property
    def header(self):
        return self.hdulist[0].header
        
    def _create_multiaccum(self, DMS=True, read_mode=None, nint=None, ngroup=None):
        """Multiaccum object based on header settings."""
        
        if DMS:
            hnames = ['READPATT', 'NINTS', 'NGROUPS']  
        else:
            hnames = ['READOUT', 'NINT', 'NGROUP']
        read_mode = header[hnames[0]] if read_mode is None else read_mode
        nint      = header[hnames[0]]    if nint      is None else nint
        ngroup    = header[hnames[0]] if ngroup is None else ngroup
        
        self.multiaccum = multiaccum(read_mode, nint, ngroup)
        
    #@property
    #ref_vals(self):





def reffix_hxrg(cube, nchans=4, in_place=True, fixcol=False, **kwargs):
    """
    This program performs a reference pixel correction
    on HAWAII-[1,2,4]RG detector data read out using N outputs.
    Top and bottom reference pixels are used first to remove 
    channel offsets.
    
    Parameters
    ===========
    cube   (ndarray) : Input datacube. Can be two or three dimensions (nz,ny,nx).
    in_place  (bool) : Perform calculations in place. Input array is overwritten.
    nchans     (int) : Number of output amplifier channels in the detector. Default=4.
    fix_col   (bool) : Perform reference column corrections?

    Channel amplifier keywords
    --------------------------
    altcol    (bool) : Calculate separate reference values for even/odd columns.
    supermean (bool) : Add back the overall mean of the reference pixels.
    top_ref   (bool) : Include top reference rows when correcting channel offsets.
    bot_ref   (bool) : Include bottom reference rows when correcting channel offsets.
    ntop       (int) : Specify the number of top reference rows.
    nbot       (int) : Specify the number of bottom reference rows.

    Reference column keywords
    --------------------------
    left_ref  (bool) : Include left reference cols when correcting 1/f noise.
    right_ref (bool) : Include right reference cols when correcting 1/f noise.
    nleft      (int) : Specify the number of left reference columns.
    nright     (int) : Specify the number of right reference columns.
    perint    (bool) : Smooth side reference pixel per integration, 
                       otherwise do frame-by-frame.
    avg_type   (str) : Type of ref col averaging to perform. Allowed values are
                       'pixel', 'frame', or 'int'.    
    """


    # Check the number of dimensions are valid.
    ndim = len(cube.shape)
    if not (ndim==2 or ndim==3):
        raise ValueError('Input data can only have 2 or 3 dimensions. \
                          Found {} dimensions.'.format(ndim))

    # Convert to float
    if 'float' not in cube.dtype.name:
        copy = (not in_place)
        cube = cube.astype(np.float, copy=copy)

    if not in_place:
        cube = np.copy(cube)
    
    # Remove channel offsets
    cube = reffix_amps(cube, in_place=True, **kwargs)
    
    # Fix 1/f noise using vertical reference pixels
    if fixcol:
        cube = ref_filter(cube, in_place=True, **kwargs)
    
    return cube


def reffix_amps(cube, nchans=4, in_place=True, altcol=True, supermean=False,
                top_ref=True, bot_ref=True, ntop=4, nbot=4, **kwargs):
    """
    Matches all amplifier outputs of the detector to a common level.
    
    This routine subtracts the average of the top and bottom reference rows
    for each amplifier and frame individually.
    
    By default, reference pixel corrections are performed in place since it's
    faster and consumes less memory.
    
    Parameters
    ===========
    cube   (ndarray) : Input datacube. Can be two or three dimensions (nz,ny,nx).
    nchans     (int) : Number of output amplifier channels in the detector. Default=4.
    altcol    (bool) : Calculate separate reference values for even/odd columns.
    supermean (bool) : Add back the overall mean of the reference pixels
    in_place  (bool) : Perform calculations in place. Input array is overwritten.
    top_ref   (bool) : Include top reference rows when correcting channel offsets.
    bot_ref   (bool) : Include bottom reference rows when correcting channel offsets.
    ntop       (int) : Specify the number of top reference rows.
    nbot       (int) : Specify the number of bottom reference rows.
    """
    
    if not in_place:
        cube = np.copy(cube)
    
    # Check the number of dimensions are valid.
    ndim = len(cube.shape)
    if ndim==2:
        ny,nx = cube.shape
        nz = 1
        cube = cube.reshape((nz,ny,nx))
    elif ndim==3:
        nz, ny, nx = cube.shape
    else:
        raise ValueError('Input data can only have 2 or 3 dimensions. \
                          Found {} dimensions.'.format(ndim))        

    chsize = int(nx / nchans)
    
    # Number of reference rows to use
    # Set nt or nb equal to 0 if we don't want to use either
    nt = ntop if top_ref else 0
    nb = nbot if bot_ref else 0
    
    if (nt+nb)==0: 
        print("No reference pixels available for use. Returning...")
        return
    
    # Slice out reference pixels
    refs_bot = cube[:,:nb,:]
    refs_top = cube[:,-nt:,:]
    if nt==0:
        refs_all = refs_bot
    elif nb==0:
        refs_all = refs_top
    else:
        refs_all = np.hstack((refs_bot, refs_top))
        
    assert refs_all.shape[1] == (nb+nt)

    # Supermean
    # the average of the average is the DC level of the output channel
    smean = robust.mean(refs_all) if supermean else 0.0
    
    if altcol:
        for ch in range(nchans):
            # Channel indices
            ich1 = ch*chsize
            ich2 = ich1 + chsize

            # Slice out alternating columns
            refs_ch1 = refs_all[:,:,ich1:ich2-1:2].reshape((nz,-1))
            refs_ch2 = refs_all[:,:,ich1+1:ich2:2].reshape((nz,-1))

            # Take the resistant mean and reshape for broadcasting
            chmed1 = robust.mean(refs_ch1,axis=1).reshape(-1,1,1)
            chmed2 = robust.mean(refs_ch2,axis=1).reshape(-1,1,1)
            
            # In-place subtraction of channel medians
            # This utilizes numpy array broadcasting
            #cube[:,:,ich1:ich2-1:2] -= chmed1
            #cube[:,:,ich1+1:ich2:2] -= chmed2
            # Looping over nz seems to be faster, though
            for i in range(nz):
                cube[i,:,ich1:ich2-1:2] -= chmed1[i]
                cube[i,:,ich1+1:ich2:2] -= chmed2[i]
    else:
        for ch in range(nchans):
            # Channel indices
            ich1 = ch*chsize
            ich2 = ich1 + chsize

            # Slice out alternating columns
            refs_ch = refs_all[:,:,ich1:ich2].reshape((nz,-1))

            # Take the resistant mean and reshape for broadcasting
            chmed = robust.mean(refs_ch,axis=1).reshape(-1,1,1)
            
            # In-place subtraction of channel medians
            # This utilizes numpy array broadcasting
            #cube[:,:,ich1:ich2] -= chmed
            # Looping over nz seems to be faster, though
            for i in range(nz):
                cube[i,:,ich1:ich2] -= chmed[i]
                
    # Add back supermean
    if supermean: cube += smean
    
    if ndim==2: return cube[0]
    else: return cube



def ref_filter(cube, nchans=4, in_place=True, perint=False, avg_type=None, 
               left_ref=True, right_ref=True, nleft=4, nright=4, **kwargs):
    """
    Performs an optimal filtering of the vertical reference pixel to 
    reduce 1/f noise (horizontal stripes).
    
    Adapted from M. Robberto IDL code:
    http://www.stsci.edu/~robberto/Main/Software/IDL4pipeline/

    Parameters
    ===========
    cube   (ndarray) : Input datacube. Can be two or three dimensions (nz,ny,nx).
    nchans     (int) : Number of output amplifier channels in the detector. Default=4.
    in_place  (bool) : Perform calculations in place. Input array is overwritten.    
    perint    (bool) : Smooth side reference pixel per integration, 
                       otherwise do frame-by-frame.
    avg_type   (str) : Type of ref col averaging to perform. Allowed values are
                       'pixel', 'frame', or 'int'.
    left_ref  (bool) : Include left reference cols when correcting 1/f noise.
    right_ref (bool) : Include right reference cols when correcting 1/f noise.
    nleft      (int) : Specify the number of left reference columns.
    nright     (int) : Specify the number of right reference columns.
    """               
               
    if not in_place:
        cube = np.copy(cube)
    
    # Check the number of dimensions are valid.
    ndim = len(cube.shape)
    if ndim==2:
        ny,nx = cube.shape
        nz = 1
        cube = cube.reshape((nz,ny,nx))
    elif ndim==3:
        nz, ny, nx = cube.shape
    else:
        raise ValueError('Input data can only have 2 or 3 dimensions. \
                          Found {} dimensions.'.format(ndim))        

    # Number of reference rows to use
    # Set nt or nb equal to 0 if we don't want to use either
    nl = nleft  if left_ref  else 0
    nr = nright if right_ref else 0
    
    assert nl>=0, 'Number of left reference pixels must not be negative.'
    assert nr>=0, 'Number of right reference pixels must not be negative.'
    
    if (nl+nr)==0: 
        print("No reference pixels available for use. Returning...")
        return

    # Slice out reference pixels
    refs_left  = cube[:,:,:nl]
    refs_right = cube[:,:,-nr:]
    
    # Set the average of left and right reference pixels to zero
    # By default, pixel averaging is best for large groups
    if avg_type is None:
        avg_type = 'pixel'
    # If there is only 1 frame, then we have to do per frame averaging.
    # Set to per int, which produces the same result as per frame for nz=1.
    if nz==1:
        avg_type = 'int'
    
    # Average over entire integration
    if 'int' in avg_type:
        if nl>0: refs_left  -= np.median(refs_left)
        if nr>0: refs_right -= np.median(refs_right)
    # Average over each frame
    elif 'frame' in avg_type:
        if nl>0: refs_left_mean  = np.median(refs_left.reshape((nz,-1)), axis=1)
        if nr>0: refs_right_mean = np.median(refs_right.reshape((nz,-1)), axis=1)
        # Subtract estimate of each ref pixel "intrinsic" value
        for i in range(nz):
            if nl>0: refs_left[i]  -= refs_left_mean[i]
            if nr>0: refs_right[i] -= refs_right_mean[i]
    # Take the average of each reference pixel 
    elif 'pix' in avg_type:
        if nl>0: refs_left_mean  = np.median(refs_left, axis=0)
        if nr>0: refs_right_mean = np.median(refs_right, axis=0)
        # Subtract estimate of each ref pixel "intrinsic" value
        for i in range(nz):
            if nl>0: refs_left[i]  -= refs_left_mean
            if nr>0: refs_right[i] -= refs_right_mean
        
    if nl==0:
        refvals = refs_right.mean(axis=2)
    elif nr==0:
        refvals = refs_left.mean(axis=2)    
    else:
        refvals = (refs_right.mean(axis=2) + refs_left.mean(axis=2)) / 2

    # The delta time does't seem to make any difference in the final data product
    # Just for vizualization purposes...
    delt = 10E-6 * (nx/nchans + 12.)

    # May want to revisit the do-all-at-once or break-up decision
    # This may depend on preamp reset per frame or per integration
    # For now, we'll do it frame-by-frame by default (perint=False)
    if perint:
        refvals_smoothed = smooth_fft(refvals, delt).reshape(refvals.shape)
    else:
        refvals_smoothed = np.array([smooth_fft(ref, delt) for ref in refvals])
        
    # Final correction
    #for i,im in enumerate(cube): im -= refvals_smoothed[i].reshape([ny,1])
    cube -= refvals_smoothed.reshape([nz,ny,1])
    
    return cube


def smooth_fft(data, delt, first_deriv=False, second_deriv=False):
    """
    Smoothing algorithm to perform optimal filtering of the 
    vertical reference pixel to reduce 1/f noise (horizontal stripes),
    based on the Kosarev & Pantos algorithm. This assumes that the
    data to be filtered/smoothed has been sampled evenly.

    Parameters
    ===========
    data (ndarray) : Signal to be filtered.
    delt   (float) : Delta time between samples.
    
    first_deriv  (bool) : Return the first derivative.    
    second_deriv (bool) : Return the second derivative (along with first).
    
    If first_deriv is set, then returns two results
    if second_deriv is set, then returns three results.
    """

    Dat = data.flatten()
    N = Dat.size
    Pi2 = 2*np.pi
    OMEGA = Pi2 / (N*delt)
    X = np.arange(N) * delt

    ##------------------------------------------------
    ## Center and Baselinefit of the data
    ##------------------------------------------------
    Dat_m = Dat - np.mean(Dat)
    SLOPE = (Dat_m[-1] - Dat_m[0]) / (N-2)
    Dat_b = Dat_m - Dat_m[0] - SLOPE * X / delt
    
    ##------------------------------------------------
    ## Compute fft- / power- spectrum
    ##------------------------------------------------
    Dat_F = np.fft.fft(Dat_b) #/ N
    Dat_P = np.abs(Dat_F)**2
    
    # Frequency for abscissa axis
    # F = [0.0, 1.0/(N*delt), ... , 1.0/(2.0*delt)]:
    #F = np.arange(N/2+1) / (N*delt)
    #F = np.fft.fftfreq(Dat_F.size, delt)

    ##------------------------------------------------
    ## Noise spectrum from 'half' to 'full'
    ## Mind: half means N/4, full means N/2
    ##------------------------------------------------
    i1 = int((N-1) / 4)
    i2 = int((N-1) / 2) + 1
    Sigma = np.sum(Dat_P[i1:i2])
    Noise = Sigma / ((N-1)/2 - (N-1)/4)

    ##------------------------------------------------
    ## Get Filtercoeff. according to Kosarev/Pantos
    ## Find the J0, start search at i=1 (i=0 is the mean)
    ##------------------------------------------------
    J0 = 2
    for i in np.arange(1, int(N/4)+1):
        sig0, sig1, sig2, sig3 = Dat_P[i:i+4]
        if (sig0<Noise) and ((sig1<Noise) or (sig2<Noise) or (sig3<Noise)):
            J0 = i
            break
    
    ##------------------------------------------------
    ## Compute straight line extrapolation to log(Dat_P)
    ##------------------------------------------------
    ii = np.arange(1,J0+1)
    logvals = np.log(Dat_P[1:J0+1])
    XY = np.sum(ii * logvals)
    XX = np.sum(ii**2)
    S  = np.sum(logvals)
    # Find parameters A1, B1
    XM = (2. + J0) / 2
    YM = S / J0
    A1 = (XY - J0*XM*YM) / (XX - J0*XM*XM)
    B1 = YM - A1 * XM
    
    # Compute J1, the frequency for which straight
    # line extrapolation drops 20dB below noise
    J1 = int(np.ceil((np.log(0.01*Noise) - B1) / A1 ))

    
    ##------------------------------------------------
    ## Compute the Kosarev-Pantos filter windows
    ## Frequency-ranges: 0 -- J0 | J0+1 -- J1 | J1+1 -- N2
    ##------------------------------------------------
    nvals = int((N-1)/2 + 1)
    LOPT = np.zeros(nvals)
    LOPT[0:J0+1] = Dat_P[0:J0+1] / (Dat_P[0:J0+1] + Noise)
    i_arr = np.arange(J1-J0) + J0+1
    LOPT[J0+1:J1+1] = np.exp(A1*i_arr+B1) / (np.exp(A1*i_arr+B1) + Noise)

    ##--------------------------------------------------------------------
    ## De-noise the Spectrum with the filter
    ## Calculate the first and second derivative (i.e. multiply by iW)
    ##--------------------------------------------------------------------

    # first loop gives smoothed data
    # second loop produces first derivative
    # third loop produces second derivative
    if second_deriv:
        ndiff = 3
    elif first_deriv:
        ndiff = 2
    else:
        ndiff = 1
    
    for diff in range(ndiff):
        Fltr_Spectrum = np.zeros(N,dtype=np.complex)
        # make the filter complex
        i1 = 1; n2 = int((N-1)/2); i2 = i1+n2 
        FltrCoef = LOPT[i1:i2].astype(np.complex)
        # differentitation in frequency domain
        iW = ((np.arange(n2)+i1)*OMEGA*1j)**diff
        # multiply spectrum with filter coefficient
        Fltr_Spectrum[i1:i2] = Dat_F[i1:i2] * FltrCoef * iW
        # copy first half of modified spectrum to last half
        Fltr_Spectrum[-n2:] = np.conj(Fltr_Spectrum[i1:i2])[::-1]
    
        Fltr_Spectrum[0] = Dat_F[0]
    
        # Fltr_Spectrum[0] values
        # The derivatives of Fltr_Spectrum[0] are 0
        # Mean if diff = 0
        Fltr_Spectrum[0] = 0 if diff>0 else Dat_F[0]
        
        # Inverse fourier transform back in time domain
        Dat_T = np.fft.ifft(Fltr_Spectrum)
        Dat_T[-1] = np.real(Dat_T[0]) + 1j*np.imag(Dat_T[-1])

        # This ist the smoothed time series (baseline added)
        if diff==0:
            Smoothed_Data = np.real(Dat_T) + Dat[0] + SLOPE * X / delt
        elif diff==1:
            First_Diff = np.real(Dat_T) + SLOPE / delt
        elif diff==2:
            Secnd_Diff = np.real(Dat_T)
        
    if second_deriv:
        return Smoothed_Data, First_Diff, Secnd_Diff
    elif first_deriv:
        return Smoothed_Data, First_Diff
    else:
        return Smoothed_Data
        


