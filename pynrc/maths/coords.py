from __future__ import absolute_import, division, print_function, unicode_literals

#__all__ = ['pad_or_cut_to_size', 'frebin', \
#           'fshift', 'fourier_imshift', 'shift_subtract', 'align_LSQ']
import numpy as np
import logging
_log = logging.getLogger('pynrc')

__epsilon = np.finfo(float).eps

def dist_image(image, pixscale=None, center=None, return_theta=False):
    """Pixel distances
    
    Returns radial distance in units of pixels, unless pixscale is specified.
    Use the center keyword to specify the position (in pixels) to measure from.
    If not set, then the center of the image is used.

    return_theta will also return the angular position of each pixel relative 
    to the specified center
    
    Parameters
    ----------
    image : ndarray
        Input image to find pixel distances (and theta).
    pixscale : int, None
        Pixel scale (such as arcsec/pixel or AU/pixel) that
        dictates the units of the output distances. If None,
        then values are in units of pixels.
    center : tuple
        Location (x,y) in the array calculate distance. If set 
        to None, then the default is the array center pixel.
    return_theta : bool
        Also return the angular positions as a 2nd element.
    """
    y, x = np.indices(image.shape)
    if center is None:
        center = tuple((a - 1) / 2.0 for a in image.shape[::-1])
    x = x - center[0]
    y = y - center[1]

    rho = np.sqrt(x**2 + y**2)
    if pixscale is not None: rho *= pixscale

    if return_theta:
        return rho, np.arctan2(-x,y)*180/np.pi
    else:
        return rho

def xy_to_rtheta(x, y):
    """Convert (x,y) to (r,theta)
    
    Input (x,y) coordinates and return polar cooridnates that use
    the WebbPSF convention (theta is CCW of +Y)
    
    Input can either be a single value or numpy array.

    Parameters
    ---------
    x : float or array
        X location values
    y : float or array
        Y location values
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(-x,y)*180/np.pi

    if np.size(r)==1:
        if np.abs(x) < __epsilon: x = 0
        if np.abs(y) < __epsilon: y = 0
    else:
        r[np.abs(r) < __epsilon] = 0
        theta[np.abs(theta) < __epsilon] = 0

    return r, theta

def rtheta_to_xy(r, theta):
    """Convert (r,theta) to (x,y)
    
    Input polar cooridnates (WebbPSF convention) and return Carteesian coords
    in the imaging coordinate system (as opposed to RA/DEC)

    Input can either be a single value or numpy array.

    Parameters
    ---------
    r : float or array
        Radial offset from the center in pixels
    theta : float or array
        Position angle for offset in degrees CCW (+Y).
    """
    x = -r * np.sin(theta*np.pi/180.)
    y =  r * np.cos(theta*np.pi/180.)

    if np.size(x)==1:
        if np.abs(x) < __epsilon: x = 0
        if np.abs(y) < __epsilon: y = 0
    else:
        x[np.abs(x) < __epsilon] = 0
        y[np.abs(y) < __epsilon] = 0

    return x, y
    
def xy_rot(x, y, ang):

    """Rotate (x,y) positions to new coords
    
    Rotate (x,y) values by some angle. 
    Positive ang values rotate counter-clockwise.
    
    Parameters
    -----------
    x : float or array
        X location values
    y : float or array
        Y location values
    ang : float or array
        Rotation angle in degrees CCW
    """

    r, theta = xy_to_rtheta(x, y)    
    return rtheta_to_xy(r, theta+ang)

###########################################################################
#
#    Coordinate Systems
#
###########################################################################

def Tel2Sci_info(channel, coords, output="Sci"):
    """Telescope coords converted to Science coords
    
    Returns the detector name associated with these coordinates

    Parameters
    ----------
    channel : str
        'SW' or 'LW'
    coords : tuple
        Telescope coordinates (V2,V3) in arcsec.
    output : str
        Type of desired output coordinates. 

            * Det: pixels, in raw detector read out axes orientation
            * Sci: pixels, in conventional DMS axes orientation
            * Idl: arcsecs relative to aperture reference location.
            * Tel: arcsecs V2,V3
    """
    
    V2, V3 = coords
    
    # Figure out the detector and pixel position for some (V2,V3) coord
    mysiaf = webbpsf.webbpsf_core.SIAF('NIRCam')
    swa = ['A1', 'A2', 'A3', 'A4']
    swb = ['B1', 'B2', 'B3', 'B4']
    lwa = ['A5']
    lwb = ['B5']
    
    detnames = swa + swb if 'SW' in channel else lwa + lwb
    apnames = ['NRC'+det+'_FULL' for det in detnames]
    
    # Find center positions for each apname
    cens = []
    for apname in apnames:
        ap = mysiaf[apname]
        cens.append(ap.Tel2Sci(V2, V3))
    cens = np.array(cens)

    # Select that with the closest position
    dist = np.sqrt((cens[:,0]-1024)**2 + (cens[:,1]-1024)**2)
    ind = np.where(dist==dist.min())[0][0]
    
    # Find detector "science" coordinates
    detector = detnames[ind]
    apname = apnames[ind]
    ap = mysiaf[apname]
    detector_position = ap.convert(V2, V3, frame_from='Tel', frame_to=output)
    
    return detector, detector_position
    


def det_to_V2V3(image, detid):
    """Detector to V2/V3 coordinates
    
    Reorient image from detector coordinates to V2/V3 coordinate system.
    This places +V3 up and +V2 to the LEFT. Detector pixel (0,0) is assumed 
    to be in the bottom left. For now, we're simply performing axes flips.
    
    Parameters
    ----------
    image : ndarray
        Input image to tranform.
    detid : int or str
        NIRCam detector/SCA ID, either 481-490 or A1-B5.
    """
    
    # Check if SCA ID (481-489) where passed through detname rather than A1-B5
    try:
        detid = int(detid)
    except ValueError:
        detname = detid
    else:
        scaids = {481:'A1', 482:'A2', 483:'A3', 484:'A4', 485:'A5',
                  486:'B1', 487:'B2', 488:'B3', 489:'B4', 490:'B5'}
        detname = scaids[detid]
    
    xflip = ['A1','A3','A5','B2','B4']
    yflip = ['A2','A4','B1','B3','B5']
    
    for s in xflip:
        if detname in s:
            image = image[:,::-1] 
    for s in yflip:
        if detname in s:
            image = image[::-1,:] 
    
    return image
    
def V2V3_to_det(image, detid):
    """V2/V3 coordinate to detector orientation
    
    Reorient image from V2/V3 coordinates to detector coordinate system.
    Assumes +V3 up and +V2 to the LEFT. The result places the detector
    pixel (0,0) in the bottom left. For now, we're simply performing 
    axes flips.

    Parameters
    ----------
    image : ndarray
        Input image to tranform.
    detid : int or str
        NIRCam detector/SCA ID, either 481-490 or A1-B5.
    """
    
    # Flips occur along the same axis and manner as in det_to_V2V3()
    return det_to_V2V3(image, detid)
    
    
def plotAxes(ax, position=(0.9,0.1), label1='V2', label2='V3', dir1=[-1,0], dir2=[0,1],
             angle=0, alength=0.12, width=2, headwidth=8, color='w'):
    """Compass arrows
    
    Show V2/V3 coordinate axis on a plot. By default, this function will plot
    the compass arrows in the lower right position in sky-right coordinates
    (ie., North/V3 up, and East/V2 to the left). 
    
    Parameters
    ==========
    ax : axis
        matplotlib axis to plot coordiante arrows.
    position : tuple
        XY-location of joined arrows as a fraction (0.0-1.0).
    label1 : str
        Label string for horizontal axis (ie., 'E' or 'V2').
    label2 : str
        Label string for vertical axis (ie, 'N' or 'V3').
    dir1 : array like
        XY-direction values to point "horizontal" arrow.
    dir2 : array like 
        XY-direction values to point "vertical" arrow.
    angle : float
        Rotate coordinate axis by some angle. 
        Positive values rotate counter-clockwise.
    alength : float
        Length of arrow vectors as fraction of plot axis.
    width : float
        Width of the arrow in points.
    headwidth : float
        Width of the base of the arrow head in points.
    color : color
        Self-explanatory.
    """
    arrowprops={'color':color, 'width':width, 'headwidth':headwidth}
    
    dir1 = xy_rot(dir1[0], dir1[1], angle)
    dir2 = xy_rot(dir2[0], dir2[1], angle)

    
    for (label, direction) in zip([label1,label2], np.array([dir1,dir2])):
        ax.annotate("", xytext=position, xy=position + alength * direction,
                    xycoords='axes fraction', arrowprops=arrowprops)
        textPos = position + alength * direction*1.3
        ax.text(textPos[0], textPos[1], label, transform=ax.transAxes,
                horizontalalignment='center', verticalalignment='center',
                color=color, fontsize=12)
