from __future__ import absolute_import, division, print_function, unicode_literals

#__all__ = ['pad_or_cut_to_size', 'frebin', \
#           'fshift', 'fourier_imshift', 'shift_subtract', 'align_LSQ']
import numpy as np
import logging
_log = logging.getLogger('pynrc')

__epsilon = np.finfo(float).eps

def dist_image(image, pixscale=None, center=None, return_theta=False):
    """
    Returns radial distance in units of pixels, unless pixscale is specified.
    Use the center keyword to specify the position (in pixels) to measure from.
    If not set, then the center of the image is used.

    return_theta will also return the angular position of each pixel relative 
    to the specified center
    
    center should be entered as (x,y)
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
    """
    Input (x,y) coordinates and return polar cooridnates that use
    the WebbPSF convention (theta is CCW of +Y)
    
    Input can either be a single value or numpy array.
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
    """
    Input polar cooridnates (WebbPSF convention) and return Carteesian coords
    in the imaging coordinate system (as opposed to RA/DEC)

    Input can either be a single value or numpy array.

    r     : Radial offset from the center in pixels
    theta : Position angle for offset in degrees CCW (+Y).
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

    """
    Rotate (x,y) values by some angle. 
    Positive ang values rotate counter-clockwise.
    """

    r, theta = xy_to_rtheta(x, y)    
    return rtheta_to_xy(r, theta+ang)

###########################################################################
#
#    Coordinate Systems
#
###########################################################################

def det_to_V2V3(image, detid):
    """
    Reorient image from detector coordinates to V2/V3 coordinate system.
    This places +V3 up and +V2 to the LEFT. Detector pixel (0,0) is assumed 
    to be in the bottom left. For now, we're simply performing axes flips. 
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
    """
    Reorient image from V2/V3 coordinates to detector coordinate system.
    Assumes +V3 up and +V2 to the LEFT. The result plances the detector
    pixel (0,0) in the bottom left. For now, we're simply performing 
    axes flips.
    """
    
    # Flips occur along the same axis and manner as in det_to_V2V3()
    return det_to_V2V3(image, detid)
    
    
def plotAxes(ax, position=(0.9,0.1), label1='V2', label2='V3', dir1=[-1,0], dir2=[0,1],
             angle=0, alength=0.12, width=2, headwidth=8, color='w'):
    """
    Show V2/V3 coordinate axis on a plot. By default, this function will plot
    the compass arrows in the lower right position in sky-right coordinates
    (ie., North/V3 up, and East/V2 to the left). 
    
    Parameters
    ==========
    ax - matplotlib axis to plot coordiante arrows.
    position - XY-location of joined arrows as a fraction (0.0-1.0).
    label1 - Label string for horizontal axis (ie., 'E' or 'V2').
    label2 - Label string for vertical axis (ie, 'N' or 'V3').
    dir1 - XY-direction values to point "horizontal" arrow.
    dir2 - XY-direction values to point "vertical" arrow.
    angle - Rotate coordinate axis by some angle. 
            Positive values rotate counter-clockwise.
    
    alength   - Length of arrow vectors as fraction of plot axis.
    width     - Width of the arrow in points.
    headwidth - Width of the base of the arrow head in points.
    color     - Self-explanatory.
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
