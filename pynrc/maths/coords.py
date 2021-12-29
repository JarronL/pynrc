"""Telescope coordinate information"""

import numpy as np
import logging
_log = logging.getLogger('pynrc')

__epsilon = np.finfo(float).eps

import pysiaf
from pysiaf import JWST_PRD_VERSION, rotations, Siaf
# Create this once since it takes time to call multiple times
siaf_nrc = Siaf('NIRCam')
siaf_nrc.generate_toc()

# Functions transferred to webbpsf_ext
from webbpsf_ext.coords import dist_image
from webbpsf_ext.coords import xy_to_rtheta, rtheta_to_xy, xy_rot
from webbpsf_ext.coords import ap_radec, radec_to_v2v3, v2v3_to_pixel
from webbpsf_ext.coords import get_NRC_v2v3_limits, NIRCam_V2V3_limits
from webbpsf_ext.coords import get_NRC_v2v3_limits as get_v2v3_limits # Original name
from webbpsf_ext.coords import plotAxes

# New stuff from webbpsf_ext
from webbpsf_ext.coords import gen_sgd_offsets, get_idl_offset, radec_offset
from webbpsf_ext.coords import jwst_point

###########################################################################
#
#    NIRCam Coordinate Systems
#
###########################################################################

def Tel2Sci_info(channel, coords, pupil=None, output='sci', return_apname=False, **kwargs):
    """Telescope coords converted to Science coords
    
    Returns the detector name and position associated with input coordinates.
    This is alway relative to a full frame detector aperture. The detector
    that is chosen is the one whose center is closest to the input coords.

    Parameters
    ----------
    channel : str
        'SW' or 'LW'
    coords : tuple
        Telescope coordinates (V2,V3) in arcsec.
    output : str
        Type of desired output coordinates. 

            * det: pixels, in raw detector read out axes orientation
            * sci: pixels, in conventional DMS axes orientation
            * idl: arcsecs relative to aperture reference location.
            * tel: arcsecs V2,V3
    """
    
    # Figure out the detector and pixel position for some (V2,V3) coord
    V2, V3 = coords
    
    if (pupil is not None) and ("LYOT" in pupil):
        # There are no full frame SIAF aperture for ModB coron
        if 'SW' in channel:
            detnames = ['A2', 'A4']
            apnames = ['NRCA2_FULL_MASK210R', 'NRCA4_FULL_MASKSWB']
        else:
            detnames = ['A5']
            apnames = ['NRCA5_FULL_MASK335R']
    else:
        swa = ['A1', 'A2', 'A3', 'A4']
        swb = ['B1', 'B2', 'B3', 'B4']
        lwa = ['A5']
        lwb = ['B5']

        detnames = swa + swb if 'SW' in channel else lwa + lwb
        apnames = ['NRC'+det+'_FULL' for det in detnames]

    # Find center positions for each apname
    cens = []
    for apname in apnames:
        ap = siaf_nrc[apname]
        try:
            vals = ap.tel_to_sci(V2, V3)
        except AttributeError:
            vals = ap.Tel2Sci(V2, V3)
        cens.append(vals)
    cens = np.array(cens)

    # Select that with the closest position
    dist = np.sqrt((cens[:,0]-1024)**2 + (cens[:,1]-1024)**2)
    ind = np.where(dist==dist.min())[0][0]
    
    # Find detector "science" coordinates
    detector = 'NRC'+detnames[ind]
    apname = apnames[ind]
    ap = siaf_nrc[apname]
    try:
        detector_position = ap.convert(V2, V3, 'tel', output.lower())
    except TypeError:
        detector_position = ap.convert(V2, V3, frame_from='Tel', frame_to=output)

    if return_apname:
        return detector, detector_position, apname
    else:
        return detector, detector_position

def siafap_sci_coords(inst, coord_vals=None, coord_frame='tel'):
    """
    Return the detector, sci position, and full frame aperture name
    for a set of coordinate values.
    """

    self = inst
    # Get a reference point if coord_vals not set
    if coord_vals is None:
        try:
            coord_vals = self.siaf_ap.reference_point(coord_frame)
        except:
            _log.warning("`self.siaf_ap` may not be set")
            apname = self.get_siaf_apname()
            if apname is None:
                _log.warning('No suitable aperture name defined to determine ref coords')
                return None, None, None
            else:
                _log.warning('`self.siaf_ap` not defined; assuming {}'.format(apname))
                ap = self.siaf[apname]
                coord_vals = ap.reference_point(coord_frame)

    # Determine V2/V3 coordinates
    detector = detector_position = apname = None
    v2 = v3 = None
    cframe = coord_frame.lower()
    if cframe=='tel':
        v2, v3 = coord_vals
    elif cframe in ['det', 'sci', 'idl']:
        x, y = coord_vals[0], coord_vals[1]
        try:
            v2, v3 = self.siaf_ap.convert(x,y, cframe, 'tel')
        except: 
            apname = self.get_siaf_apname()
            if apname is None:
                _log.warning('No suitable aperture name defined to determine V2/V3 coordinates')
            else:
                _log.warning('`self.siaf_ap` not defined; assuming {}'.format(apname))
                ap = self.siaf[apname]
                v2, v3 = ap.convert(x,y,cframe, 'tel')
            _log.warning('Update `self.siaf_ap` for more specific conversions to V2/V3.')
    else:
        _log.warning("coord_frame setting '{}' not recognized.".format(coord_frame))

    # Update detector, pixel position, and apname to pass to 
    if v2 is not None:
        res = Tel2Sci_info(self.channel, (v2, v3), output='sci', return_apname=True)
        detector, detector_position, apname = res

    return (detector, detector_position, apname)

def det_to_sci(image, detid):
    """ Detector to science orientation
    
    Reorient image from detector coordinates to 'sci' coordinate system.
    This places +V3 up and +V2 to the LEFT. Detector pixel (0,0) is assumed 
    to be in the bottom left. Simply performs axes flips.
    
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

    # Handle multiple array of images
    ndim = len(image.shape)
    if ndim==2:
        # Convert to image cube
        ny, nx = image.shape
        image = image.reshape([1,ny,nx])
    
    for s in xflip:
        if detname in s:
            image = image[:,:,::-1] 
    for s in yflip:
        if detname in s:
            image = image[:,::-1,:] 
    
    # Convert back to 2D if input was 2D
    if ndim==2:
        image = image.reshape([ny,nx])

    return image
    
def sci_to_det(image, detid):
    """ Science to detector orientation
    
    Reorient image from 'sci' coordinates to detector coordinate system.
    Assumes +V3 up and +V2 to the LEFT. The result places the detector
    pixel (0,0) in the bottom left. Simply performs axes flips.

    Parameters
    ----------
    image : ndarray
        Input image to tranform.
    detid : int or str
        NIRCam detector/SCA ID, either 481-490 or A1-B5.
    """
    
    # Flips occur along the same axis and manner as in det_to_sci()
    return det_to_sci(image, detid)
