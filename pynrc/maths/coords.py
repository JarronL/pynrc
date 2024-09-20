"""Telescope coordinate information"""

import numpy as np
import logging
_log = logging.getLogger('pynrc')

__epsilon = np.finfo(float).eps

import pysiaf
from pysiaf import JWST_PRD_VERSION, rotations, Siaf
# Create this once since it takes time to call multiple times
# from ..nrc_utils import siaf_nrc
siaf_nrc = Siaf('NIRCam')
siaf_nrc.generate_toc()

# Functions transferred to webbpsf_ext
from webbpsf_ext.coords import dist_image
from webbpsf_ext.coords import xy_to_rtheta, rtheta_to_xy, xy_rot
from webbpsf_ext.coords import ap_radec, radec_to_coord, radec_to_v2v3, v2v3_to_pixel
from webbpsf_ext.coords import get_NRC_v2v3_limits, NIRCam_V2V3_limits
from webbpsf_ext.coords import get_NRC_v2v3_limits as get_v2v3_limits # Original name
from webbpsf_ext.coords import plotAxes

# New stuff from webbpsf_ext
from webbpsf_ext.coords import gen_sgd_offsets, get_idl_offset, radec_offset
from webbpsf_ext.coords import jwst_point
from webbpsf_ext.coron_masks import det_to_sci, sci_to_det

def oversampled_coords(coords, oversample):
    """Transform image index to oversampled image index
    
    Assumes 0-based indexing where the center of a pixel
    corresponds to the integer index. That is, the center
    of the first pixel is at 0.0, and the center of the last
    pixel is at N-1.0 where N is the number of pixels.

    Parameters
    ----------
    coords : float or array-like
        Image index or indices
    oversample : int
        Oversampling factor

    Returns
    -------
    float or array-like
        Oversampled image index or indices
    """
    return (coords + 0.5) * oversample - 0.5

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
            ap = self.siaf_ap
        except:
            _log.warning("`self.siaf_ap` may not be set")
            apname = self.get_siaf_apname()
            if apname is None:
                _log.warning('No suitable aperture name defined to determine ref coords')
                return None, None, None
            else:
                _log.warning('`self.siaf_ap` not defined; assuming {}'.format(apname))
                ap = self.siaf[apname]
                
        if coord_frame=='tel':
            coord_vals = (ap.V2Ref, ap.V3Ref)
        elif coord_frame=='sci':
            coord_vals = (ap.XSciRef, ap.YSciRef)
        elif coord_frame=='det':
            coord_vals = (ap.XDetRef, ap.YDetRef)
        elif coord_frame=='idl':
            coord_vals = (0.0, 0.0)

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

