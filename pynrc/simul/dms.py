# Necessary for Python 2.6 and later
#from __future__ import division, print_function
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import datetime, os

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
from astropy import units as u

# STScI modules
import pysiaf
from pysiaf import rotations
from jwst.datamodels import Level1bModel

from pynrc.maths.coords import det_to_sci, sci_to_det

# Program bar
from tqdm.auto import trange, tqdm

def dec_to_base36(val):
    """Convert decimal number to base 36 (0-Z)"""

    digits ='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    res = '' 
    while val:
        val, i = np.divmod(val, 36)
        res = digits[i] + res
        
    return res

def jw_obs_id(pid, obs_num, visit_num, visit_gp, seq_id, act_id, exp_num):
    """ JWST Observation info and file naming convention
    
    jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>(-<”seg”NNN>)_<detector>_<prodType>.fits

    ppppp: program ID number
    ooo: observation number
    vvv: visit number
    
    gg: visit group
    s: parallel sequence ID (1=prime, 2-5=parallel)
    aa: activity number (base 36)
    
    eeeee: exposure number
    segNNN: the text “seg” followed by a three-digit segment number (optional)
    
    detector: detector name (e.g. ‘nrca1’, ‘nrcblong’, ‘mirimage’)
    
    prodType: product type identifier (e.g. ‘uncal’, ‘rate’, ‘cal’)
    
    An example Stage 1 product FITS file name is:
    jw93065002001_02101_00001_nrca1_rate.fits
    """
    
    act_id_b36 = dec_to_base36(act_id)
    
    res = {}
    res['program_number']     = '{:05d}'.format(int(pid))       # Program ID number
    res['observation_number'] = '{:03d}'.format(int(obs_num))   # Observation number
    res['visit_number']       = '{:03d}'.format(int(visit_num)) # Visit number
    res['visit_group']        = '{:02d}'.format(int(visit_gp))  # Visit group identifier
    res['sequence_id']        = '{:01d}'.format(int(seq_id))    # Parallel sequence ID (1=prime, 2-5=parallel)
    res['activity_id']        = '{:0>2}'.format(act_id_b36)     # Activity number (base 36)
    res['exposure_number']    = '{:05d}'.format(int(exp_num))   # Exposure Number
    
    # Visit identifer
    visit_id = res['program_number'] + res['observation_number'] + res['visit_number']
    # Parallel program info
    par_pid    = '{:05d}'.format(0)
    par_obsnum = '{:03d}'.format(0)
    par_info = par_pid + par_obsnum + res['visit_group'] + res['sequence_id'] + res['activity_id']
    # Observation ID
    obs_id = 'V' + visit_id + 'P' + par_info
    
    res['visit_id'] = visit_id
    res['obs_id']   = obs_id
        
    return res

def DMS_filename(obs_id_info, detname, segNum=None, prodType='uncal'):
    """
    jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>(-<”seg”NNN>)_<detector>_<prodType>.fits
    """
    
    vid = obs_id_info['visit_id']
    vgp = obs_id_info['visit_group']
    sid = obs_id_info['sequence_id']
    aid = obs_id_info['activity_id']
    eid = obs_id_info['exposure_number']
    
    detname = detname.lower()
    if 'nrc' in detname:
        detname = detname[3:]
    if 'long' in detname:
        detname = detname[0] + '5'
    detname = 'nrc' + detname

    #fname = f'jw{vid}_{vgp}{sid}{aid}_{eid}_nrc[a-b][1-5]_[uncal,rate,cal].fits' 
        
    part1 = f'jw{vid}_{vgp}{sid}{aid}_{eid}'
    part2 = "" if segNum is None else "-seg{:.0f}".format(segNum)
    part3 = '_' + detname + '_' + prodType + '.fits'
    
    fname = part1 + part2 + part3
    return fname

###  Copied and modified from MIRAGE
def create_group_entry(integration, groupnum, endday, endmilli, endsubmilli, endgroup,
                       xd, yd, gap, comp_code, comp_text, barycentric, heliocentric):
    """Add the GROUP extension to the output file

    From an example Mark Kyprianou sent:
    Parameters
    ----------
    integration : int
        Integration number
    group_number : int
        Group number
    endday : int
        Days since Jan 1 2000
    endmilli : integer
        Milliseconds of the day for given time
    endsubmilli : int
        Time since last millisecond?
    endgroup : str
        End group time, e.g. '2016-01-18T02:43:26.061'
    xd : int
        Number_of_columns e.g. 2048
    yd : int
        Number_of_rows e.g. 2048
    gap : int
        Number of gaps in telemetry
    comp_code : int
        Completion code number e.g. 0 (nominal?)
    comp_text : str
        Completion code text e.g. 'COMPLETE'-from howard
                                  'Normal Completion' - from mark
    barycentric : float
        Barycentric end time (mjd) 57405.11165225
    heliocentric : float
        Heliocentric end time (mjd) 57405.1163058
    Returns
    -------
    group : nump.ndarray
        Input values organized into format needed for group entry in
        JWST formatted file
    """
    group = np.ndarray(
        (1, ),
        dtype=[
            ('integration_number', '<i2'),
            ('group_number', '<i2'),
            ('end_day', '<i2'),
            ('end_milliseconds', '<i4'),
            ('end_submilliseconds', '<i2'),
            ('group_end_time', 'S26'),
            ('number_of_columns', '<i2'),
            ('number_of_rows', '<i2'),
            ('number_of_gaps', '<i2'),
            ('completion_code_number', '<i2'),
            ('completion_code_text', 'S36'),
            ('bary_end_time', '<f8'),
            ('helio_end_time', '<f8')
        ]
    )
    group[0]['integration_number'] = integration
    group[0]['group_number'] = groupnum
    group[0]['end_day'] = endday
    group[0]['end_milliseconds'] = endmilli
    group[0]['end_submilliseconds'] = endsubmilli
    group[0]['group_end_time'] = endgroup
    group[0]['number_of_columns'] = xd
    group[0]['number_of_rows'] = yd
    group[0]['number_of_gaps'] = gap
    group[0]['completion_code_number'] = comp_code
    group[0]['completion_code_text'] = comp_text
    group[0]['bary_end_time'] = barycentric
    group[0]['helio_end_time'] = heliocentric
    return group
    
###  Copied and modified from MIRAGE
def populate_group_table(starttime, grouptime, ramptime, numint, numgroup, ny, nx):
    """Create some reasonable values to fill the GROUP extension table.
    These will not be completely correct because access to other ssb
    scripts and more importantly, databases, is necessary. But they should be
    close.
    Parameters
    ----------
    starttime : astropy.time.Time
        Starting time of exposure
    grouptime : float
        Exposure time of a single group (seconds)
    ramptime : float
        Exposure time of the entire exposure (seconds)
    numint : int
        Number of integrations in data
    numgroup : int
        Number of groups per integration
    ny : int
        Number of pixels in the y dimension
    nx : int
        Number of pixels in the x dimension
    Returns
    -------
    grouptable : numpy.ndarray
        Group extension data for all groups in the exposure
    """
    # Create the table with a first row populated by garbage
    grouptable = create_group_entry(999, 999, 0, 0, 0, 'void', 0, 0, 0, 0, 'void', 1., 1.)

    # Quantities that are fixed for all exposures
    compcode = 0
    comptext = 'Normal Completion'
    numgap = 0

    # May want to ignore warnings as astropy.time.Time will give a warning
    # related to unknown leap seconds if the date is too far in
    # the future.
    baseday = Time('2020-01-01T00:00:00')

    # Integration start times
    rampdelta  = TimeDelta(ramptime, format='sec')
    groupdelta = TimeDelta(grouptime, format='sec')
    intstarts  = starttime + (np.arange(numint)*rampdelta)

    for integ in range(numint):
        groups = np.arange(1, numgroup+1)
        groupends = intstarts[integ] + (np.arange(1, numgroup+1)*groupdelta)
        endday = (groupends - baseday).jd

        # If the integration has a single group, force endday to be an array
        if isinstance(endday, float):
            endday = np.array([endday])
        enddayint = [np.int(s) for s in endday]

        # Now to get end_milliseconds, we need milliseconds from the beginning
        # of the day
        inday = TimeDelta(endday - enddayint, format='jd')
        endmilli = inday.sec * 1000.
        endmilli_int = [np.int(s) for s in endmilli]

        # Submilliseconds - just use a random number
        endsubmilli = (endmilli - endmilli_int)*1000
        # endsubmilli = np.random.randint(0, 1000, len(endmilli))

        # Group end time. need to remove : and - and make lowercase t
        groupending = groupends.isot

        # Approximate these as just the group end time in mjd
        barycentric = groupends.mjd
        heliocentric = groupends.mjd

        # For the case of an integration with a single group, force quantities to be
        # arrays so that everything is consistent
        if isinstance(groupending, str):
            groupending = np.array([groupending])
            barycentric = np.array([barycentric])
            heliocentric = np.array([heliocentric])

        for grp, day, milli, submilli, grpstr, bary, helio in zip(groups, endday, endmilli,
                                                                  endsubmilli, groupending,
                                                                  barycentric, heliocentric):
            entry = create_group_entry(integ+1, grp, day, milli, submilli, grpstr, nx, ny,
                                       numgap, compcode, comptext, bary, helio)
            grouptable = np.vstack([grouptable, entry])

    # Now remove the top garbage row from the table
    grouptable = grouptable[1:]
    return grouptable


def update_dms_headers(filename):
    """
    Given the name of a valid partially populated level 1b JWST file,
    add a couple simple WCS parameters from the SIAF keywords in that
    file that contain information about the telescope pointing. 

    It presumes all the accessed keywords are present from the JWST
    pipeline data model.

    Parameters
    ----------
    filename : str
        file name

    """
    hdulist = fits.open(filename, mode='update')
    pheader = hdulist[0].header
    fheader = hdulist[1].header

    local_roll  = fheader['ROLL_REF']
    v3idlyang   = fheader['V3I_YANG']
    vparity     = fheader['VPARITY']
    pa_aper_deg = local_roll - vparity * v3idlyang

    D2R = np.pi / 180.0
    fheader['PC1_1'] = -np.cos(pa_aper_deg * D2R)
    fheader['PC1_2'] = np.sin(pa_aper_deg * D2R)
    fheader['PC2_1'] = np.sin(pa_aper_deg * D2R)
    fheader['PC2_2'] = np.cos(pa_aper_deg * D2R)
    fheader['RA_REF'] = fheader['CRVAL1']
    fheader['DEC_REF'] = fheader['CRVAL2']
    
    # Now we need to adjust the datamodl header keyword
    # If we leave it as Level1bModel, the pipeline doesn't
    # work properly
    if '1b' in pheader['DATAMODL']:
        pheader['DATAMODL'] = 'RampModel'

    hdulist.flush()
    hdulist.close()

def obs_params_populate(obs, target_name, date_obs, time_obs, pa_v3, obs_id_info, 
                        time_exp_offset=0, segNum=None, int_range=None, **kwargs):
    """
    Populate the observation parameters dictionary with keywords
    necessary to input into JWST pipeline data model container
    in preparation for saving the DMS FITS file.
    """
    
    det = obs.Detectors[0]
    siaf_ap = obs.siaf_ap
    
    c = SkyCoord.from_name(target_name)
    
    if int_range is None:
        integration_start = 1
        integration_end   = det.nint
        nint_seg = det.nint
    else:
        integration_start = int_range[0] + 1
        integration_end   = int_range[1]
        nint_seg = integration_end - integration_start + 1
        
    # Start time for integrations considered in this segment
    start_time_string = date_obs + 'T' + time_obs
    t_offset_sec = (integration_start-1)*det.time_total_int2 + time_exp_offset
    start_time_int = Time(start_time_string) + t_offset_sec*u.second
    
    obs_params = {
        # Proposal info
        'pi_name'          : 'UNKNOWN',
        'title'            : 'UNKNOWN',
        'category'         : 'UNKNOWN',
        'sub_category'     : 'UNKNOWN',
        'science_category' : 'UNKNOWN',

        # Target info
        'target_name'  : target_name,
        'catalog_name' : 'UNKNOWN',
        'ra'           : c.ra.deg,
        'dec'          : c.dec.deg,
        'pa_v3'        : pa_v3,
        'siaf_ap'      : siaf_ap,

        # Observation info
        'obs_id_info' : obs_id_info,
        'obs_label'   : 'UNKNOWN',
        'date-obs'    : date_obs,
        'time-obs'    : time_obs,
        
        # Instrument configuration
        'module'  : obs.module,
        'channel' : 'LONG' if 'LW' in obs.channel else 'SHORT', 
        'detector': det.detname,
        'filter'  : obs.filter,
        'pupil'   : obs.pupil,
        # Observation Type
        'exp_type' : 'UNKNOWN',

        'subarray_name' : obs.get_subarray_name(),
        # subarray_bounds indexed to zero, but values in header should be indexed to 1.
        'xstart'   : det.x0+1,
        'ystart'   : det.y0+1,
        'xsize'    : det.xpix,
        'ysize'    : det.ypix,   
        'fastaxis' : det.fastaxis,
        'slowaxis' : det.slowaxis,

        # MULTIACCUM
        'readpatt'         : det.multiaccum.read_mode,
        'nframes'          : det.multiaccum.nf,
        'ngroups'          : det.multiaccum.ngroup,
        'nints'            : det.multiaccum.nint,
        'sample_time'      : int(1e6/det._pixel_rate),
        'frame_time'       : det.time_frame,
        'group_time'       : det.time_group,
        'groupgap'         : det.multiaccum.nd2,
        'nresets1'         : det.multiaccum.nr1,
        'nresets2'         : det.multiaccum.nr2,
        'integration_time' : det.time_int,
        'exposure_time'    : det.time_exp,
        'tint_plus_overhead' : det.time_total_int2,
        'texp_plus_overhead' : det.time_total,

        # Exposure Start time relative to TIME-OBS (seconds)
        'texp_start_relative' : time_exp_offset,
        # Create INT_TIMES table, to be saved in INT_TIMES extension
        # Currently, this is all integrations within the exposure, despite segment
        'int_times' : det.int_times_table(date_obs, time_obs, offset_seconds=time_exp_offset),
        'integration_start' : integration_start,
        'integration_end'   : integration_end,
        # Group times only populate for the current 
        'group_times'       : populate_group_table(start_time_int, det.time_group, det.time_total_int2, 
                                                   nint_seg, det.multiaccum.ngroup, det.xpix, det.ypix),

        # Dither information defaults (update later)
        'primary_type'          : 'NONE',     # Primary dither pattern name
        'position_number'       : 1,          # Primary dither position number
        'total_points'          : 1,          # Total number of primary dither positions
        'pattern_size'          : 'DEFAULT',  # Primary dither pattern size 
        'subpixel_type'         : 'NONE',     # Subpixel dither pattern name
        'subpixel_number'       : 1,          # Subpixel dither position number
        'subpixel_total_points' : 1,          # Total number of subpixel dither positions
        'x_offset'              : 0.0,        # Dither pointing offset from starting position in x (arcsec)
        'y_offset'              : 0.0,        # Dither pointing offset from starting position in y (arcsec)
    }
    
    for key in kwargs:
        obs_params[key] = kwargs[key]
    
    # Create output filename
    obs_params['filename'] = DMS_filename(obs_id_info, det.detname, segNum=segNum, prodType='uncal')
    
    return obs_params

def create_DMS_HDUList(sci_data, zero_data, obs_params):
    
    outModel = Level1bModel(data=sci_data, zeroframe=zero_data)
    outModel.meta.model_type = 'RampModel'
    
    outModel.meta.origin = 'STScI'
    outModel.meta.filetype = 'raw'
    
    # Proposal information
    outModel.meta.program.pi_name          = obs_params['pi_name']
    outModel.meta.program.title            = obs_params['title']
    outModel.meta.program.category         = obs_params['category']
    outModel.meta.program.sub_category     = obs_params['sub_category']
    outModel.meta.program.science_category = obs_params['science_category']
    outModel.meta.program.continuation_id  = 0

    # Date and time of observation
    outModel.meta.observation.date = obs_params['date-obs']
    outModel.meta.observation.time = obs_params['time-obs']
    start_time_string = obs_params['date-obs'] + 'T' + obs_params['time-obs']
    outModel.meta.date = start_time_string

    # Science target information
    ra, dec = (obs_params['ra'], obs_params['dec'])
    outModel.meta.target.proposer_name = obs_params['target_name']
    outModel.meta.target.catalog_name  = obs_params['catalog_name']
    outModel.meta.target.ra  = ra
    outModel.meta.target.dec = dec
    outModel.meta.coordinates.reference_frame = 'ICRS'
    
    # Observation Type
    # Possible types:
    #   NRC_DARK, NRC_FLAT, NRC_LED, NRC_GRISM
    #   NRC_TACQ, NRC_TACONFIRM, NRC_FOCUS
    #   NRC_IMAGE, NRC_CORON, NRC_TSIMAGE, NRC_TSGRISM, NRC_WFSS
    outModel.meta.exposure.type = obs_params['exp_type']
    # Specify whether the exposure is part of a TSO observation
    if outModel.meta.exposure.type.lower() in ['nrc_tsimage', 'nrc_tsgrism']:
        outModel.meta.visit.tsovisit = True
    else:
        outModel.meta.visit.tsovisit = False

    # Instrument info
    outModel.meta.telescope           = 'JWST'
    outModel.meta.instrument.name     = 'NIRCAM'
    outModel.meta.instrument.module   = obs_params['module']
    outModel.meta.instrument.channel  = obs_params['channel']
    outModel.meta.instrument.detector = obs_params['detector']
    
    # Filter and pupil info
    filt = obs_params['filter']
    pupil = obs_params['pupil']
    if 'GRISM0' in pupil:
        pupil = 'GRISMR'
    if 'GRISM90' in pupil:
        pupil = 'GRISMC'
    filt  = 'UNKNOWN' if filt  is None else filt
    pupil = 'UNKNOWN' if pupil is None else pupil
    outModel.meta.instrument.filter = filt
    outModel.meta.instrument.pupil  = pupil

    # Detector information 
    outModel.meta.subarray.name = obs_params['subarray_name']
    # subarray_bounds indexed to zero, but values in header should be indexed to 1.
    outModel.meta.subarray.xstart = obs_params['xstart']
    outModel.meta.subarray.ystart = obs_params['ystart']
    outModel.meta.subarray.xsize  = obs_params['xsize']
    outModel.meta.subarray.ysize  = obs_params['ysize']
    outModel.meta.subarray.fastaxis = obs_params['fastaxis']
    outModel.meta.subarray.slowaxis = obs_params['slowaxis']
    
    # MULTIACCUM Settings
    outModel.meta.exposure.readpatt = obs_params['readpatt']
    outModel.meta.exposure.nframes  = obs_params['nframes']
    outModel.meta.exposure.ngroups  = obs_params['ngroups']
    outModel.meta.exposure.nints    = obs_params['nints']
    outModel.meta.exposure.integration_start = obs_params['integration_start']
    outModel.meta.exposure.integration_end   = obs_params['integration_end']
    outModel.meta.exposure.nresets_at_start      = obs_params['nresets1']
    outModel.meta.exposure.nresets_between_ints  = obs_params['nresets2']
        
    outModel.meta.exposure.sample_time           = obs_params['sample_time']
    outModel.meta.exposure.frame_time            = obs_params['frame_time']
    outModel.meta.exposure.group_time            = obs_params['group_time']
    outModel.meta.exposure.groupgap              = obs_params['groupgap']
    outModel.meta.exposure.integration_time      = obs_params['integration_time']
    outModel.meta.exposure.exposure_time         = obs_params['exposure_time']
    # INT_TIMES table to be saved in INT_TIMES extension
    outModel.int_times = obs_params['int_times']
    
    # Total time to complete an integration (including reset fraems)
    tint_tot = obs_params['tint_plus_overhead']
    texp_tot = obs_params['texp_plus_overhead']
    start_time = Time(start_time_string) + obs_params['texp_start_relative']*u.second

    # set the exposure start time
    outModel.meta.exposure.start_time = start_time.mjd 
    outModel.meta.exposure.end_time = start_time.mjd + texp_tot / (24*3600.)
    outModel.meta.exposure.mid_time = start_time.mjd + texp_tot / (24*3600.) / 2.
    outModel.meta.exposure.duration = texp_tot

    # Observation/program ID information
    outModel.meta.observation.observation_label = obs_params['obs_label']
    obs_id_info = obs_params['obs_id_info']
    outModel.meta.observation.program_number     = obs_id_info['program_number']
    outModel.meta.observation.observation_number = obs_id_info['observation_number']
    outModel.meta.observation.visit_number       = obs_id_info['visit_number']
    outModel.meta.observation.visit_group     = obs_id_info['visit_group']
    outModel.meta.observation.sequence_id     = obs_id_info['sequence_id']
    outModel.meta.observation.activity_id     = obs_id_info['activity_id']
    outModel.meta.observation.exposure_number = obs_id_info['exposure_number']
    outModel.meta.observation.visit_id        = obs_id_info['visit_id']
    outModel.meta.observation.obs_id          = obs_id_info['obs_id']

    # Telescope pointing
    pa_v3 = obs_params['pa_v3']
    siaf_ap = obs_params['siaf_ap']
    # ra_v1, dec_v1, and pa_v3 are not used by the level 2 pipelines
    # compute pointing of V1 axis
    attitude_matrix = rotations.attitude(siaf_ap.V2Ref, siaf_ap.V3Ref, ra, dec, pa_v3)
    pointing_ra_v1, pointing_dec_v1 = rotations.pointing(attitude_matrix, 0., 0.)
    outModel.meta.pointing.ra_v1 = pointing_ra_v1
    outModel.meta.pointing.dec_v1 = pointing_dec_v1
    outModel.meta.pointing.pa_v3 = pa_v3
    
    # Dither information
    outModel.meta.dither.primary_type    = obs_params['primary_type']
    outModel.meta.dither.position_number = obs_params['position_number']
    outModel.meta.dither.total_points    = obs_params['total_points']
    outModel.meta.dither.pattern_size    = obs_params['pattern_size']
    outModel.meta.dither.subpixel_type   = obs_params['subpixel_type']
    outModel.meta.dither.subpixel_number = obs_params['subpixel_number']
    outModel.meta.dither.subpixel_total_points = obs_params['subpixel_total_points']
    outModel.meta.dither.x_offset = obs_params['x_offset']
    outModel.meta.dither.y_offset = obs_params['y_offset']

    # WCS Info
    outModel.meta.aperture.name = siaf_ap.AperName
    outModel.meta.wcsinfo.wcsaxes = 2
    outModel.meta.wcsinfo.crval1 = ra
    outModel.meta.wcsinfo.crval2 = dec
    outModel.meta.wcsinfo.crpix1 = siaf_ap.XSciRef
    outModel.meta.wcsinfo.crpix2 = siaf_ap.YSciRef
    outModel.meta.wcsinfo.ctype1 = 'RA---TAN'
    outModel.meta.wcsinfo.ctype2 = 'DEC--TAN'
    outModel.meta.wcsinfo.cunit1 = 'deg'
    outModel.meta.wcsinfo.cunit2 = 'deg'
    outModel.meta.wcsinfo.v2_ref = siaf_ap.V2Ref
    outModel.meta.wcsinfo.v3_ref = siaf_ap.V3Ref
    outModel.meta.wcsinfo.vparity = siaf_ap.VIdlParity
    outModel.meta.wcsinfo.v3yangle = siaf_ap.V3IdlYAngle
    outModel.meta.wcsinfo.cdelt1 = siaf_ap.XSciScale / 3600.
    outModel.meta.wcsinfo.cdelt2 = siaf_ap.YSciScale / 3600.
    outModel.meta.wcsinfo.siaf_xref_sci = siaf_ap.XSciRef
    outModel.meta.wcsinfo.siaf_yref_sci = siaf_ap.YSciRef
    # V3 roll angle at the ref point
    roll_ref = rotations.posangle(attitude_matrix, siaf_ap.V2Ref, siaf_ap.V3Ref)
    if roll_ref < 0:
        roll_ref += 360
    outModel.meta.wcsinfo.roll_ref = roll_ref
    
    outModel.meta.filename = obs_params['filename']
    
    return outModel
