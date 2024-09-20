"""DMS data format routines"""

import numpy as np
import os

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time, TimeDelta
from astropy import units as u

# STScI modules
import pysiaf
from pysiaf import rotations
from jwst.datamodels import Level1bModel
import warnings

from .apt import DMS_input
NONE_STR = str(None).upper()

import logging
_log = logging.getLogger('pynrc')

def dec_to_base36(val):
    """Convert decimal integer to base 36 (0-Z)"""

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
    aa: activity number (base 36) (only for WFSC, coarse and fine phasing)
    
    eeeee: exposure number
    segNNN: the text “seg” followed by a three-digit segment number (optional)
    
    detector: detector name (e.g. 'nrca1', 'nrcblong', 'mirimage')
    
    prodType: product type identifier (e.g. 'uncal', 'rate', 'cal')
    
    An example Stage 1 product FITS file name is:
    jw93065002001_02101_00001_nrca1_rate.fits
    """
    
    act_id_b36 = act_id if isinstance(act_id, str) else dec_to_base36(int(act_id))
    
    res = {}
    res['program_number']     = '{:05d}'.format(int(pid))       # Program ID number
    res['observation_number'] = '{:03d}'.format(int(obs_num))   # Observation number
    res['visit_number']       = '{:03d}'.format(int(visit_num)) # Visit number
    res['visit_group']        = '{:02d}'.format(int(visit_gp))  # Visit group identifier
    res['sequence_id']        = '{:01d}'.format(int(seq_id))    # Parallel sequence ID (1=prime, 2-5=parallel)
    res['activity_id']        = '{:0>2}'.format(act_id_b36)     # Activity number (base 36)
    res['exposure_number']    = '{:05d}'.format(int(exp_num))   # Exposure Number
    
    # Visit identifier
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
    part2 = "" if segNum is None else f"-seg{segNum:.0f}"
    part3 = '_' + detname + '_' + prodType + '.fits'
    
    fname = part1 + part2 + part3
    return fname

def filename_visit_info(filename):
    """
    Extract visit information from filename

    jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>(-<”seg”NNN>)_<detector>_<prodType>.fits

    ppppp: program ID number
    ooo: observation number
    vvv: visit number
    
    gg: visit group
    s: parallel sequence ID (1=prime, 2-5=parallel)
    aa: activity number (base 36) (only for WFSC, coarse and fine phasing)
    
    eeeee: exposure number
    segNNN: the text “seg” followed by a three-digit segment number (optional)
    
    detector: detector name (e.g. 'nrca1', 'nrcblong', 'mirimage')

    prodType: product type identifier (e.g. 'uncal', 'rate', 'cal')
    """
    from ..nrc_utils import get_detname
    
    fname = filename.split('/')[-1]
    fname_arr = fname.split('_')
    
    i0 = fname_arr[0].find('jw') + 2 # Increment index by 2
    pid = int(fname_arr[0][i0:i0+5])
    oid = int(fname_arr[0][i0+5:i0+8])
    vid = int(fname_arr[0][i0+8:])
    
    grp = int(fname_arr[1][0:2])
    seq = int(fname_arr[1][2])
    act = int(fname_arr[1][3:])
    
    expid = fname_arr[2]
    if '-seg' in expid:
        segid = int(expid.split('-')[1][3:])
        expid = int(expid.split('-')[0])
    else:
        expid = int(expid)
        segid = None
    
    det = fname_arr[3]
    
    res = jw_obs_id(pid, oid, vid, grp, seq, act, expid)
    res['detector'] = get_detname(det)
    res['segid'] = segid
    res['filename'] = fname
    
    return res

###  Copied and modified from MIRAGE
def create_group_entry(integration, groupnum, endday, endmilli, endsubmilli, endgroup,
                       xd, yd, gap, comp_code, comp_text, barycentric, heliocentric):
    """Add the GROUP extension to the output file

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

    import warnings

    # Create the table with a first row populated by garbage
    grouptable = create_group_entry(999, 999, 0, 0, 0, 'void', 0, 0, 0, 0, 'void', 1., 1.)

    # Quantities that are fixed for all exposures
    compcode = 0
    comptext = 'Normal Completion'
    numgap = 0

    # May want to ignore warnings as astropy.time.Time will give a warning
    # related to unknown leap seconds if the date is too far in
    # the future.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
        enddayint = [int(s) for s in endday]

        # Now to get end_milliseconds, we need milliseconds from the beginning
        # of the day
        inday = TimeDelta(endday - enddayint, format='jd')
        endmilli = inday.sec * 1000.
        endmilli_int = [int(s) for s in endmilli]

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


def level1b_data_model(obs_params, sci_data=None, zero_data=None):

    """
    obs_params : dict
        Dictionary of parameters to populate DMS header. 
        See `create_DMS_HDUList` in dms.py.
    """

    if sci_data is None:
        sci_shape = (obs_params['nints'], obs_params['ngroups'], obs_params['ysize'], obs_params['xsize'])
        zero_shape = (obs_params['nints'], obs_params['ysize'], obs_params['xsize'])
        sci_data  = np.zeros(sci_shape, dtype='uint16')
        zero_data = np.zeros(zero_shape, dtype='uint16')

    # Make sure data is a 4D array
    if len(sci_data.shape)<4:
        nz, ny, nx = sci_data.shape
        sci_data  = sci_data.reshape([1,nz,ny,nx])
        zero_data = zero_data.reshape([1,ny,nx])

    try:
        # A ModuleNotFoundError bug exists in current version of jwst pipeline
        # Should work to just try again if encountered
        outModel = Level1bModel(data=sci_data, zeroframe=zero_data)
    except ModuleNotFoundError:
        outModel = Level1bModel(data=sci_data, zeroframe=zero_data)
    # Update from 'Level1bModel' to 'RampModel'
    # TODO: Is this correct??
    # outModel.meta.model_type = 'RampModel'
    
    outModel.meta.origin = 'STScI'
    outModel.meta.filetype = 'uncalibrated' # 'raw'
    
    # Proposal information
    outModel.meta.program.pi_name          = obs_params['pi_name']
    outModel.meta.program.title            = obs_params['title']
    outModel.meta.program.category         = obs_params['category']
    outModel.meta.program.sub_category     = obs_params['sub_category']
    outModel.meta.program.science_category = obs_params['science_category']
    outModel.meta.program.continuation_id  = 0

    # Science target information
    outModel.meta.target.proposer_name = obs_params['target_name']
    outModel.meta.target.catalog_name  = obs_params['catalog_name']
    outModel.meta.target.ra  = obs_params['ra']
    outModel.meta.target.dec = obs_params['dec']
    outModel.meta.target.proposer_ra  = obs_params['ra']
    outModel.meta.target.proposer_dec = obs_params['dec']
    outModel.meta.target.proper_motion_ra = obs_params.get('mu_RA')
    outModel.meta.target.proper_motion_dec = obs_params.get('mu_DEC')
    outModel.meta.target.proper_motion_epoch = '2000-01-01 00:00:00.0000000'
    outModel.meta.coordinates.reference_frame = 'ICRS'
    
    # Exposure Type
    # Possible types:
    #   NRC_DARK, NRC_FLAT, NRC_LED, NRC_FOCUS, 
    #   NRC_TACQ, NRC_TACONFIRM
    #   NRC_IMAGE, NRC_CORON, NRC_GRISM
    #   NRC_WFSS, NRC_TSIMAGE, NRC_TSGRISM
    outModel.meta.exposure.type = obs_params['exp_type'].upper()

    # Specify whether the exposure is part of a TSO observation
    if outModel.meta.exposure.type.lower() in ['nrc_tsimage', 'nrc_tsgrism']:
        outModel.meta.visit.tsovisit = True
    else:
        outModel.meta.visit.tsovisit = False
    # Background Target?
    outModel.meta.observation.bkgdtarg = False
    # Number of expected exposures
    outModel.meta.visit.total_exposures = obs_params['nexposures']

    # Instrument info
    outModel.meta.telescope           = 'JWST'
    outModel.meta.instrument.name     = 'NIRCAM'
    outModel.meta.instrument.module   = obs_params['module']
    outModel.meta.instrument.channel  = obs_params['channel']
    outModel.meta.instrument.detector = obs_params['detector']
    outModel.meta.exposure.sca_num    = obs_params['det_obj'].scaid
    
    # Filter and pupil info
    filt = obs_params['filter']
    pupil = obs_params['pupil']
    if 'GRISM0' in pupil:
        pupil = 'GRISMR'
    elif 'GRISM90' in pupil:
        pupil = 'GRISMC'
    elif 'CIRC' in pupil:
        pupil = 'MASKRND'
    elif 'WEDGE' in pupil:
        pupil = 'MASKBAR'
    filt  = 'CLEAR' if filt  is None else filt
    pupil = 'CLEAR' if pupil is None else pupil

    # Update filter and pupil settings for narrowband filters in pupil wheel
    if filt in ['F162M', 'F164N']:
        pupil = filt
        filt = 'F150W2'
    elif filt=='F323N':
        pupil = filt
        filt = 'F322W2'
    elif filt in ['F405N', 'F466N','F470N']:
        pupil = filt
        filt = 'F444W'

    # Set fiter and pupil in datamodel
    outModel.meta.instrument.filter = filt
    outModel.meta.instrument.pupil  = pupil

    # Coronagraphic mask for non-TA observations
    apername = obs_params['siaf_ap'].AperName
    coronmsk = obs_params['coron_mask'].upper()
    if ('MASK' in coronmsk) and ('TA' not in apername):
        # Need to add module A or B after MASK string
        coronmsk = coronmsk[:4] + obs_params['module'] + coronmsk[4:]
        outModel.meta.instrument.coronagraph = coronmsk

    # Detector information 
    outModel.meta.subarray.name     = obs_params['subarray_name']
    # subarray_bounds indexed to zero, but values in header should be indexed to 1.
    outModel.meta.subarray.xstart   = obs_params['xstart']
    outModel.meta.subarray.ystart   = obs_params['ystart']
    outModel.meta.subarray.xsize    = obs_params['xsize']
    outModel.meta.subarray.ysize    = obs_params['ysize']
    outModel.meta.subarray.fastaxis = obs_params['fastaxis']
    outModel.meta.subarray.slowaxis = obs_params['slowaxis']
    
    # MULTIACCUM Settings
    outModel.meta.exposure.readpatt              = obs_params['readpatt']
    outModel.meta.exposure.noutputs              = obs_params['noutputs']
    outModel.meta.exposure.nframes               = obs_params['nframes']
    outModel.meta.exposure.ngroups               = obs_params['ngroups']
    outModel.meta.exposure.frame_divisor         = obs_params['nframes']
    outModel.meta.exposure.nints                 = obs_params['nints']
    outModel.meta.exposure.integration_start     = obs_params['integration_start']
    outModel.meta.exposure.integration_end       = obs_params['integration_end']
    outModel.meta.exposure.nresets_at_start      = obs_params['nresets1']
    outModel.meta.exposure.nresets_between_ints  = obs_params['nresets2']
        
    outModel.meta.exposure.sample_time           = obs_params['sample_time']
    outModel.meta.exposure.frame_time            = obs_params['frame_time']
    outModel.meta.exposure.group_time            = obs_params['group_time']
    outModel.meta.exposure.groupgap              = obs_params['groupgap']
    outModel.meta.exposure.drop_frames1          = 0  # Always 0 for NIRCam
    outModel.meta.exposure.drop_frames3          = 0  # Always 0 for NIRCam
    outModel.meta.exposure.nsamples              = 1  # Always 1 for NIRCam
    outModel.meta.exposure.integration_time      = obs_params['integration_time']
    outModel.meta.exposure.exposure_time         = obs_params['exposure_time']
    # INT_TIMES table to be saved in INT_TIMES extension
    outModel.int_times = obs_params['int_times']
    
    # Start date and time as specified in obs params
    # This is absolute start time of entire set of visits / exposures
    start_time_string = obs_params['date-obs'] + 'T' + obs_params['time-obs']
    # Get visit absolute start time
    visit_time = Time(start_time_string) + obs_params.get('visit_start_relative',0)*u.second
    visit_time_str = visit_time.utc.value
    visit_time_str = ' '.join(visit_time_str.split('T'))

    # Add offset time to get beginning of observation and update start_time_string
    texp_start_relative = obs_params.get('texp_start_relative', 0)
    start_time = Time(start_time_string) + texp_start_relative * u.second
    start_time_string = start_time.utc.value
    # Total time to complete an integration and exposures (including reset fraems)
    tint_tot = obs_params['tint_plus_overhead']
    texp_tot = obs_params['texp_plus_overhead']
    # Get observation end time
    end_time = start_time + texp_tot*u.second
    end_time_string = end_time.utc.value

    # Date and time of this observation
    date_obs, time_obs = start_time_string.split('T')
    outModel.meta.observation.date = date_obs
    outModel.meta.observation.time = time_obs
    outModel.meta.time_sys = 'UTC'
    outModel.meta.time_unit = 's'
    
    # UTC ISO string values
    # tnow = start_time.now()
    # outModel.meta.date = 'T'.join(tnow.iso.split(' '))
    outModel.meta.visit.start_time     = visit_time_str
    outModel.meta.observation.date_beg = start_time_string
    outModel.meta.observation.date_end = end_time_string

    # set the exposure start time
    outModel.meta.exposure.start_time = start_time.mjd 
    outModel.meta.exposure.end_time   = start_time.mjd + texp_tot / (24*3600.)
    outModel.meta.exposure.mid_time   = start_time.mjd + texp_tot / (24*3600.) / 2.
    outModel.meta.exposure.duration   = texp_tot

    # populate the GROUP extension table
    # n_int, n_group, n_y, n_x = outModel.data.shape
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     outModel.group = populate_group_table(start_time, obs_params['group_time'], obs_params['integration_time'],
    #                                           n_int, n_group, n_y, n_x)
    outModel.group = obs_params['group_times']

    # Observation/program ID information
    outModel.meta.observation.observation_label = obs_params.get('obs_label')
    try:
        obs_id_dict = obs_params['obs_id_info']
    except KeyError:
        _log.warning("'obs_id_info' is not a valid key in obs_params")
    else:
        outModel.meta.observation.program_number     = obs_id_dict['program_number']
        outModel.meta.observation.observation_number = obs_id_dict['observation_number']
        outModel.meta.observation.visit_number       = obs_id_dict['visit_number']
        outModel.meta.observation.visit_group        = obs_id_dict['visit_group']
        outModel.meta.observation.sequence_id        = obs_id_dict['sequence_id']
        outModel.meta.observation.activity_id        = obs_id_dict['activity_id']
        exp_num_str = str(int(obs_id_dict['exposure_number'])) # Remove leading 0s
        outModel.meta.observation.exposure_number    = exp_num_str
        outModel.meta.observation.visit_id           = obs_id_dict['visit_id']
        outModel.meta.observation.obs_id             = obs_id_dict['obs_id']

    # Telescope pointing
    siaf_ap = obs_params['siaf_ap']
    pa_v3   = obs_params['pa_v3']
    # ra_v1, dec_v1, and pa_v3 are not used by the level 2 pipelines
    # compute pointing of V1 axis
    ra_obs, dec_obs = (obs_params['ra_obs'], obs_params['dec_obs'])
    attitude_matrix = rotations.attitude(siaf_ap.V2Ref, siaf_ap.V3Ref, ra_obs, dec_obs, pa_v3)
    pointing_ra_v1, pointing_dec_v1 = rotations.pointing(attitude_matrix, 0., 0.)
    outModel.meta.pointing.ra_v1  = pointing_ra_v1
    outModel.meta.pointing.dec_v1 = pointing_dec_v1
    outModel.meta.pointing.pa_v3  = pa_v3
    

    
    # Dither information
    # Certain dither info should be None if string is 'NONE'
    pri_dithers = obs_params.get('pridith_points_packing')
    if (pri_dithers is not None) and (pri_dithers.upper()=='NONE'):
        pri_dithers = None
    sub_dith_type = obs_params.get('subpixel_pattern')
    if (sub_dith_type is not None) and (sub_dith_type.upper()=='NONE'):
        sub_dith_type = None
    sgd_type = obs_params.get('sgd_pattern')
    if (sgd_type is not None) and (sgd_type.upper()=='NONE'):
        sgd_type = None

    # If SGD, then PATT_NUM holds the current position number
    position_number = obs_params.get('position_number') if sgd_type is None else obs_params.get('subpixel_number')

    outModel.meta.dither.primary_type          = obs_params.get('pridith_pattern_type')
    outModel.meta.dither.primary_dither_type   = pri_dithers
    outModel.meta.dither.primary_points        = obs_params.get('pridith_npoints')
    outModel.meta.dither.position_number       = position_number
    # outModel.meta.dither.pattern_start         = obs_params.get('pattern_start')
    outModel.meta.dither.total_points          = obs_params.get('total_points')
    # outModel.meta.dither.dither_points         = obs_params.get('dither_points')
    outModel.meta.dither.pattern_size          = obs_params.get('pattern_size')
    outModel.meta.dither.small_grid_pattern    = sgd_type
    outModel.meta.dither.subpixel_pattern      = sub_dith_type
    outModel.meta.dither.subpixel_total_points = obs_params.get('subpixel_total_points')
    outModel.meta.dither.subpixel_number       = obs_params.get('subpixel_number')  # not valid
    outModel.meta.dither.x_offset = obs_params.get('x_offset')
    outModel.meta.dither.y_offset = obs_params.get('y_offset')

    ### Header keywords and associated attributes
    # PATTTYPE = primary_type         / Primary dither pattern type                    
    # PRIDTYPE = primary_dither_type  / Primary dither points and packing              
    # PRIDTPTS = primary_points       / Number of points in primary dither pattern     
    # PATT_NUM = position_number      / Position number in primary pattern             
    # PATTSTRT = pattern_start        / Starting point in pattern                      
    # NUMDTHPT = total_points         / Total number of points in pattern              
    # NRIMDTPT = dither_points        / Number of points in image dither pattern       
    # PATTSIZE = pattern_size         / Primary dither pattern size
    # SMGRDPAT = small_grid_pattern   / Name of small grid dither pattern              
    # SUBPXPTS = subpixel_total_points/ Number of points in subpixel dither pattern  
    # SUBPXPAT = subpixel_pattern     / Subpixel dither pattern type

    ### Examples of types
    # outModel.meta.dither.primary_type = 'NONE'
    # outModel.meta.dither.primary_dither_type = '1'
    # outModel.meta.dither.primary_points = 1
    # outModel.meta.dither.position_number = 1
    # outModel.meta.dither.pattern_start = 1
    # outModel.meta.dither.total_points = 5
    # outModel.meta.dither.dither_points = 10
    # outModel.meta.dither.pattern_size = 'DEFAULT'
    # outModel.meta.dither.small_grid_pattern = '5-POINT-DIAMOND'
    # outModel.meta.dither.subpixel_total_points = 5
    # outModel.meta.dither.subpixel_pattern = 'SMALL-GRID-DITHER'



    # WCS Info
    outModel.meta.aperture.name = siaf_ap.AperName
    outModel.meta.wcsinfo.wcsaxes = 2
    outModel.meta.wcsinfo.crval1 = ra_obs
    outModel.meta.wcsinfo.crval2 = dec_obs
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

    # Grism TSO data have the XREF_SCI and YREF_SCI keywords populated.
    # These are used to describe the location of the source on the detector.
    try:
        outModel.meta.wcsinfo.siaf_xref_sci = obs_params['XREF_SCI']
        outModel.meta.wcsinfo.siaf_yref_sci = obs_params['YREF_SCI']
    except KeyError:
        outModel.meta.wcsinfo.siaf_xref_sci = siaf_ap.XSciRef
        outModel.meta.wcsinfo.siaf_yref_sci = siaf_ap.YSciRef

    # V3 roll angle at the ref point
    roll_ref = compute_local_roll(pa_v3, ra_obs, dec_obs, siaf_ap.V2Ref, siaf_ap.V3Ref)
    outModel.meta.wcsinfo.roll_ref = roll_ref
    
    outModel.meta.filename = obs_params['filename']
    
    return outModel

def save_level1b_fits(outModel, obs_params, save_dir=None, **kwargs):
    """Save Level1bModel to FITS and update headers"""

    # Check if save directory specified in obs_params
    if save_dir is None:
        save_dir = obs_params.get('save_dir')

    file_path = outModel.meta.filename
    if save_dir is not None:
        # Create directory and intermediates if they don't exist
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_path)
        
    # Save model to DMS FITS file 
    print(f'  Saving: {file_path}')
    outModel.save(file_path)

    # Update header information
    update_dms_headers(file_path, obs_params)
    update_headers_pynrc_info(file_path, obs_params, **kwargs)


def create_DMS_HDUList(sci_data, zero_data, obs_params, save_dir=None, **kwargs):
    """Save Level 1b to FITS file"""
    
    outModel = level1b_data_model(obs_params, sci_data=sci_data, zero_data=zero_data)
    save_level1b_fits(outModel, obs_params, save_dir=save_dir, **kwargs)


def update_dms_headers(filename, obs_params):
    """
    Given the name of a valid partially populated level 1b JWST file,
    add a couple simple WCS parameters from the SIAF keywords, which
    contain information about the telescope pointing. 

    It presumes all the accessed keywords are present from the JWST
    pipeline data model.

    Parameters
    ----------
    filename : str
        file name
    obs_params : dict
        Dictionary of parameters to populate DMS header. 
        See `create_DMS_HDUList` in dms.py.
    """
    hdulist = fits.open(filename, mode='update')
    pheader = hdulist[0].header
    fheader = hdulist[1].header

    try:
        v2ref = float(pheader['V2_REF'])
        v3ref = float(pheader['V3_REF'])
        v3idlyang = float(pheader['V3I_YANG'])
        vparity   = int(pheader['VPARITY'])
        pa_v3 = float(pheader['PA_V3'])
    except:
        v2ref = float(fheader['V2_REF'])
        v3ref = float(fheader['V3_REF'])
        v3idlyang = float(fheader['V3I_YANG'])
        vparity   = int(fheader['VPARITY'])
        pa_v3 = float(fheader['PA_V3'])

    ra_ref = fheader['CRVAL1']
    dec_ref = fheader['CRVAL2']

    local_roll  = compute_local_roll(pa_v3, ra_ref, dec_ref, v2ref, v3ref)
    pa_aper_deg = local_roll - vparity * v3idlyang

    D2R = np.pi / 180.0
    fheader['PC1_1']   = -np.cos(pa_aper_deg * D2R)
    fheader['PC1_2']   = np.sin(pa_aper_deg * D2R)
    fheader['PC2_1']   = np.sin(pa_aper_deg * D2R)
    fheader['PC2_2']   = np.cos(pa_aper_deg * D2R)
    fheader['RA_REF']  = ra_ref
    fheader['DEC_REF'] = dec_ref
    fheader['ROLL_REF'] = local_roll
    fheader['WCSAXES'] = len(fheader['CTYPE*'])

    fheader['SIAF'] = (pysiaf.JWST_PRD_VERSION, "SIAF PRD version")

    # Segment exposure information
    if obs_params['EXSEGNUM'] is not None:
        pheader['EXSEGNUM'] = obs_params['EXSEGNUM']
    if obs_params['EXSEGTOT'] is not None:
        pheader['EXSEGTOT'] = obs_params['EXSEGTOT']
    
    # Now we need to adjust the data model header keyword
    # If we leave it as Level1bModel, the pipeline doesn't
    # work properly
    # TODO: Check if this is actually the case
    # if '1b' in pheader['DATAMODL']:
    #     pheader['DATAMODL'] = 'RampModel'

    hdulist.flush()
    hdulist.close()

def update_headers_pynrc_info(filename, obs_params, **kwargs):
    """Add pynrc info to headers"""

    hdulist = fits.open(filename, mode='update')
    pheader = hdulist[0].header
    fheader = hdulist[1].header

    # Add file info from kwargs
    kw_to_hkey = {
        # Keyword Arg   : (Header key, Header comment)
        'json_file'     : ('APTJSON',  'APT JSON file input.'),
        'sm_acct_file'  : ('APTSMRT',  'APT smart accounting file input.'),
        'pointing_file' : ('APTPOINT', 'APT pointing file input.'),
        'xml_file'      : ('APTXML',   'APT XML file input.'),
    }
    for kw in kw_to_hkey.keys():
        hkey, comment = kw_to_hkey[kw]
        pheader[hkey] = (kwargs.get(kw, NONE_STR), comment)

    # Insert pynrc header comment
    hkey_first = 'APTJSON'
    pheader.insert(hkey_first, '', after=False)
    pheader.insert(hkey_first, ('', 'pyNRC information'), after=False)
    pheader.insert(hkey_first, '', after=False)

    # Input telescope pointing info
    kw_to_hkey = {
        # Keyword Arg : (Header key, Header comment)
        'ra_obs'      : ('RA_IN',    'RA of observered SIAF aperture.'),
        'dec_obs'     : ('DEC_IN',   'DEC of observered SIAF aperture.'),
        'pa_v3'       : ('PAV3_IN',  'Telescope position angle relative to V3.'),
        'roll_offset' : ('ROLL_IN',  'Roll angle relative to nominal V3 PA.'),
        'solar_elong' : ('ELONG_IN', 'Solar elongation (deg).'),
        'pitch_ang'   : ('PITCH_IN', 'Telescope pitch angle relative to sun (deg).'),
        'wfe_drift'   : ('WFEDRIFT', 'Delta WFE drift from baseline OPD (nm RMS)'),
    }
    for kw in kw_to_hkey.keys():
        hkey, comment = kw_to_hkey[kw]
        pheader[hkey] = (obs_params.get(kw, NONE_STR), comment)

    # Add actual dither offset information
    kw_to_hkey = {
        # Keyword Arg        : (Header key, Header comment)
        'xoffset_act'   : ('XOFFACT', 'x dither offset in arcsec (idl coords)'),
        'yoffset_act'   : ('YOFFACT', 'y dither offset in arcsec (idl coords)'),
    }
    for kw in kw_to_hkey.keys():
        hkey, comment = kw_to_hkey[kw]
        pheader[hkey] = (kwargs.get(kw, NONE_STR), comment)

    # Add random seed info
    kw_to_hkey = {
        # Keyword Arg     : (Header key, Header comment)
        'rand_seed_init'  : ('RANDINIT', 'Random seed to init program sim'),
        'rand_seed_dith'  : ('RANDDITH', 'Random seed for dither errors'),
        'rand_seed_noise' : ('RANDNOIS', 'Random seed for ramp noise init'),
        'rand_seed_dwfe'  : ('RANDDWFE', 'Random seed for delta WFE IEC component'),
    }
    for kw in kw_to_hkey.keys():
        hkey, comment = kw_to_hkey[kw]
        pheader[hkey] = (obs_params.get(kw, NONE_STR), comment)

    # Add noise parameter settings
    kw_to_hkey = {
        # Keyword Arg        : (Header key, Header comment)
        'include_poisson'    : ('POISSON', 'Include Poisson noise?'),
        'include_dark'       : ('ADDDARK', 'Include dark current?'), 
        'include_bias'       : ('ADDBIAS', 'Include bias image?'),
        'include_ktc'        : ('ADDKTC',  'Include kTC noise?'),
        'include_rn'         : ('ADDRN',   'Include read noise?'),
        'include_cpink'      : ('CPINK',   'Include correlated amplifer 1/f noise?'),
        'include_upink'      : ('UPINK',   'Include uncorrelated amplifer 1/f noise?'),
        'include_acn'        : ('ACN',     'Include alternating column noise?'),
        'apply_ipc'          : ('ADDIPC',  'Add interpixel capacitance?'),
        'apply_ppc'          : ('ADDPPC',  'Add post-pixel coupling signal bleed?'),
        'amp_crosstalk'      : ('XTALK',   'Add amplifier crosstalk?'),
        'include_refoffsets' : ('REFOFFS', 'Include reference offsets?'),
        'include_refinst'    : ('REFINST', 'Include reference pixel instability?'),
        'include_colnoise'   : ('COLNOISE','Include sporatic column noise?'),
        'add_crs'            : ('ADDCR',   'Add cosmic rays events?'),
        'cr_model'           : ('CRMODEL', 'Cosmic ray model SUNMAX, SUNMIN, or FLARES.'),
        'cr_scale'           : ('CRSCALE', 'Cosmic ray model scaling.'),
        'apply_nonlinearity' : ('ADDLIN',  'Include pixel nonlinearity?'),
        'random_nonlin'      : ('RANDLIN', 'Add noise to nonlinearity function?'),
        'apply_flats'        : ('ADDFLATS','Include pflat and lflat field variations?'),
    }
    for kw in kw_to_hkey.keys():
        hkey, comment = kw_to_hkey[kw]
        pheader[hkey] = (kwargs.get(kw, NONE_STR), comment)

    # Add ice and nvr info
    kw_to_hkey = {
        # Keyword Arg     : (Header key, Header comment)
        'ice_scale' : ('ICESCALE', 'Scale relative to 0.0131 um thickness'),
        'nvr_scale' : ('NVRSCALE', 'Scale relative to 0.280 um thickness'),
        'nc_scale'  : ('NCSCALE',  'Scale of NVR=0.189 um and H2O=0.050 um'),
        'ote_scale' : ('OTESCALE', 'Scale relative to 0.0131 um thickness'),
    }
    for kw in kw_to_hkey.keys():
        hkey, comment = kw_to_hkey[kw]
        pheader[hkey] = (kwargs.get(kw, NONE_STR), comment)

    hdulist.flush()
    hdulist.close()

def compute_local_roll(pa_v3, ra_ref, dec_ref, v2_ref, v3_ref):
    """
    Computes the position angle of V3 (measured N to E) at the reference point of an aperture.

    Parameters
    ----------
    pa_v3 : float
        Position angle of V3 at (V2, V3) = (0, 0) [in deg]
    v2_ref, v3_ref : float
        Reference point in the V2, V3 frame [in arcsec]
    ra_ref, dec_ref : float
        RA and DEC corresponding to V2_REF and V3_REF, [in deg]

    Returns
    -------
    new_roll : float
        The value of ROLL_REF (in deg)

    """
    v2 = np.deg2rad(v2_ref / 3600)
    v3 = np.deg2rad(v3_ref / 3600)
    ra_ref = np.deg2rad(ra_ref)
    dec_ref = np.deg2rad(dec_ref)
    pa_v3 = np.deg2rad(pa_v3)

    M = np.array([[np.cos(ra_ref) * np.cos(dec_ref),
                   -np.sin(ra_ref) * np.cos(pa_v3) + np.cos(ra_ref) * np.sin(dec_ref) * np.sin(pa_v3),
                   -np.sin(ra_ref) * np.sin(pa_v3) - np.cos(ra_ref) * np.sin(dec_ref) * np.cos(pa_v3)],
                  [np.sin(ra_ref) * np.cos(dec_ref),
                   np.cos(ra_ref) * np.cos(pa_v3) + np.sin(ra_ref) * np.sin(dec_ref) * np.sin(pa_v3),
                   np.cos(ra_ref) * np.sin(pa_v3) - np.sin(ra_ref) * np.sin(dec_ref) * np.cos(pa_v3)],
                   [np.sin(dec_ref),
                    -np.cos(dec_ref) * np.sin(pa_v3),
                    np.cos(dec_ref) * np.cos(pa_v3)]
                  ])

    return _roll_angle_from_matrix(M, v2, v3)


def _roll_angle_from_matrix(matrix, v2, v3):
    X = -(matrix[2, 0] * np.cos(v2) + matrix[2, 1] * np.sin(v2)) * np.sin(v3) + matrix[2, 2] * np.cos(v3)
    Y = (matrix[0, 0] *  matrix[1, 2] - matrix[1, 0] * matrix[0, 2]) * np.cos(v2) + \
      (matrix[0, 1] * matrix[1, 2] - matrix[1, 1] * matrix[0, 2]) * np.sin(v2)
    new_roll = np.rad2deg(np.arctan2(Y, X))
    if new_roll < 0:
        new_roll += 360
    return new_roll


