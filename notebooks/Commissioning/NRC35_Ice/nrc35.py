# Import the usual libraries
from matplotlib.pyplot import streamplot
import numpy as np
import os

# Program bar
from tqdm.auto import trange, tqdm

import pynrc
from pynrc import nrc_utils
from pynrc.nrc_utils import webbpsf, poppy, S
from pynrc.nrc_utils import segment_pupil_opd, make_grism_slope
from pynrc.nrc_utils import bias_dark_high_temp as bias_dark
from pynrc.simul import apt, ngNRC

pynrc.setup_logging('WARN', verbose=True)

from astropy import units as u
from astropy.io import fits, ascii
from astropy.time import Time
from astropy.coordinates import SkyCoord

#########################################
# Define directory paths
#########################################
apt_dir       = '../../APT_output/'
# opd_dir       = '/Users/jarron/NIRCam/Data/OTE_OPDs/'
# darks_80K_dir = '/Users/jarron/NIRCam/Data/NRC_80K/'
# save_dir      = '/Users/jarron/NIRCam/Data/NRC_Sims/Sim_NRC35/'

opd_dir       = '/home/jarronl/data/NIRData/NIRCam/OTE_OPDs/'
darks_80K_dir = '/home/jarronl/data/NIRData/NIRCam/NRC_80K/'
save_dir      = '/home/jarronl/data/NIRData/NRC_Sims/Sim_NRC35/'

webbpsf_data_path = webbpsf.utils.get_webbpsf_data_path()

# Check that directories exist
for d in [apt_dir, opd_dir, darks_80K_dir, save_dir, webbpsf_data_path]:
    if not os.path.isdir(d):
        print(f'{d} does not exist!')

# Pupil and segment information
pupil_file = os.path.join(webbpsf_data_path, "jwst_pupil_RevW_npix1024.fits.gz")
pupil_hdul = fits.open(pupil_file)

segmap_file = os.path.join(webbpsf_data_path, "JWpupil_segments_RevW_npix1024.fits.gz")
segmap_hdul = fits.open(segmap_file)


def init_obs_dict(opd_dir, seg_name='A3'):
    """Initialize observation dictionary"""

    # 10 total observations, split into 5 pairs
    obs_dict = {}

    # L+40 - Use a segment from Deployment_OPDs[7]
    hdul = fits.open(opd_dir + 'Deployment_OPDs.fits')
    # hdul = fits.open(opd_dir + 'MM_WAS-GLOBAL_ALIGNMENT_WO_TT_R2017112104.fits')
    opd_single = segment_pupil_opd(hdul[7], seg_name)
    # Post LOS-02 pointing and background check
    obs_dict['031:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':75, 'day':'L+40.0'}
    obs_dict['032:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':75, 'day':'L+40.1'}
    obs_dict['033:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':75, 'day':'L+40.2', 'offset':(5,8)}
    obs_dict['034:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':75, 'day':'L+40.3'}
    #obs_dict['035:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':86, 'day':'L+40', 'time':'16:00:00'}
    # Pre FSM heater off NIRCam with Beam Probing
    obs_dict['001:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':75, 'day':'L+40.5', 'offset':(40,20)}
    obs_dict['022:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':75, 'day':'L+40.6', 'offset':(5,10.5)}
    obs_dict['023:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':75, 'day':'L+40.7', 'offset':(5,10.5)}
    obs_dict['024:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':75, 'day':'L+40.8', 'offset':(5,10.5)}
    obs_dict['025:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':75, 'day':'L+40.9', 'offset':(5,10.5)}
    obs_dict['026:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':75, 'day':'L+41.0', 'offset':(5,10.5)}
    # L+43 - Initial Deployment
    obs_dict['002:001']  = {'opd': opd_single, 'jsig': 0.1, 'temp':70, 'day':'L+43'}

    # L+47 - Segment Image Array 1
    hdul = fits.open(opd_dir + 'MM_WAS-GLOBAL_ALIGNMENT_small_WITH_TT_R2017102404.fits')
    obs_dict['003:001']  = {'opd': segment_pupil_opd(hdul[0], 'ALL'), 'jsig': 0.1, 'temp':60, 'day':'L+47.4'}

    # L+50 - Global Alignment 1
    hdul = fits.open(opd_dir + 'MM_WAS-GLOBAL_ALIGNMENT_small_WITH_TT_N2017112103.fits')
    obs_dict['004:001']  = {'opd': segment_pupil_opd(hdul[0], 'ALL'), 'jsig': 0.01, 'temp':50, 'day':'L+50.4'}

    # L+54 - Image Stacking 1
    hdul = fits.open(opd_dir + 'MM_COARSE_MIMF_N2017102403.fits')
    obs_dict['005:001']  = {'opd': segment_pupil_opd(hdul[0], 'ALL'), 'jsig': 0.01, 'temp':45, 'day':'L+53.6'}

    # L+58 - Coarse Phasing 1 (stacked)
    hdul = fits.open(opd_dir + 'MM_COARSE_MIMF_R201710240A.fits')
    obs_dict['006:001']  = {'opd': segment_pupil_opd(hdul[0], 'ALL'), 'jsig': 0.01, 'temp':42, 'day':'L+58.2'}

    # L+67 - Fine Phasing 1 (stacked)
    hdul = fits.open(opd_dir + 'MM_WAS-GLOBAL_ALIGNMENT_WO_TT_R2017112104.fits')
    obs_dict['007:001']  = {'opd': segment_pupil_opd(hdul[0], 'ALL'), 'jsig': 0.01, 'temp':40, 'day':'L+66.8'}

    # L+77/96 - MIMF
    hdul = fits.open(opd_dir + 'MM_FINE_PHASING_R2017120405.fits')
    obs_dict['008:001']  = {'opd': segment_pupil_opd(hdul[0], 'ALL'), 'jsig': 0.01, 'temp':40, 'day':'L+77.0'}
    obs_dict['009:001']  = {'opd': segment_pupil_opd(hdul[0], 'ALL'), 'jsig': 0.01, 'temp':39, 'day':'L+96.4'}

    return obs_dict

# Define APT files
def init_visits(obs_dict, rand_seed_init=None):
    """Initialize visit and initial obs_params config"""
    
    keys = obs_dict.keys()

    # APT output files
    xml_file      = os.path.join(apt_dir, 'pid1409.xml')
    pointing_file = os.path.join(apt_dir, 'pid1409.pointing')
    json_file     = os.path.join(apt_dir, 'pid1409.timing.json')
    sm_acct_file  = os.path.join(apt_dir, 'pid1409.smart_accounting')
    
    # Define Launch date and time
    tlaunch = Time('2021-12-25 12:20:00')  
    # Populate dates and times for all observations
    for vkey in keys:
        d = obs_dict[vkey]
        lplus_days = float(d['day'][2:])
        tobs = tlaunch + lplus_days
        date_obs, time_obs = tobs.to_value('fits').split('T')
        d['date'] = date_obs
        d['time'] = time_obs
        
    # Read and interpret APT files
    # Observation of first date, not really necessary
    obs_date = obs_dict['031:001']['date']  
    apt_obj = apt.DMS_input(xml_file, pointing_file, json_file, sm_acct_file, 
                            obs_date=obs_date, rand_seed_init=rand_seed_init)
    visits = apt_obj.program_info
    
    # Create obs params for all LW observations only
    print('  Creating Observation Parameters...')
    for vkey in tqdm(keys, leave=False, desc='Visits'):
        visit = visits[vkey]
        
        # Update date and time before generating specific obs_params
        apt_obj.obs_date = obs_dict[vkey]['date']
        apt_obj.obs_time = obs_dict[vkey]['time']

        visit['exp_start_times'] = visit['exp_start_times'] - visit['exp_start_times'].min() + 1800

        # Select only a single exposure within visit
        index = 0 #if 'SUBGRISM256' in visit['subarray_name'] else 0
        oid = visit['obs_id_info'][index]
        mod = visit['ModuleAPT'][index]

        # Select only LW detector
        detid = 485 if mod=='A' else 490
        exp_id = oid['exposure_number']
        grp_id = oid['visit_group']
        seq_id = oid['sequence_id']
        act_id = oid['activity_id']

        obs_params = apt_obj.gen_obs_params(vkey, exp_id, detid, grp_id=grp_id, 
                                            seq_id=seq_id, act_id=act_id)
        
        # Store in obs_dict
        obs_dict[vkey]['obs_params'] = obs_params
        obs_dict[vkey]['visit'] = visit
        

def add_inst(obs_dict, key=None, ice_scale=None, nvr_scale=None, ote_scale=0, nc_scale=0, spec_ang=0):
    """Add NIRCam instrument class
    
    ice_scale : float
        Add in additional OTE H2O absorption. This is a scale factor
        relative to 0.0131 um thickness. Also includes about 0.0150 um of
        photolyzed Carbon.
    nvr_scale : float
        Modify NIRCam non-volatile residue. This is a scale factor relative 
        to 0.280 um thickness already built into filter throughput curves. 
        If set to None, then assumes a scale factor of 1.0. 
        Setting ``nvr_scale=0`` will remove these contributions.
    ote_scale : float
        Scale factor of OTE contaminants relative to End of Life model. 
        This is the same as setting ``ice_scale``. 
        Will override ``ice_scale`` value.
    nc_scale : float
        Scale factor for NIRCam contaminants relative to End of Life model.
        This model assumes 0.189 um of NVR and 0.050 um of water ice on
        the NIRCam optical elements. Setting this keyword will remove all
        NVR contributions built into the NIRCam filter curves.
        Overrides ``nvr_scale`` value.
    
    """
    
    
    print('  Generating Instrument Classes...')
    keys = obs_dict.keys() if key is None else np.asarray([key]).flatten()
    iter_keys = tqdm(keys, leave=False, desc='Visits') if len(keys)>1 else keys
    for vkey in iter_keys:
        d = obs_dict[vkey]
        
        # Visit info and obs
        visit = d['visit']
        obs_params = d['obs_params']
        
        filt       = obs_params['filter']
        pupil_mask = obs_params['pupil']
        image_mask = None if obs_params['coron_mask']=='NONE' else obs_params['coron_mask']
        apname     = obs_params['siaf_ap'].AperName

        input_pupil_hdul, input_opd_hdul = d['opd']
        jsig = d['jsig']
        d['ice_scale'] = ice_scale
        d['nvr_scale'] = nvr_scale
        d['ote_scale'] = ote_scale
        d['nc_scale'] = nc_scale
        nrc = pynrc.NIRCam(filter=filt, pupil_mask=pupil_mask, image_mask=image_mask,
                           fov_pix=257, oversample=2, autogen_coeffs=True,
                           ice_scale=d['ice_scale'], nvr_scale=d['nvr_scale'],
                           ote_scale=d['ote_scale'], nc_scale=d['nc_scale'],
                           pupil=input_pupil_hdul, pupilopd=input_opd_hdul,
                           save=False, force=True, quick=True)

        # Update detector information
        det = obs_params['det_obj']
        kw = {**det.to_dict(), **det.multiaccum.to_dict()}
        nrc.update_detectors(**kw)
        d['nrc'] = nrc


def add_slope_images(obs_dict, key=None, spec_ang=0, add_offset=None, **kwargs):
    """Generate slope image(s)"""
        
    print('  Creating ideal slopes, FSM background, SCA darks, biases...')
    keys = obs_dict.keys() if key is None else np.asarray([key]).flatten()
    iter_keys = tqdm(keys, leave=False, desc='Visits') if len(keys)>1 else keys
    for vkey in iter_keys:
        d = obs_dict[vkey]

        nrc = d['nrc']
        # Visit info and obs
        visit = d['visit']
        obs_params = d['obs_params']

        T = d['temp']
        
        # Change collecting area to modify flux 
        input_pupil_hdul, input_opd_hdul = d['opd']
        coll_area_seg = 25.78e4 * input_pupil_hdul[0].data.sum() / pupil_hdul[0].data.sum()
        S.refs.setref(area=coll_area_seg)

        # Create source table
        ra  = obs_params['ra']
        dec = obs_params['dec']
        coords = SkyCoord(ra, dec, unit=(u.deg, u.deg), 
                          frame='icrs', equinox='J2000', obstime='J2000')
        ra, dec = (coords.ra.deg, coords.dec.deg)
        # src_tbl = ngNRC.make_gaia_source_table(coords, remove_cen_star=False)
        src_tbl = ngNRC.make_simbad_source_table(coords, remove_cen_star=False)

        # Central star is actually a binary with primary as a sub-giant or giant type
        src_tbl[0]['SpType'] = 'K3III'
        d['src_tbl'] = src_tbl
        
        # Create pointing information
        tel_point = apt.gen_jwst_pointing(visit, obs_params)
        expnum  = int(obs_params['obs_id_info']['exposure_number'])
        print(f"  Exposure: {expnum}")

        # Get offset
        if add_offset is None:
            add_offset = d.get('offset')
        if add_offset is None:
            add_offset = np.array([0,0])
            
        ind = np.where(tel_point.exp_nums == expnum)[0][0]
        idl_off = tel_point.position_offsets_act[ind]
        obs_params['xoffset_act'] = idl_off[0] + add_offset[0]
        obs_params['yoffset_act'] = idl_off[1] + add_offset[1]

        # Get ideal slope
        if nrc.is_grism:
            res = make_grism_slope(nrc, src_tbl, tel_point, expnum,
                                   add_offset=add_offset, spec_ang=spec_ang, 
                                   **kwargs)

            wspec_all, im_slope_ideal = res
            # Add to dict
            d['wave'] = wspec_all
            d['im_slope_ideal'] = im_slope_ideal
        else:
            res = ngNRC.sources_to_slope(src_tbl, nrc, obs_params, tel_point, im_bg=0, 
                                         npsf_per_full_fov=5, cframe_out='sci',
                                         add_offset=add_offset, **kwargs)
            # Add to dict
            d['im_slope_ideal'] = res
        
        # Create background slope image
        im_bg_zodi = nrc.bg_zodi_image()
        # Scale background counts by T**4
        bg_cnts = 100 # 2000 * (T**4-39.0**4)/(80.0**4-39.0**4) 
        im_bg_pom = bg_cnts * im_bg_zodi / im_bg_zodi.max()
        d['bg_slope'] = im_bg_pom + im_bg_zodi
        
        # Add dark slope and bias images
        bias, dark = bias_dark(darks_80K_dir, T=T)
        
        # Grab subarray region of bias and dark frames
        x1 = nrc.det_info['x0']
        x2 = x1 + nrc.det_info['xpix']
        y1 = nrc.det_info['y0']
        y2 = y1 + nrc.det_info['ypix']

        # These are in 'det' coords, while slopes are in 'sci' coords
        d['im_bias'] = bias[y1:y2, x1:x2]
        d['im_dark'] = dark[y1:y2, x1:x2]        
        

def generate_level1b(obs_dict, key=None, save_dir=None, save_slope=True, **kwargs):

    from pynrc.simul.dms import level1b_data_model, save_level1b_fits
    from pynrc.simul.ngNRC import save_slope_image
    # from stdatamodels import fits_support
    
    kwargs['out_ADU'] = True
            
    keys = obs_dict.keys() if key is None else np.asarray([key]).flatten()
    iter_keys = tqdm(keys, leave=False, desc='Visits') if len(keys)>1 else keys
    for vkey in iter_keys:
        d = obs_dict[vkey]
        obs_params = d['obs_params']
        nrc        = d['nrc']
                
        # 'sci' coords
        im_slope = d['im_slope_ideal'] + d['bg_slope']
        
        if save_slope:
            kwargs2 = kwargs.copy()
            # Grab ice and nvr information for header info
            kwargs2['ice_scale'] = nrc._ice_scale
            kwargs2['nvr_scale'] = nrc._nvr_scale
            kwargs2['nc_scale']  = nrc._nc_scale
            kwargs2['ote_scale'] = nrc._ote_scale            
            save_slope_image(im_slope, obs_params, save_dir=save_dir, **kwargs2)
        
        # Get sci data ramps
        rand_seed_noise_j = obs_params['rand_seed_noise']
        res = nrc.simulate_ramps(im_slope=im_slope, rand_seed=rand_seed_noise_j,
                                 super_bias=d['im_bias'], super_dark=d['im_dark'], **kwargs)
        sci_data, zero_data = res
        
        # Generate Level-1b FITS
        outModel = level1b_data_model(obs_params, sci_data=sci_data, zero_data=zero_data)

        # Grab ice and nvr information for header info
        kwargs['ice_scale'] = nrc._ice_scale
        kwargs['nvr_scale'] = nrc._nvr_scale
        kwargs['nc_scale']  = nrc._nc_scale
        kwargs['ote_scale'] = nrc._ote_scale
        save_level1b_fits(outModel, obs_params, save_dir=save_dir, **kwargs)        


def run_all_exps(obs_dict, key=None, save_dir=None, rand_seed_init=None, save_slope=True, 
                 add_offset=None, **kwargs):
    """Run all exposures within visit
    
    Assumes visits and NIRCam objects have been initialized.
    """
    
    # APT output files
    xml_file      = os.path.join(apt_dir, 'pid1409.xml')
    pointing_file = os.path.join(apt_dir, 'pid1409.pointing')
    json_file     = os.path.join(apt_dir, 'pid1409.timing.json')
    sm_acct_file  = os.path.join(apt_dir, 'pid1409.smart_accounting')
    
    kwargs['json_file']     = os.path.basename(json_file)
    kwargs['sm_acct_file']  = os.path.basename(sm_acct_file)
    kwargs['pointing_file'] = os.path.basename(pointing_file)
    kwargs['xml_file']      = os.path.basename(xml_file)
    
    # Read and interpret APT files
    # Observation of first date, not really necessary
    apt_obj = apt.DMS_input(xml_file, pointing_file, json_file, sm_acct_file, 
                            rand_seed_init=rand_seed_init)
    
    keys = obs_dict.keys() if key is None else np.asarray([key]).flatten()
    iter_keys = tqdm(keys, leave=False, desc='Visits') if len(keys)>1 else keys
    for vkey in iter_keys:
        d = obs_dict[vkey]
        visit = d['visit']

        nexp = len(visit['obs_id_info'])
        # Select only a given exposure within visit
        for index in range(nexp):
            
            oid = visit['obs_id_info'][index]
            mod = visit['ModuleAPT'][index]

            # Select only LW detector
            detid = 485 if mod=='A' else 490
            exp_id = oid['exposure_number']
            grp_id = oid['visit_group']
            seq_id = oid['sequence_id']
            act_id = oid['activity_id']
            act_int = np.int(act_id, 36) # Convert base 36 to integer number

            # Generate new obs_params and store in obs_dict
            obs_params = apt_obj.gen_obs_params(vkey, exp_id, detid, grp_id=grp_id, 
                                                seq_id=seq_id, act_id=act_id)
            obs_dict[vkey]['obs_params'] = obs_params
        
            # Generate ideal slope image and wavelength solution
            #   obs_dict[vkey]['wave'] = wspec_all  # wavelengths for all
            #   obs_dict[vkey]['im_slope_ideal'] = im_slope_ideal
            add_slope_images(obs_dict, key=key, spec_ang=0, add_offset=add_offset)
            
            # Save Level1b FITS file
            generate_level1b(obs_dict, key=key, save_dir=save_dir, save_slope=save_slope, **kwargs)


