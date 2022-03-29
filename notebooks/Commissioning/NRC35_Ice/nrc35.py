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

from pynrc.reduce.calib import nircam_cal, apply_linearity, get_fits_data
from pynrc.maths.coords import det_to_sci, sci_to_det
from pynrc.detops import create_detops

pynrc.setup_logging('WARN', verbose=True)

from astropy import units as u
from astropy.io import fits, ascii
from astropy.time import Time
from astropy.coordinates import SkyCoord

from astropy.convolution import Gaussian1DKernel, convolve
from scipy.interpolate import interp1d

#########################################
# Define directory paths
#########################################
apt_dir       = '../../APT_output/'
opd_dir       = '/Users/jarron/NIRCam/Data/OTE_OPDs/'
darks_80K_dir = '/Users/jarron/NIRCam/Data/NRC_80K/'
save_dir      = '/Users/jarron/NIRCam/Data/NRC_Sims/Sim_NRC35/'

# opd_dir       = '/home/jarronl/data/NIRData/NIRCam/OTE_OPDs/'
# darks_80K_dir = '/home/jarronl/data/NIRData/NIRCam/NRC_80K/'
# save_dir      = '/home/jarronl/data/NIRData/NRC_Sims/Sim_NRC35/'

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
    obs_dict['031:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':75, 'day':'L+40.0', 'offset':(20,40)}
    obs_dict['032:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':75, 'day':'L+40.1'}
    obs_dict['033:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':75, 'day':'L+40.2', 'offset':(5,8)}
    obs_dict['034:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':75, 'day':'L+40.3'}
    #obs_dict['035:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':86, 'day':'L+40', 'time':'16:00:00'}
    # Pre FSM heater off NIRCam with Beam Probing
    obs_dict['001:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':75, 'day':'L+40.5', 'offset':(5,10.5)}
    obs_dict['022:001'] = {'opd': opd_single, 'jsig': 0.1, 'temp':75, 'day':'L+40.6', 'offset':(45,50)}
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


def add_slope_images(obs_dict, key=None, spec_ang=0, add_offset=None, 
    src_tbl=None, **kwargs):
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
        if src_tbl is None:
            ra  = obs_params['ra']
            dec = obs_params['dec']
            coords = SkyCoord(ra, dec, unit=(u.deg, u.deg), 
                            frame='icrs', equinox='J2000', obstime='J2000')
            ra, dec = (coords.ra.deg, coords.dec.deg)
            # src_tbl = ngNRC.make_gaia_source_table(coords, remove_cen_star=False)
            src_tbl = ngNRC.make_simbad_source_table(coords, remove_cen_star=False)

            # Central star is actually a binary with primary as a sub-giant or giant type
            src_tbl[0]['SpType'] = 'G8IV'

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
    obs_date = obs_dict['031:001']['date']
    apt_obj = apt.DMS_input(xml_file, pointing_file, json_file, sm_acct_file, 
                            obs_date=obs_date, rand_seed_init=rand_seed_init)
    
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
            src_tbl = obs_dict[vkey].get('src_tbl')
            add_slope_images(obs_dict, key=key, spec_ang=0, add_offset=add_offset, src_tbl=src_tbl)
            
            # Save Level1b FITS file
            generate_level1b(obs_dict, key=key, save_dir=save_dir, save_slope=save_slope, **kwargs)

def get_model_ice_nvr(file):
    """Get simulated Ice and NVR layers stored in FITS header
    
    Assumes a single layer.

    Returns (ice, nvr) layer thickness in units of microns.
    """

    hdul = fits.open(file)
    hdr = hdul[0].header
    hdul.close()

    ice_scale = hdr.get('ICESCALE')
    nvr_scale = hdr.get('NVRSCALE')
    nc_scale  = hdr.get('NCSCALE')
    ote_scale = hdr.get('OTESCALE')

    ice_scale = None if ice_scale == 'NONE' else ice_scale
    nvr_scale = None if nvr_scale == 'NONE' else nvr_scale
    nc_scale  = None if nc_scale  == 'NONE' else nc_scale
    ote_scale = None if ote_scale == 'NONE' else ote_scale

    # ote_scale takes priority over ice_scale
    if ote_scale is not None:
        ice_scale = ote_scale
    # nc_scale turns off nvr_scale
    if nc_scale is not None:
        nvr_scale = 0

    ice_layer = 0
    nvr_layer = 0

    if ice_scale is not None:
        ice_layer += ice_scale * 0.0131
    if nvr_scale is not None:
        nvr_layer += nvr_scale * 0.280
    if nc_scale is not None:
        ice_layer += nc_scale * 0.050
        nvr_layer += nc_scale * 0.189

    return (ice_layer, nvr_layer)

# Model fitting
from scipy import optimize
from webbpsf_ext.maths import jl_poly, jl_poly_fit
import webbpsf_ext

# Ice and NVR absorption coefficients
def abs_water(wave=None):
    path = webbpsf_ext.__path__[0]

    fname = os.path.join(path, 'throughputs/h2o_abs.txt')
    names = ['Wave', 'coeff'] # coeff is per um path length
    data_ice  = ascii.read(fname, names=names)
    
    w_ice = data_ice['Wave']
    a_ice = data_ice['coeff']
    
    # Estimates for w<2.5um
    w_ice = np.insert(w_ice, 0, [1.0]) 
    a_ice = np.insert(a_ice, 0, [0.0]) 
    # Estimates for w>5.0um
    w_ice = np.append(w_ice, [6.0])    
    a_ice = np.append(a_ice, [0.0])
        
    if wave is not None:
        a_ice = np.interp(wave, w_ice, a_ice)
        w_ice = wave
        
    return w_ice, a_ice
    
def abs_nvr(wave=None):
    path = webbpsf_ext.__path__[0]

    fname = os.path.join(path, 'throughputs/nvr_abs.txt')
    names = ['Wave', 'coeff'] # coeff is per um path length
    data_nvr  = ascii.read(fname, names=names)

    w_nvr = data_nvr['Wave']
    a_nvr = data_nvr['coeff']

    # Estimates for w>5.0um
    w_nvr = np.append(w_nvr, [6.0])    
    a_nvr = np.append(a_nvr, [0.0])

    if wave is not None:
        a_nvr = np.interp(wave, w_nvr, a_nvr)
        w_nvr = wave
        
    return w_nvr, a_nvr

def nc_eol_model(ice_thick=0.05, nvr_thick=0.189, wave=None):
    """Default NIRCam model for End of Life"""
    
    wave, a_nvr = abs_nvr(wave=wave)
    wave, a_ice = abs_nvr(wave=wave)

    t_ice = np.exp(-a_ice*ice_thick)
    t_nvr = np.exp(-a_nvr*nvr_thick)
    t_nc = t_ice * t_nvr
    
    t_nc = np.exp(-a_ice*ice_thick-a_nvr*nvr_thick)
    
    return wave, t_nc

def ote_eol_model(wave=None):
    """Default OTE model for End of Life"""
    path = webbpsf_ext.__path__[0]
    
    fname = os.path.join(path, 'throughputs/ote_nc_sim_1.00.txt')
    names = ['Wave', 't_ice', 't_nvr', 't_sys']
    data  = ascii.read(fname, data_start=1, names=names)

    wtemp = data['Wave']
    wtemp = np.insert(wtemp, 0, [1.0]) # Estimates for w<2.5um
    wtemp = np.append(wtemp, [6.0])    # Estimates for w>5.0um
    
    ttemp = data['t_ice']
    ttemp = np.insert(ttemp, 0, [1.0]) # Estimates for w<2.5um
    ttemp = np.append(ttemp, [1.0])    # Estimates for w>5.0um
    
    if wave is None:
        wave = wtemp
        th = ttemp
    else:
        th = np.interp(wave, wtemp, ttemp)
    
    return wave, th



# Fitting Functions

def abs_func(x, p, **kwargs):
    """Fit absorption coefficients to data"""

    t_abs = np.exp(-p[0]*kwargs['abs_ice'] - p[1]*kwargs['abs_nvr'])
    return np.interp(x, kwargs['wave_abs'], t_abs)

def cont_abs_func(x, p, **kwargs):
    """Fit absorption coefficients and polynomial continuum to data"""
    t_abs = np.exp(-p[-2]*kwargs['abs_ice'] - p[-1]*kwargs['abs_nvr'])
    t_abs = np.interp(x, kwargs['wave_abs'], t_abs)
    return jl_poly(x, p[0:-2])*t_abs

def fit_bootstrap(pinit, datax, datay, function, yerr_systematic=0.0, nrand=1000, 
                  return_more=False, **kwargs):
    """Bootstrap fitting routine
    
    Bootstrap fitting algorithm to determine the uncertainties on the fit parameters.
    
    Parameters
    ----------
    pinit : ndarray
        Initial guess for parameters to fit
    datax, datay : ndarray
        X and y values of data to be fit
    function : func
        Model function 
    yerr_systematic : float or array_like of floats
        Systematic uncertainites on contributing to additional error in data. 
        This is treated as independent Normal error on each data point.
        Can have unique values for each data point. If 0, then we just use
        the standard deviation of the residuals to randomize the data.
    nrand : int
        Number of random data sets to generate and fit.
    return_all : bool
        If true, then also return the full set of fit parameters for the randomized
        data to perform a more thorough analysis of the distribution. Otherewise, 
        just reaturn the mean and standard deviations.
    """

    def errfunc(p, x, y, **kwargs):
        return function(x, p, **kwargs) - y

    # Parameter limits
    lim_min = len(pinit) * [-np.inf]
    lim_max = len(pinit) * [np.inf]
    lim_min[-2] = 0
    lim_min[-1] = 0
    bounds = (lim_min, lim_max)
    
    # Fit first time
    #pfit, perr = optimize.leastsq(errfunc, pinit, args=(datax, datay), full_output=0)    
    res = optimize.least_squares(errfunc, pinit, bounds=bounds, args=(datax, datay), kwargs=kwargs)#, 
                                 #loss='huber', f_scale=0.1)
    pinit = res.x
    res = optimize.least_squares(errfunc, pinit, bounds=bounds, args=(datax, datay), kwargs=kwargs)#, 
                                 #loss='huber', f_scale=0.1)
    pfit = res.x

    if (nrand is not None) and (nrand>0):
        # Get the stdev of the residuals
        residuals = errfunc(pfit, datax, datay, **kwargs)
        sigma_res = np.std(residuals)

        sigma_err_total = np.sqrt(sigma_res**2 + yerr_systematic**2)

        # Some random data sets are generated and fitted
        randomdataY = datay + np.random.normal(scale=sigma_err_total, size=(nrand, len(datay)))
        ps = []
        for i in range(nrand):
            datay_rand = randomdataY[i]
            res = optimize.least_squares(errfunc, pinit, bounds=bounds, args=(datax, datay_rand), kwargs=kwargs)#, 
                                         #loss='huber', f_scale=0.1)
            randomfit = res.x

            ps.append(randomfit) 

        ps = np.array(ps)
        mean_pfit = np.mean(ps,axis=0)
        err_pfit = np.std(ps,axis=0)

        if return_more:
            return mean_pfit, err_pfit, ps
        else:
            return mean_pfit, err_pfit
        
    else:
        return pfit

def get_collecting_area(segment_name):
    
    from webbpsf.webbpsf_core import one_segment_pupil
    webbpsf_data_path = webbpsf.utils.get_webbpsf_data_path()

    coll_full = 25.78e4
    if 'ALL' in segment_name.upper():
        return coll_full

    # Pupil mask of segment only
    pupil_seg_hdul = one_segment_pupil(segment_name)

    # Full pupil file
    pupil_file = os.path.join(webbpsf_data_path, "jwst_pupil_RevW_npix1024.fits.gz")
    pupil_hdul = fits.open(pupil_file)

    # Change collecting area to modify flux 
    coll_area_seg = coll_full * pupil_seg_hdul[0].data.sum() / pupil_hdul[0].data.sum()
    pupil_hdul.close()

    return coll_area_seg

def sim_obs_spec(wave_out, coll_area_seg, bp_dict=None, sp=None, psf_sig_pix=5, 
                 filter='F322W2', pupil='GRISMR', module='A', new_cf=True, **kwargs):
    
    pupil='NONE' if pupil is None else pupil
    
    if bp_dict is None:
        bp = pynrc.read_filter(filter, pupil=pupil, module=module, **kwargs)
    else:
        bp = bp_dict[module][pupil][filter]
    
    if sp is None:
        bp_k = pynrc.bp_2mass('k')
        sp = pynrc.stellar_spectrum('K0V', 4.7, 'vegamag', bp_k, Teff=5000, log_g=4.3, metallicity=0)
        sp.convert('photlam')

    # Interpolate onto output grid
    # sp_val = np.interp(wave_out, sp.wave/1e4, sp.flux)
    # bp_val = np.interp(wave_out, bp.wave/1e4, bp.throughput)
    sp_func = interp1d(sp.wave/1e4, sp.flux, kind='cubic', fill_value=0)
    bp_func = interp1d(bp.wave/1e4, bp.throughput, kind='cubic', fill_value=0)
    
    sp_val = sp_func(wave_out)
    bp_val = bp_func(wave_out)
    
    if not (pupil=='NONE') and new_cf:
        bp_val /= grism_throughput(wave_out, module=module, new_cf=False)
        bp_val *= grism_throughput(wave_out, module=module, new_cf=True)
    
    # Multiply by bandpass, collecting area, and dw
    dw = np.mean(wave_out[1:] - wave_out[:-1]) * 1e4
    res = sp_val * bp_val * coll_area_seg * dw
    
    # Convolve with Gaussian
    g = Gaussian1DKernel(stddev=psf_sig_pix)
    res_conv = convolve(res, g)
    
    return res_conv

def grism_throughput(wave, module='A', grism_order=1, new_cf=False):
    """Grism transmission curve follows a 3rd-order polynomial"""
    
    # The following coefficients assume that wavelength is in um
    if (module == 'A') and (grism_order==1):
        if new_cf:
            # Tom G's blaze angle of 6.16 deg
            # cf_g = 1.14 * np.array([-0.44941915, -1.10918236, 1.09376196, -0.26981016, 0.02065479])
            # Newly fit grism efficiency estimate
            cf_g = np.array([-0.51108063, -0.46895732, 0.49194712, -0.0749169 ])
        else:
            cf_g = np.array([0.068695897, -0.943894294, 4.1768413, -5.306475735])[::-1]
    elif (module == 'B') and (grism_order==1):
        if new_cf:
            # Newly fit grism efficiency estimate
            cf_g = np.array([-3.91489419, 2.94875747, -0.61759455, 0.03881305])
        else:
            cf_g = np.array([0.050758635, -0.697433006, 3.086221627, -3.92089596])[::-1]
    elif (module == 'A') and (grism_order==2):
        cf_g = np.array([0.05172, -0.85065, 5.22254, -14.18118, 14.37131])[::-1]
    elif (module == 'B') and (grism_order==2):
        cf_g = np.array([0.03821, -0.62853, 3.85887, -10.47832, 10.61880])[::-1]
    
    return jl_poly(wave, cf_g)

##############################
## Data Reduction
##############################

def do_lincorr(data, det, bias_sub=True, superbias=None, cal_obj=None, cframe='sci', **kwargs):
    """Perform linearity corrections on image or cube"""

    cal_obj = nircam_cal(det.scaid, verbose=False) if cal_obj is None else cal_obj

    # Convert from sci to det coordinates
    if cframe=='sci':
        data = sci_to_det(data, det.detid)

    # Subtract bias
    if bias_sub:
        ny, nx = (det.ypix, det.xpix)
        x1, x2 = (det.x0, det.x0 + nx)
        y1, y2 = (det.y0, det.y0 + ny)
        superbias = cal_obj.super_bias if superbias is None else superbias
        data -= superbias[y1:y2,x1:x2]
    
    # Apply linearity correction
    data = apply_linearity(data, det, cal_obj.linear_dict)

    # Convert back to sci coordinates if input were sci
    if cframe=='sci':
        data = det_to_sci(data, det.detid)

    return data


def do_flatcorr(data, det, cal_obj=None, pflat_corr=True, lflat_corr=True, cframe='sci'):
    """Perform flat field corrections on image or cube"""
        
    if (not pflat_corr) and (not lflat_corr):
        return data
    
    cal_obj = nircam_cal(det.scaid, verbose=False) if cal_obj is None else cal_obj

    # Convert from sci to det coordinates
    if cframe=='sci':
        data = sci_to_det(data, det.detid)

    # Apply flat field corrections
    if pflat_corr and (cal_obj.pflats is not None):
        data = ngNRC.apply_flat(data, det, 1/cal_obj.pflats)
    if lflat_corr and (cal_obj.lflats is not None):
        data = ngNRC.apply_flat(data, det, 1/cal_obj.lflats)
        
    # Convert back to sci coordinates if input were sci
    if cframe=='sci':
        data = det_to_sci(data, det.detid)

    return data


def get_all_data(file, return_mean=True, mn_func=np.mean, reffix=True, 
                 lin_corr=True, pflat_corr=False, lflat_corr=False, cal_obj=None, **kwargs):
    
    
    hdul = fits.open(file)
    hdr = hdul[0].header
    hdul.close()

    det = create_detops(hdr, DMS=True)
    tarr = det.times_group_avg
    
    # Reference pixel correction keywords
    nbot, ntop, nleft, nright = det.ref_info
    kwargs_def = {
        'nchans': det.nout, 'altcol': True, 'in_place': True,    
        'fixcol': False, 'avg_type': 'pixel', 'savgol': True, 'perint': False,
        'nbot': nbot, 'ntop': ntop, 'nleft': nleft, 'nright': nright,
    }
    for k in kwargs_def.keys():
        if k not in kwargs:
            kwargs[k] = kwargs_def[k]
            
    # Cycle through each integration
    nint = det.multiaccum.nint
    imarr = []
    for i in trange(nint, desc='Integrations', leave=False):
        data = get_fits_data(file, DMS=True, int_ind=i, reffix=reffix, **kwargs)
        
        # Perform linearity correction?
        if lin_corr:
            if not reffix:
                print('Must perform reference pixel correction with linearity correction.')
            else:
                # Will subtract a superbias stored in cal_obj by default
                data = do_lincorr(data, det, cal_obj=cal_obj, **kwargs)
                
        # Perform flat field corrections (just returns input if _corr keywords are False)
        data = do_flatcorr(data, det, cal_obj=cal_obj, pflat_corr=pflat_corr, lflat_corr=lflat_corr)
            
        imarr.append(data)
    imarr = np.asarray(imarr)
    
    # Return average of all ramps or full array?
    if return_mean:
        return mn_func(imarr, axis=0)
    else:
        return imarr


def get_flat(indir, det, filt='F322W2', coords_out='det'):
    """Grab CDBS flat field data"""
    
    from pynrc.maths.coords import sci_to_det
    
    det_str = det.detname[3:].lower()
    flat_files = []
    for file in os.listdir(indir):
        if file.startswith("jwst_nircam_flat") and file.endswith(f"{filt}.fits") and (det_str in file):
            flat_files.append(file)
    # Choose more recent file to use
    file = np.sort(flat_files)[-1]
        
    hdul = fits.open(os.path.join(indir, file))
    flat_data = hdul[1].data.astype('float')
    hdul.close()

    # Place in 'det' coordinates
    if coords_out=='det':
        flat_data = sci_to_det(flat_data, det.detid)

    return flat_data

def do_flatcorr_cdbs(flatdir, data, det, filt='F322W2', cframe='sci'):
    """Perform flat field corrections on image or cube"""
        
    # Get flat data in detector coordinates
    flat_data = get_flat(flatdir, det, filt=filt, coords='det')

    # Flip to detector coordinates
    if cframe=='sci':
        data = sci_to_det(data, det.detid)
        
    data = ngNRC.apply_flat(data, det, 1/flat_data)
    
    # Convert back to sci coordinates if input were sci
    if cframe=='sci':
        data = det_to_sci(data, det.detid)
        
    return data


##############################


def average_diffs(ext_dict, interp='linear', mn_func=np.median):
    """Align all images and average"""
    
    # Get all Gaussian fit row values
    ycen_all = [ext_dict[k]['ycen'] for k in ext_dict.keys()]
    for i, yc in enumerate(ycen_all):
        if yc.size==0:
            ycen_all[i] = ycen_all[i-1]
    ycen_all = np.asarray(ycen_all)
    
    # Get average offsets
    yc_mean = np.mean(ycen_all, axis=0)
    yshift = np.mean(yc_mean - ycen_all, axis=1)

    # Shift all images
    keys = list(ext_dict.keys())
    data_all = []
    for k in keys:
        im = nrc_utils.fshift(ext_dict[k]['data'], dely=yshift[k], interp=interp)
        data_all.append(im)
        
    # Return average
    return mn_func(data_all, axis=0)



class water_analysis(object):
    
    def __init__(self, wave, flux, module, pupil, segid=None, sp=None, new_cf=True, **kwargs):
        
        self.wave = wave
        self.flux = flux
        self.segid = segid
        
        self.filter = 'F322W2'
        self.pupil  = pupil
        self.module = module
        
        # Set default spectum and interpolate onto wavelength grid
        if sp is None:
            bp_k = pynrc.bp_2mass('k')
            self.get_sp('K0V', 4.7, 'vegamag', bp_k, Teff=5000, log_g=4.3, metallicity=0)
        else:
            self.sp = sp
        
        # Load bandpass and interpolate onto wavelength grid
        self.bp = self.get_bp(new_cf=new_cf, **kwargs)

    def collecting_area(self):
        """"""
        segid = 'C5' if self.segid is None else self.segid
        return get_collecting_area(segid)
        

    def get_sp(self, spt, mag, units, bp_scale, Teff=5000, log_g=4.3, metallicity=0):
        """Spectrum in ph/s/cm^2/A"""        
        sp = pynrc.stellar_spectrum(spt, mag, units, bp_scale, 
                                    Teff=Teff, log_g=log_g, metallicity=metallicity)
        sp.convert('photlam')
        self.sp = sp
        self.set_spvals()
    
    def set_spvals(self):
        """"""
        sp = self.sp
        sp_func = interp1d(sp.wave/1e4, sp.flux, kind='cubic', fill_value=0)
        self.sp_vals = sp_func(self.wave)
        
    def get_bp(self, new_cf=True, **kwargs):
        """"""
        bp = pynrc.read_filter(self.filter, pupil=self.pupil, module=self.module, **kwargs)
        
        # Correct grism efficiency
        if new_cf:
            w = bp.wave / 1e4
            th = bp.throughput
            bp._throughputtable /= grism_throughput(w, module=self.module, new_cf=False)
            bp._throughputtable *= grism_throughput(w, module=self.module, new_cf=True)
        
        self.bp = bp
        self.set_bpvals()
        
    def set_bpvals(self):
        """"""
        bp = self.bp
        bp_func = interp1d(bp.wave/1e4, bp.throughput, kind='cubic', fill_value=0)
        bp_vals = bp_func(self.wave)

        self.bp_vals = bp_vals
        
    def normalize_spec(self, psf_sig_pix=5, include_resid=False, wmin=2.45, wmax=3.95, 
                       robust_fit=True, deg=1):
        """"""
        #Normalized value convolved with Gaussian
        norm_val = self.sp_vals*self.bp_vals
        g = Gaussian1DKernel(stddev=psf_sig_pix)
        norm_val_conv = convolve(norm_val, g)
        
        if include_resid:
            norm_val_conv *= self.estimate_resid()

        spec_corr = self.flux / norm_val_conv

        # Cut off edges and normalize at continuum location
        wave_extract = self.wave
        ind = (wave_extract>wmin) & (wave_extract<wmax)
        wind = wave_extract[ind]
        spec_fin = spec_corr[ind]
        
        # Normalize by average continuum level
        ind_cont = ((wind>2.4) & (wind<2.7)) | (wind>3.6)
        spec_fin /= np.nanmedian(spec_fin[ind_cont])
        
        # Indices to fit continuum
        w = wind
        if self.module=='A':
            ind_fit = ((w>=2.45) & (w<=2.55)) | ((w>=2.65) & (w<=2.7)) | ((w>=3.65) & (w<=3.85))
        else:
            ind_fit = (w<2.5) | ((w>=2.55) & (w<=2.7)) | ((w>=3.65) & (w<=3.85))    
        ind_fit = ind_fit & ~np.isnan(spec_fin)

        # Fit continuum
        cf = jl_poly_fit(wind[ind_fit], spec_fin[ind_fit], deg=deg, robust_fit=robust_fit)
        spec_cont = jl_poly(wind, cf)
        spec_norm = spec_fin / spec_cont
        
        dout = {'wave':wind, 'flux':spec_fin, 'flux_cont':spec_cont, 'flux_norm':spec_norm}
        return dout
        