"""
ngNRC - NIRCam Detector Noise Simulator

Modification History:

Nov 2021
    - Auto-generate source tables from Gaia DR2 or Simbad queries
    - Add WFE drifts.
Oct 2021
    - Use numpy random number generator objects to produce repeatable results.
Sept 2021
    - Major refactor, splitting out slope_to_level1b and slope_to_fitswriter
    - Added linearity, flat fields, and cosmic rays
Apr 2021
    - Deprecate nghxrg, SCANoise, and slope_to_ramp
    - Instead use slope_to_ramps
Oct 2020
    - Restructure det noise and ramp creation
    - DMS simulations using JWST pipeline data models
Feb 2017
    - Add ngNRC to pyNRC code base
Aug 2016, J.M. Leisenring, UA/Steward
    - Modified how the detector and multiaccum info is handled
    - Copied detector and multiaccum classes from pyNRC
    - In the future, we will want to integrate this directly
      so that any changes made in the pyNRC classes are accounted.
July 2016, J.M. Leisenring, UA/Steward
    - Updated many things and more for nghxrg (v3.0)
Feb 2016, J.M. Leisenring, UA/Steward
    - First Release
"""
import json
import numpy as np
import os

from copy import deepcopy

from astropy.io import fits

from datetime import datetime

from pynrc.logging_utils import setup_logging

from .dms import create_DMS_HDUList, level1b_data_model
from .dms import update_dms_headers, update_headers_pynrc_info
from .apt import DMS_input, gen_jwst_pointing
from ..nrc_utils import pad_or_cut_to_size, frebin, jl_poly, get_detname
from ..nrc_utils import gen_unconvolved_point_source_image
from ..reduce.calib import ramp_resample, nircam_cal
from ..maths.coords import det_to_sci, sci_to_det
from ..maths.image_manip import convolve_image
from .. import conf

from webbpsf_ext.image_manip import add_ipc, add_ppc
from webbpsf_ext.synphot_ext import Observation

from stdatamodels import fits_support
import astropy.units as u

# Program bar
from tqdm.auto import trange, tqdm

import logging
_log = logging.getLogger('pynrc')

def create_level1b_FITS(sim_config, detname=None, apname=None, filter=None, visit_id=None,
                        dry_run=None, save_slope=None, save_dms=None):

    """ Generate Level1b DMS-like FITS files.

    TODO:
        - Tracking image persistence
        - Static column noise that shifts every integration
        - Currently only has WFE drifts of coronagraphic on-mask stars (sci and ref)
        - Also drifts non-HCI observations, generating new PSF grids each exposure

    Keyword Args
    ============
    detname : None or int or str
        Option to supply a valid detector name. If set, only the currently
        specified SCA will be simulated.
    apname : None or str
        Similar to `detname` keyword, can supply a specific SIAF aperture
        name that exists within the observation. Only that aperture will
        be simulated.
    filter : None or str
        Specify a filter or list of filters within observation to be simulated. 
        Can be combined with `detname` and `apname` keywords.
    visit_id : None or str
        Specify the visit ID to simulate. Should have the form "ABC:XYZ",
        or "ObsNum:VisitNum" (e.g., "005:001" for observation 5, visit 1).
    dry_run : bool or None
        Won't generate any image data, but instead runs through each 
        observation, printing detector info, SIAF aperture name, filter,
        visit IDs, exposure numbers, and dither information.
        If set to None, then grabs keyword from `sim_config`, otherwise
        defaults to False if not found. 
        If paired with `save_dms`, then will generate an empty set of DMS FITS 
        files with headers populated, but data set to all zeros.
    save_slope : bool or None
        Saves noiseless slope images to a separate DMS-like FITS file
        that is names 'slope_{DMSfilename}'.
        If set to None, then grabs keyword from `sim_config`, otherwise
        defaults to False if not found. No effect if dry_run=True.
    save_dms : bool or None
        Option to disable simulation of ramp data and creation of DMS FITS.
        If dry_run=True, then setting save_dms=True will save DMS FITS
        files populated with all zeros.
        If set to None, then grabs keyword from `sim_config`; if no keyword
        is found, then defaults to True if dry_run=False, otherwise False.
    """

    from ..pynrc_core import DetectorOps, NIRCam
    from ..obs_nircam import obs_hci

    # Files and output directory
    json_file     = sim_config['json_file']
    sm_acct_file  = sim_config['sm_acct_file']
    pointing_file = sim_config['pointing_file']
    xml_file      = sim_config['xml_file']
    save_dir      = sim_config['save_dir']

    # Source information for full field observations
    params_targets = sim_config['params_targets']

    # PSF information
    kwargs_nrc = sim_config['params_webbpsf']
    kwargs_psf = sim_config['params_psfconv']
    kwargs_wfedrift = sim_config.get('params_wfedrift')
    large_grid = sim_config['large_grid']

    # Ramp noise simulation options
    kwargs_det = sim_config['params_noise']

    large_slew_uncert = sim_config['large_slew']
    ta_sam_uncert     = sim_config['ta_sam']
    std_sam_uncert    = sim_config['std_sam']
    sgd_sam_uncert    = sim_config['sgd_sam']

    # Random seed information
    rand_seed_init  = sim_config.get('rand_seed_init')

    if save_slope is None:
        save_slope = sim_config.get('save_slope', False)
    if dry_run is None:
        dry_run = sim_config.get('dry_run', False)
    # Saving DMS data is off by default if running dry run,
    # but keyword settings will take precedence.
    save_dms_def = False if dry_run else True
    if save_dms is None:
        save_dms = sim_config.get('save_dms', save_dms_def)


    #################################################
    # Create DMS Input class
    obs_input = DMS_input(xml_file, pointing_file, json_file, sm_acct_file,
                          save_dir=save_dir, rand_seed_init=rand_seed_init)

    # Update observing start date/time and V3 PA
    obs_input.obs_date = sim_config['obs_date']
    obs_input.obs_time = sim_config['obs_time']
    obs_input.pa_v3    = sim_config['pa_v3']

    # Generate all observation parameters for every visit, exposure, detector, etc
    obs_params_all = obs_input.gen_all_obs_params()
    obs_params_all = np.asarray(obs_params_all)
    
    obs_detnames = np.array([get_detname(par['detector']) for par in obs_params_all])
    obs_filters  = np.array([par['filter']                for par in obs_params_all])
    obs_apnames  = np.array([par['siaf_ap'].AperName      for par in obs_params_all])
    obs_targets  = np.array([par['target_name']           for par in obs_params_all])
    obs_visitids = np.array([par['visit_key']             for par in obs_params_all])

    # Unique labels for sorting
    obs_labels  = np.array([f'{a}_{f}_{t}' for a, f, t in zip(obs_apnames, obs_filters, obs_targets)])

    # WFE Drift information
    if kwargs_wfedrift is None:
        wfe_dict = None
        rand_seed_dwfe = None
    elif kwargs_wfedrift.get('wfe_dict') is not None:
        # wfe_dict already exists in passed parameters
        wfe_dict = kwargs_wfedrift.get('wfe_dict')
        rand_seed_dwfe = None
    else:
        plot_fig = kwargs_wfedrift.get('plot', False)
        figname = kwargs_wfedrift.get('figname', None)
        # Update figure save location
        if plot_fig and figname is not None:
            fileout = os.path.basename(figname)
            figpath = os.path.join(save_dir, fileout)
            kwargs_wfedrift['figname'] = figpath

        # Random seed info
        rand_seed_dwfe = kwargs_wfedrift.get('rand_seed')
        if rand_seed_dwfe is None:
            rng = np.random.default_rng(rand_seed_init)
            rand_seed_dwfe = rng.integers(0, 2**32-1)
            kwargs_wfedrift['rand_seed'] = rand_seed_dwfe

        wfe_dict = gen_wfe_drift(obs_input, **kwargs_wfedrift)

    # Select only a specific detector to generate simulations?
    if detname is None:
        udetnames = np.unique(obs_detnames)
    else:
        # Option to pass multiple names
        if isinstance(detname, str):
            udetnames = [get_detname(detname)]
        else:
            udetnames = np.unique(detname)
        
    if dry_run:
        print('DetID SIAFAperture Filter TargetName VisitID GSAid exp# (idl_off_act) dWFE time')


    for detname in udetnames:
        # print('detname: ', detname)

        ind = (obs_detnames == detname)
        if ind.sum()==0:
            _log.warning(f'Detector {detname} is not a valid detector for these observations.')
            continue

        # Create calibration object
        det = DetectorOps(detector=detname)
        caldir = os.path.join(conf.PYNRC_PATH, 'calib', str(det.scaid))
        if (not dry_run) and save_dms:
            cal_obj = nircam_cal(det.scaid, caldir, verbose=False)

        # Grab labels for only the specific detectors
        ulabels, ulabels_ind = np.unique(obs_labels[ind], return_index=True)
        # Grab the associated apnames and filters
        uapnames = obs_apnames[ind][ulabels_ind]
        ufilters = obs_filters[ind][ulabels_ind]
        # Create masks for desired filters and apnames
        if filter is not None:
            filt_select = [filter] if isinstance(filter, str) else np.unique(filter)
            filt_mask = np.array([ff in filt_select for ff in ufilters])
        if apname is not None:
            ap_select = [apname] if isinstance(apname, str) else np.unique(apname)
            ap_mask = np.array([aa in ap_select for aa in uapnames])
        # Select specific labels, which get parsed later
        if (apname is not None) and (filter is not None):
            ulabels = ulabels[filt_mask & ap_mask]
            log_print = _log.info
        elif (apname is not None) and (filter is None):
            ulabels = ulabels[ap_mask]
            log_print = _log.info
        elif (apname is None) and (filter is not None):
            ulabels = ulabels[filt_mask]
            log_print = _log.info
        else:
            log_print = _log.warn

        if len(ulabels)==0:
            log_print('No valid observations for specified parameters:')
            log_print(f'  SCA: {detname}, SIAF: {apname}, Filter: {filter}')
            continue

        # Cycle through all labels
        for label in ulabels:
            aname = '_'.join(label.split('_')[0:-2])
            fname = label.split('_')[-2]
            tname = label.split('_')[-1]

            # Ensure this label matches one in the full list
            ind2 = (obs_labels == label)
            if ind2.sum()==0:
                _log.warning(f'Skipping {aname} + {fname} + {tname} for {detname}...')
                continue

            # Select visit ids that have current obs config
            # Make sure we're not sorting
            _, uind = np.unique(obs_visitids[ind2], return_index=True)
            visit_ids = obs_visitids[ind2][np.sort(uind)]
            if visit_id is not None:
                ind_mask = (visit_ids == visit_id)
                visit_ids = visit_ids[ind_mask]
                log_print = _log.info
            else:
                log_print = _log.warn

            if len(visit_ids)==0:
                log_print('No valid Visit IDs for specified parameters:')
                log_print(f'  SCA: {detname}, SIAF: {apname}, Filter: {filter}, Visit: {visit_id}')
                continue

            # Grab a set of obs parameters to create some key info
            op_temp = obs_params_all[ind2][0]
            siaf_ap = op_temp['siaf_ap']

            # Check fov_pix size makes sense for aperture size
            # Reduce if too large. Make odd.
            kwargs_nrc2 = kwargs_nrc.copy()
            fov_pix = kwargs_nrc.get('fov_pix')
            if fov_pix is None:
                fov_pix = 641 if op_temp['channel'].upper()=='SHORT' else 321
            fov_pix = np.min([2*siaf_ap.XSciSize, 2*siaf_ap.YSciSize, fov_pix])
            fov_pix = fov_pix+1 if (fov_pix % 2)==0 else fov_pix
            kwargs_nrc2['fov_pix'] = fov_pix

            # Get target information
            target_id = op_temp['target_name']
            try:
                target_info = params_targets[target_id]
            except KeyError:
                try:
                    target_info = params_targets[target_id+'_ref']
                except KeyError:
                    _log.error(f"Cannot find {target_id} target! Skipping {aname} + {fname} for {detname}...")
                    continue

            # More target info
            src_tbl   = target_info.get('src_tbl')
            star_info = target_info.get('params_star')
            sp_star   = None if star_info is None else star_info.get('sp')
            dist_pc   = target_info.get('dist_pc')
            age_Myr   = target_info.get('age_Myr')
            planet_params = target_info.get('params_companions')
            disk_params   = target_info.get('params_disk_model')

            # Gather info to create NIRCam instrument object
            filt  = op_temp['filter']
            pupil = None if op_temp['pupil']=='CLEAR'             else op_temp['pupil']
            mask  = None if op_temp['coron_mask'].upper()=='NONE' else op_temp['coron_mask']
            ap_obs_name = siaf_ap.AperName
            ap_nrc_name = ap_obs_name

            # Is this a high-contrast imaging observation?
            is_hci = ('MASK' in ap_nrc_name) or (sp_star is not None)

            # Set pupil to None for some cases of coron and imaging mixing, eg. eng template
            if (pupil is not None) and ('MASK' in pupil) and ('MASK' not in ap_nrc_name):
                pupil = None

            if (not dry_run):
                # Get rid of previous instances
                try: del nrc
                except: pass

                if is_hci:
                    nrc = obs_hci(sp_star, dist_pc, filter=filt, apname=ap_nrc_name, 
                                  use_ap_info=True, disk_params=disk_params, 
                                  autogen_coeffs=False, detector=detname, **kwargs_nrc2)

                    nrc.gen_psf_coeff()
                    nrc.gen_wfemask_coeff(large_grid=large_grid)
                    if wfe_dict is not None:
                        nrc.gen_wfedrift_coeff()

                    # Create grid of PSFs
                    if (disk_params is not None) or (src_tbl is not None):
                        nrc.gen_disk_psfs(force=True)
                        hdul_psfs = nrc.psf_list

                    # Add planets
                    if planet_params is not None:
                        for kpl in planet_params:
                            if age_Myr is None:
                                _log.warning('Target age is not set. Assuming 100 Myr.')
                                age = 100
                            else:
                                age = age_Myr
                            kw = planet_params[kpl]
                            nrc.add_planet(age=age, **kw)

                else:
                    nrc = NIRCam(filter=filt, pupil_mask=pupil, image_mask=mask,
                                 detector=detname, apname=ap_nrc_name,
                                 autogen_coeffs=False, **kwargs_nrc2)

                    nrc.gen_psf_coeff()
                    nrc.gen_wfefield_coeff()
                    # nrc.gen_wfemask_coeff(large_grid=large_grid)
                    if wfe_dict is not None:
                        nrc.gen_wfedrift_coeff()

                    # Create grid of PSFs
                    # Skip if doing WFE drifts over time, since regenerated
                    # for each exposure.
                    if (src_tbl is not None) and (wfe_dict is None):
                        hdul_psfs = nrc.gen_psfs_over_fov(**kwargs_psf)
                    else:
                        hdul_psfs = None

                # Get Zodiacal background emission.
                ra_ref, dec_ref = (op_temp['ra_ref'], op_temp['dec_ref'])
                date_str = op_temp['date-obs']
                date_arg = (int(s) for s in date_str.split('-'))
                day_of_year = datetime(*date_arg).timetuple().tm_yday
                im_bg = nrc.bg_zodi_image(ra=ra_ref, dec=dec_ref, thisday=day_of_year)

            # Cycle through each of the visits
            for vid in visit_ids:
                # print('  vid: ', vid)
                visit_dict = obs_input.program_info[vid]

                # Get exposure IDs
                obs_dict_arr = visit_dict['obs_id_info']
                exp_ids = np.array([int(d['exposure_number']) for d in obs_dict_arr])
                grp_ids = np.array([int(d['visit_group'])     for d in obs_dict_arr])
                seq_ids = np.array([int(d['sequence_id'])     for d in obs_dict_arr])
                act_ids = np.array([    d['activity_id']      for d in obs_dict_arr])

                # Get type of observations (T-ACQ, CONFIRM, SCIENCE, ETC?)
                type_arr = visit_dict['type']
                type_vals = np.array(['T_ACQ', 'CONFIRM', 'SCIENCE'])
                # Cycle through each exposure in visit
                nexp = len(exp_ids)  # Total number of exposures with this aperture name
                for j in trange(nexp, desc="Exposures", leave=False):

                    # Create the observation parameters dictionary for selected exposure
                    exp_num = exp_ids[j]
                    grp_id = grp_ids[j]
                    seq_id = seq_ids[j]
                    act_id = act_ids[j]
                    act_int = int(act_id, 36) # Convert base 36 to integer number
                    # print('   exp_num: ', exp_num)
                    obs_params = obs_input.gen_obs_params(vid, exp_num, detname, grp_id=grp_id, 
                                                          seq_id=seq_id, act_id=act_id)

                    # Update some target info
                    obs_params['catalog_name'] = target_info.get('TargetArchiveName', 'UNKNOWN')
                    if 'RAProperMotion' in obs_params.keys():
                        obs_params['mu_RA']  = target_info['RAProperMotion']
                    if 'DecProperMotion' in obs_params.keys():
                        obs_params['mu_DEC']  = target_info['DecProperMotion']

                    # Random seed for noise uncertainties; each seed should be unique but also repeatable
                    rand_seed_noise_j = visit_dict['rand_seed_noise'] + grp_id*act_int*nexp + exp_num

                    # Random seed for dither uncertainties
                    tup = obs_params['visit_type'].upper() #type_arr[j].upper()
                    lup = obs_params['visit_level'].upper()
                    ddist = obs_params['ddist']
                    if tup not in type_vals:
                        _log.warning(f'Exposure type {tup} not recognized.')
                        _log.warning(f'  Visit {vid}, Exp {exp_num}, Grp {grp_id}, Seq {seq_id}, Act {act_id}')
                        continue
                    if tup=='T_ACQ':
                        # First target acq image involves small SAM (~3 pixels) from initial slew
                        # These are super accurate
                        rand_seed_base = rand_seed_dith = visit_dict['rand_seed_dith']
                        base_std = ta_sam_uncert #large_slew_uncert
                        dith_std = 0
                    elif (tup=='CONFIRM') and (ddist==0):
                        # First TA_CONF image is same position as T_ACQ
                        # These are super accurate
                        rand_seed_base = rand_seed_dith= visit_dict['rand_seed_dith']
                        base_std = ta_sam_uncert #large_slew_uncert
                        dith_std = 0
                    elif (tup=='CONFIRM'):
                        # Second TA_CONF has a small SAM from ND square to coron mask
                        rand_seed_base = rand_seed_dith = visit_dict['rand_seed_dith'] + 1
                        base_std = std_sam_uncert
                        dith_std = 0
                    elif (tup=='SCIENCE') and (lup=='TARGET'):
                        # Initial observation
                        rand_seed_base = visit_dict['rand_seed_dith'] + 1
                        rand_seed_dith = rand_seed_base + 1
                        base_std = large_slew_uncert if type_arr[0].upper()=='SCIENCE' else std_sam_uncert
                    elif (tup=='SCIENCE') and (lup=='TILE'):
                        rand_seed_base = visit_dict['rand_seed_dith'] + 1
                        rand_seed_dith = rand_seed_base + 1 + grp_id*act_int*nexp + exp_num
                        base_std = std_sam_uncert
                    elif (tup=='SCIENCE') and (lup=='FILTER'):
                        rand_seed_base = visit_dict['rand_seed_dith'] + 1
                        if ddist==0:
                            rand_seed_dith = rand_seed_base + 1
                        else:
                            rand_seed_dith = rand_seed_base + 1 + grp_id*act_int*nexp + exp_num
                        base_std = large_slew_uncert if type_arr[0].upper()=='SCIENCE' else ta_sam_uncert
                    elif (tup=='SCIENCE'):
                        pass  # No need to do anything, since no new activity
                    else:
                        raise ValueError(f'Not sure what to do with {lup}, {tup}, dDist={ddist:.3f}')

                    # New science pointing once per science activity
                    ra_ref, dec_ref = (obs_params['ra_ref'], obs_params['dec_ref'])
                    if tup=='T_ACQ' or tup=='CONFIRM':
                        tel_pointing = gen_jwst_pointing(visit_dict, obs_params, base_std=base_std, dith_std=dith_std,
                                                         rand_seed=rand_seed_dith, rand_seed_base=rand_seed_base)
                        tel_pointing.exp_nums = np.array([exp_num])
                    elif (tup=='SCIENCE') and (int(exp_num)==1):
                        # Make sure 
                        first_dith_zero = True if (lup=='TARGET' or ddist==0) else False
                        # Create jwst_pointing class
                        tel_pointing = gen_jwst_pointing(visit_dict, obs_params, base_std=base_std)
                        # Update standard and SGD uncertainties and regenerate random pointings
                        tel_pointing._std_sig = std_sam_uncert
                        tel_pointing._sgd_sig = sgd_sam_uncert
                        tel_pointing.gen_random_offsets(rand_seed=rand_seed_dith, rand_seed_base=rand_seed_base,
                                                        first_dith_zero=first_dith_zero)
                        tel_pointing.exp_nums = np.arange(tel_pointing.ndith) + 1 

                    # Save random seed in obs_params
                    obs_params['rand_seed_init']  = rand_seed_init
                    obs_params['rand_seed_dith']  = rand_seed_dith
                    obs_params['rand_seed_noise'] = rand_seed_noise_j
                    obs_params['rand_seed_dwfe']  = rand_seed_dwfe

                    # Skip slope creation if obs_label doesn't match NIRCam class
                    a = obs_params['siaf_ap'].AperName
                    f = obs_params['filter']
                    t = obs_params['target_name']
                    if label != f'{a}_{f}_{t}':
                        continue

                    # Some detectors don't have coronagraphic SIAF apertures.
                    # We want to skip creation of their FITS files, because those
                    # simulated observations would assume direct imaging apertures, 
                    # which would produce incorrect PSFs and wrong pointing info.
                    p = obs_params['pupil']
                    if (p is not None) and ('MASK' in p) and ('MASK' not in a):
                        continue

                    # Create dictionary of parameters to save to FITS header
                    kwargs_pynrc = kwargs_det.copy()
                    kwargs_pynrc['json_file']     = os.path.basename(json_file)
                    kwargs_pynrc['sm_acct_file']  = os.path.basename(sm_acct_file)
                    kwargs_pynrc['pointing_file'] = os.path.basename(pointing_file)
                    kwargs_pynrc['xml_file']      = os.path.basename(xml_file)

                    # Save simulated offset positions incl. random mispointings
                    expnum = int(obs_params['obs_id_info']['exposure_number'])
                    ind = np.where(tel_pointing.exp_nums == expnum)[0][0]
                    idl_off = tel_pointing.position_offsets_act[ind]
                    kwargs_pynrc['xoffset_act'] = idl_off[0]
                    kwargs_pynrc['yoffset_act'] = idl_off[1]
                    kwargs_pynrc['pa_v3'] = obs_params['pa_v3']

                    # WFE drift values
                    tval_exp = obs_params['texp_start_relative']  # seconds
                    if wfe_dict is not None:
                        tval_all = wfe_dict['time_sec']
                        wfe_total = wfe_dict['total']
                        wfe_drift_exp = np.interp(tval_exp, tval_all, wfe_total)
                    else:
                        wfe_drift_exp = 0
                    obs_params['wfe_drift'] = wfe_drift_exp

                    
                    if dry_run:
                        idl_off_str = f'({idl_off[0]:+0.3f}, {idl_off[1]:+0.3f})'
                        gsa_str = f'{grp_id:02d}{seq_id:01d}{act_id}'
                        dwfe = f'{wfe_drift_exp:.2f}'

                        print(detname, a, f, t, vid, gsa_str, exp_num, idl_off_str, dwfe, tval_exp)

                        # Save Level1b DMS FITS files without any data
                        if save_dms:
                            obs_params['filename'] = 'empty_' + obs_params['filename']
                            create_DMS_HDUList(None, None, obs_params, 
                                               save_dir=save_dir, **kwargs_pynrc)
                    else: # Generate dithered slope image for given exposure ID

                        # Start with background
                        # im_bg was initially created above for this 
                        # aperture, filter, target but the date may have changed
                        ra_ref, dec_ref = (obs_params['ra_ref'], obs_params['dec_ref'])
                        date_str = obs_params['date-obs']
                        date_arg = (int(s) for s in date_str.split('-'))
                        day_of_year2 = datetime(*date_arg).timetuple().tm_yday
                        if day_of_year == day_of_year2:
                            im_slope = im_bg.copy()
                        else:
                            day_of_year = day_of_year2
                            im_bg = nrc.bg_zodi_image(ra=ra_ref, dec=dec_ref, thisday=day_of_year)
                            im_slope = im_bg.copy()

                        if src_tbl is not None:
                            # Create slope image from table
                            # res = (src_tbl, nrc, obs_params, tel_pointing, hdul_psfs, wfe_drift_exp)
                            # return res
                            im_slope += sources_to_slope(src_tbl, nrc, obs_params, tel_pointing,
                                                         hdul_psfs=hdul_psfs, im_bg=0, 
                                                         wfe_drift=wfe_drift_exp)

                        # return nrc, obs_params['pa_v3'],idl_off,wfe_drift_exp

                        # Create slope image from HCI observation
                        # Only add if a stellar source was included
                        if is_hci and (sp_star is not None):
                            pa_v3 = obs_params['pa_v3']
                            im_slope += nrc.gen_slope_image(PA=pa_v3, xyoff_asec=idl_off, 
                                                            zfact=0, exclude_noise=True, 
                                                            wfe_drift0=wfe_drift_exp)

                        # Save slope image
                        if save_slope:
                            save_slope_image(im_slope, obs_params, 
                                             save_dir=save_dir, **kwargs_pynrc)

                        # Simulate ramp and stuff into a level1b file
                        if save_dms:
                            obs_params['filename'] = 'pynrc_' + obs_params['filename']
                            slope_to_level1b(im_slope, obs_params, cal_obj=cal_obj, 
                                             save_dir=save_dir, **kwargs_pynrc)

        # Get rid of previous NIRCam instances
        try: del nrc
        except: pass



def save_slope_image(im_slope, obs_params, save_dir=None, **kwargs):
    """Save slope image as Level1b-like FITS"""

    # Create data model for headers
    outModel = level1b_data_model(obs_params)
    hdul_temp, _ = fits_support.to_fits(outModel._instance, outModel._schema)

    # Create a new HDUList for the slope image
    hdu1 = fits.PrimaryHDU(header=hdul_temp[0].header.copy())
    hdu2 = fits.ImageHDU(data=im_slope, header=hdul_temp[1].header.copy(strip=True))
    hdul_slope = fits.HDUList([hdu1, hdu2])

    # Save slope FITS file
    file_slope = 'slope_' + obs_params['filename']
    if save_dir is not None:
        # Create directory and intermediates if they don't exist
        os.makedirs(save_dir, exist_ok=True)
        file_slope = os.path.join(save_dir, file_slope)

    # Save model to DMS FITS file 
    print(f'  Saving: {file_slope}')
    hdul_slope.writeto(file_slope, overwrite=True)

    hdul_temp.close()
    hdul_slope.close()

    # Update some WCS and segment info
    update_dms_headers(file_slope, obs_params)
    update_headers_pynrc_info(file_slope, obs_params, **kwargs)


def slope_to_level1b(im_slope, obs_params, cal_obj=None, save_dir=None, 
                     cframe='sci', out_ADU=True, **kwargs):
    """Simulate DMS HDUList from slope image
    
    Requires input of obs_params input dictionary as generated from
    APT input files (see `DMS_input` class in apt.py). 

    Also, make sure the `calib` directory exists in PYNRC_PATH and is 
    populated with detector calibration information.

    Look at keyword args to exclude specific detector effects.

    Output is saved to disk and will not be returned by the function.

    Parameters
    ==========
    im_slope : ndarray
        Slope in e-/sec of image from all sky sources, including
        Zodiacal background. Should exclude dark current background,
        which is handled separately from calib directory.
    obs_params : dict
        Dictionary of parameters to populate DMS header. 
        See `create_DMS_HDUList` in dms.py.
    cal_obj : :class:`~pynrc.nircam_cal`
        DMS object built from exported APT files. 
        See `DMS_input` in apt.py.
    save_dir : None or str
        Option to override output directory as specified in `obs_params` dictionary.
        If not specified as either a function keyword or in `obs_params`, then files 
        are saved in current working directory.
    cframe : str
        Coordinate frame of input slope, 'sci' or 'det'.

    Keyword Args
    ============
    include_dark : bool
        Add dark current?
    include_bias : bool
        Add detector bias?
    include_ktc : bool
        Add kTC noise?
    include_rn : bool
        Add readout noise per frame?
    include_cpink : bool
        Add correlated 1/f noise to all amplifiers?
    include_upink : bool
        Add uncorrelated 1/f noise to each amplifier?
    include_acn : bool
        Add alternating column noise?
    apply_ipc : bool
        Include interpixel capacitance?
    apply_ppc : bool
        Apply post-pixel coupling to linear analog signal?
    include_refoffsets : bool
        Include reference offests between amplifiers and odd/even columns?
    include_refinst : bool
        Include reference/active pixel instabilities?
    include_colnoise : bool
        Add in column noise per integration?
    col_noise : ndarray or None
        Option to explicitly specify column noise distribution in
        order to shift by one for subsequent integrations
    amp_crosstalk : bool
        Crosstalk between amplifiers?
    add_crs : bool
        Add cosmic ray events? See Robberto et al 2010 (JWST-STScI-001928).
    cr_model: str
        Cosmic ray model to use: 'SUNMAX', 'SUNMIN', or 'FLARES'.
    cr_scale: float
        Scale factor for probabilities.
    latents : None
        Apply persistence.
    apply_nonlinearity : bool
        Apply non-linearity?  
    random_nonlin : bool
        Add randomness to the linearity coefficients?
    prog_bar : bool
        Show a progress bar for this ramp generation?
    """
    
    rseed_init = obs_params.get('rand_seed_noise')
    rng = np.random.default_rng(rseed_init)

    det = obs_params['det_obj']
    nint = det.multiaccum.nint

    # Put into 'sci' coords
    if cframe=='det':
        im_slope = det_to_sci(im_slope, det.detid)
    
    if cal_obj is None:
        cal_obj = nircam_cal(det.scaid, verbose=False)
        
    # Simulate data for a single ramp
    sci_data = []
    zero_data = []
    for i in trange(nint, desc='Ramps', leave=False):
        # New random seed for noise generator initialization
        rand_seed = rng.integers(0, 2**32-1)
        res = simulate_detector_ramp(det, cal_obj, im_slope=im_slope, return_zero_frame=True,
                                     return_full_ramp=False, cframe='sci', out_ADU=out_ADU, 
                                     rand_seed=rand_seed, **kwargs)
        # Append to lists
        sci_data.append(res[0])
        zero_data.append(res[1])
    
    # Convert to single np.array
    sci_data = np.asarray(sci_data)
    zero_data = np.asarray(zero_data)

    # Create and save Level 1b data model to disk
    # This also updates header information
    create_DMS_HDUList(sci_data, zero_data, obs_params, save_dir=save_dir, **kwargs)

def sources_to_slope(source_table, nircam_obj, obs_params, tel_pointing, 
                     hdul_psfs=None, im_bg=None, cframe_out='sci', **kwargs):
    
    """ Create a slope image from a table or sources

    Parameters
    ==========
    source_table : astropy Table
        Table of objects in across the region, including headers
        'ra', 'dec', and object fluxes in NIRCam filter in vega mags where
        headers are labeled the filter name (e.g, 'F444W').
    nircam_obj : :class:`~pynrc.NIRCam`
        NIRCam instrument class for PSF generation.
    obs_params : dict
        Dictionary of parameters to populate DMS header. 
        See ``create_obs_params`` in apt.py and ``level1b_data_model`` in dms.py.
    tel_pointing : :class:`webbpsf_ext.jwst_point`
        JWST telescope pointing information. Holds pointing coordinates 
        and dither information for a given telescope visit.
    hdul_psfs : HDUList
        Option to pass a pre-generated HDUList of PSFs across the field of view.
        If set to None, then generated automatically.
    im_bg : None or ndarray
        Option to specify a pre-generated image (or single value) of the
        Zodiacal background emission. If not specified, then gets
        automatically generating.
    cframe_out : str
        Desired output coordinate frame, either 'sci' or 'det'

    Keyword Args
    ============
    npsf_per_full_fov : int
        Number of PSFs across one dimension of the instrument's field of 
        view. If a coronagraphic observation, then this is for the nominal
        coronagrahic field of view.
    sptype : str
        Spectral type to assume point source when generating total counts.
        Default is 'G0V'.
    wfe_drift : float
        Desired WFE drift value relative to default OPD.
    osamp : int
        Sampling of output PSF relative to detector sampling. If `hdul_psfs` is 
        specified, then the 'OSAMP' header keyword takes precedence.
    use_coeff : bool
        If True, uses ``calc_psf_from_coeff``, other WebbPSF's built-in ``calc_psf``.
        Coefficients are much faster. Default is True.
    """

    nrc = nircam_obj
    siaf_ap = obs_params['siaf_ap']
    det = obs_params['det_obj']

    # Get oversampling
    if hdul_psfs is not None: # First check hdul_psfs
        osamp = hdul_psfs[0].header['OSAMP']
        if ('osamp' in kwargs.keys()) and (kwargs['osamp']!=osamp):
            osamp2 = kwargs['osamp']
            _log.warning(f'Conflict between osamp in kwargs ({osamp2}) and osamp in PSF header ({osamp}). Using header.')
        kwargs['osamp'] = osamp
    elif 'osamp' in kwargs.keys():
        osamp = kwargs['osamp']
    else:
        osamp = 1
        kwargs['osamp'] = osamp
        
    ###############################
    # Generate unconvolved image
    
    # RA and Dec of all objects in field
    ra_deg, dec_deg = (source_table['ra'], source_table['dec'])
    # Vega magnitude values
    filt = obs_params['filter']
    mags = source_table[filt].data
    expnum = int(obs_params['obs_id_info']['exposure_number'])
    hdul_sci_image = gen_unconvolved_point_source_image(nrc, tel_pointing, ra_deg, dec_deg, mags, 
                                                        expnum=expnum, **kwargs)
    
    ###############################
    # Convolve full image with PSFs

    # Perform convolution
    if nrc.is_coron:
        # For coronagraphy, assume position-dependent PSFs are a function
        # of coronagraphic mask transmission, off-axis, and on-axis PSFs.
        # PSF(x,y) = trans(x,y)*psf_off + (1-trans(x,y))*psf_on

        # Get transmission mask and 
        trans = nrc.mask_images['OVERMASK']
        scale = kwargs['osamp'] / nrc.oversample
        trans = frebin(trans, scale=scale, total=False)
        trans_oversized = np.ones_like(hdul_sci_image[0].data)
        x0, y0 = (hdul_sci_image[0].header['XSCI0'], hdul_sci_image[0].header['YSCI0'])
        x1 = int(np.abs(x0*kwargs['osamp']))
        y1 = int(np.abs(y0*kwargs['osamp']))
        x2, y2 = (x1 + trans.shape[1], y1 + trans.shape[0])
        trans_oversized[y1:y2,x1:x2] = trans

        # Off-axis component
        hdul_sci_image_off = deepcopy(hdul_sci_image)
        hdul_sci_image_off[0].data = hdul_sci_image_off[0].data * trans_oversized
        psf_off = nrc.calc_psf_from_coeff(use_bg_psf=True, return_oversample=True, return_hdul=True)
        hdul_conv_off = convolve_image(hdul_sci_image_off, psf_off, output_sampling=1, return_hdul=True)

        # On-axis component (closest PSF convolution, only for bar mask)
        hdul_sci_image_on = deepcopy(hdul_sci_image)
        hdul_sci_image_on[0].data = hdul_sci_image_on[0].data * (1 - trans_oversized)
        hdul_psfs = nrc.psf_list if hdul_psfs is None else hdul_psfs
        hdul_conv_on = convolve_image(hdul_sci_image_on, hdul_psfs, output_sampling=1, return_hdul=True)
        
        hdul_sci_conv = hdul_conv_on
        hdul_sci_conv[0].data += hdul_conv_off[0].data
    else:
        if hdul_psfs is None:
            hdul_psfs = nrc.gen_psfs_over_fov(return_coords=None, **kwargs)

        hdul_sci_conv = convolve_image(hdul_sci_image, hdul_psfs, output_sampling=1, return_hdul=True)

    im_conv = hdul_sci_conv[0].data
    ny, nx = im_conv.shape
    xsci = np.arange(nx) + hdul_sci_conv[0].header['XSCI0']
    ysci = np.arange(ny) + hdul_sci_conv[0].header['YSCI0']

    # Crop out relevant region
    xind = (xsci>=0) & (xsci<siaf_ap.XSciSize)
    yind = (ysci>=0) & (ysci<siaf_ap.YSciSize)
    im_slope = im_conv[yind,:][:,xind]
    
    ###############################
    # Add zodiacal background

    # Get Zodiacal background emission.
    # Can be reused for all ints in same observation.
    if im_bg is None:
        date_str = obs_params['date-obs']
        date_arg = (int(s) for s in date_str.split('-'))
        day_of_year = datetime(*date_arg).timetuple().tm_yday
        ra, dec = tel_pointing.ap_radec()
        # ra, dec = (obs_params['ra_ref'], obs_params['dec_ref'])
        im_bg = nrc.bg_zodi_image(ra=ra, dec=dec, thisday=day_of_year)
        
    # Add background
    im_slope = im_slope + im_bg

    if cframe_out=='det':
        im_slope = sci_to_det(im_slope, det.detid)

    return im_slope

def sources_to_level1b(source_table, nircam_obj, obs_params, tel_pointing, 
                       hdul_psfs=None, cal_obj=None, im_bg=None, 
                       out_ADU=True, save_dir=None, **kwargs):
    """Simulate DMS HDUList from slope image
    
    Requires input of obs_params input dictionary as generated from
    APT input files (see `DMS_input` class in apt.py). 

    Also, make sure the `calib` directory exists in PYNRC_PATH and is 
    populated with detector calibration information.

    Look at keyword args to exclude specific detector effects.

    Output is saved to disk and will not be returned by the function.

    Parameters
    ==========
    source_table : astropy Table
        Table of objects in across the region, including headers
        'ra', 'dec', and object fluxes in NIRCam filter in vega mags where
        headers are labeled the filter name (e.g, 'F444W').
    nircam_obj : :class:`~pynrc.NIRCam`
        NIRCam instrument class for PSF generation.
    obs_params : dict
        Dictionary of parameters to populate DMS header. 
        See `create_obs_params` in apt.py and `level1b_data_model` in dms.py.
    tel_pointing : :class:`webbpsf_ext.jwst_point`
        JWST telescope pointing information. Holds pointing coordinates 
        and dither information for a given telescope visit.
    cal_obj : :class:`~pynrc.nircam_cal`
        NIRCam calibration class that holds the necessary calibration 
        info to simulate a ramp.
    im_bg : None or ndarray
        Option to specify a pre-generated image (or single value) of the
        Zodiacal background emission. If not specified, then gets
        automatically generating.
    save_dir : None or str
        Option to override output directory as specified in `obs_params` dictionary.
        If not specified as either a function keyword or in `obs_params`, then files 
        are saved in current working directory.

    Keyword Args
    ============
    npsf_per_full_fov : int
        Number of PSFs across one dimension of the instrument's field of 
        view. If a coronagraphic observation, then this is for the nominal
        coronagrahic field of view.
    sptype : str
        Spectral type, such as 'A0V' or 'K2III'.
    wfe_drift : float
        Desired WFE drift value relative to default OPD.
    osamp : int
        Sampling of output PSF relative to detector sampling. If `hdul_psfs` is 
        specified, then the 'OSAMP' header keyword takes precedence.
    use_coeff : bool
        If True, uses `calc_psf_from_coeff`, other WebbPSF's built-in `calc_psf`.
        Coefficients are much faster

    Ramp Gen Keywords
    =================
    include_dark : bool
        Add dark current?
    include_bias : bool
        Add detector bias?
    include_ktc : bool
        Add kTC noise?
    include_rn : bool
        Add readout noise per frame?
    include_cpink : bool
        Add correlated 1/f noise to all amplifiers?
    include_upink : bool
        Add uncorrelated 1/f noise to each amplifier?
    include_acn : bool
        Add alternating column noise?
    apply_ipc : bool
        Include interpixel capacitance?
    apply_ppc : bool
        Apply post-pixel coupling to linear analog signal?
    include_refoffsets : bool
        Include reference offests between amplifiers and odd/even columns?
    include_refinst : bool
        Include reference/active pixel instabilities?
    include_colnoise : bool
        Add in column noise per integration?
    col_noise : ndarray or None
        Option to explicitly specify column noise distribution in
        order to shift by one for subsequent integrations
    amp_crosstalk : bool
        Crosstalk between amplifiers?
    add_crs : bool
        Add cosmic ray events? See Robberto et al 2010 (JWST-STScI-001928).
    cr_model: str
        Cosmic ray model to use: 'SUNMAX', 'SUNMIN', or 'FLARES'.
    cr_scale: float
        Scale factor for probabilities.
    latents : None
        Apply persistence.
    apply_nonlinearity : bool
        Apply non-linearity?
    random_nonlin : bool
        Add randomness to the linearity coefficients?
    prog_bar : bool
        Show a progress bar for this ramp generation?
    """

    # Create a slope image in 'sci' coords
    im_slope = sources_to_slope(source_table, nircam_obj, obs_params, tel_pointing, 
                                hdul_psfs=hdul_psfs, im_bg=im_bg, **kwargs)

    kwargs['cframe'] = kwargs.get('cframe_out', 'sci')   
    slope_to_level1b(im_slope, obs_params, cal_obj=cal_obj, save_dir=save_dir, 
                     out_ADU=out_ADU, **kwargs)


def slope_to_fitswriter(det, cal_obj, im_slope=None, cframe='det',
    filter=None, pupil=None, targ_name=None, obs_time=None, file_out=None, 
    out_ADU=True, return_results=True, rand_seed=None, **kwargs):
    
    """Simulate HDUList from slope image

    FITSWriter-like output. DMS output has been depreceated and moved 
    to `slope_to_level1b` and `sources_to_level1b`.
    
    Parameters
    ==========
    det : Detector Class
        Desired detector class output
    dark_cal_obj: nircam_cal class
        NIRCam calibration class that holds the necessary calibration 
        info to simulate a ramp.
    im_slope : ndarray
        Input slope image of observed scene. Assumed to be in detector
        coordinates. If an image cube, then number of images must match 
        the number of integration (`nint`) in `det` class.
    cframe : str
        Orientation of im_slope. Either 'det' or 'sci' coordinate frame.
    filter : str
        Name of filter element for header
    pupil : str
        Name of pupil element for header
    targ_name : str
        Target name (optional)
    obs_time : datetime 
        Specifies when the observation was considered to be executed.
        If not specified, then it will choose the current time.
        This information is added to the header. Must be a datetime object:
            
            >>> datetime.datetime(2016, 5, 9, 11, 57, 5, 796686)
            
    file_out : str or None
        Name (including directory) to save FITS file. 
        If None, then won't save; make sure to set return_results=True.
    out_ADU : bool
        If true, divide by gain and convert to 16-bit UINT.
    return_results : bool
        Return HDUList result? Otherwise, 

    Keyword Args
    ============
    return_full_ramp : bool
        By default, we average groups and drop frames as specified in the
        `det` input. If this keyword is set to True, then return all raw
        frames within the ramp. The last set of `nd2` frames will be omitted.
    include_dark : bool
        Add dark current?
    include_bias : bool
        Add detector bias?
    include_ktc : bool
        Add kTC noise?
    include_rn : bool
        Add readout noise per frame?
    include_cpink : bool
        Add correlated 1/f noise to all amplifiers?
    include_upink : bool
        Add uncorrelated 1/f noise to each amplifier?
    include_acn : bool
        Add alternating column noise?
    apply_ipc : bool
        Include interpixel capacitance?
    apply_ppc : bool
        Apply post-pixel coupling to linear analog signal?
    include_refinst : bool
        Include reference/active pixel instabilities?
    include_colnoise : bool
        Add in column noise per integration?
    col_noise : ndarray or None
        Option to explicitly specify column noise distribution in
        order to shift by one for subsequent integrations
    amp_crosstalk : bool
        Crosstalk between amplifiers?
    add_crs : bool
        Add cosmic ray events?
    latents : None
        Apply persistence.
    linearity_map : ndarray
        Add non-linearity.
    """

    if kwargs.get('DMS') is not None:
        raise ValueError("DMS keyword is not valid. Instead, see `slope_to_level1b` or `sources_to_level1b`.")

    if (file_out is None) and (not return_results):
        raise ValueError("Set either file_out or return_results=True")
            
    # FITSWriter (ISIM format)
    if cframe=='sci':
        im_slope = sci_to_det(im_slope, det.detid)

    data = simulate_detector_ramp(det, cal_obj, im_slope=im_slope, cframe='det',
                                  out_ADU=out_ADU, return_zero_frame=False, 
                                  rand_seed=rand_seed, **kwargs)
    hdu = fits.PrimaryHDU(data)
    hdu.header = det.make_header(filter, pupil, obs_time, targ_name=targ_name, DMS=False)

    if file_out is not None:
        hdu.header['FILENAME'] = os.path.split(file_out)[1]
    outHDUList = fits.HDUList([hdu])

    # Write file to disk
    if file_out is not None:
        outHDUList.writeto(file_out, overwrite='True')
        
    # Only return outHDUList if return_results=True
    if return_results: 
        return outHDUList
    else:
        outHDUList.close()


def make_gaia_source_table(coords, remove_cen_star=True, radius=6*u.arcmin,
    teff_default=5800, dist_crossmatch=0.1):
    """ Create source table from GAIA DR2 query

    Generates a table of objects by performing a cone search around a set
    of input coordinates. The output table includes both coordinates and
    extrapolated photometry. All returned coordinates are for epoch=2015.5 
    for GAIA DR2.

    The process involves performing a search query around a set of input
    coordinates, then using the GAIA magnitudes and Teff to extrapolate
    magnitudes in the NIRCam filters. If Teff isn't supplied, then it
    defaults to solar temperature. If an object's parallax isn't 
    available, then the object is assumed to have a flat spectrum in
    F_lambda.


    Parameters
    ==========
    coords : SkyCoord
        Astropy SkyCoord object.
    remove_cen_star : bool
        Output will exclude the star associated with the input
        coordinates (anything with ``dist_crossmatch`` is removed 
        from table).
    radius : Units
        Radius to perfrom search. Default is 6', which should encompass
        NIRCam's full FoV, including both Modules A and B.
    teff_default : string
        Default stellar effective temperature to assume for sources 
        without GAIA information to extrapolate to longer wavelengths.
    dist_crossmatch : float
        Crossmatch GAIA coordinates with input coordinates within this distance.
        Defalult: 0.1".
    """
    
    from astroquery.gaia import Gaia
    from astropy.table import Table
    from astropy.time import Time

    from .. import stellar_spectrum, read_filter
    from ..nrc_utils import S, bp_gaia

    Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"
    Gaia.ROW_LIMIT = -1
    
    try:
        # Convert to 2015.5 epoch of GAIA coordinates
        coord_query = coords.apply_space_motion(new_obstime=Time('J2015.5'))
    except:
        _log.warning(f'Unable to convert to J2015.5 epoch. Continuing with {coords.obstime}...')
        coord_query = coords
    gaia_tbl = Gaia.query_object_async(coord_query, radius=radius)
    
    # Remove items without any photometry
    ind_nomag = gaia_tbl['phot_g_mean_mag'].data.mask
    if remove_cen_star:
        dist_asec = gaia_tbl['dist'].data * 3600
        ind_dist  = dist_asec.data < dist_crossmatch
        ind_remove = np.where(ind_nomag | ind_dist)
    else:
        ind_remove = np.where(ind_nomag)
    gaia_tbl.remove_rows(ind_remove)

    ra_name  = 'ra'
    dec_name = 'dec'
        
    # Create new source table
    index = np.arange(len(gaia_tbl)) + 1
    ra = gaia_tbl[ra_name]
    dec = gaia_tbl[dec_name]
    gband = gaia_tbl['phot_g_mean_mag'].data.data
    src_tbl = Table([index, ra, dec, gband], names=('index', 'ra', 'dec', 'g-band'))
    # Add distances
    src_tbl.add_column(gaia_tbl['dist'].data * 3600, name='dist')
    src_tbl['dist'].unit = 'arcsec'

    # Get effective temperature for each object rounded to the nearest 100K
    teff = (gaia_tbl['teff_val'].data.data / 100).astype('int') * 100
    # Assume solar temperature for null data
    teff[gaia_tbl['teff_val'].data.mask] = teff_default
    
    # Create stellar spectra of each unique temperature
    sp_dict = {}
    for tval in np.unique(teff):
        key = int(tval)
        sp_dict[key] = stellar_spectrum('G2V', Teff=tval, log_g=4.5, metallicity=0)
    # Add a flat spectrum for those objects without parallax measurements
    sp_dict['flat'] = stellar_spectrum('flat')
        
    # SW Filters
    filts_sw = [
        'F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F150W2',
        'F162M', 'F164N', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N'
    ]
    # LW Filters
    filts_lw = [
        'F250M', 'F277W',  'F300M','F322W2', 'F323N', 'F335M', 'F356W', 'F360M', 
        'F405N', 'F410M', 'F430M', 'F444W', 'F460M', 'F466N', 'F470N', 'F480M'
    ]
    filts_all = filts_sw + filts_lw
    
    # Read in all bandpasses
    bp_dict = {}
    for f in filts_sw:
        bp_dict[f] = read_filter(f)
    for f in filts_lw:
        bp_dict[f] = read_filter(f)
    
    # Cycle through all filters and sources to get 0-mag values
    # Gaia bandpass filter for normalization
    bp_g = bp_gaia('g', release='DR2')
    bp_sp_dict = {}
    for f in tqdm(filts_all, leave=False, desc='Filters'):
        bp = bp_dict[f]
        d = {}
        for k in sp_dict.keys():
            sp = sp_dict[k]
            sp = sp.renorm(0, 'vegamag', bp_g)
            obs = Observation(sp, bp, binset=bp.waveset)
            d[k] = obs.effstim('vegamag')
        bp_sp_dict[f] = d
    
    # Add Teff data to table
    src_tbl['Teff'] = teff

    # Add type column (star or galaxy)
    ind_nopara = gaia_tbl['parallax'].data.mask | (gaia_tbl['parallax'].data.data < 0)
    src_type = np.array(['galaxy' if v else 'star' for v in ind_nopara])
    src_tbl['Type'] = src_type

    # Generate new columns for each filter
    for f in filts_all:

        coldata = []
        for i in range(len(gaia_tbl)):
            # Select spectrum for given object
            key = 'flat' if src_tbl['Type'][i]=='galaxy' else int(teff[i])
            # Offset filter 0-mag value
            bp_mag = bp_sp_dict[f][key] + gband[i] 
            coldata.append(bp_mag)
            
        # Round to the nearest milli-mag
        coldata = np.round(coldata, decimals=3)

        # Add filter column to table
        src_tbl.add_column(coldata, name=f)
        src_tbl[f].unit = 'mag'
        
    return src_tbl


def make_simbad_source_table(coords, remove_cen_star=True, radius=6*u.arcmin,
    spt_default='G2V', dist_crossmatch=0.1):

    from astroquery.simbad import Simbad
    from astropy.table import Table

    from .. import stellar_spectrum, read_filter
    from ..nrc_utils import S, bp_2mass

    sim_obj = Simbad()
    
    sim_obj.reset_votable_fields()
    sim_obj.remove_votable_fields('coordinates')
    sim_obj.add_votable_fields('ra(d;ICRS)', 'dec(d;ICRS)', 'flux(K)', 'sp', 'otype(V)', 'distance_result')
    
    # coords = targ_dict['HR8799']['sky_coords']
    sim_tbl = sim_obj.query_region(coords, radius=radius)
    
    # Remove non-stellar sources and 
    ind_star = np.array(['Star' in val for val in sim_tbl['OTYPE_V'].data.data])
    ind_nokband = sim_tbl['FLUX_K'].data.mask
    if remove_cen_star:
        ind_dist  = (sim_tbl['DISTANCE_RESULT'] < dist_crossmatch).data
        # ind_remove = np.where((~ind_star) | ind_nok | ind_dist)
        ind_remove = np.where(ind_nokband | ind_dist)
    else:
        # ind_remove = np.where((~ind_star) | ind_nok)
        ind_remove = np.where(ind_nokband)
    sim_tbl.remove_rows(ind_remove)

    ra_name  = sim_tbl.colnames[1]
    dec_name = sim_tbl.colnames[2]
        
    # Create new source table
    index = np.arange(len(sim_tbl)) + 1
    ra = sim_tbl[ra_name]
    dec = sim_tbl[dec_name]
    kmag = sim_tbl['FLUX_K']
    src_tbl = Table([index, ra, dec, kmag], names=('index', 'ra', 'dec', 'K-Band'))
    # Add distances
    src_tbl.add_column(sim_tbl['DISTANCE_RESULT'].data, name='dist')
    src_tbl['dist'].unit = 'arcsec'
    
    # Get effective temperature for each object rounded to the nearest 100K
    sptype = sim_tbl['SP_TYPE'].data.data
    # Assume solar temperature for null data
    sptype[sim_tbl['SP_TYPE'].data.mask] = spt_default
    sptype[sim_tbl['SP_TYPE'].data==''] = spt_default
    for i in range(len(sptype)):
        spt = sptype[i]
        if (spt=='') or (len(spt)<2) or (spt[0] not in 'OBAFGKM'):
            sptype[i] = spt_default

    # Create stellar spectra of each unique temperature
    sp_dict = {}
    for spt in np.unique(sptype):
        if (spt==''):
            continue
        spt2 = spt.split('+')[0]
        # print(spt, spt2)
        spt2 = spt2 + 'V' if len(spt2)==2 else spt2
        sp_dict[spt2] = stellar_spectrum(spt2)
        
    # SW Filters
    filts_sw = [
        'F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F150W2',
        'F162M', 'F164N', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N'
    ]
    # LW Filters
    filts_lw = [
        'F250M', 'F277W',  'F300M','F322W2', 'F323N', 'F335M', 'F356W', 'F360M', 
        'F405N', 'F410M', 'F430M', 'F444W', 'F460M', 'F466N', 'F470N', 'F480M'
    ]
    filts_all = filts_sw + filts_lw

    # Read in all bandpasses
    bp_dict = {}
    for f in filts_sw:
        bp_dict[f] = read_filter(f)
    for f in filts_lw:
        bp_dict[f] = read_filter(f)

    # Cycle through all filters and sources to get 0-mag values
    bp_k = bp_2mass('k')
    bp_sp_dict = {}
    for f in tqdm(filts_all, leave=False, desc='Filters'):
        bp = bp_dict[f]
        d = {}
        for k in sp_dict.keys():
            sp = sp_dict[k]
            sp = sp.renorm(0, 'vegamag', bp_k)
            obs = Observation(sp, bp, binset=bp.waveset)
            d[k] = obs.effstim('vegamag')
        bp_sp_dict[f] = d

    src_tbl['SpType'] = sptype
    for row in src_tbl:
        spt = 'G2V' if row['SpType']=='' else row['SpType']
        spt = spt.split('+')[0]
        spt = spt + 'V' if len(spt)==2 else spt
        row['SpType'] = spt

    # Generate new columns for each filter
    for f in filts_all:
        bp = bp_dict[f]

        coldata = []
        for row in src_tbl:
            spt = row['SpType']
            # sptype = 'G2V' if row['SP_TYPE']=='' else row['SP_TYPE']
            # sptype = sptype.split('+')[0]
            # sptype = sptype + 'V' if len(sptype)==2 else sptype
            bp_mag = bp_sp_dict[f][spt] + row['K-Band']
            coldata.append(bp_mag)

        # Round to the nearest milli-mag
        coldata = np.round(coldata, decimals=3)

        # Add filter column to table
        src_tbl.add_column(coldata, name=f)
        src_tbl[f].unit = 'mag'
        
    return src_tbl


def gen_wfe_drift(obs_input, case='BOL', iec_period=300, slew_init=10, rand_seed=None,
                  t0_offset=False, plot=False, figname=None):
    """ Create WFE drift information over time

    Parameters
    ==========
    obs_input : DMS_input class
        Class to generate a series of observation dictionaries in order to 
        build DMS-like files. Loads APT files to generate the necessary 
        observation information.
    case : string
        Either "BOL" for current best estimate at beginning of life, or
        "EOL" for more conservative prediction at end of life.
    iec_period : float
        IEC heater switching period in seconds.
    slew_init : float
        Assumed slew difference relative to previous program (degress of pitch angle).
    rand_seed : None or int
        Seed value to initialize random number generator to obtain
        repeatable values.
    t0_offset : bool
        Shift delta WFE drift values to 0 at time t=0 (beginning of program)? 
        If set to False, then relative drift values correspond to beginning
        of the previous randomly-generated program.
    plot : bool
        Create a plot of the slew angles and associated RMS drift components.
    figname : string
        Output name (path) to save plot figure.
    """

    import webbpsf
    from webbpsf_ext.opds import OTE_WFE_Drift_Model
    from webbpsf.utils import get_webbpsf_data_path
    from ..opds import opd_default, opd_dir, pupil_file

    def plot_wfe(figname=None):

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2,1, figsize=(12,8), sharex=True)
        axes = axes.flatten()

        tunit = 'hr'
        tvals = t_all.to(tunit)

        ax = axes[0]
        ax.plot(tvals.value, slew_angles, marker='.')
        ax.set_ylabel('Pitch Angle (deg)')
        k0 = list(obs_input.program_info.keys())[0]
        pid = int(obs_input.program_info[k0]['obs_id_info'][0]['program_number'])
        ax.set_title(f'OPD w/ Initial Slew of {slew_init:.0f} deg (PID {pid}, {case})')

        ylims = ax.get_ylim()
        dy = ylims[1]-ylims[0]

        for i, texp in enumerate(exp_start):
            t = (texp * u.s).to(tunit).value
            label = 'NRC Exps' if i==0 else None
            ax.plot([t,t], [ylims[0],ylims[0]+0.25*dy] , color='k', ls=':', lw=1, alpha=0.5, label=label)
        for i, tvisit in enumerate(visit_start):
            t = (tvisit * u.s).to(tunit).value
            label = 'Visit Start' if i==0 else None
            ax.plot([t,t], [ylims[0],ylims[0]+0.5*dy] , color='C2', ls='--', lw=1.5, label=label)
        for i, tslew in enumerate(slew_start):
            t = (tslew * u.s).to(tunit).value
            label = 'Slew Start' if i==0 else None
            ax.plot([t,t], [ylims[0],ylims[0]+0.75*dy] , color='C1', ls='--', lw=2, label=label)
            
        ax.set_ylim(ylims)
        ax.legend()

        # Plot WFE drift components
        # Offset relative to first visit in sequence
        wfe_dict2 = wfe_dict.copy()
        keys = list(wfe_dict.keys())#[0:2]
        for k in ['frill', 'thermal']:
            wfe_val = wfe_dict[k]
            trel = (visit_start[0]*u.s).to(tunit).value
            wfe_dict2[k] = wfe_val - np.interp(trel, tvals.value, wfe_val)
        # Update total RMS
        wfe_dict2['total'] = np.sqrt(wfe_dict2['frill']**2 + wfe_dict2['thermal']**2 + wfe_dict2['iec']**2)

        ax = axes[1]
        for k in keys:
            # Don't plot time vs time!!
            if 'time' in k:
                continue
            lw = 2 if 'total' in k else 1
            alpha = 0.5 if 'total' in k else 1
            ax.plot(tvals.value, wfe_dict2[k], label=k, lw=lw, alpha=alpha)
        ax.set_xlabel(f'Time ({tunit})')
            
        for ax in axes[1:]:
            ax.legend()
            ax.set_ylabel('$\Delta$WFE (nm RMS)')

        # xlim = [-0.1,ax.get_xlim()[1]]
        xlim = [-0.1,tvals.value.max()]
        ax.set_xlim(xlim)
        ylim = ax.get_ylim()
        if np.abs(ylim[0])>ylim[1]:
            ylim = np.array([-1,1]) * np.max(np.abs(ylim))
        ax.set_ylim(ylim)
            
        fig.tight_layout()
        if figname is not None:
            fig.savefig(figname)
            print(f'Saveing: {figname}')
    
    # Get total time for program
    temp, _, _, _ = obs_input.gen_pitch_array(nvals=1000)
    total_time = temp.max()

    # Create a series of time values to evolve over
    dt = iec_period / 2 # Timing sample
    tarr = np.arange(0, total_time, dt) # seconds

    # Required size
    nvals = len(tarr)

    # Create an initial fake scenario to occur before ours
    res = obs_input.gen_pitch_array(nvals=nvals, pitch_init=None)
    tprior, pitch_prior, _, _ = res
    tprior = tprior[::5] - tprior[-1]
    pitch_prior = pitch_prior[::5][::-1] - slew_init
    pitch_prior[0] -= slew_init

    # Create desired observations
    res = obs_input.gen_pitch_array(nvals=nvals, pitch_init=pitch_prior[-1])
    tarr, pitch_arr, slew_start, visit_start = res

    # Change anything that's outside of pitch bounds
    pitch_prior[pitch_prior<-5] = -5
    pitch_prior[pitch_prior>45] = 45
    pitch_arr[pitch_arr<-5] = -5
    pitch_arr[pitch_arr>45] = 45

    # All NIRCam exposure start times
    exp_start = []
    for k in obs_input.program_info.keys():
        d = obs_input.program_info[k]
        exp_start.append(d['exp_start_times'])
    exp_start = np.concatenate(exp_start)

    # Data directories
    # OPD directory defined above
    webbpsf_path = get_webbpsf_data_path()
    pupil_dir    = webbpsf_path

    # Pupil and OPD file path names
    opd_file, opd_index = opd_default
    pupil_path = os.path.join(pupil_dir, pupil_file)
    opd_path   = os.path.join(opd_dir, opd_file)

    # Initiate OTE drift class
    name = "Modified OPD from " + str(opd_file)
    ote = OTE_WFE_Drift_Model(name=name, opd=opd_path, opd_index=opd_index, 
                              transmission=pupil_path)

    # Generate delta OPDs for each time step
    # Also outputs a dictionary of each component's WFE drift value (nm RMS)
    t_all = np.concatenate([tprior, tarr]) * u.s
    slew_angles = np.concatenate([pitch_prior, pitch_arr])

    wfe_dict_all = ote.evolve_dopd(t_all, slew_angles, case=case, 
                                   return_dopd_fin=False, random_seed=rand_seed)

    tunit = 'hr'
    tvals = t_all.to(tunit)

    # Offset relative to first visit in sequence
    wfe_dict = wfe_dict_all.copy()
    if t0_offset:
        for k in ['frill', 'thermal']:
            wfe_val = wfe_dict[k]
            trel = (visit_start[0]*u.s).to(tunit).value
            wfe_dict[k] = wfe_val - np.interp(trel, tvals.value, wfe_val)
        # Update total RMS
        wfe_dict['total'] = np.sqrt(wfe_dict['frill']**2 + wfe_dict['thermal']**2 + wfe_dict['iec']**2)

    wfe_dict['time_sec'] = tvals.to('s').value
    if plot:
        plot_wfe(figname=figname)

    return wfe_dict


def gen_col_noise(ramp_column_varations, prob_bad, nz=108, nx=2048, rand_seed=None):
    """ Generate RTN Column Noise

    This function takes the random telegraph noise templates derived from 
    CV3 data and generates a random noise set to add to an dark ramp sim.
    These column variations likely come from RTN in column-specifc FETs 
    jumping between two discrete states, possibly within the detector column bus.

    This function randomly draws a number of template column variation ramps,
    then randomly assigns them to different columns in `super_dark_ramp`.
    The nubmer of columns (and whether or not a column is assigned a random
    variation) is based on the `prob_bad` variable.
    """

    rng = np.random.default_rng(rand_seed)

    # Number of samples in ramp templates
    nz0 = ramp_column_varations.shape[0]

    # if nz>nz0:
    #     raise ValueError('nz should not be greater than {} frames'.format(nz0))

    # Variable to store column offsets for all NX columns
    cols_all_add = np.zeros([nz0,nx])

    # Mask of random columns to include ramp excursions
    # Create set of random values between 0 and 1
    # Mark those with values less than prob_bad for 
    # adding some random empirically measured column
    xmask_random = rng.random(size=nx) <= prob_bad
    nbad_random = len(xmask_random[xmask_random])

    # Grab some random columns from the stored templates
    ntemplates = ramp_column_varations.shape[1]
    ind_rand = rng.integers(0, high=ntemplates, size=ntemplates)
    # Make sure we get unique values (no repeats)
    _, ind_rand = np.unique(ind_rand, return_index=True)
    ind_rand = ind_rand[0:nbad_random]
    # If we don't have enough random columns, append more
    # This should be very unlikely to occur, but just in case...
    if len(ind_rand) < nbad_random:
        ndiff = nbad_random - len(ind_rand)
        ind_rand = np.append(ind_rand, rng.integers(0, high=ntemplates, size=ndiff))
        
    # Select the set of random column variation templates
    cols_rand = ramp_column_varations[:,ind_rand]

    # Add a random phase shift to each of those template column
    tshifts = rng.integers(0, high=nz0, size=nbad_random)
    for i in range(nbad_random):
        cols_rand[:,i] = np.roll(cols_rand[:,i], tshifts[i])

    # Add to columns variable
    cols_all_add[:, xmask_random] = cols_rand

    # Reshape to (nz0,1,nx) to easily add to a ramp of size (nz,ny,nx)
    cols_all_add = cols_all_add.reshape([nz0,1,-1])

    # Only return number of requested frames
    if nz>nz0:
        cols_all_add2 = np.zeros([nz,1,nx])
        cols_all_add2[0:nz0] = cols_all_add
        return cols_all_add2
    else:
        return cols_all_add[0:nz, :, :]

def add_col_noise(data_in, ramp_column_varations, prob_bad, rand_seed=None):
    """ Add RTN Column Noise
    
    This function takes the random telegraph noise templates derived from 
    CV3 data and adds it to an idealized dark ramp. These column variations 
    likely come from noise in column-specifc FETs jumping between two discrete 
    states, possibly within the detector column bus.

    This function randomly draws a number of template column variation ramps,
    then randomly assigns them to different columns in `super_dark_ramp`.
    The nubmer of columns (and whether or not a column is assigned a random
    variation) is based on the `prob_bad` variable.
    
    Parameters
    ==========
    
    data_in : ndarray
        Idealized ramp of size (nz,ny,nx)
    ramp_column_variations : ndarray
        The column-average ramp variations of size (nz,nx). 
        These are added to a given columnn.
    prob_bad : float
        Probability that a given column is subject to these column variations.
    """
    
    nz, ny, nx = data_in.shape
    
    cols_all_add = gen_col_noise(ramp_column_varations, prob_bad, nz=nz, nx=nx, 
                                 rand_seed=rand_seed)
    # Add to dark ramp
    return data_in + cols_all_add

def gen_ramp_biases(ref_dict, nchan=None, data_shape=(2,2048,2048), 
                    include_refinst=True, ref_border=[4,4,4,4], rand_seed=None):
    """ Generate a ramp of bias offsets

    Parameters
    ==========
    ref_dict : dict
        Dictionary of reference behaviors.
    nchan : int
        Specify number of output channels. If not set, then will
        automatically determine from `ref_dict`. This allows us
        to set nchan=1 for Window Mode while using the first channel
        info provided in `ref_dict`.
    data_shape : array like
        Shape of output (nz,ny,nx) 
    include_refinst : bool
        Include instabilities in the offsets?
    ref_border: list
        Number of references pixels [lower, upper, left, right]
    """

    rng = np.random.default_rng(rand_seed)

    
    if nchan is None:
        nchan = len(ref_dict['amp_offset_mean'])

    cube = np.zeros(data_shape)
    nz, ny, nx = data_shape
    chsize = int(nx/nchan)
    
    ######################
    # Add overall bias
    # TODO: Add temperature dependence
    bias_off = ref_dict['master_bias_mean'] + rng.normal(scale=ref_dict['master_bias_std'])
    cube += bias_off

    # Add amplifier offsets
    # These correlate to bias offset
    cf = ref_dict['master_amp_cf']
    amp_off = jl_poly(bias_off, cf) + rng.normal(scale=ref_dict['amp_offset_std'])

    for ch in range(nchan):
        cube[:,:,ch*chsize:(ch+1)*chsize] += amp_off[ch]
    
    # Include frame-to-frame bias variation
    ######################
    bias_off_f2f = rng.normal(scale=ref_dict['master_bias_f2f'], size=nz)
    amp_off_f2f = rng.normal(scale=ref_dict['amp_offset_f2f'][0:nchan], size=(nz,nchan))

    for i, im in enumerate(cube):
        im += bias_off_f2f[i]
        for ch in range(nchan):
            im[:,ch*chsize:(ch+1)*chsize] += amp_off_f2f[i,ch]
    
    # Add some reference pixel instability relative to active pixels
    ######################

    # Mask of all reference pixels in detector coordiantes
    # Active and reference pixel masks
    lower, upper, left, right = ref_border
    mask_ref = np.zeros([ny,nx], dtype='bool')
    if lower>0: mask_ref[0:lower,:] = True
    if upper>0: mask_ref[-upper:,:] = True
    if left>0:  mask_ref[:,0:left] = True
    if right>0: mask_ref[:,-right:] = True

    rseed_nchan = rng.integers(0, 2**32-1, size=nchan)
    if include_refinst:
        for ch in range(nchan):
            mask_ch = np.zeros([ny,nx]).astype('bool')
            mask_ch[:,ch*chsize:(ch+1)*chsize] = True

            std = ref_dict['amp_ref_inst_f2f'][ch]
            ref_noise = std * pink_noise(nz, rand_seed=rseed_nchan[ch])
            cube[:, mask_ref & mask_ch] += ref_noise.reshape([-1,1])

    # Set even/odd offsets
    ######################
    mask_even = np.zeros([ny,nx]).astype('bool')
    mask_even[:,0::2] = True

    mask_odd = np.zeros([ny,nx]).astype('bool')
    mask_odd[:,1::2] = True

    for ch in range(nchan):
        mask_ch = np.zeros([ny,nx]).astype('bool')
        mask_ch[:,ch*chsize:(ch+1)*chsize] = True

        cube[:, mask_even & mask_ch] += ref_dict['amp_even_col_offset'][ch]
        cube[:, mask_odd & mask_ch]  += ref_dict['amp_odd_col_offset'][ch]
    
    return cube



def fft_noise(pow_spec, nstep_out=None, nseq=1, fmin=None, f=None, 
              pad_mode='edge', rand_seed=None, use_mkl=False, **kwargs):
    """ Random Noise from Power Spectrum
    
    Returns a noised array where the instrinsic distribution
    follows that of the input power spectrum. The output has an
    instrinsic standard deviation scaled to 1.0.
    
    Parameters
    ==========
    pow_spec : ndarray
        Input power spectrum from which to generate noise distribution.
    nstep_out : int
        Desired size of the output noise array. If smaller than `pow_spec`
        then it just truncates the results to the appropriate size.
        If larger, then pow_spec gets padded by the specified `pad_mode`.
    nseq : int
        How many independent random time series do we want to produce 
        simultaneoulsy? If `nseq>1`, then returns an array of size
        (nseq, nstep_out).
    fmin : float or None
        Low-frequency cutoff. Power spectrum values below this cut-off
        point get set equal to the power spectrum value at fmin.
    f : ndarray or None
        An array the same size as pow_spec and is only used when fmin
        is set. If set to None, then `f = np.fft.rfftfreq(n_ifft)`
        where `n_ifft` is the size of the result of `rifft(pow_spec)`
        assuming a delta time of unity.
    pad_mode : str or function
        One of the following string values or a user supplied function.
        Default is 'edge'.

        'constant' (default)
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.
        'linear_ramp'
            Pads with the linear ramp between end_value and the
            array edge value.
        'maximum'
            Pads with the maximum value of all or part of the
            vector along each axis.
        'mean'
            Pads with the mean value of all or part of the
            vector along each axis.
        'median'
            Pads with the median value of all or part of the
            vector along each axis.
        'minimum'
            Pads with the minimum value of all or part of the
            vector along each axis.
        'reflect'
            Pads with the reflection of the vector mirrored on
            the first and last values of the vector along each
            axis.
        'symmetric'
            Pads with the reflection of the vector mirrored
            along the edge of the array.
        'wrap'
            Pads with the wrap of the vector along the axis.
            The first values are used to pad the end and the
            end values are used to pad the beginning.

    """
    
    from ..nrc_utils import do_fft

    rng = np.random.default_rng(rand_seed)

    if nstep_out is None:
        nstep_out = 2 * (len(pow_spec) - 1)
        
    nstep = nstep_out
    # For large data sets, this is faster than going to the next power of 2
    nstep2 = int(2**np.ceil(np.log2(nstep-1))) + 1

    lin_spec = np.sqrt(pow_spec) 
    # Set cuf-off frequency
    if (fmin is not None) and (fmin>0):
        n_ifft = 2 * (len(lin_spec) - 1)
        f = np.fft.rfftfreq(n_ifft) if f is None else f
        fstep = f[1] - f[0]
        fmin = np.max([fmin, fstep])
        ix  = np.sum(f < fmin)   # Index of the cutoff
        if ix > 1 and ix < len(f):
            lin_spec[:ix] = lin_spec[ix]
    
    # Padding to add lower frequencies
    pad = nstep2-len(lin_spec)
    pad = 0 if pad <0 else pad
    if pad>0:
        lin_spec = np.pad(lin_spec, (pad,0), mode=pad_mode, **kwargs)
    
    # Build scaling factors for all frequencies
    
    # Calculate theoretical output standard deviation from scaling
    w = lin_spec[1:-1]
    n_ifft = 2 * (len(lin_spec) - 1)
    w_last = lin_spec[-1] * (1 + (n_ifft % 2)) / 2. # correct f = +-0.5
    the_std = 2 * np.sqrt(np.sum(w**2) + w_last**2) / n_ifft
    
    # Generate scaled random power + phase
    # For large numbers, faster to gen with scale=1, then multiply
    # sr = rng.normal(size=len(lin_spec)) * lin_spec
    # si = rng.normal(size=len(lin_spec)) * lin_spec
    sr = rng.normal(size=[nseq, len(lin_spec)]) * lin_spec.reshape([1,-1])
    si = rng.normal(size=[nseq, len(lin_spec)]) * lin_spec.reshape([1,-1])
    del lin_spec

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if (nstep2 % 2) == 0: 
        si[:,-1] = 0

    # Regardless of signal length, the DC component must be real
    si[:,0] = 0

    # Combine power + corrected phase to Fourier components
    thefft  = sr + 1J * si
    del sr, si

    # Apply the pinkening filter.
    result = do_fft(thefft, real=True, inverse=True, use_mkl=use_mkl)
    del thefft
    
    # Keep requested nstep and scale to unit variance
    result = result[:, :nstep_out] / the_std
    return result.squeeze()


def pink_noise(nstep_out, pow_spec=None, f=None, fmin=None, alpha=-1, **kwargs):
    """ Generate random pink noise

    Parameters
    ==========
    nstep_out : int
        Desired size of the output noise array. If smaller than `pow_spec`
        then it just truncates the results to the appropriate size.
        If larger, then pow_spec gets padded by the specified `pad_mode`.
    pow_spec : ndarray
        Option to input the power spectrum instead of regenerating it
        every time. Make sure this was generated with powers of 2 for
        faster processing.
    f : ndarray or None
        An array the same size as pow_spec. If set to None, then 
        will create an array of appropriate size assuming a delta
        time of unity.
    fmin : float or None
        Low-frequency cutoff. Power spectrum values below this cut-off
        point get set equal to the power spectrum value at fmin.
    alpha : float
        Power spectrum index to generate if `pow_spec` is not specified directly.

    Keyword Args
    ============
    pad_mode : str or function
        One of the following string values or a user supplied function.
        Default is 'edge'.

        'constant' (default)
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.
        'linear_ramp'
            Pads with the linear ramp between end_value and the
            array edge value.
        'maximum'
            Pads with the maximum value of all or part of the
            vector along each axis.
        'mean'
            Pads with the mean value of all or part of the
            vector along each axis.
        'median'
            Pads with the median value of all or part of the
            vector along each axis.
        'minimum'
            Pads with the minimum value of all or part of the
            vector along each axis.
        'reflect'
            Pads with the reflection of the vector mirrored on
            the first and last values of the vector along each
            axis.
        'symmetric'
            Pads with the reflection of the vector mirrored
            along the edge of the array.
        'wrap'
            Pads with the wrap of the vector along the axis.
            The first values are used to pad the end and the
            end values are used to pad the beginning.
    rand_seed : None or int
        Seed value to initialize random number generator to obtain
        repeatable values.

    """
    
    if ((fmin is not None) and (fmin>0)) or (pow_spec is None):
        # Set up to a power of 2 for faster processing
        nstep2 = 2 * int(2**np.ceil(np.log2(nstep_out)))
        f = np.fft.rfftfreq(nstep2)
        f[0] = f[1] # First element should not be 0
    if pow_spec is None:
        pow_spec = f**alpha
        pow_spec[0] = 0.
        
    if f is not None:
        assert len(f)==len(pow_spec), "f and pow_spec must be same size"
        
    assert len(pow_spec)>=nstep_out, "Power spectrum must be greater than nstep_out"
            
    res = fft_noise(pow_spec, nstep_out=nstep_out, fmin=fmin, f=f, **kwargs)
        
    return res

def sim_noise_data(det, rd_noise=[5,5,5,5], u_pink=[1,1,1,1], c_pink=3,
    acn=0, pow_spec_corr=None, corr_scales=None, fcorr_lim=[1,10],
    ref_ratio=0.8, rand_seed=None, verbose=True, **kwargs):
    
    """ Simulate Noise Ramp
    
    Simulate the noise components of a ramp, including white noise as well as
    1/f (pink) noise components that are uncorrelated and correlated between
    amplifier channels.

    Parameters
    ==========
    det : `det_timing` class
        Class holding detector operations information. See `detops.det_timing`
        for generic class, or `pynrc_core.DetectorOps` for NIRCam specific
        timing.
    rd_noise : array like or float or None
        Array of white noise values (std dev per frame) for each output channel, 
        or a single value. If an array, must match the number amplifier values 
        specified in `det.nchan`.
    u_pink : array like or float or None
        Array of uncorrelated pink noise (std dev per frame) for each output channel, 
        or a single value. If an array, must match the number amplifier values 
        specified in `det.nchan`.
    c_pink : float or None
        Standard deviation of the pink noise correlated between channels.
    pow_spec_corr : ndarray
        Option to input a custom power spectrum for the correlated noise.
    corr_scales : array like
        Instead of `pow_spec_corr`, input the scale factors of the two 1/f
        components [low freq, highfreq]).
    fcorr_lim : array like
        Low- and high- frequency cut-off points for `corr_scales` factors.
        The first element of `corr_scales` is applied to those frequencies
        below `fcorr_lim[0]`, while the second element corresponds to frequencies
        above `fcorr_lim[1]`.

    """
    
    from pynrc.reduce.calib import fit_corr_powspec, broken_pink_powspec
    import time

    ################################
    # Initialize a random number generator
    rng = np.random.default_rng(rand_seed)
    # Create individual random number generators for each noise element
    # (eg., read noise, uncorrelated pink noise, correlated pink noise)
    # This allows for repeatable noise values for each element as long as
    # the same rand_seed is passed, even if others elemented are excluded 
    # on different runs.
    # If rand_seed=None, then everthing here will be random.
    rng_keys = ['rd_noise', 'u_pink', 'c_pink', 'acn']
    rng_dict = {}
    for k in rng_keys:
        # Create a random seed to pass to individual RNGs
        rseed = rng.integers(0, 2**32-1)
        rng_dict[k] = np.random.default_rng(rseed)

    nchan = det.nout
    nx = det.xpix
    ny = det.ypix
    chsize = det.chsize

    # Number of total frames up the ramp (including drops)
    ma     = det.multiaccum
    nd1    = ma.nd1
    nd2    = ma.nd2
    nf     = ma.nf
    ngroup = ma.ngroup
    nz     = nd1 + ngroup*nf + (ngroup-1)*nd2

    nroh = det._line_overhead
    nfoh = det._extra_lines

    same_scan_direction = det.same_scan_direction
    reverse_scan_direction = det.reverse_scan_direction
    
    result = np.zeros([nz,ny,nx])
                            
    # Make white read noise. This is the same for all pixels.
    if rd_noise is not None:
        rng = rng_dict['rd_noise']
        # We want rd_noise to be an array or list
        if isinstance(rd_noise, (np.ndarray,list)):
            temp = np.asarray(rd_noise)
            if temp.size != nchan:
                _log.error('Number of elements in rd_noise not equal to n_out')
                return
        else: # Single value as opposed to an array or list
            rd_noise = np.ones(nchan) * rd_noise
    
        w = det.ref_info
        rr = ref_ratio #reference_pixel_noise_ratio 
        
        if np.any(rd_noise):
            if verbose:
                _log.info('Generating read noise...')

            # Go frame-by-frame
            for z in np.arange(nz):
                here = np.zeros((ny,nx))

                # First assume no ref pixels and just add in random noise
                for ch in np.arange(nchan):
                    x1 = ch * chsize
                    x2 = x1 + chsize
                    here[:,x1:x2] = rng.normal(scale=rd_noise[ch], size=(ny,chsize))

                # If there are reference pixels, overwrite with appropriate noise values
                # Noisy reference pixels for each side of detector
                rd_ref = rr * np.mean(rd_noise)
                if w[0] > 0: # lower
                    here[:w[0],:] = rng.normal(scale=rd_ref, size=(w[0],nx))
                if w[1] > 0: # upper
                    here[-w[1]:,:] = rng.normal(scale=rd_ref, size=(w[1],nx))
                if w[2] > 0: # left
                    here[:,:w[2]] = rng.normal(scale=rd_ref, size=(ny,w[2]))
                if w[3] > 0: # right
                    here[:,-w[3]:] = rng.normal(scale=rd_ref, size=(ny,w[3]))

                # Add the noise in to the result
                result[z,:,:] += here

                
    # Finish if no 1/f noise specified
    if (c_pink is None) and (u_pink is None) and (acn is None):
        return result
                
    #################################
    # 1/f noise
        
    ch_poh = chsize + nroh
    ny_poh = ny + nfoh

    # Compute the number of time steps per integration, per output
    nstep_frame = ch_poh * ny_poh
    nstep = nstep_frame * nz
    # Pad nsteps to a power of 2, which is much faster
    nstep2 = int(2**np.ceil(np.log2(nstep)))
    
    f2 = np.fft.rfftfreq(2*nstep2)
    f2[0] = f2[1] # First element should not be 0
    alpha = -1
    p_filter2 = f2**alpha
    p_filter2[0] = 0.

    # Add correlated pink noise.
    if (c_pink is not None) and (c_pink > 0):
        rng = rng_dict['c_pink']
        if verbose:
            _log.info('Adding correlated pink noise...')

        if corr_scales is not None:
            scales = np.array(corr_scales)
            fcut1, fcut2 = np.array(fcorr_lim) / det._pixel_rate
            pf = broken_pink_powspec(f2, scales, fcut1=fcut1, fcut2=fcut2, alpha=alpha)
            pf[0] = 0
        elif pow_spec_corr is not None:
            n_ifft = 2 * (len(pow_spec_corr) - 1)
            freq_corr = np.fft.rfftfreq(n_ifft, d=1/det._pixel_rate)
            freq_corr[0] = freq_corr[1]
            # Fit power spectrum and remake for f2
            scales = fit_corr_powspec(freq_corr, pow_spec_corr, **kwargs)
            fcut1, fcut2 = np.array(fcorr_lim) / det._pixel_rate
            pf = broken_pink_powspec(f2, scales, fcut1=fcut1, fcut2=fcut2, alpha=alpha)
            pf[0] = 0
        else:
            pf = p_filter2

        # Pass through a random seed to pink_noise function
        rseed = rng.integers(0, 2**32-1)
        tt = c_pink * pink_noise(nstep, pow_spec=pf, rand_seed=rseed)
        tt = tt.reshape([nz, ny_poh, ch_poh])[:,0:ny,0:chsize]
        # print('  Corr Pink Noise (input, output): {:.2f}, {:.2f}'
        #       .format(c_pink, np.std(tt)))

        for ch in np.arange(nchan):
            x1 = ch*chsize
            x2 = x1 + chsize
            
            if (same_scan_direction) or (np.mod(ch,2)==0):
                flip = True if reverse_scan_direction else False
            else:
                flip = False if reverse_scan_direction else True

            if flip: 
                result[:,:,x1:x2] += tt[:,:,::-1]
            else:
                result[:,:,x1:x2] += tt
        del tt

    # Add uncorrelated pink noise. Because this pink noise is stationary and
    # different for each output, we don't need to flip it (but why not?)
    if u_pink is not None:
        rng = rng_dict['u_pink']
        # We want u_pink to be an array or list
        if isinstance(u_pink, (np.ndarray,list)):
            temp = np.asarray(u_pink)
            if temp.size != nchan:
                _log.error('Number of elements in u_pink not equal to n_out')
                return
        else: # Single value as opposed to an array or list
            u_pink = np.ones(nchan) * u_pink

        # Only do the rest if any values are not 0
        if np.any(u_pink):
            if verbose:
                _log.info('Adding uncorrelated pink noise...')
            
            for ch in trange(nchan, desc='Uncorr 1/f', leave=False):
                x1 = ch*chsize
                x2 = x1 + chsize

                # Pass through a random seed to pink_noise function
                rseed = rng.integers(0, 2**32-1)
                tt = u_pink[ch] * pink_noise(nstep, pow_spec=p_filter2, rand_seed=rseed)
                tt = tt.reshape([nz, ny_poh, ch_poh])[:,0:ny,0:chsize]
                # print('  Ch{} Pink Noise (input, output): {:.2f}, {:.2f}'
                #       .format(ch, u_pink[ch], np.std(tt)))

                if (same_scan_direction) or (np.mod(ch,2)==0):
                    flip = True if reverse_scan_direction else False
                else:
                    flip = False if reverse_scan_direction else True

                if flip: 
                    result[:,:,x1:x2] += tt[:,:,::-1]
                else:
                    result[:,:,x1:x2] += tt

                del tt

    # Add ACN
    if (acn is not None) and (acn>0):
        rng = rng_dict['acn']
        if verbose:
            _log.info('Adding ACN noise...')

        facn = np.fft.rfftfreq(nstep2)
        facn[0] = facn[1] # First element should not be 0
        alpha = -2
        pf_acn = np.sqrt(facn**alpha)
        pf_acn[0] = 0.

        for ch in trange(nchan, desc='ACN', leave=False):
            x1 = ch*chsize
            x2 = x1 + chsize

            # Generate new pink noise for each even and odd vector.
            rseed_a, rseed_b = rng.integers(0, 2**32-1, size=2)
            a = acn * pink_noise(int(nstep/2), pow_spec=pf_acn, rand_seed=rseed_a)
            b = acn * pink_noise(int(nstep/2), pow_spec=pf_acn, rand_seed=rseed_b)
            # print('  Ch{} ACN Noise (input, [outa, outb]): {:.2f}, [{:.2f}, {:.2f}]'
            #         .format(ch, acn, np.std(a), np.std(b)))

            # Reformat into an image.
            tt = np.reshape(np.transpose(np.vstack((a, b))),
                            (nz, ny_poh, ch_poh))[:, 0:ny, 0:chsize]

            if (same_scan_direction) or (np.mod(ch,2)==0):
                flip = True if reverse_scan_direction else False
            else:
                flip = False if reverse_scan_direction else True

            if flip: 
                result[:,:,x1:x2] += tt[:,:,::-1]
            else:
                result[:,:,x1:x2] += tt

            del tt

    return result

def gen_dark_ramp(dark, out_shape, tf=10.73677, gain=1, ref_info=None,
                  avg_ramp=None, include_poisson=True, rand_seed=None, **kwargs):
    
    """
    Assumes a constant dark current rate, either in image form or single value.
    If gain is supplied, then input is assumed to be in DN/sec, otherwise e-/sec. 
    Output will be e-.

    Parameters
    ----------
    dark : ndarray or float
        Dark slope image or constant value. Assumed to be DN/sec. If gain=1,
        then also e-/sec. If this value is intended to be e-/sec, then simply
        set gain=1.
    out_shape : tuple, list, ndarray
        Desired shape of output ramp (nframes, ny, nx). If `dark` is an
        array, then dark.shape == out_shape[1:] == (ny,nx).
    tf : float
        Frame time in seconds
    gain : float
        Gain of detector in e-/sec. If specified to be other than 1, then we
        assume `dark` to be in units of DN/sec.
    avg_ramp : ndarray
        Time-dependent flux of average dark ramp.
    """

    rng = np.random.default_rng(rand_seed)

    nz, ny, nx = out_shape

    if avg_ramp is not None:
        assert len(avg_ramp)>=nz, "avg_ramp size must be >= to number of requested frames (out_shape[0])"
    
    # Count accumulation for a single frame (e-)
    dark_frame = np.ones([ny,nx]) * dark * tf * gain

    # Set negative values to median
    med = np.median(dark_frame)
    med = 0 if med<0 else med
    dark_frame[dark_frame<0] = med

    # Return an array of 0s if all dark current is 0
    if np.all(dark_frame==0):
        result = np.zeros(out_shape)
    else:
        # Add Poisson noise at each frame step
        if include_poisson:
            result = rng.poisson(lam=dark_frame, size=out_shape).astype('float')
        else:
            result = np.array([dark_frame for i in range(nz)])
        # Perform cumulative sum in place
        result = np.cumsum(result, axis=0, out=result)

    # Modulate "ideal" slope by emperical "average ramp" behavior
    if avg_ramp is not None:
        tarr = np.arange(1,nz+1)*tf
        avg_dark = np.median(dark)
        del_ramp = avg_ramp[0:nz] - avg_dark*tarr   # DN
        result += gain * del_ramp.reshape([-1,1,1]) # e-
            
    # Set reference pixels' dark current equal to 0
    if ref_info is not None:
        w = ref_info
        if w[0] > 0: # lower
            result[:,:w[0],:] = 0
        if w[1] > 0: # upper
            result[:,-w[1]:,:] = 0
        if w[2] > 0: # left
            result[:,:,:w[2]] = 0
        if w[3] > 0: # right
            result[:,:,-w[3]:] = 0
            
    # Return in units of e-
    return result 

def sim_dark_ramp(det, slope_image, ramp_avg_ch=None, ramp_avg_tf=10.73677, 
                  out_ADU=False, verbose=False, **kwargs):
    """
    Simulate a dark current ramp based on input det class and a
    super dark image. 
    
    By default, returns ramp in terms of e- using gain information 
    provide in `det` input. To return in terms of ADU, set 
    `out_ADU=True` (divides by gain).

    Parameters
    ----------
    det : Detector Class
        Desired detector class output
    slope_image : ndarray
        Input slope image (DN/sec).
        Can either be full frame or match `det` subarray. 
        Returns `det` subarray shape.
    
    Keyword Args
    ------------
    ramp_avg_ch : ndarray or None
        Time-dependent flux of average dark ramp for each amplifier channel
        for dark current simulations.
    ramp_avg_tf : float
        Delta time between between `ramp_avg_ch` points.
    out_ADU : bool
        Divide by gain to get value in ADU (float).
    include_poisson : bool
        Include Poisson noise from photoelectrons?
    verbose : bool
        Print some info messages.
    """
    
    nchan = det.nout
    nx, ny = (det.xpix, det.ypix)
    chsize = det.chsize
    tf = det.time_frame
    gain = det.gain
    ref_info = det.ref_info

    # Do we need to crop out subarray?
    if slope_image.shape[0]==ny:
        y1, y2 = (0, ny)
    else: # Will crop a subarray out of slope_image 
        y1 = det.y0
        y2 = int(y1 + ny)

    # Number of total frames up the ramp (including drops)
    ma     = det.multiaccum
    nd1    = ma.nd1
    nd2    = ma.nd2
    nf     = ma.nf
    ngroup = ma.ngroup
    nz     = nd1 + ngroup*nf + (ngroup-1)*nd2

    # Interpolate ramp_avg_ch onto tarr grid
    if (ramp_avg_ch is not None):
        tarr = np.arange(1,nz+1) * tf
        if tarr.max() < ramp_avg_tf:
            if verbose:
                msg = "Max ramp time {:.1f} is less than ramp_avg_tf. \
                    Not applying ramp_avg_ch.".format(tarr.max())
                _log.warning(msg)
            ramp_avg_ch = None
        else:
            # Insert 0 DN at t=0
            tvals = np.arange(0,ramp_avg_ch.shape[1]) * ramp_avg_tf
            ramp_avg_ch = np.insert(ramp_avg_ch, 0,0, axis=0)

            # Interpolate onto new time grid
            ramp_avg_ch_new = []
            for ramp_avg in ramp_avg_ch:
                avg_interp = np.interp(tarr, tvals, ramp_avg)
                ramp_avg_ch_new.append(avg_interp)
            ramp_avg_ch = np.array(ramp_avg_ch_new)

    if verbose:
        _log.info('Generating dark current ramp...')

    res = np.zeros([nz,ny,nx])
    for ch in np.arange(nchan):
        if nchan==1: # Subarray window case
            if slope_image.shape[1]==nx:
                x1, x2 = (0, nx)
            else: # Will crop a subarray out of slope_image 
                x1 = det.x0
                x2 = int(x1 + nx)
        else: # STRIPE or FULL frame
            x1 = ch*chsize 
            x2 = x1 + chsize

        slope_sub = slope_image[y1:y2,x1:x2]
        
        avg_ramp = None if ramp_avg_ch is None else ramp_avg_ch[ch]
        # Convert from DN to e-
        res[:,:,x1:x2] = gen_dark_ramp(slope_sub, (nz,ny,chsize), gain=gain, tf=tf,
                                       avg_ramp=avg_ramp, ref_info=None, **kwargs)

    if out_ADU:
        res /= gain
        
    # Set reference pixels' dark current equal to 0
    if ref_info is not None:
        w = ref_info
        if w[0] > 0: # lower
            res[:,:w[0],:] = 0
        if w[1] > 0: # upper
            res[:,-w[1]:,:] = 0
        if w[2] > 0: # left
            res[:,:,:w[2]] = 0
        if w[3] > 0: # right
            res[:,:,-w[3]:] = 0
        
    return res

def sim_image_ramp(det, im_slope, verbose=False, **kwargs):
    """
    Simulate an image ramp based on input det class and slope image.
    Uses the `sim_dark_ramp` function. By default, returns ramp in
    terms of e- using gain information provide in `det` input. To
    return in terms of ADU, set `out_ADU=True` (divides by gain).

    Parameters
    ----------
    det : Detector Class
        Desired detector class output
    im_slope : ndarray
        Input slope image (e-/sec). 
        *NOTE* - This is different than sim_dark_ramp, which assumed DN/sec.
        Can either be full frame or match `det` subarray. 
        Returns `det` subarray shape.
    include_poisson : bool
        Include Poisson noise from photons? Default: True.

    Keyword Args
    ------------
    out_ADU : bool
        Divides by gain to get output value in ADU (float).
    verbose : bool
        Print some messages.
    """
    if verbose:
        _log.info('Generating image acquisition ramp...')

    # Convert to DN/sec
    return sim_dark_ramp(det, im_slope/det.gain, ramp_avg_ch=None, verbose=False, **kwargs)


def apply_flat(cube, det, imflat_full):
    """ Apply flat field

    Includes pixel-to-pixel QE variations or instrument's optical
    flatfield (e.g., throughput variations across the field).

    Parameters
    ==========
    cube : ndarray
        Simulated ramp data in e-. These should be intrinsic
        flux values with Poisson noise, but prior to read noise,
        kTC, IPC, etc. Size (nz,ny,nx). In 'det' coords.
        Can either be full frame or match ``det`` subarray. 
        Returns ``det`` subarray shape.
    det : Detector Class
        Desired detector class output. Mainly just utilizes detector
        subarray position and shape.
    imflat_full : ndarray
        Full field image of flat field in det coords. Will get trimmed
        if necessary.
    """

    sh = cube.shape
    if len(sh)==2:
        ny, nx = sh
    else:
        _, ny, nx = sh

    # Need to crop in the event of subarrays
    x1, x2 = (det.x0, det.x0 + nx)
    y1, y2 = (det.y0, det.y0 + ny)

    imflat_sub = imflat_full[y1:y2, x1:x2]
    try:
        # Assume cube is already paired down to correct subarray size
        res = cube * imflat_sub
    except:
        # Assume cube is full frame
        cube_sub = cube[y1:y2, x1:x2] if len(sh)==2 else cube[:, y1:y2, x1:x2]
        res = cube_sub * imflat_sub

    return res


def add_cosmic_rays(data, scenario='SUNMAX', scale=1, tframe=10.73677, ref_info=[4,4,4,4], 
                    rand_seed=None):
    """ Add random cosmic rays to data cube"""

    import json

    # Load from JSON file
    file = 'cosmic_rays.json'
    file_path = os.path.join(conf.PYNRC_PATH, 'sim_params', file)
    with open(file_path, 'r') as fp:
        cr_dict = json.load(fp)
    # Only care about input scenario
    type_dict = cr_dict[scenario]

    sh = data.shape
    if len(sh)==2:
        ny, nx = sh
        nz = 1
        data = data.reshape([nz,ny,nx])
    else:
        nz, ny, nx = sh

    # Number of reference pixels [bottom, top, left, right]
    rb, rt, rl, rr = ref_info

    # Active detector area
    npix = (ny - rb - rt) * (nx - rl - rr)
    area_cm = npix * (18e-4)**2

    # For each ion type, add random events
    ion_keys = type_dict.keys()
    rng = np.random.default_rng(rand_seed)
    for k in ion_keys:
        rate   = type_dict[k]['rates'] # events / cm^2 / sec
        # How many event per frame on average?
        nhits = rate * area_cm * tframe * scale

        # Want to sample this distribution for each frame
        counts = np.asarray(type_dict[k]['counts'])

        for ii in range(nz):
            # Assume Poisson statistics on the hits
            nhits_i = rng.poisson(nhits)

            # Do a random sample
            counts_rand = rng.choice(counts, nhits_i)

            # Random position for each hit
            xpos_rand = rng.uniform(low=rl, high=nx-rr, size=nhits_i)
            ypos_rand = rng.uniform(low=rb, high=ny-rt, size=nhits_i)

            # Add CRs jump to current frame and all sequentional
            # separate into an integers and fractions
            intx = xpos_rand.astype(int)
            inty = ypos_rand.astype(int)
            fracx = xpos_rand - intx
            fracy = ypos_rand - inty
            
            # flip negative shift values
            ind = fracx < 0
            fracx[ind] += 1
            intx[ind] -= 1
            ind = fracy<0
            fracy[ind] += 1
            inty[ind] -= 1

            # Bilinear interpolation of all sources
            val1 = counts_rand * ((1-fracx)*(1-fracy))
            val2 = counts_rand * ((1-fracx)*fracy)
            val3 = counts_rand * ((1-fracy)*fracx)
            val4 = counts_rand * (fracx*fracy)

            # Add source-by-source in case of overlapped indices
            for i, (iy, ix) in enumerate(zip(inty,intx)):
                if (iy>=0) and (ix>=0):
                    data[ii:, iy,   ix]   += val1[i]
                if (iy+1<ny): 
                    data[ii:, iy+1, ix]   += val2[i]
                if (ix+1<nx): 
                    data[ii:, iy,   ix+1] += val3[i]
                if (iy+1<ny) and (ix+1<nx):
                    data[ii:, iy+1, ix+1] += val4[i]

    return data.reshape(sh)

def xtalk_image(frame, det, coeffs=None):
    """Create image of crosstalk signal

    Add amplifier crosstalk to each frame in data cube
    
    Parameters
    ----------
    frame : ndarray
        An image to calculate and add crosstalk to.
    det : :class:`~pynrc.DetectorOps`
        Detector class corresponding to data.
    coeffs : None or Table
        Table of coefficients corresponding to detector
        crosstalk behavior.

    """

    im_xtalk = np.zeros_like(frame)
    if det.nout<=1:
        # No crosstalk if only a single output channel
        return im_xtalk

    # Pixel shifts for each sub-channel
    subch_shift = {"0": 1, "1": -1, "2": 1, "3": -1}
    
    if coeffs is None:
        coeffs = det.xtalk()
    nchans = det.nout
    chsize = det.chsize
    
    ssd = det.same_scan_direction
    
    for ch in range(nchans):
        ix1, ix2 = int(ch*chsize), int((ch+1)*chsize)
        
        im_ch = frame[:,ix1:ix2]
        receivers = [i for i in range(nchans) if i != ch]
        
        for subch in receivers:
            jx1, jx2 = int(subch*chsize), int((subch+1)*chsize)

            # Reverse if amplifiers are not both even or both odd
            flip = False if ssd or (np.mod(ch-subch,2)==0) else True
            
            # Primary cross talk coefficients
            index = 'xt'+str(ch+1)+str(subch+1)
            corr_amp = im_ch[:,::-1] * coeffs[index] if flip else im_ch * coeffs[index]
            im_xtalk[:, jx1:jx2] += corr_amp
            
            # Post-pixel crosstalk coeffs require shift
            index = 'xt'+str(ch+1)+str(subch+1)+'post'
            corr_amp = im_ch[:,::-1] * coeffs[index] if flip else im_ch * coeffs[index]
            corr_amp = np.roll(corr_amp, subch_shift[str(subch)], axis=1)
            im_xtalk[:, jx1:jx2] += corr_amp

    return im_xtalk

def add_xtalk(data, det, coeffs=None):
    """Add amplifier crosstalk to each frame in data cube
    
    Parameters
    ----------
    data : ndarray
        2D or 3D data cube
    det : :class:`~pynrc.DetectorOps`
        Detector class corresponding to data.
    coeffs : None or Table
        Table of coefficients corresponding to detector
        crosstalk behavior.
    """

    sh = data.shape
    if len(sh)==2:
        ny, nx = sh
        nz = 1
        data = data.reshape([nz,ny,nx])
    else:
        nz, ny, nx = sh

    if coeffs is None:
        coeffs = det.xtalk()

    for frame in data:
        frame += xtalk_image(frame, det, coeffs=coeffs)

    return data.reshape(sh)


def simulate_detector_ramp(det, cal_obj, im_slope=None, cframe='sci', out_ADU=False, 
                           include_poisson=True, include_dark=True, include_bias=True, 
                           include_ktc=True, include_rn=True, apply_ipc=True, apply_ppc=True, 
                           include_cpink=True, include_upink=True, include_acn=True, 
                           include_refoffsets=True, include_refinst=True, 
                           include_colnoise=True, col_noise=None, amp_crosstalk=True,
                           add_crs=True, cr_model='SUNMAX', cr_scale=1, apply_flats=None, 
                           apply_nonlinearity=True, random_nonlin=False, latents=None,
                           rand_seed=None, return_zero_frame=None, return_full_ramp=False, 
                           prog_bar=True, super_bias=None, super_dark=None, **kwargs):
    
    """ Return a single simulated ramp
    
    Parameters
    ==========
    det : Detector Class
        Desired detector class output
    cal_obj: nircam_cal class
        NIRCam calibration class that holds the necessary calibration 
        info to simulate a ramp.
    im_slope : ndarray
        Input slope image of observed scene.
        Can either be full frame or match `det` subarray. 
        Returns `det` subarray shape.
    cframe : str
        Coordinate frame of input slope image, 'sci' or 'det'.
        Output will be in same coordinates.

    Keyword Args
    ============
    return_zero_frame : bool or None
        For DMS data, particularly readout patterns with averaged frames,
        this returns the very first raw read in the ramp.
    return_full_ramp : bool
        By default, we average groups and drop frames as specified in the
        `det` input. If this keyword is set to True, then return all raw
        frames within the ramp. The last set of `nd2` drop frames are omitted.
    out_ADU : bool
        If true, divide by gain and convert to 16-bit UINT.
    super_bias : ndarray or None
        Option to include a custom super bias image. If set to None, then
        grabs from ``cal_obj``. Can either be same shape as ``im_slope``
        or full frame version. Values assumed to be in units of DN.
        Assumed to be in 'det' coordinates.
    super_dark : ndarray or None
        Option to include a custom super dark image. If set to None, then
        grabs from ``cal_obj``. Can either be same shape as ``im_slope``
        or full frame version. Values assumed to be in units of DN.
        Assumed to be in 'det' coordinates.
    include_dark : bool
        Add dark current?
    include_bias : bool
        Add detector bias?
    include_poisson : bool
        Include photon noise?
    include_ktc : bool
        Add kTC noise?
    include_rn : bool
        Add readout noise per frame?
    include_cpink : bool
        Add correlated 1/f noise to all amplifiers?
    include_upink : bool
        Add uncorrelated 1/f noise to each amplifier?
    include_acn : bool
        Add alternating column noise?
    apply_ipc : bool
        Include interpixel capacitance?
    apply_ppc : bool
        Apply post-pixel coupling to linear analog signal?
    include_refoffsets : bool
        Include reference offsts between amplifiers and odd/even columns?
    include_refinst : bool
        Include reference/active pixel instabilities?
    include_colnoise : bool
        Add in column noise per integration?
    col_noise : ndarray or None
        Option to explicitly specifiy column noise distribution in
        order to shift by one for subsequent integrations
    amp_crosstalk : bool
        Crosstalk between amplifiers?
    add_crs : bool
        Add cosmic ray events? See Robberto et al 2010 (JWST-STScI-001928).
    cr_model: str
        Cosmic ray model to use: 'SUNMAX', 'SUNMIN', or 'FLARES'.
    cr_scale: float
        Scale factor for probabilities.
    apply_nonlinearity : bool
        Apply non-linearity? If False, then warning if out_ADU=True
    random_nonlin : bool
        Add randomness to the linearity coefficients?
    apply_flats: bool
        Apply sub-pixel QE variations (crosshatching and illumination)?
    latents : None or ndarray
        (TODO) Apply persistence from previous integration.
    prog_bar : bool
        Show a progress bar for this ramp generation?
    """
    
    import gc
    from ..reduce.calib import apply_nonlin

    ################################
    # Dark calibration properties
    dco = cal_obj

    # Super bias and darks
    if super_bias is None:
        super_bias = dco.super_bias_deconv # DN
    if super_dark is None:
        super_dark = dco.super_dark_deconv # DN/sec

    # IPC/PPC kernel information
    k_ipc = dco.kernel_ipc
    k_ppc = dco.kernel_ppc
    
    # Noise info
    cds_dict = dco.cds_act_dict
    keys = ['spat_det', 'spat_pink_corr', 'spat_pink_uncorr']
    cds_vals = [np.sqrt(np.mean(cds_dict[k]**2, axis=0)) for k in keys]
    # CDS Noise values
    # rd_noise_cds, c_pink_cds, u_pink_cds = cds_vals
    # Noise per frame
    rn, cp, up = cds_vals / np.sqrt(2)
    acn = 1

    # kTC Reset Noise
    ktc_noise = dco.ktc_noise

    # Power spectrum for correlated noise
    # freq = dco.pow_spec_dict['freq']
    scales = dco._pow_spec_dict['ps_corr_scale']
    # pcorr_fit = broken_pink_powspec(freq, scales)

    # Reference noise info
    ref_ratio = np.mean(dco.cds_ref_dict['spat_det'] / dco.cds_act_dict['spat_det'])
    
    ################################
    # Detector output configuration

    # Detector Gain
    gain = det.gain
    
    # Pixel readout
    nchan = det.nout
    ny, nx = (det.ypix, det.xpix)
    x1, x2 = (det.x0, det.x0 + nx)
    y1, y2 = (det.y0, det.y0 + ny)
    
    # Do we need to crop out subarray?
    if super_bias.shape!=(ny,nx):
        super_bias = super_bias[y1:y2,x1:x2]
    if super_dark.shape!=(ny,nx):
        super_dark = super_dark[y1:y2,x1:x2]
    
    # Number of total frames up the ramp (including drops)
    ma     = det.multiaccum
    nd1    = ma.nd1
    nd2    = ma.nd2
    nf     = ma.nf
    ngroup = ma.ngroup
    nz     = nd1 + ngroup*nf + (ngroup-1)*nd2

    tframe = det.time_frame

    # Scan direction info
    ssd = det.same_scan_direction
    rsd = det.reverse_scan_direction

    # Number of reference pixels (lower, upper, left, right)
    ref_info = det.ref_info
    
    ################################
    # Random seed information

    rng = np.random.default_rng(rand_seed)
    # Create random seed values to pass to each function.
    # This will all be determinate as long as `rand_seed` is defined.
    # If rand_seed=None, then everthing here will not be repeatable.
    rseed_keys = [
        'sim_dark_ramp', 'sim_image_ramp', 'add_cosmic_rays', 'apply_nonlin', 
        'ktc', 'sim_noise_data', 'gen_ramp_biases', 'gen_col_noise'
    ]
    rseed_dict = {}
    for k in rseed_keys:
        rseed_dict[k] = rng.integers(0, 2**32-1)

    ################################
    # Begin...
    ################################
    if prog_bar: 
        pbar = tqdm(total=14, leave=False)

    # Init data cube
    data = np.zeros([nz,ny,nx])

    # Work in detector coordinates
    if (im_slope is not None) and (cframe=='sci'):
        im_slope = sci_to_det(im_slope, det.detid)

    ####################
    # Create a super dark ramp (Units of e-)
    # Average shape of ramp
    if prog_bar: pbar.set_description("Dark Current")
    ramp_avg_ch = dco.dark_ramp_dict['ramp_avg_ch']
    # Create dark (adds Poisson noise)
    if include_dark:
        rseed = rseed_dict['sim_dark_ramp']
        data += sim_dark_ramp(det, super_dark, ramp_avg_ch=ramp_avg_ch, 
                              include_poisson=include_poisson, rand_seed=rseed, verbose=False)
    if prog_bar: pbar.update(1)

    ####################
    # Flat field QE variations (crosshatching)
    # TODO: Add sub-pixel QE varations 
    if prog_bar: pbar.set_description("Flat fields")
    if im_slope is not None:
        if apply_flats and (dco.pflats is not None):
            im_slope = apply_flat(im_slope, det, dco.pflats)
        if apply_flats and (dco.lflats is not None):
            im_slope = apply_flat(im_slope, det, dco.lflats)
    if prog_bar: pbar.update(1)

    ####################
    # Add on-sky source image
    if prog_bar: pbar.set_description("Sky Image")
    if im_slope is not None:
        rseed = rseed_dict['sim_image_ramp']
        data += sim_image_ramp(det, im_slope, include_poisson=include_poisson, 
                               rand_seed=rseed, verbose=False)
    if prog_bar: pbar.update(1)

    ####################
    # Add cosmic rays
    if prog_bar: pbar.set_description("Cosmic Rays")
    if add_crs:
        rseed = rseed_dict['add_cosmic_rays']
        data = add_cosmic_rays(data, scenario=cr_model, scale=cr_scale, tframe=tframe, 
                               ref_info=ref_info, rand_seed=rseed)
    if prog_bar: pbar.update(1)
    
    ####################
    # Add non-linearity
    if prog_bar: pbar.set_description("Non-Linearity")
    # The apply_nonlin function goes from e- to DN
    if apply_nonlinearity:
        rseed = rseed_dict['apply_nonlin']
        data = gain * apply_nonlin(data, det, dco.nonlinear_dict, 
                                   randomize=random_nonlin, rand_seed=rseed)
    # elif out_ADU:
    #     _log.warning("Assuming perfectly linear ramp, but convert to 16-bit UINT (out_ADU=True)")
    if prog_bar: pbar.update(1)

    ####################
    # TODO: Apply persistence/latent image
    # Latent signals are anomylous signals
    # Before or after non-linearity adjustments?
    if prog_bar: pbar.set_description("Persistence")
    if latents is not None:
        pass
    if prog_bar: pbar.update(1)

    ####################
    # Apply IPC 
    # TODO: Before or after non-linearity??
    if prog_bar: pbar.set_description("Include IPC")
    if apply_ipc:
        data = add_ipc(data, kernel=k_ipc)
    if prog_bar: pbar.update(1)

    ####################
    # Add kTC noise:
    if prog_bar: pbar.set_description("kTC Noise")
    if include_ktc:
        rng2 = np.random.default_rng(rseed_dict['ktc'])
        ktc_offset = gain * rng2.normal(scale=ktc_noise, size=(ny,nx))
        data += ktc_offset
    if prog_bar: pbar.update(1)
        
    ####################
    # Add super bias
    if prog_bar: pbar.set_description("Super Bias")
    if include_bias:
        data += gain * super_bias
    if prog_bar: pbar.update(1)
    
    ####################
    # Apply PPC (is this best location for this to occur?)
    if prog_bar: pbar.set_description("Include PPC")
    if apply_ppc:
        data = add_ppc(data, nchans=nchan, kernel=k_ppc, in_place=True,
                       same_scan_direction=ssd, reverse_scan_direction=rsd)
    if prog_bar: pbar.update(1)
    
    ####################
    # Add amplifier channel crosstalk
    if prog_bar: pbar.set_description("Amplifier Crosstalk")
    if amp_crosstalk:
        data = add_xtalk(data, det, coeffs=None)
    if prog_bar: pbar.update(1)

    ####################
    # Add read and 1/f noise
    if prog_bar: pbar.set_description("Detector & ASIC Noise")
    if nchan==1:
        rn, up = (rn[0], up[0])
    rn  = None if (not include_rn)    else rn
    up  = None if (not include_upink) else up
    cp  = None if (not include_cpink) else cp*1.2
    acn = None if (not include_acn)   else acn
    rseed = rseed_dict['sim_noise_data']
    data += gain * sim_noise_data(det, rd_noise=rn, u_pink=up, c_pink=cp, acn=acn, 
                                  corr_scales=scales, ref_ratio=ref_ratio, 
                                  rand_seed=rseed, verbose=False)
    if prog_bar: pbar.update(1)

    ####################
    # Add reference offsets
    if prog_bar: pbar.set_description("Ref Pixel Offsets")
    if include_refoffsets:
        rseed = rseed_dict['gen_ramp_biases']
        data += gain * gen_ramp_biases(dco.ref_pixel_dict, nchan=nchan, include_refinst=include_refinst,
                                       data_shape=data.shape, ref_border=ref_info, rand_seed=rseed)
    if prog_bar: pbar.update(1)

    ####################
    # Add column noise
    if prog_bar: pbar.set_description("Column Noise")
    # Passing col_noise allows for shifting of noise 
    # by one col ramp-to-ramp in higher level function
    if include_colnoise and (col_noise is None):
        rseed = rseed_dict['gen_col_noise']
        col_noise = gain * gen_col_noise(dco.column_variations, dco.column_prob_bad, 
                                         nz=nz, nx=nx, rand_seed=rseed)
    elif (include_colnoise==False):
        col_noise = 0
    # Add to data
    data += col_noise
    if prog_bar: pbar.update(1)

    # Convert to DN (16-bit int)
    if out_ADU:
        data /= gain
        data[data < 0] = 0
        data[data >= 2**16] = 2**16 - 1
        data = data.astype('uint16')

    if prog_bar: pbar.close()
    
    # return_zero_frame not set, True if not RAPID (what about BRIGHT1??)
    if return_zero_frame is None:
        return_zero_frame = False if 'RAPID' in det.multiaccum.read_mode else True
    
    # Return to sci coordinates
    if cframe=='sci':
        data = det_to_sci(data, det.detid)

    gc.collect()
    if return_full_ramp:
        if return_zero_frame:
            return data, data[0].copy()
        else:
            return data
    else:
        return ramp_resample(data, det, return_zero_frame=return_zero_frame)


def make_ramp_poisson(im_slope, det, out_ADU=True, zero_data=False, 
                      include_poisson=True, rand_seed=None):
    """
    Create a ramp with only photon noise. Useful for quick simulations.
    
    im_slope : Slope image (detector coordinates) in e-/sec
    det      : Detector information class
    out_ADU  : Convert to 16-bit UINT?
    zero_data: Return the so-called "zero frame"?
    """

    # from copy import deepcopy
    # xpix = det.xpix
    # ypix = det.ypix
    
    rng = np.random.default_rng(rand_seed)

    ma  = det.multiaccum

    nd1     = ma.nd1
    nd2     = ma.nd2
    nf      = ma.nf
    ngroup  = ma.ngroup
    t_frame = det.time_frame

    # Number of total frames up the ramp (including drops)
    naxis3 = nd1 + ngroup*nf + (ngroup-1)*nd2

    # Set reference pixels' slopes equal to 0
    w = det.ref_info
    if w[0] > 0: # lower
        im_slope[:w[0],:] = 0
    if w[1] > 0: # upper
        im_slope[-w[1]:,:] = 0
    if w[2] > 0: # left
        im_slope[:,:w[2]] = 0
    if w[3] > 0: # right
        im_slope[:,-w[3]:] = 0
        
    # Remove any negative values
    im_slope[im_slope<0] = 0

    # Count accumulation for a single frame
    frame = im_slope * t_frame
    # Add Poisson noise at each frame step
    sh0, sh1 = im_slope.shape
    new_shape = (naxis3, sh0, sh1)
    if include_poisson:
        ramp = rng.poisson(lam=frame, size=new_shape).astype(np.float64)
    else:
        ramp = np.array([frame for i in range(naxis3)])
    # Perform cumulative sum in place
    data = np.cumsum(ramp, axis=0)

    # Convert to ADU (16-bit UINT)
    # return data
    if out_ADU:
        gain = det.gain
        data /= gain
        data[data < 0] = 0
        data[data >= 2**16] = 2**16 - 1
        data = data.astype('uint16')
        
    return ramp_resample(data, det, return_zero_frame=zero_data)

