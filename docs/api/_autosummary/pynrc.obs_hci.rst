pynrc.obs\_hci
==============

.. currentmodule:: pynrc

.. autoclass:: obs_hci
   :members:
   :inherited-members:
   :show-inheritance:

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~obs_hci.__init__
      ~obs_hci.add_planet
      ~obs_hci.attenuate_with_mask
      ~obs_hci.bg_zodi
      ~obs_hci.bg_zodi_image
      ~obs_hci.calc_contrast
      ~obs_hci.calc_datacube
      ~obs_hci.calc_datacube_fast
      ~obs_hci.calc_psf
      ~obs_hci.calc_psf_from_coeff
      ~obs_hci.calc_psf_offset_from_center
      ~obs_hci.calc_psfs_grid
      ~obs_hci.calc_psfs_sgd
      ~obs_hci.delete_planets
      ~obs_hci.display
      ~obs_hci.drift_opd
      ~obs_hci.gen_disk_hdulist
      ~obs_hci.gen_disk_image
      ~obs_hci.gen_disk_psfs
      ~obs_hci.gen_mask_image
      ~obs_hci.gen_mask_transmission_map
      ~obs_hci.gen_offset_psf
      ~obs_hci.gen_planets_image
      ~obs_hci.gen_pointing_offsets
      ~obs_hci.gen_psf_coeff
      ~obs_hci.gen_psfs_over_fov
      ~obs_hci.gen_ref_det
      ~obs_hci.gen_roll_image
      ~obs_hci.gen_save_name
      ~obs_hci.gen_slope_image
      ~obs_hci.gen_wfedrift_coeff
      ~obs_hci.gen_wfefield_coeff
      ~obs_hci.gen_wfemask_coeff
      ~obs_hci.get_bar_offset
      ~obs_hci.get_opd_file_full_path
      ~obs_hci.get_opd_info
      ~obs_hci.get_optical_system
      ~obs_hci.get_siaf_apname
      ~obs_hci.get_subarray_name
      ~obs_hci.get_wfe
      ~obs_hci.interpolate_was_opd
      ~obs_hci.load_wss_opd
      ~obs_hci.load_wss_opd_by_date
      ~obs_hci.planet_spec
      ~obs_hci.plot_bandpass
      ~obs_hci.psf_grid
      ~obs_hci.ramp_optimize
      ~obs_hci.recenter_psf
      ~obs_hci.sat_limits
      ~obs_hci.saturation_levels
      ~obs_hci.sensitivity
      ~obs_hci.set_position_from_aperture_name
      ~obs_hci.simulate_level1b
      ~obs_hci.simulate_ramps
      ~obs_hci.star_flux
      ~obs_hci.update_detectors
      ~obs_hci.update_detectors_ref
      ~obs_hci.update_from_SIAF
      ~obs_hci.update_psf_coeff
      ~obs_hci.visualize_wfe_budget
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~obs_hci.LONG_WAVELENGTH_MAX
      ~obs_hci.LONG_WAVELENGTH_MIN
      ~obs_hci.ND_acq
      ~obs_hci.SHORT_WAVELENGTH_MAX
      ~obs_hci.SHORT_WAVELENGTH_MIN
      ~obs_hci.aperturename
      ~obs_hci.bandpass
      ~obs_hci.bar_offset
      ~obs_hci.channel
      ~obs_hci.coron_substrate
      ~obs_hci.det_info
      ~obs_hci.detector
      ~obs_hci.detector_list
      ~obs_hci.detector_position
      ~obs_hci.disk_params
      ~obs_hci.fastaxis
      ~obs_hci.filter
      ~obs_hci.filter_list
      ~obs_hci.fov_pix
      ~obs_hci.image_mask
      ~obs_hci.is_coron
      ~obs_hci.is_dark
      ~obs_hci.is_grism
      ~obs_hci.is_lyot
      ~obs_hci.module
      ~obs_hci.multiaccum
      ~obs_hci.multiaccum_times
      ~obs_hci.name
      ~obs_hci.ndeg
      ~obs_hci.npsf
      ~obs_hci.options
      ~obs_hci.oversample
      ~obs_hci.pixelscale
      ~obs_hci.planets
      ~obs_hci.psf_info
      ~obs_hci.pupil
      ~obs_hci.pupil_mask
      ~obs_hci.pupilopd
      ~obs_hci.quick
      ~obs_hci.save_dir
      ~obs_hci.save_name
      ~obs_hci.scaid
      ~obs_hci.siaf_ap
      ~obs_hci.siaf_ap_names
      ~obs_hci.slowaxis
      ~obs_hci.telescope
      ~obs_hci.use_fov_pix_plus1
      ~obs_hci.wave_fit
      ~obs_hci.well_level
      ~obs_hci.wfe_ref_drift
      ~obs_hci.wfe_roll_drift
      ~obs_hci.include_ote_field_dependence
      ~obs_hci.image_mask_list
      ~obs_hci.pupil_mask_list
   
   