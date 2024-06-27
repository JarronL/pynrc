pynrc.nrc\_hci
==============

.. currentmodule:: pynrc

.. autoclass:: nrc_hci
   :members:
   :inherited-members:
   :show-inheritance:

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~nrc_hci.__init__
      ~nrc_hci.attenuate_with_mask
      ~nrc_hci.bg_zodi
      ~nrc_hci.bg_zodi_image
      ~nrc_hci.calc_datacube
      ~nrc_hci.calc_datacube_fast
      ~nrc_hci.calc_psf
      ~nrc_hci.calc_psf_from_coeff
      ~nrc_hci.calc_psf_offset_from_center
      ~nrc_hci.calc_psfs_grid
      ~nrc_hci.calc_psfs_sgd
      ~nrc_hci.display
      ~nrc_hci.drift_opd
      ~nrc_hci.gen_mask_image
      ~nrc_hci.gen_mask_transmission_map
      ~nrc_hci.gen_offset_psf
      ~nrc_hci.gen_pointing_offsets
      ~nrc_hci.gen_psf_coeff
      ~nrc_hci.gen_psfs_over_fov
      ~nrc_hci.gen_save_name
      ~nrc_hci.gen_wfedrift_coeff
      ~nrc_hci.gen_wfefield_coeff
      ~nrc_hci.gen_wfemask_coeff
      ~nrc_hci.get_bar_offset
      ~nrc_hci.get_opd_file_full_path
      ~nrc_hci.get_opd_info
      ~nrc_hci.get_optical_system
      ~nrc_hci.get_siaf_apname
      ~nrc_hci.get_subarray_name
      ~nrc_hci.get_wfe
      ~nrc_hci.interpolate_was_opd
      ~nrc_hci.load_wss_opd
      ~nrc_hci.load_wss_opd_by_date
      ~nrc_hci.plot_bandpass
      ~nrc_hci.psf_grid
      ~nrc_hci.ramp_optimize
      ~nrc_hci.recenter_psf
      ~nrc_hci.sat_limits
      ~nrc_hci.saturation_levels
      ~nrc_hci.sensitivity
      ~nrc_hci.set_position_from_aperture_name
      ~nrc_hci.simulate_level1b
      ~nrc_hci.simulate_ramps
      ~nrc_hci.update_detectors
      ~nrc_hci.update_from_SIAF
      ~nrc_hci.update_psf_coeff
      ~nrc_hci.visualize_wfe_budget
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~nrc_hci.LONG_WAVELENGTH_MAX
      ~nrc_hci.LONG_WAVELENGTH_MIN
      ~nrc_hci.ND_acq
      ~nrc_hci.SHORT_WAVELENGTH_MAX
      ~nrc_hci.SHORT_WAVELENGTH_MIN
      ~nrc_hci.aperturename
      ~nrc_hci.bandpass
      ~nrc_hci.bar_offset
      ~nrc_hci.channel
      ~nrc_hci.coron_substrate
      ~nrc_hci.det_info
      ~nrc_hci.detector
      ~nrc_hci.detector_list
      ~nrc_hci.detector_position
      ~nrc_hci.fastaxis
      ~nrc_hci.filter
      ~nrc_hci.filter_list
      ~nrc_hci.fov_pix
      ~nrc_hci.image_mask
      ~nrc_hci.is_coron
      ~nrc_hci.is_dark
      ~nrc_hci.is_grism
      ~nrc_hci.is_lyot
      ~nrc_hci.module
      ~nrc_hci.multiaccum
      ~nrc_hci.multiaccum_times
      ~nrc_hci.name
      ~nrc_hci.ndeg
      ~nrc_hci.npsf
      ~nrc_hci.options
      ~nrc_hci.oversample
      ~nrc_hci.pixelscale
      ~nrc_hci.psf_info
      ~nrc_hci.pupil
      ~nrc_hci.pupil_mask
      ~nrc_hci.pupilopd
      ~nrc_hci.quick
      ~nrc_hci.save_dir
      ~nrc_hci.save_name
      ~nrc_hci.scaid
      ~nrc_hci.siaf_ap
      ~nrc_hci.siaf_ap_names
      ~nrc_hci.slowaxis
      ~nrc_hci.telescope
      ~nrc_hci.use_fov_pix_plus1
      ~nrc_hci.wave_fit
      ~nrc_hci.well_level
      ~nrc_hci.include_ote_field_dependence
      ~nrc_hci.image_mask_list
      ~nrc_hci.pupil_mask_list
   
   