pynrc.NIRCam
============

.. currentmodule:: pynrc

.. autoclass:: NIRCam
   :members:
   :inherited-members:
   :show-inheritance:

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~NIRCam.__init__
      ~NIRCam.bg_zodi
      ~NIRCam.bg_zodi_image
      ~NIRCam.calc_datacube
      ~NIRCam.calc_datacube_fast
      ~NIRCam.calc_psf
      ~NIRCam.calc_psf_from_coeff
      ~NIRCam.calc_psf_offset_from_center
      ~NIRCam.calc_psfs_grid
      ~NIRCam.calc_psfs_sgd
      ~NIRCam.display
      ~NIRCam.drift_opd
      ~NIRCam.gen_mask_image
      ~NIRCam.gen_mask_transmission_map
      ~NIRCam.gen_psf_coeff
      ~NIRCam.gen_psfs_over_fov
      ~NIRCam.gen_save_name
      ~NIRCam.gen_wfedrift_coeff
      ~NIRCam.gen_wfefield_coeff
      ~NIRCam.gen_wfemask_coeff
      ~NIRCam.get_bar_offset
      ~NIRCam.get_opd_file_full_path
      ~NIRCam.get_opd_info
      ~NIRCam.get_optical_system
      ~NIRCam.get_siaf_apname
      ~NIRCam.get_subarray_name
      ~NIRCam.get_wfe
      ~NIRCam.interpolate_was_opd
      ~NIRCam.load_wss_opd
      ~NIRCam.load_wss_opd_by_date
      ~NIRCam.plot_bandpass
      ~NIRCam.psf_grid
      ~NIRCam.ramp_optimize
      ~NIRCam.recenter_psf
      ~NIRCam.sat_limits
      ~NIRCam.saturation_levels
      ~NIRCam.sensitivity
      ~NIRCam.set_position_from_aperture_name
      ~NIRCam.simulate_level1b
      ~NIRCam.simulate_ramps
      ~NIRCam.update_detectors
      ~NIRCam.update_from_SIAF
      ~NIRCam.update_psf_coeff
      ~NIRCam.visualize_wfe_budget
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~NIRCam.LONG_WAVELENGTH_MAX
      ~NIRCam.LONG_WAVELENGTH_MIN
      ~NIRCam.ND_acq
      ~NIRCam.SHORT_WAVELENGTH_MAX
      ~NIRCam.SHORT_WAVELENGTH_MIN
      ~NIRCam.aperturename
      ~NIRCam.bandpass
      ~NIRCam.channel
      ~NIRCam.coron_substrate
      ~NIRCam.det_info
      ~NIRCam.detector
      ~NIRCam.detector_list
      ~NIRCam.detector_position
      ~NIRCam.fastaxis
      ~NIRCam.filter
      ~NIRCam.filter_list
      ~NIRCam.fov_pix
      ~NIRCam.image_mask
      ~NIRCam.is_coron
      ~NIRCam.is_dark
      ~NIRCam.is_grism
      ~NIRCam.is_lyot
      ~NIRCam.module
      ~NIRCam.multiaccum
      ~NIRCam.multiaccum_times
      ~NIRCam.name
      ~NIRCam.ndeg
      ~NIRCam.npsf
      ~NIRCam.options
      ~NIRCam.oversample
      ~NIRCam.pixelscale
      ~NIRCam.psf_info
      ~NIRCam.pupil
      ~NIRCam.pupil_mask
      ~NIRCam.pupilopd
      ~NIRCam.quick
      ~NIRCam.save_dir
      ~NIRCam.save_name
      ~NIRCam.scaid
      ~NIRCam.siaf_ap
      ~NIRCam.siaf_ap_names
      ~NIRCam.slowaxis
      ~NIRCam.telescope
      ~NIRCam.use_fov_pix_plus1
      ~NIRCam.wave_fit
      ~NIRCam.well_level
      ~NIRCam.include_ote_field_dependence
      ~NIRCam.image_mask_list
      ~NIRCam.pupil_mask_list
   
   