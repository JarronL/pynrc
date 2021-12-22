.. currentmodule:: pynrc

Utilities
=========

.. toctree::

    math_tools.rst
    nrc_utils.rst
    spectral_tools.rst
    speckle_tools.rst

.. rubric:: Coordinates Summary

.. autosummary::

    ~pynrc.maths.coords.dist_image
    ~pynrc.maths.coords.xy_to_rtheta
    ~pynrc.maths.coords.rtheta_to_xy
    ~pynrc.maths.coords.xy_rot
    ~pynrc.maths.coords.det_to_V2V3
    ~pynrc.maths.coords.V2V3_to_det
    ~pynrc.maths.coords.plotAxes

.. rubric:: Image Manipulation Summary

.. autosummary::

    ~pynrc.maths.image_manip.hist_indices
    ~pynrc.maths.image_manip.binned_statistic
    ~pynrc.maths.image_manip.frebin
    ~pynrc.maths.image_manip.fshift
    ~pynrc.maths.image_manip.fourier_imshift
    ~pynrc.maths.image_manip.shift_subtract
    ~pynrc.maths.image_manip.align_LSQ
    ~pynrc.maths.image_manip.scale_ref_image
    ~pynrc.maths.image_manip.optimal_difference
    ~pynrc.maths.image_manip.pad_or_cut_to_size
    ~pynrc.maths.image_manip.fix_nans_with_med

.. rubric:: Polynomial Fitting Summary

.. autosummary::

    ~pynrc.maths.fast_poly.jl_poly_fit
    ~pynrc.maths.fast_poly.jl_poly

.. rubric:: Robust Summary

.. autosummary::

    ~pynrc.maths.robust.biweightMean
    ~pynrc.maths.robust.checkfit
    ~pynrc.maths.robust.linefit
    ~pynrc.maths.robust.mean
    ~pynrc.maths.robust.medabsdev
    ~pynrc.maths.robust.mode
    ~pynrc.maths.robust.polyfit
    ~pynrc.maths.robust.std

.. rubric:: NIRCam Tools Summary

.. autosummary::

    ~nrc_utils.read_filter
    ~nrc_utils.psf_coeff
    ~nrc_utils.gen_image_coeff
    ~nrc_utils.bg_sensitivity
    ~nrc_utils.sat_limit_webbpsf
    ~nrc_utils.pix_noise
    ~nrc_utils.channel_select
    ~nrc_utils.grism_res

.. rubric:: Spectral Tools Summary

.. autosummary::

    ~nrc_utils.stellar_spectrum
    ~nrc_utils.BOSZ_spectrum
    ~nrc_utils.planets_sb12
    ~nrc_utils.sp_accr
    ~nrc_utils.zodi_spec
    ~nrc_utils.zodi_euclid
    ~nrc_utils.bp_2mass
    ~nrc_utils.bin_spectrum
    
.. rubric:: Speckle Noise Summary

.. autosummary::

    ~speckle_noise.OPD_extract
    ~speckle_noise.opd_extract_mp
    ~speckle_noise.opd_sci_gen
    ~speckle_noise.opd_sci_gen_mp
    ~speckle_noise.opd_ref_gen
    ~speckle_noise.ODP_drift_all
    ~speckle_noise.get_psf
    ~speckle_noise.gen_psf_ref_all
    ~speckle_noise.get_contrast
    ~speckle_noise.residual_speckle_image
    ~speckle_noise.speckle_noise_image
    ~speckle_noise.read_opd_file
    ~speckle_noise.read_opd_slice

