.. currentmodule:: pynrc.maths

Math Tools
==============

.. automodapi:: pynrc.maths.coords
    
.. automodapi:: pynrc.maths.image_manip

.. automodapi:: pynrc.maths.fast_poly

.. autosummary::

    ~pynrc.maths.fast_poly.jl_poly
    ~pynrc.maths.fast_poly.jl_poly_fit

.. automodapi:: webbpsf_ext.robust
    :no-main-docstr:


----------------------------

``pynrc.maths.coords``
----------------------------

.. autofunction:: pynrc.maths.coords.dist_image
.. autofunction:: pynrc.maths.coords.xy_to_rtheta
.. autofunction:: pynrc.maths.coords.rtheta_to_xy
.. autofunction:: pynrc.maths.coords.xy_rot
.. autofunction:: pynrc.maths.coords.Tel2Sci_info
.. autofunction:: pynrc.maths.coords.det_to_sci
.. autofunction:: pynrc.maths.coords.sci_to_det
.. autofunction:: pynrc.maths.coords.ap_radec
.. autofunction:: pynrc.maths.coords.radec_to_v2v3
.. autofunction:: pynrc.maths.coords.v2v3_to_pixel
.. autofunction:: pynrc.maths.coords.NIRCam_V2V3_limits
.. autofunction:: pynrc.maths.coords.get_NRC_v2v3_limits
.. autofunction:: pynrc.maths.coords.get_v2v3_limits
.. autofunction:: pynrc.maths.coords.gen_sgd_offsets
.. autofunction:: pynrc.maths.coords.get_idl_offset
.. autofunction:: pynrc.maths.coords.radec_offset
.. autofunction:: pynrc.maths.coords.jwst_point
.. autofunction:: pynrc.maths.coords.plotAxes

----------------------------

``pynrc.maths.image_manip``
----------------------------

.. autofunction:: pynrc.maths.image_manip.hist_indices
.. autofunction:: pynrc.maths.image_manip.binned_statistic
.. autofunction:: pynrc.maths.image_manip.frebin
.. autofunction:: pynrc.maths.image_manip.fshift
.. autofunction:: pynrc.maths.image_manip.fourier_imshift
.. autofunction:: pynrc.maths.image_manip.shift_subtract
.. autofunction:: pynrc.maths.image_manip.align_LSQ
.. autofunction:: pynrc.maths.image_manip.scale_ref_image
.. autofunction:: pynrc.maths.image_manip.optimal_difference
.. autofunction:: pynrc.maths.image_manip.pad_or_cut_to_size
.. autofunction:: pynrc.maths.image_manip.crop_zero_rows_cols
.. autofunction:: pynrc.maths.image_manip.fix_nans_with_med
.. autofunction:: pynrc.maths.image_manip.rotate_offset
.. autofunction:: pynrc.maths.image_manip.rotate_shift_image
.. autofunction:: pynrc.maths.image_manip.image_rescale
.. autofunction:: pynrc.maths.image_manip.model_to_hdulist
.. autofunction:: pynrc.maths.image_manip.convolve_image
.. autofunction:: pynrc.maths.image_manip.fit_bootstrap


----------------------------

``pynrc.maths.fast_poly``
--------------------------

.. autofunction:: pynrc.maths.fast_poly.jl_poly_fit
.. autofunction:: pynrc.maths.fast_poly.jl_poly

----------------------------

``pynrc.maths.robust``
----------------------------

Small collection of robust statistical estimators based on functions from
Henry Freudenriech (Hughes STX) statistics library (called ROBLIB) that have
been incorporated into the AstroIDL User's Library.

.. autofunction:: pynrc.maths.robust.medabsdev
.. autofunction:: pynrc.maths.robust.mean
.. autofunction:: pynrc.maths.robust.biweightMean
.. autofunction:: pynrc.maths.robust.mode
.. autofunction:: pynrc.maths.robust.std
.. autofunction:: pynrc.maths.robust.checkfit
.. autofunction:: pynrc.maths.robust.linefit
.. autofunction:: pynrc.maths.robust.polyfit
