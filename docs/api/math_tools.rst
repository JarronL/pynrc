.. currentmodule:: pynrc

Math Tools
==============

.. rubric:: Coordinates Summary

.. autosummary::

    ~pynrc.maths.coords.dist_image
    ~pynrc.maths.coords.xy_to_rtheta
    ~pynrc.maths.coords.rtheta_to_xy
    ~pynrc.maths.coords.xy_rot
    ~pynrc.maths.coords.Tel2Sci_info
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
    ~pynrc.maths.image_manip.rotate_offset

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


----------------------------

``pynrc.maths.coords``
--------------------------

.. autofunction:: pynrc.maths.coords.dist_image
.. autofunction:: pynrc.maths.coords.xy_to_rtheta
.. autofunction:: pynrc.maths.coords.rtheta_to_xy
.. autofunction:: pynrc.maths.coords.xy_rot
.. autofunction:: pynrc.maths.coords.Tel2Sci_info
.. autofunction:: pynrc.maths.coords.det_to_V2V3
.. autofunction:: pynrc.maths.coords.V2V3_to_det
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
.. autofunction:: pynrc.maths.image_manip.fix_nans_with_med
.. autofunction:: pynrc.maths.image_manip.rotate_offset

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

.. autofunction:: pynrc.maths.robust.biweightMean
.. autofunction:: pynrc.maths.robust.checkfit
.. autofunction:: pynrc.maths.robust.linefit
.. autofunction:: pynrc.maths.robust.mean
.. autofunction:: pynrc.maths.robust.medabsdev
.. autofunction:: pynrc.maths.robust.mode
.. autofunction:: pynrc.maths.robust.polyfit
.. autofunction:: pynrc.maths.robust.std
