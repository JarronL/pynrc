Ref. Pixel Correction
=====================

.. currentmodule:: pynrc.reduce.ref_pixels

.. rubric:: Summary

.. autosummary::

    ~pynrc.reduce.ref_pixels.NRC_refs
    ~pynrc.reduce.ref_pixels.reffix_hxrg
    ~pynrc.reduce.ref_pixels.reffix_amps
    ~pynrc.reduce.ref_pixels.ref_filter
    ~pynrc.reduce.ref_pixels.calc_avg_amps
    ~pynrc.reduce.ref_pixels.calc_avg_cols
    ~pynrc.reduce.ref_pixels.calc_col_smooth
    ~pynrc.reduce.ref_pixels.smooth_fft


.. rubric:: Class Documentation

.. autoclass:: pynrc.reduce.ref_pixels.NRC_refs
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~pynrc.reduce.ref_pixels.NRC_refs.multiaccum
      ~pynrc.reduce.ref_pixels.NRC_refs.multiaccum_times
      ~pynrc.reduce.ref_pixels.NRC_refs.refs_bot
      ~pynrc.reduce.ref_pixels.NRC_refs.refs_top
      ~pynrc.reduce.ref_pixels.NRC_refs.refs_right
      ~pynrc.reduce.ref_pixels.NRC_refs.refs_left

   .. rubric:: Methods Summary

   .. autosummary::

      ~pynrc.reduce.ref_pixels.NRC_refs.calc_avg_amps
      ~pynrc.reduce.ref_pixels.NRC_refs.calc_avg_cols
      ~pynrc.reduce.ref_pixels.NRC_refs.calc_col_smooth
      ~pynrc.reduce.ref_pixels.NRC_refs.correct_amp_refs
      ~pynrc.reduce.ref_pixels.NRC_refs.correct_col_refs

   .. rubric:: Methods Documentation

   .. automethod:: calc_avg_amps
   .. automethod:: calc_avg_cols
   .. automethod:: calc_col_smooth
   .. automethod:: correct_amp_refs
   .. automethod:: correct_col_refs

------------------

.. rubric:: Functions Documentation

.. autofunction:: pynrc.reduce.ref_pixels.reffix_hxrg
.. autofunction:: pynrc.reduce.ref_pixels.reffix_amps
.. autofunction:: pynrc.reduce.ref_pixels.ref_filter
.. autofunction:: pynrc.reduce.ref_pixels.calc_avg_amps
.. autofunction:: pynrc.reduce.ref_pixels.calc_avg_cols
.. autofunction:: pynrc.reduce.ref_pixels.calc_col_smooth
.. autofunction:: pynrc.reduce.ref_pixels.smooth_fft
