.. currentmodule:: pynrc.speckle_noise

Speckle Noise Tools
===================

.. admonition:: Disclaimer

    These tools were created for investigations into speckle maps during
    development of pynrc. Documentation is relatively sparse and functions 
    may not be maintained very well with respect to updated versions of
    Python and webbpsf. They are provided here for completeness, and may 
    even prove useful to someone in the future, present, or past(?).

.. rubric:: Class Summary

.. autosummary::

    ~OPD_extract

.. rubric:: Functions Summary

.. autosummary::

    ~opd_extract_mp
    ~opd_sci_gen
    ~opd_sci_gen_mp
    ~opd_ref_gen
    ~ODP_drift_all
    ~get_psf
    ~gen_psf_ref_all
    ~get_contrast
    ~residual_speckle_image
    ~speckle_noise_image
    ~read_opd_file
    ~read_opd_slice


.. rubric:: Documentation

.. autoclass:: OPD_extract
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~OPD_extract.mask_pupil
      ~OPD_extract.mask_opd
      ~OPD_extract.coeff_pupil
      ~OPD_extract.coeff_segs

   .. rubric:: Methods Summary

   .. autosummary::

      ~OPD_extract.mask_seg
      ~OPD_extract.opd_seg
      ~OPD_extract.combine_opd_segs

   .. rubric:: Methods Documentation

   .. automethod:: mask_seg
   .. automethod:: opd_seg
   .. automethod:: combine_opd_segs

.. autofunction:: opd_extract_mp
.. autofunction:: opd_sci_gen
.. autofunction:: opd_sci_gen_mp
.. autofunction:: opd_ref_gen
.. autofunction:: ODP_drift_all
.. autofunction:: get_psf
.. autofunction:: gen_psf_ref_all
.. autofunction:: get_contrast
.. autofunction:: residual_speckle_image
.. autofunction:: speckle_noise_image
.. autofunction:: read_opd_file
.. autofunction:: read_opd_slice
