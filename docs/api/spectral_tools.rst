.. currentmodule:: pynrc

Spectral Tools
==============

.. rubric:: Class Summary

.. autosummary::

    ~nrc_utils.source_spectrum
    ~nrc_utils.planets_sb12

.. rubric:: Functions Summary

.. autosummary::

    ~nrc_utils.stellar_spectrum
    ~nrc_utils.BOSZ_spectrum
    ~nrc_utils.sp_accr
    ~nrc_utils.zodi_spec
    ~nrc_utils.zodi_euclid
    ~nrc_utils.bp_2mass
    ~nrc_utils.bin_spectrum

.. rubric:: Documentation

.. autoclass:: pynrc.nrc_utils.source_spectrum
   :show-inheritance:

   .. rubric:: Methods Summary

   .. autosummary::

      ~source_spectrum.bb_jy
      ~source_spectrum.model_scale
      ~source_spectrum.model_IRexcess
      ~source_spectrum.func_resid
      ~source_spectrum.fit_SED
      ~source_spectrum.plot_SED

   .. rubric:: Methods Documentation

   .. automethod:: bb_jy
   .. automethod:: model_scale
   .. automethod:: model_IRexcess
   .. automethod:: func_resid
   .. automethod:: fit_SED
   .. automethod:: plot_SED


.. autoclass:: pynrc.nrc_utils.planets_sb12
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~planets_sb12.age
      ~planets_sb12.atmo
      ~planets_sb12.distance
      ~planets_sb12.entropy
      ~planets_sb12.flux
      ~planets_sb12.fluxunits
      ~planets_sb12.mass
      ~planets_sb12.mdot
      ~planets_sb12.wave
      ~planets_sb12.waveunits

   .. rubric:: Methods Summary

   .. autosummary::

      ~planets_sb12.export_pysynphot

   .. rubric:: Methods Documentation

   .. automethod:: export_pysynphot

.. autofunction:: pynrc.nrc_utils.stellar_spectrum
.. autofunction:: pynrc.nrc_utils.BOSZ_spectrum
.. autofunction:: pynrc.nrc_utils.sp_accr
.. autofunction:: pynrc.nrc_utils.zodi_spec
.. autofunction:: pynrc.nrc_utils.zodi_euclid
.. autofunction:: pynrc.nrc_utils.bp_2mass
.. autofunction:: pynrc.nrc_utils.bin_spectrum
