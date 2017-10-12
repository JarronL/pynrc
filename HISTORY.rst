Revision History
================

v0.6.0 (Oct 2017)
-----------------

- Support for Python 3
- Updated code comments for ``sphinx`` and ``readthedocs`` documentation
- Create ``setup.py`` install file
- Modify grism PSF shapes due to aperture shape
- Detector frames times based on ASIC microcode build 10
- Headers for DMS data

v0.5.0 (Feb 2017)
-----------------

- Initial GitHub release
- Match version numbering to ``WebbPSF`` equivalent
- ND Acquisition mode
- Ramp settings optimizer
- Can now simulate ramps with detector noise
- Query Euclid's IPAC server for position-dependent Zodiacal emission
- Added example Jupyter notebooks

v0.1.2 (Jan 2017)
-----------------
- Observations subclass for coronagraphs and direct imaging

v0.1.1 (Sep 2016)
-----------------
- Add support for LW slitless grism
- Add support for extended sources

v0.1.0 (Aug 2016)
-----------------
- Rewrite of ``SimNRC`` and rename ``pynrc``
- Object oriented ``multiaccum``, ``DetectorOps``, and ``NIRCam`` classes
- Create separate detector instances in ``NIRCam`` class


Planned Updates
---------------

FoV aware positions
+++++++++++++++++++
    - Background roll off at grism edges
    - Filter location relative offsets
    - SIAF stuff
    - Subarray positions
    - SCA Gaps and Module gaps

Detector updates in ngNRC.py
++++++++++++++++++++++++++++
    - Pixel non-linearity
    - Intrapixel Capacitance (IPC)
    - Post-pixel Coupling (PPC) due to ADC "smearing"
    - Persistence/latent image
    - Optical distortions
    - QE variations across a pixel's surface
    - RTN Noise
    - Flat field variations

PSF Related
+++++++++++
    - Include dispersion of coronagraph PSF
    - WebbPSF needs to center PSF in pixel center
    - PSF Jitter options
    - Bar Mask offsets (using nrc.options['bar_offset'])
    - PSF updates (convolution) based on geometric spot size

Miscellaneous
+++++++++++++
    - Option to exclude flat field noise
    - Option to exclude cosmic ray noise
    - Ramp optimizer warning for large number of group loops?
    - multi-thread ramp optimizer?
    - Random cosmic ray hits in exposure simulator
