Revision History
================

v1.0.1 (Dec 14, 2021)
---------------------

- Default OPD JWST_OTE_OPD_RevAA_prelaunch_predicted.fits

v1.0.0 (Nov 22, 2021)
---------------------

- Updates to work with WebbPSF v1 release candidate
- Move PSF generation to new ``webbpsf_ext`` package (https://github.com/JarronL/webbpsf_ext)
- Create DMS-like level1b FITS files using pipeline data models for imaging and coronagraphy
- PSF coefficients now use Legendre polynomials by default
- Create calibration files for each SCA (darks, IPC, noise, flats, linearity, etc)
- Background roll-off at grism edges
- SIAF-aware locations

v0.9.0beta (no release)
-----------------------

- Updates to work with WebbPSF 0.9.0.
- Start working on commissioning and DMS-like data
- Add more advanced time-dependent detector effects
- BEX model isochrones for low-mass companions from Linder et al (2019)
- There was a pandemic...

v0.8.0beta (no release)
-----------------------

- Updates to work with WebbPSF 0.8.0.
- Phasing out support for Python 2
- Add info on saturation limits in terms of surface brightness 
- Include option to create grism 2nd order
- Detector pixel timing bugs
- Field-dependent WFE extrapolated beyond FoV for better sampling diversity
- Included field-dependent WFE for coronagraphy
- Added wavelength dispersion of LW coronagraphic PSF

v0.7.0 (Jun 2018)
-----------------

- Did not make it out of development before WebbPSF 0.8.0 release.
- Works with WebbPSF 0.7.0.

  - Field-dependent WFE
  - Image plane distortions

- Implemented ``jwst_backgrounds`` (not required)


v0.6.5 (Mar 2018)
-----------------

- Fixed a critical bug where the off-axis PSF size was incorrect
  when performing WFE drift calculations.

v0.6.4 (Mar 2018)
-----------------

- Off-axis PSFs now get drifted in the same way as their on-axis
  counterparts.
- Created an intermediate ``nrc_hci`` class to enable offsets of WFE drifted PSFs.


v0.6.3 (Mar 2018)
-----------------

- First PyPI release.
- Effectively the same as 0.6.2, but better documentation of packaging and distributing.


v0.6.2 (Mar 2018)
-----------------

- Implemented coronagraphic wedges, including arbitrary offsets along bar
- Renamed ``obs_coronagraphy`` to ``~pynrc.obs_hci``

  - Faster modeling of off-axis PSFs
  - Include coronagraphic features (e.g.: ND squares) in slope images
  - Roll subtracted images include option to use Roll1-Roll2
  - Fixed bug that was slowing down PSF convolution of disks

- Can now generate docs directly from Jupyter notebooks using nbsphinx extension
- Coronagraphic tutorials for docs
- Create the ``source_spectrum`` class to fit spectra to observed photometry.

v0.6.0 (Dec 2017)
-----------------

- Support for Python 3 (mostly ``map``, ``dict``, and index fixes)
- Updated code comments for ``sphinx`` and ``readthedocs`` documentation
- Create ``setup.py`` install file
- Modify grism PSF shapes due to aperture shape
- Detector frames times based on ASIC microcode build 10
- Headers for DMS data
- Three major changes to PSF coefficients

  - coefficients based on module (SWA, SWB, LWA, LWB), rather than filter
  - WFE drift coefficient relations
  - field-dependent coefficient relation

v0.5.0 (Feb 2017)
-----------------

- Initial GitHub release
- Match version numbering to ``WebbPSF`` equivalent
- ND Acquisition mode
- Ramp settings optimizer
- Can now simulate ramps with detector noise
- Query Euclid's IPAC server for time/position-dependent Zodiacal emission
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
