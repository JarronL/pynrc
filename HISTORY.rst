Revision History
================

v1.3.0 (Sept 19, 2024)
----------------------

 - Bump to version 1.3.0 to attempt to fix PyPI upload issue

v1.2.2 (Sept 19, 2024)
----------------------

 - Bug fix in charge migration to check to break out if no saturated pixels
 - Correctly crop stellar PSF in ``obs_hci`` class while generating slope image

v1.2.1 (May 24, 2024)
---------------------

 - Bug fix for poppy requirement version. (should have been >1.1.0 rather than >1.2.0)

v1.2.0 (May 13, 2024)
---------------------

 - Bring up to date with WebbPSF v1.2.0 and webbpsf_ext v1.2.0
 - Add analysis tools for PCA and NMF PSF subtraction
 - Add a number of simulation features to match flight performance

v1.0.4 (Dec 28, 2021)
---------------------

 - check if ``im_star`` is int or float if not None
 - set ``nexposures = 1`` for level1b using ``NIRCam()`` class function
 - deprecate nghxrg.py
 - add tutorial ipynb files
 - update api docs auto generation
 - use ``webbpsf_ext`` v1.0.4

v1.0.3 (Dec 23, 2021)
---------------------

- Minor updates to seamlessly generate new releases on PyPI and new docs on readthedocs

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
