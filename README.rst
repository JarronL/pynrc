=====
pyNRC
=====

A JWST NIRCam ETC and Simulator
===============================

.. image:: https://img.shields.io/pypi/v/pynrc.svg
        :target: https://pypi.python.org/pypi/pynrc

.. image:: https://img.shields.io/travis/JarronL/pynrc.svg
        :target: https://travis-ci.org/JarronL/pynrc

.. image:: https://readthedocs.org/projects/pynrc/badge/?version=latest
        :target: https://pynrc.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

*Authors:* Jarron Leisenring (UA)

*Contributors:* Everett Schlawin (UA), Jonathan Fraine (STScI)

**!!Under Development!!**

pyNRC is a set of Python-based tools for planning observations with JWST NIRCam, 
such as an ETC, a simple image slope simulator, and an enhanced data simulator.

While special attention has been placed on NIRCam coronagraphic modes, 
this package also works for a variety of NIRCam observing modes including: 
- direct imaging 
- coronagraphic imaging
- weak lens imaging
- slitless grism spectroscopy
- DHS observations (TBI)

All PSFs are generated via WebbPSF (https://webbpsf.readthedocs.io) to reproduce 
realistic JWST images and spectra.

Documentation can be found at https://pynrc.readthedocs.io.

**Note:** pyNRC enables more modes than are officially allowed by the Observatory,
(ie., filter + coronagraphic combinations, subarray sizes, etc.). 
Just because you can do something with pyNRC does not mean it will be supported.
Check out https://jwst-docs.stsci.edu/display/JTI/NIRCam+Observing+Modes for more information.

Similar to some of its dependencies, pyNRC requires a host of input data files in
order to generate simulations. Due to the size of these files, they are not included
with this source distribution. Please see the documentation for instructions on how to
to download the required data files.
