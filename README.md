pyNRC - A JWST NIRCam ETC
=========================

*Authors:* Jarron Leisenring (UA)

*Contributors:* Everett Schlawin (UA), Jonathan Fraine (STScI)

**!!Under Development!!**

pyNRC is a set of Python-based tools for planning observations with JWST NIRCam, 
such as an ETC, ~~rudimentary overhead calculator~~ (TBI), simple image slope 
simulator, and full-blown DMS simulator.

The module works for a variety of NIRCam observing modes including direct imaging, 
coronagraphic imaging, slitless grism spectroscopy, ~~DHS observations~~ (TBI), 
and weak lens imaging.
All PSFs are generated via WebbPSF (https://webbpsf.readthedocs.io) to reproduce 
realistic JWST images and spectra.

**Note**: pyNRC allows for more modes than are officially allowed by the Observatory,
(ie., filter + coronagraphic combinations, subarray sizes, etc.). 
Just because you can do something with pyNRC does not mean it will be supported.
Check out https://jwst-docs.stsci.edu/display/JTI/NIRCam+Observing+Modes for more information.

The documentation is available at https://pynrc.readthedocs.org