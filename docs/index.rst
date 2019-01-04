.. pynrc documentation master file, created by
   sphinx-quickstart on Mon Oct  9 11:42:06 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyNRC's documentation!
=================================

pyNRC - Python ETC and Simulator for JWST NIRCam
----------------------------------------------------------------------------

pyNRC is a set of Python-based tools for planning observations with JWST NIRCam. 
It includes an ETC, a simple image slope simulator, and an enhanced data simulator.
This package works for a variety of NIRCam observing modes including direct imaging, 
coronagraphic imaging, slitless grism spectroscopy, DHS observations, 
and weak lens imaging. All PSFs are generated via WebbPSF 
(https://webbpsf.readthedocs.io) to reproduce realistic JWST images and spectra.

Developed by Jarron Leisenring and contributors at University of Arizona (2015 - 2019).
    

.. toctree::
   :caption: Getting Started
   :maxdepth: 1
   
   readme.rst
   installation.rst
   install_clean.rst

.. toctree::
   :caption: User Guides
   :maxdepth: 1

   tutorials/Basic_Usage.ipynb   
   tutorials/Ramp_Optimization_Examples.ipynb
   tutorials/Coronagraph_Basics.ipynb
   tutorials/Coronagraph_Wedges.ipynb
   
.. toctree::
   :caption: Reference Info
   :maxdepth: 1
   
   api_new.rst
   change_log.rst
   license.rst
   contributing.rst
   


License & Attribution
---------------------

pyNRC is free software made available under the MIT License. For details
see :ref:`LICENSE <license>`.

.. attention:: Citing pyNRC

    If you make use of pyNRC in your work, please cite the following paper: 
    `Leisenring et al., "pyNRC: A NIRCam ETC and Simulation Toolset"`
    (in prep).

