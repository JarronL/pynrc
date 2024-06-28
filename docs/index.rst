.. pynrc documentation master file, created by
   sphinx-quickstart on Mon Oct  9 11:42:06 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyNRC Documentation
===================

pyNRC - Python ETC and Simulator for JWST NIRCam
----------------------------------------------------------------------------

pyNRC is a set of Python-based tools for planning observations with JWST NIRCam. It includes an ETC, a simple image slope simulator, and an enhanced data simulator compatible with the JWST pipeline. This package works for a variety of NIRCam observing modes including direct imaging, coronagraphic imaging, slitless grism spectroscopy, and weak lens imaging. All PSFs are generated via `WebbPSF <https://webbpsf.readthedocs.io>`_ and `WebbPSF Extensions <https://github.com/JarronL/webbpsf_ext>`_ to reproduce realistic JWST images and spectra.

Developed by Jarron Leisenring and contributors at University of Arizona (2015 - 2024). 

.. toctree::
   :caption: Getting Started
   :maxdepth: 1
   
   readme.rst

.. toctree::
   :maxdepth: 1

   installation.rst
   install_clean.rst

.. toctree::
   :caption: User Guides
   :maxdepth: 1

   tutorials/Basic_Usage.ipynb
   tutorials/Ramp_Optimization_Examples.ipynb
   tutorials/Coronagraph_Basics.ipynb
   tutorials/Coronagraph_Wedges.ipynb
   tutorials/HR8799_DMS_Level1b.ipynb
   
.. toctree::
   :caption: Reference Info
   :maxdepth: 1
   
   api.rst
   change_log.rst
   license.rst
   contributing.rst


License & Attribution
---------------------

pyNRC is free software made available under the MIT License. For details
see :ref:`LICENSE <license>`.

.. admonition:: Citing pyNRC

   If you make use of pyNRC in your work, please cite the following paper:

    * Leisenring, J. (2024). pyNRC: Python ETC and Simulator for JWST NIRCam (Version v1.2.0) [Computer software]. https://doi.org/10.5281/zenodo.5829552
    * BibTeX entry: ``@software{Leisenring_pyNRC_Python_ETC_2024, author = {Leisenring, Jarron}, doi = {10.5281/zenodo.5829552}, month = may, title = {{pyNRC: Python ETC and Simulator for JWST NIRCam}}, url = {https://github.com/JarronL/pynrc}, version = {v1.2.0}, year = {2024} }``


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
