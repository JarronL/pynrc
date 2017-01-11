pyNRC - A JWST NIRCam ETC
=========================

*Authors:* Jarron Leisenring (University of Arizona)

**!!Under Development!!**

pyNRC is a set of Python-based tools for planning observations with JWST NIRCam, 
such as an ETC, ~~rudimentary overhead calculator~~ (TBI), and simple image slope 
simulator. 
pyNRC works for a vareity NIRCam observing modes including direct imaging, 
coronagraphic imaging, slitless grism spectroscopy, ~~DHS observations~~ (TBI), 
and weak lens imaging.
All PSFs are generated via WebbPSF (https://pythonhosted.org/webbpsf/) to reproduce 
realistic JWST images and spectra.

## Installation

At the moment, the quickest way to install pyNRC into your existing Python installation 
is to download or clone this repository and then create a .pth file in your Python 
site-packages directory that points to the pyNRC directory. 

For instance, if this git repository exists on your local machine at 
``/Users/Bob/GitHub/pynrc/`` and you have an Anaconda Python 2.7 installation at 
``/Users/Bob/anaconda/``, then you merely need to create some 
file ``/Users/Bob/anaconda/lib/python2.7/site-packages/somefile.pth`` with the GitHub 
folder path as context (``/Users/Bob/GitHub/``).
This method will add the GitHub directory to your Python path so that pynrc can be imported.

In addition, you must create a data directory that will hold cached information 
about the PSFs. Nominally, this is called ``pynrc_data`` and can be placed anywhere 
on your local machine. Create an environment variable ``PYNRC_PATH`` in your shell 
startup file, such as ``.bashrc``:

- ``export PYNRC_PATH="/Users/Bob/pynrc_data/"``

## Requirements

+ Python 2.7.x (3.x compatability built in, but not well vetted)
+ WebbPSF >=0.5.0 (https://pythonhosted.org/webbpsf/)
+ Astropy >=1.2.0 (http://www.astropy.org/)
+ Pysynphot >=0.9.8.2 (https://pysynphot.readthedocs.io)