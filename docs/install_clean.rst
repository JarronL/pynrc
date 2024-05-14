===================================
Install with new Conda Environment
===================================

This installation tutorial assumes a clean installation with Anaconda via:

.. code-block:: sh

    $ conda create -n py39 python=3.9 anaconda

and has been verified on Python 3.9 using the following modules:

* Numpy 1.20
* Matplotlib 3.5
* Scipy 1.7
* Astropy 5.0
* Astroquery 0.4.3

-------------------------------------

.. _configure_astroconda_channel:

Configure Conda Channels
========================

We will first install a few packages that live in the AstroConda and Conda-Forge channels. If you're already working in an AstroConda environment, then you should be all set and can probably skip most of these steps and jump to :ref:`install_wpsf_ext`.

If you have some other Conda installation, such as indicated above, then you can simply add the AstroConda and Conda-Forge channels to your ``.condarc`` file, which appends the appropriate URL to Conda's channel search path:

.. code-block:: sh

    # Writes changes to ~/.condarc
    $ conda config --append channels https://ssb.stsci.edu/astroconda
    $ conda config --append channels conda-forge

Now your ``.condarc`` file should look something like the following:

.. code-block:: sh

    channels:
      - defaults
      - https://ssb.stsci.edu/astroconda
      - conda-forge

-------------------------------------

.. _install_synphot:

Installing **stsynphot** and **synphot**
=========================================

From PyPi:

.. code-block:: sh

    $ pip install synphot
    $ pip install stsynphot

Data files for Synphot are distributed through the
`Calibration Reference Data System <https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools>`_. 
They are expected to follow a certain directory structure under the root
directory, identified by the ``PYSYN_CDBS`` environment variable that *must* be
set prior to using this package.

1. Download 
   `cdbs.tar.gz <https://arizona.box.com/shared/static/cbkxlwvokml7n1gref8nw3neg98kzcwn.gz>`_ [approx. 900 MB] to obtain the full set of Synphot data files. Or download a minimum subset of files `here <https://arizona.box.com/shared/static/wgq7ymqsp8e7jfno8yk6o2igbnqlad8z.zip>`_ [approx 50 MB].
2. Untar into a directory of your choosing.
3. Set the environment variable ``PYSYN_CDBS`` to point to that directory. 
   For example, in .bashrc shell file, add:

   .. code-block:: sh

       export PYSYN_CDBS='$HOME/data/cdbs/'

You should now be able to successfully ``import synphot`` and ``import stsynphot`` in a Python session.

-------------------------------------

.. _install_webbpsf:

Installing WebbPSF
====================

The easiest way to install WebbPSF without inducing package conflicts is to install some of its main dependencies, then WebbPSF using the ``--no-deps`` flag. In this particular example, we use a combination of ``conda`` and ``pip``, because of minor issues installing ``photutils`` dependencies. 

.. code-block:: sh

    $ pip install photutils 
    $ pip install pysiaf poppy
    $ pip install webbpsf

This will install WebbPSF without installing its dependencies, which should mostly be already installed. 

WebbPSF Data Files
--------------------------

You will also need to download and install `WebbPSF data files 
<https://webbpsf.readthedocs.io/en/stable/installation.html#installing-the-required-data-files>`_  [approx. 70 MB]. Follow the same procedure as with the **stsynphot** data files, setting the ``WEBBPSF_PATH`` environment variable to point towards your ``webbpsf-data`` directory.


Matplotlib Backends
--------------------------

In many cases ``matplotlib`` crashes when using the default backend (at least on Mac OS X and certain Linux distributions). Given the propensity for these crashes, it may be preferable to use a different graphics backend such as ``TkAgg``. This can either be accomplished by setting ``matplotlib.use("TkAgg")`` after importing ``matplotlib`` or setting the default backend via your `matplotlibrc file <https://matplotlib.org/stable/tutorials/introductory/customizing.html#customizing-with-matplotlibrc-files>`_. The latter option is probably preferred for most cases.

-------------------------------------

.. _install_jwb_clean:

Installing JWST Backgrounds
============================

``jwst_bakcgrounds`` is a a simple program to predict the levels of background emission in JWST observations. It accesses a precompiled background cache prepared by STScI, requiring an internet connection to access. However, ``pynrc`` comes with a simpler background estimator in the event ``jwst_background`` is not installed or no functioning internet. In this sense, ``jwst_backgrounds`` is not a strict requirement for running ``pynrc``.

This module requires ``healpy`` to run:

.. code-block:: sh

    $ conda install healpy
    
Then install JWST Backgrounds with pip:

.. code-block:: sh

    $ pip install jwst_backgrounds

-------------------------------------

.. _install_astroquery:

Installing Astroquery
============================

Astroquery is a set of tools for querying astronomical web forms and databases. It is used within ``pynrc`` to query Simbad and Gaia databases to search for sources and obtain basic astrometry, fluxes, and spectral types.

From PyPi:

.. code-block:: sh

    $ pip install astroquery

-------------------------------------

.. _install_pipeline:

Installing JWST Pipeline
========================

In order to create DMS-like datasets, pyNRC uses data models from the JWST pipeline (https://github.com/spacetelescope/jwst). Again, easiest to install via ``pip``:

.. code-block:: sh

    $ pip install jwst

The JWST pipeline is under significant development, so it's a good idea to keep this up-to-date with new releases by regularly running: 

.. code-block:: sh

    $ pip install jwst --upgrade

CRDS Data Files
---------------

Configure the calibration reference database (CRDS) by defining the CRDS directory that will store downloaded cal files. For example, in ``.bashrc`` shell file:

.. code-block:: sh

    export CRDS_PATH='$HOME/data/crds_cache/'
    export CRDS_SERVER_URL='https://jwst-crds.stsci.edu'

-------------------------------------

.. _install_wpsf_ext:

Installing WebbPSF Extensions
=============================

The ``webbpsf_ext`` package calculates and stores polynomial relationships between PSFs with respect to wavelength, focal plane position, and WFE drift in order to quickly generate arbitrary NIRCam PSFs without having to simulate a new PSF on the fly.  

.. code-block:: sh

    pip install webbpsf_ext

Set the environment variable ``WEBBPSF_EXT_PATH`` to point to some data directory. All PSF coefficients will be saved here as they are generated to be reused later. For example, in ``.bashrc`` shell file, add:

.. code-block:: sh

   export WEBBPSF_EXT_PATH='$HOME/data/webbpsf_ext_data/'

-------------------------------------

.. _install_pynrc_clean:

Installing pyNRC
====================

Finally, we are ready to install ``pynrc``!

Installing with pip
--------------------

You can install the ``pynrc`` package through pip:

.. code-block:: sh

    $ pip install pynrc

Note that the pip command only installs the program code.
You still must download and install the data files, 
as described below.

Installing from source
----------------------

To get the most up to date version of ``pynrc``, install directly from source, though stability is not guaranteed. The `development version <https://github.com/JarronL/pynrc/tree/develop>`_ can be found on GitHub.

In this case, you will need to clone the git repository:

.. code-block:: sh

    $ git clone https://github.com/JarronL/pynrc

Then install the package with:

.. code-block:: sh

    $ cd pynrc
    $ pip install .
    
For development purposes:

.. code-block:: sh

    $ cd pynrc
    $ pip install -e .

This creates an editable installation, which is great for helping to develop the code, create bug reports, pull requests to GitHub, etc. Make sure to switch to the ``develop`` branch after installation in order to get access to the latest code base.

pyNRC Data Files
--------------------------

Similarly, ``pynrc`` comes with its own set of data files, such as 
instrument throughputs, SCA biases and darks, stellar models, 
and exoplanet models. To run ``pynrc``, you must download these 
files and define the ``PYNRC_PATH`` environment variable. This is
also the location that PSF coefficients will be saved to during
normal operations of ``pynrc``.

Files containing information such as the instrument throughputs, stellar models, and exoplanet models are already distributed through ``webbpsf_ext``. 
In addition, ``pynrc`` requires a number of files to simulate realistic detector data with DMS-like formatting and headers. In general, these are not necessary to run ``pynrc``'s ETC capabilities and simple simulations. 
But, in order to create DMS and pipeline-compliant data, you must download these files and define the ``PYNRC_PATH`` environment variable. 

1. Download the following file: 
   `pynrc_data_all_v1.0.0.tar <http://mips.as.arizona.edu/~jleisenring/pynrc/pynrc_data_all_v1.0.0.tar>`_  [approx. 17.0 GB]
2. Untar into a directory of your choosing.
3. Set the environment variable ``PYNRC_PATH`` to point to that directory. 
   For example, in .bashrc shell file, add:

   .. code-block:: sh

       export PYNRC_PATH='$HOME/data/pynrc_data'

You should now be able to successfully ``import pynrc`` in a Python session.

Environment Variables
=============================

In the end, you should have a number of environment variables in your ``.bashrc`` (or equivalnet):

.. code-block:: sh

   export CRDS_PATH='$HOME/data/crds_cache/'
   export CRDS_SERVER_URL='https://jwst-crds.stsci.edu'
   export PYSYN_CDBS='$HOME/data/cdbs/'
   export WEBBPSF_PATH='$HOME/data/webbpsf-data/'
   export WEBBPSF_EXT_PATH='$HOME/data/webbpsf_ext_data/'
   export PYNRC_DATA='$HOME/data/pynrc_data/'