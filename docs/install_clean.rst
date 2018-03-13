.. _clean_install:

===================================
Install from New Conda Environment
===================================

This installation tutorial assumes a clean installation with 
Anaconda and has been verified on both Python 2.7 and 3.6 using 
the following modules:

* Numpy 1.14   
* Matplotlib 2.1
* Scipy 1.0    
* Astropy 2.0


.. _configure_astroconda_channel:

Configure Conda to use the AstroConda Channel
=============================================

We will be install a few packages that live in AstroConda. 
If you're already working in an AstroConda environment, 
then you should be all set and can probably skip this step.

If you have some other Conda, installation, then you can simply 
add the AstroConda channel to your ``.condarc`` file, which appends 
the appropriate URL to Conda's channel search path::

    $ conda config --add channels http://ssb.stsci.edu/astroconda
    # Writes changes to ~/.condarc


.. _install_pysynphot:

Installing Pysynphot
====================

With the AstroConda channel added, it's a simple matter to run::

    $ conda install pysynphot

Otherwise, install the
`standalone <https://github.com/spacetelescope/pysynphot/releases>`_ release::

    $ pip install git+https://github.com/spacetelescope/pysynphot.git@0.9.8.8

Pysynphot Data Files
--------------------

Data files for Pysynphot are distributed through the
`Calibration Reference Data System <http://www.stsci.edu/hst/observatory/crds/throughput.html>`_. 
They are expected to follow a certain directory structure under the root
directory, identified by the ``PYSYN_CDBS`` environment variable that *must* be
set prior to using this package.

1. Download the following file: 
   `cdbs.tar.gz <http://mips.as.arizona.edu/~jleisenring/pynrc/cdbs.tar.gz>`_  [approx. 760 MB]
2. Untar into a directory of your choosing.
3. Set the environment variable ``PYSYN_CDBS`` to point to that directory. 
   For example, in .bashrc shell file, add::

    export PYSYN_CDBS='$HOME/data/cdbs/'

You should now be able to successfully ``import pysynphot`` in a Python session.


.. _install_webbpsf:

Installing WebbPSF
====================

The AstroConda copy of WebbPSF has a ``webbpsf-data`` installation dependency, 
which we do not want in our slightly customized installation, because the WebbPSF 
data files get downloaded separately. Instead, we will do this in two parts to 
first install the rest of the dependencies first, then WebbPSF with the 
``--no-deps`` flag::

    $ conda install jwxml poppy
    $ conda install webbpsf --no-deps

For other installation methods see the `WebbPSF documentation <https://webbpsf.readthedocs.io>`_.

.. caution::
    A note about backends.
    
    In many cases ``matplotlib`` crashes when using the default backend (at least
    on Mac OS X and certain Linux distributions). 
    Given the propensity for these crashes, it may be preferable to 
    use a different graphics backend such as ``TkAgg``. This can either be
    accomplished by setting ``matplotlib.use("TkAgg")`` after
    importing ``matplotlib`` or setting the default backend via your 
    `matplotlibrc file <https://matplotlib.org/users/customizing.html#the-matplotlibrc-file`.
    The latter option is probably preferred for most cases.


WebbPSF Data Files
--------------------------

For the user's convenience, WebbPSF data files can be found here: 
`webbpsf-data-0.6.0.tar.gz <http://mips.as.arizona.edu/~jleisenring/pynrc/webbpsf-data-0.6.0.tar.gz>`_  [approx. 240 MB]
Follow the same procedure as with the Pysynphot data files, 
setting the ``WEBBPSF_PATH`` environment variable to point 
towards your ``webbpsf-data`` directory.


.. _install_pynrc_clean:

Installing pyNRC
====================

To get the most up to date version of ``pynrc``, install directly 
from source, though stability is not guarenteed. The 
`development version <https://github.com/JarronL/pynrc>`_ 
can be found on GitHub.

In this case, you will need to clone the git repository::

    $ git clone https://github.com/JarronL/pynrc

Then install the package with::

    $ cd pynrc
    $ pip install .
    
For development purposes::

    $ cd pynrc
    $ pip install -e .

in order to create editable installations. This is great for helping
to develop the code, create bug reports, pull requests to GitHub, etc.


pyNRC Data Files
--------------------------

Similarly, ``pynrc`` comes with its own set of data files, such as 
instrument throughputs, SCA biases and darks, stellar models, 
and exoplanet models. To run ``pynrc``, you must download these 
files and define the ``PYNRC_PATH`` environment variable. This is
also the location that PSF coefficients will be saved to during
normal operations of ``pynrc``.

1. Download the following file: 
   `pynrc_data_v0.6.1.tar.gz <http://mips.as.arizona.edu/~jleisenring/pynrc/pynrc_data_v0.6.1.tar.gz>`_  [approx. 2.3 GB]
2. Untar into a directory of your choosing.
3. Set the environment variable ``PYNRC_PATH`` to point to that directory. 
   For example, in .bashrc shell file, add::

    export PYNRC_PATH='$HOME/data/pynrc_data'

   You will probably want to add this to your ``.bashrc``.

You should now be able to successfully ``import pynrc`` in a Python session.

