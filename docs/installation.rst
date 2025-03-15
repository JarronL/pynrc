******************
Basic Installation
******************

Requirements
============

pyNRC requires Python 3.10+ along with the following packages:

* Recent version of `Numpy <http://www.numpy.org>`_, `Scipy <https://www.scipy.org>`_, and `matplotlib <https://www.matplotlib.org>`_
* `Astropy <http://www.astropy.org>`_ 5.3+
* `Astroquery <https://astroquery.readthedocs.io/>`_ 0.4.3+
* `synphot <https://synphot.readthedocs.io>`_ 1.2.0+
* `stsynphot <https://stsynphot.readthedocs.io>`_ 1.2.0+
* `STPSF <https://stpsf.readthedocs.io>`_ 2.0.0+
* `WebbPSF Extensions <https://github.com/JarronL/webbpsf_ext>`_ 2.0.0+
* `JWST Pipeline <https://github.com/spacetelescope/jwst>`_ 1.16.0+
  
**Recommended Python packages**:
  
* `jwst_backgrounds <https://github.com/spacetelescope/jwst_backgrounds>`_ 1.3.0+
* `psutil <https://pypi.python.org/pypi/psutil>`_ Library to retrieve information on system utilization and profiling
* `tqdm <https://tqdm.github.io/>`_ Progress bar for for loops


.. _install_pip:

Installing with pip
===================

You can install the ``pynrc`` package through pip:

.. code-block:: sh

    $ pip install pynrc

If you want to make sure that none of your existing dependencies get upgraded, instead you can do (assuming all dependencies are met!):

.. code-block:: sh

    $ pip install pynrc --no-deps


.. _install_dev_version:

Installing from source
======================

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

in order to create editable installations. This is great for helping to develop the code, create bug reports, pull requests to GitHub, etc.


.. _data_install:

Installing the data files
=========================

The above commands only installs the program code. If you're planning on generating DMS simulations, download the full set of data files. In future version, these will be replaced with JWST CRDS files.

Files containing information such as the instrument throughputs, stellar models, and exoplanet models are already distributed through ``webbpsf_ext``. 
In addition, ``pynrc`` requires a number of files to simulate realistic detector data with DMS-like formatting and headers. In general, these are not necessary to run ``pynrc`` and use its ETC capabilities and simple simulations. 
But, in order to create DMS and pipeline-compliant data, you must download these files and define the ``PYNRC_PATH`` environment variable. 

1. Download the following file: 
   `pynrc_data_all.tar <http://mips.as.arizona.edu/~jleisenring/pynrc/pynrc_data_all.tar>`_  [approx. 17.0 GB]
2. Untar into a directory of your choosing.
3. Set the environment variable ``PYNRC_PATH`` to point to that directory. 
   For example, in ``.bashrc`` shell file, add:

   .. code-block:: sh

       $ export PYNRC_PATH=$HOME/data/pynrc_data

You should now be able to successfully ``import pynrc`` in a Python session.


