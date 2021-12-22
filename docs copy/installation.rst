************
Installation
************

.. _installation:

Requirements
============

pyNRC requires Python 3.7+ along with the following packages:

* Recent version of `Numpy <http://www.numpy.org>`_, `Scipy <https://www.scipy.org>`_, and `matplotlib <https://www.matplotlib.org>`_
* `Astropy <http://www.astropy.org>`_ 4.2+
* `pysynphot <https://pysynphot.readthedocs.io>`_ 2.0.0+
* `WebbPSF <https://webbpsf.readthedocs.io>`_ 1.0.0+ 
  and its dependencies.
  
**Recommended Python packages**:
  
* `jwst_backgrounds <https://github.com/spacetelescope/jwst_backgrounds>`_ 1.1.2+
* `psutil <https://pypi.python.org/pypi/psutil>`_

.. _install_conda:

Installing with conda
=====================

.. todo::

  **Not yet implemented**

  pyNRC can be installed with `conda <https://docs.conda.io/en/latest/>`_ if you have installed `Anaconda <https://www.anaconda.com/products/individual>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_. To install pyNRC using the `conda-forge Anaconda channel <https://anaconda.org/conda-forge/pynrc>`_, simply add ``-c conda-forge`` to the install command::
    
        $ conda install -c conda-forge pynrc


.. _install_pip:

Installing with pip
===================

You can install the ``pynrc`` package through pip::

    $ pip install pynrc

If you want to make sure that none of your existing dependencies get
upgraded, instead you can do:

.. code-block:: sh

    $ pip install pynrc --no-deps

Note that the pip command only installs the program code.
You still must download and install the data files, 
as :ref:`described below <data_install>`.


.. _install_dev_version:

Installing from source
----------------------

To get the most up to date version of ``pynrc``, install directly 
from source, though stability is not guarenteed. The 
`development version <https://github.com/JarronL/pynrc>`_ 
can be found on GitHub.

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

in order to create editable installations. This is great for helping
to develop the code, create bug reports, pull requests to GitHub, etc.


.. _data_install:

Installing the data files
--------------------------

Files containing such information as the instrument throughputs, 
SCA biases and darks, stellar models, and exoplanet models are 
distributed separately. To run ``pynrc``, you must download these 
files and define the ``PYNRC_PATH`` environment variable.
This is also the location that PSF coefficients will be saved to during
normal operations of ``pynrc``.

1. Download the following file: 
   `pynrc_data_v0.6.1.tar.gz <http://mips.as.arizona.edu/~jleisenring/pynrc/pynrc_data_v0.6.1.tar.gz>`_  [approx. 2.3 GB]
2. Untar into a directory of your choosing.
3. Set the environment variable ``PYNRC_PATH`` to point to that directory. 
   For bash, for example::

    $ export PYNRC_PATH=$HOME/data/pynrc_data

   You will probably want to add this to your ``.bashrc``.

You should now be able to successfully ``import pynrc`` in a Python session.

Testing
--------

.. todo::

   **Not yet implemented**

    If you want to check that all the tests are running correctly with your Python
    configuration, you can also run::

        $ python setup.py test

    in the source directory. If there are no errors, you are good to go!    