Installation
============

.. _installation:

Requirements
------------

pyNRC requires Python 2.7+ or 3.5+ along with the following packages:

* Recent version of `NumPy <http://www.numpy.org>`_, 
  `SciPy <https://www.scipy.org>`_,
  and `matplotlib <https://www.matplotlib.org>`_
* `Astropy <http://www.astropy.org>`_ 1.2.0+
* `pysynphot <https://pysynphot.readthedocs.io>`_ 0.9.8+
* `WebbPSF <https://webbpsf.readthedocs.io>`_ 0.6.0+ 
  and its dependencies.
* `psutil <https://pypi.python.org/pypi/psutil>`_


Installing with conda
---------------------

.. todo::

    pyNRC has been placed on conda-forge so you can manage the package
    through your Conda installation. Simply add conda-forge to your 
    ``.condarc`` file, which appends the appropriate URL to Conda's 
    channel search path::
    
        $ conda config --add channels conda-forge 

    With the conda-forge channel added, it's a simple matter to run::

        $ conda install pynrc

Installing with pip
--------------------

You can install the ``pynrc`` package through pip::

    $ pip install pynrc

Note that the pip command only installs the program code.
You still must download and install the data files, 
as :ref:`described below <data_install>`.


Installing from source
----------------------

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

    If you want to check that all the tests are running correctly with your Python
    configuration, you can also run::

        $ python setup.py test

    in the source directory. If there are no errors, you are good to go!    