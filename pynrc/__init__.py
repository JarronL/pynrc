"""
pyNRC - Python ETC and Simulator for JWST NIRCam
----------------------------------------------------------------------------

pyNRC is a set of Python-based tools for planning observations with JWST NIRCam, 
such as an ETC, a simple image slope simulator, and an enhanced data simulator.
This package works for a variety of NIRCam observing modes including direct imaging, 
coronagraphic imaging, slitless grism spectroscopy, DHS observations, 
and weak lens imaging. All PSFs are generated via webbpsf (https://webbpsf.readthedocs.io) 
and webbpsf_ext (https://github.com/JarronL/webbpsf_ext) to reproduce realistic JWST 
images and spectra.

Developed by Jarron Leisenring and contributors at University of Arizona (2015 - 2024).
"""
from .version import __version__

import astropy
from astropy import config as _config

import tempfile

class Conf(_config.ConfigNamespace):

    # Path to data files for pynrc. 
    # The environment variable $PYNRC_PATH takes priority.
    import os
    
    on_rtd = os.environ.get('READTHEDOCS') == 'True'
    
    if on_rtd:
        path = tempfile.gettempdir()
    else:
        path = os.getenv('PYNRC_PATH')
        if path is None:
            print("WARNING: Environment variable $PYNRC_PATH is not set!")
            print("  Setting PYNRC_PATH to temporary directory.")
            path = tempfile.gettempdir()
            print("  {}".format(path))
            #raise EnvironmentError("Environment variable $PYNRC_PATH is not set!")
        if not os.path.isdir(path):
            #print ("WARNING: PYNRC_PATH ({}) is not a valid directory path!".format(path))
            raise IOError("PYNRC_PATH ({}) is not a valid directory path!".format(path))
            
    if '/' not in path[-1]: 
        # Make sure there is a '/' at the end of the path name
        path = path + '/'

    PYNRC_PATH = _config.ConfigItem(path, 'Directory path to data files \
                                    required for pynrc calculations.')

    autoconfigure_logging = _config.ConfigItem(
        False,
        'Should pynrc configure logging for itself and others?'
    )
    logging_level = _config.ConfigItem(
        ['INFO', 'DEBUG', 'WARN', 'ERROR', 'CRITICAL', 'NONE'],
        'Desired logging level for pyNRC.'
    )
    default_logging_level = _config.ConfigItem('INFO', 
        'Logging verbosity: one of {DEBUG, INFO, WARN, ERROR, or CRITICAL}')
    logging_filename = _config.ConfigItem("none", "Desired filename to save log messages to.")
    logging_format_screen = _config.ConfigItem(
        '[%(name)10s:%(levelname)s] %(message)s', 'Format for lines logged to the screen.'
    )
    logging_format_file = _config.ConfigItem(
        '%(asctime)s [%(name)s:%(levelname)s] %(filename)s:%(lineno)d: %(message)s',
        'Format for lines logged to a file.'
    )

conf = Conf()

# from webbpsf_ext import robust

from .logging_utils import setup_logging
setup_logging(conf.default_logging_level, verbose=False)

from .nrc_utils import read_filter, bp_2mass, bp_wise, bp_gaia
from .nrc_utils import pix_noise
from .nrc_utils import stellar_spectrum, source_spectrum, planets_sb12

from .pynrc_core import DetectorOps, NIRCam

from .obs_nircam import obs_hci, nrc_hci
from .detops import multiaccum, det_timing, nrc_header

#from .ngNRC import slope_to_ramp, nproc_use_ng

from .maths import coords, robust, fast_poly, image_manip
from .reduce import ref_pixels

from .reduce.calib import nircam_dark, nircam_cal
from .simul.apt import DMS_input
from .simul import ngNRC

from .testing import perform_benchmarks


def _reload(name="pynrc"):
    """
    Simple reload function to test code changes without restarting python.
    There may be some weird consequences and bugs that show up, such as
    functions and attributes deleted from the code still stick around after
    the reload. Although, this is even true with ``importlib.reload(pynrc)``.

    Other possible ways to reload on-the-fly: 
       
    from importlib import reload
    reload(pynrc)

    # Delete classes/modules to reload
    import sys
    del sys.modules['pynrc.obs_nircam'] 
    """
    import imp
    imp.load_module(name,*imp.find_module(name))

    print("{} reloaded".format(name)) 


