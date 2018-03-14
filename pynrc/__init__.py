"""
pyNRC - Python ETC and Simulator for JWST NIRCam
----------------------------------------------------------------------------

pyNRC is a set of Python-based tools for planning observations with JWST NIRCam, 
such as an ETC, a simple image slope simulator, and an enhanced data simulator.
This package works for a variety of NIRCam observing modes including direct imaging, 
coronagraphic imaging, slitless grism spectroscopy, DHS observations, 
and weak lens imaging. All PSFs are generated via WebbPSF 
(https://webbpsf.readthedocs.io) to reproduce realistic JWST images and spectra.

Developed by Jarron Leisenring and contributors at University of Arizona (2015-18).
"""
from __future__ import absolute_import, division, print_function, unicode_literals


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

    logging_level = _config.ConfigItem(
        ['INFO', 'DEBUG', 'WARN', 'WARNING', 'ERROR', 'CRITICAL', 'NONE'],
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


from .logging_utils import setup_logging#, restart_logging
setup_logging(conf.default_logging_level, verbose=False)

from .nrc_utils import (read_filter, pix_noise, nrc_header, bp_2mass, bp_wise, \
                        stellar_spectrum, source_spectrum)

from .pynrc_core import (multiaccum, DetectorOps, NIRCam, planets_sb11, planets_sb12)

from .obs_nircam import (obs_coronagraphy, obs_hci, nrc_hci)

#from .ngNRC import slope_to_ramp, nproc_use_ng

from .maths import *

from .simul import ngNRC

from .reduce import ref_pixels


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


