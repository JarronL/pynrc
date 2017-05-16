"""
pyNRC - Python ETC and Simulator for JWST NIRCam
----------------------------------------------------------------------------

pyNRC contains a number of tools for planning observations with JWST NIRCam,
such as an ETC and simple image simulator. pyNRC is instended to work for 
all NIRCam observing modes including direct imaging, coronagraphic imaging, 
slitless grism spectroscopy, DHS observations, and weak lens imaging.
All PSFs are generated via WebbPSF (https://pythonhosted.org/webbpsf/) to
reproduce realistic JWST images and spectra.

Developed by Jarron Leisenring and contributors at University of AZ (2015-17).
"""


__version__ = u'0.3'
#__author__ = 'Jarron Leisenring'
#__email__  = 'jarronl@email.arizona.edu'

import astropy
from astropy import config as _config

class Conf(_config.ConfigNamespace):

    # Path to data files for pynrc. 
    # The environment variable $PYNRC_PATH takes priority.
    import os
    path = os.getenv('PYNRC_PATH')
    if path is None:
        #export PYNRC_PATH='/Volumes/NIRData/pynrc_data'
        raise EnvironmentError("Environment variable $PYNRC_PATH is not set!")
    if not os.path.isdir(path):
        raise IOError("PYNRC_PATH ({}) is not a valid directory path!".format(path))
    # Make sure there is a '/' at the end of the path name
    if '/' not in path[-1]: 
        path += '/'
    PYNRC_PATH = _config.ConfigItem(path, 'Directory path to data files required for pynrc calculations.')

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

#from . import logging_utils
from .logging_utils import setup_logging#, restart_logging
setup_logging(conf.default_logging_level, verbose=False)

from .nrc_utils import (read_filter, pix_noise, nrc_header, stellar_spectrum)

from .pynrc_core import (multiaccum, DetectorOps, NIRCam, planets_sb11)

from .obs_nircam import (obs_coronagraphy, nrc_diskobs, observe_disk, observe_star)

#from .ngNRC import slope_to_ramp, nproc_use_ng

from . import robust, ngNRC