# Import libraries
import os

# Default OPD info
# opd_default = ('OPD_RevW_ote_for_NIRCam_predicted.fits', 0)
opd_default = ('JWST_OTE_OPD_RevAA_prelaunch_predicted.fits', 0)
pupil_file  = 'jwst_pupil_RevW_npix1024.fits.gz'

# The following won't work on readthedocs compilation
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if not on_rtd:
    from webbpsf.utils import get_webbpsf_data_path
    from webbpsf_ext.utils import check_fitsgz

    # Set up initial OPD file info
    opd_file = opd_default[0]
    try:
        opd_file = check_fitsgz(opd_file)
    except OSError:
        # Fall back to RevW if cannot find newer version
        opd_file = check_fitsgz('OPD_RevW_ote_for_NIRCam_predicted.fits')
    opd_default = (opd_file, 0)

    # If a NIRCam-specific OPD file, grab from NIRCam OPD directory,
    # otherwise OPD file will be found in main webbpsf-data directory.
    if 'NIRCam' in opd_file:
        opd_dir = os.path.join(get_webbpsf_data_path(),'NIRCam','OPD')
    else:
        opd_dir = get_webbpsf_data_path()

else:
    opd_dir = ''

from webbpsf_ext.opds import OPDFile_to_HDUList, OTE_WFE_Drift_Model, plot_im, plot_opd
