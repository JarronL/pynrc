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
    # Check .fits or .fits.gz?

    opd_file = opd_default[0]
    if 'JWST_OTE_OPD' in opd_file:
        opd_dir = get_webbpsf_data_path()
    else:
        opd_dir = os.path.join(get_webbpsf_data_path(),'NIRCam','OPD')

    opd_fullpath = os.path.join(opd_dir, opd_file)
    if not os.path.exists(opd_fullpath):
        opd_file_alt = opd_file + '.gz'
        opd_path_alt = os.path.join(opd_dir, opd_file_alt)
        if not os.path.exists(opd_path_alt):
            err_msg = f'Cannot find either {opd_file} or {opd_file_alt} in {opd_dir}'
            raise OSError(err_msg)
        else:
            opd_default = (opd_file_alt, 0)
else:
    opd_dir = ''

from webbpsf_ext.opds import OPDFile_to_HDUList, OTE_WFE_Drift_Model, plot_im, plot_opd
