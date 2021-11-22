# Import libraries
import os

# Default OPD info
opd_default = ('OPD_RevW_ote_for_NIRCam_predicted.fits', 0)

# The following won't work on readthedocs compilation
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if not on_rtd:
    # .fits or .fits.gz?
    import webbpsf
    opd_dir = os.path.join(webbpsf.utils.get_webbpsf_data_path(),'NIRCam','OPD')
    opd_file = opd_default[0]
    opd_fullpath = os.path.join(opd_dir, opd_file)
    if not os.path.exists(opd_fullpath):
        opd_file_alt = opd_file + '.gz'
        opd_path_alt = os.path.join(opd_dir, opd_file_alt)
        if not os.path.exists(opd_path_alt):
            err_msg = f'Cannot find either {opd_file} or {opd_file_alt} in {opd_dir}'
            raise OSError(err_msg)
        else:
            opd_default = (opd_file_alt, 0)

        #import errno
        #raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), opd_file)

from webbpsf_ext.opds import OPDFile_to_HDUList, OTE_WFE_Drift_Model, plot_im, plot_opd