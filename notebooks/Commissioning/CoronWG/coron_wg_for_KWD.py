# Import the usual libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy

from scipy.interpolate import interp1d

from astropy.io import fits

# Progress bar
from tqdm.auto import tqdm, trange
import webbpsf_ext

from webbpsf_ext.image_manip import fourier_imshift, fshift, frebin
from webbpsf_ext.opds import OPDFile_to_HDUList


# Scenario to consider [best, nominal, requirements]
scenarios = ['Best Case', 'Nominal', 'Requirements']
# LOS Jitter [2.5, 3.8, 5.8] per axis
# Samples from random distribution, 1 sample per time step. 
# hdul_jitter = fits.open(opd_dir + 'LOS_JITTER.fits')
jitter_modes = np.array([0, 2.5, 3.8, 5.8])

# Target Acqs
# Three values for best, nominal, and requirements: 6.2568703,  8.759604 , 12.513741
#   - these are per axis
#   - include sqrt(2) relative to sci target
tacq_ref = np.array([0, 6.2568703,  8.759604 , 12.513741])
tacq_ref = tacq_ref.repeat(2).reshape([-1,2])

# Science target acq uncertainty is always set to 0 (perfect pointing)
tacq_sci = 0

# Load OPD and other information
opd_dir = 'NIRCAM_OPDS/'
tvals_sec = fits.getdata(opd_dir + 'time_vector.fits') * 60

# Static OPDs
hdul_opds_static = OPDFile_to_HDUList(opd_dir + 'STATIC_NIRCAM-A_INPUT.fits')

# Thermal, frill, and IEC
hdul_opds_thermal = fits.open(opd_dir + 'TD_NIRCAM.fits')
hdul_opds_frill   = fits.open(opd_dir + 'FRILLCO_NIRCAM.fits')
hdul_opds_iec     = fits.open(opd_dir + 'IEC_NIRCAM.fits')

# Convert OPDS to units of microns
for hdul in [hdul_opds_static, hdul_opds_thermal, hdul_opds_frill, hdul_opds_iec]:
    for hdu in hdul:
        hdu.data *= 1e6
        hdu.header['BUNIT'] = 'micron'


def create_delta_opds(imode, tint, ref_opds=False):
    """
    Generate a series of delta OPDs at given time steps.
    Uses the hdul_opds_thermal, hdul_opds_frill, hdul_opds_iec files.
    """

    tvals_sec = fits.getdata(opd_dir + 'time_vector.fits') * 60

    nint = len(tint)

    # Create drifted OPDs for each integration
    # Interpolate OPDS for each integration
    dopds = np.zeros([nint,1024,1024])
    for hdul in tqdm([hdul_opds_thermal, hdul_opds_frill, hdul_opds_iec], leave=False):
        # Flip along y-axis for correct orientation
        opds = hdul[imode].data[:,::-1,:]

        # Interpolation function for dOPDs w.r.t. time
        func = interp1d(tvals_sec, opds, axis=0, kind='linear', bounds_error=True)

        # Interpolate dOPDs
        # For reference thermal and frill, start at t=0, but flip sign
        if ref_opds and ((hdul is hdul_opds_thermal) or (hdul is hdul_opds_frill)):
            tint0 = tint - tint.min()
            tint0 += (tint[1]-tint[0]) / 2
            opds_interp = -1 * func(tint0)
        else:
            opds_interp = func(tint)

        # Rebin and add to output array
        dopds += frebin(opds_interp, dimensions=1024, total=False)

    return dopds

def create_drifted_opds(imode, tint, ref_opds=False):

    nint = len(tint)
    dopds = create_delta_opds(imode, tint, ref_opds=ref_opds)


    # If reference, initial OPD is last science OPD
    hdul_opds0 = deepcopy(hdul_opds_static)
    if ref_opds:
        # delta time between integrations
        dt = tint[1] - tint[0]
        tint_sci = np.array([tint.min()-dt])
        dopds_sci = create_delta_opds(imode, tint_sci, ref_opds=False)
        hdul_opds0[0].data += dopds_sci[0]

    # Create copy of OPD and add delta
    res = []
    desc='Ref INTs' if ref_opds else 'Sci INTs'
    for i in trange(nint, desc=desc, leave=False):
        hdul_int = deepcopy(hdul_opds0)
        hdul_int[0].data += dopds[i]

        res.append(hdul_int)

    return res


#################################################
# Interpolate OPDS for a given mode and observation setup

# scenario index [0,1, or 2] for [best, nominal, or requirements]
imode = 0

# Define some number of integrations
nint_sci = 100
nint_ref = 100

# Integration time (sec)
t_int = 30.0

# Time step of integration (average) relative to beginning of observation
tint_sci = np.arange(nint_sci) * t_int + t_int / 2
tint_ref = tint_sci.max() + (np.arange(nint_ref) + 1) * t_int

# Create lists of drifted OPDs at time of each integration
dopds_sci = create_drifted_opds(imode, tint_sci, ref_opds=False)
dopds_ref = create_drifted_opds(imode, tint_ref, ref_opds=True)

# Can now update the OPD of any webbpsf or webbpsf_ext class at each integration
inst = webbpsf_ext.NIRCam_ext()
webbpsf_ext.setup_logging('WARN')
psfs_sci = []
for i in trange(nint_sci, desc='Sci INT'):
    inst.pupilopd = dopds_sci[i]
    psf = inst.calc_psf()
    psfs_sci.append(psf)

psfs_ref = []
for i in trange(nint_ref, desc='Ref INT'):
    inst.pupilopd = dopds_ref[i]
    psf = inst.calc_psf()
    psfs_ref.append(psf)

