# from .nrc_utils import *
import numpy as np
from .nrc_utils import webbpsf, poppy, read_filter
from .logging_utils import setup_logging
from .pynrc_core import NIRCam
from .opds import opd_default

def perform_benchmarks(filter='F430M', pupil=None, mask=None, module='A',
                       fov_pix=33, oversample=4, include_si_wfe=True, 
                       include_ote_field_dependence=True, include_distortions=True,
                       use_legendre=True, force=False, save=True, 
                       do_webbpsf=True, do_webbpsf_only=False, return_nrc=False,
                       use_mp=None, nproc=None, **kwargs):
    
    import datetime, time
    
    # PSF setup
#     kwargs = {}
    kwargs['pupil'] = 'CLEAR' if pupil is None else pupil
    kwargs['mask'] = mask
    kwargs['module'] = module

    kwargs['fov_pix'] = fov_pix
    kwargs['oversample'] = oversample

    # kwargs['opd'] = ('OPD_RevW_ote_for_NIRCam_predicted.fits.gz', 0)
    kwargs['opd'] = opd_default
    kwargs['jitter'] = 'gaussian'
    kwargs['jitter_sigma'] = 0.007

    kwargs['include_si_wfe'] = include_si_wfe
    kwargs['include_ote_field_dependence'] = include_ote_field_dependence
    kwargs['include_distortions'] = include_distortions
    kwargs['use_legendre'] = use_legendre

    kwargs['force']     = force
    kwargs['save']      = save
    kwargs['save_name'] = None
    
    setup_logging('WARN', verbose=False)
    
    tdict = {
        'webbpsf_init': None,
        'webbpsf_psf': None,
        'pynrc_coeff': None,
        'pynrc_drift': None,
        'pynrc_field': None,
        'pynrc_psf': None,
    }
    
    # Multiprocessing cases
    use_mp_def = poppy.conf.use_multiprocessing
    nproc_def = poppy.conf.n_processes
    if use_mp is not None:
        poppy.conf.use_multiprocessing = use_mp
    if nproc is not None:
        poppy.conf.n_processes = nproc

    ####
    # WebbPSF timings
    ####
    if do_webbpsf or do_webbpsf_only:
        # Initialization overheads
        tarr = []
        for i in range(5):
            t0 = time.time()
            inst = webbpsf.NIRCam()
            inst.detector = 'NRCA5'
            t1 = time.time()
            tarr.append(t1-t0)
        tdict['webbpsf_init'] = dt = np.mean(tarr)
        time_string = 'Took {:.2f} seconds to init WebbPSF'.format(dt)
        print(time_string)    

        # PSF Generation
        nrc = NIRCam(filter=filter, **kwargs)
        tarr = []
        for i in range(5):
            t0 = time.time()
            _ = nrc.calc_psf()
            t1 = time.time()
            dt = t1-t0
            tarr.append(dt)
        tdict['webbpsf_psf'] = dt = np.mean(tarr)
        time_string = 'Took {:.2f} seconds to generate WebbPSF PSF'.format(dt)
        print(time_string)    

    if do_webbpsf_only and (not return_nrc):
        return tdict
    
    ####
    # pyNRC coefficients
    ####
    t0 = time.time()
    nrc = NIRCam(filter=filter, **kwargs)
    t1 = time.time()
    
    if do_webbpsf_only and return_nrc:
        return nrc

    tdict['pynrc_coeff'] = dt = t1-t0
    time_string = 'Took {:.2f} seconds to generate pyNRC coefficients'.format(dt)
    print(time_string)

    ####
    # WFE drift coefficient generation
    ####
    t0 = time.time()
    nrc.gen_wfedrift_coeff()
    t1 = time.time()
    dt = t1-t0

    tdict['pynrc_drift'] = dt = t1-t0
    time_string = 'Took {:.2f} seconds to generate WFE Drift coefficients'.format(dt)
    print(time_string)

    ####
    # Field coefficient generation
    ####
    t0 = time.time()
    nrc.gen_wfefield_coeff()
    t1 = time.time()

    tdict['pynrc_field'] = dt = t1-t0
    time_string = 'Took {:.2f} seconds to generate WFE Field coefficients'.format(dt)
    print(time_string)
    
    tarr = []
    for i in range(10):
        t0 = time.time()
        
        _ = nrc.calc_psf_from_coeff(wfe_drift=5, coord_vals=(1024,1024), coord_frame='sci', 
                                    return_oversample=True, return_hdul=False)
        t1 = time.time()
        dt = t1-t0
        tarr.append(dt)
        if dt>10:
            break
    tdict['pynrc_psf'] = dt = np.mean(tarr)
    time_string = 'Took {:.2f} seconds to generate pynrc PSF'.format(dt)
    print(time_string)
    
    # Return defaults
    poppy.conf.use_multiprocessing = use_mp_def
    poppy.conf.n_processes = nproc_def
    
    if return_nrc:
        return nrc
    else:
        return tdict
