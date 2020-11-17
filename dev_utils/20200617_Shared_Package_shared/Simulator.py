# because the function move_global_zernikes does not have the delay_update parameter anymore need to add ote.update_opd
# however the latest version of poppy creates the following error: module 'poppy' has no attribute 'opd_from_zernikes'
# problem is the anaconda environment does not support pynrc => creating a class for machine learning to by-pass those problems, will need to solve this later
# Simulator_V2.py

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#import webbpsf
#import pysynphot
from astropy.io import fits
import sys, time, os

os.environ['WEBBPSF_PATH'] = "/Users/mygouf/Python/webbpsf/webbpsf-data4/"
os.environ['PYSYN_CDBS'] = "/Users/mygouf/git/pynrc/cdbs/"

import pynrc
from pynrc import nrc_utils
from pynrc.nrc_utils import S
from pynrc.obs_nircam import model_to_hdulist
pynrc.setup_logging('WARNING', verbose=False)

import json
import webbpsf
import poppy
#from Configuration import Target, Observatory, NIRCam_pynrc, VISIR_poppy
from datetime import date

class Simulation(object):

    def __init__(self, dict_simu = None):
        self.dict_simu = dict_simu
        #self.json_file = dict_simu['json_file']
        #self.wfe_budget = dict_simu['wfe_budget']
        #self.noise = dict_simu['noise']
        self.fov_arcsec = dict_simu['fov_arcsec']
        
        #with open(json_file) as json_data_file:
        #    self.observation_parameters = json.load(json_data_file)

        seed = 1234
        self.rs = np.random.RandomState(seed)

    def generate_coefficients(self,wfe_budget,weight=1):
        coefficients = []
        for term in wfe_budget:
            coefficients.append(
                                # convert nm to meters, get value in range
                                #np.random.uniform(low=-1e-9 * term, high=1e-9 * term)*weight
                                # convert nm to microns, get value in range
                                self.rs.uniform(low=-1e-6 * term, high=1e-6 * term)*weight
                                )
        return coefficients

    def create_image(self, coefficient_set_init=None, input_noise=None):
        # This is the function that creates the image (will probably call the config file) from some zernike coefficients
        # Copy paste the code that creates the image here
        
        import poppy
        from astropy.io import fits

        pupil_diameter = 6.559 # (in meter) As used in WebbPSF
        pupil_radius = pupil_diameter/2
        
        osys = poppy.OpticalSystem()

        transmission = '/Users/mygouf/Python/webbpsf/webbpsf-data4/jwst_pupil_RevW_npix1024.fits.gz'
        #opd = '/Users/mygouf/Python/webbpsf/webbpsf-data/NIRCam/OPD/OPD_RevV_nircam_115.fits'
        opd = '/Users/mygouf/Python/webbpsf/webbpsf-data4/NIRCam/OPD/OPD_RevW_ote_for_NIRCam_requirements.fits.gz'
        hdul = fits.open(opd)
        hdul2 = fits.open(transmission)
        
        # Create wavefront map
        #print(coefficient_set_init)
        zernike_coefficients = np.append(0,coefficient_set_init)
        #zernike_coefficients *= 1e6 # conversion from meters to microns
        #wavefront_map = poppy.ZernikeWFE(radius=pupil_radius,
        #                                     coefficients=zernike_coefficients,
        #                                     aperture_stop=False)
        #print(zernike_coefficients)
        wavefront_map = poppy.zernike.opd_from_zernikes(zernike_coefficients,
                                               npix=1024, basis=poppy.zernike.zernike_basis_faster)
        wavefront_map = np.nan_to_num(wavefront_map)*hdul2[0].data
        
        fits.writeto('wavefront_map.fits',wavefront_map,hdul[0].header,overwrite=True)
        #opd = wavefront_map
        opd = 'wavefront_map.fits'
        
        #myoptic = poppy.FITSOpticalElement(transmission='transfile.fits', opd='opdfile.fits', pupilscale="PIXELSCL")
        #opd = '/Users/mygouf/Python/webbpsf/webbpsf-data4/NIRCam/OPD/OPD_RevW_ote_for_NIRCam_requirements.fits.gz'

        jwst_opd = poppy.FITSOpticalElement(transmission=transmission, opd=opd)
        #jwst_opd = poppy.FITSOpticalElement(transmission=transmission)
        
        osys.add_pupil(jwst_opd)    # JWST pupil
        osys.add_detector(pixelscale=0.063, fov_arcsec=self.fov_arcsec, oversample=4)  # image plane coordinates in arcseconds

        psf = osys.calc_psf(4.44e-6)                            # wavelength in microns
        psf_poppy = np.array(psf[0].data)
        poppy.display_psf(psf, title='JWST NIRCam test')

        #psf1 = osys.calc_psf(4.44e-6)                            # wavelength in microns
        #psf2 = osys.calc_psf(2.50e-6)                            # wavelength in microns
        #psf_poppy = psf1[0].data/2 + psf2[0].data/2 
        
        psf_poppy = psf_poppy*1e7/np.max(psf_poppy)
        # Adding photon noise
        #image = np.random.normal(loc=psf_poppy, scale=np.sqrt(psf_poppy))
        if np.all(input_noise) == None:
            #noise =  np.random.normal(loc=psf_poppy, scale=np.sqrt(psf_poppy>1000))
            #noise =  self.rs.normal(loc=psf_poppy, scale=np.sqrt(psf_poppy>np.max(psf_poppy)/1000)) 
            noise =  np.random.normal(loc=psf_poppy, scale=np.sqrt(psf_poppy>np.max(psf_poppy)/1000))
            #noise = rs.poisson(psf_poppy)
            #print('Estimated Noise', np.mean(noise))
        else:
            noise = input_noise
            #print('Input Noise', np.mean(noise))
            
        image = psf_poppy + noise

        dict_ = {'image': image, 'noise': noise, 'wavefront_map':wavefront_map}
        #print(np.mean(image),np.mean(noise))
        
        return dict_

    def create_image2(self, coefficient_set_init=None, input_noise=None):
        # This is the function that creates the image (will probably call the config file) from some zernike coefficients
        # Copy paste the code that creates the image here
        
        import poppy
        from astropy.io import fits

        pupil_diameter = 6.559 # (in meter) As used in WebbPSF
        pupil_radius = pupil_diameter/2
        
        osys = poppy.OpticalSystem()

        transmission = '/Users/mygouf/Python/webbpsf/webbpsf-data4/jwst_pupil_RevW_npix1024.fits.gz'
        #opd = '/Users/mygouf/Python/webbpsf/webbpsf-data/NIRCam/OPD/OPD_RevV_nircam_115.fits'
        opd = '/Users/mygouf/Python/webbpsf/webbpsf-data4/NIRCam/OPD/OPD_RevW_ote_for_NIRCam_requirements.fits.gz'
        hdul = fits.open(opd)
        hdul2 = fits.open(transmission)
        
        # Create wavefront map
        #print(coefficient_set_init)
        zernike_coefficients = np.append(0,coefficient_set_init)
        #zernike_coefficients *= 1e6 # conversion from meters to microns
        #wavefront_map = poppy.ZernikeWFE(radius=pupil_radius,
        #                                     coefficients=zernike_coefficients,
        #                                     aperture_stop=False)
        #print(zernike_coefficients)
        wavefront_map = poppy.zernike.opd_from_zernikes(zernike_coefficients,
                                               npix=1024, basis=poppy.zernike.zernike_basis_faster)
        wavefront_map = np.nan_to_num(wavefront_map)*hdul2[0].data
        
        fits.writeto('wavefront_map.fits',wavefront_map,hdul[0].header,overwrite=True)
        #opd = wavefront_map
        opd = 'wavefront_map.fits'
        
        #myoptic = poppy.FITSOpticalElement(transmission='transfile.fits', opd='opdfile.fits', pupilscale="PIXELSCL")
        #opd = '/Users/mygouf/Python/webbpsf/webbpsf-data4/NIRCam/OPD/OPD_RevW_ote_for_NIRCam_requirements.fits.gz'

        jwst_opd = poppy.FITSOpticalElement(transmission=transmission, opd=opd)
        #jwst_opd = poppy.FITSOpticalElement(transmission=transmission)
        
        osys.add_pupil(jwst_opd)    # JWST pupil
        osys.add_detector(pixelscale=0.063, fov_arcsec=self.fov_arcsec, oversample=4)  # image plane coordinates in arcseconds

        psf = osys.calc_psf(4.44e-6)                            # wavelength in microns
        psf_poppy = np.array(psf[0].data)
        poppy.display_psf(psf, title='JWST NIRCam test')

        #psf1 = osys.calc_psf(4.44e-6)                            # wavelength in microns
        #psf2 = osys.calc_psf(2.50e-6)                            # wavelength in microns
        #psf_poppy = psf1[0].data/2 + psf2[0].data/2 
        
        psf_poppy = psf_poppy*1e7/np.max(psf_poppy)
        # Adding photon noise
        #image = np.random.normal(loc=psf_poppy, scale=np.sqrt(psf_poppy))
        if np.all(input_noise) == None:
            #noise =  np.random.normal(loc=psf_poppy, scale=np.sqrt(psf_poppy>1000))
            #noise =  self.rs.normal(loc=psf_poppy, scale=np.sqrt(psf_poppy>np.max(psf_poppy)/1000)) 
            noise =  np.random.normal(loc=psf_poppy, scale=np.sqrt(psf_poppy>np.max(psf_poppy)/1000))
            #noise = rs.poisson(psf_poppy)
            #print('Estimated Noise', np.mean(noise))
        else:
            noise = input_noise
            #print('Input Noise', np.mean(noise))
            
        image = (np.ndarray.flatten(psf_poppy + noise))

        dict_ = {'image': image, 'noise': noise, 'wavefront_map':wavefront_map}
        #print(np.mean(image),np.mean(noise))
        
        return dict_

    
    def create_image_from_opd_file(self, opd=None, input_noise=None):
        # This is the function that creates the image (will probably call the config file) from some zernike coefficients
        # Copy paste the code that creates the image here
        
        import poppy
        from astropy.io import fits

        pupil_diameter = 6.559 # (in meter) As used in WebbPSF
        pupil_radius = pupil_diameter/2
        
        osys = poppy.OpticalSystem()

        transmission = '/Users/mygouf/Python/webbpsf/webbpsf-data4/jwst_pupil_RevW_npix1024.fits.gz'
        #opd = '/Users/mygouf/Python/webbpsf/webbpsf-data/NIRCam/OPD/OPD_RevV_nircam_115.fits'
        #opd = '/Users/mygouf/Python/webbpsf/webbpsf-data4/NIRCam/OPD/OPD_RevW_ote_for_NIRCam_requirements.fits.gz'
        hdul = fits.open(opd)
        hdul2 = fits.open(transmission)

        wavefront_map = hdul[0].data
        
        jwst_opd = poppy.FITSOpticalElement(transmission=transmission, opd=opd)
        #jwst_opd = poppy.FITSOpticalElement(transmission=transmission)
        
        osys.add_pupil(jwst_opd)    # JWST pupil
        osys.add_detector(pixelscale=0.063, fov_arcsec=self.fov_arcsec, oversample=4)  # image plane coordinates in arcseconds

        psf = osys.calc_psf(4.44e-6)                            # wavelength in microns
        psf_poppy = np.array(psf[0].data)
        poppy.display_psf(psf, title='JWST NIRCam test')

        #psf1 = osys.calc_psf(4.44e-6)                            # wavelength in microns
        #psf2 = osys.calc_psf(2.50e-6)                            # wavelength in microns
        #psf_poppy = psf1[0].data/2 + psf2[0].data/2 
        
        psf_poppy = psf_poppy*1e7/np.max(psf_poppy)
        # Adding photon noise
        #image = np.random.normal(loc=psf_poppy, scale=np.sqrt(psf_poppy))
        if np.all(input_noise) == None:
            #noise =  np.random.normal(loc=psf_poppy, scale=np.sqrt(psf_poppy>1000))
            #noise =  self.rs.normal(loc=psf_poppy, scale=np.sqrt(psf_poppy>np.max(psf_poppy)/1000)) 
            noise =  np.random.normal(loc=psf_poppy, scale=np.sqrt(psf_poppy>np.max(psf_poppy)/1000))
            #noise = rs.poisson(psf_poppy)
            #print('Estimated Noise', np.mean(noise))
        else:
            noise = input_noise
            #print('Input Noise', np.mean(noise))
            
        image = psf_poppy + noise

        dict_ = {'image': image, 'noise': noise, 'wavefront_map':wavefront_map}
        #print(np.mean(image),np.mean(noise))
        
        return dict_
    

    def generate_psfs(self,):
        
        from datetime import date
        today = date.today()
        date = str(today)
        print(today)

        outdir  = '/Users/mygouf/git/0-Python_Notebooks/20190726_JWST_PP_Pipeline/Tests/'+date+'/poppy/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
#        dict_target = self.observation_parameters['target']
#        dict_companions = self.observation_parameters['target']['companions']
#        dict_reference = self.observation_parameters['reference']
#        dict_instrument = self.observation_parameters['instrument']
#        dict_observatory = self.observation_parameters['observatory']
#        
##        target = Target()
##        target.from_dict(dict_target)
##        companions =
##        reference = Target()
##        reference.from_dict(dict_reference)
##        observatory = Observatory()
##        observatory.from_dict(dict_observatory)
##        instrument = VISIR_poppy()
##        instrument.from_dict(dict_instrument)
#
#        dist = distance = dict_target['distance']
#        #mag_alpha_cen_A= dict_target['mag_alpha_cen_A']
#        sptype_alpha_cen_A= dict_target['spType']
#        
#        sep= dict_companions[0]['sep']
#        PA = dict_companions[0]['PA']
#        #mag_alpha_cen_B= dict_companions['mag_alpha_cen_B']
#        sptype_alpha_cen_B= dict_companions[0]['spType']
#        
#        pupil_diameter= dict_observatory['diameter']
#                                            
#        nb_of_images= dict_instrument['nb_of_images']
#        norm_alpha_cen_a= dict_instrument['norm_alpha_cen_a']
#        norm_alpha_cen_b= dict_instrument['norm_alpha_cen_b']
#        px_scale_x= dict_instrument['px_scale_x']
#        px_scale_y= dict_instrument['px_scale_y']
#        size_x= dict_instrument['size_x']
#        size_y= dict_instrument['size_y']
#        tint= dict_instrument['tint']
#        total_tint= dict_instrument['total_tint']
#        wavelength= dict_instrument['wavelength']
#        delta_PA= dict_instrument['delta_PA']
#
#        size = size_x
#        px_scale = px_scale_x
#        fov_arcsec = size*px_scale
#        norm_alpha_cen_a_count_per_second =  norm_alpha_cen_a/tint
#        norm_alpha_cen_b_count_per_second = norm_alpha_cen_b/tint
#        pupil_radius = pupil_diameter/2

        coefficient_set_init = self.generate_coefficients(self.dict_simu['wfe_budget'])
        print(coefficient_set_init)

        dict_ = self.create_image(coefficient_set_init[1:len(self.dict_simu['wfe_budget'])])
        image = dict_['image']

        #fig, axes = plt.subplots(1,1, figsize=(13,4.5))
        #ax1 = axes
        #ax1.imshow(image)
        #plt.show()

        print(len(self.dict_simu['wfe_budget']))

        return dict_, coefficient_set_init


