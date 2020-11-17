

#def write_fits(path,array):
#    from astropy.io import fits
#    hdul = fits.PrimaryHDU(array)
#    hdul.writeto(path,overwrite=True)
#    return

def write_fits(path,array):
    from astropy.io import fits
    opd = '/Users/mygouf/Python/webbpsf/webbpsf-data4/NIRCam/OPD/OPD_RevW_ote_for_NIRCam_requirements.fits.gz'
    hdul = fits.open(opd)
    hdul[0].header['BUNIT']    
    hdu2 = fits.PrimaryHDU(array)
    hdu2.header['BUNIT'] = 'micron'
    fits.writeto(path, np.nan_to_num(hdu2.data*1e6),hdu2.header,overwrite=True)
    return

def display_ote_and_psf(inst,ote, opd_vmax=500, psf_vmax=0.1, title="OPD and PSF", **kwargs):
    import matplotlib.pyplot as plt
    import webbpsf
    psf = inst.calc_psf(monochromatic=2e-6,)
    plt.figure(figsize=(12,8))
    ax1=plt.subplot(121)
    ote.display_opd(ax=ax1, vmax=opd_vmax, 
                    colorbar_orientation='horizontal',
                   title='OPD modified for mirror moves') #, cbpad=0.05)
    ax2=plt.subplot(122)
    webbpsf.display_psf(psf, ext=1, vmax=psf_vmax, vmin=psf_vmax/1e4,
                        colorbar_orientation='horizontal',
                       title="PSF sim, 2 microns") #, cbpad=0.05)
    plt.suptitle(title, fontsize=16)
    
def show_telescope_wfe(instr, ax=None, title=None, ticks=True, **kwargs):
    if ax is None:
        ax=plt.gca()
    osys = instr._getOpticalSystem()
    tel_wfe = osys.planes[0]
    tel_wfe.display(what='opd', ax=ax,
                    colorbar_orientation='vertical',
                   **kwargs)
    if title is None:
        title=tel_wfe.name+" for\n"+instr.name
    ax.set_title(title)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        
def show_inst_wfe(instr, ax=None, **kwargs):
    if ax is None:
        plt.gca()
    osys = instr._getOpticalSystem()
    pupils = [p for p in osys.planes if p.planetype==poppy.poppy_core.PlaneType.pupil]
    inst_wfe = pupils[-1]
    inst_wfe.display(what='opd', ax=ax, 
                     colorbar_orientation='vertical', **kwargs)
    plt.title(inst_wfe.name.replace(',','\n')+ "field point")
    
def show_tel_inst_wfes(instr):
    plt.figure(figsize=(12,4))
    ax1 = plt.subplot(121)
    show_telescope_wfe(instr, ax=ax1)
    ax2 = plt.subplot(122)
    show_inst_wfe(instr, ax=ax2)
    return
  
def dist(yc,xc,y1,x1):
    """ Returns the Euclidean distance between two points.
    """
    return np.sqrt((yc-y1)**2+(xc-x1)**2)

def find_coords(rad, sep, init_angle, fin_angle):
    angular_range = fin_angle-init_angle
    npoints = (np.deg2rad(angular_range)*rad)/sep   #(2*np.pi*rad)/sep
    ang_step = angular_range/npoints   #360/npoints
    x = []
    y = []
    for i in range(int(npoints)): 
        newx = rad * np.cos(np.deg2rad(ang_step * i + init_angle))
        newy = rad * np.sin(np.deg2rad(ang_step * i + init_angle))
        x.append(newx)
        y.append(newy)
    return np.array(y), np.array(x)

def contrast_curve(data):

    import matplotlib.pyplot as plt
    import photutils
    
    data_crop = data

    fwhm = 4
    wedge=(0,360)
    init_angle, fin_angle = wedge
    init_rad=fwhm

    NP = data_crop.shape[0]
    print(NP)
    array = data_crop
    centery, centerx = np.array([NP/2,NP/2])

    separation = 1.1
    separation = 0.5
    n_annuli = int(np.floor((centery)/separation))

    x = centerx
    y = centery
    total = []
    mean = []
    noise = []
    vector_radd = []

    #plt.figure(figsize=(5,5)) 
    #vmin,vmax=np.min(data_crop),np.max(data_crop)
    #plt.imshow(data_crop, cmap='CMRmap', origin = 'lower', vmin = vmin , vmax = vmax)

    for i in range(n_annuli-1):
        y = centery + init_rad + separation*(i)
        rad = dist(centery, centerx, y, x)
        yy, xx = find_coords(rad, fwhm, init_angle, fin_angle)
        yy += centery
        xx += centerx

        apertures = photutils.CircularAperture((xx, yy), fwhm/2.)
        #fluxes = photutils.aperture_photometry(array, apertures,mask = stis_mask)
        fluxes = photutils.aperture_photometry(array, apertures)
        fluxes = np.array(fluxes['aperture_sum'])

        noise_ann = np.std(fluxes)
        noise.append(noise_ann)
        vector_radd.append(rad)

        mean_ann = np.mean(fluxes)
        mean.append(mean_ann)

        nb_apertures = apertures.positions.shape[0]
        total_ann = np.sum(fluxes)/nb_apertures
        total.append(total_ann)
        #print(total_ann)
        if i <= 9:
            apertures.plot(color='blue', lw=1.5, alpha=0.5)
    #plt.show()

    total0 = np.array(total)    
    mean0 = np.array(mean)
    noise0 = np.array(noise)
    vector_rad0 = np.array(vector_radd)   
    total0_stis = np.array(total)
    mean0_stis = np.array(mean)
    noise0_stis = np.array(noise)
    vector_rad0_stis = np.array(vector_radd)  
    
    return noise0_stis, vector_rad0_stis

def generate_wavefront_errors(nb_of_maps,errors,nb_zernikes,path):

    import poppy,webbpsf
    import random
    import matplotlib.pyplot as plt

    # intial wavefront map
    nc = webbpsf.NIRCam()
    nc, ote = webbpsf.enable_adjustable_ote(nc)
    osys = nc._get_aberrations()

    # perturbed wavefront map
    nc_perturb = webbpsf.NIRCam()
    nc_perturb, ote_perturb = webbpsf.enable_adjustable_ote(nc_perturb)
    osys_perturb = nc_perturb._get_aberrations()

    # final wavefront map
    nc_final = webbpsf.NIRCam()
    nc_final, ote_final = webbpsf.enable_adjustable_ote(nc_final)
    osys_final = nc_final._get_aberrations()

    tab_opd_final = []
    for n, error in zip(range(nb_of_maps), errors):
        print(n, error)
        
        # change aberrations in wavefront map: example with random zernikes
        # this map will be our perturbation map and we will add it to the initial map with a certain weight

        # creating the perturbation map
        #weight = 0.2
        weight = error/100
        for i in range(nb_zernikes):
            #tmp = random.randint(-10,10)
            tmp = random.randint(-1,1)
            osys_perturb.zernike_coeffs[i] = weight*tmp*osys.zernike_coeffs[i]
            osys_final.zernike_coeffs[i] = osys.zernike_coeffs[i] + weight*tmp*osys.zernike_coeffs[i]

        # implementing and displaying the wavefront maps    
        #display_ote_and_psf(nc, ote, title="Initial OPD and PSF")

        ote_perturb.reset()
        ote_perturb.move_global_zernikes(osys_perturb.zernike_coeffs[0:10])
        #display_ote_and_psf(nc_perturb, ote_perturb, title="Perturbed OPD and PSF")

        ote_final.reset()
        ote_final.move_global_zernikes(osys_final.zernike_coeffs[0:10])
        #display_ote_and_psf(nc_final, ote_final, title="Final OPD and PSF")

        rms = ote.rms()
        rms_perturb = ote_perturb.rms()
        rms_final = ote_final.rms()
        print(rms,rms_perturb,rms_final)
        print('')

        #print(osys.zernike_coeffs)
        #print('')
        #print(osys_perturb.zernike_coeffs)
        #print('')
        #print(osys_final.zernike_coeffs)
        #print('')

        opd = poppy.zernike.opd_from_zernikes(osys.zernike_coeffs[0:10],
                                                       npix=1024, basis=poppy.zernike.zernike_basis_faster)

        opd_perturb = poppy.zernike.opd_from_zernikes(osys_perturb.zernike_coeffs[0:10],
                                                       npix=1024, basis=poppy.zernike.zernike_basis_faster)

        opd_final = poppy.zernike.opd_from_zernikes(osys_final.zernike_coeffs[0:10],
                                                       npix=1024, basis=poppy.zernike.zernike_basis_faster)
        
        #tab_opd_final.append(opd_final)
        
        write_fits(path+'_opd'+str(n)+'.fits',opd)
        write_fits(path+'_opd_perturb'+str(n)+'.fits',opd_perturb)
        write_fits(path+'_opd_final'+str(n)+'.fits',opd_final)
        
        #plt.figure(figsize=(12,4))
        #ax1 = plt.subplot(131)
        #ax1.imshow(opd)
        #ax1.set_title('initial wavefront map')
        #ax2 = plt.subplot(132)
        #ax2.imshow(opd_perturb)
        #ax2.set_title('perturbed wavefront map')
        #ax3 = plt.subplot(133)
        #ax3.imshow(opd_final)
        #ax3.set_title('sum of maps')
        #plt.show()

        wavefront_error = mse(np.nan_to_num(opd), np.nan_to_num(opd_final))
        print('mse',error,wavefront_error)
        print("MSE: %.2f" % (wavefront_error*100))
        
    return tab_opd_final

def mse(image, reference):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    import numpy as np
    n_points = float(image.shape[0] * image.shape[1])
    err = np.sum((image.astype("float") - reference.astype("float")) ** 2)
    err /= n_points
    err = np.sqrt(err)
    #norm = 1
    #norm = np.sum(reference) / n_points
    norm = np.sqrt(np.sum((np.abs(reference))**2)/n_points)
    print('erreur:',err/norm)
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err/norm

def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    #s = ssim(imageA, imageB)
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f" % (m))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")
    # show the images
    plt.show()
    return m

def generate_wavefront_errors_correction(nb_of_maps,errors,nb_zernikes):
    import poppy, webbpsf
    import matplotlib.pyplot as plt
    
    # intial wavefront map
    nc = webbpsf.NIRCam()
    nc, ote = webbpsf.enable_adjustable_ote(nc)
    osys = nc._get_aberrations()

    # final wavefront map
    nc_final = webbpsf.NIRCam()
    nc_final, ote_final = webbpsf.enable_adjustable_ote(nc_final)
    osys_final = nc_final._get_aberrations()

    tab_wavefront_error = np.zeros(nb_of_maps)
    tab_error = np.zeros(nb_of_maps)
    for n, error in zip(range(nb_of_maps), errors):
        print(n, error)
        #print(zip(range(nb_of_maps)))
        #print(errors)      
        # change aberrations in wavefront map: example with random zernikes
        # this map will be our perturbation map and we will add it to the initial map with a certain weight

        # creating the perturbation map
        #weight = 0.2
        #weight = error/100
        osys_corrected = osys.zernike_coeffs.copy()

        for i in range(nb_zernikes):
            if i<error+1:
                osys_corrected[i] = 0
            
        print(osys.zernike_coeffs)
        print(osys_corrected)
        
        opd = poppy.zernike.opd_from_zernikes(osys.zernike_coeffs,
                                                       npix=1024, basis=poppy.zernike.zernike_basis_faster)

        opd_corrected = poppy.zernike.opd_from_zernikes(osys_corrected,
                                                       npix=1024, basis=poppy.zernike.zernike_basis_faster)
        
        wavefront_error = mse(np.nan_to_num(opd), np.nan_to_num(opd_corrected))
        print('mse',error,wavefront_error)
        
        #tab_opd_final.append(opd_final)
        
        #write_fits('_opd'+str(n)+'.fits',opd)
        #write_fits('_opd_corrected'+str(n)+'.fits',opd_corrected)
        
        plt.figure(figsize=(12,4))
        ax1 = plt.subplot(131)
        #ax1.imshow(opd,vmin=np.min(opd),vmax=np.max(opd))
        ax1.imshow(opd)
        ax1.set_title('initial wavefront map')
        ax2 = plt.subplot(132)
        #ax2.imshow(opd_corrected,vmin=np.min(opd),vmax=np.max(opd))
        ax2.imshow(opd_corrected)
        ax2.set_title('corrected wavefront map')
        ax3 = plt.subplot(133)
        ax3.imshow(opd - opd_corrected)
        ax3.set_title('sum of maps')
        plt.show()
    
        tab_wavefront_error[n] = wavefront_error
        tab_error[n] = error
        
    return tab_wavefront_error, tab_error


def test_wavefront_errors():

    # Import packages
    #################
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from datetime import date
    from datetime import datetime
    from astropy.io import fits

    import random
    import photutils

    from Simulator import Simulation
    #from Estimation import Estimation

    os.environ['WEBBPSF_PATH'] = "/Users/mygouf/Python/webbpsf/my_webbpsf-data3"
    os.environ['PYSYN_CDBS'] = "/Users/mygouf/git/pynrc/cdbs/"

    import poppy
    import webbpsf

    # Set up directories
    ####################
    
    today = date.today()
    test_date = date = today.strftime("%Y%m%d")
    print(test_date)
    
    tests_directory = './Tests/'+test_date+'/'

    if not os.path.exists(tests_directory):
        os.makedirs(tests_directory)

    directory = tests_directory+test_date+'_wavefront_errors/'
    directory1 = directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = directory+date

    # Parameters simulation images
    ##############################

    #transmission = '/Users/mygouf/Python/webbpsf/webbpsf-data4/jwst_pupil_RevW_npix1024.fits.gz'
    #opd = '/Users/mygouf/Python/webbpsf/webbpsf-data4/NIRCam/OPD/OPD_RevW_ote_for_NIRCam_requirements.fits.gz'

    # poppy paramaters
    
    pixelscale = 0.063
    fov_arcsec = 10
    #oversample = 4
    #wavelength = 4.441e-6
    
    # webbPSF parameters

    #filt = 'F444W'

    
    # Generate wavefront errors
    ###########################

    nb_of_maps = 10
    errors = [1,2,3,4,5,10,20,30,40,50]
    #nb_of_maps = 2
    #errors = [1,2]
    nb_zernikes = 35

    tab_opd_final = generate_wavefront_errors(nb_of_maps,errors,nb_zernikes,path)


    # Generating images with those wavefronts
    #########################################

    dict_simulation_parameters = {'fov_arcsec': fov_arcsec}
    simulation = Simulation(dict_simulation_parameters)
    
    tab_images_initial = np.zeros((nb_of_maps,636,636))
    tab_images_final = np.zeros((nb_of_maps,636,636))
    for i in range(nb_of_maps):
        dict_initial = simulation.create_image_from_opd_file(opd=path+'_opd'+str(i)+'.fits', input_noise=None)
        dict_final = simulation.create_image_from_opd_file(opd=path+'_opd_final'+str(i)+'.fits', input_noise=None)

        image_initial0 = dict_initial['image']
        image_final0 = dict_final['image']

        tab_images_initial[i] = image_initial0
        tab_images_final[i] = image_final0
    

    # Compute contrast curves
    #########################

    contrast_initial_image1, vector_rad_initial_image1 = contrast_curve(tab_images_initial[0])
    contrast_final_image1, vector_rad_final_image1 = contrast_curve(tab_images_initial[0]-tab_images_final[0])
    contrast_initial_image2, vector_rad_initial_image2 = contrast_curve(tab_images_initial[1])
    contrast_final_image2, vector_rad_final_image2 = contrast_curve(tab_images_initial[1]-tab_images_final[1])
    contrast_initial_image3, vector_rad_initial_image3 = contrast_curve(tab_images_initial[2])
    contrast_final_image3, vector_rad_final_image3 = contrast_curve(tab_images_initial[2]-tab_images_final[2])
    contrast_initial_image4, vector_rad_initial_image4 = contrast_curve(tab_images_initial[3])
    contrast_final_image4, vector_rad_final_image4 = contrast_curve(tab_images_initial[3]-tab_images_final[3])
    contrast_initial_image5, vector_rad_initial_image5 = contrast_curve(tab_images_initial[4])
    contrast_final_image5, vector_rad_final_image5 = contrast_curve(tab_images_initial[4]-tab_images_final[4])
    contrast_initial_image6, vector_rad_initial_image6 = contrast_curve(tab_images_initial[5])
    contrast_final_image6, vector_rad_final_image6 = contrast_curve(tab_images_initial[5]-tab_images_final[5])
    contrast_initial_image7, vector_rad_initial_image7 = contrast_curve(tab_images_initial[6])
    contrast_final_image7, vector_rad_final_image7 = contrast_curve(tab_images_initial[6]-tab_images_final[6])
    contrast_initial_image8, vector_rad_initial_image8 = contrast_curve(tab_images_initial[7])
    contrast_final_image8, vector_rad_final_image8 = contrast_curve(tab_images_initial[7]-tab_images_final[7])
    contrast_initial_image9, vector_rad_initial_image9 = contrast_curve(tab_images_initial[8])
    contrast_final_image9, vector_rad_final_image9 = contrast_curve(tab_images_initial[8]-tab_images_final[8])
    contrast_initial_image10, vector_rad_initial_image10 = contrast_curve(tab_images_initial[9])
    contrast_final_image10, vector_rad_final_image10 = contrast_curve(tab_images_initial[9]-tab_images_final[9])


    # Create and saving figures
    ###########################

    pxscale = pixelscale

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(111)
    plt.plot(vector_rad_initial_image1*pxscale, contrast_initial_image1/np.max(tab_images_initial[0]), label='Raw')
    plt.plot(vector_rad_final_image1*pxscale, contrast_final_image1/np.max(tab_images_initial[0]), label='1% error',linestyle='--')
    plt.plot(vector_rad_final_image5*pxscale, contrast_final_image5/np.max(tab_images_initial[4]), label='5% error',linestyle='--')
    plt.plot(vector_rad_final_image6*pxscale, contrast_final_image6/np.max(tab_images_initial[5]), label='10% error',linestyle='--')
    plt.plot(vector_rad_final_image10*pxscale, contrast_final_image10/np.max(tab_images_initial[9]), label='50% error',linestyle='--')

    plt.xlabel('Angular separation [arcsec]')
    plt.ylabel('Contrast')
    plt.grid('on', which='both', alpha=0.2, linestyle='solid')
    ax1.set_yscale('log')
    ax1.set_xlim(0, 4)
    plt.legend()
    plt.show()

    fname = path+'_contrast_curves.pdf'       
    fig.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, pad_inches=0.1,
                frameon=None, metadata=None,bbox_inches = 'tight')
    print('Saving file:',fname)


    # Create and saving figures
    ###########################

    contrast_raw = contrast_initial_image1/np.max(tab_images_initial[0])
    contrast1 = contrast_final_image1/np.max(tab_images_initial[0])
    contrast5 = contrast_final_image5/np.max(tab_images_initial[4])
    contrast6 = contrast_final_image6/np.max(tab_images_initial[5])
    contrast10 = contrast_final_image10/np.max(tab_images_initial[9])
                                              
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(111)
    plt.plot(vector_rad_initial_image1*pxscale, contrast_raw/contrast_initial_image1/np.max(tab_images_initial[0]), label='Raw')
    plt.plot(vector_rad_final_image1*pxscale, contrast_raw/contrast1, label='1% error',linestyle='--')
    plt.plot(vector_rad_final_image5*pxscale, contrast_raw/contrast5, label='5% error',linestyle='--')
    plt.plot(vector_rad_final_image6*pxscale, contrast_raw/contrast6, label='10% error',linestyle='--')
    plt.plot(vector_rad_final_image10*pxscale, contrast_raw/contrast10, label='50% error',linestyle='--')
    
    plt.xlabel('Angular separation [arcsec]')
    plt.ylabel('Contrast Gain')
    plt.grid('on', which='both', alpha=0.2, linestyle='solid')
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 50)
    ax1.hlines([0,10], 0, 10, colors='k', linestyles='solid', label='', data=None)
    
    plt.legend()
    plt.show()

    fname = path+'_contrast_gain_curves.pdf'       
    fig.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, pad_inches=0.1,
                frameon=None, metadata=None,bbox_inches = 'tight')
    print('Saving file:',fname)


    # How many Zernike coeff. does that correspond to?
    ##################################################

    nb_zernikes = 36
    errors = np.linspace(0,nb_zernikes,nb_zernikes+1) # actually number of corrected zernike coefficients
    nb_of_maps = len(errors)
    print(nb_of_maps)
    print(errors)
    # nb_of_maps = 2
    # errors = [1,36]

    # nb_of_maps = 1
    # errors = [1]

    tab_wavefront_error, tab_error = generate_wavefront_errors_correction(nb_of_maps,errors,nb_zernikes)

    print('Number of corrected Zernike coefficients:',tab_error)
    print('Error in percent:',tab_wavefront_error)

    
    # Save fits files
    #################

    #

    
if __name__ == "__main__":
    
    from datetime import date
    from astropy.io import fits
    import os
    import numpy as np
    
    test_wavefront_errors()

    today = date.today()
    date = test_date = today.strftime("%Y%m%d")

    date_ref = '20200308'
    
    tests_directory = './Tests/'+test_date+'/'

    if not os.path.exists(tests_directory):
        os.makedirs(tests_directory)
        
    directory = tests_directory+test_date+'_wavefront_errors/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    #filename = directory+date+test+'_Zernike_coefficients_estimation.fits'
    #filename_ref = '../Reference_Tests/'+date_ref+'_wavefront_errors/'+date_ref+'_wavefront_errors.fits'
    #hdul = fits.open(filename)
    #hdul_ref = fits.open(filename)
    #image = hdul[0].data
    #image_ref = hdul[0].data

    #diff1 = np.sum(image-image_ref)

    #filename = directory+date+test+'_Zernike_coefficients_estimation.fits'
    #filename_ref = '../Reference_Tests/'+date_ref+'_wavefront_errors/'+date_ref+'_psf_webbpsf.fits'
    #hdul = fits.open(filename)
    #hdul_ref = fits.open(filename)
    #image = hdul[0].data
    #image_ref = hdul[0].data

    #diff1 = np.sum(image-image_ref)
    
    #print(diff1)

    #if np.sum([diff1,diff2]) == 0:
    #    print("Test 'Wavefront errors' passed")
    #else:
    #    print("Test 'Wavefront errors' not passed")
    
