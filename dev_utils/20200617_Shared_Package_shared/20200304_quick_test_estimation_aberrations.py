
def write_fits(path,array):
    from astropy.io import fits
    hdul = fits.PrimaryHDU(array)
    hdul.writeto(path,overwrite=True)
    return

def figure_images(image_simu,final_image,path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1,3, figsize=(13,3))
    ax1,ax2,ax3= axes
    im1 = ax1.imshow(image_simu,cmap=plt.get_cmap('gist_heat'),norm=LogNorm(vmin=1e-7, vmax=1e4))
    ax1.set_title('Simulated Image')
    im2 = ax2.imshow(final_image,cmap=plt.get_cmap('gist_heat'),norm=LogNorm(vmin=1e-7, vmax=1e4))
    ax2.set_title('Final image')
    im3 = ax3.imshow(image_simu-final_image,cmap=plt.get_cmap('gist_heat'),norm=LogNorm(vmin=1e-7, vmax=1e4))
    ax3.set_title('Difference')
    plt.colorbar(im1,ax=ax1)
    plt.colorbar(im2,ax=ax2)
    plt.colorbar(im3,ax=ax3)
    plt.show()
    fname = path+'_image_estimation.pdf'       
    fig.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, pad_inches=0.1,
                frameon=None, metadata=None,bbox_inches = 'tight')
    print('Saving file:',fname)
    return 

def figure_wavefronts(wavefront_map_simu,wavefront_map_estimated,path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3, figsize=(13,3))
    ax1,ax2,ax3= axes
    im1 = ax1.imshow(wavefront_map_simu,cmap=plt.get_cmap('gist_heat'))
    ax1.set_title('Simulated Image')
    im2 = ax2.imshow(wavefront_map_estimated,cmap=plt.get_cmap('gist_heat'))
    ax2.set_title('Final image')
    im3 = ax3.imshow(wavefront_map_simu-wavefront_map_estimated,cmap=plt.get_cmap('gist_heat'))
    ax3.set_title('Difference')
    plt.colorbar(im1,ax=ax1)
    plt.colorbar(im2,ax=ax2)
    plt.colorbar(im3,ax=ax3)
    plt.show()

    fname = path+'_wavefront_map_estimation.pdf'    
    #fname = date+'_image_estimation.pdf'    
    fig.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
             orientation='portrait', papertype=None, format=None,
             transparent=False, pad_inches=0.1,
             frameon=None, metadata=None,bbox_inches = 'tight')
    print('Saving file:',fname)
    return

def figure_coefficients(coefficient_set_init,estimated_coefficients,path):

    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(1,3, figsize=(13,4.5))

    tmp = estimated_coefficients[0]

    i = 0
    for ax in axes.flatten():
        ax.set_xlabel('Zernike Coefficients')
        #ax.legend(ncol=1)

        ax.xaxis.get_major_locator().set_params(nbins=9, steps=[1, 2, 5, 10])
        ax.yaxis.get_major_locator().set_params(nbins=9, steps=[1, 2, 5, 10])

        if i == 0: 
            ax.plot(range(1,len(tmp)+1),coefficient_set_init[1:len(tmp)+1])
            ax.set_title('Simulated Coefficients')
            ax.set_ylabel('Value')
            ax.set_ylim(np.min(coefficient_set_init[1:len(tmp)+1]),np.max(coefficient_set_init[1:len(tmp)+1]))
        if i == 1: 
            ax.plot(range(1,len(tmp)+1),estimated_coefficients[0])
            ax.set_title('Estimated Coefficients')
            ax.set_ylim(np.min(coefficient_set_init[1:len(tmp)+1]),np.max(coefficient_set_init[1:len(tmp)+1]))
        if i == 2: 
            ax.plot(range(1,len(tmp)+1),coefficient_set_init[1:len(tmp)+1]-estimated_coefficients[0])
            ax.set_title('Difference')
            ax.set_ylim(np.min(coefficient_set_init[1:len(tmp)+1]),np.max(coefficient_set_init[1:len(tmp)+1]))
        i += 1
    
    fig.tight_layout()

    fig.suptitle('', fontsize=16);
    fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.85, bottom=0.1 , left=0.05, right=0.97)

    plt.show()

    fname = path+'_Zernike_coefficients_estimation.pdf'    
    fig.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, pad_inches=0.1,
                frameon=None, metadata=None,bbox_inches = 'tight')
    print('Saving file:',fname)
    return

def quick_test_estimation_aberrations():

    # Import packages
    #################
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from datetime import date
    from datetime import datetime
    from astropy.io import fits

    from Simulator import Simulation
    from Estimation import Estimation

    # Set up directories
    ####################
    
    today = date.today()
    date = test_date = today.strftime("%Y%m%d")
    print(test_date)
    
    tests_directory = './Tests/'+test_date+'/'

    if not os.path.exists(tests_directory):
        os.makedirs(tests_directory)

    directory = tests_directory+test_date+'_quick_test_estimation_aberrations/'
    if not os.path.exists(directory):
        os.makedirs(directory)


    # Parameters simulation images
    ##############################        

    #transmission = '/Users/mygouf/Python/webbpsf/webbpsf-data4/jwst_pupil_RevW_npix1024.fits.gz'
    #opd = '/Users/mygouf/Python/webbpsf/webbpsf-data4/NIRCam/OPD/OPD_RevW_ote_for_NIRCam_requirements.fits.gz'

    wfe_budget = [0, 2000, 2000, 1500, 1000, 1000, 500, 360, 360, 250, 100, 100, 80, 70, 70, 60, 50, 50, 40, 30, 30, 20, 10, 10, 9, 8, 8, 7, 6, 6, 5, 4, 4, 3, 2, 2, 1]
    #wfe_budget = [0, 2000, 2000] #, 1500, 1000, 1000, 500, 360, 360, 250, 100, 100, 80, 70, 70, 60, 50, 50, 40, 30, 30, 20, 10, 10, 9, 8, 8, 7, 6, 6, 5, 4, 4, 3, 2, 2, 1]
        
    # poppy paramaters
    
    #pixelscale = 0.063
    fov_arcsec = 10
    #oversample = 4
    #wavelength = 4.441e-6
    
    # webbPSF parameters

    #filt = 'F444W'
    
    # Simulation PSF
    ################
        
    dict_simulation_parameters = {'wfe_budget': wfe_budget, 'fov_arcsec': fov_arcsec}
    simulation = Simulation(dict_simulation_parameters)
    dict_, coefficient_set_init = simulation.generate_psfs()
    image = dict_['image']
    noise = dict_['noise']
    wavefront_map = dict_['wavefront_map']
    dict_['wfe_budget'] = wfe_budget
    dict_['fov_arcsec'] = fov_arcsec

    image_simu = image.copy()
    noise_simu = noise.copy()
    wavefront_map_simu = wavefront_map.copy()
    coefficient_set_init_simu = coefficient_set_init.copy()

    
    # Estimation Zernike coefficients - without noise same budget
    #################################

    test = '_without_noise_same_budget'
    wfe_budget_estimation = wfe_budget

    dict_['wfe_budget'] = wfe_budget_estimation
    dict_['noise'] = noise_simu

    now = datetime.now()

    estimation = Estimation(dict_) 
    final_image, estimated_coefficients = estimation.estimate_zernikes()
    print('Simulated coefficients',coefficient_set_init)
    dict_estimated = simulation.create_image(estimated_coefficients[0])
    wavefront_map_estimated = dict_estimated['wavefront_map']

    now2 = datetime.now()
    print("Time to estimate", now2-now)
                
    # Save fits files
    write_fits(directory+date+test+'_image_simulation.fits',image_simu)
    write_fits(directory+date+test+'_noise_simulation.fits',noise_simu)
    write_fits(directory+date+test+'_Zernike_coefficients_simulation.fits',coefficient_set_init_simu)
    write_fits(directory+date+test+'_wavefront_map_simulation.fits',wavefront_map_simu)
    write_fits(directory+date+test+'_image_estimation.fits',final_image)
    write_fits(directory+date+test+'_Zernike_coefficients_estimation.fits',estimated_coefficients[0])
    write_fits(directory+date+test+'_wavefront_map_estimation.fits',wavefront_map_estimated)
    
    # Create and saving figures
    figure_images(image_simu,final_image,directory+date+test)
    figure_wavefronts(wavefront_map_simu,wavefront_map_estimated,directory+date+test)
    figure_coefficients(coefficient_set_init_simu,estimated_coefficients,directory+date+test)

    # Estimation Zernike coefficients - with noise same budget
    #################################

    test = '_with_noise_same_budget'
    wfe_budget_estimation = wfe_budget
    
    dict_['wfe_budget'] = wfe_budget_estimation
    dict_['noise'] = None

    now = datetime.now()

    estimation = Estimation(dict_) 
    final_image, estimated_coefficients = estimation.estimate_zernikes()
    print('Simulated coefficients',coefficient_set_init)
    dict_estimated = simulation.create_image(estimated_coefficients[0])
    wavefront_map_estimated = dict_estimated['wavefront_map']

    now2 = datetime.now()
    print("Time to estimate", now2-now)
                
    # Save fits files
    write_fits(directory+date+test+'_image_simulation.fits',image_simu)
    write_fits(directory+date+test+'_noise_simulation.fits',noise_simu)
    write_fits(directory+date+test+'_Zernike_coefficients_simulation.fits',coefficient_set_init_simu)
    write_fits(directory+date+test+'_wavefront_map_simulation.fits',wavefront_map_simu)
    write_fits(directory+date+test+'_image_estimation.fits',final_image)
    write_fits(directory+date+test+'_Zernike_coefficients_estimation.fits',estimated_coefficients[0])
    write_fits(directory+date+test+'_wavefront_map_estimation.fits',wavefront_map_estimated)    
    # Create and saving figures
    figure_images(image_simu,final_image,directory+date+test)
    figure_wavefronts(wavefront_map_simu,wavefront_map_estimated,directory+date+test)
    figure_coefficients(coefficient_set_init_simu,estimated_coefficients,directory+date+test)

    # Estimation Zernike coefficients - with noise different budget
    #################################

    test = '_with_noise_different_budget'
    
    wfe_budget_estimation = [0, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]
 
    dict_['wfe_budget'] = wfe_budget_estimation
    dict_['noise'] = None

    now = datetime.now()

    estimation = Estimation(dict_) 
    final_image, estimated_coefficients = estimation.estimate_zernikes()
    print('Simulated coefficients',coefficient_set_init)
    dict_estimated = simulation.create_image(estimated_coefficients[0])
    wavefront_map_estimated = dict_estimated['wavefront_map']

    now2 = datetime.now()
    print("Time to estimate", now2-now)
                
    # Save fits files
    write_fits(directory+date+test+'_image_simulation.fits',image_simu)
    write_fits(directory+date+test+'_noise_simulation.fits',noise_simu)
    write_fits(directory+date+test+'_Zernike_coefficients_simulation.fits',coefficient_set_init_simu)
    write_fits(directory+date+test+'_wavefront_map_simulation.fits',wavefront_map_simu)
    write_fits(directory+date+test+'_image_estimation.fits',final_image)
    write_fits(directory+date+test+'_Zernike_coefficients_estimation.fits',estimated_coefficients[0])
    write_fits(directory+date+test+'_wavefront_map_estimation.fits',wavefront_map_estimated)    
    # Create and saving figures
    figure_images(image_simu,final_image,directory+date+test)
    figure_wavefronts(wavefront_map_simu,wavefront_map_estimated,directory+date+test)
    figure_coefficients(coefficient_set_init_simu,estimated_coefficients,directory+date+test)

    
if __name__ == "__main__":

    from datetime import date
    from astropy.io import fits
    import os
    import numpy as np
    
    #quick_test_estimation_aberrations()

    today = date.today()
    date = test_date = today.strftime("%Y%m%d")

    date_ref = '20200308'
    
    tests_directory = './Tests/'+test_date+'/'

    if not os.path.exists(tests_directory):
        os.makedirs(tests_directory)

    directory = tests_directory+test_date+'_quick_test_estimation_aberrations/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    test = '_without_noise_same_budget'    
    filename = directory+date+test+'_Zernike_coefficients_estimation.fits'
    filename_ref = '../Reference_Tests/'+date_ref+'_quick_test_estimation_aberrations/'+date_ref+test+'_Zernike_coefficients_estimation.fits'
    hdul = fits.open(filename)
    hdul_ref = fits.open(filename_ref)
    coefficients = hdul[0].data
    coefficients_ref = hdul_ref[0].data

    diff1 = np.sum(coefficients-coefficients_ref)
    
    test = '_with_noise_same_budget'    
    filename = directory+date+test+'_Zernike_coefficients_estimation.fits'
    filename_ref = '../Reference_Tests/'+date_ref+'_quick_test_estimation_aberrations/'+date_ref+test+'_Zernike_coefficients_estimation.fits'
    hdul = fits.open(filename)
    hdul_ref = fits.open(filename_ref)
    coefficients = hdul[0].data
    coefficients_ref = hdul_ref[0].data

    diff2 = np.sum(coefficients-coefficients_ref)
    
    test = '_with_noise_different_budget'    
    filename = directory+date+test+'_Zernike_coefficients_estimation.fits'
    filename_ref = '../Reference_Tests/'+date_ref+'_quick_test_estimation_aberrations/'+date_ref+test+'_Zernike_coefficients_estimation.fits'
    hdul = fits.open(filename)
    hdul_ref = fits.open(filename_ref)
    coefficients = hdul[0].data
    coefficients_ref = hdul_ref[0].data

    #print(coefficients)
    #print(coefficients_ref)
    diff3 = np.sum(coefficients-coefficients_ref)

    print(diff1,diff2,diff3)

    if np.sum([diff1,diff2,diff3]) <= 1e-3:
        print("Test 'Estimation Aberrations' passed")
    else:
        print("Test 'Estimation Aberrations' not passed")
        
