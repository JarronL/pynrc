import numpy as np

from scipy.optimize import leastsq
from scipy.optimize import least_squares
from scipy.optimize import minimize

from Simulator import Simulation

from astropy.io import fits
import poppy

class Estimation(object):
    
    def __init__(self, dict_ = None, guess = None, guess2 = None):

        self.dict_ = dict_
        self.image = dict_['image']
        self.noise = dict_['noise']
        self.wfe_budget = dict_['wfe_budget']
        self.fov_arcsec = dict_['fov_arcsec']
        self.guess = guess
        self.guess0 = guess

        seed = 12345
        self.rs = np.random.RandomState(seed)
        
        #Computing guess for the estimation of Zernike coefficients using an error budget
        #This does not constrain the values of the estimated parameters during the estimation
        if np.all(guess) == None:
            x0 = np.zeros(len(self.wfe_budget))
            print(x0)
            for i in range(0,len(self.wfe_budget)):
                term = self.wfe_budget[i]
                #x0.append(np.random.uniform(low=-1. * term, high=1. * term))
                #x0[i] = np.random.uniform(low=-1. * term, high=1. * term)
                x0[i] = self.rs.uniform(low=-1e-6 * term, high=1e-6 * term)
                self.guess = np.array(x0[1:len(self.wfe_budget)])
                guess = np.array(x0[1:len(self.wfe_budget)])
        print('Guess')
        print(self.guess)
        print('')
        
        # Computing guess for the estimation of the noise map
        # This does not constrain the values of the estimated parameters during the estimation
        #if np.all(guess2) == None:
        #    shape_noise = self.image.shape
        #    guess2 = np.zeros(shape_noise)

        #print('Noise guess', np.sum(guess2))
        #print('')
        
    def func(self, zernike_coefficients, image):
        dict_ = Simulation.create_image(self,zernike_coefficients,self.noise)
        image_ = dict_['image']
        
        result = np.ndarray.flatten(image - image_)
        criterion = np.sum(result**2,axis=0)
        print('Criterion:', criterion)
        return result

    def func2(self, zernike_coefficients, image):
        dict_ = Simulation.create_image(self,zernike_coefficients,self.noise)
        image_ = np.ndarray.flatten(dict_['image'])
        
        result = np.ndarray.flatten(image - image_)
        criterion = np.sum(result**2,axis=0)
        print('Criterion:', criterion)
        return criterion
    

    def estimate_zernikes(self,method='leastsq'):
        init_criterion = np.sum(self.func(self.guess,self.image)**2,axis=0)
        print(init_criterion)

        #dict_ = Simulation.create_image(self,self.guess,self.noise)
        #guess_image = dict_['image']

        print(self.image.shape)

        # Leastsq optimizer
        ###################

        if method == 'leastsq':
            estimated_coefficients = leastsq(self.func, self.guess, args=(self.image), Dfun=None, full_output=1, col_deriv=0, ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=100, epsfcn=0.01, factor=100, diag=None)
            #estimated_coefficients = leastsq(self.func, self.guess, args=(self.image), Dfun=None, full_output=1, col_deriv=0, ftol=1.49012e-08, xtol=1.49012e-08, gtol=1, maxfev=100, epsfcn=1e2, factor=0.1, diag=None)
        
            final_criterion = np.sum(self.func(estimated_coefficients[0],self.image)**2,axis=0)
            dict_ = Simulation.create_image(self,estimated_coefficients[0],self.noise)
            final_image = dict_['image']
        
        # L-BFGS-L optimizer
        ####################

        if method == 'L-BFGS-B':

            precision = 1
            # Putting some bounds on the parameters such that we know them with a precision of 1%
            #guess_min = [self.guess0[i]-self.guess0[i]*precision for i in range(len(self.guess0))]
            #guess_max = [self.guess0[i]+self.guess0[i]*precision for i in range(len(self.guess0))]
            #guess_min = np.zeros(len(self.guess0))
            #guess_max = np.zeros(len(self.guess0))
            #bounds = np.zeros((len(self.guess0),2))
            #for i in range(len(self.guess0)):
            #    guess_min[i] = self.guess0[i]-np.abs(self.guess0[i])*precision
            #    guess_max[i] = self.guess0[i]+np.abs(self.guess0[i])*precision
            #    bounds[i] = (guess_min[i],guess_max[i])
                
            #bounds = ((guess_min[i],guess_max[i]) for i in range(len(self.guess0)))
            #print('bounds',bounds)
            bounds = None
            #print(guess_min)
            #print(guess_max)
            #bounds = ((guess_min[0], guess_max[0]), (guess_min[1], guess_max[1]))
            #print('bounds',bounds)
            #bounds = ((-1, 1), (-1, 1))
            
            #estimated_coefficients = minimize(self.func2, self.guess, args=(np.ndarray.flatten(self.image)), method='L-BFGS-B', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
           #estimated_coefficients = minimize(self.func2, self.guess, args=(np.ndarray.flatten(self.image)), method='L-BFGS-B', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': -1, 'maxls': 20})
            estimated_coefficients = minimize(self.func2, self.guess, args=(np.ndarray.flatten(self.image)), method='L-BFGS-B', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options={'disp': None, 'maxcor': 100, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-02, 'maxfun': 15000, 'maxiter': 15000, 'iprint': -1, 'maxls': 20})
            

            estimated_coefficients = [estimated_coefficients.x]
        
            final_criterion = np.sum(self.func(estimated_coefficients[0],self.image)**2,axis=0)

            #tmp = 1
            #estimated_coefficients = least_squares(self.func2, self.guess, args=(self.image,tmp))
            #final_criterion = np.sum(self.func2(estimated_coefficients[0],self.image,tmp)**2,axis=0)
            
            print(final_criterion)
        
            #image = Simulation.create_image(coefficient_set_init[1:len(wfe_budget)])
            #final_image = Simulation.create_image(self,estimated_coefficients[0])
            dict_ = Simulation.create_image(self,estimated_coefficients[0],self.noise)
            final_image = dict_['image']

        ####################

        print('')
        print('Guess')
        print(np.append(0,self.guess))
        print('')
        print('Estimated coefficients')
        print(np.append(0,estimated_coefficients[0]))
        print('')
        #print('Bounds')
        #print(bounds)
        #print('')
#        print('Simulated coefficients')
#        print(coefficient_set_init)
#        print('')
        print('Flag')
        #print(estimated_coefficients[4])

#print(coefficient_set_init[1:len(wfe_budget)])
#        print(estimated_coefficients[0])
    #
        return final_image, estimated_coefficients

    def generate_opd(self,coefficient_set_init=None,path ='wavefront_map.fits'):

        transmission = '/Users/mygouf/Python/webbpsf/webbpsf-data4/jwst_pupil_RevW_npix1024.fits.gz'
        opd = '/Users/mygouf/Python/webbpsf/webbpsf-data4/NIRCam/OPD/OPD_RevW_ote_for_NIRCam_requirements.fits.gz'
        hdul = fits.open(opd)
        hdul2 = fits.open(transmission)
        
        # Create wavefront map
        zernike_coefficients = np.append(0,coefficient_set_init)
        wavefront_map = poppy.zernike.opd_from_zernikes(zernike_coefficients,
                                               npix=1024, basis=poppy.zernike.zernike_basis_faster)
        wavefront_map = np.nan_to_num(wavefront_map)*hdul2[0].data
        
        fits.writeto(path,wavefront_map,hdul[0].header,overwrite=True)

        return wavefront_map


class Estimation_noise(object):
    
    def __init__(self, dict_ = None, guess = None, guess2 = None):

        self.dict_ = dict_
        self.image = dict_['image']
        self.noise = dict_['noise']
        self.guess2 = guess2

        # Computing guess for the estimation of the noise map
        # This does not constrain the values of the estimated parameters during the estimation
        if np.all(guess2) == None:
            #shape_noise = self.noise.shape
            #shape_noise = self.image.shape
            #self.guess2 = np.zeros(shape_noise)
            #print('shape2',self.guess2.shape)
            self.guess2 =  np.random.normal(loc=self.image, scale=np.sqrt(self.image>0))
    
    def func2(self,noise_map,image):

        #noise_map_tmp = np.ndarray.flatten(noise_map)
        #image_tmp = np.ndarray.flatten(image)
        noise_map_tmp = noise_map
        image_tmp = image

        result = image_tmp - noise_map_tmp
        criterion = np.sum(result**2,axis=0)
        print('Criterion:', criterion)

        #noise_map = np.reshape(noise_map_tmp,noise_map.shape)
        #image = np.reshape(image_tmp,image.shape)
        noise_map = noise_map_tmp
        image = image_tmp

        return result

    
    def estimate_noise(self):

        init_criterion = np.sum(self.func2(self.guess2,self.image)**2,axis=0)
        
        #estimated_noise = leastsq(self.func2, self.guess2, args=(self.image), Dfun=None, full_output=1, col_deriv=0, ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=100, epsfcn=0.01, factor=100, diag=None)
        estimated_noise = leastsq(self.func2, self.guess2, args=(self.image), Dfun=None, full_output=1, col_deriv=0, ftol=1.49012e-01, xtol=1.49012e-01, gtol=0.0, maxfev=1, epsfcn=0.01, factor=0.1, diag=None)

        print(estimated_noise[0].shape)
        #print(image.shape)
        final_criterion = np.sum(self.func2(estimated_noise[0],self.image)**2,axis=0)
        print(final_criterion)
        
        return estimated_noise 

    #func : callable
    #should take at least one (possibly length N vector) argument and
    #returns M floating point numbers. It must not return NaNs or
    #fitting might fail.
#x0 : ndarray
#    The starting estimate for the minimization.
#args : tuple, optional
#    Any extra arguments to func are placed in this tuple.
#Dfun : callable, optional
#    A function or method to compute the Jacobian of func with derivatives
#    across the rows. If this is None, the Jacobian will be estimated.
#full_output : bool, optional
#    non-zero to return all optional outputs.
#col_deriv : bool, optional
#    non-zero to specify that the Jacobian function computes derivatives
#    down the columns (faster, because there is no transpose operation).
#ftol : float, optional
#    Relative error desired in the sum of squares.
#xtol : float, optional
#    Relative error desired in the approximate solution.
#gtol : float, optional
#    Orthogonality desired between the function vector and the columns of
#    the Jacobian.
#maxfev : int, optional
#    The maximum number of calls to the function. If `Dfun` is provided
#    then the default `maxfev` is 100*(N+1) where N is the number of elements
#    in x0, otherwise the default `maxfev` is 200*(N+1).
#epsfcn : float, optional
#    A variable used in determining a suitable step length for the forward-
#    difference approximation of the Jacobian (for Dfun=None).
#    Normally the actual step length will be sqrt(epsfcn)*x
#    If epsfcn is less than the machine precision, it is assumed that the
#    relative errors are of the order of the machine precision.
#factor : float, optional
#    A parameter determining the initial step bound
#    (``factor * || diag * x||``). Should be in interval ``(0.1, 100)``.
#diag : sequence, optional
#    N positive entries that serve as a scale factors for the variables.

