# From https://github.com/joolof/DDiT as a comparison to VIP's scattered light module
# This code is not used in the current version of pynrc, but is kept here for reference.

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
# -----------------------------------------------------------
np.seterr(over='ignore')
# -----------------------------------------------------------------------------
# Define some constants
# -----------------------------------------------------------------------------
CC = 2.9979e+14     # in microns/s 
HH = 6.6262e-27
KK = 1.3807e-16
# -----------------------------------------------------------
class Disk(object):
    """
    Class to compute images of debris disks, in polarized light and total intensity.
    """
    def __init__(self, nx = 300, pixscale = 0.01226, nframe = 150, thermal = False, dpc = None, gaussian = False, nm = 2., engler = False, threshold = 1.e-2):
        """
        Class to compute synthetic images of debris disks. To compute a model, simply do the following:
        > disk = Disk()
        > disk.compute_model(a = 0.8, e = 0.1, incl = 60.)

        And the images will be stored in disk.intensity and disk.polarized. More details in the README.md file.
        """

        """
        Some checks on the input parameters
        """
        if type(nx) is not int: self._error_msg('The parameter \'nx\' should be an integer ')
        if type(nframe) is not int: self._error_msg('The parameter \'nframe\' should be an integer ')
        if nframe%2 == 1:
            print('It is probably better if \'nframe\' is an even number.')
        if ((thermal) and (dpc is None)):
            self._error_msg('To compute thermal images, you need to provide a distance \'dpc\' in pc. This is because of the way the T(r) is defined.')
        self._threshold = threshold
        self._nm = nm
        self._engler = engler
        self._nx = nx
        if nx%2 ==0:
            self._cx = self._nx//2 - 0.5
        else:
            self._cx = self._nx//2
        self._nframe = nframe
        self._pixscale = pixscale
        self._thermal = thermal
        self._gaussian = gaussian
        self._dpc = dpc
        self._xlim = self._cx * self._pixscale
        self._Xin, self._Yin = (np.mgrid[0:self._nx, 0:self._nx] - self._cx) * self._pixscale
        self._xm, self._ym = np.zeros(shape=(self._nx, self._nx)), np.zeros(shape=(self._nx, self._nx))
        """
        Define some variables that will contain the:
            - total intensity image
            - polarized intensity image
            - scattering angle at the midplane
            - distance to the star at the midplane
            - azimuth angle at the midplane
        """
        self.polarized = np.zeros(shape=(self._nx, self._nx))
        self.intensity = np.zeros(shape=(self._nx, self._nx))
        self.scattering = np.zeros(shape=(self._nx, self._nx))
        self.distance = np.zeros(shape=(self._nx, self._nx))
        self.azimuth = np.zeros(shape=(self._nx, self._nx))
        """
        Some trigonometry variables
        """
        self._cs, self._ss, self._st, self._st2, self._to = 0., 0., 0., 0., 0.
        """
        The geometric parameters for the disk, with some default values.
        """
        self._a, self._incl, self._pa, self._pin, self._pout, self._e, self._omega, self._opang, self._dr, self._pmid, self._da, self._gamma = 1., 0.1, 152.1*np.pi/180., 25., -2.5, 0., 0.,0.04, 0.05, None, None, 2.0
        """
        Parameters for the phase functions
        """
        self._gsca, self._gpol = 0., 0.
        self.theta, self.s11, self.s12 = None, None, None
        """
        Parameters for the thermal images
        """
        self._nu = None

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        self._a = a

    @property
    def da(self):
        return self._da

    @da.setter
    def da(self, da):
        self._da = da

    @property
    def e(self):
        return self._e

    @e.setter
    def e(self, e):
        self._e = e

    @property
    def pin(self):
        return self._pin

    @pin.setter
    def pin(self, pin):
        self._pin = pin

    @property
    def pmid(self):
        return self._pmid

    @pmid.setter
    def pmid(self, pmid):
        self._pmid = pmid

    @property
    def opang(self):
        return self._opang

    @opang.setter
    def opang(self, opang):
        self._opang = opang

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    @property
    def gpol(self):
        return self._gpol

    @gpol.setter
    def gpol(self, gpol):
        self._gpol = gpol

    @property
    def gsca(self):
        return self._gsca

    @gsca.setter
    def gsca(self, gsca):
        self._gsca = gsca

    @property
    def dr(self):
        return self._dr

    @dr.setter
    def dr(self, dr):
        self._dr = dr

    @property
    def pout(self):
        return self._pout

    @pout.setter
    def pout(self, pout):
        self._pout = pout

    @property
    def incl(self):
        return self._incl * 180. / np.pi

    @incl.setter
    def incl(self, incl):
        self._incl = incl * np.pi/180.

    @property
    def pa(self):
        return -self._pa * 180. / np.pi

    @pa.setter
    def pa(self, pa):
        self._pa = -pa * np.pi/180.

    @property
    def omega(self):
        return self._omega * 180. / np.pi

    @omega.setter
    def omega(self, omega):
        self._omega = omega * np.pi/180.

    """
    Get the entry and exit points of an ellipse in the [y,z] plane
    """
    def _get_sphere(self):
        """
        Intersection points with an ellipse along the [y,z] plane
        """
        ye, ys = np.zeros(shape=(self._nx, self._nx)), np.zeros(shape=(self._nx, self._nx))
        ze, zs = np.zeros(shape=(self._nx, self._nx)), np.zeros(shape=(self._nx, self._nx))
        xe = np.zeros(shape=(self._nx, self._nx))
        """
        rmax is the maximum radius that should contain most of the density of the disk.
        It should be a * (1 - e**2) / (1-e) which simplyfies as a*(1+e)
        """
        rmax = self._threshold**(1.e0 / self.pout) * self._a * (1.e0 + self._e)
        radius = rmax**2. - self._xm**2.
        radius[(radius<0)] = 0.
        radius = np.sqrt(radius)
        height = 3. * radius * self._to
        """
        Here I am solving the intersections of two equations:
          + (y/radius)**2. + (z/height)**2. = 1.
          + z = (y - ym) / tan(inclination)
        The first one is an ellipse and the second one is the line of sight intercepting the midplane at ym.
        Replacing z in the first equation with the expression from the second equation yields a second
        degree equation.
        """
        delta = self._ym**2. * self._st2**2. * radius ** 4. - (height**2. + self._st2 * radius**2.) * (self._st2 * self._ym**2. * radius**2. - height**2. * radius**2.)
        sel = (delta > 0.)
        ye[sel] = ((self._ym[sel] * self._st2 * radius[sel]**2.) + np.sqrt(delta[sel])) / (height[sel]**2. + self._st2 * radius[sel]**2.)
        ys[sel] = ((self._ym[sel] * self._st2 * radius[sel]**2.) - np.sqrt(delta[sel])) / (height[sel]**2. + self._st2 * radius[sel]**2.)
        ze[sel] = self._st * (ye[sel] - self._ym[sel])
        zs[sel] = self._st * (ys[sel] - self._ym[sel])
        xe[sel] = self._xm[sel]

        return xe, ye, ze, ys, zs

    """
    Compute the model for the total and polarized intensity.
    """
    def compute_model(self, **kwargs):
        """
        Method to compute a synthetic image of a debris disk

        INPUTS:
          - a: reference semi-major axis in arcseconds
          - incl: inclination in degrees
          - pa: position angle in degrees
          - pin: inner slope
          - pout: outer slope
          - gsca: HG coefficient for total intensity
          - gpol: HG coefficient for polarized intensity
          - e: eccentricity
          - omega: argument of pericenter in degrees
          - opang: opening angle of the disk
          - s11: the scattered light phase function
          - s12: the polarized light phase function
        """
        """
        Check the input parameters
        And put the angles in radians
        """
        self._check_parameters(kwargs)
        """
        Compute some cos, sin, and tan
        """
        self._trigonometry()
        """
        Need to re-initialize the arrays to zero, in case I compute two models one after the other
        """
        self.polarized = np.zeros(shape=(self._nx, self._nx))
        self.intensity = np.zeros(shape=(self._nx, self._nx))
        """
        Project the disk according to the inclination and position angle
        """
        self._xm = ((np.cos(self._pa) * self._Xin + np.sin(self._pa) * self._Yin))
        self._ym = ((np.sin(self._pa) * self._Xin - np.cos(self._pa) * self._Yin)) / self._cs
        self.distance = np.sqrt(self._xm**2. + self._ym**2.)
        if self._nx % 2 == 1:
            self.distance[self._cx, self._cx] = 1. # This is to avoid a division by zero when computing the scattering angle in the midplane.
        """
        Compute the azimuth and scattering angles in the disk midplane, so
        that they can be used outside of the class.

        The azimuth angle should be zero along the projected minor axis of the disk.
        """
        self.azimuth = (np.arctan2(self._ym,self._xm) + np.pi/2.) % (2. * np.pi) - np.pi
        self.scattering = np.arccos((self._ss * self._ym)/self.distance)
        """
        Get the entry and exit points for the upper and lower layers
        """
        xe, ye, ze, ys, zs = self._get_sphere()
        """
        Compute the final image
        """
        self._get_flux(xe, ye, ze, ys, zs)

    """
    Method to compute the emission between the entry and exit points.
    """
    def _get_flux(self, xe, ye, ze, ys, zs):
        """
        The volume of each cells
        The line of sight is divided as follows:
        for i in range(nframe):
            zi = (ze-zs) * i / (nframe - 1) + zs
        So then I can compute the lenght of a given cell which is (ze-zs)/(nframe -1)
        And the volume of one cell is sqrt(Delta_z**2. + Delta_y**2) * pixelscale**2.
        """
        volume = np.sqrt(((ze-zs)**2. + (ye-ys)**2.))/(self._nframe-1.)*self._pixscale**2.
        for i in range(self._nframe-1):
            """
            Define some variables, that need to be re-initialized for every iteration
            This slows down the code a bit, but it is more accurate and prevents some unfortunate
            NaNs in some cases (especially if the position angle is 0 degrees for instance.

            I had tried to define them once, and set them to 0., but that did not improve the time.
            For clarity, I leave it as it is now.
            """
            densr = np.zeros(shape=(self._nx, self._nx))
            densz = np.zeros(shape=(self._nx, self._nx))
            """
            Get the middle points of each cells.
            """
            zi = zs + (ze - zs) * (i+0.5) / (self._nframe-1)
            yi = ys + (ye - ys) * (i+0.5) / (self._nframe-1)
            """
            Compute the 2D and 3D distances, with their respective
            selection where they are different than 0
            """
            dist2d = np.sqrt(xe**2. + yi**2.)
            sel2d = (dist2d > 0.)
            dist3d = np.sqrt(xe**2. + yi**2. + zi**2.)
            sel3d = (dist3d > 0.)
            """
            Define the radial and vertical density structure.
            For the radial one, I need the azimuthal angle at the midplane, which is np.arctan2(yi, xe)
            """
            r_ref = self._a * (1.e0 - self._e * self._e) / (1.e0 + self._e * np.cos(np.arctan2(yi, xe) + self._omega))
            if self._gaussian:
                densr[sel2d] = np.exp(-(r_ref[sel2d] - dist2d[sel2d])**2. / (2 * self._dr**2.))
            else:
                if self._pmid is None:
                    densr[sel2d] = ((dist2d[sel2d]/r_ref[sel2d])**(- self._nm * self._pout) + (dist2d[sel2d]/r_ref[sel2d])**(- self._nm * self._pin))**(-1./self._nm)
                else:
                    r_ref2 = (self._a + self._da) * (1.e0 - self._e * self._e) / (1.e0 + self._e * np.cos(np.arctan2(yi, xe) + self._omega))
                    dens2 = np.zeros(shape=(self._nx, self._nx))
                    dens3 = np.zeros(shape=(self._nx, self._nx))
                    dens2[sel2d] = ((dist2d[sel2d]/r_ref[sel2d])**(-self._nm*self._pin) + (dist2d[sel2d]/r_ref[sel2d])**(-self._nm * self._pmid))**(-1./self._nm)
                    dens3[sel2d] = dens2[sel2d] * (dist2d[sel2d]/r_ref2[sel2d])**(self._pout-self._pmid)
                    densr[sel2d] = (dens2[sel2d]**(-self._nm) + dens3[sel2d]**(-self._nm))**(-1./self._nm)
                    densr[sel2d] = densr[sel2d] / np.max(densr[sel2d])
            if self._engler:
                densz[sel2d] = np.exp(-np.log(2) * zi[sel2d]**2 / ((self._to * dist2d[sel2d])**2.))
            else:
                densz[sel2d] = np.exp(-(np.abs(zi[sel2d]) / (self._to * dist2d[sel2d]))**self._gamma)
            # densz[sel2d] = np.exp(-zi[sel2d]**2 / (2. * (self._to * dist2d[sel2d])**2.))
            if self._thermal:
                temperature = np.zeros(shape=(self._nx, self._nx))
                expterm = np.zeros(shape=(self._nx, self._nx))
                bplanck = np.zeros(shape=(self._nx, self._nx))
                """
                Following Eq. 3 of Wyatt 2008, without the stellar luminosity
                """
                temperature[sel3d] = 278.3 / np.sqrt(dist3d[sel3d] * self._dpc)
                expterm[sel3d] = np.exp(self._nu * HH / KK / temperature[sel3d])
                bplanck[sel3d] = 2. * HH * self._nu**3. / (expterm[sel3d]-1.0) / (CC * CC)
                self.intensity[sel3d] += densr[sel3d] * densz[sel3d] * volume[sel3d] * bplanck[sel3d]
            else:
                psca = np.zeros(shape=(self._nx, self._nx, 2))
                ppol = np.zeros(shape=(self._nx, self._nx, 2))
                theta = np.zeros(shape=(self._nx, self._nx))
                costheta = np.zeros(shape=(self._nx, self._nx))
                density = np.zeros(shape=(self._nx, self._nx))
                """
                Compute the cosine of the scattering angle
                I need to make a selection where dist3d != 0, to avoid division by 0 when computing it.
                """
                if self.theta is None:
                    costheta[sel3d] = (self._ss * yi[sel3d] - self._cs * zi[sel3d]) / dist3d[sel3d]
                    hg = (1.e0 - self._gsca**2.) / (4. * np.pi * (1.e0 + self._gsca**2. - 2. * self._gsca * costheta[sel3d])**(1.5))
                    phg = (1.e0 - self._gpol**2.) / (4. * np.pi * (1.e0 + self._gpol**2. - 2. * self._gpol * costheta[sel3d])**(1.5)) * (1.e0 - costheta[sel3d]**2.) / (1.e0 + costheta[sel3d]**2.)
                    psca[sel3d,0] = hg
                    psca[sel3d,1] = hg
                    ppol[sel3d,0] = phg
                    ppol[sel3d,1] = phg
                else:
                    """
                    The interpolation method needs self.theta to be in increasing order, so I
                    cannot use costheta directly.
                    """
                    theta[sel3d] = np.arccos((self._ss * yi[sel3d] - self._cs * zi[sel3d]) / dist3d[sel3d])
                    psca[sel3d,0] = np.interp(theta[sel3d], self.theta, self.s11[:,0])
                    psca[sel3d,1] = np.interp(theta[sel3d], self.theta, self.s11[:,1])
                    ppol[sel3d,0] = np.interp(theta[sel3d], self.theta, self.s12[:,0])
                    ppol[sel3d,1] = np.interp(theta[sel3d], self.theta, self.s12[:,1])
                """
                This is the number density for the scatterd light images
                """
                density[sel3d] = densr[sel3d] * densz[sel3d] * volume[sel3d] / (dist3d[sel3d]**2.)
                """
                Differentiate the north and south sides of the disk
                """
                sel = (self.azimuth <=0.)
                self.intensity[sel] += density[sel] * psca[sel,0]
                self.intensity[~sel] += density[~sel] * psca[~sel,1]
                self.polarized[sel] += density[sel] * ppol[sel,0]
                self.polarized[~sel] += density[~sel] * ppol[~sel,1]
                """
                Some debugging plots
                """
                #tmp = np.zeros(shape=(self._nx, self._nx))
                #tmp[sel] = density[sel] * psca[sel,0]
                #tmp[~sel] = density[~sel] * psca[~sel,1]

                #xlim = self._cx * self._pixscale
                #fig = plt.figure(figsize=(7,7 * 9./16.))
                #ax1 = fig.add_axes([0.0, 0.0, 1., 1.])
                #ax1.imshow(tmp, origin = 'lower', vmin = 0., vmax = 3.5e-7, cmap = 'inferno', extent=[xlim, -xlim, -xlim, xlim])
                #ax1.plot(0.,0., marker = '+', ms = 6 , color = 'w')
                #ax1.set_xlim(xlim, -xlim)
                #ax1.set_ylim(-xlim * 9./16., xlim * 9./16.)
                #ax1.axis('off')
                #plt.savefig('debug/image_'+format(i,'04d')+'.png', edgecolor='black', dpi=100, facecolor='black')
                ##if i==0:
                    ##plt.show()
                #plt.close()
        """
        For thermal emission, the flux is normalized to the total flux in the image,
        so that one can fit directly for the total flux in the image. Note that if not
        all the emission is contained in the image, you will miss some of the flux.
        """
        if self._thermal:
            self.intensity /= np.sum(self.intensity)

    """
    Plot the images
    """
    def plot(self, cmap = 'inferno'):
        """
        Method to make some plot
        """
        if self._thermal:
            fig = plt.figure(figsize=(7,7))
            ax1 = fig.add_axes([0.16, 0.14, 0.8, 0.79])
            im = ax1.imshow(self.intensity, origin = 'lower', extent = [self._xlim, -self._xlim, -self._xlim, self._xlim], cmap = cmap, vmin = np.percentile(self.intensity, 1.), vmax = np.percentile(self.intensity, 99.9))
            ax1.set_xlabel('$\Delta \\alpha$ [$^{\prime\prime}$]')
            ax1.set_ylabel('$\Delta \delta$ [$^{\prime\prime}$]')
            plt.show()
        else:
            fig = plt.figure(figsize=(13,6))
            ax1 = fig.add_axes([0.1, 0.12, 0.4, 0.8])
            im = ax1.imshow(self.intensity, origin = 'lower', extent = [self._xlim, -self._xlim, -self._xlim, self._xlim], cmap = cmap, vmin = np.percentile(self.intensity, 1.), vmax = np.percentile(self.intensity, 99.9))
            ax1.set_xlabel('$\Delta \\alpha$ [$^{\prime\prime}$]')
            ax1.set_ylabel('$\Delta \delta$ [$^{\prime\prime}$]')

            ax2 = fig.add_axes([0.55, 0.12, 0.4, 0.8])
            im = ax2.imshow(self.polarized, origin = 'lower', extent = [self._xlim, -self._xlim, -self._xlim, self._xlim], cmap = cmap, vmin = np.percentile(self.polarized, 1.), vmax = np.percentile(self.polarized, 99.9))
            ax2.set_xlabel('$\Delta \\alpha$ [$^{\prime\prime}$]')
            plt.show()

    """
    Check the parameters that are passed as kwargs
    """
    def _check_parameters(self, kwargs):
        """
        Check parameters that are being passed
        """
        if 'a' in kwargs:
            self._a = kwargs['a']
        if 'da' in kwargs:
            self._da = kwargs['da']
        if 'incl' in kwargs:
            self._incl = kwargs['incl'] * np.pi / 180.
        if 'pa' in kwargs:
            self._pa = -kwargs['pa'] * np.pi / 180.
        if 'pin' in kwargs:
            self._pin = kwargs['pin']
        if 'pmid' in kwargs:
            self._pmid = kwargs['pmid']
        if 'pout' in kwargs:
            self._pout = kwargs['pout']
        if 'gamma' in kwargs:
            self._gamma = kwargs['gamma']
        if 'gsca' in kwargs:
            self._gsca = kwargs['gsca']
        if 'gpol' in kwargs:
            self._gpol = kwargs['gpol']
        if 'e' in kwargs:
            self._e = kwargs['e']
        if 'omega' in kwargs:
            self._omega = kwargs['omega'] * np.pi / 180.
        if 'opang' in kwargs:
            self._opang = kwargs['opang']
        if 's11' in kwargs:
            self.s11 = kwargs['s11']
            self.theta = np.linspace(0., np.pi, num = np.shape(self.s11)[0])
            if 's12' not in kwargs: self.s12 = np.ones(shape=(np.shape(self.s11)[0],2))
        if 's12' in kwargs:
            self.s12 = kwargs['s12']
            self.theta = np.linspace(0., np.pi, num = np.shape(self.s12)[0])
            if 's11' not in kwargs: self.s11 = np.ones(shape=(np.shape(self.s12)[0],2))
        if 'wave' in kwargs:
            self._nu = CC / kwargs['wave']
        if ((self._thermal) and (self._nu is None)):
            self._error_msg('To compute thermal images, you need to pass a wavelength \'wave\' in units of microns.')
        if ((self._pmid is not None) and (self._da is None)):
            self._error_msg('If you need a third power-law you need to provide \'da\'')

    """
    Define some cos, sin, and tan
    """
    def _trigonometry(self):
        """
        Define some variables
        """
        if self._incl == 0.:
            self._incl = 0.1 * np.pi / 180.
        if self._incl == np.pi/2.:
            self._incl = 89.9 * np.pi / 180.
        self._cs, self._ss = np.cos(self._incl), np.sin(self._incl)
        self._st = self._cs / self._ss
        self._st2 = self._st**2.
        self._to = np.tan(self._opang)

    """
    Print some error message and quit
    """
    def _error_msg(self, message):
        """
        Print something and quit the program.
        """
        print(message)
        sys.exit()

if __name__ == '__main__':
    disk = Disk(nframe = 50, thermal = False, dpc= 71.)
    t0 = time.time()
    disk.compute_model(e = 0.2, incl = 70.1, pa = 110., a = 0.89, gsca = 0.4, gpol = 0.6, omega = 80., opang = 0.035, pin = 20.0, pout = -5.5, pmid = 0.5, da = 0.5)
    print('Took: ' + format(time.time()-t0, '0.2f') + ' seconds.')
    disk.plot()

