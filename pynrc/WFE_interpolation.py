from astropy.table import Table
from astropy.io import fits
from scipy.interpolate import griddata

import pynrc, webbpsf, os
#inst = webbpsf.NIRCam()

outdir = pynrc.conf.path

data_dir = webbpsf.utils.get_webbpsf_data_path() + '/'
zernike_file = data_dir + 'si_zernikes_isim_cv3.fits'

if not os.path.exists(zernike_file):
    print('Zernike file does not exist:')
    print('  {}'.format(zernike_file))
else:
    ztable_full = Table.read(zernike_file)

    keys = np.array(ztable_full.keys())
    ind_z = ['Zernike' in k for k in keys]
    zkeys = keys[ind_z]

    for mod in ['SW', 'LW', 'SWA', 'LWA', 'SWB', 'LWB']:
        ind_nrc = ['NIRCam'+mod in row['instrument'] for row in ztable_full]

        # Grab V2/V3 coordinates
        # In units of arcsec
        v2 = ztable_full[ind_nrc]['V2']
        v3 = ztable_full[ind_nrc]['V3']

        # Create finer mesh grid
        dstep = 1. / 60. # 1" steps
        xgrid = np.arange(v2.min()-dstep/2, v2.max()+dstep/2, dstep)
        ygrid = np.arange(v3.min()-dstep/2, v3.max()+dstep/2, dstep)
        X, Y = np.meshgrid(xgrid,ygrid)

        # Create a Zernike cube
        zcube = []
        for k in zkeys:
    
            z = ztable_full[ind_nrc][k]

            # There will be some NaNs along the outer borders
            zgrid = griddata((v2, v3), z, (X, Y), method='cubic')
            ind_nan = np.isnan(zgrid)

            # Replace NaNs with nearest neighbors
            zgrid2 = griddata((X[~ind_nan], Y[~ind_nan]), zgrid[~ind_nan], (X,Y), method='nearest')
            zgrid[ind_nan] = zgrid2[ind_nan]

            #ind_min = zgrid<z.min()
            #ind_max = zgrid>z.max()
            #zgrid[ind_min] = z.min()
            #zgrid[ind_max] = z.max()
    
            zcube.append(zgrid)

        zcube = np.array(zcube)

        hdu = fits.PrimaryHDU(zcube)
        hdr = hdu.header

        hdr['units'] = 'meters'
        hdr['xunits'] = 'Arcsec'
        hdr['xmin'] = X.min()
        hdr['xmax'] = X.max()
        hdr['xdel'] = dstep
        hdr['yunits'] = 'Arcsec'
        hdr['ymin'] = Y.min()
        hdr['ymax'] = Y.max()
        hdr['ydel'] = dstep
    
        #hdr['wave'] = 

        hdr['comment'] = 'X and Y values correspond to V2 and V3 coordinates (arcsec).'
        hdr['comment'] = 'Slices in the cube correspond to Zernikes 1 to 36.'
        hdr['comment'] = 'Zernike values calculated using 2D cubic interpolation.'

        fname = 'NIRCam{}_zernikes_isim_cv3.fits'.format(mod)
        hdu.writeto(outdir + fname, overwrite=True)


    #plt.clf()
    #plt.contourf(xgrid,ygrid,zcube[i],20)
    #plt.scatter(v2,v3,marker='o',c='k',s=5)
    #plt.xlim([X.min(),X.max()])
    #plt.ylim([Y.min(),Y.max()])
    #plt.axes().set_aspect('equal', 'datalim')

