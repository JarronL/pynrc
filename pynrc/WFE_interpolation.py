from astropy.table import Table, join
from astropy.io import fits
from scipy.interpolate import griddata, RegularGridInterpolator

import pynrc, webbpsf, os
#inst = webbpsf.NIRCam()

outdir = pynrc.conf.path + 'opd_mod/'

# Read in measured SI Zernike data
data_dir = webbpsf.utils.get_webbpsf_data_path() + '/'
zernike_file = data_dir + 'si_zernikes_isim_cv3.fits'

# Read in Zemax Zernike data to remove edge effects
# zemax_file = outdir + 'si_zernikes_Zemax_wfe.csv'

# Coordinate limits (oversized) for each FoV
v2v3_limits = {}
v2v3_limits['SW'] = {'V2':[-160, 160], 'V3':[-570, -420]}
v2v3_limits['LW'] = v2v3_limits['SW']
v2v3_limits['SWA'] = {'V2':[0, 160], 'V3':[-570, -420]}
v2v3_limits['LWA'] = v2v3_limits['SWA']
v2v3_limits['SWB'] = {'V2':[-160, 0], 'V3':[-570, -420]}
v2v3_limits['LWB'] = v2v3_limits['SWB']

if not os.path.exists(zernike_file):
    print('Zernike file does not exist:')
    print('  {}'.format(zernike_file))
else:
    ztable_full = Table.read(zernike_file)
#     zemax_full = Table.read(zemax_file, format='ascii.csv')
#     zemax_full.rename_column('\ufeffinstrument', 'instrument')
#     
#     for k in zemax_full.keys():
#         zemax_full[k] = zemax_full[k].astype(ztable_full[k].dtype)
        
#     ztable_full = join(ztable_full, zemax_full, join_type='outer')

    keys = np.array(ztable_full.keys())
    ind_z = ['Zernike' in k for k in keys]
    zkeys = keys[ind_z]

#    for mod in ['SW', 'LW', 'SWA', 'LWA', 'SWB', 'LWB']:
    for mod in ['SWA', 'LWA', 'SWB', 'LWB']:
        ind_nrc = ['NIRCam'+mod in row['instrument'] for row in ztable_full]
        ind_nrc = np.where(ind_nrc)

        # Grab V2/V3 coordinates
        # In units of arcmin
        v2 = ztable_full[ind_nrc]['V2']
        v3 = ztable_full[ind_nrc]['V3']

        # Create finer mesh grid
        v2_lims = np.array(v2v3_limits[mod]['V2']) / 60.
        v3_lims = np.array(v2v3_limits[mod]['V3']) / 60.
        dstep = 1. / 60. # 1" steps
        xgrid = np.arange(v2_lims[0], v2_lims[1]+dstep, dstep)
        ygrid = np.arange(v3_lims[0], v3_lims[1]+dstep, dstep)
        X, Y = np.meshgrid(xgrid,ygrid)

        extent = [X.min(), X.max(), Y.min(), Y.max()]

        # Create a Zernike cube
        zcube = []
        for k in zkeys:
    
            z = ztable_full[ind_nrc][k]

            # There will be some NaNs along the outer borders
            zgrid = griddata((v2, v3), z, (X, Y), method='cubic')
            ind_nan = np.isnan(zgrid)
            
            # Cut out a square region whose values are not NaNs
            xnans = ind_nan.sum(axis=0)
            ynans = ind_nan.sum(axis=1)
            x_ind = xnans < ygrid.size
            y_ind = ynans < xgrid.size
            zgrid2 = zgrid[y_ind, :][:, x_ind]
            ygrid2 = ygrid[y_ind]
            xgrid2 = xgrid[x_ind]
            # Remove rows/cols 1 by 1 until no NaNs
            while np.isnan(zgrid2.sum()):
                zgrid2 = zgrid2[1:-1,1:-1]
                ygrid2 = ygrid2[1:-1]
                xgrid2 = xgrid2[1:-1]

            # Create regular grid interpolator function for extrapolation at NaN's
            func = RegularGridInterpolator((ygrid2,xgrid2), zgrid2, method='linear',
                                           bounds_error=False, fill_value=None)

            pts = np.array([Y[ind_nan], X[ind_nan]]).transpose()
            zgrid[ind_nan] = func(pts)
            zcube.append(zgrid)

        zcube = np.array(zcube)

        hdu = fits.PrimaryHDU(zcube)
        hdr = hdu.header

        hdr['units'] = 'meters'
        hdr['xunits'] = 'Arcmin'
        hdr['xmin'] = X.min()
        hdr['xmax'] = X.max()
        hdr['xdel'] = dstep
        hdr['yunits'] = 'Arcmin'
        hdr['ymin'] = Y.min()
        hdr['ymax'] = Y.max()
        hdr['ydel'] = dstep
    
        hdr['wave'] = 2.10 if 'SW' in mod else 3.23

        hdr['comment'] = 'X and Y values correspond to V2 and V3 coordinates (arcmin).'
        hdr['comment'] = 'Slices in the cube correspond to Zernikes 1 to 36.'
        hdr['comment'] = 'Zernike values calculated using 2D cubic interpolation'
        hdr['comment'] = 'and linear extrapolation outside gridded data.'

        fname = 'NIRCam{}_zernikes_isim_cv3.fits'.format(mod)
        hdu.writeto(outdir + fname, overwrite=True)


#     plt.clf()
#     plt.contourf(xgrid,ygrid,zcube[i],20)
#     #plt.imshow(zcube[i], extent = [X.min(), X.max(), Y.min(), Y.max()])
#     plt.scatter(v2,v3,marker='o',c='r',s=5)
#     plt.xlim([X.min(),X.max()])
#     plt.ylim([Y.min(),Y.max()])
#     plt.axes().set_aspect('equal', 'datalim')

# names = ['Guider1', 'Guider2', 'MIRI', 'NIRCamLWA', 'NIRCamLWB', 
# 'NIRCamSWA', 'NIRCamSWB', 'NIRISS', 'NIRSpec']
