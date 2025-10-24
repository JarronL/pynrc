import numpy as np
import pynrc

from matplotlib import pyplot as plt
import matplotlib 

pynrc.setup_logging('WARNING', verbose=False)

nrc1 = pynrc.NIRCam(filter='F335M', image_mask='MASK335R', pupil_mask='CIRCLYOT', autogen_coeffs=False)
nrc2 = pynrc.NIRCam(filter='F335M', image_mask='MASK335R', pupil_mask=None, autogen_coeffs=False)

nrc1.fov_pix = 129
nrc2.fov_pix = 129
nrc1._nrc_bg.fov_pix = 129
nrc2._nrc_bg.fov_pix = 129

hdul1 = nrc1.calc_psf()
hdul2 = nrc2.calc_psf()

hdul1_off = nrc1.calc_psf(use_bg_psf=True)
hdul2_off = nrc2.calc_psf(use_bg_psf=True)

im1_on = hdul1[1].data.copy()
im2_on = hdul2[1].data.copy()
im1_off = hdul1_off[1].data.copy()
im2_off = hdul2_off[1].data.copy()

# Scale by off-axis max PSF value
im1_on /= np.max(im1_off)
im2_on /= np.max(im2_off)
# im1_off /= np.max(im1_off)
# im2_off /= np.max(im2_off)

# Plot images in log scale with colorbars
import matplotlib.colors as colors
fig, axes = plt.subplots(1,2, figsize=(12, 6))

norm = colors.LogNorm(vmin=1e-6, vmax=1e-3)
extent = np.array([-1,1,-1,1]) * nrc1.fov_pix * nrc1.pixelscale / 2

ax = axes[0]
pim = ax.imshow(im1_on, norm=norm, cmap='viridis', extent=extent)
ax.set_title('NIRCam with Pupil Mask (On-axis PSF)')
ax.set_xlabel('arcsec')
ax.set_ylabel('arcsec')
fig.colorbar(pim, ax=ax, label='Intensity normalized to Off-axis PSF')

ax = axes[1]
pim = ax.imshow(im2_on, norm=norm, cmap='viridis', extent=extent)
ax.set_title('NIRCam without Pupil Mask (On-axis PSF)')
ax.set_xlabel('arcsec')
ax.set_ylabel('arcsec')
fig.colorbar(pim, ax=ax, label='Intensity normalized to Off-axis PSF')

fig.tight_layout()
