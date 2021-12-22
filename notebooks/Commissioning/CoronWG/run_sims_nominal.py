# Import the usual libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import coron_wg
from tqdm.auto import trange, tqdm

filt, mask, pupil = ('F335M', 'MASK335R', 'CIRCLYOT')

import os

# Update output directories
coron_wg.fig_dir           = coron_wg.base_dir + 'output_M335R/'
coron_wg.contrast_maps_dir = coron_wg.base_dir + 'contrast_maps_M335R/'

for path in [coron_wg.fig_dir, coron_wg.contrast_maps_dir]:
    if not os.path.isdir(path):
        os.makedirs(path)
        print(f'Creating: {path}')
    else:
        print(f'Path alreayd exists: {path}')

# For Nominal WFE, cycle through jitters, tacq, and IEC scenarios
imode = 1
for imode_jitt in trange(4, leave=False, desc='Jitter/TACQ'):
    for imode_iec in trange(3, leave=False, desc='IEC'):
        coron_wg.run_obs(filt, mask, pupil, imode=imode, imode_iec=imode_iec, imode_jitt=imode_jitt)

