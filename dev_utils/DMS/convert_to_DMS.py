#! /usr/bin/env python

"""Run the conversion scripts to convert a raw fits file to DMS format
"""

import os
from glob import glob
import sci2ssb


files = sorted(glob('NRCNRCA2-DARK-53510600011_1_*_SE_2015-12-17T06h11m14.fits'))
output_dir = './'

for filename in files:
    basename = os.path.basename(filename)
    basename = basename.replace('.fits', '')
    proc = sci2ssb.sci2ssbclass()
    proc.image2ssb(filename, outfilebasename=basename, outdir=output_dir)
