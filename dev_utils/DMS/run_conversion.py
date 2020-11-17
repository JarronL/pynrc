#! /usr/bin/env python

'''
run level1b conversion tool on all cv3 darks
'''

import convert_to_Level1b
from glob import glob

files = glob('/ifs/jwst/wit/nircam/isim_cv3_files_for_calibrations/darks/*/*uncal.fits')

for file in files:
    c = convert_to_Level1b.Level1b()
    c.infile = file
    c.convert()
