#! /usr/bin/env python

"""Create Mirage-formatted linearized dark ramps for NIRISS starting from raw versions"""

from glob import glob
from mirage.dark import dark_prep

#yamlfile = 'linearize_niriss_dark_001.yaml'

yaml_files = sorted(glob('dark_file_001.yaml'))

for yamlfile in yaml_files:
    d = dark_prep.DarkPrep()
    d.paramfile = yamlfile
    d.prepare()
