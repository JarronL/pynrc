#! /usr/bin/env python

'''
Convert a given file to a Level1b file.
This has been created essentially to remove
empty ERR and DQ extensions from uncal files
that may have been created with RampModel.

This conversion will also remove any non-Level1b
metadata before saving the Level1b file. This has 
the advantage of removing any ground testing 
keywords that should not be made public.

Inputs:

   infile: Name of fits file to be converted
   datamodel: Name of data model to place into the
              DATAMODL header keyword. This controls
              what type of data model is used to open
              the file in the event that it is opened
              using datamodels.open(). In order to 
              run the outputs through the JWST
              calibration pipeline properly, this
              needs to be reset from the default of
              'Level1bModel' (which is set upon saving
              the Level1bModel instance) to 'RampModel'.

Optional Inputs:

   outfile: Name of output file saved as Level1b.
            If not provided, a suffix will be 
            added to the input file name.

Outputs:
   Level1b fits file
'''

import argparse
from astropy.io import fits
from jwst.datamodels import Level1bModel

class Level1b:

    def __init__(self):
        self.infile = ''
        self.outfile = None
        self.datamodel = 'RampModel'
        
        
    def convert(self):
        # Perform the conversion
        model = Level1bModel(self.infile)

        # Tests have shown that we need to explicitly
        # not include extensions other than data and
        # zeroframe. e.g if ERR extension is present,
        # it will be saved to the new file if we were
        # to simply save the Level1bModel instance.
        # So, create a new, empty instance, and populate
        # only the extensions we want.
        outmodel = Level1bModel()

        # Populate the new model instance
        outmodel.data = model.data

        # Bug in Level1bModel at the moment. If zeroframe
        # is not present, it's grabbing the wrong dimensions
        # to make one.
        print("Zeroframe is being manually added!!!!!")
        print("Make sure this is ok! If the data are not")
        print("RAPID, it will be incorrect!")
        #outmodel.zeroframe = model.zeroframe
        outmodel.zeroframe = model.data[:,0,:,:]
        outmodel.meta = model.meta

        # Save the updated model instance
        if self.outfile is None:
            try:
                self.outfile = self.infile.replace('uncal.fits',\
                                                   'level1b_uncal.fits')
            except:
                self.outfile = self.infile.replace('.fits','level1b.fits')

        outmodel.save(self.outfile)

        # Now we optionally update the DATAMODL header keyword,
        # which needs to be 'RampModel' in order to run the
        # file through the JWST Calibration pipeline
        if self.datamodel != 'Level1bModel':
            h = fits.open(self.outfile, mode='update')
            h[0].header['DATAMODL'] = self.datamodel
            h.flush()
            
    def add_options(self,parser = None, usage = None):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,\
                                             description='Convert fits file to Level1bModel')
        parser.add_argument("infile",help='File to convert to Level1b')
        parser.add_argument("--outfile",help='Output filename',default = None)
        parser.add_argument("--datamodel",help='Value to place in DATAMODL header keyword',default = 'RampModel')
        return parser

if __name__ == '__main__':
    usagestring = 'USAGE: convert_to_Level1b.py myfile.fits'
    mod = Level1b()
    parser = mod.add_options(usage = usagestring)
    args = parser.parse_args(namespace = mod)
    mod.convert()
