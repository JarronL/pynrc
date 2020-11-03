#! /usr/bin/env python

# This script converts the fits files from the NIRCam CRYO runs
# into ssb-conform fits files.

import sys, os,re,math
import optparse,scipy
from jwst import datamodels as models
from astropy.io import fits as pyfits
import numpy as np






class nircam2ssbclass:
    def __init__(self):
        self.version = 1.0

        self.runID=None

        self.outputmodel=None
        self.data = None
        self.hdr = None

        #dictionary to translate between part number and detector/channel/module
        self.part2mod = {}
        self.modApartIDs = ['16989','17023','17024','17048','17158','C072','C067','C104','C073','C090',481,482,483,484,485]
        self.modBpartIDs = ['16991','17005','17011','17047','17161','C045','C043','C101','C044','C084',486,487,488,489,490]
        for i in range(len(self.modApartIDs)):
            self.part2mod[self.modApartIDs[i]]={}
            self.part2mod[self.modBpartIDs[i]]={}
            self.part2mod[self.modApartIDs[i]]['module']='A'
            self.part2mod[self.modBpartIDs[i]]['module']='B'
            if i == 4 or i == 9 or i==14:
                self.part2mod[self.modApartIDs[i]]['channel']='LONG'
                self.part2mod[self.modApartIDs[i]]['detector'] = 'NRCALONG'
                self.part2mod[self.modBpartIDs[i]]['channel']='LONG'
                self.part2mod[self.modBpartIDs[i]]['detector'] = 'NRCBLONG'
            elif i < 4:
                self.part2mod[self.modApartIDs[i]]['channel']='SHORT'
                self.part2mod[self.modApartIDs[i]]['detector']='NRCA'+str(i+1)
                self.part2mod[self.modBpartIDs[i]]['channel']='SHORT'
                self.part2mod[self.modBpartIDs[i]]['detector']='NRCB'+str(i+1)
            elif i > 4 and i < 9:
                self.part2mod[self.modApartIDs[i]]['channel']='SHORT'
                self.part2mod[self.modApartIDs[i]]['detector']='NRCA'+str(i+1-5)
                self.part2mod[self.modBpartIDs[i]]['channel']='SHORT'
                self.part2mod[self.modBpartIDs[i]]['detector']='NRCB'+str(i+1-5)
            elif i > 9 and i < 14:
                self.part2mod[self.modApartIDs[i]]['channel']='SHORT'
                self.part2mod[self.modApartIDs[i]]['detector']='NRCA'+str(i+1-10)
                self.part2mod[self.modBpartIDs[i]]['channel']='SHORT'
                self.part2mod[self.modBpartIDs[i]]['detector']='NRCB'+str(i+1-10)



    def add_options(self, parser=None, usage=None):
        if parser == None:
            parser = optparse.OptionParser(usage=usage, conflict_handler="resolve")

        parser.add_option('-v', '--verbose', action="count", dest="verbose",default=0)
        parser.add_option('-o','--outfilebasename'  , default='auto' , type="string",
                          help='file basename of output file. If \'auto\', then basename is input filename with fits removed (default=%default)')
        parser.add_option('-d','--outdir'  , default=None , type="string",
                          help='if specified output directory (default=%default)')
        parser.add_option('-s','--outsubdir'  , default=None , type="string",
                          help='if specified gets added to output directory (default=%default)')
        parser.add_option('--outsuffix'  , default=None , type="string",
                          help='if specified: output suffix, otherwise _uncal.fits (default=%default)')
        return(parser)


    def copy_comments(self,filename):
        incomments = self.hdr['COMMENT']
        return

    def copy_history(self,filename):
        return


    def mkoutfilebasename(self,filename, outfilebasename='auto',outdir=None,outsuffix=None,outsubdir=None):
        if  outfilebasename.lower() == 'auto':
            outfilebasename = re.sub('\.fits$','',filename)
            if outfilebasename==filename:
                raise RuntimeError('BUG!!! %s=%s' % (outfilebasename,filename))

        # new outdir?
        if outdir!=None:
           (d,f)=os.path.split(outfilebasename)
           outfilebasename = os.path.join(outdir,f)

        # append suffix?
        if outsuffix!=None:
            outfilebasename += '.'+outsuffix

        # add subdir?
        if outsubdir!=None:
            (d,f)=os.path.split(outfilebasename)
            outfilebasename = os.path.join(d,outsubdir,f)

        # make sure output dir exists
        dirname = os.path.dirname(outfilebasename)
        if dirname!='' and not os.path.isdir(dirname):
            os.makedirs(dirname)
            if not os.path.isdir(dirname):
                raise RuntimeError('ERROR: Cannot create directory %s' % dirname)

        return(outfilebasename)

    def cryo_update_meta_detector(self,runID=None,filename=None,reffileflag=True):

        if runID==None:
            runID=self.runID

        if runID=='TUCSONNEW':
            self.outputmodel.meta.instrument.module = self.hdr['MODULE']
            if self.hdr['DETECTOR']=='SW':
                self.outputmodel.meta.instrument.channel = 'SHORT'
            elif self.hdr['DETECTOR']=='LW':
                self.outputmodel.meta.instrument.channel = 'LONG'
            else:
                raise RuntimeError('wrong DETECTOR=%s' % self.hdr['DETECTOR'])
            self.outputmodel.meta.instrument.detector = 'NRC%s%d' % (self.outputmodel.meta.instrument.module,self.hdr['SCA'])

            print('TEST!!!',self.outputmodel.meta.instrument.module,self.outputmodel.meta.instrument.channel,self.outputmodel.meta.instrument.detector)
        elif runID=='TUCSON_PARTNUM':
            idInFilename = filename[0:5]
            self.outputmodel.meta.instrument.detector = self.part2mod[idInFilename]['detector']
            self.outputmodel.meta.instrument.channel = self.part2mod[idInFilename]['channel']
            self.outputmodel.meta.instrument.module = self.part2mod[idInFilename]['module']
        elif runID=='CRYO2' or runID=='CRYO3':
            detectorname=self.hdr['DETECTOR']
            self.outputmodel.meta.instrument.filetype= 'UNCALIBRATED'
            if re.search('^NRCA',detectorname):
                self.outputmodel.meta.instrument.module = 'A'
            elif  re.search('^NRCB',detectorname):
                self.outputmodel.meta.instrument.module = 'B'
            else:
                raise RuntimeError('wrong DETECTOR=%s' % detectorname)

            if re.search('LONG$',detectorname):
                self.outputmodel.meta.instrument.channel = 'LONG'
            else:
                self.outputmodel.meta.instrument.channel = 'SHORT'
            self.outputmodel.meta.instrument.detector = self.hdr['DETECTOR']
            print(self.outputmodel.meta.instrument.module)
            print(self.outputmodel.meta.instrument.channel)
            print(self.outputmodel.meta.instrument.detector)
        elif runID=='CV2':
            if 'TLDYNEID' in self.hdr:
                detectorname=self.hdr['TLDYNEID']
            elif 'SCA_ID' in self.hdr:
                detectorname=self.hdr['SCA_ID']
            else:
                print('ERROR! could not get detector!!!')
                sys.exit(0)

            self.outputmodel.meta.instrument.detector = self.part2mod[detectorname]['detector']
            self.outputmodel.meta.instrument.channel = self.part2mod[detectorname]['channel']
            self.outputmodel.meta.instrument.module = self.part2mod[detectorname]['module']

            # Below three lines added

            if 'DESCRIP' in self.hdr:
               print('DESCRIP already exist')
            elif reffileflag:
               self.outputmodel.meta.reffile.description = self.hdr['DESCRIPT']

            #if reffileflag:
            #    self.outputmodel.meta.reffile.description = self.hdr['DESCRIPT']
            #    #self.outputmodel.meta.reffile.author = self.hdr['AUTHOR']
        elif runID=='CV3':
            if 'SCA_ID' in self.hdr:
                detectorname=self.hdr['SCA_ID']
            else:
                print("ERROR! could not get detector!!!")
            self.outputmodel.meta.instrument.detector = self.part2mod[detectorname]['detector']
            self.outputmodel.meta.instrument.channel = self.part2mod[detectorname]['channel']
            self.outputmodel.meta.instrument.module = self.part2mod[detectorname]['module']

            # Below three lines added

            if 'DESCRIP' in self.hdr:
               print('DESCRIP already exist')
            elif reffileflag:
               self.outputmodel.meta.reffile.description = self.hdr['DESCRIPT']
        elif runID=='OTIS':
            if 'SCA_ID' in self.hdr:
                detectorname=self.hdr['SCA_ID']
            else:
                print("ERROR! could not get detector!!!")
            self.outputmodel.meta.instrument.detector = self.part2mod[detectorname]['detector']
            self.outputmodel.meta.instrument.channel = self.part2mod[detectorname]['channel']
            self.outputmodel.meta.instrument.module = self.part2mod[detectorname]['module']

            # Below three lines added

            if 'DESCRIP' in self.hdr:
               print('DESCRIP already exist')
            elif reffileflag:
               self.outputmodel.meta.reffile.description = self.hdr['DESCRIPT']
        else:
            print('ERROR!!! dont know runID=%s' % runID)
            sys.exit(0)

    def getRunID(self,filename=None,hdr=None):
        if hdr!=None:
            if 'TERROIR' in hdr:
                if hdr['TERROIR']=='ISIM-CV2':
                    runID = 'CV2'
                    return(runID)
                else:
                    print('TERROIR=%s unknown, fix me in nircam2ssb.getRunID!' % hdr['TERROIR'])
                    sys.exit(0)

        if filename!=None:

            basename = os.path.basename(filename)
            if re.search('^Run\d\d\_',filename):
                runID='TUCSONNEW'
            elif filename[0:5] in self.modApartIDs or filename[0:5] in self.modBpartIDs:
                runID='TUCSON_PARTNUM'
            elif filename[6:11] in self.modApartIDs or filename[6:11] in self.modBpartIDs:
                print('VVVVVVVVVVVVVVVVVV',filename)
                if self.hdr['DATE']>'2014-09':
                    runID='CV2'
                else:
                    runID='TUCSON_PARTNUM'
            elif filename[0:4] == 'jwst':
                runID='CRDS'
            elif re.search('cvac1',filename):
                runID = 'CRYO1'
            elif re.search('cvac2',filename):
                runID = 'CRYO2'
            elif re.search('cvac3',filename):
                runID = 'CRYO3'
            elif re.search('SE\_2014',filename):
                runID = 'CV2'
            elif re.search('SE\_2015',filename):
                runID = 'CV3'
            elif re.search('SE\_2016',filename):
                runID = 'CV3'
            elif re.search('SE\_2017',filename):
                runID = 'OTIS'
            elif filename[0:4] in self.modApartIDs or filename[0:4] in self.modBpartIDs:
                runID='OLD_DET'
            else:
                print('FIX ME getRunID!!!!',filename)
                sys.exit(0)
        else:
            print('FIX ME getRunID!!!!',filename)
            sys.exit(0)
        return(runID)

    def updatemetadata_CRYOX(self,runID,filename=None):
        test = self.hdr.get('DATE-OBS',default=-1)
        if test == -1:
            print('DATE-OBS keyword not found.')
            test2 = self.hdr.get('DATE',default=-1)
            if test2 == -1:
                print('DATE keyword also not found. Defaulting to dummy value.')
                self.outputmodel.meta.observation.date = '2000-01-01T00:00:00'
            else:
                if not re.search('T',test2):
                    self.outputmodel.meta.observation.date = '%sT%s' % (test2,'00:00:00')
                else:
                    self.outputmodel.meta.observation.date = test2
        else:
            if not re.search('T',self.hdr['DATE-OBS']):
                tmp = '%sT%s' % (self.hdr['DATE-OBS'],self.hdr['TIME-OBS'])
                self.outputmodel.meta.observation.date = tmp


    def updatemetadata_TUCSONNEW(self,runID,filename=None,reffileflag=True):
        if reffileflag:
            self.outputmodel.meta.reffile.author = 'Misselt'

        test = self.hdr.get('DATE-OBS',default=-1)
        if test == -1:
            print('DATE-OBS keyword not found')
            test2 = self.hdr.get('DATE',default=-1)
            if test2 == -1:
                print('DATE keyword not found. Checking filename as last-ditch effort.')
                test3 = filename[-13:-5]
                if test3[0:4] in ['2011','2012','2013','2014']:
                    print('date string found in filename.')
                    dt = test3[0:4] + '-' + test3[4:6] + '-' + test3[6:8] + 'T00:00:00'
                    print('using: %s' %dt)
                    self.outputmodel.meta.observation.date = dt
                else:
                    print('No date string found in filename check. Using dummy value.')
                    self.outputmodel.meta.observation.date = '2000-01-01T00:00:00'
            else:
                print('DATE keyword found. Using this for DATE-OBS')
                self.outputmodel.meta.observation.date = '%sT%s' % (self.hdr['DATE'],'00:00:00')
        else:
            self.outputmodel.meta.observation.date = '%sT%s' % (self.hdr['DATE-OBS'],'00:00:00')

        print('FIXING DATE',self.outputmodel.meta.observation.date)


    def updatemetadata_CV2(self,runID,filename=None,reffileflag=True):
        if (not reffileflag) and self.outputmodel.meta.observation.date==None:
            #timeobs=re.sub('\.*','',self.hdr['TIME-OBS']
            self.outputmodel.meta.observation.date = '%sT%s' % (self.hdr['DATE-OBS'],self.hdr['TIME-OBS'])

    def updatemetadata_CV3(self,runID,filename=None,reffileflag=True):
        if (not reffileflag) and self.outputmodel.meta.observation.date==None:
            #timeobs=re.sub('\.*','',self.hdr['TIME-OBS']
            self.outputmodel.meta.observation.date = '%sT%s' % (self.hdr['DATE-OBS'],self.hdr['TIME-OBS'])

    def updatemetadata_OTIS(self,runID,filename=None,reffileflag=True):
        if (not reffileflag) and self.outputmodel.meta.observation.date==None:
            #timeobs=re.sub('\.*','',self.hdr['TIME-OBS']
            self.outputmodel.meta.observation.date = '%sT%s' % (self.hdr['DATE-OBS'],self.hdr['TIME-OBS'])

    def updatemetadata(self,filename,runID=None,cpmetadata=True,reffileflag=True):
        if runID==None:
            runID=self.runID

        if cpmetadata:
            # Update output model with meta data from input
            with pyfits.open(filename) as tmp:
                tmp['PRIMARY'].header['SUBARRAY'] = 'FULL'
                tmp.writeto(filename,overwrite=True)
            dummy4hdr = models.DataModel(filename)
            self.outputmodel.update(dummy4hdr) #, primary_only=True)
            dummy4hdr.close()

        print('within nircam2ssb, runID is',runID)
        if runID in ['CRYO1','CRYO2','CRYO3','OLD_DET']:
            self.updatemetadata_CRYOX(runID,filename=filename)
        elif  runID in ['TUCSONNEW','TUCSON_PARTNUM']:
            self.updatemetadata_TUCSONNEW(runID,filename=filename,reffileflag=reffileflag)
        elif  runID in ['CV2']:
            self.updatemetadata_CV2(runID,filename=filename,reffileflag=reffileflag)
        elif runID in ['CV3']:
            self.updatemetadata_CV3(runID,filename=filename,reffileflag=reffileflag)
        elif runID in ['OTIS']:
            self.updatemetadata_CV3(runID,filename=filename,reffileflag=reffileflag)
        else:
            print('ERROR: runID=%s not yet implemented into "updatemetadata"' % runID)
            sys.exit(0)

    def get_subarray_name(self,subarrays,colstart, colstop, rowstart, rowstop):
        for i in np.arange(0,len(subarrays)):
            subarray_row = subarrays[i]
            if rowstart == subarray_row['xstart'] and rowstop == subarray_row['xend'] and colstart == subarray_row['ystart'] and colstop == subarray_row['yend']:
                return subarray_row['Name']
        return 'UNKNOWN'

    def image2ssb(self,inputfilename, outfilebasename='auto',outdir=None,outsuffix=None,outsubdir=None):
        outfilebasename = self.mkoutfilebasename(inputfilename, outfilebasename=outfilebasename,outdir=outdir,outsuffix=outsuffix,outsubdir=outsubdir)
        return(outfilebasename)

if __name__=='__main__':

    usagestring='USAGE: nircam2ssb.py infile1 infile2 ...'

    nircam2ssb=nircam2ssbclass()
    parser = nircam2ssb.add_options(usage=usagestring)
    options,  args = parser.parse_args()

    if len(args)<1:
        parser.parse_args(['--help'])
        sys.exit(0)

    nircam2ssb.verbose=options.verbose

    for infile in args:
        nircam2ssb.image2ssb(infile,outfilebasename=options.outfilebasename,outdir=options.outdir,outsuffix=options.outsuffix,outsubdir=options.outsubdir)
