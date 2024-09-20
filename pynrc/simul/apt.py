"""APT conversion procedures"""

import copy
import os
import logging
from pickle import NONE
import re
import argparse
import pkg_resources
import warnings
from collections import OrderedDict
from lxml import etree


import numpy as np
import yaml, json

import pysiaf
from pysiaf import JWST_PRD_VERSION, rotations, Siaf
# Create this once since it takes time to call multiple times
from ..nrc_utils import siaf_nrc
# siaf_nrc = Siaf('NIRCam')
# siaf_nrc.generate_toc()

from astropy.table import Table
from astropy.io import ascii
from astropy.time import Time, TimeDelta
from astropy import units as u

from tqdm.auto import tqdm, trange

from .. import conf
from ..nrc_utils import get_detname
from ..maths.coords import jwst_point

from webbpsf_ext.opds import slew_time

import logging
_log = logging.getLogger('pynrc')

instrument_abbreviations = {'nircam': 'NRC', 'fgs': 'FGS', 'niriss': 'NIS',
                            'nirspec': 'NRS', 'miri': 'MIR'}
NIRCAM_UNSUPPORTED_PUPIL_VALUES = ['GDHS0', 'GDHS60', 'MASKIPR', 'PINHOLES']

NONE_STR = str(None).upper()

class AptInput:
    """Summary

    Attributes:
        exposure_tab (TYPE): Description
        input_xml (str): Description
        observation_list_file (str): Description
        obstab (TYPE): Description
        output_csv (TYPE): Description
        pointing_file (str): Description
    """

    def __init__(self, input_xml=None, pointing_file=None, output_dir=None, output_csv=None,
                 observation_list_file=None):
        self.logger = logging.getLogger('mirage.apt.apt_inputs')

        self.input_xml = input_xml
        self.pointing_file = pointing_file
        self.output_dir = output_dir
        self.output_csv = output_csv
        self.observation_list_file = observation_list_file

        # Locate the module files, so that we know where to look
        # for config subdirectory
        self.config_path = os.path.join(pkg_resources.resource_filename('mirage', ''), 'config')

    def add_epochs(self, intab):
        """NOT CURRENTLY USED"""
        # add information on the epoch of each observation
        # if the user entered a list of epochs, read that in
        default_date = '2020-10-14'

        if self.epoch_list is not None:
            epochs = ascii.read(self.epoch_list, header_start=0, data_start=1)
        else:
            epochs = Table()
            epochs['observation'] = intab['obs_label']
            epochs['date'] = ['2018-10-14'] * len(intab['obs_label'])
            epochs['pav3'] = [0.] * len(intab['obs_label'])

        # insert epoch info for each observation into dictionary
        epoch_start = []
        epoch_pav3 = []
        for obs in intab['obs_label']:
            match = obs == epochs['observation'].data
            if np.sum(match) == 0:
                self.logger.error("No valid epoch line found for observation {}".format(obs))
                self.logger.error('{}'.format(epochs['observation'].data))
                epoch_start.append(default_date)
                epoch_pav3.append(0.)
            else:
                epoch_start.append(epochs['date'][match].data[0])
                epoch_pav3.append(epochs['pav3'][match].data[0])
        intab['epoch_start_date'] = epoch_start
        intab['pav3'] = epoch_pav3
        return intab

    def add_observation_info(self, intab):
        """Add information about each observation.

        Catalog names, dates, PAV3 values, etc., which are retrieved from the observation list
        yaml file.

        Parameters
        ----------
        intab : obj
            astropy.table.Table containing exposure information

        Returns
        -------
        intab : obj
            Updated table with information from the observation list
            yaml file added.

        """
        with open(self.observation_list_file, 'r') as infile:
            self.obstab = yaml.safe_load(infile)

        OBSERVATION_LIST_FIELDS = 'Date PAV3 Filter PointSourceCatalog GalaxyCatalog ' \
                                  'ExtendedCatalog ExtendedScale ExtendedCenter MovingTargetList ' \
                                  'MovingTargetSersic MovingTargetExtended ' \
                                  'MovingTargetConvolveExtended MovingTargetToTrack ' \
                                  'ImagingTSOCatalog GrismTSOCatalog ' \
                                  'BackgroundRate DitherIndex CosmicRayLibrary CosmicRayScale'.split()

        nircam_mapping = {'ptsrc': 'PointSourceCatalog',
                          'galcat': 'GalaxyCatalog',
                          'ext': 'ExtendedCatalog',
                          'extscl': 'ExtendedScale',
                          'extcent': 'ExtendedCenter',
                          'movptsrc': 'MovingTargetList',
                          'movgal': 'MovingTargetSersic',
                          'movext': 'MovingTargetExtended',
                          'movconv': 'MovingTargetConvolveExtended',
                          'solarsys': 'MovingTargetToTrack',
                          'img_tso': 'ImagingTSOCatalog',
                          'grism_tso': 'GrismTSOCatalog',
                          'bkgd': 'BackgroundRate',
                          }

        unique_instrument_names = [name.lower() for name in np.unique(intab['Instrument'])]

        # initialize dictionary keys
        for key in OBSERVATION_LIST_FIELDS:
            intab[key] = []

        if 'nircam' in unique_instrument_names:
            for channel in ['SW', 'LW']:
                for name, item in nircam_mapping.items():
                    key = '{}_{}'.format(channel.lower(), name)
                    intab[key] = []

        # loop over entries in input dictionary
        for index, instrument in enumerate(intab['Instrument']):
            instrument = instrument.lower()

            # retrieve corresponding entry from observation list
            entry = _get_entry(self.obstab, intab['entry_number'][index])

            if instrument == 'nircam':
                # keep the number of entries in the dictionary consistent
                for key in OBSERVATION_LIST_FIELDS:
                    if key in ['Date', 'PAV3', 'Instrument', 'CosmicRayLibrary', 'CosmicRayScale']:
                        value = str(entry[key])
                    else:
                        value = NONE_STR

                    intab[key].append(value)

                for channel in ['SW', 'LW']:
                    for name, item in nircam_mapping.items():
                        key = '{}_{}'.format(channel.lower(), name)
                        if item in 'ExtendedScale ExtendedCenter MovingTargetConvolveExtended BackgroundRate'.split():
                            intab[key].append(entry['FilterConfig'][channel][item])
                        else:
                            intab[key].append(self.full_path(entry['FilterConfig'][channel][item]))

            else:
                for key in OBSERVATION_LIST_FIELDS:
                    value = str(entry[key])

                    # Expand catalog names to contain full paths
                    catalog_names = 'PointSourceCatalog GalaxyCatalog ' \
                                    'ExtendedCatalog MovingTargetList ' \
                                    'MovingTargetSersic MovingTargetExtended ' \
                                    'MovingTargetToTrack ImagingTSOCatalog ' \
                                    'GrismTSOCatalog'.split()
                    if key in catalog_names:
                        value = self.full_path(value)
                    intab[key].append(value)

                # keep the number of entries in the dictionary consistent
                if 'nircam' in unique_instrument_names:
                    for channel in ['SW', 'LW']:
                        for name, item in nircam_mapping.items():
                            key = '{}_{}'.format(channel.lower(), name)
                            intab[key].append(NONE_STR)

        intab['epoch_start_date'] = intab['Date']
        return intab

    def base36encode(self, integer):
        """
        Translate a base 10 integer to base 36

        Parameters
        ----------
        integer : int
            a base 10 integer

        Returns
        -------
        integer : int
            The integer translated to base 36
        """
        chars, encoded = '0123456789abcdefghijklmnopqrstuvwxyz', ''

        while integer > 0:
            integer, remainder = divmod(integer, 36)
            encoded = chars[remainder] + encoded

        return encoded.zfill(2)

    def combine_dicts(self, dict1, dict2):
        """Combine two dictionaries into a single dictionary.

        Parameters
        ----------
        dict1 : dict
            dictionary
        dict2 : dict
            dictionary

        Returns
        -------
        combined : dict
            Combined dictionary
        """
        combined = dict1.copy()
        combined.update(dict2)
        return combined

    def create_input_table(self, verbose=False):
        """
        Main function for creating a table of parameters for each
        exposure

        Parameters
        ----------
        verbose : bool
            If True, extra information is printed to the log
        """
        # Expand paths to full paths
        # self.input_xml = os.path.abspath(self.input_xml)
        # self.pointing_file = os.path.abspath(self.pointing_file)
        if self.output_csv is not None:
            self.output_csv = os.path.abspath(self.output_csv)
        if self.observation_list_file is not None:
            self.observation_list_file = os.path.abspath(self.observation_list_file)

        # if APT.xml content has already been generated during observation list creation
        # (generate_observationlist.py) load it here
        if self.apt_xml_dict is None:
            raise RuntimeError('self.apt_xml_dict is not defined')

        # Read in the pointing file and produce dictionary
        pointing_dictionary = self.get_pointing_info(self.pointing_file, propid=self.apt_xml_dict['ProposalID'][0])

        # Check that the .xml and .pointing files agree
        assert len(self.apt_xml_dict['ProposalID']) == len(pointing_dictionary['obs_num']),\
            ('Inconsistent table size from XML file ({}) and pointing file ({}). Something was not '
             'processed correctly in apt_inputs.'.format(len(self.apt_xml_dict['ProposalID']),
                                                         len(pointing_dictionary['obs_num'])))

        # Combine the dictionaries
        observation_dictionary = self.combine_dicts(self.apt_xml_dict, pointing_dictionary)

        # Add epoch and catalog information
        observation_dictionary = self.add_observation_info(observation_dictionary)

        if verbose:
            self.logger.info('Summary of observation dictionary:')
            for key in observation_dictionary.keys():
                self.logger.info('{:<25}: number of elements is {:>5}'.format(key, len(observation_dictionary[key])))

        # Global Alignment observations need to have the pointing information for the
        # FGS exposures updated
        if 'WfscGlobalAlignment' in observation_dictionary['APTTemplate']:
            observation_dictionary = self.global_alignment_pointing(observation_dictionary)

        # Expand the dictionary to have one entry for each detector in each exposure
        self.exposure_tab = self.expand_for_detectors(observation_dictionary)

        # For fiducial point overrides, save the pointing aperture and actual aperture separately
        self.check_aperture_override()

        # Add start times for each exposure
        # Ignore warnings as astropy.time.Time will give a warning
        # related to unknown leap seconds if the date is too far in
        # the future.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.exposure_tab = _make_start_times(self.exposure_tab)

        # Fix data for filename generation
        # Set parallel seq id
        for j, isparallel in enumerate(self.exposure_tab['ParallelInstrument']):
            if isparallel:
                self.exposure_tab['sequence_id'][j] = '2'

        # set exposure number (new sequence for every combination of seq id and act id and observation number and detector)
        temp_table = Table([self.exposure_tab['sequence_id'], self.exposure_tab['exposure'], self.exposure_tab['act_id'], self.exposure_tab['obs_num'], self.exposure_tab['detector']], names=('sequence_id', 'exposure', 'act_id', 'obs_num', 'detector'))

        for obs_num in np.unique(self.exposure_tab['obs_num']):
            for act_id in np.unique(temp_table['act_id']):
                # prime_index = np.where((temp_table['sequence_id'] == '1') & (temp_table['act_id'] == act_id) & (temp_table['obs_num'] == obs_num))[0]
                # parallel_index = np.where((temp_table['sequence_id'] == '2') & (temp_table['act_id'] == act_id) & (temp_table['obs_num'] == obs_num))[0]
                for detector in np.unique(temp_table['detector']):
                    prime_index = np.where((temp_table['sequence_id'] == '1') & (temp_table['act_id'] == act_id) & (temp_table['obs_num'] == obs_num) & (temp_table['detector']==detector))[0]
                    parallel_index = np.where((temp_table['sequence_id'] == '2') & (temp_table['act_id'] == act_id) & (temp_table['obs_num'] == obs_num) & (temp_table['detector']==detector))[0]

                    temp_table['exposure'][prime_index] = ['{:05d}'.format(n+1) for n in np.arange(len(prime_index))]
                    temp_table['exposure'][parallel_index] = ['{:05d}'.format(n+1) for n in np.arange(len(parallel_index))]
        self.exposure_tab['exposure'] = list(temp_table['exposure'])

        if verbose:
            for key in self.exposure_tab.keys():
                self.logger.info('{:>20} has {:>10} items'.format(key, len(self.exposure_tab[key])))

        # Create a pysiaf.Siaf instance for each instrument in the proposal
        self.siaf = {}
        for instrument_name in np.unique(observation_dictionary['Instrument']):
            self.siaf[instrument_name] = Siaf(instrument_name)

        # Output to a csv file.
        if self.output_csv is None:
            indir, infile = os.path.split(self.input_xml)
            self.output_csv = os.path.join(self.output_dir, 'Observation_table_for_' + infile.split('.')[0] + '.csv')
        ascii.write(Table(self.exposure_tab), self.output_csv, format='csv', overwrite=True)
        self.logger.info('csv exposure list written to {}'.format(self.output_csv))

    def check_aperture_override(self):
        if bool(self.exposure_tab['FiducialPointOverride']) is True:
            instruments = self.exposure_tab['Instrument']
            apertures = self.exposure_tab['aperture']

            aperture_key = instrument_abbreviations

            fixed_apertures = []
            for i, (instrument, aperture) in enumerate(zip(instruments, apertures)):
                inst_match_ap = aperture.startswith(aperture_key[instrument.lower()])
                if not inst_match_ap:
                    # Handle the one case we understand, for now
                    if instrument.lower() == 'fgs' and aperture[:3] == 'NRC':
                        obs_num = self.exposure_tab['obs_num'][i]

                        if self.exposure_tab['APTTemplate'][i] == 'WfscGlobalAlignment':
                            guider_number = self.exposure_tab['aperture'][i][3]
                        elif self.exposure_tab['APTTemplate'][i] == 'FgsExternalCalibration':
                            guider_number = _get_guider_number(self.input_xml, obs_num)
                        else:
                            raise ValueError("WARNING: unsupported APT template with Fiducial Override.")
                        guider_aperture = 'FGS{}_FULL'.format(guider_number)
                        fixed_apertures.append(guider_aperture)
                    else:
                        self.logger.error('{} {} {} {}'.format(instrument, aperture, inst_match_ap, aperture_key[instrument.lower()]))
                        raise ValueError('Unknown FiducialPointOverride in program. Instrument = {} but aperture = {}.'.format(instrument, aperture))
                else:
                    fixed_apertures.append(aperture)

            # Add new dictionary entry to document the FiducialPointOverride (pointing aperture)
            self.exposure_tab['pointing_aperture'] = self.exposure_tab['aperture']
            # Rewrite the existing imaging aperture entry to match the primary instrument
            self.exposure_tab['aperture'] = fixed_apertures
        else:
            # Add new dictionary entry to document that the imaging aperture is the
            # same as the pointing aperture
            self.exposure_tab['pointing_aperture'] = self.exposure_tab['aperture']

    def expand_for_detectors(self, input_dictionary):
        """Expand dictionary to have one entry per detector, rather than the
        one line per module that is in the input

        Parameters
        ----------
        input_dictionary : dict
            dictionary containing one entry per module

        Returns
        -------
        observation_dictionary : dict
            dictionary expanded to have one entry per detector
        """
        observation_dictionary = {}
        for key in input_dictionary:
            observation_dictionary[key] = []
        observation_dictionary['detector'] = []

        # Read in tables of aperture information
        nircam_subarray_file = 'NIRCam_subarray_definitions.list'
        file = os.path.join(conf.PYNRC_PATH, 'sim_params', nircam_subarray_file)
        nircam_apertures = read_subarray_definition_file(file)

        for index, instrument in enumerate(input_dictionary['Instrument']):
            instrument = instrument.lower()
            if instrument == 'nircam' and input_dictionary['Mode'][index] == 'coron':
                # For coronagraphic observations, there is no need to expand for detectors.
                # Coronographic observations will always use only a single detector, which
                # we already know from the 'aperture' key in the input dictionary
                detectors = [input_dictionary['aperture'][index][3:5]]

                # Reset the mode of the coronagraphic observations to be imaging, since
                # 'coron' is not a supported mode, and those running Mirage with these
                # files currently have to manually switch the mode over to 'imaging'
                input_dictionary['Mode'][index] = 'imaging'

                n_detectors = len(detectors)
                for key in input_dictionary:
                    observation_dictionary[key].extend(([input_dictionary[key][index]] * n_detectors))
                observation_dictionary['detector'].extend(detectors)

            elif instrument == 'nircam':
            # if instrument == 'nircam' and input_dictionary['Mode'][index] != 'coron':
                # NIRCam case: Expand for detectors. Create one entry in each list for each
                # detector, rather than a single entry for 'ALL' or 'BSALL'

                # Determine module and subarray of the observation
                sub = input_dictionary['Subarray'][index]
                module = input_dictionary['Module'][index]
                if module == 'ALL':
                    module = ['A', 'B']
                else:
                    module = list(module)

                # Match up `sub` with aperture names in the aperture table
                # FULL matches up with the standard full frame imaging
                # apertures as well as full frame Grism apertures
                matches = np.where(nircam_apertures['Name'] == sub)[0]

                if len(matches) == 0:
                    raise ValueError("ERROR: aperture {} in not present in the subarray definition file {}"
                                     .format(sub, nircam_subarray_file))

                # Keep only apertures in the correct module
                matched_allmod_apertures = nircam_apertures['AperName'][matches].data
                matched_apertures = []
                for mod in module:
                    good = [ap for ap in matched_allmod_apertures if 'NRC{}'.format(mod) in ap]
                    matched_apertures.extend(good)

                if sub in ['FULL', 'SUB160', 'SUB320', 'SUB640', 'SUB64P', 'SUB160P', 'SUB400P', 'FULLP']:
                    mode = input_dictionary['Mode'][index]
                    template = input_dictionary['APTTemplate'][index]
                    if (sub == 'FULL'):
                        if mode in ['imaging', 'ts_imaging', 'wfss', 'coron']:
                            # This block should catch full-frame observations
                            # in either imaging (including TS imaging) or
                            # wfss mode
                            matched_aps = np.array([ap for ap in matched_apertures if 'GRISM' not in ap])
                            matched_apertures = []
                            detectors = []
                            for ap in matched_aps:
                                detectors.append(ap[3:5])
                                split = ap.split('_')
                                if len(split) == 3:
                                    ap_string = '{}_{}'.format(split[1], split[2])
                                elif len(split) == 2:
                                    ap_string = split[1]
                                matched_apertures.append(ap_string)

                        elif mode == 'ts_grism':
                            # This block should get Grism Time Series
                            # observations that use the full frame
                            matched_apertures = np.array([ap for ap in matched_apertures if 'GRISM' in ap])
                            filtered_aperture = input_dictionary['aperture'][index]
                            filtered_splits = filtered_aperture.split('_')
                            filtered_ap_no_det = '{}_{}'.format(filtered_splits[1], filtered_splits[2])
                            detectors = [filtered_splits[0][3:5]]

                            # Get correct apertures
                            apertures_to_add = []

                            final_matched_apertures = []
                            final_detectors = []
                            filtered_ap_det = filtered_aperture[3:5]
                            for ap in matched_apertures:
                                det = ap[3:5]
                                if ap == filtered_aperture or det != filtered_ap_det:
                                    final_matched_apertures.append(ap)
                                    final_detectors.append(det)
                            matched_apertures = final_matched_apertures
                            detectors = final_detectors
                    else:
                        # 'Standard' imaging subarrays: SUB320, SUB400P, etc
                        matched_apertures = [ap for ap in matched_apertures if sub in ap]
                        detectors = [ap.split('_')[0][3:5] for ap in matched_apertures]
                        matched_apertures = [sub] * len(detectors)

                elif 'SUBGRISM' in sub:
                    # This should catch only Grism Time Series observations
                    # and engineering imaging observations, which are the
                    # only 2 templates that can use SUBGRISM apertures
                    long_filter = input_dictionary['LongFilter'][index]
                    filter_dependent_apertures = [ap for ap in matched_apertures if len(ap.split('_')) == 3]

                    filtered_aperture = input_dictionary['aperture'][index]
                    filtered_splits = filtered_aperture.split('_')
                    filtered_ap_no_det = '{}_{}'.format(filtered_splits[1], filtered_splits[2])
                    detectors = [filtered_splits[0][3:5]]

                    # Get correct apertures
                    apertures_to_add = []

                    final_matched_apertures = []
                    final_detectors = []
                    filtered_ap_det = filtered_aperture[3:5]
                    for ap in matched_apertures:
                        det = ap[3:5]
                        if ap == filtered_aperture or det != filtered_ap_det:
                            final_matched_apertures.append(ap)
                            final_detectors.append(det)
                    matched_apertures = []
                    for ap in final_matched_apertures:
                        split = ap.split('_')
                        if len(split) == 3:
                            ap_string = '{}_{}'.format(split[1], split[2])
                        elif len(split) == 2:
                            ap_string = split[1]
                        matched_apertures.append(ap_string)
                    detectors = final_detectors
                else:
                    # TA, WFSC apertures
                    stripped_apertures = []
                    detectors = []
                    for ap in matched_apertures:
                        detectors.append(ap[3:5])
                        split = ap.split('_')
                        if len(split) == 3:
                            ap_string = '{}_{}'.format(split[1], split[2])
                        elif len(split) == 2:
                            ap_string = split[1]
                        stripped_apertures.append(ap_string)
                    matched_apertures = stripped_apertures

                full_apertures = ['NRC{}_{}'.format(det, sub) for det, sub in zip(detectors, matched_apertures)]

                # Add entries to observation dictionary
                num_entries = len(detectors)
                #observation_dictionary['Subarray'].extend(matched_apertures) extend? or replace?
                for key in input_dictionary:
                    #if key not in ['Subarray']:
                    if key not in ['aperture', 'detector', 'Subarray']:
                        observation_dictionary[key].extend(([input_dictionary[key][index]] * num_entries))
                observation_dictionary['detector'].extend(detectors)
                observation_dictionary['aperture'].extend(full_apertures)
                observation_dictionary['Subarray'].extend(matched_apertures)

            else:
                if instrument == 'niriss':
                    detectors = ['NIS']

                elif instrument == 'nirspec':
                    detectors = ['NRS']

                elif instrument == 'fgs':
                    if input_dictionary['APTTemplate'][index] == 'WfscGlobalAlignment':
                        guider_number = input_dictionary['aperture'][index][3]
                    elif input_dictionary['APTTemplate'][index] == 'FgsExternalCalibration':
                        guider_number = _get_guider_number(self.input_xml, input_dictionary['obs_num'][index])
                    detectors = ['G{}'.format(guider_number)]

                elif instrument == 'miri':
                    detectors = ['MIR']

                n_detectors = len(detectors)
                for key in input_dictionary:
                    observation_dictionary[key].extend(([input_dictionary[key][index]] * n_detectors))
                observation_dictionary['detector'].extend(detectors)

        """
        # Correct NIRCam aperture names for commissioning subarrays
        for index, instrument in enumerate(observation_dictionary['Instrument']):
            instrument = instrument.lower()
            if instrument == 'nircam':
                detector = observation_dictionary['detector'][index]
                sub = observation_dictionary['Subarray'][index]

                # this should probably better be handled by using the subarray_definitions file upstream
                if 'DHSPIL' in sub:
                    subarray, module = sub.split('DHSPIL')
                    subarray_size = subarray[3:]
                    detector = 'A3' if module == 'A' else 'B4'
                    aperture_name = 'NRC{}_DHSPIL_SUB{}'.format(detector, subarray_size)
                elif 'FP1' in sub:
                    subarray, module = sub.split('FP1')
                    subarray_size = subarray[3:]
                    detector = 'A3' if module == 'A' else 'B4'
                    aperture_name = 'NRC{}_FP1_SUB{}'.format(detector, subarray_size)
                else:
                    aperture_name = 'NRC' + detector + '_' + sub

                observation_dictionary['aperture'][index] = aperture_name
                observation_dictionary['detector'][index] = detector
        """

        return observation_dictionary

    def extract_grism_aperture(self, apertures, filter_name):
        """In the case of a Grism observation (WFSS or GRISM TSO), where a
        given crossing filter is used, find the appropriate aperture to
        use.

        Parameters
        ----------
        apertures : list
            List of possible grism apertures

        filter_name : str
            Name of crossing filter

        Returns
        -------
        apertures : list
            Modified list containig the correct aperture
        """
        filter_match = [True if filter_name in mtch else False for mtch in apertures]
        if any(filter_match):
            self.logger.debug('EXACT FILTER MATCH')
            self.logger.debug('{}'.format(filter_match))
            apertures = list(np.array(apertures)[filter_match])
        else:
            self.logger.debug('NO EXACT FILTER MATCH')
            filter_int = int(filter_name[1:4])
            aperture_int = np.array([int(ap.split('_')[-1][1:4]) for ap in apertures])
            wave_diffs = np.abs(aperture_int - filter_int)
            min_diff_index = np.where(wave_diffs == np.min(wave_diffs))[0]
            apertures = list(apertures[min_diff_index])

            self.logger.debug('{} {} {} {}'.format(filter_int, aperture_int, min_diff_index, apertures))

        return apertures

    def extract_value(self, line):
        """Extract text from xml line

        Parameters
        ----------
        line : str
            Line from xml file

        Returns
        -------
        line : str
            Text between > and < in the input line
        """
        gt = line.find('>')
        lt = line.find('<', gt)
        return line[gt + 1:lt]

    def full_path(self, in_path):
        """
        If the input path is not None, expand
        any environment variables and make an
        absolute path. Return the updated path.

        Parameters
        ----------
        in_path : str
            Path to be expanded

        Returns
        -------
        in_path : str
            Expanded, absolute path
        """
        if in_path.lower() == 'none':
            return in_path
        else:
            return os.path.abspath(os.path.expandvars(in_path))

    def get_pointing_info(self, file, propid=0, verbose=False):
        """Read in information from APT's pointing file.

        Parameters
        ----------
        file : str
            Name of APT-exported pointing file to be read
        propid : int
            Proposal ID number (integer). This is used to
            create various ID fields

        Returns
        -------
        pointing : dict
            Dictionary of pointing-related information

        TODO
        ----
            extract useful information from header?
            check visit numbers
            set parallel proposal number correctly

        """
        tar = []
        tile = []
        exp = []
        dith = []
        aperture = []
        targ1 = []
        targ2 = []
        ra = []
        dec = []
        basex = []
        basey = []
        dithx = []
        dithy = []
        v2 = []
        v3 = []
        idlx = []
        idly = []
        level = []
        type_str = []
        expar = []
        dkpar = []
        ddist = []
        observation_number = []
        visit_number = []
        visit_id = []
        visit_grp = []
        activity_id = []
        observation_label = []
        observation_id = []
        seq_id = []

        act_counter = 1
        with open(file) as f:
            for line in f:

                # Skip comments and new lines except for the line with the version of the PRD
                if (line[0] == '#') or (line in ['\n']) or ('=====' in line):

                    # Compare the version of the PRD from APT and pysiaf
                    if 'PRDOPSSOC' in line:
                        apt_prd_version = line.split(' ')[-2]
                        if apt_prd_version != JWST_PRD_VERSION:
                            self.logger.warning(('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                                                 'The pointing file from APT was created using PRD version: {},\n'
                                                 'while the current installation of pysiaf uses PRD version: {}.\n'
                                                 'This inconsistency may lead to errors in source locations or\n'
                                                 'the WCS of simulated data if the apertures being simulated are\n'
                                                 'shifted between the two versions. We highly recommend using a\n'
                                                 'consistent version of the PRD between APT and pysiaf.\n'
                                                 '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                                                 .format(apt_prd_version, JWST_PRD_VERSION)))
                    else:
                        continue
                # Extract proposal ID
                elif line.split()[0] == 'JWST':
                    propid_header = line.split()[7]
                    try:
                        propid = int(propid_header)
                    except ValueError:
                        # adopt value passed to function
                        pass
                    if verbose:
                        self.logger.info('Extracted proposal ID {}'.format(propid))
                    continue

                elif (len(line) > 1):
                    elements = line.split()

                    # Look for lines that give visit/observation numbers
                    if line[0:2] == '* ':
                        paren = line.rfind('(')
                        if paren == -1:
                            obslabel = line[2:]
                            obslabel = obslabel.strip()
                        else:
                            obslabel = line[2:paren-1]
                            obslabel = obslabel.strip()
                        if (' (' in obslabel) and (')' in obslabel):
                            obslabel = re.split(r' \(|\)', obslabel)[0]

                    skip = False

                    if line[0:2] == '**':
                        v = elements[2]
                        obsnum, visitnum = v.split(':')
                        obsnum = str(obsnum).zfill(3)
                        visitnum = str(visitnum).zfill(3)
                        if (skip is True) and (verbose):
                            self.logger.info('Skipping observation {} ({})'.format(obsnum, obslabel))

                    try:
                        # Skip the line at the beginning of each
                        # visit that gives info on the target,
                        # but is not actually an observation
                        # These lines have 'Exp' values of 0,
                        # while observations have a value of 1
                        # (that I've seen so far)

                        # The exception to this rule is TA images.
                        # These have Exp values of 0. Sigh.


                        if ((int(elements[1]) > 0) & ('NRC' in elements[4]
                                                         or 'NIS' in elements[4]
                                                         or 'FGS' in elements[4]
                                                         or 'NRS' in elements[4]
                                                         or 'MIR' in elements[4])
                            ) or (('TA' in elements[4]) & ('NRC' in elements[4]
                                                           or 'NIS' in elements[4])):
                            if (elements[18] == 'PARALLEL') and ('MIRI' in elements[4]):
                                skip = True

                            if skip:
                                # act_counter += 1
                                continue
                            act = self.base36encode(act_counter)
                            activity_id.append(act)
                            observation_label.append(obslabel)
                            observation_number.append(obsnum)
                            visit_number.append(visitnum)
                            prop_5digit = "{0:05d}".format(int(propid))
                            vid = "{}{}{}".format(prop_5digit, obsnum, visitnum)
                            visit_id.append(vid)
                            # Visit group hard coded to 1. It's not clear how APT divides visits up into visit
                            # groups. For now just keep everything in a single visit group.
                            vgrp = '01'
                            visit_grp.append(vgrp)
                            # Parallel sequence is hard coded to 1 (Simulated instrument as prime rather than
                            # parallel) at the moment. Future improvements may allow the proper sequence
                            # number to be constructed.
                            seq = '1'
                            seq_id.append(seq)
                            tar.append(int(elements[0]))
                            tile.append(int(elements[1]))
                            exnum = str(elements[2]).zfill(5)
                            exp.append(exnum)
                            dith.append(int(elements[3]))

                            ap = elements[4]
                            if ('GRISMR_WFSS' in elements[4]):
                                ap = ap.replace('GRISMR_WFSS', 'FULL')
                            elif ('GRISMC_WFSS' in elements[4]):
                                ap = ap.replace('GRISMC_WFSS', 'FULL')

                            aperture.append(ap)
                            targ1.append(int(elements[5]))
                            targ2.append(elements[6])
                            ra.append(elements[7])
                            dec.append(elements[8])
                            basex.append(elements[9])
                            basey.append(elements[10])
                            dithx.append(float(elements[11]))
                            dithy.append(float(elements[12]))
                            v2.append(float(elements[13]))
                            v3.append(float(elements[14]))
                            idlx.append(float(elements[15]))
                            idly.append(float(elements[16]))
                            level.append(elements[17])
                            type_str.append(elements[18])
                            expar.append(int(elements[19]))
                            dkpar.append(int(elements[20]))
                            ddist.append(float(elements[21]))

                            # For the moment we assume that the instrument being simulated is not being
                            # run in parallel, so the parallel proposal number will be all zeros,
                            # as seen in the line below.
                            observation_id.append("V{}P{}{}{}{}".format(vid, '00000000', vgrp, seq, act))
                            # act_counter += 1

                    except ValueError as e:
                        if verbose:
                            self.logger.info('Skipping line:\n{}\nproducing error:\n{}'.format(line, e))
                        pass

        pointing = {'exposure': exp, 'dither': dith, 'aperture': aperture,
                    'targ1': targ1, 'targ2': targ2, 'ra': ra, 'dec': dec,
                    'basex': basex, 'basey': basey, 'dithx': dithx,
                    'dithy': dithy, 'v2': v2, 'v3': v3, 'idlx': idlx, 'idly': idly, 
                    'ddist': ddist, 'obs_label': observation_label,
                    'obs_num': observation_number, 'visit_num': visit_number,
                    'act_id': activity_id, 'visit_id': visit_id, 'visit_group': visit_grp,
                    'sequence_id': seq_id, 'observation_id': observation_id}
        return pointing


    def global_alignment_pointing(self, obs_dict):
        """Adjust the pointing dictionary information for global alignment
        observations. Some of the entries need to be changed from NIRCam to
        FGS. Remember that not all observations in the dictionary will
        necessarily be WfscGlobalAlignment template. Be sure the leave all
        other templates unchanged.

        Parameters
        ----------
        obs_dict : dict
            Dictionary of observation parameters, as returned from add_observation_info()

        Returns
        -------
        obs_dict : dict
            Dictionary with modified values for FGS pointing in Global Alignment templates
        """

        # We'll always be changing NIRCam to FGS, so set up the NIRCam siaf
        # instance outside of loop
        nrc_ap = siaf_nrc['NRCA3_FULL']
        # nrc_siaf = Siaf('nircam')['NRCA3_FULL']

        ga_index = np.array(obs_dict['APTTemplate']) == 'WfscGlobalAlignment'

        observation_numbers = np.unique(np.array(obs_dict['obs_num'])[ga_index])

        for obs_num in observation_numbers:
            obs_indexes = np.where(np.array(obs_dict['obs_num']) == obs_num)[0]

            # Get the subarray and aperture entries for the observation
            aperture_values = np.array(obs_dict['aperture'])[obs_indexes]
            subarr_values = np.array(obs_dict['Subarray'])[obs_indexes]

            # Subarray values, which come from the xml file, are correct. The aperture
            # values, which come from the pointing file, are not correct. We need to
            # copy over the FGS values from the Subarray column to the aperture column
            to_fgs = [True if 'FGS' in subarr else False for subarr in subarr_values]
            aperture_values[to_fgs] = subarr_values[to_fgs]
            fgs_aperture = aperture_values[to_fgs][0]

            all_aperture_values = np.array(obs_dict['aperture'])
            all_aperture_values[obs_indexes] = aperture_values
            obs_dict['aperture'] = all_aperture_values

            # Update the pointing info for the FGS exposures
            fgs = Siaf('fgs')[fgs_aperture]
            basex, basey = fgs.tel_to_idl(nrc_ap.V2Ref, nrc_ap.V3Ref)
            dithx = np.array(obs_dict['dithx'])[obs_indexes[to_fgs]]
            dithy = np.array(obs_dict['dithy'])[obs_indexes[to_fgs]]
            idlx = basex + dithx
            idly = basey + dithy

            basex_col = np.array(obs_dict['basex'])
            basey_col = np.array(obs_dict['basey'])
            idlx_col = np.array(obs_dict['idlx'])
            idly_col = np.array(obs_dict['idly'])

            basex_col[obs_indexes[to_fgs]] = basex
            basey_col[obs_indexes[to_fgs]] = basey
            idlx_col[obs_indexes[to_fgs]] = idlx
            idly_col[obs_indexes[to_fgs]] = idly

            obs_dict['basex'] = basex_col
            obs_dict['basey'] = basey_col
            obs_dict['idlx'] = idlx_col
            obs_dict['idly'] = idly_col

        return obs_dict

    def tight_dithers(self, input_dict):
        """
        In NIRCam, when the 'FULL' dither pattern is
        used, it is possible to set the number of primary
        dithers to '3TIGHT' rather than just a number
        (e.g. '3'). If the number of dithers is set to '3TIGHT'
        remove 'TIGHT' from the entries and leave
        only the number behind.

        Parameters
        ----------

        input_dict : dict
           Dictionary where each key points to a list containing
           observation details. For example, input_dict['PrimarDither']
           is a list of the number of primary dithers for all observations

        Returns
        -------

        input_dict : dict
            Updated dictionary where 'TIGHT' has been removed from
            PrimaryDither list
        """
        inlist = input_dict['PrimaryDithers']
        modlist = [v if 'TIGHT' not in v else v.strip('TIGHT') for v in inlist]
        input_dict['PrimaryDithers'] = modlist
        return input_dict

    def add_options(self, parser=None, usage=None):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage, description='Simulate JWST ramp')
        parser.add_argument("input_xml", help='XML file from APT describing the observations.')
        parser.add_argument("pointing_file", help='Pointing file from APT describing observations.')
        parser.add_argument("--output_csv", help="Name of output file containing list of observations.",
                            default=None)
        parser.add_argument("--observation_list_file", help=('Ascii file containing a list of '
                                                             'observations, times, and roll angles, '
                                                             'catalogs'), default=None)
        return parser

def read_subarray_definition_file(filename):
    """Read in the file that contains a list of subarray names and related information

    Parameters
    ----------
    filename : str
        Name of the ascii file containing the table of subarray information

    Returns
    -------
    data : astropy.table.Table
        Table containing subarray information
    """
    try:
        data = ascii.read(filename, data_start=1, header_start=0)
    except:
        raise RuntimeError(("Error: could not read in subarray definitions file: {}"
                            .format(filename)))
    return data


def _get_entry(dict, entry_number):
    """Return a numbered entry from a dictionary that corresponds to the observataion_list.yaml.

    Parameters
    ----------
    dict
    entry_number

    Returns
    -------

    """
    entry_key = 'EntryNumber{}'.format(entry_number)
    for key, observation in dict.items():
        if entry_key in observation.keys():
            return observation[entry_key]

def _append_dictionary(base_dictionary, added_dictionary, braid=False):
    """Append the content of added_dictionary key-by-key to the base_dictionary.

    This assumes that the keys refer to lists.

    Parameters
    ----------
    base_dictionary : dict
    added_dictionary : dict
    braid : bool
        If true, the elements of added_dictionary are added in alternating sequence.
        This is used to synchronize parallel observations with the pointing file.

    Returns
    -------
    new_dictionary : dict
        Dictionary where every key holds a list of lists

    """
    new_dictionary = copy.deepcopy(base_dictionary)

    # extract an arbitrary key name
    first_key = [key for i, key in enumerate(base_dictionary.keys()) if i == 0][0]

    # Insert keys from added_dictionary that are not yet present in base_dictionary
    for key in added_dictionary.keys():
        if key not in base_dictionary.keys():
            new_dictionary[key] = ['None'] * len(base_dictionary[first_key])

    # Append the items
    for key in new_dictionary.keys():
        if key not in added_dictionary.keys():
            continue
        # print('{} {}'.format(key, new_dictionary[key]))
        if len(new_dictionary[key]) == 0:
            new_dictionary[key] = added_dictionary[key]
        else:
            if braid:
                # solution from https://stackoverflow.com/questions/3678869/pythonic-way-to-combine-two-lists-in-an-alternating-fashion
                new_dictionary[key] = [sub[i] for i in range(len(added_dictionary[key])) for sub in
                                       [new_dictionary[key], added_dictionary[key]]]
            else:
                new_dictionary[key] = new_dictionary[key] + added_dictionary[key]

    return new_dictionary


class ReadAPTXML():
    """Class to open and parse XML files from APT. Can read templates for
    NircamImaging, NircamEngineeringImaging, WfscCommissioning,
    WfscGlobaLAlignment, WfscCoarsePhasing, WfscFinePhasing (incomplete),
    and NircamWfss modes.

    Attributes
    ----------
    apt: str
        APT namespace for XML files
    APTObservationParams: dict
        Dictionary of APT parameters that accumulates all parameters in all
        tiles and all observation. Passed out to be further parsed in the
        apt_inputs script
    obs_tuple_list: list
        Compiled list of all the parameters for all tiles in a single observation
    """

    def __init__(self):
        # Initialize log
        self.logger = logging.getLogger('read_apt_xml')
        self.logger.info('Running read_apt_xml....\n')

        # Define the APT namespace
        self.apt = '{http://www.stsci.edu/JWST/APT}'

        # Set up dictionary of observation parameters to be populated
        ProposalParams_keys = ['PI_Name', 'Proposal_category', 'Proposal_subcategory', 'ProposalID',
                               'Science_category', 'Title']
        ObsParams_keys = ['Module', 'Subarray', 'Instrument',
                          'PrimaryDitherType', 'PrimaryDithers', 'DitherSize',
                          'SubpixelPositions', 'SubpixelDitherType', 'SmallGridDitherType',
                          'CoordinatedParallel', 'ParallelInstrument',
                          'ObservationID', 'TileNumber', 'APTTemplate',
                          'ApertureOverride', 'ObservationName',
                          'DitherPatternType', 'ImageDithers',  # NIRISS
                          'number_of_dithers',  # uniform name across instruments
                          'FiducialPointOverride', 'TargetID', 'TargetRA', 'TargetDec'
                          ]
        FilterParams_keys = ['ShortFilter', 'LongFilter', 'ShortPupil', 'LongPupil',
                             'ReadoutPattern', 'Groups', 'Integrations',
                             'FilterWheel', 'PupilWheel',  # for NIRISS
                             'NumOutputs'
                             ]
        OtherParams_keys = ['Mode', 'Grism', 'CoronMask',
                            'IntegrationsShort', 'GroupsShort', 'Dither',  # MIRI
                            'GroupsLong', 'ReadoutPatternShort', 'IntegrationsLong',
                            'Exposures', 'Wavelength', 'ReadoutPatternLong', 'Filter',
                            'EtcIdLong', 'EtcIdShort', 'EtcId', 'Tracking'
                            ]

        self.APTObservationParams_keys = ProposalParams_keys + ObsParams_keys + \
            FilterParams_keys + OtherParams_keys
        self.APTObservationParams = {}
        for key in self.APTObservationParams_keys:
            self.APTObservationParams[key] = []
        self.empty_exposures_dictionary = copy.deepcopy(self.APTObservationParams)
        self.observation_info = OrderedDict()

    def get_tracking_type(self, observation):
        """Determine whether the observation uses sidereal or non-sidereal
        tracking

        Parameters
        ----------
        observation : etree xml element
            xml content of observation

        Returns
        -------
        tracking_type : str
            "sidereal" or "non-sidereal" based on the target used in the
            observation
        """
        target_id = observation.find(self.apt + 'TargetID').text
        targname = target_id.split(' ')[1]
        matched_key = [key for key in self.target_type if targname == key]
        if len(matched_key) == 0:
            raise ValueError('No matching target name for {} in self.target_type'.format(targname))
        elif len(matched_key) > 1:
            raise ValueError('Multiple matching target names for {} in self.target_type.'.format(targname))
        else:
            tracking_type = self.target_type[matched_key[0]]
        return tracking_type

    def read_xml(self, infile, verbose=False):
        """Main function. Read in the .xml file from APT, and output dictionary of parameters.

        Arguments
        ---------
        infile (str):
            Path to input .xml file

        Returns
        -------
        dict:
            Dictionary with extracted observation parameters

        Raises
        ------
        ValueError:
            If an .xml file is provided that includes an APT template that is not
            supported.
            If the .xml file includes a fiducial pointing override with an
            unknown subarray specification
        """
        # Open XML file, get element tree of the APT proposal
        with open(infile) as f:
            tree = etree.parse(f)

        # Get high-level information: proposal info - - - - - - - - - - - - - -

        # Set default values
        propid_default = 42
        proptitle_default = 'Looking for my towel'
        scicat_default = 'Planets and Planet Formation'
        piname_default = 'D.N. Adams'
        propcat_default = 'GO'
        propsubcat_default = 'UNKNOWN'

        # Get just the element with the proposal information
        proposal_info = tree.find(self.apt + 'ProposalInformation')

        # Title
        try:
            prop_title = proposal_info.find(self.apt + 'Title').text
        except:
            prop_title = proptitle_default

        # Proposal ID
        try:
            prop_id = '{:05d}'.format(int(proposal_info.find(self.apt + 'ProposalID').text))
        except:
            prop_id = '{:05d}'.format(propid_default)

        # Proposal Category
        try:
            # prop_category = proposal_info.find(self.apt + 'ProposalCategory')[0]
            # prop_category = etree.QName(prop_category).localname
            prop_category = proposal_info.find(self.apt + 'ProposalCategory').text
        except:
            prop_category = propcat_default

        # Proposal Sub-Category
        try:
            prop_subcategory = proposal_info.find(self.apt + 'ProposalCategorySubtype').text
        except:
            prop_subcategory = propsubcat_default


        # Science Category
        try:
            science_category = proposal_info.find(self.apt + 'ScientificCategory').text
        except:
            science_category = scicat_default

        # Principal Investigator Name
        try:
            pi_firstname = proposal_info.find('.//' + self.apt + 'FirstName').text
            pi_lastname = proposal_info.find('.//' + self.apt + 'LastName').text
            pi_name = ' '.join([pi_firstname, pi_lastname])
        except:
            pi_name = piname_default

        # Get target names - - - - - - - - - - - - - - - - - - - - - - - - - -
        targs = tree.find(self.apt + 'Targets')
        target_elements = targs.findall(self.apt + 'Target')
        self.target_info = {}
        self.target_type = {}
        for target in target_elements:
            t_name = target.find(self.apt + 'TargetName').text
            try:
                t_coords = target.find(self.apt + 'EquatorialCoordinates').items()[0][1]
            except AttributeError:
                # Non-sidereal targets do not have EquatorialCoordinates entries in the xml
                t_coords = '00 00 00 00 00 00'
            ra_hour, ra_min, ra_sec, dec_deg, dec_arcmin, dec_arcsec = t_coords.split(' ')
            ra = '{}:{}:{}'.format(ra_hour, ra_min, ra_sec)
            dec = '{}:{}:{}'.format(dec_deg, dec_arcmin, dec_arcsec)
            self.target_info[t_name] = (ra, dec)

            type_key = [key for key in target.attrib.keys() if 'type' in key]
            if 'SolarSystem' in target.attrib[type_key[0]]:
                self.target_type[t_name] = 'non-sidereal'
            else:
                self.target_type[t_name] = 'sidereal'
        self.logger.info('target_info:')
        self.logger.info('{}'.format(self.target_info))

        # Get parameters for each observation  - - - - - - - - - - - - - - - -

        # Find all observations (but use only those that use NIRCam or are WFSC)
        observation_data = tree.find(self.apt + 'DataRequests')
        observation_list = observation_data.findall('.//' + self.apt + 'Observation')

        # Loop through observations, get parameters
        for i_obs, obs in enumerate(observation_list):
            observation_number = obs.find(self.apt + 'Number').text.zfill(3)

            # Create empty list that will be populated with a tuple of parameters
            # for every observation
            self.obs_tuple_list = []

            # Determine what template is used for the observation
            template = obs.find(self.apt + 'Template')[0]
            template_name = etree.QName(template).localname

            # Are all the templates in the XML file something that we can handle?
            known_APT_templates = ['NircamImaging', 'NircamWfss', 'WfscCommissioning',
                                   'NircamEngineeringImaging', 'WfscGlobalAlignment',
                                   'WfscCoarsePhasing', 'WfscFinePhasing',
                                   'NircamGrismTimeSeries', 'NircamTimeSeries', 'NircamCoron',
                                   'NirissExternalCalibration', 'NirissWfss', 'NirissAmi',  # NIRISS
                                   'NirspecImaging', 'NirspecInternalLamp',  # NIRSpec
                                   'MiriMRS', 'MiriImaging', 'MiriCoron', # MIRI
                                   'FgsExternalCalibration',  # FGS
                                   'NircamDark', 'NirissDark'  # Darks
                                   ]
            if template_name not in known_APT_templates:
                # If not, turn back now.
                _log.info(f'No protocol written to read {template_name} template.')

            # Get observation label
            label_ele = obs.find(self.apt + 'Label')
            if label_ele is not None:
                label = label_ele.text
                if (' (' in label) and (')' in label):
                    label = re.split(r' \(|\)', label)[0]
            else:
                label = 'None'

            # Get coordinated parallel
            coordparallel = obs.find(self.apt + 'CoordinatedParallel').text

            if verbose:
                self.logger.info('+'*100)
                self.logger.info('Observation `{}` labelled `{}` uses template `{}`'.format(observation_number, label,
                                                                                 template_name))
                number_of_entries = len(self.APTObservationParams['Instrument'])
                self.logger.info('APTObservationParams Dictionary holds {} entries before reading template'
                                 .format(number_of_entries))
                if coordparallel == 'true':
                    self.logger.info('Coordinated parallel observation')

            CoordinatedParallelSet = None
            if coordparallel == 'true':
                try:
                    CoordinatedParallelSet = obs.find(self.apt + 'CoordinatedParallelSet').text
                except AttributeError:
                    raise RuntimeError('Program does not specify parallels correctly.')

            try:
                obs_label = obs.find(self.apt + 'Label').text
            except AttributeError:
                # label tag not present
                obs_label = 'Observation 1'

            # Get target name
            try:
                targ_name = obs.find(self.apt + 'TargetID').text.split(' ')[1]
            except IndexError as e:
                self.logger.info("No target ID for observation: {}".format(obs))
                targ_name = obs.find(self.apt + 'TargetID').text.split(' ')[0]

            # For NIRSpec Internal Lamp
            if targ_name == 'NONE':
                self.target_info[targ_name] = ('0', '0')

            # extract visit numbers
            visit_numbers = [int(element.items()[0][1]) for element in obs if
                             element.tag.split(self.apt)[1] == 'Visit']

            prop_params = [pi_name, prop_id, prop_title, prop_category,
                           science_category, coordparallel, observation_number, obs_label, targ_name]

            proposal_parameter_dictionary = {'PI_Name': pi_name, 'ProposalID': prop_id,
                                             'Title': prop_title,
                                             'Proposal_category': prop_category,
                                             'Proposal_subcategory': prop_subcategory,
                                             'Science_category': science_category,
                                             'CoordinatedParallel': coordparallel,
                                             'ObservationID': observation_number,
                                             'ObservationName': obs_label,
                                             'TargetID': targ_name,
                                             'TargetRA': self.target_info[targ_name][0],
                                             'TargetDec': self.target_info[targ_name][1]
                                             }

            if template_name in ['NircamImaging', 'NircamEngineeringImaging', 'NirissExternalCalibration',
                                 'NirspecImaging', 'MiriMRS', 'FgsExternalCalibration', 'MiriImaging']:
                exposures_dictionary = self.read_generic_imaging_template(template, template_name, obs,
                                                                          proposal_parameter_dictionary,
                                                                          verbose=verbose)
                if coordparallel == 'true':
                    parallel_template_name = etree.QName(obs.find(self.apt + 'FirstCoordinatedTemplate')[0]).localname
                    if parallel_template_name in ['MiriImaging']:
                        pass
                    else:
                        parallel_exposures_dictionary = self.read_parallel_exposures(obs, exposures_dictionary,
                                                                                     proposal_parameter_dictionary,
                                                                                     verbose=verbose)
                        exposures_dictionary = _append_dictionary(exposures_dictionary, parallel_exposures_dictionary, braid=True)

            # If template is WFSC Commissioning
            elif template_name in ['WfscCommissioning']:
                exposures_dictionary, num_WFCgroups = self.read_commissioning_template(template,
                                                                                       template_name, obs,
                                                                                       prop_params)

            # If template is WFSC Global Alignment
            elif template_name in ['WfscGlobalAlignment']:
                exposures_dictionary, n_exp = self.read_globalalignment_template(template, template_name, obs,
                                                                                 prop_params)

            # If template is WFSC Coarse Phasing
            elif template_name in ['WfscCoarsePhasing']:
                exposures_dictionary, n_tiles_phasing = self.read_coarsephasing_template(template,
                                                                                         template_name, obs,
                                                                                         prop_params)

            # If template is WFSC Fine Phasing
            elif template_name in ['WfscFinePhasing']:
                exposures_dictionary, n_tiles_phasing = self.read_finephasing_template(template,
                                                                                       template_name, obs,
                                                                                       prop_params)

            # If template is NIRCam Grism Time Series
            elif template_name == 'NircamGrismTimeSeries':
                exposures_dictionary = self.read_nircam_grism_time_series(template, template_name, obs,
                                                                          proposal_parameter_dictionary)

            # If template is NIRCam Imaging Time Series
            elif template_name == 'NircamTimeSeries':
                exposures_dictionary = self.read_nircam_imaging_time_series(template, template_name, obs,
                                                                            proposal_parameter_dictionary)

            # NIRISS AMI
            elif template_name == 'NirissAmi':
                exposures_dictionary = self.read_niriss_ami_template(template, template_name, obs,
                                                                     proposal_parameter_dictionary,
                                                                     verbose=verbose)

            # If template is WFSS
            elif template_name == 'NircamWfss':
                exposures_dictionary = self.read_nircam_wfss_template(template, template_name, obs,
                                                                      proposal_parameter_dictionary)

            elif template_name == 'NirissWfss':
                exposures_dictionary = self.read_niriss_wfss_template(template, template_name, obs,
                                                                      proposal_parameter_dictionary,
                                                                      verbose=verbose)
                if coordparallel == 'true':
                    parallel_template_name = etree.QName(obs.find(self.apt + 'FirstCoordinatedTemplate')[0]).localname
                    if parallel_template_name in ['MiriImaging']:
                        pass
                    elif parallel_template_name in ['NircamImaging']:
                        parallel_exposures_dictionary = self.read_parallel_exposures(obs, exposures_dictionary,
                                                                                     proposal_parameter_dictionary,
                                                                                     verbose=verbose)
                        exposures_dictionary = _append_dictionary(exposures_dictionary, parallel_exposures_dictionary, braid=True)

                    else:
                        raise ValueError('Parallel template {} (with primary template {}) not supported.'
                                         .format(parallel_template_name, template_name))
            elif template_name == 'NircamCoron':
                exposures_dictionary = self.read_nircam_coronagraphy_template(template, template_name, obs,
                                                                      proposal_parameter_dictionary)
            elif template_name == 'MiriCoron':
                # we can't actually simulate this, but parse it anyway to allow handling joint
                # APT programs containing NIRCam and MIRI together.
                exposures_dictionary = self.read_miri_coronagraphy_template(template, template_name, obs,
                                                                              proposal_parameter_dictionary)


            else:
                self.logger.info('SKIPPED: Observation `{}` labelled `{}` uses template `{}`'.format(observation_number,
                                                                                                     label,
                                                                                                     template_name))
                continue

            if verbose:
                self.logger.info('Dictionary read from template has {} entries.'.format(len(exposures_dictionary['Instrument'])))

            # # set default number of dithers, for downstream processing
            # for i, n_dither in enumerate(exposures_dictionary['number_of_dithers']):
            #     if (template_name == 'NircamEngineeringImaging') and (n_dither == '2PLUS'):
            #         exposures_dictionary['number_of_dithers'][i] = '2'
            #     elif int(n_dither) == 0:
            #         exposures_dictionary['number_of_dithers'][i] = '1'

            # add the exposure dictionary to the main dictionary
            self.APTObservationParams = _append_dictionary(self.APTObservationParams,
                                                          exposures_dictionary)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            # Now we need to look for mosaic details, if any
            mosaic_tiles = obs.findall('.//' + self.apt + 'MosaicTiles')

            # count only tiles that are included
            tile_state = np.array([mosaic_tiles[i].find('.//' + self.apt + 'TileState').text for i in range(len(mosaic_tiles))])
            n_tiles = np.sum(np.array([True if state == 'Tile Included' else False for state in tile_state]))

            label = obs_label

            if verbose:
                self.logger.info("Found {} tile(s) for observation {} {}".format(n_tiles, observation_number, label))
                if len(visit_numbers) > 0:
                    self.logger.info('Found {} visits with numbers: {}'.format(len(visit_numbers), visit_numbers))

            if n_tiles > 1:
                for i in range(n_tiles - 1):
                    self.APTObservationParams = _append_dictionary(self.APTObservationParams, exposures_dictionary)

            self.observation_info[observation_number] = {}
            self.observation_info[observation_number]['visit_numbers'] = visit_numbers

            if verbose:
                number_of_entries_after = len(self.APTObservationParams['Instrument'])
                self.logger.info('APTObservationParams Dictionary holds {} entries after reading template ({:+d} entries)'
                                 .format(number_of_entries_after, number_of_entries_after-number_of_entries))

        if verbose:
            self.logger.info('Finished reading APT xml file.')
            self.logger.info('+'*100)
        # Temporary for creating truth tables to use in tests
        #bool_cols = ['ParallelInstrument']
        #int_cols = ['Groups', 'Integrations']
        #final_table = Table()
        #for key in self.APTObservationParams.keys():
        #    if key in bool_cols:
        #        data_type = bool
        #    elif key in int_cols:
        #        data_type = str
        #    else:
        #        data_type = str
        #    new_col = Column(data=self.APTObservationParams[key], name=key, dtype=data_type)
        #    final_table.add_column(new_col)
        #tmpoutdir = '/Users/hilbert/python_repos/mirage/tests/test_data'
        #filebase = os.path.split(infile)[1]
        #tmpoutfile = '{}{}'.format(filebase.split('.xml')[0], '.txt')
        #ascii.write(final_table, os.path.join(tmpoutdir, tmpoutfile), overwrite=True)

        return self.APTObservationParams

    def add_exposure(self, dictionary, tup):
        """Add an exposure to the exposure dictionary

        Parameters
        ----------
        dictionary : dict
            Information on individual exposures

        tup : tuple
            A tuple contianing information to add to dictionary as
            the next exposure

        Returns
        -------
        dictionary : dict
            With new exposure added
        """
        dictionary['PI_Name'].append(tup[0])
        dictionary['ProposalID'].append(tup[1])
        dictionary['Title'].append(tup[2])
        dictionary['Proposal_category'].append(tup[3])
        dictionary['Science_category'].append(tup[4])
        dictionary['Mode'].append(tup[5])
        dictionary['Module'].append(tup[6])
        dictionary['Subarray'].append(tup[7])
        dictionary['PrimaryDitherType'].append(tup[8])
        dictionary['PrimaryDithers'].append(tup[9])
        dictionary['SubpixelDitherType'].append(tup[10])
        dictionary['SubpixelPositions'].append(tup[11])
        dictionary['ShortFilter'].append(tup[12])
        dictionary['LongFilter'].append(tup[13])
        dictionary['ReadoutPattern'].append(tup[14])
        dictionary['Groups'].append(tup[15])
        dictionary['Integrations'].append(tup[16])
        dictionary['ShortPupil'].append(tup[17])
        dictionary['LongPupil'].append(tup[18])
        dictionary['Grism'].append(tup[19])
        dictionary['CoordinatedParallel'].append(tup[20])
        dictionary['ObservationID'].append(tup[21])
        dictionary['TileNumber'].append(tup[22])
        dictionary['APTTemplate'].append(tup[23])
        dictionary['Instrument'].append(tup[24])
        dictionary['ObservationName'].append(tup[25])
        dictionary['TargetID'].append(tup[26])
        dictionary['Tracking'].append(tup[27])
        return dictionary

    def read_generic_imaging_template(self, template, template_name, obs, proposal_parameter_dictionary,
                                      verbose=False, parallel=False):
        """Read imaging template content regardless of instrument.

        Save content to object attributes. Support for coordinated parallels is included.

        Parameters
        ----------
        template : etree xml element
            xml content of template
        template_name : str
            name of the template
        obs : etree xml element
            xml content of observation
        proposal_parameter_dictionary : dict
            Dictionary of proposal parameters to extract from template

        Returns
        -------
        exposures_dictionary : OrderedDict
            Dictionary containing relevant exposure parameters

        """
        if parallel:
            # boolean indicating which instrument is not prime but parallel
            parallel_instrument = True
            if template_name == 'FgsExternalCalibration':
                instrument = 'FGS'
            elif template_name == 'MiriImaging':
                instrument = 'MIRI'
            elif template_name == 'NirissImaging':
                instrument = 'NIRISS'
            elif template_name == 'NircamImaging':
                instrument = 'NIRCAM'
            prime_instrument = obs.find(self.apt + 'Instrument').text
            if verbose:
                self.logger.info('Prime: {}   Parallel: {}'.format(prime_instrument, instrument))
            prime_template = obs.find(self.apt + 'Template')[0]
            prime_template_name = etree.QName(prime_template).localname
            prime_ns = "{{{}/Template/{}}}".format(self.apt.replace('{', '').replace('}', ''), prime_template_name)
            if verbose:
                self.logger.info('PRIME TEMPLATE NAME IS: {}'.format(prime_template_name))
        else:
            instrument = obs.find(self.apt + 'Instrument').text
            parallel_instrument = False
            prime_instrument = instrument
            prime_template = template
            prime_template_name = template_name

        exposures_dictionary = copy.deepcopy(self.empty_exposures_dictionary)
        ns = "{{{}/Template/{}}}".format(self.apt.replace('{','').replace('}',''), template_name)

        DitherPatternType = None

        if ((prime_instrument in ['NIRCAM', 'FGS']) or
           (prime_instrument == 'NIRISS' and prime_template_name == 'NirissWfss')):
            dither_key_name = 'PrimaryDithers'
        elif prime_instrument in ['NIRISS', 'MIRI', 'NIRSPEC']:
            dither_key_name = 'ImageDithers'

        # number of dithers defaults to 1
        number_of_dithers = 1
        number_of_subpixel_positions = 1

        number_of_primary_dithers = 1
        number_of_subpixel_dithers = 1

        # Check the target type in order to decide whether the tracking should be
        # sidereal or non-sidereal
        tracking = self.get_tracking_type(obs)

        if instrument.lower() == 'nircam':
            # NIRCam uses FilterConfig structure to specifiy exposure parameters

            # store Module, Subarray, ... fields
            observation_dict = {}
            for field in template:
                key = field.tag.split(ns)[1]
                value = field.text
                observation_dict[key] = value

            if "PrimaryDitherType" in observation_dict.keys():
                if observation_dict["PrimaryDitherType"] == "WFSC":
                    observation_dict["SubpixelDitherType"] = "WFSC"

            # Determine if there is an aperture override
            override = obs.find('.//' + self.apt + 'FiducialPointOverride')
            FiducialPointOverride = True if override is not None else False

            # Get the number of primary and subpixel dithers
            primary_dithers_present = dither_key_name in observation_dict.keys()
            if primary_dithers_present:
                number_of_primary_dithers = observation_dict[dither_key_name]

                if (template_name == 'NircamEngineeringImaging') and (number_of_primary_dithers == '2PLUS'):
                    # Handle the special case for 2PLUS
                    number_of_primary_dithers = 2

                # Deal with cases like 2TIGHTGAPS, 8NIRSPEC, etc.
                try:
                    test = int(number_of_primary_dithers)
                except ValueError:
                    number_of_primary_dithers = observation_dict[dither_key_name][0]

            else:
                self.logger.info('Primary dither element {} not found, use default primary dithers value (1).'.format(dither_key_name))

            # Find the number of subpixel dithers
            if not parallel:
                if observation_dict['SubpixelDitherType'] in ['3-POINT-WITH-MIRI-F770W']:
                    # Handle the special case for MIRI
                    number_of_subpixel_dithers = 3
                elif "-WITH-NIRISS" in observation_dict['SubpixelDitherType']:
                    number_of_subpixel_dithers = int(observation_dict['SubpixelDitherType'][0])
                elif observation_dict['SubpixelDitherType'] in ['STANDARD', 'IMAGING', 'SMALL-GRID-DITHER']:
                    number_of_subpixel_dithers = int(observation_dict['SubpixelPositions'])
            else:
                # For parallel instrument we ignore any dither info and set values to 0
                number_of_primary_dithers = 0
                number_of_subpixel_dithers = 0

                # The exception is if NIRISS WFSS is prime, in which case the dither pattern
                # (from NIRISS) can be imposed on only the second of each trio of exposures
                # (the exposure that is matched up to the NIRISS grism exposure), OR to all
                # three of each trio of exposures (meaning those matched up to the direct,
                # grism, and direct NIRISS exposures).

                # This doesn't actually matter at the moment since parallel instrument dither values
                # in the table are ignored.
                if prime_instrument == 'NIRISS' and prime_template_name == 'NirissWfss':
                    observation_dict['PrimaryDithers'] = prime_template.find(prime_ns + dither_key_name).text

                    # In the case where PrimaryDithers is e.g. 2-POINT-WITH-NIRCam,
                    # extract the '2' and place it in the PrimaryDithers field
                    try:
                        int_dithers = int(observation_dict['PrimaryDithers'])
                    except ValueError:
                        observation_dict['PrimaryDitherType'] = copy.deepcopy(observation_dict['PrimaryDithers'])
                        observation_dict['PrimaryDithers'] = observation_dict['PrimaryDithers'][0]
                    observation_dict['DitherSize'] = prime_template.find(prime_ns + 'DitherSize').text
                    number_of_primary_dithers = int(observation_dict['PrimaryDithers'][0])
                    number_of_subpixel_dithers = 1

            # Combine primary and subpixel dithers
            number_of_dithers = str(int(number_of_primary_dithers) * number_of_subpixel_dithers)
            self.logger.info('Number of dithers: {} primary * {} subpixel = {}'.format(number_of_primary_dithers,
                                                                                       number_of_subpixel_dithers,
                                                                                       number_of_dithers))

            # Find filter parameters for all filter configurations within obs
            filter_configs = template.findall('.//' + ns + 'FilterConfig')
            # loop over filter configurations
            for filter_config_index, filter_config in enumerate(filter_configs):
                filter_config_dict = {}
                # print('Filter config index {}'.format(filter_config_index))
                for element in filter_config:
                    key = element.tag.split(ns)[1]
                    value = element.text

                    # Engineering template directly sets filters and pupils separately
                    if template_name == 'NircamEngineeringImaging':
                        for k in ['ShortFilter', 'ShortPupil', 'LongFilter', 'LongPupil']:
                            if key == k:
                                filter_config_dict[k] = value
                    else:
                        # Imaging template needs to parse some pupil settings for paired filters
                        if key == 'ShortFilter':
                            ShortPupil, ShortFilter = self.separate_pupil_and_filter(value)
                            filter_config_dict['ShortPupil'] = ShortPupil
                            filter_config_dict['ShortFilter'] = ShortFilter
                        elif key == 'LongFilter':
                            LongPupil, LongFilter = self.separate_pupil_and_filter(value)
                            filter_config_dict['LongPupil'] = LongPupil
                            filter_config_dict['LongFilter'] = LongFilter

                    if key not in ['ShortFilter', 'ShortPupil', 'LongFilter', 'LongPupil']:
                        filter_config_dict[key] = value

                for key in self.APTObservationParams_keys:
                    if key in filter_config_dict.keys():
                        value = filter_config_dict[key]
                    elif key in observation_dict.keys():
                        value = observation_dict[key]
                    elif key in proposal_parameter_dictionary.keys():
                        value = proposal_parameter_dictionary[key]
                    elif key == 'Instrument':
                        value = instrument
                    elif key == 'ParallelInstrument':
                        value = parallel_instrument
                    elif key == 'number_of_dithers':
                        value = str(number_of_dithers)
                    elif key == 'FiducialPointOverride':
                        value = str(FiducialPointOverride)
                    elif key == 'APTTemplate':
                        value = template_name
                    elif key == 'Tracking':
                        value = tracking
                    elif key == 'Mode':
                        value = 'imaging'
                    elif key == 'NumOutputs':
                        subarray = template.find(ns + 'Subarray').text
                        value = 4 if subarray=='FULL' else 1
                    else:
                        value = NONE_STR
                    exposures_dictionary[key].append(value)


            ##########################################################
            # If NIRCam is prime with NIRISS WFSS parallel, then a DITHER_DIRECT
            # field will be added to the xml, describing whether the direct images
            # on either side of the grism exposure should be ditered. In a rare case
            # of the prime instrument having to conform to what the parallel instrument
            # is doing, this means that the NIRCam exposures taken at the same time as
            # the NIRISS direct images will also be dithered or not to match NIRISS. In
            # this special mode, NIRISS direct images are taken before and after each
            # grism exposure. Therefore for the NIRCam prime exposures, you can collect
            # them into groups of 3, and the dither_direct value will affect the first
            # and third exposures in each group. If dither_direct is false then all dithering,
            # primary and subpixel, are skipped. If dither_direct is true, then primary
            # and subpixel dithers are both done. It is guaranteed (by APT) in this case that
            # then number of exposures is a multiple of 3.
            try:
                dither_direct = observation_dict['DitherNirissWfssDirectImages']
                if dither_direct == 'NO_DITHERING':
                    if verbose:
                        self.logger.info(('NIRISS WFSS parallel and NO_DITHERING set for direct imgages. Adjusting '
                                          'number_of_dithers to 1 for the matching NIRCam exposures.'))
                    num_dithers = exposures_dictionary['number_of_dithers']
                    for counter in range(0, len(num_dithers), 3):
                        num_dithers[counter: counter+3] = ['1', num_dithers[counter+1], '1']
            except:
                pass
            ############################################################

        else:

            # Determine if there is an aperture override
            override = obs.find('.//' + self.apt + 'FiducialPointOverride')
            FiducialPointOverride = True if override is not None else False

            for element in template:
                element_tag_stripped = element.tag.split(ns)[1]

                # loop through exposures and collect dither parameters
                if element_tag_stripped == 'DitherPatternType':
                    DitherPatternType = element.text
                elif element_tag_stripped == 'ImageDithers':
                    number_of_primary_dithers = int(element.text)
                elif element_tag_stripped == 'SubpixelPositions':
                    if element.text != 'NONE':
                        number_of_subpixel_positions = int(element.text)
                elif element_tag_stripped == 'PrimaryDithers':
                    if (element.text is not None) & (element.text != 'NONE'):
                        number_of_primary_dithers = int(element.text)
                elif element_tag_stripped == 'Dithers':
                    dither_key = 'MrsDitherSpecification' if template_name=='MiriMRS' else 'DitherSpecification'
                    DitherPatternType = element.find(ns + dither_key).find(ns + 'DitherType').text
                    number_of_primary_dithers = int(DitherPatternType[0])
                elif element_tag_stripped == 'SubpixelDithers':
                    if element.text is not None:
                        number_of_subpixel_dithers = int(element.text)


                # handle the NIRISS AMI case
                if number_of_subpixel_positions > number_of_subpixel_dithers:
                    number_of_subpixel_dithers = np.copy(number_of_subpixel_positions)

                # Determine if there is an aperture override
                override = obs.find('.//' + self.apt + 'FiducialPointOverride')
                FiducialPointOverride = True if override is not None else False

                # To reduce confusion, if this is the parallel instrument,
                # set the number of dithers to zero, since the prime
                # instrument controls the number of dithers
                if parallel:
                    number_of_primary_dithers = 0
                    number_of_subpixel_dithers = 0

                # Combine primary and subpixel dithers
                number_of_dithers = str(number_of_primary_dithers * number_of_subpixel_dithers)

                # Different SI conventions of how to list exposure parameters
                # The MIRI Imaging template is extra different from the others, so adjust
                # which XML tags we're looking for if that's being used
                miri_exposure_list_tag = 'Filters' if template_name=='MiriImaging' else 'ExposureList'
                individual_exposure_tag = 'FilterConfig' if template_name=='MiriImaging' else 'Exposure'
                if ((instrument.lower() == 'niriss') and (element_tag_stripped == 'ExposureList')) | \
                        ((instrument.lower() == 'fgs') and (element_tag_stripped == 'Exposures'))| \
                        ((instrument.lower() == 'miri') and (element_tag_stripped == miri_exposure_list_tag))| \
                        ((instrument.lower() == 'nirspec') and (element_tag_stripped == 'Exposures')):
                    for exposure in element.findall(ns + individual_exposure_tag):
                        exposure_dict = {}

                        # Load dither information into dictionary
                        exposure_dict['DitherPatternType'] = DitherPatternType

                        if (number_of_dithers is None) | (number_of_dithers == 'NONE'):
                            number_of_dithers = 1 * number_of_subpixel_positions

                        exposure_dict[dither_key_name] = int(number_of_dithers)
                        exposure_dict['number_of_dithers'] = exposure_dict[dither_key_name]

                        for exposure_parameter in exposure:
                            parameter_tag_stripped = exposure_parameter.tag.split(ns)[1]
                            # if verbose:
                            #     print('{} {}'.format(parameter_tag_stripped, exposure_parameter.text))
                            exposure_dict[parameter_tag_stripped] = exposure_parameter.text

                        # fill dictionary to return
                        for key in self.APTObservationParams_keys:
                            if key in exposure_dict.keys():
                                value = exposure_dict[key]
                            elif key in proposal_parameter_dictionary.keys():
                                value = proposal_parameter_dictionary[key]
                            elif key == 'Instrument':
                                value = instrument
                            elif key == 'ParallelInstrument':
                                value = parallel_instrument
                            elif key == 'FiducialPointOverride':
                                value = str(FiducialPointOverride)
                            elif key == 'APTTemplate':
                                value = template_name
                            elif key == 'Tracking':
                                value = tracking
                            else:
                                value = NONE_STR

                            if (key in ['PrimaryDithers', 'ImageDithers']) and (str(value) == 'None'):
                                value = '1'

                            if (key == 'Mode'):
                                if template_name not in ['NirissAmi']:
                                    value = 'imaging'
                                else:
                                    value = 'ami'

                            exposures_dictionary[key].append(value)

                        # add keys that were not defined in self.APTObservationParams_keys
                        # (to be fixed in Class.__init__ later )
                        for key in exposure_dict.keys():
                            if key not in self.APTObservationParams_keys:
                                # if key not yet present, create entry
                                if key not in exposures_dictionary.keys():
                                    exposures_dictionary[key] = [str(exposure_dict[key])]
                                else:
                                    exposures_dictionary[key].append(str(exposure_dict[key]))

            if not parallel:
                self.logger.info('Number of dithers: {} primary * {} subpixel = {}'.format(number_of_primary_dithers,
                                                                                           number_of_subpixel_dithers,
                                                                                           number_of_dithers))

        for key in exposures_dictionary.keys():
            if type(exposures_dictionary[key]) is not list:
                exposures_dictionary[key] = list(exposures_dictionary[key])

        # make sure all list items in the returned dictionary have the same length
        for key, item in exposures_dictionary.items():
            if len(item) == 0:
                exposures_dictionary[key] = [0] * len(exposures_dictionary['Instrument'])

        return exposures_dictionary

    def read_commissioning_template(self, template, template_name, obs, prop_params):
        # Get proposal parameters
        pi_name, prop_id, prop_title, prop_category, science_category, coordparallel, i_obs, obs_label, target_name = prop_params

        # dictionary that holds the content of this observation only
        exposures_dictionary = copy.deepcopy(self.empty_exposures_dictionary)

        # Set namespace
        ns = "{http://www.stsci.edu/JWST/APT/Template/WfscCommissioning}"

        # Set parameters that are constant for all WFSC obs
        #typeflag = template_name
        typeflag = 'imaging'
        grismval = 'N/A'
        subarr = 'FULL'
        amps = 4
        pdithtype = 'NONE'
        pdither = '1'
        sdithtype = 'STANDARD'
        sdither = '1'

        # Find observation-specific parameters
        mod = template.find(ns + 'Module').text
        num_WFCgroups = int(template.find(ns + 'ExpectedWfcGroups').text)

        # Find filter parameters for all filter configurations within obs
        filter_configs = template.findall('.//' + ns + 'FilterConfig')

        # Check the target type in order to decide whether the tracking should be
        # sidereal or non-sidereal
        tracking = self.get_tracking_type(obs)

        for filt in filter_configs:
            sfilt = filt.find(ns + 'ShortFilter').text
            try:
                lfilt = filt.find(ns + 'LongFilter').text
            except AttributeError:
                lfilt = 'F480M'
            rpatt = filt.find(ns + 'ReadoutPattern').text
            grps = filt.find(ns + 'Groups').text
            ints = filt.find(ns + 'Integrations').text

            # Separate pupil and filter in case of filter that is
            # mounted in the pupil wheel
            if ' + ' in sfilt:
                split_ind = sfilt.find(' + ')
                short_pupil = sfilt[0:split_ind]
                sfilt = sfilt[split_ind + 1:]
            else:
                short_pupil = 'CLEAR'

            if ' + ' in lfilt:
                p = lfilt.find(' + ')
                long_pupil = lfilt[0:p]
                lfilt = lfilt[p + 1:]
            else:
                long_pupil = 'CLEAR'

            # Repeat for the number of expected WFSC groups + 1
            for j in range(num_WFCgroups + 1):
                # Add all parameters to dictionary
                tup_to_add = (pi_name, prop_id, prop_title, prop_category,
                              science_category, typeflag, mod, subarr, pdithtype,
                              pdither, sdithtype, sdither, sfilt, lfilt,
                              rpatt, grps, ints, short_pupil,
                              long_pupil, grismval, coordparallel,
                              i_obs , j + 1, template_name, 'NIRCAM', obs_label,
                              target_name, tracking)

                exposures_dictionary = self.add_exposure(exposures_dictionary, tup_to_add)
                self.obs_tuple_list.append(tup_to_add)

            # Add the number of dithers
            number_of_dithers = int(pdither) * int(sdither)
            exposures_dictionary['number_of_dithers'] = [str(number_of_dithers)] * len(exposures_dictionary['Instrument'])

            # Force 4 amp readout
            exposures_dictionary['NumOutputs'] = [amps] * len(exposures_dictionary['Instrument'])

        # make sure all list items in the returned dictionary have the same length
        for key, item in exposures_dictionary.items():
            if len(item) == 0:
                exposures_dictionary[key] = [0] * len(exposures_dictionary['Instrument'])
                # self.APTObservationParams = self.add_exposure(self.APTObservationParams, tup_to_add)
                # self.obs_tuple_list.append(tup_to_add)

        return exposures_dictionary, num_WFCgroups

    def read_globalalignment_template(self, template, template_name, obs, prop_params):
        # Get proposal parameters
        pi_name, prop_id, prop_title, prop_category, science_category, coordparallel, i_obs, obs_label, target_name = prop_params

        # dictionary that holds the content of this observation only
        exposures_dictionary = copy.deepcopy(self.empty_exposures_dictionary)

        ns = "{http://www.stsci.edu/JWST/APT/Template/WfscGlobalAlignment}"

        # Set parameters that are constant for all WFSC obs
        #typeflag = template_name
        typeflag = 'imaging'
        grismval = 'N/A'
        short_pupil = 'CLEAR'
        subarr = 'FULL'
        pdither = '1'
        pdithtype = 'NONE'
        sdithtype = 'STANDARD'
        sdither = '1'

        # Determine the Global Alignment Iteration Type
        GA_iteration = obs.find('.//' + ns + 'GaIteration').text

        if GA_iteration == 'ADJUST1' or GA_iteration == 'CORRECT':
            # 3 NIRCam and 1 FGS images
            n_exp = 4
        elif GA_iteration == 'ADJUST2' or GA_iteration == 'CORRECT+ADJUST':
            # 5 NIRCam and 2 FGS
            n_exp = 7
        elif GA_iteration == 'BSCORRECT':
            # 2 NIRCam and 1 FGS
            n_exp = 3

        # Find observation-specific parameters
        mod = template.find(ns + 'Module').text
        # num_WFCgroups = int(template.find(ns + 'ExpectedWfcGroups').text)

        # Find filter parameters for all filter configurations within obs
        ga_nircam_configs = template.findall('.//' + ns + 'NircamParameters')
        ga_fgs_configs = template.findall('.//' + ns + 'FgsParameters')

        # Check the target type in order to decide whether the tracking should be
        # sidereal or non-sidereal
        tracking = self.get_tracking_type(obs)

        for conf in ga_nircam_configs:
            sfilt = conf.find(ns + 'ShortFilter').text
            try:
                lfilt = conf.find(ns + 'LongFilter').text
            except AttributeError:
                lfilt = 'F480M'
            rpatt = conf.find(ns + 'ReadoutPattern').text
            grps = conf.find(ns + 'Groups').text
            ints = conf.find(ns + 'Integrations').text

            # Separate pupil and filter in case of filter that is
            # mounted in the pupil wheel
            if ' + ' in sfilt:
                split_ind = sfilt.find(' + ')
                short_pupil = sfilt[0:split_ind]
                sfilt = sfilt[split_ind + 1:]
            else:
                short_pupil = 'CLEAR'

            if ' + ' in lfilt:
                p = lfilt.find(' + ')
                long_pupil = lfilt[0:p]
                lfilt = lfilt[p + 1:]
            else:
                long_pupil = 'CLEAR'

        for fgs_conf in ga_fgs_configs:
            fgs_grps = fgs_conf.find(ns + 'Groups').text
            fgs_ints = fgs_conf.find(ns + 'Integrations').text

        guider_det_num = _get_guider_number_from_special_requirements(self.apt, obs)
        fgs_subarr = "FGS{}_FULL".format(guider_det_num)

        # Repeat for the number of exposures
        for j in range(n_exp):
            # Add all parameters to dictionary

            if j==2 or j==5:
                # This is an FGS image as part of GA

                # Add FGS exposure to the dictionary
                tup_to_add = (pi_name, prop_id, prop_title, prop_category,
                              science_category, typeflag, 'N/A', fgs_subarr, pdithtype,
                              pdither, sdithtype, sdither, 'N/A', 'N/A',
                              'FGSRAPID', fgs_grps, fgs_ints, 'N/A',
                              'N/A', 'N/A', coordparallel,
                              i_obs, j + 1, template_name, 'FGS', obs_label,
                              target_name, tracking)
            else:
                # This is a NIRCam image as part of GA

                tup_to_add = (pi_name, prop_id, prop_title, prop_category,
                              science_category, typeflag, mod, subarr, pdithtype,
                              pdither, sdithtype, sdither, sfilt, lfilt,
                              rpatt, grps, ints, short_pupil,
                              long_pupil, grismval, coordparallel,
                              i_obs, j + 1, template_name, 'NIRCAM', obs_label,
                              target_name, tracking)

            exposures_dictionary = self.add_exposure(exposures_dictionary, tup_to_add)
            self.obs_tuple_list.append(tup_to_add)

        # Add the number of dithers
        number_of_dithers = int(pdither) * int(sdither)
        exposures_dictionary['number_of_dithers'] = [str(number_of_dithers)] * len(
            exposures_dictionary['Instrument'])

        # make sure all list items in the returned dictionary have the same length
        for key, item in exposures_dictionary.items():
            if len(item) == 0:
                exposures_dictionary[key] = [0] * len(exposures_dictionary['Instrument'])

            # self.APTObservationParams = self.add_exposure(self.APTObservationParams, tup_to_add)
            # self.obs_tuple_list.append(tup_to_add)

        # All exposures are full frame 4 amp readouts
        exposures_dictionary['NumOutputs'] = [4] * len(exposures_dictionary['NumOutputs'])

        # Add the target RA and Dec to the exposure dictionary
        exposures_dictionary['TargetRA'] = [self.target_info[target_name][0]] * len(exposures_dictionary['NumOutputs'])
        exposures_dictionary['TargetDec'] = [self.target_info[target_name][1]] * len(exposures_dictionary['NumOutputs'])

        return exposures_dictionary, n_exp

    def read_coarsephasing_template(self, template, template_name, obs, prop_params):
        # Get proposal parameters
        pi_name, prop_id, prop_title, prop_category, science_category, coordparallel, i_obs, obs_label, target_name = prop_params

        # dictionary that holds the content of this observation only
        exposures_dictionary = copy.deepcopy(self.empty_exposures_dictionary)

        ns = "{http://www.stsci.edu/JWST/APT/Template/WfscCoarsePhasing}"

        # Set parameters that are constant for all WFSC obs
        #typeflag = template_name
        typeflag = 'imaging'
        grismval = 'N/A'
        pdither = '1'
        pdithtype = 'NONE'
        sdithtype = 'STANDARD'
        sdither = '1'

        # Find the module and derive the subarrays
        mod = template.find(ns + 'Module').text
        mods = [mod] * 12
        if mod == 'A':
            subarrs = ['SUB96DHSPILA'] + ['FULL'] * 6
        if mod == 'B':
            subarrs = ['SUB96DHSPILB'] + ['FULL'] * 6

        # Find the exposure parameters for the In Focus, DHS, and Defocus modes
        readouts = [r.text for r in obs.findall('.//' + ns + 'ReadoutPattern')]
        groups = [g.text for g in obs.findall('.//' + ns + 'Groups')]
        integrations = [i.text for i in obs.findall('.//' + ns + 'Integrations')]
        inds = [0, 1, 1, 1, 1, 2, 2]
        readouts = np.array(readouts)[inds]
        groups = np.array(groups)[inds]
        integrations = np.array(integrations)[inds]

        # List the pupils and filters in the appropriate order
        sw_pupils = ['CLEAR', 'GDHS0', 'GDHS0', 'GDHS60', 'GDHS60', 'WLP8', 'WLM8']
        sw_filts = ['F212N', 'F150W2', 'F150W2', 'F150W2', 'F150W2', 'F212N', 'F212N']
        lw_pupils = ['F405N'] * 7
        lw_filts = ['F444W'] * 7

        # Check the target type in order to decide whether the tracking should be
        # sidereal or non-sidereal
        tracking = self.get_tracking_type(obs)

        for i in range(7):
            mod = mods[i]
            subarr = subarrs[i]

            sfilt = sw_filts[i]
            lfilt = lw_filts[i]
            short_pupil = sw_pupils[i]
            long_pupil = lw_pupils[i]

            rpatt = readouts[i]
            grps = groups[i]
            ints = integrations[i]

            # Repeat for two dithers
            for j in range(2):
                # Add all parameters to dictionary
                tup_to_add = (pi_name, prop_id, prop_title, prop_category,
                              science_category, typeflag, mod, subarr, pdithtype,
                              pdither, sdithtype, sdither, sfilt, lfilt,
                              rpatt, grps, ints, short_pupil,
                              long_pupil, grismval, coordparallel,
                              i_obs, j + 1, template_name, 'NIRCAM', obs_label,
                              target_name, tracking)

                exposures_dictionary = self.add_exposure(exposures_dictionary, tup_to_add)
                self.obs_tuple_list.append(tup_to_add)

        # Add the number of dithers
        number_of_dithers = int(pdither) * int(sdither)
        exposures_dictionary['number_of_dithers'] = [str(number_of_dithers)] * len(
            exposures_dictionary['Instrument'])

        # make sure all list items in the returned dictionary have the same length
        for key, item in exposures_dictionary.items():
            if len(item) == 0:
                exposures_dictionary[key] = [0] * len(exposures_dictionary['Instrument'])

                    # self.APTObservationParams = self.add_exposure(self.APTObservationParams, tup_to_add)
                    # self.obs_tuple_list.append(tup_to_add)
        n_tiles_phasing = 14

        return exposures_dictionary, n_tiles_phasing

    def read_finephasing_template(self, template, template_name, obs, prop_params):
        # Get proposal parameters
        pi_name, prop_id, prop_title, prop_category, science_category, coordparallel, i_obs, obs_label, target_name = prop_params

        # dictionary that holds the content of this observation only
        exposures_dictionary = copy.deepcopy(self.empty_exposures_dictionary)

        ns = "{http://www.stsci.edu/JWST/APT/Template/WfscFinePhasing}"

        # Set parameters that are constant for all WFSC obs
        #typeflag = template_name
        typeflag = 'imaging'
        grismval = 'N/A'
        pdither = '1'
        pdithtype = 'NONE'
        sdithtype = 'STANDARD'
        sdither = '1'

        # Find the module and derive the subarrays
        mod = template.find(ns + 'Module').text

        # Determine the sensing type, and list the pupils and filters
        # in the appropriate order
        sensing_type = obs.find('.//' + ns + 'SensingType').text

        # Check the target type in order to decide whether the tracking should be
        # sidereal or non-sidereal
        tracking = self.get_tracking_type(obs)

        n_configs = 0
        n_dithers = []
        subarrs = []
        mods = []
        sw_pupils = []
        sw_filts = []
        lw_pupils = []
        lw_filts = []
        readouts = []
        groups = []
        integrations = []

        if sensing_type in ['LOS Jitter', 'Both']:
            ########## IF WE WANT TO MODEL TARGET ACQ:
            # n_configs += 2
            # n_dithers += [1] * n_configs

            # subarrs += ['SUB64FP1' + mod, 'SUB8FP1' + mod]
            # mods += subarrs

            # sw_pupils += ['CLEAR', 'CLEAR']
            # sw_filts += ['F212N', 'F200W']
            # lw_pupils += ['F405N'] * n_configs # default?
            # lw_filts += ['F444W'] * n_configs # default?

            # # Find/define the exposure parameters for the target
            # # acquisition and LOS imaging modes
            # readouts += ['RAPID', 'RAPID']
            # acq_groups = obs.find('.//' + ns + 'AcqNumGroups').text
            # LOSimg_groups = obs.find('.//' + ns + 'LosImgNumGroups').text
            # groups += [acq_groups, LOSimg_groups]
            # LOSimg_ints = obs.find('.//' + ns + 'LosImgNumInts').text
            # integrations += [1, LOSimg_ints]

            n_configs += 1
            n_dithers += [1] * n_configs

            subarrs += ['SUB8FP1{}'.format(mod)]
            mods += [mod]

            sw_pupils += ['CLEAR']
            sw_filts += ['F200W']
            lw_pupils += ['F405N'] * n_configs # default?
            lw_filts += ['F444W'] * n_configs # default?

            # Find/define the exposure parameters for the target
            # acquisition and LOS imaging modes
            readouts += ['RAPID']
            groups += [obs.find('.//' + ns + 'LosImgNumGroups').text]
            integrations += [obs.find('.//' + ns + 'LosImgNumInts').text]

        if sensing_type in ['Fine Phasing', 'Both']:
            # Deterimine what diversity of sensing
            diversity = obs.find('.//' + ns + 'Diversity').text
            if diversity == 'ALL':
                n_configs_fp = 5
                n_configs += n_configs_fp
            elif diversity == 'ALL+187N':
                n_configs_fp = 7
                n_configs += n_configs_fp
            elif diversity == 'PM8':
                n_configs_fp = 2
                n_configs += n_configs_fp

            n_dithers += [2] * n_configs_fp

            subarrs += ['FULL'] * n_configs_fp
            mods += [mod] * n_configs_fp

            sw_pupils += ['WLM8', 'WLP8', 'WLP8', 'WLM8', 'CLEAR', 'WLM8', 'WLP8'][:n_configs_fp]
            sw_filts += ['F212N', 'F212N', 'WLP4', 'WLP4', 'WLP4', 'F187N', 'F187N'][:n_configs_fp]
            lw_pupils += ['F405N'] * n_configs_fp
            lw_filts += ['F444W'] * n_configs_fp

            # Find the exposure parameters for the +/- 8, + 12, and +/-4 modes
            readouts_fp = [r.text for r in obs.findall('.//' + ns + 'ReadoutPattern')]
            groups_fp = [g.text for g in obs.findall('.//' + ns + 'Groups')]
            integrations_fp = [i.text for i in obs.findall('.//' + ns + 'Integrations')]
            inds = [0, 0, 1, 2, 2, 3, 3][:n_configs_fp]
            readouts += list(np.array(readouts_fp)[inds])
            groups += list(np.array(groups_fp)[inds])
            integrations += list(np.array(integrations_fp)[inds])


        sensing = obs.find('.//' + self.apt + 'WavefrontSensing').text
        if sensing == 'SENSING_ONLY':
            n_repeats = 1
        else:
            n_repeats = 2

        for z in range(n_repeats):
            for i in range(n_configs):
                subarr = subarrs[i]
                mod = mods[i]

                sfilt = sw_filts[i]
                lfilt = lw_filts[i]
                short_pupil = sw_pupils[i]
                long_pupil = lw_pupils[i]

                rpatt = readouts[i]
                grps = groups[i]
                ints = integrations[i]

                n_dith = n_dithers[i]

                # Add all parameters to dictionary
                tup_to_add = (pi_name, prop_id, prop_title, prop_category,
                              science_category, typeflag, mod, subarr, pdithtype,
                              pdither, sdithtype, sdither, sfilt, lfilt,
                              rpatt, grps, ints, short_pupil,
                              long_pupil, grismval, coordparallel,
                              i_obs, 1, template_name, 'NIRCAM', obs_label,
                              target_name, tracking)

                exposures_dictionary = self.add_exposure(exposures_dictionary, tup_to_add)
                exposures_dictionary['number_of_dithers'] += str(n_dith)

                self.obs_tuple_list.append(tup_to_add)

        # make sure all list items in the returned dictionary have the same length
        for key, item in exposures_dictionary.items():
            if len(item) == 0:
                exposures_dictionary[key] = [0] * len(exposures_dictionary['Instrument'])
                    # self.APTObservationParams = self.add_exposure(self.APTObservationParams, tup_to_add)
                    # self.obs_tuple_list.append(tup_to_add)

        n_tiles_phasing = sum(n_dithers) * n_repeats

        return exposures_dictionary, n_tiles_phasing

    def read_nircam_grism_time_series(self, template, template_name, obs, proposal_parameter_dictionary):
        """Parse a NIRCam Grism Time Series observation template from an APT xml file.
        Produce an exposure dictionary that lists all exposures (excluding dithers)
        from the template

        Parameters
        ----------
        template : lxml.etree._Element
            Template section from APT xml

        template_name : str
            The type of template (e.g. 'NirissWfss')

        obs : lxml.etree._Element
            Observation section from APT xml

        proposal_param_dict : dict
            Dictionary of proposal level information from the xml file
            (e.g. PI, Science Category, etc)

        Returns
        -------
        exposures_dictionary : dict
            Dictionary containing details on all exposures contained within
            the template. These details include things like filter, pupil,
            readout pattern, subarray, etc. Specifically for Grism Time Series,
            there will be entries for the TA exposure and the Time Series
            exposure.
        """
        # Dictionary that holds the content of this observation only
        exposures_dictionary = copy.deepcopy(self.empty_exposures_dictionary)

        # Set namespace
        ns = "{http://www.stsci.edu/JWST/APT/Template/NircamGrismTimeSeries}"

        # Observation-wide info
        instrument = obs.find(self.apt + 'Instrument').text

        # Get target name
        try:
            targ_name = obs.find(self.apt + 'TargetID').text.split(' ')[1]
        except IndexError as e:
            self.logger.info("No target ID for observation: {}".format(obs))
            targ_name = obs.find(self.apt + 'TargetID').text.split(' ')[0]

        # Mode specific info, including target acq
        acq_target = template.find(ns + 'AcqTargetID').text
        if acq_target == 'Same Target as Observation':
            acq_target = targ_name

        acq_readout_pattern = template.find(ns + 'AcqReadoutPattern').text
        acq_groups = template.find(ns + 'AcqGroups').text
        acq_filter = 'F335M'
        acq_subarray = 'SUB32TATSGRISM'
        acq_integrations = '1'
        module = template.find(ns + 'Module').text
        subarray = template.find(ns + 'Subarray').text
        readout_pattern = template.find(ns + 'ReadoutPattern').text
        groups = template.find(ns + 'Groups').text
        integrations = template.find(ns + 'Integrations').text
        num_exps = template.find(ns + 'NumExps').text
        num_outputs = template.find(ns + 'NumOutputs').text  # Number of amplifiers used

        # Check the target type in order to decide whether the tracking should be
        # sidereal or non-sidereal
        tracking = self.get_tracking_type(obs)

        # Neither TA exposures nor Grism Time Series exposures allow dithering
        number_of_dithers = '1'
        number_of_subpixel_positions = '1'
        number_of_primary_dithers = '1'
        number_of_subpixel_dithers = '1'

        short_pupil_filter = template.find(ns + 'ShortPupilFilter').text
        long_pupil_filter = template.find(ns + 'LongPupilFilter').text
        short_pupil, short_filter = self.separate_pupil_and_filter(short_pupil_filter)
        long_pupil, long_filter = self.separate_pupil_and_filter(long_pupil_filter)

        # Populate observation dictionary with TA and Time Series exposures
        exposures_dictionary['Instrument'] = [instrument] * 2
        exposures_dictionary['Module'] = [module] * 2
        exposures_dictionary['TargetID'] = [acq_target, targ_name]
        exposures_dictionary['APTTemplate'] = [template_name] * 2
        exposures_dictionary['Mode'] = ['imaging', 'ts_grism']
        exposures_dictionary['Subarray'] = [acq_subarray, subarray]
        exposures_dictionary['ReadoutPattern'] = [acq_readout_pattern, readout_pattern]
        exposures_dictionary['Groups'] = [acq_groups, groups]
        exposures_dictionary['Integrations'] = [acq_integrations, integrations]
        exposures_dictionary['Exposures'] = ['1', num_exps]
        exposures_dictionary['NumOutputs'] = ['1', num_outputs]
        exposures_dictionary['PrimaryDithers'] = [number_of_primary_dithers, number_of_primary_dithers]
        exposures_dictionary['SubpixelPositions'] = [number_of_subpixel_dithers, number_of_subpixel_dithers]
        exposures_dictionary['ImageDithers'] = [number_of_dithers, number_of_dithers]
        exposures_dictionary['number_of_dithers'] = [number_of_dithers, number_of_dithers]
        exposures_dictionary['ShortFilter'] = [NONE_STR, short_filter]
        exposures_dictionary['ShortPupil'] = [NONE_STR, short_pupil]
        exposures_dictionary['LongFilter'] = [acq_filter, long_filter]
        exposures_dictionary['LongPupil'] = ['CLEAR', long_pupil]
        exposures_dictionary['FiducialPointOverride'] = [str(False)] * 2
        exposures_dictionary['ParallelInstrument'] = [False] * 2
        exposures_dictionary['Tracking'] = [tracking] * 2

        # Populate other keywords with None
        for key in self.APTObservationParams_keys:
            value = 'reset_value'
            if key in proposal_parameter_dictionary.keys() and key != 'TargetID':
                value = [proposal_parameter_dictionary[key]] * 2
            elif exposures_dictionary[key] == []:
                value = [NONE_STR] * 2
            if value != 'reset_value':
                exposures_dictionary[key].extend(value)

        return exposures_dictionary

    def read_nircam_imaging_time_series(self, template, template_name, obs, proposal_parameter_dictionary):
        """Parse a NIRCam Imaging Time Series observation template from an APT xml file.
        Produce an exposure dictionary that lists all exposures (excluding dithers)
        from the template

        Parameters
        ----------
        template : lxml.etree._Element
            Template section from APT xml

        template_name : str
            The type of template (e.g. 'NirissWfss')

        obs : lxml.etree._Element
            Observation section from APT xml

        proposal_param_dict : dict
            Dictionary of proposal level information from the xml file
            (e.g. PI, Science Category, etc)

        Returns
        -------
        exposures_dictionary : dict
            Dictionary containing details on all exposures contained within
            the template. These details include things like filter, pupil,
            readout pattern, subarray, etc. Specifically for Grism Time Series,
            there will be entries for the TA exposure and the Time Series
            exposure.
        """
        # Dictionary that holds the content of this observation only
        exposures_dictionary = copy.deepcopy(self.empty_exposures_dictionary)

        # Set namespace
        ns = "{http://www.stsci.edu/JWST/APT/Template/NircamTimeSeries}"

        # Observation-wide info
        instrument = obs.find(self.apt + 'Instrument').text

        # Check the target type in order to decide whether the tracking should be
        # sidereal or non-sidereal
        tracking = self.get_tracking_type(obs)

        # Get target name
        try:
            targ_name = obs.find(self.apt + 'TargetID').text.split(' ')[1]
        except IndexError as e:
            self.logger.info("No target ID for observation: {}".format(obs))
            targ_name = obs.find(self.apt + 'TargetID').text.split(' ')[0]

        # Mode specific info, including target acq
        acq_target = template.find(ns + 'AcqTargetID').text
        if acq_target == 'Same Target as Observation':
            acq_target = targ_name

        acq_readout_pattern = template.find(ns + 'AcqReadoutPattern').text
        acq_groups = template.find(ns + 'AcqGroups').text
        acq_filter = 'F335M'
        acq_subarray = 'SUB32TATS'
        acq_integrations = '1'
        module = template.find(ns + 'Module').text
        subarray = template.find(ns + 'Subarray').text
        readout_pattern = template.find(ns + 'ReadoutPattern').text
        groups = template.find(ns + 'Groups').text
        integrations = template.find(ns + 'Integrations').text
        num_exps = template.find(ns + 'NumExps').text

        if subarray == 'FULL':
            num_outputs = 4
        else:
            num_outputs = 1

        # Neither TA exposures nor Grism Time Series exposures allow dithering
        number_of_dithers = '1'
        number_of_subpixel_positions = '1'
        number_of_primary_dithers = '1'
        number_of_subpixel_dithers = '1'

        short_pupil = template.find(ns + 'ShortPupil').text
        short_filter = template.find(ns + 'ShortFilter').text
        long_pupil = template.find(ns + 'LongPupil').text
        long_filter = template.find(ns + 'LongFilter').text

        # Populate observation dictionary with TA and Time Series exposures
        exposures_dictionary['Instrument'] = [instrument] * 2
        exposures_dictionary['Module'] = [module] * 2
        exposures_dictionary['TargetID'] = [acq_target, targ_name]
        exposures_dictionary['APTTemplate'] = [template_name] * 2
        exposures_dictionary['Mode'] = ['imaging', 'ts_imaging']
        exposures_dictionary['Subarray'] = [acq_subarray, subarray]
        exposures_dictionary['ReadoutPattern'] = [acq_readout_pattern, readout_pattern]
        exposures_dictionary['Groups'] = [acq_groups, groups]
        exposures_dictionary['Integrations'] = [acq_integrations, integrations]
        exposures_dictionary['Exposures'] = ['1', num_exps]
        exposures_dictionary['NumOutputs'] = ['1', num_outputs]
        exposures_dictionary['PrimaryDithers'] = [number_of_primary_dithers, number_of_primary_dithers]
        exposures_dictionary['SubpixelPositions'] = [number_of_subpixel_dithers, number_of_subpixel_dithers]
        exposures_dictionary['ImageDithers'] = [number_of_dithers, number_of_dithers]
        exposures_dictionary['number_of_dithers'] = [number_of_dithers, number_of_dithers]
        exposures_dictionary['ShortFilter'] = [NONE_STR, short_filter]
        exposures_dictionary['ShortPupil'] = [NONE_STR, short_pupil]
        exposures_dictionary['LongFilter'] = [acq_filter, long_filter]
        exposures_dictionary['LongPupil'] = ['CLEAR', long_pupil]
        exposures_dictionary['FiducialPointOverride'] = [str(False)] * 2
        exposures_dictionary['ParallelInstrument'] = [False] * 2
        exposures_dictionary['Tracking'] = [tracking] * 2

        # Populate other keywords with None
        for key in self.APTObservationParams_keys:
            value = 'reset_value'
            if key in proposal_parameter_dictionary.keys() and key != 'TargetID':
                value = [proposal_parameter_dictionary[key]] * 2
            elif exposures_dictionary[key] == []:
                value = [NONE_STR] * 2
            if value != 'reset_value':
                exposures_dictionary[key].extend(value)

        return exposures_dictionary

    def read_nircam_wfss_template(self, template, template_name, obs, proposal_param_dict):
        """Parse a NIRCam WFSS observation template from an APT xml file. Produce an exposure dictionary
        that lists all exposures (excluding dithers) from the template.

        Parameters
        ----------
        template : lxml.etree._Element
            Template section from APT xml

        template_name : str
            The type of template (e.g. 'NirissWfss')

        obs : lxml.etree._Element
            Observation section from APT xml

        proposal_param_dict : dict
            Dictionary of proposal level information from the xml file
            (e.g. PI, Science Category, etc)

        Returns
        -------
        exposures_dictionary : dict
            Dictionary containing details on all exposures contained within the template. These details
            include things like filter, pupil, readout pattern, subarray, etc
        """
        # Dictionary that holds the content of this observation only
        exposures_dictionary = copy.deepcopy(self.empty_exposures_dictionary)

        # Set namespace
        ns = "{http://www.stsci.edu/JWST/APT/Template/NircamWfss}"

        # Observation-wide info
        instrument = obs.find(self.apt + 'Instrument').text
        module = template.find(ns + 'Module').text
        subarr = template.find(ns + 'Subarray').text

        # Check the target type in order to decide whether the tracking should be
        # sidereal or non-sidereal
        tracking = self.get_tracking_type(obs)

        # Determine if there is an aperture override
        override = obs.find('.//' + self.apt + 'FiducialPointOverride')
        FiducialPointOverride = True if override is not None else False

        # Get primary and subpixel dither values for the grism exposures
        primary_dither_type_grism = template.find(ns + 'PrimaryDitherType').text
        if primary_dither_type_grism.lower() != 'none':
            primary_dither_grism = template.find(ns + 'PrimaryDithers').text
        else:
            primary_dither_grism = '1'
        subpix_dither_type_grism = template.find(ns + 'SubpixelPositions').text
        if subpix_dither_type_grism.lower() != 'none':
            subpix_dither_grism = subpix_dither_type_grism[0]
        else:
            subpix_dither_grism = '1'
        grism_number_of_dithers = str(int(primary_dither_grism) * int(subpix_dither_grism))

        # Direct and out of field images are never dithered
        primary_dither_direct = '1'
        primary_dither_type_direct = 'None'
        subpix_dither_direct = '1'
        subpix_dither_type_direct = 'None'
        direct_number_of_dithers = '1'

        # Which grism(s) are to be used
        grismval = template.find(ns + 'Grism').text
        if grismval == 'BOTH':
            grismval = ['GRISMR', 'GRISMC']
        else:
            grismval = [grismval]

        explist = template.find(ns + 'ExposureList')
        expseqs = explist.findall(ns + 'ExposureSequences')

        # if BOTH was specified for the grism,
        # then we need to repeat the sequence of
        # grism/direct/grism/direct/outoffield for each grism
        for grism_number, grism in enumerate(grismval):
            out_of_field_dict = {}
            for expseq in expseqs:
                # sequence = grism, direct, grism, direct, outoffield, outoffield
                # if grism == both, entire sequence is done for grismr,
                # then repeated for grismc

                # Mini dictionary just for exposure sequence
                exp_seq_dict = {}

                # Switch the order of the grism and direct
                # exposures from that within xml file in order for them to be chronological
                grismexp = expseq.find(ns + 'GrismExposure')
                grism_typeflag = 'wfss'
                grism_short_filter = grismexp.find(ns + 'ShortFilter').text
                grism_long_filter = grismexp.find(ns + 'LongFilter').text
                grism_readpatt = grismexp.find(ns + 'ReadoutPattern').text
                grism_groups = grismexp.find(ns + 'Groups').text
                grism_integrations = grismexp.find(ns + 'Integrations').text

                # Separate SW pupil and filter in case of filter
                # that is mounted in the pupil wheel
                grism_short_pupil, grism_short_filter = self.separate_pupil_and_filter(grism_short_filter)
                grism_long_pupil = grism

                # Check to see if the direct image will be collected
                # This will be either 'true' or 'false'
                direct_done = expseq.find(ns + 'DirectImage').text

                # We need to collect the information about the direct image even if it is
                # not being observed, because the same info will be used for the out of field
                # observations later.
                directexp = expseq.find(ns + 'DiExposure')
                direct_typeflag = 'imaging'
                direct_grism = 'N/A'
                direct_short_filter = directexp.find(ns + 'ShortFilter').text
                direct_long_filter = directexp.find(ns + 'LongFilter').text
                direct_readpatt = directexp.find(ns + 'ReadoutPattern').text
                direct_groups = directexp.find(ns + 'Groups').text
                direct_integrations = directexp.find(ns + 'Integrations').text

                # Separate pupil and filter in case of filter
                # that is mounted in the pupil wheel
                direct_short_pupil, direct_short_filter = self.separate_pupil_and_filter(direct_short_filter)
                direct_long_pupil, direct_long_filter = self.separate_pupil_and_filter(direct_long_filter)

                # Only add the direct image if APT says it will be observed
                if direct_done == 'true':
                    exp_seq_dict['Mode'] = [grism_typeflag, direct_typeflag]
                    exp_seq_dict['Module'] = [module] * 2
                    exp_seq_dict['Subarray'] = [subarr] * 2
                    exp_seq_dict['PrimaryDitherType'] = [primary_dither_type_grism, primary_dither_type_direct]
                    exp_seq_dict['PrimaryDithers'] = [primary_dither_grism, primary_dither_direct]
                    exp_seq_dict['SubpixelPositions'] = [subpix_dither_grism, subpix_dither_direct]
                    exp_seq_dict['SubpixelDitherType'] = [subpix_dither_type_grism, subpix_dither_type_direct]
                    exp_seq_dict['CoordinatedParallel'] = ['false'] * 2
                    exp_seq_dict['Instrument'] = [instrument] * 2
                    exp_seq_dict['ParallelInstrument'] = [False] * 2
                    exp_seq_dict['ShortFilter'] = [grism_short_filter, direct_short_filter]
                    exp_seq_dict['LongFilter'] = [grism_long_filter, direct_long_filter]
                    exp_seq_dict['ReadoutPattern'] = [grism_readpatt, direct_readpatt]
                    exp_seq_dict['Groups'] = [grism_groups, direct_groups]
                    exp_seq_dict['Integrations'] = [grism_integrations, direct_integrations]
                    exp_seq_dict['ShortPupil'] = [grism_short_pupil, direct_short_pupil]
                    exp_seq_dict['LongPupil'] = [grism_long_pupil, direct_long_pupil]
                    exp_seq_dict['Grism'] = [grism, direct_grism]
                    exp_seq_dict['ObservationID'] = [proposal_param_dict['ObservationID']] * 2
                    exp_seq_dict['TileNumber'] = ['1'] * 2
                    exp_seq_dict['APTTemplate'] = [template_name] * 2
                    exp_seq_dict['ObservationName'] = [proposal_param_dict['ObservationName']] * 2
                    exp_seq_dict['number_of_dithers'] = [grism_number_of_dithers, direct_number_of_dithers]
                    exp_seq_dict['FilterWheel'] = ['none'] * 2  # used for NIRISS
                    exp_seq_dict['PupilWheel'] = ['none'] * 2  # used for NIRISS
                    exp_seq_dict['FiducialPointOverride'] = [FiducialPointOverride] * 2
                    exp_seq_dict['Tracking'] = [tracking] * 2
                else:
                    exp_seq_dict['Mode'] = [grism_typeflag]
                    exp_seq_dict['Module'] = [module]
                    exp_seq_dict['Subarray'] = [subarr]
                    exp_seq_dict['PrimaryDitherType'] = [primary_dither_type_grism]
                    exp_seq_dict['PrimaryDithers'] = [primary_dither_grism]
                    exp_seq_dict['SubpixelPositions'] = [subpix_dither_grism]
                    exp_seq_dict['SubpixelDitherType'] = [subpix_dither_type_grism]
                    exp_seq_dict['CoordinatedParallel'] = ['false']
                    exp_seq_dict['Instrument'] = [instrument]
                    exp_seq_dict['ParallelInstrument'] = [False]
                    exp_seq_dict['ShortFilter'] = [grism_short_filter]
                    exp_seq_dict['LongFilter'] = [grism_long_filter]
                    exp_seq_dict['ReadoutPattern'] = [grism_readpatt]
                    exp_seq_dict['Groups'] = [grism_groups]
                    exp_seq_dict['Integrations'] = [grism_integrations]
                    exp_seq_dict['ShortPupil'] = [grism_short_pupil]
                    exp_seq_dict['LongPupil'] = [grism_long_pupil]
                    exp_seq_dict['Grism'] = [grism]
                    exp_seq_dict['ObservationID'] = [proposal_param_dict['ObservationID']]
                    exp_seq_dict['TileNumber'] = ['1']
                    exp_seq_dict['APTTemplate'] = [template_name]
                    exp_seq_dict['ObservationName'] = [proposal_param_dict['ObservationName']]
                    exp_seq_dict['number_of_dithers'] = [grism_number_of_dithers]
                    exp_seq_dict['FilterWheel'] = ['none']  # used for NIRISS
                    exp_seq_dict['PupilWheel'] = ['none']  # used for NIRISS
                    exp_seq_dict['FiducialPointOverride'] = [FiducialPointOverride]
                    exp_seq_dict['Tracking'] = [tracking]

                # Add exp_seq_dict to the exposures_dictionary
                exposures_dictionary = self.append_to_exposures_dictionary(exposures_dictionary,
                                                                           exp_seq_dict,
                                                                           proposal_param_dict)

            # Now we need to add the two out-of-field exposures, which are
            # not present in the APT file (but are in the associated pointing
            # file from APT.) We can just duplicate the entries for the direct
            # images taken immediately prior. Out of field exposures are collected
            # for each grism in the observation
            out_of_field_dict['Mode'] = [direct_typeflag] * 2
            out_of_field_dict['Module'] = [module] * 2
            out_of_field_dict['Subarray'] = [subarr] * 2
            out_of_field_dict['PrimaryDitherType'] = [primary_dither_type_direct] * 2
            out_of_field_dict['PrimaryDithers'] = [primary_dither_direct] * 2
            out_of_field_dict['SubpixelPositions'] = [subpix_dither_direct] * 2
            out_of_field_dict['SubpixelDitherType'] = [subpix_dither_type_direct] * 2
            out_of_field_dict['CoordinatedParallel'] = ['false'] * 2
            out_of_field_dict['Instrument'] = [instrument] * 2
            out_of_field_dict['ParallelInstrument'] = [False] * 2
            out_of_field_dict['ShortFilter'] = [direct_short_filter] * 2
            out_of_field_dict['LongFilter'] = [direct_long_filter] * 2
            out_of_field_dict['ReadoutPattern'] = [direct_readpatt] * 2
            out_of_field_dict['Groups'] = [direct_groups] * 2
            out_of_field_dict['Integrations'] = [direct_integrations] * 2
            out_of_field_dict['ShortPupil'] = [direct_short_pupil] * 2
            out_of_field_dict['LongPupil'] = [direct_long_pupil] * 2
            out_of_field_dict['Grism'] = [direct_grism] * 2
            out_of_field_dict['ObservationID'] = [proposal_param_dict['ObservationID']] * 2
            out_of_field_dict['TileNumber'] = ['1'] * 2
            out_of_field_dict['APTTemplate'] = [template_name] * 2
            out_of_field_dict['ObservationName'] = [proposal_param_dict['ObservationName']] * 2
            out_of_field_dict['number_of_dithers'] = [direct_number_of_dithers] * 2
            out_of_field_dict['FilterWheel'] = ['none'] * 2  # used for NIRISS
            out_of_field_dict['PupilWheel'] = ['none'] * 2  # used for NIRISS
            out_of_field_dict['FiducialPointOverride'] = [FiducialPointOverride] * 2
            out_of_field_dict['Tracking'] = [tracking] * 2

            # Add out_of_field_dict to the exposures_dictionary
            exposures_dictionary = self.append_to_exposures_dictionary(exposures_dictionary,
                                                                       out_of_field_dict,
                                                                       proposal_param_dict)

        # Make sure all entries are lists
        for key in exposures_dictionary.keys():
            if type(exposures_dictionary[key]) is not list:
                exposures_dictionary[key] = list(exposures_dictionary[key])

        # Make sure all list items in the returned dictionary have the same length
        for key, item in exposures_dictionary.items():
            if len(item) == 0:
                exposures_dictionary[key] = [0] * len(exposures_dictionary['Instrument'])

        return exposures_dictionary


    def read_nircam_coronagraphy_template(self, template, template_name, obs, proposal_param_dict, parallel=False,
                                 verbose=False):
        """Parse a NIRCam coronagraphy observation template from an APT xml file. Produce an exposure dictionary
        that lists all exposures (excluding dithers) from the template.

        Parameters
        ----------
        template : lxml.etree._Element
            Template section from APT xml

        template_name : str
            The type of template (e.g. 'NirissAmi')

        obs : lxml.etree._Element
            Observation section from APT xml

        proposal_param_dict : dict
            Dictionary of proposal level information from the xml file
            (e.g. PI, Science Category, etc)

        parallel : bool
            If True, template should be for parallel observations. If False, NIRISS WFSS
            observation is assumed to be prime

        Returns
        -------
        exposures_dictionary : dict
            Dictionary containing details on all exposures contained within the template. These details
            include things like filter, pupil, readout pattern, subarray, etc

        exp_len : int
            Dictionary length to use when comparing to that from a parallel observation. This is not
            necessarily the same as the true length of the dictionary due to the way in which APT
            groups overvations
        """
        instrument = 'NIRCam'

        if verbose:
            print(f"Reading template {template_name}")

        # Dictionary that holds the content of this observation only
        exposures_dictionary = copy.deepcopy(self.empty_exposures_dictionary)

        # Set namespace
        ncc = "{http://www.stsci.edu/JWST/APT/Template/NircamCoron}"

        mod = 'A' # by policy, always mod A for coronagraphy

        if verbose:
            self.logger.info("Reading NIRCam Coronagraphy template")

        parallel_instrument = False

        dither_key_name = 'DitherPattern'


        # Check the target type in order to decide whether the tracking should be
        # sidereal or non-sidereal
        tracking = self.get_tracking_type(obs)

        # Determine if there is an aperture override
        override = obs.find('.//' + self.apt + 'FiducialPointOverride')
        FiducialPointOverride = True if override is not None else False

        subarray = template.find(ncc + 'Subarray').text

        # Find the number of dithers
        # We treat the coronagraphy dithers as subpixel dithers, with 1 primary dither.
        primary_dithers_pattern = 'NONE'
        number_of_primary_dithers = 1
        subpix_dithers_pattern = template.find(ncc + 'DitherPattern').text
        if subpix_dithers_pattern.upper() != 'NONE':
            # first character of the dither pattern name is the number of points in it, so:
            number_of_subpixel_dithers = int(subpix_dithers_pattern[0])
            subpix_dither_type = 'SMALL-GRID-DITHER'
        else:
            number_of_subpixel_dithers = 1
            subpix_dither_type = NONE_STR
        number_of_astrometric_dithers = 0 # it's optional, and off by default

        coronmask = template.find(ncc + 'CoronMask').text

        # APT outputs the incorrect aperture name. It specifies
        # SUB320 for both MASK430R and MASKLWB cases. Fix that here.
        if subarray != 'FULL':
            if coronmask == 'MASK430R':
                subarray = 'SUB320{}430R'.format(mod)
            elif coronmask == 'MASKLWB':
                subarray = 'SUB320{}LWB'.format(mod)
            elif coronmask == 'MASK335R':
                subarray = 'SUB320{}335R'.format(mod)
            elif coronmask == 'MASKSWB':
                subarray = 'SUB640{}SWB'.format(mod)
            elif coronmask == 'MASK210R':
                subarray = 'SUB640{}210R'.format(mod)

        coron_sci_detector = 'A2' if coronmask=='MASK210R' else \
                       'A4' if coronmask=='MASKSWB' else 'A5'
        using_lw = coron_sci_detector =='A5'


        if using_lw: # Long wave channel is being used
            long_pupil = 'WEDGELYOT' if coronmask=='MASKLWB' else 'CIRCLYOT'
            short_pupil = 'n/a'    # not even read out, no data downloaded
            filter_key_name = 'LongFilter'
        else: # Short wave channel
            short_pupil = 'WEDGELYOT' if coronmask == 'MASKSWB' else 'CIRCLYOT'
            long_pupil = 'n/a'
            filter_key_name = 'ShortFilter'

        # Get information about any TA exposures

        ta_targ = template.find(ncc + 'AcqTargetID').text
        if ta_targ.upper() != 'NONE':
            ta_readout = template.find(ncc + 'AcqReadoutPattern').text
            ta_groups = template.find(ncc + 'AcqGroups').text
            ta_filter = template.find(ncc + 'AcqFilter').text
            ta_brightness = template.find(ncc + 'AcqTargetBrightness').text
            ta_dithers_pattern = 'NONE'

            # TA uses subarrays
            #   from NIRCam_subarray_definitions.list we have:
            # NRCA2_FSTAMASK210R              SUBFSA210R     1
            # NRCA5_FSTAMASK335R              SUBFSA335R     1
            # NRCA5_FSTAMASK430R              SUBFSA430R     1
            # NRCA5_FSTAMASKLWB               SUBFSALWB      1
            # NRCA4_FSTAMASKSWB               SUBFSASWB      1
            # NRCA2_TAMASK210R                SUBNDA210R     1
            # NRCA5_TAMASK335R                SUBNDA335R     1
            # NRCA5_TAMASK430R                SUBNDA430R     1
            # NRCA5_TAMASKLWBL                SUBNDALWBL     1
            # NRCA5_TAMASKLWB                 SUBNDALWBS     1
            # NRCA4_TAMASKSWB                 SUBNDASWBL     1
            # NRCA4_TAMASKSWBS                SUBNDASWBS     1
            ta_subarray = 'SUB' + ('NDA' if ta_brightness=='BRIGHT' else 'FSA') + coronmask[4:]

            # For the bar occulters, which have L and S (long and short) side TA subarrays,
            # it turns out that which aperture gets used for TA depends on the science filter in a non-obvious way
            # See table 7-7 of NIRCam OSS docs, as provided by Stansberry to Perrin
            # For MIRAGE at this point we haven't yet parsed the first science filter, and for now it's not critical
            # enough to get this right, so let's just pick reasonable defaults.
            # TODO improve later, if we decide it's important
            if ta_subarray =='SUBNDASWB':
                # ta_side should be S for F182M, F187M,F210M otherwise it's L.  Default for now can be L for F200W
                ta_aperture_side = 'L'
                ta_subarray += ta_aperture_side
            elif ta_subarray == 'SUBNDALWB':
                # ta_side should be L for F460M,F480M otherwise it's S.  Default for now can be S
                ta_aperture_side = 'S'
                ta_subarray += ta_aperture_side

            self.logger.info(f"Read TA exposure parameters {ta_readout}, {ta_groups}; inferred subarray= {ta_subarray}")

        # Do not look for the text attribute yet since this keyword may not be present
        astrometric_confirmation_imaging = template.find(ncc + 'OptionalConfirmationImage').text

        if astrometric_confirmation_imaging.upper() == 'TRUE':
            number_of_astrometric_dithers = 2 # at the initial position for TA, and after move to the occulter
            astrom_readout_pattern = template.find(ncc + 'ConfirmationReadoutPattern').text
            astrom_readout_groups = template.find(ncc + 'ConfirmationGroups').text
            astrom_readout_ints = template.find(ncc + 'ConfirmationIntegrations').text
            astrom_readout_detector = 'FULL' # f'NRC{sci_detector}_FULL'

        # Combine primary and subpixel dithers
        number_of_dithers = str(number_of_primary_dithers * number_of_subpixel_dithers)

        ta_dict = {}
        ta_exposures = copy.deepcopy(self.empty_exposures_dictionary)
        if ta_targ.upper() != 'NONE':
            ta_dict['ReadoutPattern'] = ta_readout
            ta_dict['Groups'] = ta_groups
            ta_dict['Integrations'] = 1
            ta_dict[filter_key_name] = ta_filter
            ta_dict['TABrightness'] = ta_brightness
            ta_dict[dither_key_name] = ta_dithers_pattern
            ta_dict['number_of_dithers'] = 1  # TA is never dithered
            ta_dict['ShortPupil'] = short_pupil
            ta_dict['LongPupil'] = long_pupil
            ta_dict['CoronMask'] = coronmask

            for key in self.APTObservationParams_keys:
                if key in ta_dict.keys():
                    value = ta_dict[key]
                elif key in proposal_param_dict.keys():
                    value = proposal_param_dict[key]
                elif key == 'Instrument':
                    value = instrument
                elif key == 'ParallelInstrument':
                    value = parallel_instrument
                elif key == 'FiducialPointOverride':
                    value = str(FiducialPointOverride)
                elif key == 'APTTemplate':
                    value = template_name
                elif key == 'Tracking':
                    value = tracking
                elif key == 'Mode':
                    # value = 'imaging'
                    value = 'coron'
                elif key == 'Module':
                    value = mod
                elif key == 'Subarray':
                    value = ta_subarray
                elif key == 'PrimaryDithers':
                    value = ta_dithers_pattern
                else:
                    value = NONE_STR
                ta_exposures[key].append(value)

        # Setup astrometric exposures, if present
        # If direct images are requested, we need to add a separate
        # entry in the exposure dictionary for them.
        if astrometric_confirmation_imaging.upper() == 'TRUE':
            astrometric_exp_dict = {}
            astrometric_exposures = copy.deepcopy(self.empty_exposures_dictionary)

            astrometric_exp_dict[dither_key_name] = int(number_of_astrometric_dithers)
            astrometric_exp_dict['number_of_dithers'] = astrometric_exp_dict[dither_key_name]
            astrometric_exp_dict[filter_key_name] = ta_filter
            astrometric_exp_dict['ReadoutPattern'] = astrom_readout_pattern
            astrometric_exp_dict['Groups'] = astrom_readout_groups
            astrometric_exp_dict['Integrations'] = astrom_readout_ints
            astrometric_exp_dict['ImageDithers'] = number_of_astrometric_dithers
            astrometric_exp_dict['EtcId'] = '-1'
            astrometric_exp_dict['ShortPupil'] = short_pupil
            astrometric_exp_dict['LongPupil'] = long_pupil
            astrometric_exp_dict['CoronMask'] = coronmask

            for key in self.APTObservationParams_keys:
                if key in astrometric_exp_dict.keys():
                    dir_value = astrometric_exp_dict[key]
                elif key in proposal_param_dict.keys():
                    dir_value = proposal_param_dict[key]
                elif key == 'Instrument':
                    dir_value = instrument
                elif key == 'ParallelInstrument':
                    dir_value = parallel_instrument
                elif key == 'FiducialPointOverride':
                    dir_value = str(FiducialPointOverride)
                elif key == 'APTTemplate':
                    dir_value = template_name
                elif key == 'Tracking':
                    dir_value = tracking
                elif (key == 'Mode'):
                    # dir_value = 'imaging'
                    dir_value = 'coron'
                elif key == 'Module':
                    dir_value = mod
                elif key == 'Subarray':
                    dir_value = astrom_readout_detector
                elif key == 'PrimaryDithers':
                    dir_value = number_of_astrometric_dithers
                else:
                    dir_value = NONE_STR
                astrometric_exposures[key].append(dir_value)

        # Now that we have the correct number of dithers, we can
        # begin populating the exposure dictionary
        science_exposures = copy.deepcopy(self.empty_exposures_dictionary)
        for element in template.find(ncc+'Filters'):
            element_tag_stripped = element.tag.split(ncc)[1]
            # Get exposure information
            if element_tag_stripped == 'FilterConfig':
                exposure_dict = {}

                # Load dither information into dictionary
                exposure_dict[dither_key_name] = int(number_of_dithers)
                exposure_dict['number_of_dithers'] = exposure_dict[dither_key_name]
                exposure_dict['SubpixelDitherType'] = subpix_dither_type
                exposure_dict['ReadoutPattern'] = element.find(ncc + 'ReadoutPattern').text
                exposure_dict['Groups'] = element.find(ncc + 'Groups').text
                exposure_dict['Integrations'] = element.find(ncc + 'Integrations').text

                exposure_dict[filter_key_name] = element.find(ncc + 'Filter').text
                exposure_dict['ShortPupil'] = short_pupil
                exposure_dict['LongPupil'] = long_pupil
                exposure_dict['CoronMask'] = coronmask

                # print(exposure_dict )

                # Filter, ReadoutPattern, Groups, Integrations,
                # set subarray also

                # Store all entries in exposure_dict as lists, so that everything
                # is consistent regardless of whether there is a direct image
                #for exposure_parameter in exposure:
                #    parameter_tag_stripped = exposure_parameter.tag.split(ncc)[1]
                #    exposure_dict[parameter_tag_stripped] = exposure_parameter.text

                # Fill dictionary to return
                for key in self.APTObservationParams_keys:
                    if key in exposure_dict.keys():
                        value = exposure_dict[key]
                    elif key in proposal_param_dict.keys():
                        value = proposal_param_dict[key]
                    elif key == 'Instrument':
                        value = instrument
                    elif key == 'ParallelInstrument':
                        value = parallel_instrument
                    elif key == 'FiducialPointOverride':
                        value = str(FiducialPointOverride)
                    elif key == 'APTTemplate':
                        value = template_name
                    elif key == 'Tracking':
                        value = tracking
                    elif (key == 'Mode'):
                        value = 'coron'
                    elif key == 'Module':
                        value = mod
                    elif key == 'Subarray':
                        value = subarray
                    # elif key == 'PrimaryDithers':
                    #     value = primary_dithers_pattern
                    # elif key == 'SubpixelPositions':
                    #     value = subpix_dithers_pattern
                    elif key == 'PrimaryDithers':
                        value = number_of_primary_dithers
                    elif key == 'SubpixelPositions':
                        value = number_of_subpixel_dithers
                    elif key == 'PrimaryDitherType':
                        value = primary_dithers_pattern
                    elif key == 'SubpixelDitherType':
                        value = subpix_dithers_pattern
                    elif key == 'SmallGridDitherType':
                        value = subpix_dithers_pattern
                    else:
                        value = NONE_STR
                    science_exposures[key].append(value)


        # After collecting information for all exposures, we need to
        # put them in the correct order. TA exposures first, followed by
        # any astrometric confirmation images, then all science observations.
        # This is based on the order shown in the pointing file.
        if ta_targ.upper() != 'NONE':
            for key in science_exposures:
                exposures_dictionary[key] = list(ta_exposures[key])
            # print(f"Number of TA exposure specs: {len(ta_exposures[key])}")

        if astrometric_confirmation_imaging.upper() == 'TRUE':
            for key in science_exposures:
                exposures_dictionary[key] = list(exposures_dictionary[key]) + list(astrometric_exposures[key])
            # print(f"Number of astrometric exposure specs: {len(astrometric_exposures[key])}")

        for key in science_exposures:
            exposures_dictionary[key] = list(exposures_dictionary[key]) + list(science_exposures[key])
        # print(f"Number of science exposure specs: {len(science_exposures[key])}")


        self.logger.info('Number of dithers for NIRCam coron exposure: {} primary * {} subpixel = {}'.format(number_of_primary_dithers,
                                                                                                    number_of_subpixel_dithers,
                                                                                                    number_of_dithers))
        if astrometric_confirmation_imaging.upper() == 'TRUE':
            self.logger.info('Number of dithers for astrometric confirmation image: {}'.format(number_of_astrometric_dithers))

        for key in exposures_dictionary.keys():
            if type(exposures_dictionary[key]) is not list:
                exposures_dictionary[key] = list(exposures_dictionary[key])

        # Make sure all list items in the returned dictionary have the same length
        for key, item in exposures_dictionary.items():
            if len(item) == 0:
                exposures_dictionary[key] = [0] * len(exposures_dictionary['Instrument'])

        self.logger.info(f"Total number of exposures for this observation: {len(exposures_dictionary[key])}")
        return exposures_dictionary

    def read_miri_coronagraphy_template(self, template, template_name, obs, proposal_param_dict, parallel=False,
                                 verbose=False):
        """Parse a MIRI coronagraphy observation template from an APT xml file. Produce an exposure dictionary
        that lists all exposures (excluding dithers) from the template.

        Parameters
        ----------
        template : lxml.etree._Element
            Template section from APT xml

        template_name : str
            The type of template (e.g. 'NirissAmi')

        obs : lxml.etree._Element
            Observation section from APT xml

        proposal_param_dict : dict
            Dictionary of proposal level information from the xml file
            (e.g. PI, Science Category, etc)

        parallel : bool
            If True, template should be for parallel observations. If False, NIRISS WFSS
            observation is assumed to be prime

        Returns
        -------
        exposures_dictionary : dict
            Dictionary containing details on all exposures contained within the template. These details
            include things like filter, pupil, readout pattern, subarray, etc

        exp_len : int
            Dictionary length to use when comparing to that from a parallel observation. This is not
            necessarily the same as the true length of the dictionary due to the way in which APT
            groups overvations
        """
        instrument = 'MIRI'

        if verbose:
            print(f"Reading template {template_name}")
            self.logger.info(f"Reading {template_name} template")

        # Dictionary that holds the content of this observation only
        exposures_dictionary = copy.deepcopy(self.empty_exposures_dictionary)

        # Set namespace
        mc = "{http://www.stsci.edu/JWST/APT/Template/MiriCoron}"

        parallel_instrument = False

        dither_key_name = 'DitherPattern'

        # Check the target type in order to decide whether the tracking should be
        # sidereal or non-sidereal
        tracking = self.get_tracking_type(obs)

        # Determine if there is an aperture override
        override = obs.find('.//' + self.apt + 'FiducialPointOverride')
        FiducialPointOverride = True if override is not None else False

        # Find the number of dithers
        # We treat the coronagraphy dithers as subpixel dithers, with 1 primary dither.
        primary_dithers_pattern = 'NONE'
        number_of_primary_dithers = 1
        subpix_dithers_pattern = template.find(mc + 'Dither').text
        if subpix_dithers_pattern.upper() != 'NONE':
            # first character of the dither pattern name is the number of points in it, so:
            number_of_subpixel_dithers = int(subpix_dithers_pattern[0])
            subpix_dither_type = 'SMALL-GRID-DITHER'
        else:
            number_of_subpixel_dithers = 1
            subpix_dither_type = NONE_STR

        # Get information about any TA exposures

        ta_targ = template.find(mc + 'AcqTargetID').text
        if ta_targ.upper() != 'NONE':
            ta_readout = template.find(mc + 'AcqReadoutPattern').text
            ta_groups = template.find(mc + 'AcqGroups').text
            ta_filter = template.find(mc + 'AcqFilter').text
            ta_dithers_pattern = 'NONE'

            ta_subarray = 'MIRISUBARRAYTBD'

            self.logger.info(f"Read TA exposure parameters {ta_readout}, {ta_groups}; inferred subarray= {ta_subarray}")

        # Combine primary and subpixel dithers
        number_of_dithers = str(number_of_primary_dithers * number_of_subpixel_dithers)

        ta_dict = {}
        ta_exposures = copy.deepcopy(self.empty_exposures_dictionary)
        if ta_targ.upper() != 'NONE':
            ta_dict['ReadoutPattern'] = ta_readout
            ta_dict['Groups'] = ta_groups
            ta_dict['Integrations'] = 1
            ta_dict['Filter'] = ta_filter
            ta_dict[dither_key_name] = ta_dithers_pattern
            ta_dict['number_of_dithers'] = 1  # TA is never dithered
            ta_dict['Pupil'] = 'TBD'

            for key in self.APTObservationParams_keys:
                if key in ta_dict.keys():
                    value = ta_dict[key]
                elif key in proposal_param_dict.keys():
                    value = proposal_param_dict[key]
                elif key == 'Instrument':
                    value = instrument
                elif key == 'ParallelInstrument':
                    value = parallel_instrument
                elif key == 'FiducialPointOverride':
                    value = str(FiducialPointOverride)
                elif key == 'APTTemplate':
                    value = template_name
                elif key == 'Tracking':
                    value = tracking
                elif key == 'Mode':
                    value = 'imaging'
                elif key == 'Subarray':
                    value = ta_subarray
                elif key == 'PrimaryDithers':
                    value = ta_dithers_pattern
                else:
                    value = NONE_STR
                ta_exposures[key].append(value)


        # Now that we have the correct number of dithers, we can
        # begin populating the exposure dictionary
        science_exposures = copy.deepcopy(self.empty_exposures_dictionary)
        for element in template.find(mc+'Filters'):
            element_tag_stripped = element.tag.split(mc)[1]
            # Get exposure information
            if element_tag_stripped == 'FilterConfig':
                exposure_dict = {}

                # Load dither information into dictionary
                exposure_dict[dither_key_name] = int(number_of_dithers)
                exposure_dict['number_of_dithers'] = exposure_dict[dither_key_name]
                exposure_dict['SubpixelDitherType'] = subpix_dither_type
                exposure_dict['ReadoutPattern'] = element.find(mc + 'ReadoutPattern').text
                exposure_dict['Groups'] = element.find(mc + 'Groups').text
                exposure_dict['Integrations'] = element.find(mc + 'Integrations').text

                exposure_dict['Filter'] = element.find(mc + 'Filter').text
                coron_type =  element.find(mc + 'Mask').text

                if coron_type=='LYOT':
                    coron_mask='MASKLYOT'
                    coronagraph='LYOT_2300'
                else:
                    coron_mask='MASK' + exposure_dict['Filter'][1:5]
                    coronagraph = '4QPM_' + exposure_dict['Filter'][1:5]
                exposure_dict['Pupil'] = coron_mask
                exposure_dict['Subarray'] = coron_mask
                exposure_dict['CoronMask'] = coronagraph
                # print(exposure_dict )

                # Filter, ReadoutPattern, Groups, Integrations,

                # Store all entries in exposure_dict as lists, so that everything
                # is consistent regardless of whether there is a direct image
                #for exposure_parameter in exposure:
                #    parameter_tag_stripped = exposure_parameter.tag.split(ncc)[1]
                #    exposure_dict[parameter_tag_stripped] = exposure_parameter.text

                # Fill dictionary to return
                for key in self.APTObservationParams_keys:
                    if key in exposure_dict.keys():
                        value = exposure_dict[key]
                    elif key in proposal_param_dict.keys():
                        value = proposal_param_dict[key]
                    elif key == 'Instrument':
                        value = instrument
                    elif key == 'ParallelInstrument':
                        value = parallel_instrument
                    elif key == 'FiducialPointOverride':
                        value = str(FiducialPointOverride)
                    elif key == 'APTTemplate':
                        value = template_name
                    elif key == 'Tracking':
                        value = tracking
                    elif (key == 'Mode'):
                        value = 'coron'
                    elif key == 'PrimaryDithers':
                        value = primary_dithers_pattern
                    elif key == 'SubpixelPositions':
                        value = subpix_dithers_pattern
                    else:
                        value = NONE_STR
                    science_exposures[key].append(value)


        # After collecting information for all exposures, we need to
        # put them in the correct order. TA exposures first, followed by
        # any astrometric confirmation images, then all science observations.
        # This is based on the order shown in the pointing file.

        # HACK: For MIRI don't make rows for the TA exposures.
        # This is because the pointing file parsing ignores them, so we have to be consistent (for now)
        include_MIRI_TAs = False
        if ta_targ.upper() != 'NONE' and include_MIRI_TAs:
            for key in science_exposures:
                exposures_dictionary[key] = list(ta_exposures[key])
            # print(f"Number of TA exposure specs: {len(ta_exposures[key])}")

        for key in science_exposures:
            exposures_dictionary[key] = list(exposures_dictionary[key]) + list(science_exposures[key])
        # print(f"Number of science exposure specs: {len(science_exposures[key])}")


        self.logger.info('Number of dithers for MIRI coron exposure: {} primary * {} subpixel = {}'.format(number_of_primary_dithers,
                                                                                                    number_of_subpixel_dithers,
                                                                                                    number_of_dithers))

        for key in exposures_dictionary.keys():
            if type(exposures_dictionary[key]) is not list:
                exposures_dictionary[key] = list(exposures_dictionary[key])

        # Make sure all list items in the returned dictionary have the same length
        for key, item in exposures_dictionary.items():
            if len(item) == 0:
                exposures_dictionary[key] = [0] * len(exposures_dictionary['Instrument'])

        self.logger.info(f"Total number of exposures for this observation: {len(exposures_dictionary[key])}")
        return exposures_dictionary


    def read_parallel_exposures(self, obs, exposures_dictionary, proposal_parameter_dictionary, verbose=False):
        """Read the exposures of the parallel instrument.

        Parameters
        ----------
        obs : APT xml element
            Observation section of xml file
        exposures_dictionary : dict
            Exposures of the prime instrument
        proposal_parameter_dictionary : dict
            Parameters to extract
        verbose : bool
            Verbosity

        Returns
        -------
        parallel_exposures_dictionary : dict
            Parallel exposures.

        """
        # Determine what template is used for the parallel observation
        template = obs.find(self.apt + 'FirstCoordinatedTemplate')[0]
        template_name = etree.QName(template).localname
        if template_name in ['NircamImaging', 'NircamEngineeringImaging', 'NirissExternalCalibration',
                             'NirspecImaging', 'MiriMRS', 'FgsExternalCalibration']:
            parallel_exposures_dictionary = self.read_generic_imaging_template(template,
                                                                               template_name, obs,
                                                                               proposal_parameter_dictionary,
                                                                               parallel=True,
                                                                               verbose=verbose)
        elif template_name == 'NirissWfss':
            parallel_exposures_dictionary = self.read_niriss_wfss_template(template, template_name, obs,
                                                                           proposal_parameter_dictionary,
                                                                           parallel=True)
        else:
            raise ValueError('Parallel observation template {} not supported.'.format(template_name))

        # Find length of the exposures dictionary to compare
        parallel_length = len(parallel_exposures_dictionary['number_of_dithers'])
        exposures_dictionary_length = len(exposures_dictionary['number_of_dithers'])

        if parallel_length != exposures_dictionary_length:
            raise RuntimeError('Mismatch in the number of parallel observations.')

        return parallel_exposures_dictionary

    def append_to_exposures_dictionary(self, exp_dictionary, exposure_seq_dict, prop_param_dict):
        """Append exposure(s) information from a dictionary to an existing exposures dictionary

        Parameters
        ----------
        exp_dictionary : dict
            Dictionary containing information on multiple exposures

        exposure_seq_dict : dict
            Dictionary containing information on a single exposure. This dictionary should have
            the same keys as exp_dictionary. The contents of this dictionary will be added to
            exp_dictionary

        prop_param_dict : dict
            A dictionary containing proposal-wide information, such as title and PI name

        Reutrns
        -------
        exp_dictionary : dict
            With the new exposure(s) added
        """
        keys = list(exposure_seq_dict.keys())
        number_of_exposures = len(exposure_seq_dict[keys[0]])

        for key in self.APTObservationParams_keys:
            if key in exposure_seq_dict.keys():
                value = exposure_seq_dict[key]
            elif key in prop_param_dict.keys():
                value = [prop_param_dict[key]] * number_of_exposures
            else:
                value = [NONE_STR] * number_of_exposures

            if (key in ['PrimaryDithers', 'ImageDithers']) and ((value is None) or (value == 'None')):
                value = ['1'] * number_of_exposures
            exp_dictionary[key].extend(value)

        # add keys that were not defined in self.APTObservationParams_keys
        for key in exposure_seq_dict.keys():
            if key not in self.APTObservationParams_keys:
                # if key not yet present, create entry
                if key not in exp_dictionary.keys():
                    self.logger.info('Key {} not present in APTObservationParams nor exposures_dictionary'.format(key))
                    exp_dictionary[key] = [str(exposure_seq_dict[key])]
                else:
                    self.logger.info('Key {} not present in APTObservationParams'.format(key))
                    exp_dictionary[key].append(str(exposure_seq_dict[key]))
        return exp_dictionary





    def read_niriss_ami_template(self, template, template_name, obs, proposal_param_dict, parallel=False,
                                 verbose=False):
        """Parse a NIRISS AMI observation template from an APT xml file. Produce an exposure dictionary
        that lists all exposures (excluding dithers) from the template.

        Parameters
        ----------
        template : lxml.etree._Element
            Template section from APT xml

        template_name : str
            The type of template (e.g. 'NirissAmi')

        obs : lxml.etree._Element
            Observation section from APT xml

        proposal_param_dict : dict
            Dictionary of proposal level information from the xml file
            (e.g. PI, Science Category, etc)

        parallel : bool
            If True, template should be for parallel observations. If False, NIRISS WFSS
            observation is assumed to be prime

        Returns
        -------
        exposures_dictionary : dict
            Dictionary containing details on all exposures contained within the template. These details
            include things like filter, pupil, readout pattern, subarray, etc

        exp_len : int
            Dictionary length to use when comparing to that from a parallel observation. This is not
            necessarily the same as the true length of the dictionary due to the way in which APT
            groups overvations
        """
        instrument = 'NIRISS'

        # Dummy module name for NIRISS. Needed for consistency in dictionary entry
        mod = 'N'
        long_filter = 'N/A'
        long_pupil = 'N/A'

        # Dictionary that holds the content of this observation only
        exposures_dictionary = copy.deepcopy(self.empty_exposures_dictionary)

        # Set namespace
        ns = "{http://www.stsci.edu/JWST/APT/Template/NirissAmi}"

        if verbose:
            self.logger.info("Reading NIRISS AMI template")

        parallel_instrument = False

        DitherPatternType = None
        dither_key_name = 'ImageDithers'

        # number of dithers defaults to 1
        number_of_dithers = 1
        number_of_subpixel_positions = 1

        number_of_primary_dithers = 1
        number_of_subpixel_dithers = 1
        number_of_direct_dithers = 0

        # Check the target type in order to decide whether the tracking should be
        # sidereal or non-sidereal
        tracking = self.get_tracking_type(obs)

        # Determine if there is an aperture override
        override = obs.find('.//' + self.apt + 'FiducialPointOverride')
        FiducialPointOverride = True if override is not None else False

        subarray = template.find(ns + 'Subarray').text
        ta_subarray = 'SUBTAAMI'

        # Find the number of primary, subpixel, and optionally, direct image dithers
        primary_dithers = template.find(ns + 'PrimaryDithers').text
        subpix_dithers = template.find(ns + 'SubpixelPositions').text

        # Get information about any TA exposures
        ta_targ = template.find(ns + 'AcqTarget').text
        if ta_targ.upper() != 'NONE':
            ta_readout = template.find(ns + 'AcqReadoutPattern').text
            ta_groups = template.find(ns + 'AcqGroups').text
            ta_dithers = 4

        # Do not look for the text attribute yet since this keyword may not be present
        direct_imaging = template.find(ns + 'DirectImaging').text

        if primary_dithers.upper() != 'NONE':
            number_of_primary_dithers = int(primary_dithers)
        if subpix_dithers.upper() != 'NONE':
            number_of_subpixel_dithers = int(subpix_dithers)
        if direct_imaging.upper() == 'TRUE':
            image_dithers = template.find(ns + 'ImageDithers').text
            if image_dithers.upper() != 'NONE':
                number_of_direct_dithers = int(image_dithers)
            else:
                number_of_direct_dithers = 1

        # Combine primary and subpixel dithers
        number_of_dithers = str(number_of_primary_dithers * number_of_subpixel_dithers)

        ta_dict = {}
        ta_exposures = copy.deepcopy(self.empty_exposures_dictionary)
        if ta_targ.upper() != 'NONE':
            ta_dict['ReadoutPattern'] = ta_readout
            ta_dict['Groups'] = ta_groups
            ta_dict['Integrations'] = 1
            ta_dict['Filter'] = 'F480M'
            ta_dict[dither_key_name] = ta_dithers
            ta_dict['number_of_dithers'] = ta_dict[dither_key_name]

            for key in self.APTObservationParams_keys:
                if key in ta_dict.keys():
                    value = ta_dict[key]
                elif key in proposal_param_dict.keys():
                    value = proposal_param_dict[key]
                elif key == 'Instrument':
                    value = instrument
                elif key == 'ParallelInstrument':
                    value = parallel_instrument
                elif key == 'FiducialPointOverride':
                    value = str(FiducialPointOverride)
                elif key == 'APTTemplate':
                    value = template_name
                elif key == 'Tracking':
                    value = tracking
                elif key == 'Mode':
                    value = 'imaging'
                elif key == 'Module':
                    value = mod
                elif key == 'Subarray':
                    value = ta_subarray
                elif key == 'PrimaryDithers':
                    value = ta_dithers
                else:
                    value = NONE_STR
                ta_exposures[key].append(value)

        # Now that we have the correct number of dithers, we can
        # begin populating the exposure dictionary
        for element in template:
            element_tag_stripped = element.tag.split(ns)[1]

            # Get exposure information
            if element_tag_stripped == 'ExposureList':
                science_exposures = copy.deepcopy(self.empty_exposures_dictionary)
                direct_exposures = copy.deepcopy(self.empty_exposures_dictionary)

                for exposure in element.findall(ns + 'Exposure'):
                    exposure_dict = {}
                    direct_dict = {}

                    # Load dither information into dictionary
                    exposure_dict[dither_key_name] = int(number_of_dithers)
                    exposure_dict['number_of_dithers'] = exposure_dict[dither_key_name]

                    if direct_imaging.upper() == 'TRUE':
                        direct_dict[dither_key_name] = int(number_of_direct_dithers)
                        direct_dict['number_of_dithers'] = direct_dict[dither_key_name]

                    # Store all entries in exposure_dict as lists, so that everything
                    # is consistent regardless of whether there is a direct image
                    for exposure_parameter in exposure:
                        parameter_tag_stripped = exposure_parameter.tag.split(ns)[1]
                        exposure_dict[parameter_tag_stripped] = exposure_parameter.text

                    # If direct images are requested, we need to add a separate
                    # entry in the exposure dictionary for them.
                    if direct_imaging.upper() == 'TRUE':
                        direct_dict['Filter'] = exposure_dict['Filter']
                        direct_dict['ReadoutPattern'] = exposure_dict['DirectReadoutPattern']
                        direct_dict['Groups'] = exposure_dict['DirectGroups']
                        direct_dict['Integrations'] = exposure_dict['DirectIntegrations']
                        direct_dict['ImageDithers'] = image_dithers
                        direct_dict['EtcId'] = exposure_dict['DirectEtcId']

                    # Fill dictionary to return
                    for key in self.APTObservationParams_keys:
                        if key in exposure_dict.keys():
                            value = exposure_dict[key]
                        elif key in proposal_param_dict.keys():
                            value = proposal_param_dict[key]
                        elif key == 'Instrument':
                            value = instrument
                        elif key == 'ParallelInstrument':
                            value = parallel_instrument
                        elif key == 'FiducialPointOverride':
                            value = str(FiducialPointOverride)
                        elif key == 'APTTemplate':
                            value = template_name
                        elif key == 'Tracking':
                            value = tracking
                        elif (key == 'Mode'):
                            value = 'ami'
                        elif key == 'Module':
                            value = mod
                        elif key == 'Subarray':
                            value = subarray
                        elif key == 'PrimaryDithers':
                            value = primary_dithers
                        elif key == 'SubpixelPositions':
                            value = subpix_dithers
                        else:
                            value = NONE_STR
                        science_exposures[key].append(value)

                        if direct_imaging.upper() == 'TRUE':
                            if key in direct_dict.keys():
                                dir_value = direct_dict[key]
                            elif key in proposal_param_dict.keys():
                                dir_value = proposal_param_dict[key]
                            elif key == 'Instrument':
                                dir_value = instrument
                            elif key == 'ParallelInstrument':
                                dir_value = parallel_instrument
                            elif key == 'FiducialPointOverride':
                                dir_value = str(FiducialPointOverride)
                            elif key == 'APTTemplate':
                                dir_value = template_name
                            elif key == 'Tracking':
                                dir_value = tracking
                            elif (key == 'Mode'):
                                dir_value = 'imaging'
                            elif key == 'Module':
                                dir_value = mod
                            elif key == 'Subarray':
                                dir_value = subarray
                            elif key == 'PrimaryDithers':
                                dir_value = number_of_direct_dithers
                            else:
                                dir_value = NONE_STR
                            direct_exposures[key].append(dir_value)

                # After collecting information for all exposures, we need to
                # put them in the correct order. TA exposures first, followed by
                # all science observations, and then any direct images.
                # This is based on the order shown in the pointing file.
                if ta_targ.upper() != 'NONE':
                    for key in science_exposures:
                        exposures_dictionary[key] = list(ta_exposures[key]) + list(science_exposures[key])
                else:
                    for key in science_exposures:
                        exposures_dictionary[key] = list(science_exposures[key])

                if direct_imaging.upper() == 'TRUE':
                    for key in science_exposures:
                        exposures_dictionary[key] = list(exposures_dictionary[key]) + list(direct_exposures[key])

        self.logger.info('Number of dithers for AMI exposure: {} primary * {} subpixel = {}'.format(number_of_primary_dithers,
                                                                                                    number_of_subpixel_dithers,
                                                                                                    number_of_dithers))
        if direct_imaging.upper() == 'TRUE':
            self.logger.info('Number of dithers for direct image: {}'.format(number_of_direct_dithers))

        for key in exposures_dictionary.keys():
            if type(exposures_dictionary[key]) is not list:
                exposures_dictionary[key] = list(exposures_dictionary[key])

        # Make sure all list items in the returned dictionary have the same length
        for key, item in exposures_dictionary.items():
            if len(item) == 0:
                exposures_dictionary[key] = [0] * len(exposures_dictionary['Instrument'])

        return exposures_dictionary


    def read_niriss_wfss_template(self, template, template_name, obs, proposal_param_dict, parallel=False,
                                  verbose=False):
        """Parse a NIRISS WFSS observation template from an APT xml file. Produce an exposure dictionary
        that lists all exposures (excluding dithers) from the template.

        Parameters
        ----------
        template : lxml.etree._Element
            Template section from APT xml

        template_name : str
            The type of template (e.g. 'NirissWfss')

        obs : lxml.etree._Element
            Observation section from APT xml

        proposal_param_dict : dict
            Dictionary of proposal level information from the xml file
            (e.g. PI, Science Category, etc)

        parallel : bool
            If True, template should be for parallel observations. If False, NIRISS WFSS
            observation is assumed to be prime

        Returns
        -------
        exposures_dictionary : dict
            Dictionary containing details on all exposures contained within the template. These details
            include things like filter, pupil, readout pattern, subarray, etc

        exp_len : int
            Dictionary length to use when comparing to that from a parallel observation. This is not
            necessarily the same as the true length of the dictionary due to the way in which APT
            groups overvations
        """
        instrument = 'NIRISS'

        # Dummy module name for NIRISS. Needed for consistency in dictionary entry
        mod = 'N'
        subarr = 'FULL'
        long_filter = 'N/A'
        long_pupil = 'N/A'

        # Dictionary that holds the content of this observation only
        exposures_dictionary = copy.deepcopy(self.empty_exposures_dictionary)

        # Set namespace
        ns = "{http://www.stsci.edu/JWST/APT/Template/NirissWfss}"

        # Template from the prime instrument is needed if WFSS is parallel to a nircam imaging observation.
        # In that case we need to look into the nircam observation to see if the niriss direct images are
        # to be dithered
        if verbose:
            self.logger.info("Reading NIRISS WFSS template")
        if parallel:
            prime_template = obs.find(self.apt + 'Template')[0]
            prime_template_name = etree.QName(prime_template).localname
            prime_ns = "{{{}/Template/{}}}".format(self.apt.replace('{', '').replace('}', ''), prime_template_name)

            # Boolean indicating which instrument is not prime but parallel
            parallel_instrument = True
            prime_instrument = obs.find(self.apt + 'Instrument').text
            if verbose:
                self.logger.info('Prime: {}   Parallel: {}'.format(prime_instrument, instrument))
            pdither_grism = prime_template.find(prime_ns + 'PrimaryDithers').text
            pdither_type_grism = prime_template.find(prime_ns + 'PrimaryDitherType').text
            dither_direct = prime_template.find(prime_ns + 'DitherNirissWfssDirectImages').text
            sdither_type_grism = prime_template.find(prime_ns + 'CoordinatedParallelSubpixelPositions').text
            try:
                sdither_grism = str(int(sdither_type_grism[0]))
            except ValueError:
                sdither_grism = prime_template.find(prime_ns + 'SubpixelPositions').text
        else:
            parallel_instrument = False
            prime_instrument = instrument
            dither_direct = 'NO_DITHERING'
            sdither_type_grism = 'None'
            sdither_grism = '1'
            # Dither size can be SMALL, MEDIUM, LARGE. Only look for this if NIRISS is prime
            #pdither_type_grism = template.find(ns + 'DitherSize').text
            try:
                pdither_type_grism = template.find(ns + 'PrimaryDitherType').text
            except AttributeError:
                pdither_type_grism = 'None'

            # Can be various types
            # WFSS stand-alone observation or WFSS as parallel to NIRCam prime imaging:
            # this will be the number of primary dithers.
            # WFSS as prime, with NIRCam imaging as parallel, this will be a string.
            # (e.g. '2-POINT-LARGE-NIRCam')
            dvalue = template.find(ns + 'PrimaryDithers').text
            try:
                pdither_grism = str(int(dvalue))
            except ValueError:
                # When NIRISS is prime with NIRCam parallel, the PrimaryDithers field can be
                # (e.g. '2-POINT-LARGE-NIRCAM'), where the first character is always the number
                # of dither positions. Not sure how to save both this name as well as the DitherSize
                # value. I don't think there are header keywords for both, with PATTTYPE being the
                # only keyword for dither pattern names.
                pdither_grism = str(int(dvalue[0]))

        # Check if this observation has parallels
        coordinated_parallel = obs.find(self.apt + 'CoordinatedParallel').text

        explist = template.find(ns + 'ExposureList')
        expseqs = explist.findall(ns + 'ExposureSequences')

        # Check the target type in order to decide whether the tracking should be
        # sidereal or non-sidereal
        tracking = self.get_tracking_type(obs)

        # Determine if there is an aperture override
        override = obs.find('.//' + self.apt + 'FiducialPointOverride')
        FiducialPointOverride = True if override is not None else False

        delta_exp_dict_length = 0
        for expseq in expseqs:
            # Grism values are listed for each ExposureSequence
            grismval = expseq.find(ns + 'Sequence').text
            if grismval == 'BOTH':
                grismval = ['GR150R', 'GR150C']
                both_grisms = True
                entry_repeats = [2, 3]
            else:
                grismval = [grismval]
                both_grisms = False
                entry_repeats = [3]
            filter_name = expseq.find(ns + 'Filter').text

            # Loop over grism selections
            for grism_number, grism in enumerate(grismval):
                # sequence = direct pre, dithered grism, direct post
                # but if grism == both: direct pre1, dithered grism, direct pre2,
                # dithered grism2, direct post

                # Mini dictionary just for exposure sequence
                exp_seq_dict = {}

                # Collect info on the direct exposure
                directexp = expseq.find(ns + 'DiExposure')

                # Check to see if the user requested extra direct dithers.
                # Handle xml files from older versions of APT where ShouldDither
                # is not present
                try:
                    extra_direct_dithers = directexp.find(ns + 'ShouldDither').text
                except AttributeError:
                    extra_direct_dithers = 'false'

                typeflag = 'imaging'
                if dither_direct == 'NO_DITHERING':
                    pdither = '1'  # direct image has no dithers
                    pdither_type = 'None'  # direct image has no dithers
                    sdither = '1'
                    sdither_type = 'None'
                else:
                    pdither = pdither_grism
                    pdither_type = pdither_type_grism
                    sdither = sdither_grism
                    sdither_type = sdither_type_grism

                tile = '1'
                direct_grismvalue = 'N/A'
                #pupil = 'CLEARP'  # NIRISS filter MUST be in filter wheel, not PUPIL wheel
                pupil = 'CLEAR'
                rpatt = directexp.find(ns + 'ReadoutPattern').text
                grps = directexp.find(ns + 'Groups').text
                ints = directexp.find(ns + 'Integrations').text

                # Collect info on grism exposure
                grismexp = expseq.find(ns + 'GrismExposure')
                grism_typeflag = 'wfss'
                grism_pupil = grism
                grism_rpatt = grismexp.find(ns + 'ReadoutPattern').text
                grism_grps = grismexp.find(ns + 'Groups').text
                grism_ints = grismexp.find(ns + 'Integrations').text

                # Update values in dictionary
                repeats = entry_repeats[grism_number]

                exp_seq_dict['Module'] = ['N'] * repeats
                exp_seq_dict['Subarray'] = ['FULL'] * repeats  # Niriss WFSS is always full frame
                exp_seq_dict['CoordinatedParallel'] = [coordinated_parallel] * repeats
                exp_seq_dict['Instrument'] = [instrument] * repeats
                exp_seq_dict['ParallelInstrument'] = [parallel_instrument] * repeats
                exp_seq_dict['ShortFilter'] = [filter_name] * repeats
                exp_seq_dict['LongFilter'] = [long_filter] * repeats
                exp_seq_dict['LongPupil'] = [long_pupil] * repeats
                exp_seq_dict['ObservationID'] = [proposal_param_dict['ObservationID']] * repeats
                exp_seq_dict['TileNumber'] = [tile] * repeats
                exp_seq_dict['APTTemplate'] = [template_name] * repeats
                exp_seq_dict['ObservationName'] = [proposal_param_dict['ObservationName']] * repeats
                exp_seq_dict['PupilWheel'] = [filter_name] * repeats
                exp_seq_dict['FiducialPointOverride'] = [FiducialPointOverride] * repeats
                exp_seq_dict['Tracking'] = [tracking] * repeats

                if not both_grisms:
                    if extra_direct_dithers == 'true':
                        primary_dither_list = [pdither, pdither_grism, str(int(pdither)+2)]
                        num_of_dither_list = [str(int(pdither)*int(sdither)),
                                              str(int(pdither_grism)*int(sdither_grism)),
                                              str(int(pdither) * int(sdither) * 3)]
                    else:
                        primary_dither_list = [pdither, pdither_grism, pdither]
                        num_of_dither_list = [str(int(pdither)*int(sdither)),
                                              str(int(pdither_grism)*int(sdither_grism)),
                                              str(int(pdither)*int(sdither))]

                    exp_seq_dict['Mode'] = [typeflag, grism_typeflag, typeflag]
                    exp_seq_dict['PrimaryDitherType'] = [pdither_type, pdither_type_grism, pdither_type]
                    exp_seq_dict['PrimaryDithers'] = primary_dither_list
                    exp_seq_dict['SubpixelPositions'] = [sdither, sdither_grism, sdither]
                    exp_seq_dict['SubpixelDitherType'] = [sdither_type, sdither_type_grism, sdither_type]
                    exp_seq_dict['ReadoutPattern'] = [rpatt, grism_rpatt, rpatt]
                    exp_seq_dict['Groups'] = [grps, grism_grps, grps]
                    exp_seq_dict['Integrations'] = [ints, grism_ints, ints]
                    exp_seq_dict['ShortPupil'] = [pupil, grism_pupil, pupil]
                    exp_seq_dict['Grism'] = [direct_grismvalue, grism, direct_grismvalue]
                    exp_seq_dict['number_of_dithers'] = num_of_dither_list
                    exp_seq_dict['FilterWheel'] = [pupil, grism_pupil, pupil]
                else:
                    if grism_number == 0:
                        exp_seq_dict['Mode'] = [typeflag, grism_typeflag]
                        exp_seq_dict['PrimaryDitherType'] = [pdither_type, pdither_type_grism]
                        exp_seq_dict['PrimaryDithers'] = [pdither, pdither_grism]
                        exp_seq_dict['SubpixelPositions'] = [sdither, sdither_grism]
                        exp_seq_dict['SubpixelDitherType'] = [sdither_type, sdither_type_grism]
                        exp_seq_dict['ReadoutPattern'] = [rpatt, grism_rpatt]
                        exp_seq_dict['Groups'] = [grps, grism_grps]
                        exp_seq_dict['Integrations'] = [ints, grism_ints]
                        exp_seq_dict['ShortPupil'] = [pupil, grism_pupil]
                        exp_seq_dict['Grism'] = [direct_grismvalue, grism]
                        exp_seq_dict['number_of_dithers'] = [str(int(pdither)*int(sdither)),
                                                             str(int(pdither_grism)*int(sdither_grism))]
                        exp_seq_dict['FilterWheel'] = [pupil, grism_pupil]
                    elif grism_number == 1:
                        if extra_direct_dithers == 'true':
                            primary_dither_list = [str(int(pdither)+3), pdither_grism, str(int(pdither)+2)]
                            num_of_dither_list = [str((int(pdither)*int(sdither))*4),
                                                  str(int(pdither_grism)*int(sdither_grism)),
                                                  str(int(pdither)*int(sdither)*3)]
                        else:
                            primary_dither_list = [str(int(pdither)+1), pdither_grism, pdither]
                            num_of_dither_list = [str((int(pdither)*int(sdither))*2),
                                                  str(int(pdither_grism)*int(sdither_grism)),
                                                  str(int(pdither)*int(sdither))]

                        exp_seq_dict['Mode'] = [typeflag, grism_typeflag, typeflag]
                        exp_seq_dict['PrimaryDitherType'] = [pdither_type, pdither_type_grism, pdither_type]
                        exp_seq_dict['PrimaryDithers'] = primary_dither_list
                        exp_seq_dict['SubpixelPositions'] = [sdither, sdither_grism, sdither]
                        exp_seq_dict['SubpixelDitherType'] = [sdither_type, sdither_type_grism, sdither_type]
                        exp_seq_dict['ReadoutPattern'] = [rpatt, grism_rpatt, rpatt]
                        exp_seq_dict['Groups'] = [grps, grism_grps, grps]
                        exp_seq_dict['Integrations'] = [ints, grism_ints, ints]
                        exp_seq_dict['ShortPupil'] = [pupil, grism_pupil, pupil]
                        exp_seq_dict['Grism'] = [direct_grismvalue, grism, direct_grismvalue]
                        exp_seq_dict['number_of_dithers'] = num_of_dither_list
                        exp_seq_dict['FilterWheel'] = [pupil, grism_pupil, pupil]
                #######################################################################
                # Update exposure dictionary to return
                # Add out_of_field_dict to the exposures_dictionary
                exposures_dictionary = self.append_to_exposures_dictionary(exposures_dictionary,
                                                                           exp_seq_dict,
                                                                           proposal_param_dict)
        # Make sure all entries are lists
        for key in exposures_dictionary.keys():
            if type(exposures_dictionary[key]) is not list:
                exposures_dictionary[key] = list(exposures_dictionary[key])

        # Make sure all list items in the returned dictionary have the same length
        for key, item in exposures_dictionary.items():
            if len(item) == 0:
                exposures_dictionary[key] = [0] * len(exposures_dictionary['Instrument'])

        # Dictionary length to use when comparing to that from a parallel observation
        exp_len = len(exposures_dictionary['number_of_dithers']) + delta_exp_dict_length
        return exposures_dictionary

    def separate_pupil_and_filter(self, filter_string):
        """Filters listed for NIRCam observations can take the form 'F164N+F444W' in cases
        where filters in the filter wheel and the pupil wheel are used in combination. This
        function separates the two values.

        Parameters
        ----------
        filter_string : str
            Filter name as given in xml file from APT

        Returns
        -------
        filter_name : str
            Name of the filter in the filter wheel

        pupil_name : str
            Name of the filter in the pupil wheel
        """
        if '+' in filter_string:
            pupil_name, filter_name = filter_string.split('+')
            pupil_name = pupil_name.strip()
            filter_name = filter_name.strip()
        else:
            pupil_name = 'CLEAR'
            filter_name = filter_string
        return pupil_name, filter_name


def _get_guider_number(xml_file, observation_number):
    """"Parse the guider number for a particular FGSExternalCalibration or
    WfscGlobalAlignment observation.
    """
    observation_number = int(observation_number)
    apt_namespace = '{http://www.stsci.edu/JWST/APT}'
    fgs_namespace = '{http://www.stsci.edu/JWST/APT/Template/FgsExternalCalibration}'

    with open(xml_file) as f:
        tree = etree.parse(f)

    observation_data = tree.find(apt_namespace + 'DataRequests')
    observation_list = observation_data.findall('.//' + apt_namespace + 'Observation')
    for obs in observation_list:
        if int(obs.findtext(apt_namespace + 'Number')) == observation_number:
            try:
                detector = obs.findtext('.//' + fgs_namespace + 'Detector')
                number = detector[-1]
                return number
            except TypeError:
                number = _get_guider_number_from_special_requirements(apt_namespace, obs)
                return number

    raise RuntimeError('Could not find guider number in observation {} in {}'.format(observation_number, xml_file))


def _get_guider_number_from_special_requirements(apt_namespace, obs):
    """Parse the guider number from the SpecialRequirements for a particular WfscGlobalAlignment
    observation.
    """
    sr = [x for x in obs.iterchildren() if x.tag.split(apt_namespace)[1] == "SpecialRequirements"][0]
    try:
        gs = [x for x in sr.iterchildren() if x.tag.split(apt_namespace)[1] == "GuideStarID"][0]
    except IndexError:
        raise IndexError('There is no Guide Star Special Requirement for this observation')

    # Pull out the guide star ID and the guider number
    guider = [x for x in gs.iterchildren() if x.tag.split(apt_namespace)[1] == "Guider"][0].text
    if not isinstance(guider, int):
        guider = guider.lower().replace(' ', '').split('guider')[1]

    return guider

def _make_start_times(obs_info):
    """Create exposure start times for each entry in the observation dictionary.

    Parameters
    ----------
    obs_info : dict
        Dictionary of exposures. Development was around a dictionary containing
        APT xml-derived properties as well as pointing file properties. Should
        be before expanding to have one entry for each detector in each exposure.

    Returns
    -------
    obs_info : dict
        Modified dictionary with observation dates and times added
    """
    logger = logging.getLogger('mirage.apt.apt_inputs')

    date_obs = []
    time_obs = []
    expstart = []
    nframe = []
    nskip = []
    namp = []

    if 'epoch_start_date' in obs_info.keys():
        epoch_base_date = obs_info['epoch_start_date'][0]
    else:
        epoch_base_date = obs_info['Date'][0]

    base = Time(obs_info['epoch_start_date'][0])
    #base = Time(epoch_base_date + 'T' + epoch_base_time)
    base_date, base_time = base.iso.split()

    # Pick some arbirary overhead values
    act_overhead = 90  # seconds. (filter change)
    visit_overhead = 600  # seconds. (slew)

    # Get visit, activity_id, dither_id info for first exposure
    ditherid = obs_info['dither'][0]
    actid = obs_info['act_id'][0]
    visit = obs_info['visit_num'][0]
    obsid = obs_info['ObservationID'][0]
    exp = obs_info['exposure'][0]
    entry_num = obs_info['entry_number'][0]

    for i, instrument in enumerate(obs_info['Instrument']):
        # Get dither/visit
        # Files with the same activity_id should have the same start time
        # Overhead after a visit break should be large, smaller between
        # exposures within a visit
        next_actid = obs_info['act_id'][i]
        next_visit = obs_info['visit_num'][i]
        next_obsname = obs_info['obs_label'][i]
        next_ditherid = obs_info['dither'][i]
        next_obsid = obs_info['ObservationID'][i]
        next_exp = obs_info['exposure'][i]
        next_entry_num = obs_info['entry_number'][i]

        # Find the readpattern of the file
        readpatt = obs_info['ReadoutPattern'][i]
        groups = int(obs_info['Groups'][i])
        integrations = int(obs_info['Integrations'][i])

        if instrument.lower() in ['miri', 'nirspec']:
            nframe.append(0)
            nskip.append(0)
            namp.append(0)
            date_obs.append(base_date)
            time_obs.append(base_time)
            expstart.append(base.mjd)

        else:
            if instrument.lower() == 'niriss':
                readout_pattern_file = 'niriss_readout_pattern.txt'
                subarray_def_file = 'niriss_subarrays.list'
            elif instrument.lower() == 'fgs':
                readout_pattern_file = 'guider_readout_pattern.txt'
                subarray_def_file = 'guider_subarrays.list'
            elif instrument.lower() == 'nircam':
                readout_pattern_file = 'nircam_read_pattern_definitions.list'
                subarray_def_file = 'NIRCam_subarray_definitions.list'
            file = os.path.join(conf.PYNRC_PATH, 'sim_params', readout_pattern_file)
            readpatt_def = ascii.read(file)
            file = os.path.join(conf.PYNRC_PATH, 'sim_params', subarray_def_file)
            subarray_def = ascii.read(file)

            match2 = readpatt == readpatt_def['name']
            if np.sum(match2) == 0:
                raise RuntimeError(("WARNING!! Readout pattern {} not found in definition file."
                                    .format(readpatt)))

            # Now get nframe and nskip so we know how many frames in a group
            fpg = int(readpatt_def['nframe'][match2][0])
            spg = int(readpatt_def['nskip'][match2][0])
            nframe.append(fpg)
            nskip.append(spg)

            # Get the aperture name. For non-NIRCam instruments,
            # this is simply the obs_info['aperture']. But for NIRCam,
            # we need to be careful of entries like NRCBS_FULL, which is used
            # for observations using all 4 shortwave B detectors. In that case,
            # we need to build the aperture name from the combination of detector
            # and subarray name.
            aperture = obs_info['aperture'][i]

            # Get the number of amps from the subarray definition file
            match = aperture == subarray_def['AperName']

            if np.sum(match) == 0:
                if '_MASKLWB' in aperture or '_MASKSWB' in aperture:
                    apsplit = aperture.split('_')
                    no_filter = '{}_{}'.format(apsplit[0], apsplit[1])
                    match = no_filter == subarray_def['AperName']

            # needed for NIRCam case
            if np.sum(match) == 0:
                logger.info(('Aperture: {} does not match any entries in the subarray definition file. Guessing at the '
                             'aperture for the purpose of calculating the exposure time and number of amps.'.format(aperture)))
                sub = aperture.split('_')[1]
                aperture = [apername for apername, name in
                            np.array(subarray_def['AperName', 'Name']) if
                            (sub in apername) or (sub in name)]

                match = aperture == subarray_def['AperName']

                if len(aperture) > 1 or len(aperture) == 0 or np.sum(match) == 0:
                    raise ValueError('Cannot combine detector and subarray {}\
                                     into valid aperture name.'.format(sub))
                # We don't want aperture as a list
                aperture = aperture[0]

            # For grism tso observations, get the number of
            # amplifiers to use from the APT file.
            # For other modes, check the subarray def table.
            try:
                amp = int(obs_info['NumOutputs'][i])
            except ValueError:
                amp = subarray_def['num_amps'][match][0]

            # Default to amps=4 for subarrays that can have 1 or 4
            # if the number of amps is not defined. Hopefully we
            # should never enter this code block given the lines above.
            if amp == 0:
                amp = 4
                logger.info(('Aperture {} can be used with 1 or 4 readout amplifiers. Defaulting to use 4.'
                             'In the future this information should be made a user input.'.format(aperture)))
            namp.append(amp)

            # same activity ID
            # Remove this for now, since Mirage was not correctly
            # specifying activities. At the moment all exposures have
            # the same activity ID, which means we must allow the
            # the epoch_start_date to change even if the activity ID
            # does not. This will change back in the future when we
            # figure out more realistic activity ID values.
            #if next_actid == actid:
            #    # in this case, the start time should remain the same
            #    date_obs.append(base_date)
            #    time_obs.append(base_time)
            #    expstart.append(base.mjd)
            #    continue

            epoch_date = obs_info['epoch_start_date'][i]
            #epoch_time = copy.deepcopy(epoch_base_time0)

            # new epoch - update the base time
            if epoch_date != epoch_base_date:
                epoch_base_date = copy.deepcopy(epoch_date)
                #base = Time(epoch_base_date + 'T' + epoch_base_time)
                base = Time(obs_info['epoch_start_date'][i])
                base_date, base_time = base.iso.split()
                basereset = True
                date_obs.append(base_date)
                time_obs.append(base_time)
                expstart.append(base.mjd)
                actid = copy.deepcopy(next_actid)
                visit = copy.deepcopy(next_visit)
                obsid = copy.deepcopy(next_obsid)
                obsname = copy.deepcopy(next_obsname)
                ditherid = copy.deepcopy(next_ditherid)
                exp = copy.deepcopy(next_exp)
                entry_num = copy.deepcopy(next_entry_num)
                continue

            # new observation or visit (if a different epoch time has
            # not been provided)
            if ((next_obsid != obsid) | (next_visit != visit)):
                # visit break. Larger overhead
                overhead = visit_overhead
            elif ((next_actid > actid) & (next_visit == visit)):
                # This block should be updated when we have more realistic
                # activity IDs
                # same visit, new activity. Smaller overhead
                overhead = act_overhead
            elif ((next_ditherid != ditherid) & (next_visit == visit)):
                # same visit, new dither position. Smaller overhead
                overhead = act_overhead
            else:
                # same observation, activity, dither. Filter changes
                # will still fall in here, which is not accurate
                overhead = 0.  # Reset frame captured in exptime below

            # For cases where the base time needs to change
            # continue down here
            siaf_inst = obs_info['Instrument'][i].upper()
            siaf_obj = Siaf(siaf_inst)[aperture]

            # Calculate the readout time for a single frame
            frametime = _calc_frame_time(siaf_inst, aperture, siaf_obj.XSciSize, siaf_obj.YSciSize, amp)

            # Estimate total exposure time
            exptime = ((fpg + spg) * groups + fpg) * integrations * frametime

            if ((next_obsid == obsid) & (next_visit == visit) & (next_actid == actid) & (next_ditherid == ditherid) & (next_entry_num == entry_num)):
                # If we are in the same exposure (but with a different detector),
                # then we should keep the start time the same
                delta = TimeDelta(0., format='sec')
            else:
                # If we are moving on to the next exposure, activity, or visit
                # then move the start time by the expoure time of the current
                # exposure, plus the overhead
                delta = TimeDelta(exptime + overhead, format='sec')

            base += delta
            base_date, base_time = base.iso.split()

            # Add updated dates and times to the list
            date_obs.append(base_date)
            time_obs.append(base_time)
            expstart.append(base.mjd)

            # increment the activity ID and visit
            actid = copy.deepcopy(next_actid)
            visit = copy.deepcopy(next_visit)
            obsname = copy.deepcopy(next_obsname)
            ditherid = copy.deepcopy(next_ditherid)
            obsid = copy.deepcopy(next_obsid)
            exp = copy.deepcopy(next_exp)
            entry_num = copy.deepcopy(next_entry_num)

    obs_info['date_obs'] = date_obs
    obs_info['time_obs'] = time_obs
    obs_info['nframe'] = nframe
    obs_info['nskip'] = nskip
    obs_info['namp'] = namp
    return obs_info

def _calc_frame_time(instrument, aperture, xdim, ydim, amps):
    """Calculate the readout time for a single frame
    of a given size and number of amplifiers. Note that for
    NIRISS and FGS, the fast readout direction is opposite to
    that in NIRCam, so we switch xdim and ydim so that we can
    keep a single equation.

    Parameters:
    -----------
    instrument : str
        Name of the instrument being simulated

    aperture : str
        Name of aperture being simulated (e.g "NRCA1_FULL")
        Currently this is only used to check for the FGS
        ACQ1 aperture, which uses a unique value of colpad
        below.

    xdim : int
        Number of columns in the frame

    ydim : int
        Number of rows in the frame

    amps : int
        Number of amplifiers used to read out the frame

    Returns:
    --------
    frametime : float
        Readout time in seconds for the frame
    """
    instrument = instrument.lower()
    if instrument == "nircam":
        colpad = 12

        # Fullframe
        if amps == 4:
            rowpad = 1
            fullpad = 1
        else:
            # All subarrays
            rowpad = 2
            fullpad = 0

            if ((xdim <= 8) & (ydim <= 8)):
                # The smallest subarray
                rowpad = 3

    elif instrument == "niriss":
        # Reverse x and y since NIRISS's fast readout direction is
        # opposite of NIRCam's
        tmpx = copy.deepcopy(xdim)
        xdim = copy.deepcopy(ydim)
        ydim = tmpx

        colpad = 12

        # Fullframe
        if amps == 4:
            rowpad = 1
            fullpad = 1
        else:
            rowpad = 2
            fullpad = 0

    elif instrument == 'fgs':
        # Reverse x and y since FGS's fast readout direction is
        # opposite of NIRCam's
        tmpx = copy.deepcopy(xdim)
        xdim = copy.deepcopy(ydim)
        ydim = tmpx

        colpad = 12
        fullpad = 0

        if ((xdim == 2048) & (ydim == 2048)):
            rowpad = 1
            fullpad = 1
        else:
            rowpad = 2

        if ((xdim <= 32) & (ydim <= 32)):
            colpad = 6
            rowpad = 1

    return ((1.0 * xdim / amps + colpad) * (ydim + rowpad) + fullpad) * 1.e-5



###  Modified from MIRAGE
def get_pointing_info(pointing_files, propid=0, verbose=False, all_inst=False):
    """Read in information from APT's pointing file.

    Parameters
    ----------
    file : str
        Name of APT-exported pointing file to be read
    propid : int
        Proposal ID number (integer). This is used to
        create various ID fields
    all_inst : bool
        If False, only NIRCam, otherwise get all instruments.

    Returns
    -------
    pointing : dict
        Dictionary of pointing-related information

    TODO
    ----
        extract useful information from header?
        check visit numbers
        set parallel proposal number correctly

    """

    from .dms import jw_obs_id

    tar = []
    tile = []
    dith = []
    aperture = []
    targ1 = []
    targ2 = []
    ra = []
    dec = []
    basex = []
    basey = []
    dithx = []
    dithy = []
    v2 = []
    v3 = []
    idlx = []
    idly = []
    level_arr = []
    type_str = []
    expar = []
    dkpar = []
    ddist = []
    observation_label = []
    obs_id_all = []
    obs_num_int = []
    visit_num_int = []

    pri_dith_arr = []
    sub_dith_arr = []

    grp_counter = 1
    act_counter = 1
    expnum = 1
    with open(pointing_files) as f:
        for line in f:

            # Skip comments and new lines except for the line with the version of the PRD
            if (line[0] == '#') or (line in ['\n']) or ('=====' in line):

                # Compare the version of the PRD from APT and pysiaf
                if 'PRDOPSSOC' in line:
                    apt_prd_version = line.split(' ')[-2]
                    if apt_prd_version != JWST_PRD_VERSION:
                        _log.info(('The pointing file from APT was created using PRD version: {},\n'
                                    'while the current installation of pysiaf uses PRD version: {}.\n'
                                    .format(apt_prd_version, JWST_PRD_VERSION)))
                else:
                    continue
            # Extract proposal ID
            elif line.split()[0] == 'JWST':
                propid_header = line.split()[7]
                try:
                    propid = int(propid_header)
                except ValueError:
                    # adopt value passed to function
                    pass
                if verbose:
                    _log.info('Extracted proposal ID {}'.format(propid))
                continue

            elif (len(line) > 1):
                elements = line.split()

                # Look for lines that give visit/observation numbers
                if line[0:2] == '* ':
                    paren = line.rfind('(')
                    if paren == -1:
                        obslabel = line[2:]
                        obslabel = obslabel.strip()
                    else:
                        obslabel = line[2:paren-1]
                        obslabel = obslabel.strip()
                    if (' (' in obslabel) and (')' in obslabel):
                        obslabel = re.split(r' \(|\)', obslabel)[0]

                skip = False

                if line[0:2] == '**':
                    v = elements[2]
                    obsnum, visitnum = v.split(':')
                    obsnum = str(obsnum).zfill(3)
                    visitnum = str(visitnum).zfill(3)
                    # At beginning of each visit, reset numbers
                    grp_counter = 2 # Default to 2 since GS acq is 1
                    act_counter = 1
                    expnum = 1
                    if (skip is True) and (verbose):
                        _log.info('Skipping observation {} ({})'.format(obsnum, obslabel))

                try:

                    # Only care about NIRCam
                    if all_inst:
                        do_this = (int(elements[1]) > 0)
                    else:
                        do_this = ((int(elements[1]) > 0) & ('NRC' in elements[4])) or \
                            (('TA' in elements[4]) & ('NRC' in elements[4]))

                    # Skip visit name lines that might sneak past above
                    if line[0:2] == '* ':
                        do_this = False

                    if do_this:

                        level = elements[17]
                        type_val = elements[18]
                        if (type_val == 'PARALLEL'):
                            skip = True

                        if skip:
                            # act_counter += 1
                            continue

                        observation_label.append(obslabel)

                        # Visit Groups and Activity numbers
                        ap = elements[4]
                        if type_val=='T_ACQ':
                            grp_counter = 2
                            act_counter = 2 # First activity is always subarray switch
                        elif type_val=='CONFIRM':
                            grp_counter = 3
                            if 'TA' not in ap:
                                act_counter += 1 # Add additional activity for SAM
                        elif ('MASK' in ap) and (type_val=='SCIENCE'):
                            if (level=='TARGET') and ('FULL' not in ap):
                                act_counter += 1 # Add additional activity for FULL->subarray
                            elif (level=='TARGET') and (act_counter==1):
                                act_counter += 1 # Add additional activity (NRCSUBMAIN, SUBARRAY=FULL)
                            elif (level == 'FILTER'):
                                act_counter += 1 # Start of new activity
                                expnum = 1       # Reset exposure number
                            grp_counter = 3

                        # Parallel sequence is hard coded to 1 (Simulated instrument as prime rather than
                        # parallel) at the moment. Future improvements may allow the proper sequence
                        # number to be constructed.
                        seq = '1'

                        # Set activity ID to 1

                        tar.append(int(elements[0]))
                        tile.append(int(elements[1]))
                        exnum = int(elements[2])
                        # exp.append(str(exnum).zfill(5))
                        dith_pos = int(elements[3])
                        dith.append(dith_pos)
                        # If pointing file exp and dith are both 1, then reset exposure num to 1
                        # otherwise increment by 1.

                        # Reset primary dither position on new target, filter, or tile
                        if level in ['TARGET', 'FILTER', 'TILE']:
                            pri_dith_pos = 1
                            sub_dith_pos = 1
                        elif level=='DITHER':
                            pri_dith_pos += 1
                            sub_dith_pos = 1
                        elif level=='SUBDITHER':
                            sub_dith_pos += 1
                        pri_dith_arr.append(pri_dith_pos)
                        sub_dith_arr.append(sub_dith_pos)

                        # if ('GRISMR_WFSS' in elements[4]):
                        #     ap = ap.replace('GRISMR_WFSS', 'FULL')
                        # elif ('GRISMC_WFSS' in elements[4]):
                        #     ap = ap.replace('GRISMC_WFSS', 'FULL')

                        aperture.append(ap)
                        targ1.append(int(elements[5]))
                        targ2.append(elements[6])
                        ra.append(float(elements[7]))
                        dec.append(float(elements[8]))
                        basex.append(float(elements[9]))
                        basey.append(float(elements[10]))
                        dithx.append(float(elements[11]))
                        dithy.append(float(elements[12]))
                        v2.append(float(elements[13]))
                        v3.append(float(elements[14]))
                        idlx.append(float(elements[15]))
                        idly.append(float(elements[16]))
                        level_arr.append(level)
                        type_str.append(elements[18])
                        expar.append(int(elements[19]))
                        dkpar.append(int(elements[20]))
                        ddist.append(float(elements[21]))

                        # For the moment we assume that the instrument being simulated is not being
                        # run in parallel, so the parallel proposal number will be all zeros,
                        # as seen in the line below.
                        # observation_id.append("V{}P{}{}{}{}".format(vid, '00000000', vgrp, seq, act))
                        # act_counter += 1

                        # Use dither position as a proxy for exp ID. 
                        # Each dither should have it's own file
                        obs_id = jw_obs_id(propid, obsnum, visitnum, grp_counter, seq, act_counter, expnum)
                        obs_id_all.append(obs_id)

                        obs_num_int.append(int(obsnum))
                        visit_num_int.append(int(visitnum))

                        # Increment exposure number
                        expnum += 1

                        # Increment group counter after target acq
                        # useful for TSO
                        if type_val=='T_ACQ':
                            grp_counter += 1
                            expnum = 1 # Reset exposure counter
                        # Increment activity coun
                        elif type_val=='CONFIRM':
                            if 'TA' in ap:
                                act_counter += 1 
                            expnum = 1 # Reset exposure counter

                except ValueError as e:
                    if verbose:
                        _log.info('Skipping line:\n{}\nproducing error:\n{}'.format(line, e))
                    pass

    pointing = {
        'aperture': np.array(aperture), 'pri_dith': np.array(pri_dith_arr), 'sub_dith': np.array(sub_dith_arr),
        'targ1': np.array(targ1), 'targ2': np.array(targ2), 'ra': np.array(ra), 'dec': np.array(dec),
        'basex': np.array(basex), 'basey': np.array(basey), 'dithx': np.array(dithx), 'dithy': np.array(dithy), 
        'v2': np.array(v2), 'v3': np.array(v3), 'idlx': np.array(idlx), 'idly': np.array(idly), 
        'obs_label': np.array(observation_label), 'obs_num': np.array(obs_num_int), 
        'visit_num': np.array(visit_num_int), 'obs_id_info': np.array(obs_id_all), 
        'level': np.array(level_arr), 'type': np.array(type_str), 'ddist': np.array(ddist),
        }
    return pointing


def build_dict_from_xml(xml_file, keys, verbose=False):
    """Read in the .xml file from APT, and output dictionary of parameters.

    Arguments
    ---------
    infile (str):
        Path to input .xml file

    Returns
    -------
    dict:
        Dictionary with extracted observation parameters

    Raises
    ------
    ValueError:
        If an .xml file is provided that includes an APT template that is not
        supported.
        If the .xml file includes a fiducial pointing override with an
        unknown subarray specification
    """
    temp = ReadAPTXML()
    pointing_info = temp.read_xml(xml_file, verbose=verbose)
    keys_pointing = list(pointing_info.keys())

    out_dict = {}
    for k in keys:
        out_dict[k] = []

    instrument_list = set(pointing_info['Instrument'])
    for inst in instrument_list:
        good = np.where(np.array(pointing_info['Instrument']) == inst)[0]
        if inst.upper() == 'NIRCAM':
            ndithers = np.array(pointing_info['number_of_dithers'])[good]
            nel = len(ndithers)
            for k in keys:
                vals = np.array(pointing_info[k])[good]
                for i in range(nel):
                    val  = vals[i]
                    ndith = int(ndithers[i])

                    # Append filters for each dither position
                    for j in range(ndith):
                        out_dict[k].append(val)

    for k in keys:
        out_dict[k] = np.array(out_dict[k])

    return out_dict

def get_ditherinfo(xml_file, verbose=False):
    """Dither information"""

    keys = [
        'APTTemplate', 'PrimaryDitherType', 'PrimaryDithers', 'DitherSize', 
        'SubpixelPositions', 'SubpixelDitherType',
        'SmallGridDitherType', 'DitherPatternType', 
        'ImageDithers', 'number_of_dithers'
    ]

    return build_dict_from_xml(xml_file, keys, verbose=verbose)

def get_siaf_detectors(apname):
    """List detectors associated with SIAF aperture name"""
    
    str_arr = apname.split('_')
    detid = str_arr[0]
    

    det_amod = ['NRCA1', 'NRCA2', 'NRCA3', 'NRCA4', 'NRCA5']
    det_bmod = ['NRCB1', 'NRCB2', 'NRCB3', 'NRCB4', 'NRCB5']

    if detid=='NRCAS':
        # NRCAS_FULL uses all mod A detectors
        return det_amod
    elif detid=='NRCBS': 
        # NRCBS_FULL uses all mod B detectors
        return det_bmod
    elif detid=='NRCALL':
        return det_amod + det_bmod
    else:
        return [detid]

def update_subarray_imaging(visit_dict):
    """Update detectors and apertures used or subaray observations"""

    apt_template = visit_dict.get('APTTemplate')
    if (apt_template is not None) and ('nircamimaging' not in apt_template.lower()):
        _log.warning(f'APT template {apt_template} is not NircamImaging. Returning...')
        return

    # Update names of apertures and detectors
    apertures = []
    detectors = []
    for i, subname in enumerate(visit_dict['subarray_name']):
        if subname=='FULL':
            if mod=='A':
                aps = 'NRCAS_FULL'
            elif mod=='B':
                aps = 'NRCBS_FULL'
            else:
                aps = 'NRCALL_FULL'

            dets = get_siaf_detectors(aps)
        elif ('SUB' in subname):
            mod = visit_dict['ModuleAPT'][i]
            pix = subname[3:]
            subp = subname[-1]=='P'
            if subp:
                pix = pix[:-1]

            if subp:
                aps_amod = [f'NRCA{j}_SUB{pix}P' for j in [3,5]]
                aps_bmod = [f'NRCB{j}_SUB{pix}P' for j in [1,5]]
            else:
                aps_amod = [f'NRCA{j}_SUB{pix}' for j in [1,2,3,4,5]]
                aps_bmod = [f'NRCB{j}_SUB{pix}' for j in [1,2,3,4,5]]

            if mod=='A':
                aps = aps_amod
            elif mod=='B':
                aps = aps_bmod
            else:
                aps = aps_amod + aps_bmod

            # Get detector name for each aperture
            dets = [get_siaf_detectors(ap)[0] for ap in aps]

        apertures.append(aps)
        detectors.append(dets)

    visit_dict['aperture'] = np.array(apertures)
    visit_dict['detectors'] = np.array(detectors)


def update_eng_detectors(visit_dict):
    """Update detectors used for engineering templates
    
    Engineering templates always use all detectors operated with the
    same subarray settings, but there are not always SIAF apertures
    for every detector for certain subarray settings (e.g., SUBGRISM), 
    so some of the auto-generated detector from `get_siaf_detectors`
    will usually be incorrect.
    """

    apt_template = visit_dict.get('APTTemplate')
    if (apt_template is not None) and ('engineering' not in apt_template.lower()):
        _log.warning(f'APT template {apt_template} is not Engineering. Returning...')
        return
    
    # Update detectors used
    detectors = []
    for i, mod in enumerate(visit_dict['ModuleAPT']):

        swpupil = visit_dict['sw_pupils'][i]
        if ('MASKRND' in swpupil) or ('MASKBAR' in swpupil):
            # det_amod = ['NRCA2', 'NRCA4', 'NRCA5']
            # det_bmod = ['NRCB1', 'NRCB3', 'NRCB5']
            det_amod = ['NRCA1', 'NRCA2', 'NRCA3', 'NRCA4', 'NRCA5']
            det_bmod = ['NRCB1', 'NRCB2', 'NRCB3', 'NRCB4', 'NRCB5']
        else:
            det_amod = ['NRCA1', 'NRCA2', 'NRCA3', 'NRCA4', 'NRCA5']
            det_bmod = ['NRCB1', 'NRCB2', 'NRCB3', 'NRCB4', 'NRCB5']

        if mod=='A':
            dets = det_amod
        elif mod=='B':
            dets = det_bmod
        else:
            dets = det_amod + det_bmod
        detectors.append(dets)
    visit_dict['detectors'] = np.array(detectors)

    # Update names of apertures
    apertures = []
    for i, subname in enumerate(visit_dict['subarray_name']):
        mod = visit_dict['ModuleAPT'][i]
        if subname=='FULL':
            if mod=='A':
                aps = 'NRCAS_FULL'
            elif mod=='B':
                aps = 'NRCBS_FULL'
            else:
                aps = 'NRCALL_FULL'
        elif ('SUBGRISM' in subname):
            pix = subname[8:]

            # GRISMTS for SW, although some of these apertures are undefined in SIAF!
            aps_amod = [f'NRCA{j}_GRISMTS{pix}' for j in [1,2,3,4]]
            aps_bmod = [f'NRCB{j}_GRISMTS{pix}' for j in [1,2,3,4]]

            lw_filter = visit_dict['lw_filters'][i]
            # Only certain filters are allowed. Default to F322W2 if some other filter specified.
            if lw_filter not in ['F322W2', 'F277W', 'F356W', 'F444W']:
                lw_filter = 'F322W2'
            aps_amod = aps_amod + [f'NRCA5_GRISM{pix}_{lw_filter}']
            aps_bmod = aps_bmod + [f'NRCB5_GRISM{pix}_{lw_filter}']

            if mod=='A':
                aps = aps_amod
            elif mod=='B':
                aps = aps_bmod
            else:
                aps = aps_amod + aps_bmod
        elif ('SUB' in subname) and ('GRISM' not in subname) and ('DHS' not in subname):
            pix = subname[3:]
            subp = subname[-1]=='P'
            if subp:
                pix = pix[:-1]

            aps_amod = [f'NRCA{j}_SUB{pix}' for j in [1,2,3,4,5]]
            aps_bmod = [f'NRCB{j}_SUB{pix}' for j in [1,2,3,4,5]]
            if subp:
                aps_amod[2] = aps_amod[2] + 'P'
                aps_amod[4] = aps_amod[4] + 'P'
                aps_bmod[0] = aps_bmod[0] + 'P'
                aps_bmod[4] = aps_bmod[4] + 'P'

            if mod=='A':
                aps = aps_amod
            elif mod=='B':
                aps = aps_bmod
            else:
                aps = aps_amod + aps_bmod
        else:
            _log.warning(f'{subname} not yet supported. Setting apertures to None')
            aps = 'NONE'
        
        apertures.append(aps)
    visit_dict['aperture'] = np.array(apertures)


def get_readmodes(xml_file, verbose=False):
    """Readout information for each exposure"""

    key_update = {
        'Mode'           : 'mode',
        'Subarray'       : 'subarray_name',
        'ReadoutPattern' : 'readout',
        'Integrations'   : 'nints',
        'Groups'         : 'ngroups',
        'NumOutputs'     : 'noutputs',
    }

    res = build_dict_from_xml(xml_file, key_update.keys(), verbose=verbose)

    out_dict = {}
    for k in key_update.keys():
        out_dict[key_update[k]] = res[k]

    return out_dict

def get_target_info(xml_file, verbose=False):
    """Target for each each exposure"""
    key_update = {
        'TargetID' : 'TargetID',
        'TargetRA' : 'TargetRA', 
        'TargetDec': 'TargetDec',
    }
    res = build_dict_from_xml(xml_file, key_update.keys(), verbose=verbose)

    out_dict = {}
    for k in key_update.keys():
        out_dict[key_update[k]] = res[k]

    return out_dict

def get_filter_info(xml_file, verbose=False):
    """Filter information for each exposure"""

    key_update = {
        'Module'      : 'ModuleAPT',
        'ShortFilter' : 'sw_filters',
        'ShortPupil'  : 'sw_pupils',
        'LongFilter'  : 'lw_filters',
        'LongPupil'   : 'lw_pupils',
        'CoronMask'   : 'coron_mask',
    }

    res = build_dict_from_xml(xml_file, key_update.keys(), verbose=verbose)

    out_dict = {}
    for k in key_update.keys():
        out_dict[key_update[k]] = res[k]

    # Certain filters in pupil wheel should be stored in filter
    # For instance filter settings 'F405N+F444W' were parsed as
    # placing 'LongPupil':'F405N' and 'LongFilter':'F444W',
    # but in order to correctly create pynrc observations, we want
    # to set filter='F405N'.
    for i, val in enumerate(out_dict['sw_pupils']):
        if (val[0] == 'F') and (val != 'FLAT'):
            new_pup = 'WLP4' if out_dict['sw_filters'][i]=='WLP4' else 'CLEAR'
            out_dict['sw_filters'][i] = val
            out_dict['sw_pupils'][i] = new_pup
    for i, val in enumerate(out_dict['lw_pupils']):
        if (val[0] == 'F') and (val != 'FLAT'):
            out_dict['lw_filters'][i] = val
            out_dict['lw_pupils'][i] = 'CLEAR'

    # If filter is set to WLP4, set pupil depending on pupil setting
    for i, val in enumerate(out_dict['sw_filters']):
        if val=='WLP4':
            sw_pup = out_dict['sw_pupils'][i]
            if sw_pup=='WLP8':
                new_pup = 'WLP12'
            elif sw_pup=='WLM8':
                new_pup = 'WLM4'
            else:
                new_pup = 'WLP4'
            out_dict['sw_pupils'][i]  = new_pup
            out_dict['sw_filters'][i] = 'CLEAR'

    return out_dict


def get_timing_info(timing_json_file, smart_accounting_file):

    f = open(timing_json_file)
    timing_info = json.load(f)
    f.close

    # Use smart accounting file to get order of visits
    obs_visit_num = []
    with open(smart_accounting_file) as f:
        for line in f:
            good = (line[0]==' ') & (':' in line)
            if good:
                obs_visit_num.append(line[1:8])

    # Place visit dictionaries in sequence
    visit_dict = OrderedDict()
    for vn in obs_visit_num:
        obs_num, visit_num = vn.split(':')
        obs_num = int(obs_num)
        visit_num = int(visit_num)
        
        # Search for obs:visit in timing_info
        for dobs in timing_info['observations']:
            if dobs['id'] == obs_num:
                for dvisit in dobs['visits']:
                    if dvisit['id'] == visit_num:
                        visit_dict[vn] = dvisit

    # Get start times of each exposure
    tval = 0
    for k in visit_dict.keys():
        exp_start_times = []
        comp_list = visit_dict[k]['components']
        for d in comp_list:
            # Save visit start time and slew duration
            if d['type']=='VISIT_SLEW':
                visit_dict[k]['visit_start'] = tval
                visit_dict[k]['slew_duration'] = d['duration']

            # Append science exposure start times
            if d['type']=='SCIENCE':
                exp_start_times.append(tval)
            tval += d['duration']

        visit_dict[k]['exp_start_times'] = np.array(exp_start_times)

    return visit_dict

def get_proposal_info(xml_file, verbose=False):
    """Get basic program information"""

    temp = ReadAPTXML()
    program_info = temp.read_xml(xml_file, verbose=verbose)

    obs_params = {
        # Proposal info
        'pi_name'          : program_info['PI_Name'][0],
        'title'            : program_info['Title'][0],
        'pid'              : program_info['ProposalID'][0],
        'category'         : program_info['Proposal_category'][0],
        'sub_category'     : program_info['Proposal_subcategory'][0],
        'science_category' : program_info['Science_category'][0],
    }

    return obs_params

def get_roll_info(xml_file):

    with open(xml_file) as f:
        tree = etree.parse(f)

    apt = '{http://www.stsci.edu/JWST/APT}'
    LinkingRequirements = tree.find(apt + 'LinkingRequirements')
    OrientFromLink = LinkingRequirements.findall('.//' + apt + 'OrientFromLink')

    # Loop through PA offset links
    roll_dict = {}
    for i, orient in enumerate(OrientFromLink):
        obs1 = orient.get('PrimaryObs')
        obs2 = orient.get('OrientFromObs')
        obs1_num = int(obs1.split('(Obs ')[-1][:-1])
        obs2_num = int(obs2.split('(Obs ')[-1][:-1])

        pa_min = int(orient.get('MinAngle').split(' ')[0])
        pa_max = int(orient.get('MaxAngle').split(' ')[0])

        # Store PA offset link
        roll_dict[i] = {
            'PrimaryObs'    : obs1_num,
            'OrientFromObs' : obs2_num,
            'MinPA' : pa_min,
            'MaxPA' : pa_max,
        }

    if len(roll_dict)==0:
        return None
    else:
        return roll_dict

def get_orient_specreq(xml_file):
    """ Grab V3 PA Range special requirements from XML file

    TODO: Value is specified in the aperture PA, not V3 PA. Need to calculate V3 PA.
    """

    with open(xml_file) as f:
        tree = etree.parse(f)

    apt = '{http://www.stsci.edu/JWST/APT}'

    orient_dict = {}
    # Get special requirements for Orient Range
    OrientRanges = tree.findall('.//' + apt + 'OrientRange')
    # Loop through each orient ranges PA offset links
    for ORange in OrientRanges:
        obs_num = ORange.getparent().getparent().find(apt+'Number').text.zfill(3)
        orient_dict[obs_num] = {
            'OrientMin' : float(ORange.get('OrientMin').split(' ')[0]),
            'OrientMax' : float(ORange.get('OrientMax').split(' ')[0]),
        }

    if len(orient_dict)==0:
        return None
    else:
        return orient_dict


def gen_all_apt_visits(xml_file, pointing_file, sm_acct_file, json_file, rand_seed=None):
    """
    Read in APT output files and return a dictionary that holds all
    necessary visit information to create an observation in DMS
    format. Each visit is placed in an ordered dictionary according
    to that within the Smart Accounting file.
    """

    rng = np.random.default_rng(rand_seed)

    timing_info = get_timing_info(json_file, sm_acct_file)
    target_info = get_target_info(xml_file)
    pointing_info = get_pointing_info(pointing_file)
    read_modes = get_readmodes(xml_file)
    filter_info = get_filter_info(xml_file)
    dith_info = get_ditherinfo(xml_file)
    roll_info = get_roll_info(xml_file)
    orient_info = get_orient_specreq(xml_file)

    # Save visit information to its own dictionary
    visits_dict = OrderedDict()
    for k in timing_info.keys():
        obs_num, visit_num = k.split(':')
        ind = (pointing_info['obs_num'] == int(obs_num)) & (pointing_info['visit_num'] == int(visit_num))

        # Pointing might have non-NIRCam files
        # This will only keep those with relevant obs_num and visit_num
        if ind.sum()>0:
            visits_dict[k] = {
                'obs_num': int(obs_num),
                'visit_num': int(visit_num),
                'visit_start': timing_info[k]['visit_start'],
                'slew_duration': timing_info[k]['slew_duration'],
                'visit_duration': timing_info[k]['scheduling_duration'],
                'exp_start_times': timing_info[k]['exp_start_times'],
            }
        
    # Add roll offset info to relevant visits
    if orient_info is not None:
        for k_obs in orient_info.keys():
            d_orient = orient_info[k_obs]

            # Cycle through each visit to find obs num matches
            for key in visits_dict.keys():
                d = visits_dict[key]
                obs_num = d['obs_num']
                # Only update if statment is True
                d['orient_info'] = d_orient if (obs_num==int(k_obs)) else d.get('orient_info')

    # Add orientation special requierment info to relevant visits
    if roll_info is not None:
        for k_roll in roll_info.keys():
            d_roll = roll_info[k_roll]
            onum1 = d_roll['PrimaryObs']
            onum2 = d_roll['OrientFromObs']

            # Cycle through each visit to find obs num matches
            for key in visits_dict.keys():
                d = visits_dict[key]
                obs_num = d['obs_num']
                # Only update if statment is True
                d['roll_info'] = d_roll if (obs_num==onum1) or (obs_num==onum2) else d.get('roll_info')


    template_key = 'APTTemplate'
    for key in visits_dict.keys():
        d = visits_dict[key]

        # Select specific observation and visit number
        obs_num, visit_num = (d['obs_num'], d['visit_num'])
        ind = (pointing_info['obs_num'] == obs_num) & (pointing_info['visit_num'] == visit_num)

        # Only need 1 entry per visit for dither information
        for k in dith_info.keys():
            if k in d.keys():
                pass
            else:
                d[k] = dith_info[k][ind]#[0]

        # Add a random seed for random dither offsets and noise
        d['rand_seed_init']  = rand_seed
        d['rand_seed_dith']  = rng.integers(0, 2**32-1)
        d['rand_seed_noise'] = rng.integers(0, 2**32-1)
            
        # Pointing, Filter, and Readout information
        for dict_append in [target_info, pointing_info, filter_info, read_modes]:
            for k in dict_append.keys():
                if k in d.keys():
                    pass
                else:
                    d[k] = dict_append[k][ind]

        # Only need single template name
        if template_key in d.keys():
            d[template_key] = d[template_key][0]

        # Add all detectors
        apertures = d['aperture']
        detectors_all = []
        for ap in apertures:
            detnames = get_siaf_detectors(ap)
            detectors_all.append(detnames)
        detectors_all = np.array(detectors_all)
        d['detectors'] = detectors_all

        # Update detectors and apertures if an Engineering Template
        template_name = d.get(template_key)
        if (template_name is not None) and ('engineering' in template_name.lower()):
            update_eng_detectors(d)
        elif (template_name is not None) and ('nircamimaging' in template_name.lower()):
            update_subarray_imaging(d)

    return visits_dict

def get_exp_type(visit_type, visit_mode, pupil):
    """
    Possible Exposure Types:
      NRC_DARK, NRC_FLAT, NRC_LED, NRC_FOCUS, 
      NRC_TACQ, NRC_TACONFIRM, 
      NRC_IMAGE, NRC_GRISM, NRC_CORON,
      NRC_WFSS, NRC_TSIMAGE, NRC_TSGRISM
    """

    types_dict = {
        "imaging"    : "NRC_IMAGE", 
        "wfss"       : "NRC_WFSS", 
        "ts_imaging" : "NRC_TSIMAGE", 
        "ts_grism"   : "NRC_TSGRISM",
    }

    visit_type = visit_type.upper()
    visit_mode = visit_mode.lower()
    pupil = pupil.upper()

    if pupil == 'FLAT':
        exp_type = 'NRC_DARK'
    elif 'T_ACQ' in visit_type:
        exp_type = 'NRC_TACQ'
    elif 'CONFIRM' in visit_type:
        exp_type = 'NRC_TACONFIRM'
    elif ('RND' in pupil) or ('BAR' in pupil) or ('CIRC' in pupil) or ('WEDGE' in pupil):
        exp_type = 'NRC_CORON'
    elif 'SCIENCE' in visit_type:
        try:
            exp_type = types_dict[visit_mode]
        except:
            if 'GRISM' in pupil:
                exp_type = 'NRC_GRISM'
            else:
                exp_type = 'NRC_IMAGE'
    else:
        _log.error(f'type: {visit_type}, mode: {visit_mode}, pupil: {pupil}')
        raise RuntimeError('Cannot determine exp_type.')
        
    return exp_type


def create_det_class(visit_dict, exp_id, detname, grp_id=1, seq_id=1, act_id='01'):

    from ..pynrc_core import DetectorOps
    from .dms import dec_to_base36

    # Ensure standardized detector naming convention ("NRC[A/B][1-5]")
    det_id = get_detname(detname)
    
    act_id_b36 = act_id if isinstance(act_id, str) else dec_to_base36(int(act_id))

    visit_group     = '{:02d}'.format(int(grp_id))  # Visit group identifier
    sequence_id     = '{:01d}'.format(int(seq_id))  # Parallel sequence ID (1=prime, 2-5=parallel)
    activity_id     = '{:0>2}'.format(act_id_b36)   # Activity number (base 36)
    exposure_number = '{:05d}'.format(int(exp_id))  # Exposure Number
    par_info_grab = visit_group + sequence_id + activity_id + '_' + exposure_number

    # Get list of all parameters strung together
    par_info_all = []
    for d in visit_dict['obs_id_info']:
        par_info = d['visit_group'] + d['sequence_id'] + d['activity_id'] + '_' + d['exposure_number']
        par_info_all.append(par_info)
    par_info_all = np.array(par_info_all)

    # Grab indices for specified info
    ind_mask = (par_info_all == par_info_grab)
    if ind_mask.sum()==0:
        _log.error("d['visit_group'] + d['sequence_id'] + d['activity_id'] + '_' + d['exposure_number']")
        print("Valid values:\n", par_info_all)
        raise IndexError(f'Not valid: {par_info_grab}; visit_group:{grp_id}, exp_id:{exp_id}, seq_id:{seq_id}, act_id:{act_id}')

    # Grab indices for specified exposure number
    # exp_ids_visit = np.array([int(d['exposure_number']) for d in visit_dict['obs_id_info']])
    # ind_mask      = (exp_ids_visit == int(exp_id))
    
    # Get siaf aperture names
    exp_apnames = visit_dict['aperture'][ind_mask]
    if exp_apnames.size==1:
        apname = visit_dict['aperture'][ind_mask][0]
    else:
        ind_det = (visit_dict['detectors'][ind_mask] == det_id)
        apname = visit_dict['aperture'][ind_mask][ind_det][0]
    if ('NRCAS' in apname) or ('NRCBS' in apname) or ('NRCALL' in apname):
        apname = det_id + '_FULL'
    # pysiaf aperture
    siaf_ap = siaf_nrc[apname]

    # Build detector operations class
    readout = visit_dict['readout'][ind_mask][0]
    nint    = visit_dict['nints'][ind_mask][0]
    ngroup  = visit_dict['ngroups'][ind_mask][0]
    xpix, ypix = np.array([siaf_ap.XSciSize, siaf_ap.YSciSize]).astype('int')
    x0, y0 = np.array(siaf_ap.dms_corner()) - 1
    # Check if number of amplifiers was specified in APT file (e.g., grism timeseries)
    noutputs = visit_dict['noutputs'][ind_mask][0]
    if (noutputs is not None) and (noutputs==1):
        wind_mode = 'WINDOW'
    elif (xpix >= 2048) and (ypix>=2048):
        wind_mode = 'FULL'
        noutputs = 4
    elif (xpix >= 2048):
        wind_mode = 'STRIPE'
        noutputs = 4
    else:
        wind_mode = 'WINDOW'
        noutputs = 1
    # Detector class
    det = DetectorOps(detector=det_id, read_mode=readout, nint=nint, ngroup=ngroup, 
                      wind_mode=wind_mode, xpix=xpix, ypix=ypix, x0=x0, y0=y0)

    return det


def create_obs_params(filt, pupil, mask, det, siaf_ap, ra_dec, date_obs, time_obs="12:00:00.000", 
    pa_v3=None, siaf_ap_obs=None, xyoff_idl=(0,0), visit_level='TARGET', visit_type='SCIENCE', 
    time_series=False, time_visit_offset=0, time_exp_offset=0, segNum=None, segTot=None, int_range=None, 
    filename=None, **kwargs):

    """ Generate obs_params dictionary

    An obs_params dictionary is used to create a jwst data model (e.g., Level1bModel).
    Additional ``**kwargs`` will add/update elements to the final output dictionary.
    
    Parameters
    ==========
    filt : str
        Observed filter
    pupil : str
        Observed pupil mask (e.g., GRISMR, GRISMC, CIRCLYOT, etc)
    det : DetectorOps
        NIRCam detector operations class
    siaf_ap : pysiaf Aperture
        SIAF aperture class used for telescope pointing
    ra_dec : tuple, list
        RA and Dec in degrees associated with observation pointing
    data_obs : str
        YYYY-MM-DD

    Keyword Arg
    ===========
    time_obs : str
        HH:MM:SS
    pa_v3 : float or None
        Telescope V3 position angle. If set to None, then will automatically determine
        from date and ra/dec.
    siaf_ap_obs : pysiaf Aperture
        SIAF aperture class used for to observe (if different from `siaf_ap`)
    xyoff_idl : tuple, list
        (x,y) offset in arcsec ('idl' coords) to dither observation
    visit_type : str
        'T_ACQ', 'CONFIRM', or 'SCIENCE'
    time_series : bool
        Is this a time series observation?
    time_exp_offset : float
        Exposure start time (in seconds) relative to beginning of observation execution. 
    segNum : int
        The segment number of the current product. Only for TSO.
    segTot : int
        The total number of segments. Only for TSO.
    int_range : list
        Integration indices to use 
    filename : str or None  
        Name of output filename.
    """

    from .dms import populate_group_table, jw_obs_id

    pupil = 'CLEAR' if pupil is None else pupil

    # Ensure standardized detector naming convention ("NRC[A/B][1-5]")
    det_id = get_detname(det.detname)
    # Multiaccum class
    ma = det.multiaccum

    siaf_ap_ref = siaf_ap
    ra, dec = ra_dec

    # Ensure 3 decimal points in time_obs
    try:
        # Split off decimal point after string
        # Add a bunch of zeros, then truncate
        tsplit = time_obs.split('.')
        sub_sec = (tsplit[1] + '000')[:3]
        time_obs = '.'.join([tsplit[0], sub_sec])
    except:
        # If no decimal, just add zeros
        time_obs = time_obs + '.000'

    output = get_tel_angles(ra, dec, obs_date=date_obs, obs_time=time_obs)
    pitch_ang = output['pitch_deg']
    pa_v3_nom = output['v3pa_deg']
    sol_elong = pitch_ang + 90
    if pa_v3 is None:
        pa_v3 = pa_v3_nom

    # TODO: Maybe add an orient ranges constraint here?

    if siaf_ap_obs is None:
        apname = siaf_ap_ref.AperName
        if ('NRCAS' in apname) or ('NRCBS' in apname) or ('NRCALL' in apname):
            apname = det_id + '_FULL'
        siaf_ap_obs = siaf_nrc[apname]

    xoff_idl, yoff_idl = xyoff_idl
    v2_ref, v3_ref = siaf_ap_ref.convert(xoff_idl, yoff_idl, 'idl', 'tel')
    att = rotations.attitude(v2_ref, v3_ref, ra, dec, pa_v3)
    # Set attitude correction matrices for each aperture
    siaf_ap_ref.set_attitude_matrix(att)
    siaf_ap_obs.set_attitude_matrix(att)
    # Calculate their ra/dec reference locations
    ra_ref, dec_ref = siaf_ap_ref.reference_point('sky')
    ra_obs, dec_obs = siaf_ap_obs.reference_point('sky')

    # Possible Exposure Types:
    #   NRC_DARK, NRC_FLAT, NRC_LED, NRC_FOCUS, 
    #   NRC_TACQ, NRC_TACONFIRM
    #   NRC_IMAGE, NRC_CORON, NRC_GRISM
    #   NRC_WFSS, NRC_TSIMAGE, NRC_TSGRISM
    if time_series:
        visit_mode = 'ts_grism' if 'GRISM' in pupil else 'ts_imaging'
    else:
        visit_mode = 'wfss' if 'GRISM' in pupil else 'imaging'
    exp_type = get_exp_type(visit_type, visit_mode, pupil)

    if int_range is None:
        integration_start = 1
        integration_end   = ma.nint
        nint_seg          = ma.nint
    else:
        integration_start = int_range[0] + 1
        integration_end   = int_range[1]
        nint_seg = int_range[1] - int_range[0]

    # Start time for integrations considered in this segment
    start_time_string = date_obs + 'T' + time_obs
    # Time offset of exposure start relative to beginning of program execution
    # t_offset_sec = det.time_total_int1 + time_exp_offset
    # if integration_start>1:
    #     t_offset_sec += (integration_start-2) * det.time_total_int2
    # t_offset_sec = (integration_start-1) * det.time_total_int1 + time_exp_offset

    if integration_start==1:
        t_offset_sec = det.time_total_int1 + (nint_seg-1) * det.time_total_int2 + time_exp_offset
    else:
        t_offset_sec = nint_seg * det.time_total_int2 + time_exp_offset
    start_time_int = Time(start_time_string) + t_offset_sec*u.second

    ramptime = det.time_total_int1 if det.time_total_int2==0 else det.time_total_int2
    group_times = populate_group_table(start_time_int, det.time_group, ramptime, 
                                       nint_seg, det.multiaccum.ngroup, det.xpix, det.ypix)

    # Create a dummy observation ID
    obs_id_info = jw_obs_id(1337, 1, 1, 1, 1, 1, 1)

    obs_params = {
        # Proposal info
        'pi_name'          : 'UNKNOWN',
        'title'            : 'UNKNOWN',
        'category'         : 'UNKNOWN',
        'sub_category'     : 'UNKNOWN',
        'science_category' : 'UNKNOWN',

        # Target info
        'target_name'  : 'UNKNOWN',
        'catalog_name' : 'UNKNOWN',
        'ra'           : ra,            # Target RA
        'dec'          : dec,           # Target Dec

        # Pointing info
        'pa_v3'        : pa_v3,         # Telescope position angle relative to V3
        'roll_offset'  : 0,             # Roll angle relative to nominal V3 PA
        'solar_elong'  : sol_elong,     # Solar elongation (deg)
        'pitch_ang'    : pitch_ang,     # Telescope pitch angle relative to sun (deg)
        'siaf_ap'      : siaf_ap_obs,   # Observed SIAF aperture
        'ra_obs'       : ra_obs,        # RA of observered SIAF aperture ref location
        'dec_obs'      : dec_obs,       # Dec of observed SIAF aperture ref location
        'siaf_ap_ref'  : siaf_ap_ref,   # Reference SIAF aperture
        'ra_ref'       : ra_ref,        # RA of reference SIAF aperture ref location
        'dec_ref'      : dec_ref,       # Dec of reference SIAF aperture ref location

        # Observation info
        'date-obs'      : date_obs,
        'time-obs'      : time_obs,
        'obs_id_info'   : obs_id_info,
        'obs_label'     : 'UNKNOWN',
        # Exposure Start time relative to TIME-OBS (seconds)
        'texp_start_relative'  : time_exp_offset,
        # Visit start offset from date/time-obs
        'visit_start_relative' : time_visit_offset,

        # Instrument configuration
        'det_obj'    : det,
        'module'     : det.module,
        'channel'    : 'LONG' if 'LW' in det.channel else 'SHORT', 
        'detector'   : det.detname,
        'filter'     : filt,
        'pupil'      : pupil,
        'coron_mask' : mask,
        # Observation Type
        'visit_level': visit_level,  # e.g., TAREGT, DITHER, FILTER
        'visit_type' : visit_type,   # e.g., T_ACQ, CONFIRM, SCIENCE
        'visit_mode' : visit_mode,
        'exp_type'   : exp_type,
        # Subarray
        'subarray_name' : 'UNKNOWN',

        # subarray_bounds indexed to zero, but values in header should be indexed to 1.
        'xstart'   : det.x0+1 if det.fastaxis>0 else 2048-det.x0-det.xpix+1,
        'ystart'   : det.y0+1 if det.slowaxis>0 else 2048-det.y0-det.ypix+1,
        'xsize'    : det.xpix,
        'ysize'    : det.ypix,   
        'fastaxis' : det.fastaxis,
        'slowaxis' : det.slowaxis,
        'noutputs' : det.nout, 

        # MULTIACCUM
        'readpatt'           : det.multiaccum.read_mode,
        'nframes'            : det.multiaccum.nf,
        'ngroups'            : det.multiaccum.ngroup,
        'nints'              : det.multiaccum.nint,
        'sample_time'        : int(1e6/det._pixel_rate),
        'frame_time'         : det.time_frame,
        'group_time'         : det.time_group,
        'groupgap'           : det.multiaccum.nd2,
        'nresets1'           : det.multiaccum.nr1,
        'nresets2'           : det.multiaccum.nr2,
        'integration_time'   : det.time_int,
        'exposure_time'      : det.time_exp,
        'tint_plus_overhead' : det.time_total_int1,
        'texp_plus_overhead' : det.time_total,

        # Create INT_TIMES table, to be saved in INT_TIMES extension
        # Currently, this is all integrations within the exposure, despite segment
        'int_times' : det.int_times_table(date_obs, time_obs, offset_seconds=time_exp_offset),
        'integration_start' : integration_start,
        'integration_end'   : integration_end,
        # Group times only populate for the current 
        'group_times'       : group_times,

        # Dither information defaults (update later)
        'pridith_pattern_type'  : 'NONE',       # Primary dither pattern name
        'pridith_points_packing': None,         # Primary dither points and packing
        'pridith_npoints'       : 1,            # Number of points in primary dither pattern
        'position_number'       : 1,            # Primary dither position number
        'pattern_start'         : 1,            # Starting point in pattern (???)
        'total_points'          : 1,            # Total number of points in pattern
        'dither_points'         : 1,            # Number of points in image dither pattern
        'pattern_size'          : 'DEFAULT',    # Primary dither pattern size 
        'sgd_pattern'           : 'NONE',       # Small grid dither pattern name
        'subpixel_number'       : 1,            # Subpixel dither position number
        'subpixel_total_points' : 1,            # Total number of subpixel dither positions
        'subpixel_pattern'      : 'STANDARD',   # Subpixel dither pattern name
        'x_offset'              : xoff_idl,     # Dither pointing offset from starting position in x (arcsec)
        'y_offset'              : yoff_idl,     # Dither pointing offset from starting position in y (arcsec)
    }

    if segNum is not None:
        obs_params['EXSEGNUM'] = segNum
        obs_params['EXSEGTOT'] = segTot
    else:
        obs_params['EXSEGNUM'] = None
        obs_params['EXSEGTOT'] = None
    
    # Add file path parameters
    obs_params['filename'] = filename
    obs_params['save_dir'] = None

    # Add any kwargs to final dictionary output
    res = {**obs_params, **kwargs}
    return res

def populate_obs_params(visit_dict, exp_id, detname, date_obs, time_obs='12:00:00.000', 
                        pa_v3=None, segNum=None, segTot=None, int_range=None, det=None, 
                        obs_params=None, grp_id=1, seq_id=1, act_id='01', **kwargs):

    """ Create obs_params from visit dictionary

    An obs_params dictionary is used to create a jwst data model (e.g., Level1bModel).
    If passing ``obs_params`` parameter, this gets updated based on the input arguments.
    Additional ``**kwargs`` will add/update elements to the final output dictionary
    
    Parameters
    ==========
    visit_dict : dict
        Uses gen_all_apt_visits() to create a dictionary of visit information.
        Each visit has a series of exposure IDs.
    exp_id : int
        Unique exposure ID generate observations
    detname : str
        Options NRC[A/B][1-5]
    data_obs : str
        YYYY-MM-DD
    time_obs : str
        HH:MM:SS

    Keyword Arg
    ===========
    pa_v3 : float or None
        Option to specify telescope V3 position angle. If not set, then
        automatically calculated from RA/Dec and observation date/time.
    segNum : int
        The segment number of the current product. Only for TSO.
    segTot : int
        The total number of segments. Only for TSO.
    int_range : list
        Integration indices to use 
    obs_params : dict
        An initial ``obs_params`` dictionary. Any duplicate keywords will be
        updated.
    """

    from .dms import DMS_filename

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    # Ensure standardized detector naming convention ("NRC[A/B][1-5]")
    det_id = get_detname(detname)

    # Build detector operations class
    if det is None:
        det = create_det_class(visit_dict, exp_id, det_id, 
                               grp_id=grp_id, seq_id=seq_id, act_id=act_id)
    
    # Grab indices for specified exposure number
    obs_dict_arr = visit_dict['obs_id_info']
    exp_ids = np.array([int(d['exposure_number']) for d in obs_dict_arr])
    grp_ids = np.array([int(d['visit_group'])     for d in obs_dict_arr])
    seq_ids = np.array([int(d['sequence_id'])     for d in obs_dict_arr])
    act_ids = np.array([    d['activity_id']      for d in obs_dict_arr])

    nexp = len(exp_ids)
    ind_mask = (exp_ids == int(exp_id)) & (grp_ids == int(grp_id)) & \
               (seq_ids == int(seq_id)) & (act_ids == act_id)
    
    # Dictionary info of observation, visit, exposure, etc
    obs_id_info   = visit_dict['obs_id_info'][ind_mask][0]
    obs_label     = visit_dict['obs_label'][ind_mask][0]

    # Get siaf aperture names
    # First, do reference aperture
    exp_apnames = visit_dict['aperture'][ind_mask]
    if exp_apnames.size==1:
        apname = visit_dict['aperture'][ind_mask][0]
    else:
        ind_det = (visit_dict['detectors'][ind_mask] == det_id)
        apname = visit_dict['aperture'][ind_mask][ind_det][0]
    siaf_ap_ref = siaf_nrc[apname]

    # Next, update observed aperture
    if ('NRCAS' in apname) or ('NRCBS' in apname) or ('NRCALL' in apname):
        # Account for coronagraphic masks in engineering templates
        # No SIAF aperture for B Module
        sw_pupil = visit_dict['sw_pupils'][ind_mask][0]
        lw_pupil = visit_dict['lw_pupils'][ind_mask][0]
        if ('A5' in det_id) and lw_pupil=='MASKRND':
            apname = 'NRCA5_FULL_MASK335R'
        elif ('A5' in det_id) and lw_pupil=='MASKBAR':
            apname = 'NRCA5_FULL_MASKLWB'
        elif ('A2' in det_id) and ('MASK' in sw_pupil):
            apname = 'NRCA2_FULL_MASK210R'
        elif ('A4' in det_id) and ('MASK' in sw_pupil):
            apname = 'NRCA4_FULL_MASKSWB'
        else:
            apname = det_id + '_FULL'
    
        #  Check if any of the above apertures throws an exception
        try:
            _ = siaf_nrc[apname]
        except:
            apname_new = det_id + '_FULL'
            _log.error(f'{apname} does not exist. Falling back to {apname_new}.')
            apname = apname_new
    siaf_ap = siaf_nrc[apname]

    # Subarray name
    subarray_name = visit_dict['subarray_name'][ind_mask][0]

    # Make sure det_id exists in exposure
    detectors_all = visit_dict['detectors'][ind_mask][0]
    if det_id not in detectors_all:
        raise ValueError(f'{det_id} not a requested observation. Valid values: {detectors_all}')
    
    # target_name = visit_dict['targ2'][ind_mask][0]
    # ra          = visit_dict['ra'][ind_mask][0]
    # dec         = visit_dict['dec'][ind_mask][0]

    # Get target name and RA/Dec from 
    target_name = visit_dict['TargetID'][ind_mask][0]
    ra_str      = visit_dict['TargetRA'][ind_mask][0]
    dec_str     = visit_dict['TargetDec'][ind_mask][0]
    coords = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), 
                      frame='icrs', equinox='J2000', obstime='J2000')
    ra = coords.ra.deg
    dec = coords.dec.deg

    # Instrument config
    filt_key = f'{det.channel.lower()}_filters' 
    pupil_key = f'{det.channel.lower()}_pupils' 
    filt  = visit_dict[filt_key][ind_mask][0]
    pupil = visit_dict[pupil_key][ind_mask][0]
    mask  = visit_dict['coron_mask'][ind_mask][0]
    
    # Possible Exposure Types:
    #   NRC_DARK, NRC_FLAT, NRC_LED,
    #   NRC_TACQ, NRC_TACONFIRM, 
    #   NRC_IMAGE, NRC_GRISM, NRC_CORON,
    #   NRC_TSIMAGE, NRC_TSGRISM, 
    #   NRC_FOCUS, NRC_WFSS
    level_val = visit_dict['level'][ind_mask][0]
    type_val  = visit_dict['type'][ind_mask][0]
    mode_val  = visit_dict['mode'][ind_mask][0]
    time_series = True if 'ts' in mode_val.lower() else False
    exp_type = get_exp_type(type_val, mode_val, pupil)
                            
    # Dither offset position
    xoffset = visit_dict['idlx'][ind_mask][0]
    yoffset = visit_dict['idly'][ind_mask][0]
    ddist   = visit_dict['ddist'][ind_mask][0]

    # Dither information
    pri_dith_type = visit_dict['PrimaryDitherType'][ind_mask][0]
    pri_dith_size = visit_dict['DitherSize'][ind_mask][0]
    sub_dith_type = visit_dict['SubpixelDitherType'][ind_mask][0]
    sgd_type      = visit_dict['SmallGridDitherType'][ind_mask][0]
    dither_points = visit_dict['number_of_dithers'][ind_mask][0]
    try: 
        pri_dithers = visit_dict['PrimaryDithers'][ind_mask][0] # e.g., '3TIGHT'
    except ValueError: 
        pri_dithers = NONE_STR
    try: 
        npri_dith = int(visit_dict['PrimaryDithers'][ind_mask][0])
    except ValueError: 
        try: 
            npri_dith = int(visit_dict['PrimaryDithers'][ind_mask][0][0])
        except ValueError: 
            npri_dith = NONE_STR
    try: nsub_dith = int(visit_dict['SubpixelPositions'][ind_mask][0])
    except ValueError: nsub_dith = 1
    pri_pos_num = int(visit_dict['pri_dith'][ind_mask][0])
    sub_pos_num = int(visit_dict['sub_dith'][ind_mask][0])

    # # Certain dither info should be None if string is 'NONE'
    # if (pri_dithers is not None) and (pri_dithers.upper()=='NONE'):
    #     pri_dithers = None
    # if (sub_dith_type is not None) and (sub_dith_type.upper()=='NONE'):
    #     sub_dith_type = None
    # if (sgd_type is not None) and (sgd_type.upper()=='NONE'):
    #     sgd_type = None

    # Create output filename
    filename = DMS_filename(obs_id_info, det.detname, segNum=segNum, prodType='uncal')

    # Exposure start times offset
    time_exp_offset = visit_dict['exp_start_times'][ind_mask][0]
    # Visit start time
    time_visit_offset = visit_dict['visit_start']

    # Ensure 3 decimal points in time_obs
    try:
        # Split off decimal point after string
        # Add a bunch of zeros, then truncate
        tsplit = time_obs.split('.')
        sub_sec = (tsplit[1] + '000')[:3]
        time_obs = '.'.join([tsplit[0], sub_sec])
    except:
        # If no decimal, just add zeros
        time_obs = time_obs + '.000'

    # Create intial obs_params dictionary
    obs_params_init = create_obs_params(filt, pupil, mask, det, siaf_ap_ref, (ra, dec), 
        date_obs, time_obs=time_obs, pa_v3=pa_v3, siaf_ap_obs=siaf_ap, xyoff_idl=(xoffset,yoffset), 
        visit_level=level_val, visit_type=type_val, time_series=time_series, 
        time_visit_offset=time_visit_offset, time_exp_offset=time_exp_offset, 
        segNum=segNum, segTot=segTot, int_range=int_range, filename=filename)
    
    # Update V3 PA with Orient Ranges special requirement
    # Update V3 PA with roll information
    orient_info = visit_dict.get('orient_info')
    roll_info = visit_dict.get('roll_info')
    if (orient_info is not None) or (roll_info is not None):
        pa_v3_input = pa_v3
        pa_v3 = obs_params_init['pa_v3']

        # Make sure we constrain within V3 PA limits within field of regard
        output = get_tel_angles(ra, dec, obs_date=date_obs, obs_time=time_obs)
        if orient_info is not None:
            v3pa_min = np.max([output['v3pa_min'], orient_info['OrientMin']])
            v3pa_max = np.min([output['v3pa_max'], orient_info['OrientMax']])
        else:
            v3pa_min, v3pa_max = (output['v3pa_min'], output['v3pa_max'])

        # If initial input was None, then we have free reign to calculate nominal values
        if pa_v3_input is None:
            # Set min value to negative if it's greater than max value
            if v3pa_min > v3pa_max:
                v3pa_min = v3pa_min - 360
            # Nominal value is then average of min and max constraints
            pa_v3 = (v3pa_min + v3pa_max) / 2
            # Ensure PA is 0-360
            if pa_v3 < 0:
                pa_v3 += 360
        # Otherwise, check if specified pa_v3 is outside of bounds
        elif pa_v3 < v3pa_min:
            _log.warning(f'{pa_v3:.2f} is less than allowed {v3pa_min:.2f} for the given date!')
        elif pa_v3 > v3pa_max:
            _log.warning(f'{pa_v3:.2f} is greater than allowed {v3pa_max:.2f} for the given date!')

    if roll_info is not None:
        obs_num = visit_dict['obs_num']

        dpa_max = np.abs([roll_info['MinPA'], roll_info['MaxPA']]).max()
        if roll_info['MaxPA'] < 0:
            dpa_max *= -1

        # Add or subtract max roll depending obs
        if obs_num==roll_info['PrimaryObs']:
            pa_v3 = pa_v3 + dpa_max / 2
        elif obs_num==roll_info['OrientFromObs']:
            pa_v3 = pa_v3 - dpa_max / 2

        if pa_v3 < v3pa_min:
            pa_v3 = v3pa_min
        elif pa_v3 > v3pa_max:
            pa_v3 = v3pa_max

        # Update Roll and V3 PA
        obs_params_init['roll_offset'] = pa_v3 - obs_params_init['pa_v3']
        obs_params_init['pa_v3'] = pa_v3

        # Update pitch angle (solar elongation stays the same)
        pitch_orig = obs_params_init['pitch_ang']
        roll_ang = obs_params_init['roll_offset']
        vpitch_rad = np.deg2rad(pitch_orig)
        vroll_rad = np.deg2rad(roll_ang)
        pitch_new = np.rad2deg(np.arctan(np.tan(vpitch_rad) / np.cos(vroll_rad)))
        del_pitch = np.abs(pitch_orig - pitch_new)
        if roll_ang>0:
            pitch_new += del_pitch
        else:
            pitch_new -= del_pitch
        obs_params_init['pitch_ang'] = pitch_new


    obs_params_temp = {
        # Proposal info
        'pi_name'          : 'UNKNOWN',
        'title'            : 'UNKNOWN',
        'category'         : 'UNKNOWN',
        'sub_category'     : 'UNKNOWN',
        'science_category' : 'UNKNOWN',

        # Target info
        'target_name'  : target_name,
        'catalog_name' : 'UNKNOWN',

        # Observation info
        'obs_id_info'   : obs_id_info,
        'obs_label'     : obs_label,
        'exp_type'      : exp_type,
        'subarray_name' : subarray_name,
        'nexposures'    : nexp,   # Total number of planned exposures in visit

        # Dither information defaults (update later)
        'pridith_pattern_type'  : pri_dith_type,# Primary dither pattern name
        'pridith_points_packing': pri_dithers,  # Primary dither points and packing
        'pridith_npoints'       : npri_dith,    # Number of points in primary dither pattern
        'position_number'       : pri_pos_num,  # Primary dither position number
        'pattern_start'         : 1,            # Starting point in pattern (???)
        'total_points'          : dither_points,# Total number of points in pattern
        'dither_points'         : dither_points,# Number of points in image dither pattern
        'pattern_size'          : pri_dith_size,# Primary dither pattern size 
        'sgd_pattern'           : sgd_type,     # Small grid dither pattern name
        'subpixel_number'       : sub_pos_num,  # Subpixel dither position number
        'subpixel_total_points' : nsub_dith,    # Total number of subpixel dither positions
        'subpixel_pattern'      : sub_dith_type,# Subpixel dither pattern name
        'x_offset'              : xoffset,      # Dither pointing offset from starting position in x (arcsec)
        'y_offset'              : yoffset,      # Dither pointing offset from starting position in y (arcsec)
        'ddist'                 : ddist,        # Movement relative to previous exposure (arcsec)
    }

    obs_params_temp = {**obs_params_init, **obs_params_temp}

    # Merge dictionaries
    # obs_params => obs_params_temp => kwargs
    if obs_params is None:
        obs_params = {}
    res = {**obs_params, **obs_params_temp, **kwargs}

    return res

def file_segmenting(det, max_size_MB=320):

    ma = det.multiaccum

    # Determine number of segments to break FITS file into
    npix_int = det.xpix * det.ypix * ma.nread_tot
    int_size_MB = 2 * npix_int / (1024*1024)  # 2 bytes/pix
    nint_max = np.max([int(max_size_MB / int_size_MB), 1])

    # Number of FITS segments to split
    nseg = ma.nint / nint_max
    nseg = int(nseg) if nseg.is_integer() else int(nseg)+1

    # Number of ints in each segment
    nint_per_seg = ma.nint / nseg
    nint_per_seg = int(nint_per_seg) if nint_per_seg.is_integer() else int(nint_per_seg)+1

    # Built integer indices
    iseg1 = np.arange(nseg)*nint_per_seg
    iseg2 = iseg1 + nint_per_seg
    iseg2[-1] = ma.nint
    iseg_list = [(i1,i2) for i1,i2 in zip(iseg1,iseg2)]

    # print(nseg, nint_per_seg)

    return iseg_list



class DMS_input():
    """
    Class to generate a series of observation dictionaries in order
    to build DMS-like files. Loads APT files to generate the 
    necessary observation information.

    Usage:

        json_file     = 'NRC-21.timing.json'
        sm_acct_file  = 'NRC-21.smart_accounting'
        pointing_file = 'NRC-21.pointing'
        xml_file      = 'NRC-21.xml'

        obs_input = DMS_input(xml_file, pointing_file, json_file, sm_acct_file)
        # Update observation time and telescope PA
        obs_input.obs_date = '2022-03-01'
        obs_input.obs_time = '12:00:00.000'
        obs_input.pa_v3 = 45

        # Create a list of observation parameters
        obs_params_all = obs_input.gen_all_obs_params()

        # Select one of the parameter dictionaries
        obs_params = obs_params_all[0]

        # Create a series of science data based on observation params
        # that results in sci_data and zero_data 16-bit numpy arrays
        out_model = level1b_data_model(obs_params, sci_data, zero_data)

        # Save to FITS file
        # Performs minor updates to the saved FITS file
        save_level1b_fits(out_model, obs_params, save_dir)
    """

    def __init__(self, xml_file, pointing_file, json_file, sm_acct_file, save_dir=None, 
                 obs_date='2022-03-01', obs_time='12:00:00', pa_v3=None, rand_seed_init=None):

        self.files = {
            'xml_file'      : xml_file,
            'pointing_file' : pointing_file,
            'json_file'     : json_file,
            'sm_acct_file'  : sm_acct_file,
        }

        self.proposal_info = get_proposal_info(xml_file)
        # Series of dictionaries, one per visit
        # Dictionaries are ordered according to smart accounting order information
        self.program_info = gen_all_apt_visits(xml_file, pointing_file, sm_acct_file, json_file, 
                                               rand_seed=rand_seed_init)

        # Create unique labels for each exposure
        self.labels = self._gen_obs_labels()

        self._obs_date = obs_date
        self._obs_time = obs_time
        self._pa_v3 = pa_v3

        self.save_dir = save_dir

    @property
    def obs_date(self):
        """Date of observations"""
        if self._obs_date is None:
            return '2022-03-01'
        else:
            return self._obs_date
    @obs_date.setter
    def obs_date(self, value):
        self._obs_date = value

    @property
    def obs_time(self):
        """Start time of observations"""
        if self._obs_time is None:
            return '12:00:00.000'
        else:
            return self._obs_time
    @obs_time.setter
    def obs_time(self, value):
        self._obs_time = value

    @property
    def pa_v3(self):
        return self._pa_v3
    @pa_v3.setter
    def pa_v3(self, value):
        self._pa_v3 = value
        
    def _gen_label(self, visit_id, exp_id, det_id, grp_id, seq_id, act_id):
        "Create unique label ID"
        detname = get_detname(det_id)
        return f'{visit_id}_{exp_id}_{detname}_{grp_id}_{seq_id}_{act_id}'

    def _gen_obs_labels(self):
        """Create unique label for each observation"""
        
        labels = []
        
        visit_ids = self.program_info.keys()
        for vid in visit_ids:
            visit_dict = self.program_info[vid]
            exp_ids = np.array([int(d['exposure_number']) for d in visit_dict['obs_id_info']])
            grp_ids = np.array([int(d['visit_group'])     for d in visit_dict['obs_id_info']])
            seq_ids = np.array([int(d['sequence_id'])     for d in visit_dict['obs_id_info']])
            act_ids = np.array([    d['activity_id']      for d in visit_dict['obs_id_info']])
            # For each exposure id, get detector id
            for i in range(len(exp_ids)):
                det_ids = visit_dict['detectors'][i]
                eid = exp_ids[i]
                gid = grp_ids[i]
                sid = seq_ids[i]
                aid = act_ids[i]
                for detname in det_ids:
                    label = self._gen_label(vid, eid, detname, gid, sid, aid)
                    labels.append(label)
                    
        return labels

    def _parse_label(self, label):
        """Parse label to get visit_id, exp_id, det_id, grp_id, seq_id, act_id"""
        
        # visit_id, exp_id, det_id = label.split('_')
        return label.split('_')
        
    def gen_obs_params(self, visit_id, exp_id, det_id, det=None, 
                       seg_num=None, seg_tot=None, int_range=None,
                       grp_id=1, seq_id=1, act_id='01'):
        """Generate a single set of observation parameters for a given exposure"""

        # Visit dictionary
        visit_dict = self.program_info[visit_id]

        kwargs = self.proposal_info
        date_obs = self.obs_date
        time_obs = self.obs_time
        pa_v3 = self.pa_v3

        # observation id information
        kwargs['grp_id'] = grp_id
        kwargs['seq_id'] = seq_id
        kwargs['act_id'] = act_id
        act_int = int(act_id, 36) # Convert base 36 to integer number

        # Populate random seed information
        nexp = len(visit_dict['obs_id_info'])
        rand_seed_noise = visit_dict.get('rand_seed_noise')
        if rand_seed_noise is not None:
            rand_seed_noise_j = rand_seed_noise + int(grp_id)*act_int*nexp + int(exp_id)
            kwargs['rand_seed_noise'] = rand_seed_noise_j
        kwargs['rand_seed_init']  = visit_dict.get('rand_seed_init')
        kwargs['rand_seed_dith']  = visit_dict.get('rand_seed_dith')

        detname = get_detname(det_id)
        res = populate_obs_params(visit_dict, exp_id, detname, date_obs, time_obs=time_obs, pa_v3=pa_v3, 
                                  det=det, segNum=seg_num, segTot=seg_tot, int_range=int_range, **kwargs)
        res['visit_key'] = visit_id
        res['save_dir'] = self.save_dir

        return res
    
    # def gen_obs_param(self, visit_id, exp_id, det_id, det=None, 
    #                    seg_num=None, seg_tot=None, int_range=None):
    #     """Generate a single set of observation parameters for a given exposure"""

    #     args = (visit_id, exp_id, det_id)
    #     kwargs = {'det': det, 'seg_num': seg_num, 'seg_tot': seg_tot, 'int_range': int_range}
    #     return self.gen_obs_params(*args, **kwargs)

    def gen_all_obs_params(self):
        """Generate a full set of parameters for all exposures"""
        
        obs_params_all = []
        
        all_labels = self.labels
        for label in tqdm(all_labels, desc='Obs Params', leave=False):
            visit_id, exp_id, det_id, grp_id, seq_id, act_id = self._parse_label(label)
            visit_dict = self.program_info[visit_id]
            det = create_det_class(visit_dict, exp_id, det_id, 
                                   grp_id=grp_id, seq_id=seq_id, act_id=act_id)
            
            # Get FITS segmenting
            iseg_list = file_segmenting(det)
            seg_tot = len(iseg_list)
            if seg_tot>1:
                for ii in range(seg_tot):
                    obspar = self.gen_obs_params(visit_id, exp_id, det_id, det=det, 
                                                 seg_num=ii, seg_tot=seg_tot, int_range=iseg_list[ii],
                                                 grp_id=grp_id, seq_id=seq_id, act_id=act_id)
                    obs_params_all.append(obspar)
            else:
                obspar = self.gen_obs_params(visit_id, exp_id, det_id, det=det,
                                             grp_id=grp_id, seq_id=seq_id, act_id=act_id)
                obs_params_all.append(obspar)
                
        return obs_params_all

    def gen_pitch_array(self, nvals=1000, pitch_init=None):

        f1 = self.files['xml_file']
        f2 = self.files['pointing_file']
        f3 = self.files['json_file']
        f4 = self.files['sm_acct_file']

        return pitch_vs_time(f1, f2, f3, f4, obs_date=self.obs_date, obs_time=self.obs_time, 
            pitch_init=pitch_init, nvals=nvals)

    def make_jwst_point(self, visit_id, exp_id, detname, obs_params=None, 
        base_std=0, dith_std=0, rand_seed=None, grp_id=1, seq_id=1, act_id='01'):
        """Create jwst_point object
        

        Parameters
        ==========
        visit_id : str
            obsnum:visitnum, for example: '007:001' 
        exp_id : int
            Exposure number
        detname : str
            Name of detector, such as 'NRCA5'.

        Keyword Args
        ============
        obs_params : dict
            Option to pass any already generated obs_params dictionary
            rather than generating it automatically. Should be part of
            the associated visit.
        base_std : float or None
            The 1-sigma pointing uncertainty per axis for telescope slew. 
            If None, then standard deviation is chosen to be either 5 mas 
            or 100 mas, depending on `use_ta` attribute (default: True).
        dith_std : float or None
            The 1-sigma pointing uncertainty per axis for dithers. If None,
            then standard deviation is chosen to be either 2.5 or 5 mas, 
            depending on `use_sgd` attribute (default: True).
        rand_seed : None or int
            Random seed to use for generating repeatable random offsets.
        """

        # Visit dictionary
        visit_dict = self.program_info[visit_id]

        # Create a single observation parameter dictionary if not passed
        if obs_params is None:
            obs_params = self.gen_obs_params(visit_id, exp_id, detname,
                                             grp_id=grp_id, seq_id=seq_id, act_id=act_id)

        # Filter by type value (SCIENCE, T_ACQ, CONFIRM)
        type_arr = visit_dict['type']
        if obs_params['visit_type'] == 'SCIENCE':
            ind = (type_arr == obs_params['visit_type'])
        else:
            # Get exposure IDs
            obs_dict_arr = visit_dict['obs_id_info']
            exp_ids = np.array([d['exposure_number'] for d in obs_dict_arr])
            ind = (exp_ids == obs_params['obs_id_info']['exposure_number'])

        # Ensure detector name is valid
        detname = get_detname(detname)
        det_ids = visit_dict['detectors'][ind]
        if detname not in det_ids:
            raise ValueError(f'{detname} not valid for Visit {visit_id}.')

        return gen_jwst_pointing(visit_dict, obs_params, base_std=base_std, dith_std=dith_std, rand_seed=rand_seed)


def gen_pointing_info(*args, **kwargs):
    """
    **Deprecated. Use :func:`gen_jwst_pointing` instead.**
    Create telescope pointing sequence for a given visit / exposure.
    """

    _log.warning("Deprecated. Use `gen_jwst_pointing` function instead in the future.")
    return gen_jwst_pointing(*args, **kwargs)

def gen_jwst_pointing(visit_dict, obs_params, base_std=None, dith_std=None, 
                      rand_seed=None, rand_seed_base=None, **kwargs):
    """
    Create telescope pointing sequence for a given visit / exposure.

    Keyword Args
    ============
    base_std : float or array-like or None
        The 1-sigma pointing uncertainty per axis for telescope slew. 
        If None, then standard deviation is chosen to be either 5 mas 
        or 100 mas, depending on `use_ta` attribute (default: True).
    dith_std : float or array-like or None
        The 1-sigma pointing uncertainty per axis for dithers. If None,
        then standard deviation is chosen to be either 2.5 or 5 mas, 
        depending on `use_sgd` attribute (default: True).
    rand_seed : None or int
        Random seed to use for generating repeatable random offsets.
    rand_seed_base : None or int
        Use a separate random seed for telescope slew offset.
        Then, `rand_seed` corresponds only to relative dithers.
        Useful for multiple exposures with same initial slew, but
        independent dither pattern realizations. 
    """
    
    # Reference aperture (e.g., NRCALL_FULL) for telescope pointing
    ap_ref_name = obs_params['siaf_ap_ref'].AperName
    # Observed aperture (e.g., NRCA5_FULL) for detector
    ap_obs_name = obs_params['siaf_ap'].AperName
    # Reference RA/Dec of aperture prior to offseting/dithering
    ra_ref, dec_ref = (obs_params['ra'], obs_params['dec']) 
    pos_ang = obs_params['pa_v3']

    # Filter by type value (SCIENCE, T_ACQ, CONFIRM)
    type_arr = visit_dict['type']
    if obs_params['visit_type'] == 'SCIENCE':
        ind = (type_arr == obs_params['visit_type'])
        base_offset  = (visit_dict['basex'][ind][0], visit_dict['basey'][ind][0])
        dith_offsets = list(zip(visit_dict['dithx'][ind], visit_dict['dithy'][ind]))
    else:
        # For T_ACQ and CONFIRM, index by individual exposure ID
        obs_dict_arr = visit_dict['obs_id_info']
        exp_ids = np.array([d['exposure_number'] for d in obs_dict_arr])
        ind = exp_ids == obs_params['obs_id_info']['exposure_number']
        base_offset  = (visit_dict['basex'][ind][0], visit_dict['basey'][ind][0])
        dith_offsets = [(visit_dict['dithx'][ind][0], visit_dict['dithy'][ind][0])]

    if rand_seed is None:
        rand_seed = visit_dict.get('rand_seed_dith')

    tel_pointing = jwst_point(ap_obs_name, ap_ref_name, ra_ref, dec_ref, pos_ang=pos_ang, 
                              base_offset=base_offset, dith_offsets=dith_offsets, 
                              base_std=base_std, dith_std=dith_std, 
                              rand_seed=rand_seed, rand_seed_base=rand_seed_base)

    # Standard SAM or SGD
    subpixel_pattern = obs_params.get('subpixel_pattern', 'NONE')
    tel_pointing.use_sgd = True if 'SMALL-GRID-DITHER' in subpixel_pattern.upper() else False
    
    return tel_pointing


def get_tel_angles(ra, dec, obs_date='2022-03-01', obs_time='12:00:00'):
    """
    For a given RA, Dec and date, return the nominal telescope pitch and V3 PA angles.
    """
    
    import datetime
    from .skyvec2ins import skyvec2ins

    # Create datetime object
    args1 = np.array(obs_date.split('-'))
    args2 = np.array(obs_time.split(':'))
    try:
        sec_split = args2[-1].split('.')
        args2[-1] = sec_split[0]
        args3 = (np.array([sec_split[1]]).astype('float')*1e5).astype('int')
    except:
        args3 = np.array(['0'])
    args = np.concatenate((args1, args2, args3)).astype('int')
    time_obs = datetime.datetime(*args)
    
    # Get array of information
    res = skyvec2ins(ra, dec, time_obs, npoints=3, nrolls=100)
    mask_obs = res[1].astype('bool') # Observabilitiy mask
    elong_deg = np.rad2deg(res[2])   # Solar elongation
    v3pa_deg = np.rad2deg(res[3])    # V3 Position Angle
    pitch_deg = elong_deg - 90       # Telescope pitch angles (valid from -5 to 45)
    
    pitch = pitch_deg[0,0]
    mask = mask_obs[:,0] 
    if mask.sum()==0:
        _log.warning(f"Source RA, Dec = ({ra:.1f}, {dec:.1f}) deg is not visible on {obs_date}!")
        v3pa_lims = np.array([v3pa_deg[0,0], v3pa_deg[-1,0]])
    else:
        v3pa_masked = v3pa_deg[mask,0]
        v3pa_lims = np.array([v3pa_masked[0], v3pa_masked[-1]])

    # Check if max-min is too large, indicating min should actually be negative
    # This might be a case where the V3PA range is something like 355 to 5.
    if np.abs(v3pa_lims[0]-v3pa_lims[1]) > 30:
        v3pa_max = v3pa_lims.min()
        v3pa_min = v3pa_lims.max() - 360
    else:
        v3pa_min = v3pa_lims.min()
        v3pa_max = v3pa_lims.max()

    v3pa = (v3pa_max + v3pa_min) / 2

    return {'pitch_deg': pitch, 'v3pa_deg': v3pa, 
            'v3pa_min': v3pa_min,'v3pa_max': v3pa_max}

def pitch_vs_time(xml_file, pointing_file, timing_json_file, smart_accounting_file,
    obs_date='2022-03-01', obs_time='12:00:00', pitch_init=None, nvals=1000):
    
    timing_info = get_timing_info(timing_json_file, smart_accounting_file)
    pointing_info = get_pointing_info(pointing_file, all_inst=True)
    roll_info = get_roll_info(xml_file)
    orient_info = get_orient_specreq(xml_file)

    slew_durations  = np.array([timing_info[k]['slew_duration'] for k in timing_info.keys()])
    visit_durations = np.array([timing_info[k]['scheduling_duration'] for k in timing_info.keys()])

    # Get pitch angle for each visit
    pitch_angles = []
    for k in list(timing_info.keys()):
        obs_num, visit_num = np.array(k.split(':')).astype('int')
        ind = (pointing_info['obs_num'] == obs_num) & (pointing_info['visit_num'] == visit_num)
        ra  = pointing_info['ra'][ind][0]
        dec = pointing_info['dec'][ind][0]

        res = get_tel_angles(ra, dec, obs_date=obs_date, obs_time=obs_time)
        # Nominal pitch angle for time and date
        pitch_angle = res['pitch_deg']

        # V3 PA limits
        pa_v3_nom = res['v3pa_deg']
        pa_v3_min, pa_v3_max = (res['v3pa_min'], res['v3pa_max'])

        # print(f'Init (Obs{obs_num}): {pa_v3_nom:.2f} ({pa_v3_min:.2f}, {pa_v3_max:.2f})')

        # Update V3 PA limits if Orient Ranges Special Requirement exists
        if orient_info is not None:
            for kobs in orient_info.keys():
                if int(kobs)==obs_num:
                    orient_min = orient_info[kobs]['OrientMin']
                    orient_max = orient_info[kobs]['OrientMax']
                    if pa_v3_min<0:
                        orient_min = orient_min - 360
                    pa_v3_min = np.max([orient_min, pa_v3_min])
                    pa_v3_max = np.min([orient_max, pa_v3_max])
                    pa_v3_nom = (pa_v3_min + pa_v3_max) / 2

                    # print(f'Orient SpecReq: {pa_v3_nom:.2f} ({pa_v3_min:.2f}, {pa_v3_max:.2f})')

        # Update pitch angles if there is a roll angle
        if roll_info is not None:
            for kroll in roll_info.keys():
                rdict = roll_info[kroll]
                if (obs_num==rdict['PrimaryObs']) or (obs_num==rdict['OrientFromObs']):

                    dpa_max = np.abs([rdict['MinPA'], rdict['MaxPA']]).max()
                    if rdict['MaxPA'] < 0:
                        dpa_max *= -1

                    # Add or subtract max roll depending obs
                    if obs_num==rdict['PrimaryObs']:
                        pa_v3 = pa_v3_nom + dpa_max / 2
                    elif obs_num==rdict['OrientFromObs']:
                        pa_v3 = pa_v3_nom - dpa_max / 2

                    # Make sure we constrain within V3 PA limits within field of regard
                    if pa_v3 < pa_v3_min:
                        pa_v3 = pa_v3_min
                    elif pa_v3 > pa_v3_max:
                        pa_v3 = pa_v3_max

                    # Calculate change in pitch based on roll angle
                    roll_ang = pa_v3 - pa_v3_nom
                    vpitch_rad = np.deg2rad(pitch_angle)
                    vroll_rad = np.deg2rad(roll_ang)
                    pitch_new = np.rad2deg(np.arctan(np.tan(vpitch_rad) / np.cos(vroll_rad)))
                    del_pitch = np.abs(pitch_angle - pitch_new)
                    # Determine positive or negative depending on roll direction
                    if roll_ang>0:
                        pitch_angle += del_pitch
                    else:
                        pitch_angle -= del_pitch

                    # print(f'Roll Link: {pa_v3_nom:.2f} ({pa_v3_min:.2f}, {pa_v3_max:.2f}); {pa_v3:.2f}')

        pitch_angles.append(pitch_angle)

    pitch_angles = np.array(pitch_angles)
    pitch_init = pitch_angles[0] if pitch_init is None else pitch_init

    # Create a timing array broken up into events
    tmax = (visit_durations+slew_durations).cumsum()[-1]
    tarr = np.linspace(0,tmax,nvals)

    # Start time for slew and visits
    slew_start = np.cumsum(slew_durations + visit_durations)
    slew_start = np.concatenate(([0], slew_start[:-1]))
    visit_start = slew_start + slew_durations

    # Fill pitch angle array w.r.t. time
    pitch_arr = np.zeros(len(tarr)) + pitch_init
    for tval, pval in zip(visit_start, pitch_angles):
        ind = tarr>(tval)
        pitch_arr[ind] = pval

    # Assume linear change during slew times
    for i in range(len(pitch_angles)):
        tstart = slew_start[i]
        tend = visit_start[i]
        pitch_start = pitch_init if i==0 else pitch_angles[i-1]
        pitch_end = pitch_angles[i]

        # Linear interpolation
        ind = (tarr>=tstart) & (tarr<=tend)
        pitch_arr[ind] = np.interp(tarr[ind], [tstart,tend], [pitch_start,pitch_end])

    return tarr, pitch_arr, slew_start, visit_start

