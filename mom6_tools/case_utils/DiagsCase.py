import yaml
import os, sys
import logging as log
from collections import namedtuple


DiagFileEntry = namedtuple('DiagFileEntry',
                    ("output_freq", "output_freq_units", "file_format", "time_axis_units",
                     "time_axis_name", "new_file_freq", "new_file_freq_units", "start_time",
                     "file_duration", "file_duration_units"))
DiagFieldEntry = namedtuple('DiagFieldEntry',
                    ("module_name", "output_name", "time_sampling", "reduction_method",
                     "regional_section", "packing"))

class DiagsCase(object,):

    def __init__(self, diag_config_yml_path):

        # initialize members:
        self._cime_case = None

        # load diag_config.yml file:
        self._config = yaml.load(open(diag_config_yml_path,'r'), Loader=yaml.Loader)

        rundir_provided = "RUNDIR" in self._config
        dout_s_root_provided = "DOUT_S_ROOT" in self._config
        caseroot_provided = "CASEROOT" in self._config
        cimeroot_provided = "CIMEROOT" in self._config

        # check if required keywords are in diag_config.yml
        if not (rundir_provided or dout_s_root_provided):
            assert (caseroot_provided and cimeroot_provided),\
                    "If 'RUNDIR' or 'DOUT_S_ROOT' are not provided,"\
                    " both 'CASEROOT' and 'CIMEROOT' must be provided."

        # if available, instantiate a cime case object
        if caseroot_provided and cimeroot_provided:
            cimeroot = self._config['CIMEROOT']
            caseroot = self._config['CASEROOT']
            sys.path.append(os.path.join(cimeroot, "scripts", "lib"))
            from CIME.case.case import Case
            self._cime_case = Case(caseroot)

    def get_value(self, var):
        """ Returns the value of a variable in yaml config file. If var is not in yaml config
            file, then checks to see if it can retrive the var from _cime_case instance """

        val = None
        if var in self._config:
            val =  self._config[var]
        elif self._cime_case:
            val = self._cime_case.get_value(var)

        if val.lower() == "none":
            val = None

        log.info(f"get_value - requsted variable: {var}, returning value: {val}")
        return val

    @staticmethod
    def convert_prefix_to_regex(prefix):
        prefix_split = prefix.split('%')

        # first add the pre-prefix:
        regex = prefix_split[0]

        # now add date sections:
        for date_str in prefix_split[1:]:
            nchars  = int(date_str[0])
            regex += f'_\d{{{nchars}}}'

        # add .nc
        regex += '.nc'

        return regex

    def get_file_prefix(self, fld_to_search:str, output_freq=None, output_freq_units=None) -> str:
        """Returns the prefix of file including a given field"""

        # first, determine all the files that include the field
        candidate_files = set()
        for fld_name,file_name in self.diag_fields:
            if fld_to_search==fld_name:
                log.info(f"{fld_to_search}, {fld_name}, {file_name}")
                candidate_files.add(file_name)
        log.info(f"{fld_to_search} found in {candidate_files}")

        # second, determine all the files with unmatcing output frequency
        if output_freq!=None or output_freq_units!=None:
            non_matching_files = set()
            for matched_file in candidate_files:
                if (output_freq and self.diag_files[matched_file].output_freq != output_freq) or\
                   (output_freq_units and self.diag_files[matched_file].output_freq_units != output_freq_units):
                    non_matching_files.add(matched_file)

            # final list of candidate files
            candidate_files -= non_matching_files

        # there must be one matching file only
        if len(candidate_files) == 0:
            raise RuntimeError(f"Cannot find '{fld_to_search}' in diag_table")
        elif len(candidate_files) > 1:
            raise RuntimeError(f"Multiple '{fld_to_search}' entries in diag_table. Provide output frequency!")
        else: # only one file including field found
            pass

        file_prefix = candidate_files.pop()
        log.info(f"returning {file_prefix} including {fld_to_search}")
        return file_prefix



    def _parse_diag_table(self):
        diag_table_path = os.path.join(self.get_value('RUNDIR'), 'diag_table')

        with open(diag_table_path,'r') as diag_table:

            # first read the two header files:
            ctr = 0
            for line in diag_table:
                line = line.strip()
                if len(line)>0 and line[0] != '#': # not an empty line or comment line
                    ctr+=1
                    if ctr==2: break

            # now read the file and field blocks
            self.diag_files = dict()
            self.diag_fields = dict()
            within_file_list = True # if false, within field list
            for line in diag_table:
                line = line.strip()
                line = line.replace("'"," ").replace('"',' ').replace(",","")
                line_split = line.split()
                if len(line)>0 and line[0] != '#': # not an empty line or comment line

                    if len(line)>11 and line[1:12]=="ocean_model":
                        within_file_list = False

                    if within_file_list:
                        file_name = line_split[0]
                        self.diag_files[file_name] = DiagFileEntry(
                            output_freq = line_split[1],
                            output_freq_units = line_split[2],
                            file_format = line_split[3],
                            time_axis_units = line_split[4],
                            time_axis_name = line_split[5],
                            new_file_freq = line_split[6] if len(line_split)>6 else None,
                            new_file_freq_units = line_split[7] if len(line_split)>7 else None,
                            start_time = line_split[8] if len(line_split)>8 else None,
                            file_duration = line_split[9] if len(line_split)>9 else None,
                            file_duration_units = line_split[10] if len(line_split)>10 else None
                        )

                    else: # within field list
                        fld_name = line_split[1]
                        file_name = line_split[3]
                        self.diag_fields[fld_name,file_name] = DiagFieldEntry(
                            module_name = line_split[0],
                            output_name = line_split[2],
                            time_sampling = line_split[4],
                            reduction_method = line_split[5],
                            regional_section = line_split[6],
                            packing = line_split[7])

