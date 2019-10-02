import yaml
import os, sys
import re
import logging as log
from collections import OrderedDict, namedtuple
import xarray as xr


DiagFileEntry = namedtuple('DiagFileEntry',
                    ("output_freq", "output_freq_units", "file_format", "time_axis_units",
                     "time_axis_name", "new_file_freq", "new_file_freq_units", "start_time",
                     "file_duration", "file_duration_units"))
DiagFieldEntry = namedtuple('DiagFieldEntry',
                    ("module_name", "output_name", "time_sampling", "reduction_method",
                     "regional_section", "packing"))

class DiagsCase(object,):

    def __init__(self, case_config:OrderedDict):

        self._config = case_config
        self._cime_case = None
        self.diag_files = None
        self.diag_fields = None

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

        if val != None and val.lower() == "none":
            val = None

        log.info(f"get_value::\n\trequsted variable: {var} \n\treturning value: {val}"\
                 f"\n\ttype: {type(val)}")
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
            raise RuntimeError(f"Multiple '{fld_to_search}' entries in diag_table. Provide output frequency "\
                                "or HIST_FILE_PREFIX!")
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


    def _get_file_list(self, fields:list):

        field0 = fields[0]

        # from diag_table, get prefix of file including field:
        hist_file_prefix = self.get_value("HIST_FILE_PREFIX")
        if hist_file_prefix == None:
            if self.diag_files == None:
                self._parse_diag_table()
            hist_file_prefix = self.get_file_prefix(field0)
            for f in fields[1:]:
                if self.get_file_prefix(f) != hist_file_prefix:
                    raise RuntimeError(f"The following fields are spreaded across multiple "+\
                                        "netcdf files with different prefixes")

        # create a list of all files including the requested fields:
        rundir = self.get_value("RUNDIR")
        dout_s_root = self.get_value("DOUT_S_ROOT")
        regex = DiagsCase.convert_prefix_to_regex(hist_file_prefix)
        log.info(f"regex to determine all files including {field0}: {regex}")
        all_nc_files = []
        if rundir != None:
            all_nc_files += [os.path.join(rundir,f) for f in os.listdir(rundir) if f[-3:]=='.nc']
        if dout_s_root != None:
            all_nc_files += [os.path.join(dout_s_root,f) for f in os.listdir(dout_s_root) if f[-3:]=='.nc']
        all_matched_files = [f for f in all_nc_files if re.search(regex,f)]
        all_matched_files.sort()
        log.info(f"number of files including {field0}: {len(all_matched_files)}")

        # sanity check:
        assert len(all_matched_files)>0, f"Cannot find any history files including {fields}"

        return all_matched_files

    def create_esmlab_dataset(self, fields:list):
        log.info(f"Constructing a dataset for fields: {fields}")
        file_list = self._get_file_list(fields)
        #log.info(f"Files to read:")
        #for f in file_list:
        #    log.info(f"\t{f}")
        esm = xr.open_mfdataset(file_list, decode_times=False).esm
        return esm
