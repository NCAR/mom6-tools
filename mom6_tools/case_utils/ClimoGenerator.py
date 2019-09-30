#import esmlab
import xarray as xr
import logging
import os
import re
import logging as log
from esmlab import climatology

class ClimoGenerator(object,):

    def __init__(self,diags_case):
        self.diags_case = diags_case
        self.diags_case._parse_diag_table()

    def _get_file_list(self, fields:list):

        field0 = fields[0]

        # from diag_table, get prefix of file including field:
        hist_file_prefix = diags_case.get_value("HIST_FILE_PREFIX")
        if hist_file_prefix == None:
            hist_file_prefix = diags_case.get_file_prefix(field0)
            for f in fields[1:]:
                if diags_case.get_file_prefix(f) != hist_file_prefix:
                    raise RuntimeError(f"The following fields are spreaded across multiple "+\
                                        "netcdf files with different prefixes")

        # create a list of all files including the requested fields:
        rundir = diags_case.get_value("RUNDIR")
        dout_s_root = diags_case.get_value("DOUT_S_ROOT")
        regex = DiagsCase.convert_prefix_to_regex(hist_file_prefix)
        log.info(f"regex to determine all files including {field0}: {regex}")
        all_nc_files = []
        if rundir != None:
            all_nc_files += [f for f in os.listdir(rundir) if f[-3:]=='.nc']
        if dout_s_root != None:
            all_nc_files += [f for f in os.listdir(dout_s_root) if f[-3:]=='.nc']
        all_matched_files = [f for f in all_nc_files if re.search(regex,f)]
        all_matched_files.sort()
        log.info(f"number of files including {field0}: {len(all_matched_files)}")

        # sanity check:
        assert len(all_matched_files)>0, f"Cannot find any history files including {fields}"

        return all_matched_files

    def _construct_dataset(self, fields:list):
        file_list = self._get_file_list(fields)
        self.dset = xr.open_mfdataset(file_list, decode_times=False)
