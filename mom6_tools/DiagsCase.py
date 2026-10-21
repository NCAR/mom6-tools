import yaml
import os
import re
import logging as log
from collections import namedtuple
import xarray as xr
from mom6_tools.MOM6grid import MOM6grid
from mom6_tools.m6toolbox import cime_xmlquery
import dask


DiagFileEntry = namedtuple('DiagFileEntry',
                    ("output_freq", "output_freq_units", "file_format", "time_axis_units",
                     "time_axis_name", "new_file_freq", "new_file_freq_units", "start_time",
                     "file_duration", "file_duration_units"))
DiagFieldEntry = namedtuple('DiagFieldEntry',
                    ("module_name", "output_name", "time_sampling", "reduction_method",
                     "regional_section", "packing"))

class DiagsCase(object,):
    """ CESM case manager. Provides methods to get the values of CESM case parameters,
        to parse case files such as diag_table, and to generate datasets from history
        output files.

    Attributes
    -------
    casename
        CIME case name (cime_xmlquery(CASEROOT, 'CASE'))
    ocn_diag_root
        Directory used for diagnostic outputs (created at construction time)
    hist_dir
        Directory containing MOM6 history (output) files, read from -- distinct
        from ocn_diag_root, which is written to

    Methods
    -------
    get_value(var)
        Returns the value of a variable defined in yaml config file.
    get_fname(key)
        Returns the full path (glob pattern) of the history files for a given stream.
    get_grid(xrformat)
        Returns a MOM6grid instance built from this case's static and geom files.
    create_png_dir(subdir)
        Creates and returns the directory used for PNG figures (independent of OCN_DIAG_ROOT).
    stage_dset(fields)
        Returns an xarray dataset that contain the specified fields.
    """

    def __init__(self, yaml_path, xrformat=False):
        """ Parameters
            ----------
            yaml_path : str
                Full path to the diag_config.yml file. Its layout:

                Case:
                    CASEROOT: ... # required; path to the case root directory
                    OCN_DIAG_ROOT: ... # required; path to diagnostics output files
                    SNAME: ... # required; short name of the case
                    HIST_FILE_PREFIX: ... # optional; prefix of history output files
                Jobqueue: ...  # required; dask cluster settings, see mom6_tools.jobqueue
                Avg:           # optional; time-averaging window (None: unbounded)
                    start_date: ...
                    end_date: ...
                TS:            # optional; time series window (None: entire record)
                    start_date: ...
                    end_date: ...
                Fnames: ...    # history file suffixes per stream, see get_fname
                Misc:          # optional
                    savefigs: ... # defaults to True
                Transports: ...   # optional; sections for volume/mass transports
                oce_cat: ...      # optional; ocean catalog file for observational datasets

            xrformat : boolean, optional
            If True, returns an xarray Dataset with the grid. Otherwise (default), returns an
            object with numpy arrays.

            Raises
            ------
            ValueError
                If a required Case key or the Jobqueue section is missing or empty.
        """

        with open(yaml_path, 'r') as f:
            self.full_config = yaml.load(f, Loader=yaml.Loader)
        self._config = self.full_config['Case']

        # Check if required keys are available and non-empty in diag_config.yml
        required_keys = ("CASEROOT", "OCN_DIAG_ROOT", "SNAME")
        missing_keys = [key for key in required_keys if not self.get_value(key)]
        if missing_keys:
            raise ValueError(
                f"Required key(s) missing or empty: {', '.join(missing_keys)}"
            )

        self.jobqueue_config = self.full_config.get('Jobqueue')
        if not self.jobqueue_config:
            self.jobqueue_config = dask.config.get("jobqueue", {}).get("pbs")
            if not self.jobqueue_config:
                # This should fall back to some default configuration instead
                raise ValueError("'Jobqueue' section missing or empty in diag_config.yml. \n Consult https://jobqueue.dask.org/en/latest/clusters-configuration.html#configuration-files for information on how to set up a user-level jobqueue.yaml file.")

        self.caseroot = self.get_value('CASEROOT')
        self.casename = cime_xmlquery(self.caseroot, 'CASE')

        # Location of MOM6 output files
        dout_s = cime_xmlquery(self.caseroot, 'DOUT_S')
        if dout_s.strip().lower() == 'true':
            self.hist_dir = cime_xmlquery(self.caseroot, 'DOUT_S_ROOT') + '/ocn/hist/'
        else:
            self.hist_dir = cime_xmlquery(self.caseroot, 'RUNDIR')
        print('History directory is:', self.hist_dir)

        self.diag_files = None
        self.diag_fields = None
        self.xrformat = xrformat

        # `or {}` also covers sections present in the yaml but left blank (loaded as None)
        avg = self.full_config.get('Avg') or {}
        self.start_date = avg.get('start_date')
        self.end_date = avg.get('end_date')
        ts = self.full_config.get('TS') or {}
        self.ts_start_date = ts.get('start_date')
        self.ts_end_date = ts.get('end_date')
        self.savefigs = (self.full_config.get('Misc') or {}).get('savefigs', True)
        self.label = self.get_value('SNAME')
        self.ocn_diag_root = self.create_output_dir()
        self.oce_cat = self.full_config.get('oce_cat')
        self.transports = self.full_config.get('Transports') or {}

    def get_fname(self, key):
        """Returns the full path (glob pattern) of the history files for a given
        stream, i.e. `hist_dir`/`casename` + `full_config['Fnames'][key]`.

        Parameters
        ----------
        key : str
            Key into the yaml file's 'Fnames' section (e.g. 'native', 'static',
            'geom', 'z', 'rho2').
        """
        fnames = self.full_config.get('Fnames') or {}
        if key not in fnames:
            raise KeyError(f"'{key}' not found in the Fnames section of diag_config.yml")
        return os.path.join(self.hist_dir, self.casename + fnames[key])

    def get_grid(self, xrformat=None):
        """Returns a MOM6grid instance built from this case's static and geom files.

        Parameters
        ----------
        xrformat : boolean, optional
            If provided, overrides self.xrformat for this call (e.g. to get both
            an object-with-numpy-arrays grid and an xarray-Dataset grid from the
            same case).
        """
        static_file = self.get_fname('static')
        geom_file = self.get_fname('geom')
        return MOM6grid(static_file, geom_file,
                         xrformat=self.xrformat if xrformat is None else xrformat)

    def get_value(self, var):
        """ Returns the value of a variable in yaml config file.

        Parameters
        ----------
        var : string
            Variable name

        """

        val = self._config.get(var)

        if type(val) == type("") and val.lower() == "none":
            val = None

        log.info(f"get_value::\n\trequsted variable: {var} \n\treturning value: {val}"\
                 f"\n\ttype: {type(val)}")
        return val

    def create_output_dir(self, subdir=None):
        """Create and return the directory used for diagnostic outputs.

        Parameters
        ----------
        subdir : str, optional
            Optional subdirectory appended beneath OCN_DIAG_ROOT.

        Returns
        -------
        str
            Path to the output directory.
        """

        output_dir = self.get_value('OCN_DIAG_ROOT')
        if subdir:
            output_dir = os.path.join(output_dir, subdir)

        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def create_png_dir(self, subdir=None):
        """Create and return the directory used for PNG figures.

        Independent of OCN_DIAG_ROOT: always relative to the current working
        directory (e.g. 'PNG/<subdir>'), since figures and diagnostic output
        files are kept in separate locations.

        Parameters
        ----------
        subdir : str, optional
            Optional subdirectory appended beneath 'PNG'.

        Returns
        -------
        str
            Path to the PNG output directory.
        """

        png_dir = os.path.join('PNG', subdir) if subdir else 'PNG'
        os.makedirs(png_dir, exist_ok=True)
        return png_dir

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
            raise RuntimeError(f"Multiple '{fld_to_search}' entries in diag_table. Provide HIST_FILE_PREFIX!")
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
        else:
            # check if given hist_file_prefix actually exists in diag_table:
            self._parse_diag_table()
            assert hist_file_prefix in self.diag_files, f"Cannot find "+hist_file_prefix+" in diag_table"

        # create a list of all files including the requested fields:
        rundir = self.get_value("RUNDIR")
        dout_s = self.get_value('DOUT_S')
        dout_s_root = self.get_value("DOUT_S_ROOT")
        regex = DiagsCase.convert_prefix_to_regex(hist_file_prefix)
        log.info(f"regex to determine all files including {field0}: {regex}")
        all_nc_files = []
        if rundir != None:
            all_nc_files += [os.path.join(rundir,f) for f in os.listdir(rundir) if f[-3:]=='.nc']
        if dout_s_root and dout_s==True:
            all_nc_files += [os.path.join(dout_s_root,f) for f in os.listdir(dout_s_root) if f[-3:]=='.nc']
        all_matched_files = [f for f in all_nc_files if re.search(regex,f)]
        all_matched_files.sort()
        log.info(f"number of files including {field0}: {len(all_matched_files)}")

        # sanity check:
        assert len(all_matched_files)>0, f"Cannot find any history files including {fields}"

        return all_matched_files

    def stage_dset(self, fields:list):
        """ Generates a dataset containing the given fields for the entire
        duration of a run

        Parameters
        ----------
        fields : list
            The list of fields of this case to include in the dataset to be generated."

        Returns
        -------
        xarray.Dataset
        """

        log.info(f"Constructing a dataset for fields: {fields}")
        file_list = self._get_file_list(fields)
        dset = xr.open_mfdataset(file_list)#, decode_times=False)

        # confine dataset to given list of fields
        if "average_T1" not in fields:
            fields.append("average_T1")
        if "average_T2" not in fields:
            fields.append("average_T2")
        dset = dset[fields]

        return dset

