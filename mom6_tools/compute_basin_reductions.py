#!/usr/bin/env python

import os
import yaml
import xarray as xr
import numpy as np
import glob
import textwrap
import re
import subprocess
import nbformat
import argparse, warnings
from datetime import datetime
from ncar_jobqueue import NCARCluster
from dask.distributed import Client
from mom6_tools.m6toolbox import weighted_temporal_mean_vars, add_global_attrs
from mom6_tools.m6toolbox import cime_xmlquery,filter_vars_2D_tracers
from mom6_tools.m6toolbox import replace_cell_content
from mom6_tools.MOM6grid import MOM6grid

# Suppress warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description=
        "This script computes area-weighted mean and integral time series of a given ocean variable "
        "over predefined ocean basin masks. It takes as input a dataset file and loops "
        "over specified basin masks to extract and average/integrate the variable within each region.")
    parser.add_argument('config_yml', type=str, help='Path to YAML configuration file.')
    parser.add_argument('-v', '--variable', type=str, default='', help='Variable to be processed (default is empty, it will process all 2D variables on tracer points).')
    parser.add_argument('-s', '--stream', type=str, default='.mom6.h.native.????-??.nc', help='History file stream (default is .mom6.h.native.????-??.nc)')
    parser.add_argument('-f', '--fname', type=str, default='native', help='Name of the history file stream (default is native)')
    parser.add_argument('-sd', '--start_date', type=str, default='', help='Start date for averaging (YYYY-MM).')
    parser.add_argument('-ed', '--end_date', type=str, default='', help='End date for averaging (YYYY-MM).')
    parser.add_argument('-debug', action='store_true', help='Enable debug mode.')
    return parser.parse_args()

# Function to submit PBS script for each variable
def submit_pbs_script(var, stream, fname):
    """Create and submit a PBS script for generating area-weighted mean time series."""

    pbs_script = textwrap.dedent(f"""\
    #!/bin/bash
    #PBS -N {fname}_{var}
    #PBS -A p93300012
    #PBS -l select=1:ncpus=1:mem=4GB
    #PBS -l walltime=02:00:00
    #PBS -q casper
    #PBS -o {fname}/{fname}_{var}.o
    #PBS -j oe

    source ~/.bashrc
    conda activate /glade/work/gmarques/conda-envs/mom6-tools

    compute_basin_reductions.py diag_config.yml -v {var} -s {stream} -f {fname}
    """)

    # Create the directory if it does not exist
    os.makedirs(fname, exist_ok=True)
    # Write the PBS script to a temporary file
    pbs_filename = os.path.join(fname, f"basin_means_{fname}_{var}.sh")
    with open(pbs_filename, 'w') as f:
        f.write(pbs_script)

    # Submit the PBS script
    subprocess.run(['qsub', pbs_filename])
    print(f'{pbs_filename} was submitted...')
    return

def remove_m2_from_units(units):
    """
    Removes 'm-2' from the unit string while keeping the rest intact.

    Parameters:
        units (str): The original unit string.

    Returns:
        str: The modified unit string without 'm-2'.
    """
    return re.sub(r"\s*m-2\s*", " ", units).strip()

def process_dataset(ds1, basin_code, area, output_dir, casename, dataset_label):
    """Compute area-weighted mean and integral time series for all 2D variables in the given dataset."""

    start_date = str(ds1.time[0].values)
    end_date = str(ds1.time[-1].values)

    print(f'Processing data from {start_date} to {end_date}')

    ds_sel = ds1.sel(time=slice(start_date, end_date))

    print(f'Computing annual mean...')
    startTime = datetime.now()
    ds_ann = weighted_temporal_mean_vars(ds_sel)
    print('Time elasped: ', datetime.now() - startTime)

    # Iterate over all variables and save to separate files
    print(f'Iterate over all variables and save to separate files...')
    print(f'Variables to be processed:',ds1.data_vars)

    start_date_label = start_date[:4]+start_date[5:7]
    end_date_label = end_date[:4]+end_date[5:7]

    for var in ds1.data_vars:
        if var in ds1.coords:  # Skip coordinates like 'time'
          print(f'Skipping {var}...')
          continue

        units = ds1[var].units
        units_int = ''
        long_name = ds1[var].long_name
        print(f'Processing {var}...')
        print(f'long_name: {long_name}')
        print(f'units: {units}')
        startTime = datetime.now()
        output_file = os.path.join(output_dir, f'{casename}.{dataset_label}.{var}.{start_date_label}-{end_date_label}.nc')

        startTime = datetime.now()
        print(f'Computing area-weighted mean time series for {var}...')
        ds_mean_wgt = (ds_ann[var] * basin_code).weighted(area*basin_code).mean(dim=["yh", "xh"]).rename(var+'_mean')
        print('Time elasped: ', datetime.now() - startTime)

        # Decide if integral should be computed. Only for following units:
        units_list = [ "psu kg m-2", "degC kg m-2", "W m-2", "kg m-2 s-1"]
        if units in units_list:
          startTime = datetime.now()
          print(f'Computing area-weighted integral time series for {var}...')
          ds_int_wgt = (ds_ann[var] * basin_code * area).sum(dim=["yh", "xh"]).rename(var+'_int')
          print('Time elasped: ', datetime.now() - startTime)
          ds_mean_wgt = xr.Dataset({
            f'{var}_mean': ds_mean_wgt,
            f'{var}_int': ds_int_wgt
          })

          units_int = remove_m2_from_units(units)
        else:
          ds_mean_wgt = xr.Dataset({
            f'{var}_mean': ds_mean_wgt,
          })

        add_global_attrs(ds_mean_wgt, {
            'description': f'Area-weighted mean {var}_mean and, in some cases, integral {var}_int time series for over certain regions ({dataset_label})',
            'casename': casename,
            'long_name': long_name,
            'units_mean': units,
            'units_int': units_int,
            'regions': basin_code.region.values.tolist(),
            'start_date': start_date,
            'end_date': end_date
        })

        ds_mean_wgt.to_netcdf(output_file)
        print(f'Saved {dataset_label} {var} data to {output_file}')
        print('Time elasped: ', datetime.now() - startTime)

def main():
    startTime_main = datetime.now()

    args = parse_args()
    variable = args.variable
    stream = args.stream
    fname = args.fname
    # Read in the yaml file
    config = yaml.load(open(args.config_yml,'r'), Loader=yaml.Loader)

    caseroot = config['Case']['CASEROOT']
    ocn_diag_root = config['Case']['OCN_DIAG_ROOT']
    args.casename = cime_xmlquery(caseroot, 'CASE')
    DOUT_S = cime_xmlquery(caseroot, 'DOUT_S')
    if DOUT_S.lower() == "true":
      OUTDIR = cime_xmlquery(caseroot, 'DOUT_S_ROOT')+'/ocn/hist/'
    else:
      OUTDIR = cime_xmlquery(caseroot, 'RUNDIR')

    print('DOUT_S:', DOUT_S)
    print('Model directory with history files is:', OUTDIR)
    print('Casename is:', args.casename)
    print('Variable is:', variable)
    print('Stream is:', stream)

    # GMM, update this
    basin_code = xr.open_dataset('/glade/work/gmarques/cesm/tx2_3/basin_masks/basin_masks_tx2_3v2_20250318.nc')['basin_masks']
    args.geom = args.casename+config['Fnames']['geom']
    args.static = args.casename+config['Fnames']['static']

    # read grid info
    geom_file = OUTDIR+'/'+args.geom
    if os.path.exists(geom_file):
      grd = MOM6grid(OUTDIR+'/'+args.static, geom_file, xrformat=True)
    else:
      grd = MOM6grid(OUTDIR+'/'+args.static, xrformat=True)

    try:
      area = xr.where(grd.wet == 1, grd.area_t, 0.)
    except:
      area = xr.where(grd.wet == 1, grd.areacello, 0.)

    try:
      os.makedirs(ocn_diag_root, exist_ok=True)
    except:
      current_path = os.getcwd()
      proc_path = os.path.join(current_path, "proc")
      warnings.warn(f"Directory {ocn_diag_root} could not be created. Using {proc_path} instead.", UserWarning)
      ocn_diag_root = proc_path
      os.makedirs(ocn_diag_root, exist_ok=True)

    ts_path = f"{ocn_diag_root}../notebooks/ts/"
    os.makedirs(ts_path, exist_ok=True)
    print(f"created {ts_path}")

    if not variable:
      print("The variable is an empty string. Processing all variables in {}".format(stream))

      # Select all files that contain 'native' in their name
      file = glob.glob(os.path.join(OUTDIR, args.casename+stream))[0]

      if args.debug:
        print(f'file: {file}')

      ds_file = filter_vars_2D_tracers(xr.open_dataset(file))

      if args.debug:
        print(ds_file)

      # Write to a markdown file
      md_path = f"{ts_path}ts.md"
      # Open the markdown file to write
      with open(md_path, 'w') as f:
        # Write the header
        f.write("# Time series\n")
        f.write("""This section contains area-weighted mean and integrated time
                 series for a set of two-dimensional variables. These time series
                 were generated using the following basin masks:\n\n""")
        f.write("![basins](../images/basins.png)\n\n")
        f.write("Variables processed:\n\n")

        # Iterate through the variables and write the list
        for var in ds_file.data_vars:
            long_name = ds_file[var].attrs.get('long_name', 'No long_name available')
            units = ds_file[var].attrs.get('units', 'No units available')

            # Write the variable, long_name, and units to the file
            f.write(f"- **{var}** ({long_name}, {units})\n")

      print(f"Markdown file has been created at {ts_path}ts.md")

      # Loop over the variables in the dataset and submit a PBS job for each
      for var in ds_file.data_vars:
        # Submit a PBS script for the variable
        submit_pbs_script(var, stream, fname)

    else:
      print(f'Processing {variable}')

      start_date = args.start_date or config['Avg']['start_date']
      end_date = args.end_date or config['Avg']['end_date']

      def preprocess(ds, variable):
        """Preprocess function that selects the specified variable."""
        return ds[[variable]]

      files = os.path.join(OUTDIR, args.casename+stream)
      ds = xr.open_mfdataset(files,
                       parallel=False,
                       combine="nested",
                       concat_dim="time",
                       data_vars="minimal",
                       coords="minimal",
                       compat="override",
                       preprocess=lambda ds: preprocess(ds, variable)
                       )

      # Process variable in dataset
      process_dataset(ds, basin_code, area, ocn_diag_root, args.casename, fname)

      print(f'Generating notebook for {variable}')
      long_name = ds[variable].long_name
      startTime = datetime.now()
      # Get the directory of the current script
      script_dir = os.path.dirname(os.path.realpath(__file__))
      # Define path to the template notebook
      template_path = os.path.join(script_dir, 'nb_templates', 'ts.ipynb')
      cwd = os.getcwd()
      os.chdir(ts_path)
      file_in  = f"template.ipynb"
      file_out = f"{variable}.ipynb"
      #cmd = f"papermill {file_in} {file_out} -p variable {variable}"
      cmd = f"papermill {template_path} {file_out} -p variable {variable} -p long_name '{long_name}'"
      print(cmd)
      file_out = f"{ts_path}{variable}.ipynb"
      subprocess.run(cmd, shell=True, check=True)
      replace_cell_content(file_out, variable, file_out)
      os.chdir(cwd)
      print('Time elasped: ', datetime.now() - startTime)

    print('Total time elasped: ', datetime.now() - startTime_main)
    print('{} was run successfully!'.format(os.path.basename(__file__)))
    return

if __name__ == '__main__':
    main()

