#!/usr/bin/env python

import os
import yaml
import xarray as xr
import numpy as np
import glob
import textwrap
import subprocess
import argparse, warnings
from datetime import datetime
from ncar_jobqueue import NCARCluster
from dask.distributed import Client
from mom6_tools.m6toolbox import weighted_temporal_mean_vars, add_global_attrs
from mom6_tools.m6toolbox import cime_xmlquery, filter_vars, replace_cell_content
from mom6_tools.MOM6grid import MOM6grid

# Suppress warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Create monthly climatology and averages for datasets.')
    parser.add_argument('config_yml', type=str, help='Path to YAML configuration file.')
    parser.add_argument('-v', '--variable', type=str, default='', help='Variable to be processed (default is empty).')
    parser.add_argument('-s', '--stream', type=str, default='.mom6.h.z.????-??.nc', help='History file stream (default is .mom6.h.z.????-??.nc)')
    parser.add_argument('-f', '--fname', type=str, default='z', help='Name of the history file stream (default is z)')
    parser.add_argument('-sd', '--start_date', type=str, default='', help='Start date for averaging (YYYY-MM).')
    parser.add_argument('-ed', '--end_date', type=str, default='', help='End date for averaging (YYYY-MM).')
    parser.add_argument('-debug', action='store_true', help='Enable debug mode.')
    return parser.parse_args()

# Function to submit PBS script for each variable
def submit_pbs_script(var, stream, fname):
    """Create and submit a PBS script for generating climatology."""
    pbs_script = textwrap.dedent(f"""\
    #!/bin/bash
    #PBS -N {fname}_{var}
    #PBS -A NCGD0011
    #PBS -l select=1:ncpus=1:mem=4GB
    #PBS -l walltime=06:00:00
    #PBS -q casper
    #PBS -o {fname}/{fname}_{var}.o
    #PBS -j oe

    source ~/.bashrc
    conda activate /glade/work/gmarques/conda-envs/mom6-tools

    create_climatology.py diag_config.yml -v {var} -s {stream} -f {fname}
    """)

    # Create the directory if it does not exist
    os.makedirs(fname, exist_ok=True)
    # Write the PBS script to a temporary file
    pbs_filename = os.path.join(fname, f"climo_{fname}_{var}.sh")
    with open(pbs_filename, 'w') as f:
        f.write(pbs_script)

    # Submit the PBS script
    subprocess.run(['qsub', pbs_filename])
    print(f'{pbs_filename} was submitted...')
    return

def process_dataset(ds1, grd, start_date, end_date, output_dir, casename, dataset_label):
    """Compute climatology and annual mean for all variables in the given dataset."""

    ds_sel = ds1.sel(time=slice(start_date, end_date))

    print(f'Computing annual mean for {dataset_label}...')
    startTime = datetime.now()
    ds_ann = weighted_temporal_mean_vars(ds_sel)
    ds_mean = ds_ann.mean('time').load()
    print('Time elasped: ', datetime.now() - startTime)

    startTime = datetime.now()
    print(f'Computing monthly climatology for {dataset_label}...')
    ds_clim = ds_sel.groupby('time.month').mean('time').load()
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
        long_name = ds1[var].long_name
        print(f'Processing {var}...')
        startTime = datetime.now()
        output_file = os.path.join(output_dir, f'{casename}.{dataset_label}.{var}.{start_date_label}-{end_date_label}.nc')

        ds_combined = xr.Dataset({
            f'{var}_annual_mean': ds_mean[var],
            f'{var}_monthly_climatology': ds_clim[var]
        })

        # add coords
        x = ds_mean[var].dims[-1]
        y = ds_mean[var].dims[-2]
        ds_combined = ds_combined.assign_coords(geolat=((y,x),grd.geolat.values))
        ds_combined = ds_combined.assign_coords(geolon=((y,x),grd.geolon.values))

        add_global_attrs(ds_combined, {
            'description': f'Annual mean and monthly climatology for {var} ({dataset_label})',
            'long_name': long_name,
            'casename': casename,
            'units': units,
            'start_date': start_date,
            'end_date': end_date
        })

        ds_combined.to_netcdf(output_file)
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
    ocn_diag_root = os.path.join(ocn_diag_root, "climo/")
    args.casename = cime_xmlquery(caseroot, 'CASE')
    args.geom = args.casename+config['Fnames']['geom']
    args.static = args.casename+config['Fnames']['static']
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

    try:
      os.makedirs(ocn_diag_root, exist_ok=True)
    except:
      current_path = os.getcwd()
      proc_path = os.path.join(current_path, "proc")
      warnings.warn(f"Directory {ocn_diag_root} could not be created. Using {proc_path} instead.", UserWarning)
      ocn_diag_root = proc_path
      os.makedirs(ocn_diag_root, exist_ok=True)

    climo_path = f"{ocn_diag_root}../../notebooks/climo_{fname}/"
    os.makedirs(climo_path, exist_ok=True)

    if not variable:
      print("The variable is an empty string. Processing all variables in {}".format(stream))

      # Select all files that contain 'native' in their name
      file = glob.glob(os.path.join(OUTDIR, args.casename+stream))[0]

      if args.debug:
        print(f'file: {file}')

      ds_file = filter_vars(xr.open_dataset(file))

      if args.debug:
        print(ds_file)

      # Write to a markdown file
      md_path = f"{climo_path}climo_{fname}.md"
      # Open the markdown file to write
      with open(md_path, 'w') as f:
        # Write the header
        f.write(f"# Climo {fname}\n")
        f.write("""This section presents area-weighted maps—and, when applicable,
                   vertical profiles—for the following ocean basins: \n\n""")
        f.write("![basins](../images/basins.png)\n\n")
        f.write("Variables processed:\n\n")

        # Iterate through the variables and write the list
        for var in ds_file.data_vars:
            long_name = ds_file[var].attrs.get('long_name', 'No long_name available')
            units = ds_file[var].attrs.get('units', 'No units available')

            # Write the variable, long_name, and units to the file
            f.write(f"- **{var}** ({long_name}, {units})\n")

      print(f"Markdown file has been created at {climo_path}ts.md")

      # Loop over the variables in the dataset and submit a PBS job for each
      for var in ds_file.data_vars:
        # Submit a PBS script for the variable
        submit_pbs_script(var, stream, fname)

    else:
      print("The variable is not an empty string.")

      start_date = args.start_date or config['Avg']['start_date']
      end_date = args.end_date or config['Avg']['end_date']

      print(f'Processing data from {start_date} to {end_date}')

      def preprocess(ds, variable):
        """Preprocess function that selects the specified variable."""
        return ds[[variable]]

      files = os.path.join(OUTDIR, args.casename+stream)
      ds = xr.open_mfdataset(files,
                       parallel=True,
                       combine="nested",
                       concat_dim="time",
                       data_vars="minimal",
                       coords="minimal",
                       compat="override",
                       preprocess=lambda ds: preprocess(ds, variable)
                       )

      # read grid info
      geom_file = OUTDIR+'/'+args.geom
      if os.path.exists(geom_file):
        grd_xr = MOM6grid(OUTDIR+'/'+args.static, geom_file, xrformat=True);
      else:
        grd_xr = MOM6grid(OUTDIR+'/'+args.static, xrformat=True);

      # Process variable in dataset
      process_dataset(ds, grd_xr, start_date, end_date, ocn_diag_root, args.casename, fname)

      # run notebook
      print(f'Generating notebook for {variable}')
      long_name = ds[variable].long_name
      startTime = datetime.now()
      # Get the directory of the current script
      script_dir = os.path.dirname(os.path.realpath(__file__))
      # Define path to the template notebook
      template_path = os.path.join(script_dir, 'nb_templates', 'climo.ipynb')
      cwd = os.getcwd()
      #path_out = f"{ocn_diag_root}/../../notebooks/climo/"
      os.chdir(climo_path)
      file_out = f"{variable}.ipynb"
      cmd = f"papermill {template_path} {file_out} -p variable {variable} -p stream {fname} -p long_name '{long_name}'"
      print(cmd)
      file_out = f"{climo_path}{variable}.ipynb"
      subprocess.run(cmd, shell=True, check=True)
      replace_cell_content(file_out, variable, file_out)
      os.chdir(cwd)
      print('Time elasped: ', datetime.now() - startTime)

    print('Total time elasped: ', datetime.now() - startTime_main)
    print('{} was run successfully!'.format(os.path.basename(__file__)))
    return

if __name__ == '__main__':
    main()

