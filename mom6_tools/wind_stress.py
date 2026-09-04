#!/usr/bin/env python

"""
Compute zonal-mean zonal wind stress and wind stress curl as functions of
latitude and time. Produce Hovmoller diagrams for the Southern Ocean.

Usage example (standalone):
  python wind_stress.py '/path/to/archive/ocn/hist/*native*0001-??.nc'

Usage example (yaml-based workflow):
  python wind_stress.py diag_config.yml
"""

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os, yaml, argparse, glob
from datetime import datetime
from scipy.ndimage import uniform_filter1d
from xgcm import Grid
from mom6_tools.m6toolbox import add_global_attrs, cime_xmlquery
from mom6_tools.MOM6grid import MOM6grid
from mom6_tools.DiagsCase import DiagsCase
from pathlib import Path


def parseCommandLine():
  """
  Parse the command line positional and optional arguments.
  """
  parser = argparse.ArgumentParser(description=
      '''
      Compute zonal-mean zonal wind stress and wind stress curl as functions
      of latitude and time. Produce Hovmoller diagrams for the Southern Ocean.
      ''',
  epilog='Written by Gustavo Marques (gmarques@ucar.edu).')
  parser.add_argument('input_path', type=str, help='''Path to yaml config file
    or glob pattern for native history files (e.g. '/path/*native*0001-??.nc').''')
  parser.add_argument('-asd', '--avg_start_date', type=str, default='',
                      help='''Start date to select data. Default is to use all available data
                      (or value set in yaml config).''')
  parser.add_argument('-aed', '--avg_end_date', type=str, default='',
                      help='''End date to select data. Default is to use all available data
                      (or value set in yaml config).''')
  parser.add_argument('-tsd', '--ts_start_date', type=str, default='',
                      help='''Start date for time-series (TS) analysis. Default is to use value set in diag_config_yml_path, or the whole record if not set there.''')
  parser.add_argument('-ted', '--ts_end_date', type=str, default='',
                      help='''End date for time-series (TS) analysis. Default is to use value set in diag_config_yml_path, or the whole record if not set there.''')
  parser.add_argument('-o','--output_dir', type=str, default='ncfiles',
                      help='''Directory for output NetCDF files (default: ncfiles).''')
  parser.add_argument('-p','--plot_dir', type=str, default='PNG/WIND',
                      help='''Directory for output plots (default: PNG/WIND).''')
  parser.add_argument('-label','--label', type=str, default='',
                      help='''Label for the case (used in plot titles).''')
  parser.add_argument('-nw','--number_of_workers',  type=int, default=0,
                      help='''Number of workers to use (default=0, serial job).''')
  optCmdLineArgs = parser.parse_args()
  return optCmdLineArgs


def driver(args):
  nw = args.number_of_workers

  # Determine if input is a yaml config or a file glob pattern
  if args.input_path.endswith('.yml') or args.input_path.endswith('.yaml'):
    # yaml-based workflow
    diag_config_yml = yaml.load(open(args.input_path,'r'), Loader=yaml.Loader)
    dcase = DiagsCase(diag_config_yml['Case'])
    caseroot = diag_config_yml['Case']['CASEROOT']
    args.casename = cime_xmlquery(caseroot, 'CASE')
    DOUT_S = cime_xmlquery(caseroot, 'DOUT_S')
    if DOUT_S.lower() == "true":
      OUTDIR = cime_xmlquery(caseroot, 'DOUT_S_ROOT')+'/ocn/hist/'
    else:
      OUTDIR = cime_xmlquery(caseroot, 'RUNDIR')

    dcase.set_dates(args, diag_config_yml)
    dcase.set_fnames(args, diag_config_yml, {'native': 'native'})
    file_pattern = OUTDIR + '/' + args.native
    if not args.label: args.label = diag_config_yml['Case'].get('SNAME', args.casename)

    # Use OCN_DIAG_ROOT (via DiagsCase) for output directories
    args.output_dir = dcase.create_output_dir()
    args.plot_dir = dcase.create_output_dir(subdir=os.path.join('PNG', 'WIND'))
  else:
    # standalone mode: input_path is a glob pattern
    file_pattern = args.input_path
    files = sorted(glob.glob(file_pattern))
    if not files:
      raise FileNotFoundError(f'No files matched: {file_pattern}')
    args.casename = os.path.basename(files[0]).split('.mom6.')[0]
    if not args.label: args.label = args.casename
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

  print(f'Case: {args.casename}')
  print(f'File pattern: {file_pattern}')
  print(f'Number of workers: {nw}')

  # Set up parallel processing
  parallel = False
  if nw > 1:
    parallel = True
    from ncar_jobqueue import NCARCluster
    from dask.distributed import Client
    cluster = NCARCluster()
    cluster.scale(nw)
    client = Client(cluster)

  # Open dataset
  print('Opening dataset...')
  startTime = datetime.now()

  def preprocess(ds):
    variables = ['tauuo', 'tauvo']
    return ds[[v for v in variables if v in ds]]

  ds = xr.open_mfdataset(file_pattern,
                         preprocess=preprocess,
                         data_vars='minimal',
                         coords='minimal',
                         compat='override',
                         chunks={'time': 12})
  print(f'Time elapsed: {datetime.now() - startTime}')

  # Select time range if specified
  if args.avg_start_date and args.avg_end_date:
    print(f'Selecting data between {args.avg_start_date} and {args.avg_end_date}...')
    ds = ds.sel(time=slice(args.avg_start_date, args.avg_end_date))

  taux = ds['tauuo']  # (time, yh, xq) - zonal wind stress
  tauy = ds['tauvo']  # (time, yq, xh) - meridional wind stress

  # ---- 1. Zonal-mean zonal wind stress ----
  print('Computing zonal-mean zonal wind stress...')
  startTime = datetime.now()
  taux_zmean = taux.mean(dim='xq').compute()
  taux_zmean.name = 'taux_zmean'
  taux_zmean.attrs['long_name'] = 'Zonal-mean zonal wind stress'
  taux_zmean.attrs['units'] = 'N m-2'
  print(f'Time elapsed: {datetime.now() - startTime}')

  taux_ds = taux_zmean.to_dataset()
  taux_ds.attrs = {
      'description': 'Zonal-mean zonal wind stress as f(latitude, time)',
      'casename': args.casename,
      'module': os.path.basename(__file__)
  }
  taux_file = os.path.join(args.output_dir, f'{args.casename}_taux_zmean.nc')
  taux_ds.to_netcdf(taux_file)
  print(f'Saved: {taux_file}')

  # ---- 2. Wind stress curl ----
  print('Setting up xgcm grid...')
  coords = {
      'X': {'center': 'xh', 'right': 'xq'},
      'Y': {'center': 'yh', 'right': 'yq'},
  }
  grid = Grid(ds, coords=coords, periodic=['X'], autoparse_metadata=False)

  print('Computing wind stress curl...')
  startTime = datetime.now()
  curl = (
      grid.diff(tauy, "X", boundary="fill")
      - grid.diff(taux, "Y", boundary="fill")
  ).load()
  curl.name = 'wind_stress_curl'
  print(f'Time elapsed: {datetime.now() - startTime}')

  # Determine dimension names after diff (should be xq, yq)
  x_dim = [d for d in curl.dims if d.startswith('x')][0]
  y_dim_curl = [d for d in curl.dims if d.startswith('y')][0]

  print('Computing zonal-mean wind stress curl...')
  curl_zmean = curl.mean(dim=x_dim).compute()
  curl_zmean.name = 'curl_zmean'
  curl_zmean.attrs['long_name'] = 'Zonal-mean wind stress curl'

  curl_ds = curl_zmean.to_dataset()
  curl_ds.attrs = {
      'description': 'Zonal-mean wind stress curl as f(latitude, time)',
      'casename': args.casename,
      'module': os.path.basename(__file__)
  }
  curl_file = os.path.join(args.output_dir, f'{args.casename}_curl_zmean.nc')
  curl_ds.to_netcdf(curl_file)
  print(f'Saved: {curl_file}')

  # ---- 3. Hovmoller diagrams ----
  print('Creating Hovmoller diagrams...')

  # Southern Ocean latitude range
  lat_range = slice(-70, -40)

  # Taux Hovmoller
  taux_so = taux_zmean.sel(yh=lat_range)
  plot_hovmoller(
      taux_so, 'yh',
      title=f'Zonal-mean zonal wind stress\n{args.label}',
      cbar_label=r'$\tau_x$ [N m$^{-2}$]',
      outfile=os.path.join(args.plot_dir, f'{args.casename}_hovmoller_taux.png')
  )

  # Curl Hovmoller
  curl_so = curl_zmean.sel({y_dim_curl: lat_range})
  plot_hovmoller(
      curl_so, y_dim_curl,
      title=f'Zonal-mean wind stress curl\n{args.label}',
      cbar_label='Wind stress curl',
      outfile=os.path.join(args.plot_dir, f'{args.casename}_hovmoller_curl.png'),
      zero_contour=True
  )

  if parallel:
    print('Releasing workers...')
    client.close(); cluster.close()

  print(f'{os.path.basename(__file__)} completed successfully!')
  return


def plot_hovmoller(da, y_dim, title, cbar_label, outfile,
                   zero_contour=False, smooth_window=12):
  """Plot Hovmoller diagram with smoothed latitude of maximum overlaid."""
  fig, ax = plt.subplots(figsize=(12, 6))

  # Plot Hovmoller (time on x-axis, latitude on y-axis)
  p = da.plot(ax=ax, x='time', y=y_dim, cmap='RdBu_r',
              add_colorbar=True, cbar_kwargs={'label': cbar_label})

  # Overlay zero contour for curl
  if zero_contour:
    da.plot.contour(ax=ax, x='time', y=y_dim, levels=[0],
                    colors='gray', linewidths=1.0, linestyles='--')

  # Find latitude of maximum at each time step
  lat = da[y_dim]
  idx_max = da.argmax(dim=y_dim, skipna=True)
  lat_max = lat.isel({y_dim: idx_max}).values

  # Smooth the lat-of-max line (running mean)
  if len(lat_max) > smooth_window:
    lat_max_smooth = uniform_filter1d(lat_max.astype(float),
                                      size=smooth_window, mode='nearest')
  else:
    lat_max_smooth = lat_max

  ax.plot(da.time, lat_max_smooth, 'k-', linewidth=1.5, label='Lat of max')

  ax.legend(loc='upper right')
  ax.set_title(title)
  ax.set_xlabel('Time')
  ax.set_ylabel('Latitude')
  plt.tight_layout()
  plt.savefig(outfile, dpi=150, bbox_inches='tight')
  plt.close()
  print(f'  Saved: {outfile}')


def main():
  '''
  Main procedure that calls the driver.
  '''
  args = parseCommandLine()
  driver(args)

# Invoke main() which calls parseCommandLine() and the driver.
if __name__ == '__main__':
  main()

