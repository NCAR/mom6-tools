#!/usr/bin/env python

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import warnings, os, yaml, argparse
import pandas as pd
import dask
from datetime import datetime, date
from ncar_jobqueue import NCARCluster
from dask.distributed import Client
from mom6_tools.DiagsCase import DiagsCase
from mom6_tools.m6toolbox import add_global_attrs
from mom6_tools.m6plot import xycompare, xyplot
from mom6_tools.MOM6grid import MOM6grid
from distributed import Client

def parseCommandLine():
  """
  Parse the command line positional and optional arguments.
  This is the highest level procedure invoked from the very end of the script.
  """
  parser = argparse.ArgumentParser(description=
      '''
      Compute time-averages of surface variables (SST, SSS, SSU, SSV and MLD), and (when possible)
      compare results against observational datasets.
      ''',
  epilog='Written by Gustavo Marques (gmarques@ucar.edu).')
  parser.add_argument('diag_config_yml_path', type=str, help='''Full path to the yaml file  \
    describing the run and diagnostics to be performed.''')
  parser.add_argument('-sd','--start_date', type=str, default='',
                      help='''Start year to compute averages. Default is to use value set in diag_config_yml_path''')
  parser.add_argument('-ed','--end_date', type=str, default='',
                      help='''End year to compute averages. Default is to use value set in diag_config_yml_path''')
  parser.add_argument('-fname','--file_name', type=str, default='.mom6.hm_*.nc',
                      help='''File(s) where vmo should be read. Default .mom6.hm_*.nc''')
  parser.add_argument('-nw','--number_of_workers',  type=int, default=0,
                      help='''Number of workers to use (default=0, serial job).''')
  parser.add_argument('-debug',   help='''Add priting statements for debugging purposes''', action="store_true")
  optCmdLineArgs = parser.parse_args()
  driver(optCmdLineArgs)

#-- This is where all the action happends, i.e., functions for each diagnostic are called.

def driver(args):
  nw = args.number_of_workers
  fname = args.file_name
  if not os.path.isdir('ncfiles'):
    print('Creating a directory to place netCDF files (ncfiles)... \n')
    os.system('mkdir ncfiles')

  # Read in the yaml file
  diag_config_yml = yaml.load(open(args.diag_config_yml_path,'r'), Loader=yaml.Loader)

  # Create the case instance
  dcase = DiagsCase(diag_config_yml['Case'])
  RUNDIR = dcase.get_value('RUNDIR')
  args.casename = dcase.casename
  print('Run directory is:', RUNDIR)
  print('Casename is:', args.casename)
  print('Number of workers: ', nw)

  # set avg dates
  avg = diag_config_yml['Avg']
  if not args.start_date : args.start_date = avg['start_date']
  if not args.end_date : args.end_date = avg['end_date']

  # read grid info
  grd = MOM6grid(RUNDIR+'/'+args.casename+'.mom6.static.nc')

  parallel = False
  if nw > 1:
    parallel = True
    cluster = NCARCluster()
    cluster.scale(args.number_of_workers)
    client = Client(cluster)

  print('Reading {} dataset...'.format(args.file_name))
  startTime = datetime.now()

  def preprocess(ds):
    ''' Return the dataset with variables'''
    #variables = ['taux','tauy', 'time_bnds']
    variables = ['tauuo','tauvo', 'time_bnds']
    return ds[variables]

  ds = xr.open_mfdataset(RUNDIR+'/'+dcase.casename+fname, parallel=parallel)

  ds = preprocess(ds)

  print('Time elasped: ', datetime.now() - startTime)

  print('Selecting data between {} and {}...'.format(args.start_date, args.end_date))
  startTime = datetime.now()
  ds = ds.sel(time=slice(args.start_date, args.end_date))
  print('Time elasped: ', datetime.now() - startTime)

  print('Averaging in time...')
  startTime = datetime.now()
  frc = ds.mean('time').compute()
  print('Time elasped: ', datetime.now() - startTime)

  print('\n Plotting...')
  if not os.path.isdir('PNG/WSTRESS'):
    print('Creating a directory to place figures (PNG/WSTRESS)... \n')
    os.system('mkdir -p PNG/WSTRESS')

  taux_val = np.ma.masked_invalid(frc.tauuo.values)
  tauy_val = np.ma.masked_invalid(frc.tauvo.values)

  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,7), gridspec_kw={'width_ratios': [3, 1]})
  xyplot(taux_val, grd.mgeolon_u, grd.geolat_u,
         axis=ax[0], suptitle='Zonal surface stress [Pa] - '+str(args.casename),
         title=str(args.start_date) + ' to '+ str(args.end_date))
  frc.tauuo.mean(dim='xq').plot(ax=ax[1], y="yh")
  ax[1].grid()
  plt.savefig('PNG/WSTRESS/'+str(args.casename)+'_taux.png')
  plt.close()

  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,7), gridspec_kw={'width_ratios': [3, 1]})
  xyplot(tauy_val, grd.mgeolon_v, grd.geolat_v,
         axis=ax[0], suptitle='Meridional surface stress [Pa] - '+str(args.casename),
         title=str(args.start_date) + ' to '+ str(args.end_date))
  frc.tauvo.mean(dim='xh').plot(ax=ax[1], y="yq")
  ax[1].grid()
  plt.savefig('PNG/WSTRESS/'+str(args.casename)+'_tauy.png')
  plt.close()

  # create dataarays
  wind_da = xr.Dataset(
             data_vars=dict(
                taux=(["yh", "xq"], taux_val),
                tauy=(["yq", "xh"], tauy_val),
             ),
             coords=dict(
                lon_u=(["yh", "xq"], grd.geolon_u),
                lat_u=(["yh", "xq"], grd.geolat_u),
                lat_v=(["yq", "xh"], grd.geolat_v),
                lon_v=(["yq", "xh"], grd.geolon_v)
             ),
             attrs = {'start_date': args.start_date,
                      'end_date': args.end_date,
                      'casename': args.casename,
                      'description': 'Mean wind stress (Pa)',
                      'module': os.path.basename(__file__)}
             )

  wind_da.to_netcdf('ncfiles/'+str(args.casename)+'_wind_stress.nc')

  if parallel:
    print('\n Releasing workers...')
    client.close(); cluster.close()

  return

# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': parseCommandLine()

