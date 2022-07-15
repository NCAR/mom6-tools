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
      Compute time-averages of forcing variables.
      ''',
  epilog='Written by Gustavo Marques (gmarques@ucar.edu).')
  parser.add_argument('diag_config_yml_path', type=str, help='''Full path to the yaml file  \
    describing the run and diagnostics to be performed.''')
  parser.add_argument('-sd','--start_date', type=str, default='',
                      help='''Start year to compute averages. Default is to use value set in diag_config_yml_path''')
  parser.add_argument('-ed','--end_date', type=str, default='',
                      help='''End year to compute averages. Default is to use value set in diag_config_yml_path''')
  parser.add_argument('-nw','--number_of_workers',  type=int, default=0,
                      help='''Number of workers to use (default=0, serial job).''')
  parser.add_argument('-debug',   help='''Add priting statements for debugging purposes''', action="store_true")
  optCmdLineArgs = parser.parse_args()
  driver(optCmdLineArgs)

#-- This is where all the action happends, i.e., functions for each diagnostic are called.

def driver(args):
  nw = args.number_of_workers
  if not os.path.isdir('ncfiles'):
    print('Creating a directory to place netCDF files (ncfiles)... \n')
    os.system('mkdir ncfiles')

  # Read in the yaml file
  diag_config_yml = yaml.load(open(args.diag_config_yml_path,'r'), Loader=yaml.Loader)

  # Create the case instance
  dcase = DiagsCase(diag_config_yml['Case'])
  DOUT_S = dcase.get_value('DOUT_S')
  if DOUT_S:
    OUTDIR = dcase.get_value('DOUT_S_ROOT')+'/ocn/hist/'
  else:
    OUTDIR = dcase.get_value('RUNDIR')

  args.casename = dcase.casename
  print('Output directory is:', OUTDIR)
  print('Casename is:', args.casename)
  print('Number of workers: ', nw)

  # set avg dates
  avg = diag_config_yml['Avg']
  if not args.start_date : args.start_date = avg['start_date']
  if not args.end_date : args.end_date = avg['end_date']

  # read grid info
  grd = MOM6grid(OUTDIR+'/'+args.casename+'.mom6.static.nc')

  parallel = False
  if nw > 1:
    parallel = True
    cluster = NCARCluster()
    cluster.scale(args.number_of_workers)
    client = Client(cluster)

  print('Reading forcing dataset...')
  startTime = datetime.now()


  ds = xr.open_mfdataset(OUTDIR+'/'+dcase.casename+'.mom6.frc_*.nc', parallel=parallel)
  # use datetime
  #ds1['time'] = ds1.indexes['time'].to_datetimeindex()

  print('Time elasped: ', datetime.now() - startTime)

  print('Selecting data between {} and {}...'.format(args.start_date, args.end_date))
  startTime = datetime.now()
  ds = ds.sel(time=slice(args.start_date, args.end_date))
  print('Time elasped: ', datetime.now() - startTime)


  if parallel:
    print('\n Releasing workers...')
    client.close(); cluster.close()

  print('{} was run successfully!'.format(os.path.basename(__file__)))

  return

# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': parseCommandLine()

