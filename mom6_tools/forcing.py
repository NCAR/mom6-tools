#!/usr/bin/env python

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import warnings, os, yaml
import pandas as pd
import dask
from datetime import datetime, date
from mom6_tools.jobqueue import get_cluster, release_workers
from mom6_tools.cli import get_parser
from mom6_tools.DiagsCase import DiagsCase
from mom6_tools.m6toolbox import add_global_attrs
from mom6_tools.m6plot import xycompare, xyplot
from mom6_tools.MOM6grid import MOM6grid

def parse_args():
  parser = get_parser(description='''Compute time-averages of forcing variables.''',
                      epilog='Written by Gustavo Marques (gmarques@ucar.edu).')
  return parser.parse_args()

#-- This is where all the action happends, i.e., functions for each diagnostic are called.

def driver(args):
  nw = args.number_of_workers

  # Read in the yaml file and create the case instance
  dcase = DiagsCase(args.diag_config_yml_path)
  ocn_diag_root = dcase.ocn_diag_root

  args.casename = dcase.casename
  print('Casename is:', args.casename)
  print('Number of workers: ', nw)

  # set avg dates
  if not args.start_date : args.start_date = dcase.start_date
  if not args.end_date : args.end_date = dcase.end_date

  # read grid info
  grd = dcase.get_grid()

  parallel, cluster, client = get_cluster(args.number_of_workers, args=args,
                                          config=dcase.jobqueue_config)

  print('Reading forcing dataset...')
  startTime = datetime.now()


  ds = xr.open_mfdataset(dcase.get_fname('native'), parallel=parallel)
  # use datetime
  #ds1['time'] = ds1.indexes['time'].to_datetimeindex()

  print('Time elasped: ', datetime.now() - startTime)

  print('Selecting data between {} and {}...'.format(args.start_date, args.end_date))
  startTime = datetime.now()
  ds = ds.sel(time=slice(args.start_date, args.end_date))
  print('Time elasped: ', datetime.now() - startTime)


  release_workers(parallel, cluster, client)

  print('{} was run successfully!'.format(os.path.basename(__file__)))

  return



def main():
  '''
  Main procedure that calls the driver.
  '''
  args = parse_args()
  driver(args)

# Invoke main() which calls parse_args() and the driver.
if __name__ == '__main__':
  main()

