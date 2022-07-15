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
from mom6_tools.wright_eos import alpha_wright_eos, beta_wright_eos
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
  parser.add_argument('-g','--gravity',  type=float, default=9.8,
                      help='''The gravitational acceleration of the Earth. (default = 9.8 m s-2)''')
  parser.add_argument('-rho_0','--mean_density',  type=float, default=1035.0,
                      help='''The mean ocean density. (default = 1035.0 kg m-3)''')
  parser.add_argument('-c_p','--heat_capacity',  type=float, default=3992.0,
                      help='''The heat capacity of sea water. (default = 3992.0 J kg-1 K-1)''')
  parser.add_argument('-debug',   help='''Add priting statements for debugging purposes''', action="store_true")
  optCmdLineArgs = parser.parse_args()
  driver(optCmdLineArgs)

#-- This is where all the action happends, i.e., functions for each diagnostic are called.

def driver(args):
  nw = args.number_of_workers
  fname = args.file_name
  g = args.gravity
  rho_0 = args.mean_density
  c_p = args.heat_capacity
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

  def preprocess1(ds):
    ''' Return the dataset with variables'''
    variables = ['hfds','PRCmE', 'time_bnds']
    return ds[variables]

  ds1 = xr.open_mfdataset(RUNDIR+'/'+dcase.casename+fname, parallel=parallel)

  ds1 = preprocess1(ds1)

  def preprocess2(ds):
    ''' Return the dataset with variables'''
    variables = ['tos', 'sos', 'time_bnds']
    return ds[variables]

  ds2 = xr.open_mfdataset(RUNDIR+'/'+dcase.casename+'.mom6.hm_*.nc', parallel=parallel)

  ds2 = preprocess2(ds2)

  print('Time elasped: ', datetime.now() - startTime)

  print('Selecting data between {} and {}...'.format(args.start_date, args.end_date))
  startTime = datetime.now()
  ds1 = ds1.sel(time=slice(args.start_date, args.end_date))
  ds2 = ds2.sel(time=slice(args.start_date, args.end_date))
  print('Time elasped: ', datetime.now() - startTime)

  print('Averaging in time...')
  startTime = datetime.now()
  frc = ds1.mean('time').compute()
  state = ds2.mean('time').compute()
  print('Time elasped: ', datetime.now() - startTime)

  # bouyancy flux computed using Eq.(1) in https://doi.org/10.1002/2016GL070058

  # alpha [kg m-2 C-1] need to divide by rho_0 to get C-1
  alpha = alpha_wright_eos(state.tos,state.sos,np.zeros(state.sos.shape))/rho_0
  # beta [kg m-3 PSU-1] need to divide by rho_0 to get PSU-1
  beta = beta_wright_eos(state.tos,state.sos,np.zeros(state.sos.shape))/rho_0

  alpha.to_netcdf('ncfiles/'+str(args.casename)+'_alpha.nc')
  # BHF [m s-3], alpha is negative so we need a - sign below
  BHF = -alpha * g * frc.hfds / (c_p * rho_0)

  # BFW [m s-3] need to x 10-3 to covert from kg to m3
  BFW = beta * state.sos * frc.PRCmE * g * 1.0e-3

  print('\n Plotting...')
  if not os.path.isdir('PNG/BFLUX'):
    print('Creating a directory to place figures (PNG/BFLUX)... \n')
    os.system('mkdir -p PNG/BFLUX')

  bhf_val = np.ma.masked_invalid(BHF.values*1.0e8)
  bfw_val = np.ma.masked_invalid(BFW.values*1.0e8)
  b_val = np.ma.masked_invalid((BHF+BFW).values*1.0e8)

  fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(14,24))
  xyplot(bhf_val, grd.geolon, grd.geolat, area=grd.area_t,
         axis=ax[0], title='Bouyancy flux due to heat [10$^{-8}$ m$^2$ s^{-3}]', #clim=(0,0.4),
         suptitle=str(args.casename) + ' ' +str(args.start_date) + ' to '+ str(args.end_date))
  xyplot(bfw_val, grd.geolon, grd.geolat, area=grd.area_t, #clim=(0,0.4)
         axis=ax[1], title='Bouyancy flux due to fresh water [10$^{-8}$ m$^2$ s^{-3}]')
  xyplot(b_val, grd.geolon, grd.geolat, area=grd.area_t,
         axis=ax[2], title='Total bouyancy flux  [10$^{-8}$ m$^2$ s^{-3}]') #clim=(-0.2,0.2))

  plt.savefig('PNG/BFLUX/'+str(args.casename)+'_bouyancy_flux.png')
  plt.close()

  fig, ax = plt.subplots(nrows=1, ncols=1)
  (BHF.mean(dim='xh')*1.0e8).plot(ax=ax,label='Heat')
  (BFW.mean(dim='xh')*1.0e8).plot(ax=ax,label='Freshwater')
  ((BFW+BHF).mean(dim='xh')*1.0e8).plot(ax=ax,label='Total')
  ax.legend(); ax.grid()
  ax.set_title('Bouyancy Flux [10$^{-8}$ m$^2$ s$^{-3}$]')
  plt.suptitle(str(args.casename) + ' ' +str(args.start_date) + ' to '+ str(args.end_date))
  plt.savefig('PNG/BFLUX/'+str(args.casename)+'_bouyancy_flux_profile.png')
  plt.close()

  # create dataarays
  bflux_da = xr.Dataset(
             data_vars=dict(
                bhf=(["yh", "xh"], bhf_val),
                bfw=(["yh", "xh"], bfw_val),
                btot=(["yh", "xh"], b_val),
             ),
             coords=dict(
                lon=(["yh", "xh"], grd.geolon),
                lat=(["yh", "xh"], grd.geolat)
             ),
             attrs = {'start_date': args.start_date,
                      'end_date': args.end_date,
                      'casename': args.casename,
                      'description': 'Mean Bouyancy flux: BHF, BFW, and total (10-8 m2 s-3)',
                      'module': os.path.basename(__file__)}
             )

  #add_global_attrs(bflux_da,attrs)
  bflux_da.to_netcdf('ncfiles/'+str(args.casename)+'_bouyancy_flux.nc')

  if parallel:
    print('\n Releasing workers...')
    client.close(); cluster.close()

  print('{} was run successfully!'.format(os.path.basename(__file__)))

  return

# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': parseCommandLine()

