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
from mom6_tools.m6toolbox import request_workers
from mom6_tools.m6plot import xycompare, polarcomparison
from mom6_tools.MOM6grid import MOM6grid

def parseCommandLine():
  """
  Parse the command line positional and optional arguments.
  This is the highest level procedure invoked from the very end of the script.
  """
  parser = argparse.ArgumentParser(description=
      '''
      Compares (model - obs) T and S at depth levels
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
  if not os.path.isdir('PNG/TS_levels'):
    print('Creating a directory to place figures (PNG)... \n')
    os.system('mkdir -p PNG/TS_levels')
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

  # load PHC2 data
  phc_path = '/glade/p/cesm/omwg/obs_data/phc/'
  phc_temp = xr.open_mfdataset(phc_path+'PHC2_TEMP_tx0.66v1_34lev_ann_avg.nc', decode_coords=False, decode_times=False)
  phc_salt = xr.open_mfdataset(phc_path+'PHC2_SALT_tx0.66v1_34lev_ann_avg.nc', decode_coords=False, decode_times=False)

  parallel, cluster, client = request_workers(nw)

  print('Reading surface dataset...')
  startTime = datetime.now()
  variables = ['thetao', 'so', 'time', 'time_bnds']

  def preprocess(ds):
    ''' Compute montly averages and return the dataset with variables'''
    return ds[variables]#.resample(time="1Y", closed='left', \
           #keep_attrs=True).mean(dim='time', keep_attrs=True)

  if parallel:
    ds = xr.open_mfdataset(RUNDIR+'/'+dcase.casename+'.mom6.h_*.nc', \
         parallel=True, data_vars='minimal', \
         coords='minimal', compat='override', preprocess=preprocess)
  else:
    ds = xr.open_mfdataset(RUNDIR+'/'+dcase.casename+'.mom6.h_*.nc', \
         data_vars='minimal', coords='minimal', compat='override', preprocess=preprocess)

  print('Time elasped: ', datetime.now() - startTime)

  print('Selecting data between {} and {}...'.format(args.start_date, args.end_date))
  startTime = datetime.now()
  ds = ds.sel(time=slice(args.start_date, args.end_date))
  print('Time elasped: ', datetime.now() - startTime)

  print('\n Computing yearly means...')
  startTime = datetime.now()
  ds = ds.resample(time="1Y", closed='left',keep_attrs=True).mean('time',keep_attrs=True)
  print('Time elasped: ', datetime.now() - startTime)

  print('Time averaging...')
  startTime = datetime.now()
  temp = np.ma.masked_invalid(ds.thetao.mean('time').values)
  salt = np.ma.masked_invalid(ds.so.mean('time').values)
  print('Time elasped: ', datetime.now() - startTime)

  print('Saving netCDF files...')
  startTime = datetime.now()
  thetao = xr.DataArray(ds.thetao.mean('time'), dims=['z_l','yh','xh'],
              coords={'z_l' : ds.z_l, 'yh' : grd.yh, 'xh' : grd.xh}).rename('thetao')
  thetao.to_netcdf('ncfiles/'+str(args.casename)+'_thetao_time_mean.nc')
  so = xr.DataArray(ds.so.mean('time'), dims=['z_l','yh','xh'],
              coords={'z_l' : ds.z_l, 'yh' : grd.yh, 'xh' : grd.xh}).rename('so')
  so.to_netcdf('ncfiles/'+str(args.casename)+'_so_time_mean.nc')
  print('Time elasped: ', datetime.now() - startTime)

  if parallel:
    print('\n Releasing workers...')
    client.close(); cluster.close()

  print('Global plots...')
  km = len(phc_temp['depth'])
  for k in range(km):
    if ds['z_l'][k].values < 1200.0:
      figname = 'PNG/TS_levels/'+str(dcase.casename)+'_'+str(ds['z_l'][k].values)+'_'
      temp_obs = np.ma.masked_invalid(phc_temp['TEMP'][k,:].values)
      xycompare(temp[k,:] , temp_obs, grd.geolon, grd.geolat, area=grd.area_t,
              title1 = 'model temperature, depth ='+str(ds['z_l'][k].values)+ 'm',
              title2 = 'observed temperature, depth ='+str(phc_temp['depth'][k].values)+ 'm',
              suptitle=dcase.casename + ', averaged '+str(args.start_date)+ ' to ' +str(args.end_date),
              extend='both', dextend='neither', clim=(-1.9,30.), dlim=(-2,2), dcolormap=plt.cm.bwr,
              save=figname+'global_temp.png')
      salt_obs = np.ma.masked_invalid(phc_salt['SALT'][k,:].values)
      xycompare( salt[k,:] , salt_obs, grd.geolon, grd.geolat, area=grd.area_t,
              title1 = 'model salinity, depth ='+str(ds['z_l'][k].values)+ 'm',
              title2 = 'observed salinity, depth ='+str(phc_temp['depth'][k].values)+ 'm',
              suptitle=dcase.casename + ', averaged '+str(args.start_date)+ ' to ' +str(args.end_date),
              extend='both', dextend='neither', clim=(30.,39.), dlim=(-2,2), dcolormap=plt.cm.bwr,
              save=figname+'global_salt.png')

  print('Antarctic plots...')
  for k in range(km):
    if (ds['z_l'][k].values < 1200.):
      temp_obs = np.ma.masked_invalid(phc_temp['TEMP'][k,:].values)
      polarcomparison(temp[k,:] , temp_obs, grd,
              title1 = 'model temperature, depth ='+str(ds['z_l'][k].values)+ 'm',
              title2 = 'observed temperature, depth ='+str(phc_temp['depth'][k].values)+ 'm',
              extend='both', dextend='neither', clim=(-1.9,10.5), dlim=(-2,2), dcolormap=plt.cm.bwr,
              suptitle=dcase.casename + ', averaged '+str(args.start_date)+ ' to ' +str(args.end_date),
              proj='SP', save=figname+'antarctic_temp.png')
      salt_obs = np.ma.masked_invalid(phc_salt['SALT'][k,:].values)
      polarcomparison( salt[k,:] , salt_obs, grd,
              title1 = 'model salinity, depth ='+str(ds['z_l'][k].values)+ 'm',
              title2 = 'observed salinity, depth ='+str(phc_temp['depth'][k].values)+ 'm',
              extend='both', dextend='neither', clim=(33.,35.), dlim=(-2,2), dcolormap=plt.cm.bwr,
              suptitle=dcase.casename + ', averaged '+str(args.start_date)+ ' to ' +str(args.end_date),
              proj='SP', save=figname+'antarctic_salt.png')

  print('Arctic plots...')
  for k in range(km):
    if (ds['z_l'][k].values < 100.):
      temp_obs = np.ma.masked_invalid(phc_temp['TEMP'][k,:].values)
      polarcomparison(temp[k,:] , temp_obs, grd,
              title1 = 'model temperature, depth ='+str(ds['z_l'][k].values)+ 'm',
              title2 = 'observed temperature, depth ='+str(phc_temp['depth'][k].values)+ 'm',
              extend='both', dextend='neither', clim=(-1.9,11.5), dlim=(-2,2), dcolormap=plt.cm.bwr,
              suptitle=dcase.casename + ', averaged '+str(args.start_date)+ ' to ' +str(args.end_date),
              proj='NP', save=figname+'arctic_temp.png')
      salt_obs = np.ma.masked_invalid(phc_salt['SALT'][k,:].values)
      polarcomparison( salt[k,:] , salt_obs, grd,
              title1 = 'model salinity, depth ='+str(ds['z_l'][k].values)+ 'm',
              title2 = 'observed salinity, depth ='+str(phc_temp['depth'][k].values)+ 'm',
              extend='both', dextend='neither', clim=(31.5,35.), dlim=(-2,2), dcolormap=plt.cm.bwr,
              suptitle=dcase.casename + ', averaged '+str(args.start_date)+ ' to ' +str(args.end_date),
              proj='NP', save=figname+'arctic_salt.png')
  return

# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': parseCommandLine()

