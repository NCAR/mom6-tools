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
from mom6_tools.m6plot import xycompare, xyplot
from mom6_tools.MOM6grid import MOM6grid

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

  parallel, cluster, client = request_workers(nw)

  print('Reading surface dataset...')
  startTime = datetime.now()
  variables = ['oml','mlotst','tos','SSH', 'SSU', 'SSV', 'speed', 'time_bnds']

  def preprocess(ds):
    ''' Compute montly averages and return the dataset with variables'''
    for v in variables:
      if v not in ds.variables:
        ds[v] = xr.zeros_like(ds.SSH)
    return ds[variables].resample(time="1M", closed='left', \
           keep_attrs=True).mean(dim='time', keep_attrs=True)

  if parallel:
    ds = xr.open_mfdataset(RUNDIR+'/'+dcase.casename+'.mom6.sfc_*.nc', \
         chunks={'time': 365}, parallel=True, data_vars='minimal', \
         coords='minimal', compat='override', preprocess=preprocess)
  else:
    ds = xr.open_mfdataset(RUNDIR+'/'+dcase.casename+'.mom6.sfc_*.nc', \
         data_vars='minimal', coords='minimal', compat='override', preprocess=preprocess)

  print('Time elasped: ', datetime.now() - startTime)

  print('Selecting data between {} and {}...'.format(args.start_date, args.end_date))
  startTime = datetime.now()
  ds = ds.sel(time=slice(args.start_date, args.end_date))
  print('Time elasped: ', datetime.now() - startTime)

  # MLD
  get_MLD(ds, 'mlotst', grd, args)

  # SSH
  get_SSH(ds, 'SSH', grd, args)

  if parallel:
    print('\n Releasing workers...')
    client.close(); cluster.close()

  return

def get_SSH(ds, var, grd, args):
  '''
  Compute a sea level anomaly climatology and compare against obs.
  '''

  if not os.path.isdir('PNG/SLA'):
    print('Creating a directory to place figures (PNG/SLA)... \n')
    os.system('mkdir -p PNG/SLA')

  print('Computing mean sea level climatology...')
  startTime = datetime.now()
  mean_sl_model =ds[var].mean(dim='time').compute()
  print('Time elasped: ', datetime.now() - startTime)

  print('Computing monthly SLA climatology...')
  startTime = datetime.now()
  rms_sla_model = np.sqrt(((ds[var]-ds[var].mean(dim='time'))**2).mean(dim='time')).compute()
  print('Time elasped: ', datetime.now() - startTime)

  # fix month values using pandas. We just want something that xarray an understand
  #mld_model['month'] = pd.date_range('2000-01-15', '2001-01-01',  freq='2SMS')

  # read obs
  filepath = '/glade/work/gmarques/cesm/datasets/Aviso/rms_sla_climatology_remaped_to_tx0.6v1.nc'
  print('\n Reading climatology from: ', filepath)
  rms_sla_aviso = xr.open_dataset(filepath)

  print('\n Plotting...')
  model = np.ma.masked_invalid(rms_sla_model.values)
  aviso = np.ma.masked_invalid(rms_sla_aviso.rms_sla.values)

  fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(14,24))
  xyplot(model, grd.geolon, grd.geolat, area=grd.area_t,
         axis=ax[0], suptitle='RMS of SSH anomaly [m]', clim=(0,0.4),
         title=str(args.casename) + ' ' +str(args.start_date) + ' to '+ str(args.end_date))
  xyplot(aviso, grd.geolon, grd.geolat, area=grd.area_t,
         axis=ax[1], clim=(0,0.4), title='RMS of SSH anomaly (AVISO, 1993-2018)')
  xyplot(model - aviso, grd.geolon, grd.geolat, area=grd.area_t,
         axis=ax[2], title='model - AVISO', clim=(-0.2,0.2))

  plt.savefig('PNG/SLA/'+str(args.casename)+'_RMS_SLA_vs_AVISO.png')
  plt.close()

  # create dataarays
  model_rms_sla_da = xr.DataArray(model, dims=['yh','xh'],
                           coords={'yh' : grd.yh, 'xh' : grd.xh}).rename('rms_sla')

  model_rms_sla_da.to_netcdf('ncfiles/'+str(args.casename)+'_RMS_SLA.nc')

  model_mean_sl_da = xr.DataArray(mean_sl_model.values, dims=['yh','xh'],
                           coords={'yh' : grd.yh, 'xh' : grd.xh}).rename('mean_sl')

  model_mean_sl_da.to_netcdf('ncfiles/'+str(args.casename)+'_mean_sea_level.nc')

  return

def get_MLD(ds, var, grd, args):
  '''
  Compute a MLD climatology and compare against obs.
  '''

  if not os.path.isdir('PNG/MLD'):
    print('Creating a directory to place figures (PNG/MLD)... \n')
    os.system('mkdir -p PNG/MLD')

  print('Computing monthly MLD climatology...')
  startTime = datetime.now()
  mld_model = ds[var].groupby("time.month").mean('time').compute()
  print('Time elasped: ', datetime.now() - startTime)

  # fix month values using pandas. We just want something that xarray an understand
  mld_model['month'] = pd.date_range('2000-01-15', '2001-01-01',  freq='2SMS')

  # read obs
  #filepath = '/glade/work/gmarques/cesm/datasets/Mimoc/MIMOC_ML_v2.2_PT_S_MLP_remapped_to_tx06v1.nc'
  #filepath = '/glade/work/gmarques/cesm/datasets/MLD/ARGO_MLD_remapped_to_tx06v1.nc'
  filepath = '/glade/work/gmarques/cesm/datasets/MLD/deBoyer/deBoyer_MLD_remapped_to_tx06v1.nc'
  print('\n Reading climatology from: ', filepath)
  mld_obs = xr.open_dataset(filepath)

  print('\n Plotting...')
  # March and Sep, noticed starting from 0
  months = [2,8]
  for t in months:
    model = np.ma.masked_invalid(mld_model[t,:].values)
    obs = np.ma.masked_invalid(mld_obs.mld[t,:].values)
    month = date(1900, t+1, 1).strftime('%B')
    xycompare(model , obs, grd.geolon, grd.geolat, area=grd.area_t,
            title1 = 'model, '+str(month),
            title2 = 'obs (deBoyer), '+str(month),
            suptitle=args.casename +', ' + str(args.start_date) + ' to ' + str(args.end_date),
            colormap=plt.cm.Spectral_r, dcolormap=plt.cm.bwr, clim = (0,1500), extend='max',
            save = 'PNG/MLD/'+str(args.casename)+'_MLD_'+str(month)+'.png')

    xyplot(model, grd.geolon, grd.geolat, area=grd.area_t,
           save='PNG/MLD/'+str(args.casename)+'_MLD_model_'+str(month)+'.png',
           suptitle=ds[var].attrs['long_name'] +' ['+ ds[var].attrs['units']+']', clim=(0,1500),
           title=str(args.casename) + ' ' +str(args.start_date) + ' to '+ str(args.end_date))

  # JFM, starting from 0
  months = [0,1,2]
  model_JFM = np.ma.masked_invalid(mld_model.isel(month=months).mean('month').values)
  obs_JFM = np.ma.masked_invalid(mld_obs.mld.isel(time=months).mean('time').values)
  month = 'JFM'
  xycompare(model_JFM , obs_JFM, grd.geolon, grd.geolat, area=grd.area_t,
            title1 = 'model, '+str(month),
            title2 = 'obs (deBoyer), '+str(month),
            suptitle=args.casename +', ' + str(args.start_date) + ' to ' + str(args.end_date),
            colormap=plt.cm.Spectral_r, dcolormap=plt.cm.bwr, clim = (0,1500), extend='max',
            save = 'PNG/MLD/'+str(args.casename)+'_MLD_'+str(month)+'.png')

  # JAS, starting from 0
  months = [6,7,8]
  model_JAS = np.ma.masked_invalid(mld_model.isel(month=months).mean('month').values)
  obs_JAS = np.ma.masked_invalid(mld_obs.mld.isel(time=months).mean('time').values)
  month = 'JAS'
  xycompare(model_JAS , obs_JAS, grd.geolon, grd.geolat, area=grd.area_t,
            title1 = 'model, '+str(month),
            title2 = 'obs (deBoyer), '+str(month),
            suptitle=args.casename +', ' + str(args.start_date) + ' to ' + str(args.end_date),
            colormap=plt.cm.Spectral_r, dcolormap=plt.cm.bwr, clim = (0,1500), extend='max',
            save = 'PNG/MLD/'+str(args.casename)+'_MLD_'+str(month)+'.png')

  # Winter, JFM (NH) and JAS (SH)
  model_winter = model_JAS.copy(); obs_winter = obs_JAS.copy()
  # find point closest to eq. and select data
  j = np.abs( grd.geolat[:,0] - 0. ).argmin()
  model_winter[j::,:] = model_JFM[j::,:]; obs_winter[j::,:] = obs_JFM[j::,:]
  # create dataarays
  model_winter_da = xr.DataArray(model_winter, dims=['yh','xh'],
                           coords={'yh' : grd.yh, 'xh' : grd.xh}).rename('MLD_winter')

  month = 'winter'
  model_winter_da.to_netcdf('ncfiles/'+str(args.casename)+'_MLD_'+month+'.nc')
  xycompare(model_winter , obs_winter, grd.geolon, grd.geolat, area=grd.area_t,
            title1 = 'model, JFM (NH), JAS (SH)',
            title2 = 'obs (deBoyer), JFM (NH), JAS (SH)',
            suptitle=args.casename +', ' + str(args.start_date) + ' to ' + str(args.end_date),
            colormap=plt.cm.Spectral_r, dcolormap=plt.cm.bwr, clim = (0,1500), extend='max',
            save = 'PNG/MLD/'+str(args.casename)+'_MLD_'+str(month)+'.png')

  xyplot(model_winter, grd.geolon, grd.geolat, area=grd.area_t,
         save='PNG/MLD/'+str(args.casename)+'_MLD_model_'+str(month)+'.png',
         suptitle=ds[var].attrs['long_name'] +' ['+ ds[var].attrs['units']+']', clim=(0,1500),
         title=str(args.casename) + ' ' +str(args.start_date) + ' to '+ str(args.end_date) + \
              ' JFM (NH), JAS (SH)')

  # Summer, JFM (SH) and JAS (NH)
  model_summer = model_JAS.copy(); obs_summer = obs_JAS.copy()
  model_summer[0:j,:] = model_JFM[0:j,:]; obs_summer[0:j,:] = obs_JFM[0:j,:]
  # create dataarays
  model_summer_da = xr.DataArray(model_summer, dims=['yh','xh'],
                           coords={'yh' : grd.yh, 'xh' : grd.xh}).rename('MLD_summer')

  month = 'summer'
  model_summer_da.to_netcdf('ncfiles/'+str(args.casename)+'_MLD_'+month+'.nc')
  xycompare(model_summer , obs_summer, grd.geolon, grd.geolat, area=grd.area_t,
            title1 = 'model, JFM (SH), JAS (NH)',
            title2 = 'obs (deBoyer), JFM (SH), JAS (NH)',
            suptitle=args.casename +', ' + str(args.start_date) + ' to ' + str(args.end_date),
            colormap=plt.cm.Spectral_r, dcolormap=plt.cm.bwr, clim = (0,150), extend='max',
            save = 'PNG/MLD/'+str(args.casename)+'_MLD_'+str(month)+'.png')

  xyplot(model_summer, grd.geolon, grd.geolat, area=grd.area_t,
         save='PNG/MLD/'+str(args.casename)+'_MLD_model_'+str(month)+'.png',
         suptitle=ds[var].attrs['long_name'] +' ['+ ds[var].attrs['units']+']', clim=(0,150),
         title=str(args.casename) + ' ' +str(args.start_date) + ' to '+ str(args.end_date) + \
              ' JFM (SH), JAS (NH)')
  return

# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': parseCommandLine()

