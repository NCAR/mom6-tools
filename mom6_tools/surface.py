#!/usr/bin/env python

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import warnings, os, yaml, argparse
import pandas as pd
import dask, intake
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
  parser.add_argument('-mld_obs','--mld_obs', type=str, default='mld-deboyer-tx2_3v2',
                      help='''Name of the observation-based MLD dataset in the oce-catalog. Default is mld-deboyer-tx2_3v2''')
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
    OUTDIR = dcase.get_value('RUNDIR')+'/'

  args.casename = dcase.casename
  print('Output directory is:', OUTDIR)
  print('Casename is:', args.casename)
  print('Number of workers: ', nw)

  # set avg dates + other params
  avg = diag_config_yml['Avg']
  if not args.start_date : args.start_date = avg['start_date']
  if not args.end_date : args.end_date = avg['end_date']
  args.native = dcase.casename+diag_config_yml['Fnames']['native']
  args.static = dcase.casename+diag_config_yml['Fnames']['static']
  args.savefigs = True

  # read grid info
  grd = MOM6grid(OUTDIR+args.static)

  parallel = False
  if nw > 1:
    parallel = True
    cluster = NCARCluster()
    cluster.scale(args.number_of_workers)
    client = Client(cluster)

  print('Reading surface dataset...')
  startTime = datetime.now()

  def preprocess(ds):
    ''' Compute montly averages and return the dataset with variables'''
    variables = ['oml','mlotst','tos','SSH', 'SSU', 'SSV', 'speed']
    if 'time_bounds' in ds.variables:
      variables.append('time_bounds')
    elif 'time_bnds' in ds.variables:
      variables.append('time_bnds')
    for v in variables:
      if v not in ds.variables:
        ds[v] = xr.zeros_like(ds.SSH)
    return ds[variables]

  ds1 = xr.open_mfdataset(OUTDIR+'/'+args.native, parallel=parallel)
  # use datetime
  #ds1['time'] = ds1.indexes['time'].to_datetimeindex()

  ds = preprocess(ds1)

  print('Time elasped: ', datetime.now() - startTime)

  print('Selecting data between {} and {}...'.format(args.start_date, args.end_date))
  startTime = datetime.now()
  ds = ds.sel(time=slice(args.start_date, args.end_date))
  print('Time elasped: ', datetime.now() - startTime)

  # load obs-based mld from oce-catalog
  catalog = intake.open_catalog(diag_config_yml['oce_cat'])
  mld_obs = catalog[args.mld_obs].to_dask()

  # MLD
  get_MLD(ds, 'mlotst', mld_obs, grd, args)

  # BLD
  get_BLD(ds, 'oml', grd, args)

  # TODO: SSH
  #get_SSH(ds, 'SSH', grd, args)

  if parallel:
    print('\n Releasing workers...')
    client.close(); cluster.close()

  print('{} was run successfully!'.format(os.path.basename(__file__)))

  return

def get_SSH(ds, var, grd, args):
  '''
  Compute a sea level anomaly climatology and compare against obs.
  '''

  if args.savefigs:
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

  try:
    area = grd.area_t
  except:
    area = grd.areacello

  fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(14,24))
  xyplot(model, grd.geolon, grd.geolat, area=area,
         axis=ax[0], suptitle='RMS of SSH anomaly [m]', clim=(0,0.4),
         title=str(args.casename) + ' ' +str(args.start_date) + ' to '+ str(args.end_date))
  xyplot(aviso, grd.geolon, grd.geolat, area=area,
         axis=ax[1], clim=(0,0.4), title='RMS of SSH anomaly (AVISO, 1993-2018)')
  xyplot(model - aviso, grd.geolon, grd.geolat, area=area,
         axis=ax[2], title='model - AVISO', clim=(-0.2,0.2))

  if args.savefigs:
    plt.savefig('PNG/SLA/'+str(args.casename)+'_RMS_SLA_vs_AVISO.png')
  plt.close()

  # create dataarays
  model_rms_sla_da = xr.DataArray(model, dims=['yh','xh'],
                           coords={'yh' : grd.yh, 'xh' : grd.xh}).rename('rms_sla')

  attrs = {'start_date': args.start_date,
           'end_date': args.end_date,
           'casename': args.casename,
           'description': 'RMS of SSH anomaly (AVISO, 1993-2018)',
           'obs': 'AVISO',
           'module': os.path.basename(__file__)}
  add_global_attrs(model_rms_sla_da,attrs)
  model_rms_sla_da.to_netcdf('ncfiles/'+str(args.casename)+'_RMS_SLA.nc')

  model_mean_sl_da = xr.DataArray(mean_sl_model.values, dims=['yh','xh'],
                           coords={'yh' : grd.yh, 'xh' : grd.xh}).rename('mean_sl')
  attrs['description'] = 'Mean sea level climatology'
  model_mean_sl_da.to_netcdf('ncfiles/'+str(args.casename)+'_mean_sea_level.nc')

  return

def get_MLD(ds, var, mld_obs, grd, args):
  '''
  Compute a MLD climatology and compare against obs.
  '''

  if args.savefigs:
    if not os.path.isdir('PNG/MLD'):
      print('Creating a directory to place figures (PNG/MLD)... \n')
      os.system('mkdir -p PNG/MLD')

  print('Computing monthly MLD climatology...')
  startTime = datetime.now()
  mld_model = ds[var].groupby("time.month").mean('time').compute()
  print('Time elasped: ', datetime.now() - startTime)

  # fix month values using pandas. We just want something that xarray understands
  mld_model['month'] = pd.date_range('2000-01-15', '2001-01-01',  freq='2SMS')

  try:
    area = grd.area_t
  except:
    area = grd.areacello

  print('\n Plotting...')
  # March and Sep, noticed starting from 0
  months = [2,8]
  fname = None

  for t in months:
    month = date(1900, t+1, 1).strftime('%B')
    if args.savefigs:
      fname = 'PNG/MLD/'+str(args.casename)+'_MLD_'+str(month)+'.png'
      model = np.ma.masked_invalid(mld_model[t,:].values)
      obs = np.ma.masked_invalid(mld_obs.mld[t,:].values)
      obs = np.ma.masked_where(grd.wet == 0, obs)
      xycompare(model , obs, grd.geolon, grd.geolat, area=area,
            title1 = 'model, '+str(month),
            title2 = 'obs (deBoyer), '+str(month),
            suptitle=args.casename +', ' + str(args.start_date) + ' to ' + str(args.end_date),
            colormap=plt.cm.Spectral_r, dcolormap=plt.cm.bwr, clim = (0,1500), extend='max',
            save = fname)

    if args.savefigs:
      fname = 'PNG/MLD/'+str(args.casename)+'_MLD_model_'+str(month)+'.png'
      xyplot(model, grd.geolon, grd.geolat, area=area,
           save=fname,
           suptitle=ds[var].attrs['long_name'] +' ['+ ds[var].attrs['units']+']', clim=(0,1500),
           title=str(args.casename) + ' ' +str(args.start_date) + ' to '+ str(args.end_date))

  # JFM, starting from 0
  months = [0,1,2]
  model_JFM = np.ma.masked_invalid(mld_model.isel(month=months).mean('month').values)
  obs_JFM = np.ma.masked_invalid(mld_obs.mld.isel(time=months).mean('time').values)
  obs_JFM = np.ma.masked_where(grd.wet == 0, obs_JFM)
  month = 'JFM'
  if args.savefigs:
    fname = 'PNG/MLD/'+str(args.casename)+'_MLD_'+str(month)+'.png'
    xycompare(model_JFM , obs_JFM, grd.geolon, grd.geolat, area=area,
            title1 = 'model, '+str(month),
            title2 = 'obs (deBoyer), '+str(month),
            suptitle=args.casename +', ' + str(args.start_date) + ' to ' + str(args.end_date),
            colormap=plt.cm.Spectral_r, dcolormap=plt.cm.bwr, clim = (0,1500), extend='max',
            save = fname)

  # JAS, starting from 0
  months = [6,7,8]
  model_JAS = np.ma.masked_invalid(mld_model.isel(month=months).mean('month').values)
  obs_JAS = np.ma.masked_invalid(mld_obs.mld.isel(time=months).mean('time').values)
  obs_JAS = np.ma.masked_where(grd.wet == 0, obs_JAS)
  month = 'JAS'
  if args.savefigs:
    fname = 'PNG/MLD/'+str(args.casename)+'_MLD_'+str(month)+'.png'
    xycompare(model_JAS , obs_JAS, grd.geolon, grd.geolat, area=area,
            title1 = 'model, '+str(month),
            title2 = 'obs (deBoyer), '+str(month),
            suptitle=args.casename +', ' + str(args.start_date) + ' to ' + str(args.end_date),
            colormap=plt.cm.Spectral_r, dcolormap=plt.cm.bwr, clim = (0,1500), extend='max',
            save = fname)

  # Winter, JFM (NH) and JAS (SH)
  model_winter = model_JAS.copy(); obs_winter = obs_JAS.copy()
  # find point closest to eq. and select data
  j = np.abs( grd.geolat[:,0] - 0. ).argmin()
  model_winter[j::,:] = model_JFM[j::,:]; obs_winter[j::,:] = obs_JFM[j::,:]
  # create dataarays
  model_winter_da = xr.DataArray(model_winter, dims=['yh','xh'],
                           coords={'yh' : grd.yh, 'xh' : grd.xh}).rename('MLD_winter')

  month = 'winter'
  attrs = {'start_date': args.start_date,
           'end_date': args.end_date,
           'casename': args.casename,
           'description': 'Winter MLD (m)',
           'module': os.path.basename(__file__)}
  add_global_attrs(model_winter_da,attrs)
  model_winter_da.to_netcdf('ncfiles/'+str(args.casename)+'_MLD_'+month+'.nc')
  if args.savefigs:
    fname = 'PNG/MLD/'+str(args.casename)+'_MLD_'+str(month)+'.png'
    xycompare(model_winter , obs_winter, grd.geolon, grd.geolat, area=area,
            title1 = 'model, JFM (NH), JAS (SH)',
            title2 = 'obs (deBoyer), JFM (NH), JAS (SH)',
            suptitle=args.casename +', ' + str(args.start_date) + ' to ' + str(args.end_date),
            colormap=plt.cm.Spectral_r, dcolormap=plt.cm.bwr, clim = (0,1500), extend='max',
            save = fname)

  if args.savefigs:
    fname = 'PNG/MLD/'+str(args.casename)+'_MLD_model_'+str(month)+'.png'
    xyplot(model_winter, grd.geolon, grd.geolat, area=area,
         save=fname,
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
  attrs['description'] = 'Summer MLD (m)'
  add_global_attrs(model_summer_da,attrs)
  model_summer_da.to_netcdf('ncfiles/'+str(args.casename)+'_MLD_'+month+'.nc')
  if args.savefigs:
    fname = 'PNG/MLD/'+str(args.casename)+'_MLD_'+str(month)+'.png'
    xycompare(model_summer , obs_summer, grd.geolon, grd.geolat, area=area,
            title1 = 'model, JFM (SH), JAS (NH)',
            title2 = 'obs (deBoyer), JFM (SH), JAS (NH)',
            suptitle=args.casename +', ' + str(args.start_date) + ' to ' + str(args.end_date),
            colormap=plt.cm.Spectral_r, dcolormap=plt.cm.bwr, clim = (0,150), extend='max',
            save = fname)

  if args.savefigs:
    fname = 'PNG/MLD/'+str(args.casename)+'_MLD_model_'+str(month)+'.png'
    xyplot(model_summer, grd.geolon, grd.geolat, area=area,
         save=fname,
         suptitle=ds[var].attrs['long_name'] +' ['+ ds[var].attrs['units']+']', clim=(0,150),
         title=str(args.casename) + ' ' +str(args.start_date) + ' to '+ str(args.end_date) + \
              ' JFM (SH), JAS (NH)')
  return

def get_BLD(ds, var, grd, args):
  '''
  Compute and save a surface BLD climatology.
  TODO: compare against obs
  '''
  if args.savefigs:
    if not os.path.isdir('PNG/BLD'):
      print('Creating a directory to place figures (PNG/BLD)... \n')
      os.system('mkdir -p PNG/BLD')

  print('Computing monthly BLD climatology...')
  startTime = datetime.now()
  mld_model = ds[var].groupby("time.month").mean('time').compute()
  print('Time elasped: ', datetime.now() - startTime)

  # fix month values using pandas. We just want something that xarray understands
  mld_model['month'] = pd.date_range('2000-01-15', '2001-01-01',  freq='2SMS')

  print('\n Plotting...')
  try:
    area = grd.area_t
  except:
    area = grd.areacello

  # March and Sep, noticed starting from 0
  months = [2,8]
  fname = None
  for t in months:
    month = date(1900, t+1, 1).strftime('%B')
    if args.savefigs:
      fname = 'PNG/BLD/'+str(args.casename)+'_BLD_model_'+str(month)+'.png'
    model = np.ma.masked_invalid(mld_model[t,:].values)
    xyplot(model, grd.geolon, grd.geolat, area=area,
           save=fname,
           suptitle=ds[var].attrs['long_name'] +' ['+ ds[var].attrs['units']+']', clim=(0,1500),
           title=str(args.casename) + ' ' +str(args.start_date) + ' to '+ str(args.end_date))

  # JFM, starting from 0
  months = [0,1,2]
  model_JFM = np.ma.masked_invalid(mld_model.isel(month=months).mean('month').values)
  month = 'JFM'
  # JAS, starting from 0
  months = [6,7,8]
  model_JAS = np.ma.masked_invalid(mld_model.isel(month=months).mean('month').values)
  month = 'JAS'

  # Winter, JFM (NH) and JAS (SH)
  model_winter = model_JAS.copy()
  # find point closest to eq. and select data
  j = np.abs( grd.geolat[:,0] - 0. ).argmin()
  model_winter[j::,:] = model_JFM[j::,:]
  # create dataarays
  model_winter_da = xr.DataArray(model_winter, dims=['yh','xh'],
                           coords={'yh' : grd.yh, 'xh' : grd.xh}).rename('BLD_winter')

  month = 'winter'
  attrs = {'start_date': args.start_date,
           'end_date': args.end_date,
           'casename': args.casename,
           'description': 'Winter MLD (m)',
           'module': os.path.basename(__file__)}
  add_global_attrs(model_winter_da,attrs)
  model_winter_da.to_netcdf('ncfiles/'+str(args.casename)+'_BLD_'+month+'.nc')

  if args.savefigs:
    fname = 'PNG/BLD/'+str(args.casename)+'_BLD_model_'+str(month)+'.png'
  xyplot(model_winter, grd.geolon, grd.geolat, area=area,
         save=fname,
         suptitle=ds[var].attrs['long_name'] +' ['+ ds[var].attrs['units']+']', clim=(0,1500),
         title=str(args.casename) + ' ' +str(args.start_date) + ' to '+ str(args.end_date) + \
              ' JFM (NH), JAS (SH)')

  # Summer, JFM (SH) and JAS (NH)
  model_summer = model_JAS.copy()
  model_summer[0:j,:] = model_JFM[0:j,:]
  # create dataarays
  model_summer_da = xr.DataArray(model_summer, dims=['yh','xh'],
                           coords={'yh' : grd.yh, 'xh' : grd.xh}).rename('BLD_summer')

  month = 'summer'
  attrs['description'] = 'Summer BLD (m)'
  add_global_attrs(model_summer_da,attrs)
  model_summer_da.to_netcdf('ncfiles/'+str(args.casename)+'_BLD_'+month+'.nc')

  if args.savefigs:
    fname = 'PNG/BLD/'+str(args.casename)+'_BLD_model_'+str(month)+'.png'
  xyplot(model_summer, grd.geolon, grd.geolat, area=area,
         save='PNG/BLD/'+str(args.casename)+'_BLD_model_'+str(month)+'.png',
         suptitle=ds[var].attrs['long_name'] +' ['+ ds[var].attrs['units']+']', clim=(0,150),
         title=str(args.casename) + ' ' +str(args.start_date) + ' to '+ str(args.end_date) + \
              ' JFM (SH), JAS (NH)')
  return
# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': parseCommandLine()

