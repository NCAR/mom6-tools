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
from mom6_tools.m6plot import yzcompare, yzplot
from mom6_tools.MOM6grid import MOM6grid
from mom6_tools.m6toolbox import shiftgrid, add_global_attrs

def parseCommandLine():
  """
  Parse the command line positional and optional arguments.
  This is the highest level procedure invoked from the very end of the script.
  """
  parser = argparse.ArgumentParser(description=
      '''
      Compares (model vs obs) T, S and U near the Equator.
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
  parser.add_argument('-o','--obs', type=str, default='WOA18', help='''Observational product to compare agaist.  \
    Valid options are: WOA18 (default) or PHC2''')
  parser.add_argument('-debug',   help='''Add priting statements for debugging purposes''', action="store_true")
  optCmdLineArgs = parser.parse_args()
  driver(optCmdLineArgs)

#-- This is where all the action happends, i.e., functions for each diagnostic are called.

def driver(args):
  nw = args.number_of_workers
  if not os.path.isdir('PNG/Equatorial'):
    print('Creating a directory to place figures (PNG/Equatorial)... \n')
    os.system('mkdir -p PNG/Equatorial')
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
  grd = MOM6grid(OUTDIR+'/'+args.casename+'.mom6.static.nc', xrformat=True)
  # select Equatorial region
  grd_eq = grd.sel(yh=slice(-10,10))

  # load obs
  if args.obs == 'PHC2':
    # load PHC2 data
    obs_path = '/glade/p/cesm/omwg/obs_data/phc/'
    obs_temp = xr.open_dataset(obs_path+'PHC2_TEMP_tx0.66v1_34lev_ann_avg.nc', decode_times=False)
    obs_salt = xr.open_dataset(obs_path+'PHC2_SALT_tx0.66v1_34lev_ann_avg.nc', decode_times=False)
    # get theta and salt and rename coordinates to be the same as the model's
    thetao_obs = obs_temp.TEMP.rename({'X': 'xh','Y': 'yh', 'depth': 'z_l'});
    salt_obs = obs_salt.SALT.rename({'X': 'xh','Y': 'yh', 'depth': 'z_l'});
  elif args.obs == 'WOA18':
    # load WOA18 data
    obs_path = '/glade/u/home/gmarques/Notebooks/CESM_MOM6/WOA18_remapping/'
    obs_temp = xr.open_dataset(obs_path+'WOA18_TEMP_tx0.66v1_34lev_ann_avg.nc', decode_times=False)
    obs_salt = xr.open_dataset(obs_path+'WOA18_SALT_tx0.66v1_34lev_ann_avg.nc', decode_times=False)
    # get theta and salt and rename coordinates to be the same as the model's
    thetao_obs = obs_temp.theta0 #.rename({'zl': 'z_l'});
    salt_obs = obs_salt.s_an     #.rename({'zl': 'z_l'});

  else:
    raise ValueError("The obs selected is not available.")


  johnson = xr.open_dataset('/glade/p/cesm/omwg/obs_data/johnson_pmel/meanfit_m.nc')

  parallel = False
  if nw > 1:
    parallel = True
    cluster = NCARCluster()
    cluster.scale(nw)
    client = Client(cluster)

  print('Reading surface dataset...')
  startTime = datetime.now()

  # load data
  def preprocess(ds):
    variables = ['thetao', 'so', 'uo', 'time', 'time_bnds', 'e']
    return ds[variables]

  ds1 = xr.open_mfdataset(OUTDIR+'/'+dcase.casename+'.mom6.h_*.nc', parallel=parallel)
  # use datetime
  #ds1['time'] = ds1.indexes['time'].to_datetimeindex()

  ds = preprocess(ds1)

  print('Time elasped: ', datetime.now() - startTime)

  # set obs coords to be same as model
  thetao_obs['xh'] = ds.xh; thetao_obs['yh'] = ds.yh;
  salt_obs['xh'] = ds.xh; salt_obs['yh'] = ds.yh;

  print('Selecting data between {} and {} (time) and -10 to 10 (yh)...'.format(args.start_date, \
        args.end_date))
  startTime = datetime.now()
  ds = ds.sel(time=slice(args.start_date, args.end_date)).sel(yh=slice(-10,10)).isel(z_i=slice(0,15)).isel(z_l=slice(0,14))
  print('Time elasped: ', datetime.now() - startTime)

  print('Yearly mean...')
  startTime = datetime.now()
  ds = ds.resample(time="1Y", closed='left',keep_attrs=True).mean('time',keep_attrs=True).compute()
  print('Time elasped: ', datetime.now() - startTime)

  print('Time averaging...')
  startTime = datetime.now()
  thetao = ds.thetao.mean('time')
  so = ds.so.mean('time')
  uo = ds.uo.mean('time')
  eta = ds.e.mean('time')
  # find point closest to eq. and select data
  j = np.abs( grd_eq.geolat[:,0].values - 0. ).argmin()
  temp_eq = np.ma.masked_invalid(thetao.isel(yh=j).values)
  salt_eq = np.ma.masked_invalid(so.isel(yh=j).values)
  u_eq    = np.ma.masked_invalid(uo.isel(yh=j).values)
  e_eq    = np.ma.masked_invalid(eta.isel(yh=j).values)
  thetao_obs_eq = np.ma.masked_invalid(thetao_obs.sel(yh=slice(-10,10)).isel(yh=j).isel(z_l=slice(0,14)).values)
  salt_obs_eq = np.ma.masked_invalid(salt_obs.sel(yh=slice(-10,10)).isel(yh=j).isel(z_l=slice(0,14)).values)
  print('Time elasped: ', datetime.now() - startTime)

  if parallel:
    print('\n Releasing workers...')
    client.close(); cluster.close()

  print('Equatorial Upper Ocean plots...')
  y = ds.yh.values
  zz = ds.z_i.values
  x = ds.xh.values
  [X, Z] = np.meshgrid(x, zz)
  z = 0.5 * ( Z[:-1] + Z[1:])

  figname = 'PNG/Equatorial/'+str(dcase.casename)+'_'
  yzcompare(temp_eq , thetao_obs_eq, x, -Z,
            title1 = 'model temperature', ylabel='Longitude', yunits='',
            title2 = 'observed temperature (PHC/Levitus)', #contour=True,
            suptitle=dcase.casename + ', averaged '+str(args.start_date)+ ' to ' +str(args.end_date),
            extend='neither', dextend='neither', clim=(6,31.), dlim=(-5,5), dcolormap=plt.cm.bwr,
            save=figname+'Equatorial_Global_temperature.png')

  yzcompare(salt_eq , salt_obs_eq, x, -Z,
        title1 = 'model salinity', ylabel='Longitude', yunits='',
        title2 = 'observed salinity (PHC/Levitus)', #contour=True,
        suptitle=dcase.casename + ', averaged '+str(args.start_date)+ ' to ' +str(args.end_date),
        extend='neither', dextend='neither', clim=(33.5,37.), dlim=(-1,1), dcolormap=plt.cm.bwr,
        save=figname+'Equatorial_Global_salinity.png')

  # create dataarays and saving data
  temp_eq_da = xr.DataArray(temp_eq, dims=['zl','xh'],
                           coords={'zl' : z[:,0], 'xh' : x[:]}).rename('temp_eq')

  attrs = {'casename': args.casename,
           'module': os.path.basename(__file__)}
  add_global_attrs(temp_eq_da,attrs)
  temp_eq_da.to_netcdf('ncfiles/'+str(args.casename)+'_temp_eq.nc')

  salt_eq_da = xr.DataArray(salt_eq, dims=['zl','xh'],
                           coords={'zl' : z[:,0], 'xh' : x[:]}).rename('salt_eq')
  add_global_attrs(salt_eq_da,attrs)
  salt_eq_da.to_netcdf('ncfiles/'+str(args.casename)+'_salt_eq.nc')

  # Shift model data to compare against obs
  tmp, lonh = shiftgrid(thetao.xh[-1].values, thetao[0,0,:].values, ds.thetao.xh.values)
  tmp, lonq = shiftgrid(uo.xq[-1].values, uo[0,0,:].values, uo.xq.values)

  thetao['xh'].values[:] = lonh
  so['xh'].values[:] = lonh
  uo['xq'].values[:] = lonq

  # y and z from obs
  y_obs = johnson.YLAT11_101.values
  zz = np.arange(0,510,10)
  [Y, Z_obs] = np.meshgrid(y_obs, zz)
  z_obs = 0.5 * ( Z_obs[0:-1,:] + Z_obs[1:,] )

  # y and z from model
  y_model = thetao.yh.values
  z = eta.z_i.values
  [Y, Z_model] = np.meshgrid(y_model, z)
  z_model = 0.5 * ( Z_model[0:-1,:] + Z_model[1:,:] )

  # longitutes to be compared
  longitudes = [143., 156., 165., 180., 190., 205., 220., 235., 250., 265.]

  for l in longitudes:
    # Temperature
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
    dummy_model = np.ma.masked_invalid(thetao.sel(xh=l, method='nearest').values)
    dummy_obs = np.ma.masked_invalid(johnson.POTEMPM.sel(XLON=l, method='nearest').values)
    yzplot(dummy_model, y_model, -Z_model, clim=(7,30), axis=ax1, zlabel='Depth', ylabel='Latitude', title=str(dcase.casename))
    cs1 = ax1.contour( y_model + 0*z_model, -z_model, dummy_model, levels=np.arange(0,30,2), colors='k',); plt.clabel(cs1,fmt='%3.1f', fontsize=14)
    ax1.set_ylim(-400,0)
    yzplot(dummy_obs, y_obs, -Z_obs, clim=(7,30), axis=ax2, zlabel='Depth', ylabel='Latitude', title='Johnson et al (2002)')
    cs2 = ax2.contour( y_obs + 0*z_obs, -z_obs, dummy_obs, levels=np.arange(0,30,2), colors='k',); plt.clabel(cs2,fmt='%3.1f', fontsize=14)
    ax2.set_ylim(-400,0)
    plt.suptitle('Temperature [C] @ '+str(l)+ ', averaged between '+str(args.start_date)+' and '+str(args.end_date))
    plt.savefig(figname+'temperature_'+str(l)+'.png')

    # Salt
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
    dummy_model = np.ma.masked_invalid(so.sel(xh=l, method='nearest').values)
    dummy_obs = np.ma.masked_invalid(johnson.SALINITYM.sel(XLON=l, method='nearest').values)
    yzplot(dummy_model, y_model, -Z_model, clim=(32,36), axis=ax1, zlabel='Depth', ylabel='Latitude', title=str(dcase.casename))
    cs1 = ax1.contour( y_model + 0*z_model, -z_model, dummy_model, levels=np.arange(32,36,0.5), colors='k',); plt.clabel(cs1,fmt='%3.1f', fontsize=14)
    ax1.set_ylim(-400,0)
    yzplot(dummy_obs, y_obs, -Z_obs, clim=(32,36), axis=ax2, zlabel='Depth', ylabel='Latitude', title='Johnson et al (2002)')
    cs2 = ax2.contour( y_obs + 0*z_obs, -z_obs, dummy_obs, levels=np.arange(32,36,0.5), colors='k',); plt.clabel(cs2,fmt='%3.1f', fontsize=14)
    ax2.set_ylim(-400,0)
    plt.suptitle('Salinity [psu] @ '+str(l)+ ', averaged between '+str(args.start_date)+' and '+str(args.end_date))
    plt.savefig(figname+'salinity_'+str(l)+'.png')

    # uo
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
    dummy_model = np.ma.masked_invalid(uo.sel(xq=l, method='nearest').values)
    dummy_obs = np.ma.masked_invalid(johnson.UM.sel(XLON=l, method='nearest').values)
    yzplot(dummy_model, y_model, -Z_model, clim=(-0.6,1.2), axis=ax1, zlabel='Depth', ylabel='Latitude', title=str(dcase.casename))
    cs1 = ax1.contour( y_model + 0*z_model, -z_model, dummy_model, levels=np.arange(-1.2,1.2,0.1), colors='k',); plt.clabel(cs1,fmt='%3.1f', fontsize=14)
    ax1.set_ylim(-400,0)
    yzplot(dummy_obs, y_obs, -Z_obs, clim=(-0.6,1.2), axis=ax2, zlabel='Depth', ylabel='Latitude', title='Johnson et al (2002)')
    cs2 = ax2.contour( y_obs + 0*z_obs, -z_obs, dummy_obs, levels=np.arange(-1.2,1.2,0.1), colors='k',); plt.clabel(cs2,fmt='%3.1f', fontsize=14)
    ax2.set_ylim(-400,0)
    plt.suptitle('Eastward velocity [m/s] @ '+str(l)+ ', averaged between '+str(args.start_date)+' and '+str(args.end_date))
    plt.savefig(figname+'uo_'+str(l)+'.png')

  # Eastward velocity [m/s] along the Equatorial Pacific
  x_obs = johnson.XLON.values
  [X_obs, Z_obs] = np.meshgrid(x_obs, zz)
  z_obs = 0.5 * ( Z_obs[:-1,:] + Z_obs[1:,:] )

  x_model = so.xh.values
  z = eta.z_i.values
  [X, Z_model] = np.meshgrid(x_model, z)
  z_model = 0.5 * ( Z_model[:-1,:] + Z_model[1:,:] )

  #from mom6_tools.m6plot import
  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
  dummy_obs = np.ma.masked_invalid(johnson.UM.sel(YLAT11_101=0).values)
  dummy_model = np.ma.masked_invalid(uo.sel(yh=0, method='nearest').values)
  yzplot(dummy_model, x_model, -Z_model, clim=(-0.6,1.2), axis=ax1, landcolor=[0., 0., 0.], title=str(dcase.casename), ylabel='Longitude')
  cs1 = ax1.contour( x_model + 0*z_model, -z_model, dummy_model, levels=np.arange(-1.2,1.2,0.1),  colors='k'); plt.clabel(cs1,fmt='%2.1f', fontsize=14)
  ax1.set_xlim(143,265); ax1.set_ylim(-400,0)
  yzplot(dummy_obs, x_obs, -Z_obs, clim=(-0.4,1.2), ylabel='Longitude', yunits='',  axis=ax2, title='Johnson et al (2002)')
  cs1 = ax2.contour( x_obs + 0*z_obs, -z_obs, dummy_obs,  levels=np.arange(-1.2,1.2,0.1), colors='k'); plt.clabel(cs1,fmt='%2.1f', fontsize=14)
  ax2.set_xlim(143,265); ax2.set_ylim(-400,0)
  plt.suptitle('Eastward velocity [m/s] along the Equatorial Pacific, averaged between '+str(args.start_date)+' and '+str(args.end_date))
  plt.savefig(figname+'Equatorial_Pacific_uo.png')

  plt.close('all')

  print('{} was run successfully!'.format(os.path.basename(__file__)))

  return

# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': parseCommandLine()

