#!/usr/bin/env python

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import warnings, os, yaml, argparse, intake
import pandas as pd
from collections import OrderedDict
import dask
from datetime import datetime, date
from ncar_jobqueue import NCARCluster
from dask.distributed import Client
from mom6_tools.DiagsCase import DiagsCase
from mom6_tools.m6toolbox import add_global_attrs, genBasinMasks, weighted_temporal_mean_vars
from mom6_tools.m6plot import xycompare, polarcomparison, chooseColorLevels
from mom6_tools.MOM6grid import MOM6grid
from mom6_tools.stats import stats_to_ds, min_da, max_da, mean_da, rms_da, std_da

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
  parser.add_argument('-o','--obs', type=str, default='woa-2018-tx2_3v2-annual-all',
                      help='''Name of observational product in the oce-catalog  \
                              to compare against. Default is woa-2018-tx2_3v2-annual-all''')
  parser.add_argument('-debug',   help='''Add priting statements for debugging purposes''', action="store_true")
  optCmdLineArgs = parser.parse_args()
  driver(optCmdLineArgs)

#-- This is where all the action happends, i.e., functions for each diagnostic are called.

def driver(args):
  debug = args.debug
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
  DOUT_S = dcase.get_value('DOUT_S')
  if DOUT_S:
    OUTDIR = dcase.get_value('DOUT_S_ROOT')+'/ocn/hist/'
  else:
    OUTDIR = dcase.get_value('RUNDIR')

  args.casename = dcase.casename
  args.monthly = dcase.casename+diag_config_yml['Fnames']['z']
  args.static = dcase.casename+diag_config_yml['Fnames']['static']
  args.geom = dcase.casename+diag_config_yml['Fnames']['geom']

  print('Output directory is:', OUTDIR)
  print('Casename is:', args.casename)
  print('Number of workers: ', nw)
  print('Reading file stream: ', args.monthly)

  # set avg dates
  avg = diag_config_yml['Avg']
  if not args.start_date : args.start_date = avg['start_date']
  if not args.end_date : args.end_date = avg['end_date']

  # read grid info
  geom_file = OUTDIR+'/'+args.geom
  if os.path.exists(geom_file):
    grd = MOM6grid(OUTDIR+'/'+args.static, geom_file)
    grd_xr = MOM6grid(OUTDIR+'/'+args.static, geom_file, xrformat=True);
  else:
    grd = MOM6grid(OUTDIR+'/'+args.static)
    grd_xr = MOM6grid(OUTDIR+'/'+args.static, xrformat=True);

  # create masks
  try:
    depth = grd.depth_ocean
  except:
    depth = grd.deptho
  # remote Nan's, otherwise genBasinMasks won't work
  depth[np.isnan(depth)] = 0.0
  basin_code = genBasinMasks(grd.geolon, grd.geolat, depth, xda=True)

  # load obs
  catalog = intake.open_catalog(diag_config_yml['oce_cat'])
  obs = catalog[args.obs].to_dask()
  obs = obs.rename({'z_l' : 'depth'});
  obs_temp = obs.thetao
  obs_salt = obs.so

  parallel = False
  if nw > 1:
    parallel = True
    cluster = NCARCluster()
    cluster.scale(nw)
    client = Client(cluster)

  print('Reading surface dataset...')
  startTime = datetime.now()

  def preprocess(ds):
    ''' Return a dataset desired variables'''
    variables = ['thetao', 'so', 'time']
#   The following mess up time averaging
#   xarray doesn't like averaging variables that are time objects themselves
#   the calendar information is sufficient to get weighted means
#    if 'time_bounds' in ds.variables:
#      variables.append('time_bounds')
#    elif 'time_bnds' in ds.variables:
#      variables.append('time_bnds')
    return ds[variables]

  ds = xr.open_mfdataset(OUTDIR+'/'+args.monthly, \
         parallel=True, data_vars='minimal', \
         coords='minimal', compat='override', preprocess=preprocess)


  print('Time elasped: ', datetime.now() - startTime)

  print('Selecting data between {} and {}...'.format(args.start_date, args.end_date))
  startTime = datetime.now()
  ds_sel = ds.sel(time=slice(args.start_date, args.end_date))
  print('Time elasped: ', datetime.now() - startTime)

  print('\n Computing annual means and then average in time...')
  startTime = datetime.now()
  # compute annual mean and then average in time
  ds_ann = weighted_temporal_mean_vars(ds_sel)
  temp = np.ma.masked_invalid(ds_ann.thetao.mean('time').values)
  salt = np.ma.masked_invalid(ds_ann.so.mean('time').values)
  print('Time elasped: ', datetime.now() - startTime)

  print('Computing stats for different basins...')
  startTime = datetime.now()
  # construct a 3D area with land values masked
  try:
    area = np.ma.masked_where(grd.wet == 0,grd.area_t)
    area_xr = grd_xr.area_t
  except:
    area = np.ma.masked_where(grd.wet == 0,grd.areacello)
    area_xr = grd_xr.areacello

  try:
    depth = grd_xr.depth_ocean
  except:
    depth = grd_xr.deptho

  tmp = np.repeat(area[np.newaxis, :, :], len(obs_temp.depth), axis=0)
  area_mom3D = xr.DataArray(tmp, dims=('depth', 'yh','xh'),
                          coords={'depth':obs_temp.depth.values, 'yh': grd.yh,
                                  'xh':grd.xh})

  for k in range(len(area_mom3D.depth)):
      area_mom3D[k,:] = area_xr.where(depth >= area_mom3D.depth[k])

  print('Done computing area_mom3D...')
  # temp
  thetao_mean = ds_ann.thetao.mean('time').compute()
  print('Done computing thetao_mean...')
  temp_diff = thetao_mean.rename({'z_l':'depth'}) - obs_temp
  print('Done computing thetao_diff...')
  temp_stats = myStats_da(temp_diff, area_mom3D, basins=basin_code, debug=debug).rename('thetao_bias_stats')
  print('Done computing temp_stats...')
  # salt
  so_mean = ds_ann.so.mean('time').compute()
  salt_diff = so_mean.rename({'z_l':'depth'}) - obs_salt
  salt_stats = myStats_da(salt_diff, area_mom3D, basins=basin_code, debug=debug).rename('so_bias_stats')

  # plots
  depth = temp_stats.depth.values
  basin = temp_stats.basin.values
  interfaces = np.zeros(len(depth)+1)
  for k in range(1,len(depth)+1):
    interfaces[k] = interfaces[k-1] + ( 2 * (depth[k-1] - interfaces[k-1]))

  reg = np.arange(len(temp_stats.basin.values)+ 1)
  figname = 'PNG/TS_levels/'+str(dcase.casename)+'_'

  temp_label = r'Potential temperature [$^o$C]'
  salt_label = 'Salinity [psu]'
  # minimum
  score_plot2(basin, interfaces, temp_stats[:,0,:],nbins=30, cmap=plt.cm.viridis,
            cmin=temp_stats[:,0,:].min().values,
            units=temp_label,
            fname = figname+'thetao_bias_min.png',
            title='Minimun temperature difference (model-{})'.format(args.obs))
  score_plot2(basin, interfaces, salt_stats[:,0,:],nbins=30, cmap=plt.cm.viridis,
            cmin=salt_stats[:,0,:].min().values,
            units=salt_label,
            fname = figname+'so_bias_min.png',
            title='Minimun salinity difference (model-{})'.format(args.obs))

  # maximum
  score_plot2(basin, interfaces, temp_stats[:,1,:],nbins=30, cmap=plt.cm.viridis,
            cmin=temp_stats[:,1,:].min().values,
            units=temp_label,
            fname = figname+'thetao_bias_max.png',
            title='Maximum temperature difference (model-{})'.format(args.obs))
  score_plot2(basin, interfaces, salt_stats[:,1,:],nbins=30, cmap=plt.cm.viridis,
            cmin=salt_stats[:,1,:].min().values,
            units=salt_label,
            fname = figname+'so_bias_max.png',
            title='Maximum salinity difference (model-{})'.format(args.obs))

  # mean
  score_plot2(basin, interfaces, temp_stats[:,2,:],nbins=30, cmap=plt.cm.seismic,
            units=temp_label,
            fname = figname+'thetao_bias_mean.png',
            title='Mean temperature difference (model-{})'.format(args.obs))
  score_plot2(basin, interfaces, salt_stats[:,2,:],nbins=30, cmap=plt.cm.seismic,
            units=salt_label,
            fname = figname+'so_bias_mean.png',
            title='Mean salinity difference (model-{})'.format(args.obs))

  # std
  score_plot2(basin, interfaces, temp_stats[:,3,:],nbins=30, cmap=plt.cm.viridis, cmin = 1.0E-15,
            units=temp_label,
            fname = figname+'thetao_bias_std.png',
            title='Std temperature difference (model-{})'.format(args.obs))
  score_plot2(basin, interfaces, salt_stats[:,3,:],nbins=30, cmap=plt.cm.viridis, cmin = 1.0E-15,
            units=salt_label,
            fname = figname+'so_bias_std.png',
            title='Std salinity difference (model-{})'.format(args.obs))
  # rms
  score_plot2(basin, interfaces, temp_stats[:,4,:],nbins=30, cmap=plt.cm.viridis, cmin = 1.0E-15,
            units=temp_label,
            fname = figname+'thetao_bias_rms.png',
            title='Rms temperature difference (model-{})'.format(args.obs))
  score_plot2(basin, interfaces, salt_stats[:,4,:],nbins=30, cmap=plt.cm.viridis, cmin = 1.0E-15,
            units=salt_label,
            fname = figname+'so_bias_rms.png',
            title='Rms salinity difference (model-{})'.format(args.obs))
  print('Time elasped: ', datetime.now() - startTime)

  print('Saving netCDF files...')
  startTime = datetime.now()
  attrs = {'description': 'model - obs at depth levels',
           'start_date': args.start_date,
           'end_date': args.end_date,
           'casename': dcase.casename,
           'obs': args.obs,
           'module': os.path.basename(__file__)}
  # create dataset to store results
  add_global_attrs(temp_stats,attrs)
  temp_stats.to_netcdf('ncfiles/'+str(args.casename)+'_thetao_bias_ann_mean_stats.nc')
  add_global_attrs(salt_stats,attrs)
  salt_stats.to_netcdf('ncfiles/'+str(args.casename)+'_so_bias_ann_mean_stats.nc')

  thetao = xr.DataArray(thetao_mean, dims=['z_l','yh','xh'],
              coords={'z_l' : ds.z_l, 'yh' : grd.yh, 'xh' : grd.xh}).rename('thetao')
  temp_bias = np.ma.masked_invalid(thetao.values - obs_temp.values)
  ds_thetao = xr.Dataset(data_vars={ 'thetao' : (('z_l','yh','xh'), thetao.values),
                            'thetao_bias' :     (('z_l','yh','xh'), temp_bias)},
                            coords={'z_l' : ds.z_l, 'yh' : grd.yh, 'xh' : grd.xh})
  add_global_attrs(ds_thetao,attrs)

  ds_thetao.to_netcdf('ncfiles/'+str(args.casename)+'_thetao_time_mean.nc')
  so = xr.DataArray(ds.so.mean('time'), dims=['z_l','yh','xh'],
              coords={'z_l' : ds.z_l, 'yh' : grd.yh, 'xh' : grd.xh}).rename('so')
  salt_bias = np.ma.masked_invalid(so.values - obs_salt.values)
  ds_so = xr.Dataset(data_vars={ 'so' : (('z_l','yh','xh'), so.values),
                            'so_bias' :     (('z_l','yh','xh'), salt_bias)},
                            coords={'z_l' : ds.z_l, 'yh' : grd.yh, 'xh' : grd.xh})
  add_global_attrs(ds_so,attrs)
  ds_so.to_netcdf('ncfiles/'+str(args.casename)+'_so_time_mean.nc')
  print('Time elasped: ', datetime.now() - startTime)

  if parallel:
    print('\n Releasing workers...')
    client.close(); cluster.close()

  print('Global plots...')
  try:
    area = grd_xr.area_t.where(grd_xr.wet > 0).values
  except:
    area = grd_xr.areacello.where(grd_xr.wet > 0).values

  km = len(obs_temp['depth'])
  for k in range(km):
    if ds['z_l'][k].values < 1200.0:
      figname = 'PNG/TS_levels/'+str(dcase.casename)+'_'+str(ds['z_l'][k].values)+'_'
      temp_obs = np.ma.masked_invalid(obs_temp[k,:].values)
      xycompare(temp[k,:] , temp_obs, grd.geolon, grd.geolat, area=area,
              title1 = 'model temperature, depth ='+str(ds['z_l'][k].values)+ 'm',
              title2 = 'observed temperature, depth ='+str(obs_temp['depth'][k].values)+ 'm',
              suptitle=dcase.casename + ', averaged '+str(args.start_date)+ ' to ' +str(args.end_date),
              extend='both', dextend='neither', clim=(-1.9,30.), dlim=(-2,2), dcolormap=plt.cm.bwr,
              save=figname+'global_temp.png')
      salt_obs = np.ma.masked_invalid(obs_salt[k,:].values)
      xycompare( salt[k,:] , salt_obs, grd.geolon, grd.geolat, area=area,
              title1 = 'model salinity, depth ='+str(ds['z_l'][k].values)+ 'm',
              title2 = 'observed salinity, depth ='+str(obs_temp['depth'][k].values)+ 'm',
              suptitle=dcase.casename + ', averaged '+str(args.start_date)+ ' to ' +str(args.end_date),
              extend='both', dextend='neither', clim=(30.,39.), dlim=(-2,2), dcolormap=plt.cm.bwr,
              save=figname+'global_salt.png')

  #print('Antarctic plots...')
  #for k in range(km):
  #  if (ds['z_l'][k].values < 1200.):
  #    temp_obs = np.ma.masked_invalid(obs_temp['TEMP'][k,:].values)
  #    polarcomparison(temp[k,:] , temp_obs, grd,
  #            title1 = 'model temperature, depth ='+str(ds['z_l'][k].values)+ 'm',
  #            title2 = 'observed temperature, depth ='+str(obs_temp['depth'][k].values)+ 'm',
  #            extend='both', dextend='neither', clim=(-1.9,10.5), dlim=(-2,2), dcolormap=plt.cm.bwr,
  #            suptitle=dcase.casename + ', averaged '+str(args.start_date)+ ' to ' +str(args.end_date),
  #            proj='SP', save=figname+'antarctic_temp.png')
  #    salt_obs = np.ma.masked_invalid(obs_salt['SALT'][k,:].values)
  #    polarcomparison( salt[k,:] , salt_obs, grd,
  #            title1 = 'model salinity, depth ='+str(ds['z_l'][k].values)+ 'm',
  #            title2 = 'observed salinity, depth ='+str(obs_temp['depth'][k].values)+ 'm',
  #            extend='both', dextend='neither', clim=(33.,35.), dlim=(-2,2), dcolormap=plt.cm.bwr,
  #            suptitle=dcase.casename + ', averaged '+str(args.start_date)+ ' to ' +str(args.end_date),
  #            proj='SP', save=figname+'antarctic_salt.png')

  #print('Arctic plots...')
  #for k in range(km):
  #  if (ds['z_l'][k].values < 100.):
  #    temp_obs = np.ma.masked_invalid(obs_temp['TEMP'][k,:].values)
  #    polarcomparison(temp[k,:] , temp_obs, grd,
  #            title1 = 'model temperature, depth ='+str(ds['z_l'][k].values)+ 'm',
  #            title2 = 'observed temperature, depth ='+str(obs_temp['depth'][k].values)+ 'm',
  #            extend='both', dextend='neither', clim=(-1.9,11.5), dlim=(-2,2), dcolormap=plt.cm.bwr,
  #            suptitle=dcase.casename + ', averaged '+str(args.start_date)+ ' to ' +str(args.end_date),
  #            proj='NP', save=figname+'arctic_temp.png')
  #    salt_obs = np.ma.masked_invalid(obs_salt['SALT'][k,:].values)
  #    polarcomparison( salt[k,:] , salt_obs, grd,
  #            title1 = 'model salinity, depth ='+str(ds['z_l'][k].values)+ 'm',
  #            title2 = 'observed salinity, depth ='+str(obs_temp['depth'][k].values)+ 'm',
  #            extend='both', dextend='neither', clim=(31.5,35.), dlim=(-2,2), dcolormap=plt.cm.bwr,
  #            suptitle=dcase.casename + ', averaged '+str(args.start_date)+ ' to ' +str(args.end_date),
  #            proj='NP', save=figname+'arctic_salt.png')

  print('{} was run successfully!'.format(os.path.basename(__file__)))

  return

# misc functions
def myStats_da(da, weights, dims=('yh', 'xh'), basins=None, debug=False):
  rmask_od = OrderedDict()
  for reg in basins.region:
    if debug: print('Region: ', reg)
    # select region in the DataArray
    da_reg = da.where(basins.sel(region=reg).values == 1.0)
    # select weights to where region values are one
    tmp_weights = weights.where(basins.sel(region=reg).values == 1.0)
    total_weights = tmp_weights.sum(dim=dims)
    da_min  = min_da(da_reg , dims)
    da_max  = max_da(da_reg , dims)
    da_mean = mean_da(da_reg, dims, tmp_weights,  total_weights)
    da_std  = std_da(da_reg , dims, tmp_weights,  total_weights, da_mean)
    da_rms  = rms_da(da_reg , dims, tmp_weights,  total_weights)
    out = stats_to_ds(da_min, da_max, da_mean, da_std, da_rms)
    rmask_od[str(reg.values)] = out

  return dict_to_da(rmask_od) # create dataarray using rmask_od

def dict_to_da(stats_dict):
  depth = stats_dict[list(stats_dict.items())[0][0]].depth
  basins = list(stats_dict.keys())
  stats = ['da_min', 'da_max', 'da_mean', 'da_std', 'da_rms']
  var = np.zeros((len(basins),len(stats),len(depth)))
  da = xr.DataArray(var, dims=['basin', 'stats', 'depth'],
                           coords={'basin': basins,
                                   'stats': stats,
                                   'depth': depth},)
  for reg in (basins):
    da.sel(basin=reg).sel(stats='da_min').values[:] = stats_dict[reg].da_min.values
    da.sel(basin=reg).sel(stats='da_max').values[:] = stats_dict[reg].da_max.values
    da.sel(basin=reg).sel(stats='da_mean').values[:]= stats_dict[reg].da_mean.values
    da.sel(basin=reg).sel(stats='da_std').values[:] = stats_dict[reg].da_std.values
    da.sel(basin=reg).sel(stats='da_rms').values[:] = stats_dict[reg].da_rms.values

  return da

def score_plot2(x,y,vals, cmin=None, cmap=plt.cm.bwr,title='',nbins=50, units='',fname=None):
  import matplotlib
  matplotlib.rcParams.update({'font.size': 14})
  reg = np.arange(len(x)+ 1)
  autocenter=False
  if not cmin:
    cmin = vals.min().values
    autocenter=True

  cmax = vals.max().values
  #print(cmin, autocenter)
  cmap, norm, extend = chooseColorLevels(cmin, cmax, cmap,
                                         nbins=nbins, autocenter=autocenter)
  #print(val_abs)
  fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,5), sharex=True)
  ax[0].set_title(title)
  # 0 to 200 depth[0] to depth[9]
  cs = ax[1].pcolormesh(reg, y, vals.transpose().values, cmap=cmap, norm=norm)
  ax[1].set_xticks(np.arange(len(x))+0.5)
  ax[1].set_xticklabels(x);
  ax[1].set_ylim(0,200)
  ax[1].set_ylabel('Depth [m]')
  # 200 to 1000
  cs = ax[0].pcolormesh(reg, y, vals.transpose().values, cmap=cmap, norm=norm)
  ax[0].set_xticks(np.arange(len(x))+0.5)
  ax[0].set_xticklabels(x);
  ax[0].set_ylim(200,1000)
  ax[0].set_ylabel('Depth [m]')

  # Rotate the tick labels and set their alignment.
  plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
  fig.subplots_adjust(right=0.8)
  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  cbar = fig.colorbar(cs, cax=cbar_ax)
  cbar.set_label(units)
  if fname:
    plt.savefig(fname,bbox_inches='tight')

  plt.close()
  return
# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': parseCommandLine()

