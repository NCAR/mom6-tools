#!/usr/bin/env python

"""
Functions used to calculate statistics.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mom6_tools.m6toolbox import cime_xmlquery
from mom6_tools.ClimoGenerator import ClimoGenerator
from mom6_tools.m6toolbox import genBasinMasks, add_global_attrs
from mom6_tools.m6plot import ztplot, plot_stats_da, xyplot
from mom6_tools.MOM6grid import MOM6grid
import pandas as pd
import getpass
from datetime import datetime
from distributed import Client
from ncar_jobqueue import NCARCluster
from collections import OrderedDict
import yaml, os

try: import argparse
except: raise Exception('This version of python is not new enough. python 2.7 or newer is required.')

def options():
  parser = argparse.ArgumentParser(description='''Script for computing and plotting statistics.''')
  parser.add_argument('diag_config_yml_path', type=str, help='''Full path to the yaml file  \
    describing the run and diagnostics to be performed.''')
  parser.add_argument('-sd','--start_date', type=str, default='',
                      help='''Start year to compute averages. Default is to use value set in diag_config_yml_path''')
  parser.add_argument('-ed','--end_date', type=str, default='',
                      help='''End year to compute averages. Default is to use value set in diag_config_yml_path''')
  parser.add_argument('-ocean_stats', help='''Extract time series from ocean.stats and ocean.stats.nc ''', \
                      action="store_true")
  parser.add_argument('-time_series', help='''Extract time-series for thetaoga and soga and saves \
                       annual means in a netCDF file''', action="store_true")
  parser.add_argument('-forcing', help='''Compute global time averages and regionally-averaged time-series \
                      of forcing fields''', action="store_true")
  parser.add_argument('-surface', help='''Compute global time averages and regionally-averaged time-series \
                      of surface fields''', action="store_true")
  parser.add_argument('-nw','--number_of_workers',  type=int, default=0,
                      help='''Number of workers to use. Default=0 (serial).''')
  parser.add_argument('-debug',   help='''Add priting statements for debugging purposes''', action="store_true")
  cmdLineArgs = parser.parse_args()
  return cmdLineArgs

def min_da(da, dims=('yh', 'xh')):
  """
  Calculates the minimun value in DataArray da,

  ----------
  da : xarray.DataArray
        DataArray for which to compute the min.

  dims : tuple, str
    Dimension(s) over which to apply reduction. Default is ('yh', 'xh').

  Returns
  -------
  reduction : DataSet
      xarray.Dataset with min for da.
  """
  check_dims(da,dims)
  return da.min(dim=dims, keep_attrs=True)

def max_da(da, dims=('yh', 'xh')):
  """
  Calculates the maximum value in DataArray da.

  ----------
  da : xarray.DataArray
        DataArray for which to compute the max.

  dims : tuple, str
    Dimension(s) over which to apply reduction. Default is ('yh', 'xh').

  Returns
  -------
  reduction : DataSet
      xarray.Dataset with the max for da.
  """
  check_dims(da,dims)
  return da.max(dim=dims, keep_attrs=True)

def mean_da(da, dims=('yh', 'xh'), weights=None,  weights_sum=None):
  """
  Calculates the mean value in DataArray da (optional weighted mean).

  ----------
  da : xarray.DataArray
        DataArray for which to compute (weighted) mean.

  dims : tuple, str
    Dimension(s) over which to apply reduction. Default is ('yh', 'xh').

  weights : xarray.DataArray, optional
    weights to apply. It can be a masked array.

  weights_sum : xarray.DataArray, optional
    Total weight (i.e., weights.sum()). Only computed if not provided.

  Returns
  -------
  reduction : DataSet
      xarray.Dataset with (optionally weighted) mean for da.
  """
  check_dims(da,dims)
  if weights is not None:
    if weights_sum is None: weights_sum = weights.sum(dim=dims)
    out = ((da * weights).sum(dim=dims) / weights_sum)
    # copy attrs
    out.attrs = da.attrs
    return out
  else:
    return da.mean(dim=dims, keep_attrs=True)

def std_da(da, dims=('yh', 'xh'), weights=None,  weights_sum=None, da_mean=None):
  """
  Calculates the std in DataArray da (optional weighted std).

  ----------
  da : xarray.DataArray
        DataArray for which to compute (weighted) std.

  dims : tuple, str
    Dimension(s) over which to apply reduction. Default is ('yh', 'xh').

  weights : xarray.DataArray, optional
    weights to apply. It can be a masked array.

  weights_sum : xarray.DataArray, optional
    Total weight (i.e., weights.sum()). Only computed if not provided.

  da_mean : xarray.DataArray, optional
   Mean value in DataArray da. Only computed if not provided.

  Returns
  -------
  reduction : DataSet
      xarray.Dataset with (optionally weighted) std for da.
  """

  check_dims(da,dims)
  if weights is not None:
    if weights_sum is None:
      weights_sum = weights.sum(dim=dims)
    if da_mean is None: da_mean = mean_da(da, dims, weights, weights_sum)
    out = np.sqrt(((da-da_mean)**2 * weights).sum(dim=dims)/weights_sum)
    # copy attrs
    out.attrs = da.attrs
    return out
  else:
    return da.std(dim=dims, keep_attrs=True)

def rms_da(da, dims=('yh', 'xh'), weights=None,  weights_sum=None):
  """
  Calculates the rms in DataArray da (optional weighted rms).

  ----------
  da : xarray.DataArray
        DataArray for which to compute (weighted) rms.

  dims : tuple, str
    Dimension(s) over which to apply reduction. Default is ('yh', 'xh').

  weights : xarray.DataArray, optional
    weights to apply. It can be a masked array.

  weights_sum : xarray.DataArray, optional
    Total weight (i.e., weights.sum()). Only computed if not provided.

  Returns
  -------
  reduction : DataSet
      xarray.Dataset with (optionally weighted) rms for da.
  """

  check_dims(da,dims)
  if weights is not None:
    if weights_sum is None: weights_sum = weights.sum(dim=dims)
    out = np.sqrt((da**2 * weights).sum(dim=dims)/weights_sum)
    # copy attrs
    out.attrs = da.attrs
    return out
  else:
    return np.sqrt((da**2).mean(dim=dims, keep_attrs=True))

def check_dims(da,dims):
  """
  Checks if dims exists in ds.
  ----------
  da : xarray.DataArray
        DataArray for which to compute (weighted) min.

  dims : tuple, str
    Dimension(s) over which to apply reduction.
  """
  if dims[0] not in da.dims:
    print('dims[0], da.dims',dims[0], da.dims)
    raise ValueError("DataArray does not have dimensions given by dims[0]")
  if dims[1] not in da.dims:
    print('dims[1], da.dims',dims[1], da.dims)
    raise ValueError("DataArray does not have dimensions given by dims[1]")

  return

def myStats_da(da, weights, dims=('yh', 'xh'), basins=None, debug=False):
  """
  Calculates min, max, mean, standard deviation and root-mean-square for DataArray da
  and returns Dataset with values.

  Parameters
  ----------
  da : xarray.DataArray
        DataArray for which to compute weighted stats.

  dims : tuple, str
    Dimension(s) over which to apply reduction. Default is ('yh', 'xh').

  weights : xarray.DataArray
    weights to apply. It can be a masked array.

  basins : xarray.DataArray, optional
    Basins mask to apply. If True, returns horizontal mean RMSE for each basin provided. \
    Basins must be generated by genBasinMasks. Default is False.

  debug : boolean, optional
    If true, print stuff for debugging. Default is False.

  Returns
  -------
  reduced : DataSet
      New xarray.Dataset with min, max and weighted mean, standard deviation and
      root-mean-square for DataArray ds.
  """
  check_dims(da,dims)
  if weights is None:
    print('compute weights here')
    # compute weights here...

  rmask_od = OrderedDict()
  if basins is None:
    # global
    total_weights = weights.sum(dim=dims)
    da_min  = min_da(da, dims)
    da_max  = max_da(da, dims)
    da_mean = mean_da(da, dims, weights,  total_weights)
    da_std  = std_da(da, dims, weights,  total_weights, da_mean)
    da_rms  = rms_da(da, dims, weights,  total_weights)

    if debug: print_stats(da_min, da_max, da_mean, da_std, da_rms)

    out = stats_to_ds(da_min, da_max, da_mean, da_std, da_rms)
    # copy attrs
    out.attrs = da.attrs
    rmask_od['Global'] = out

  else:
    # aplpy reduction for each basin
    if 'region' not in basins.coords:
      raise ValueError("Regions does not have coordinate region. Please use genBasinMasks \
                        to construct the basins mask.")
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

      if debug:
        print_stats(da_min, da_max, da_mean, da_std, da_rms)

      out = stats_to_ds(da_min, da_max, da_mean, da_std, da_rms)
      rmask_od[str(reg.values)] = out

  return dict_to_da(rmask_od) # create dataarray using rmask_od

def print_stats(da_min, da_max, da_mean, da_std, da_rms):
  """
  Print values for debugging purposes.

  Parameters
  ----------

  da_* : xarray.DataArray
    DataArrays with min, max, std, mean, rms.
  """
  print('myStats: min(da) =' ,da_min)
  print('myStats: max(da) =' ,da_max)
  print('myStats: mean(da) =',da_mean)
  print('myStats: std(da) =' ,da_std)
  print('myStats: rms(da) =' ,da_rms)
  return

def stats_to_ds(da_min, da_max, da_mean, da_std, da_rms):
  """
  Creates a xarray.Dataset using DataArrays provided.

  Parameters
  ----------

  da_* : xarray.DataArray
    DataArrays with min, max, std, mean, rms.

  Returns
  -------
  ds : DataSet
      xarray.Dataset with min, max, mean, standard deviation and
      root-mean-square.
  """
  dim0 = da_min.dims[0]
  dim0_val = da_min[dim0]
  #if 'time' in da_min:
  #  var = np.zeros(len(da_min.time))
  #  time = da_mean['time']
  #else:
  #  var = np.zeros(1)
  #  time = np.array([0.])

  # create dataset with zeros
  ds = xr.Dataset(data_vars={ 'da_min' : ((dim0), da_min.data),
                              'da_max' : ((dim0), da_max.data),
                              'da_std' : ((dim0), da_std.data),
                              'da_rms' : ((dim0), da_rms.data),
                              'da_mean': ((dim0), da_mean.data)},
                   coords={dim0: dim0_val})
  # fill dataset with correct values
  #ds['da_mean'] = da_mean; ds['da_std'] = da_std; ds['da_rms'] = da_rms
  #ds['da_min'] = da_min; ds['da_max'] = da_max
  return ds

def dict_to_da(stats_dict):
  """
  Creates a xarray.DataArray using keys in dictionary (stats_dict).

  Parameters
  ----------

  stats_dict : OrderedDict
    Dictionary with statistics computed using function myStats_da

  Returns
  -------
  da : DataSet
      DataArray with min, max, mean, standard deviation and
      root-mean-square for different basins.
  """

  time = stats_dict[list(stats_dict.items())[0][0]].time
  basins = list(stats_dict.keys())
  stats = ['da_min', 'da_max', 'da_mean', 'da_std', 'da_rms']
  var = np.zeros((len(basins),len(stats),len(time)))
  da = xr.DataArray(var, dims=['basin', 'stats', 'time'],
                           coords={'basin': basins,
                                   'stats': stats,
                                   'time': time},)
  for reg in (basins):
    da.sel(basin=reg).sel(stats='da_min').values[:] = stats_dict[reg].da_min.values
    da.sel(basin=reg).sel(stats='da_max').values[:] = stats_dict[reg].da_max.values
    da.sel(basin=reg).sel(stats='da_mean').values[:]= stats_dict[reg].da_mean.values
    da.sel(basin=reg).sel(stats='da_std').values[:] = stats_dict[reg].da_std.values
    da.sel(basin=reg).sel(stats='da_rms').values[:] = stats_dict[reg].da_rms.values

  return da

def main(stream=False):

  # Get options
  args = options()
  args.nw = args.number_of_workers

  if not args.ocean_stats and not args.surface and not args.forcing and not args.time_series:
    raise ValueError("Please select -ocean_stats, -time_series, -surface and/or -forcing.")

  # Read in the yaml file
  diag_config_yml = yaml.load(open(args.diag_config_yml_path,'r'), Loader=yaml.Loader)

  caseroot = diag_config_yml['Case']['CASEROOT']
  # Create the case instance
  args.casename = cime_xmlquery(caseroot, 'CASE')
  DOUT_S = cime_xmlquery(caseroot, 'DOUT_S')
  if DOUT_S:
    OUTDIR = cime_xmlquery(caseroot, 'DOUT_S_ROOT')+'/ocn/hist/'
  else:
    OUTDIR = cime_xmlquery(caseroot, 'DOUT_S_ROOT')


  # set avg dates and other params
  avg = diag_config_yml['Avg']
  if not args.start_date : args.start_date = avg['start_date']
  if not args.end_date : args.end_date = avg['end_date']
  args.static = args.casename+diag_config_yml['Fnames']['static']
  args.native = args.casename+diag_config_yml['Fnames']['native']
  args.geom = args.casename+diag_config_yml['Fnames']['geom']
  args.rundir = cime_xmlquery(caseroot, 'RUNDIR')
  args.caseroot = caseroot
  args.OUTDIR = OUTDIR

  print('Output directory is:', OUTDIR)
  print('Casename is:', args.casename)
  print('Number of workers: ', args.nw)

  if not os.path.isdir('PNG/Horizontal_mean_biases'):
    print('Creating a directory to place figures (PNG)... \n')
    os.system('mkdir -p PNG/Horizontal_mean_biases')
  if not os.path.isdir('ncfiles'):
    print('Creating a directory to place netCDF files (ncfiles)... \n')
    os.system('mkdir ncfiles')

  # read grid info
  geom_file = OUTDIR+'/'+args.geom
  if os.path.exists(geom_file):
    grd = MOM6grid(OUTDIR+'/'+args.static, geom_file, xrformat=True)
  else:
    grd = MOM6grid(OUTDIR+'/'+args.static, xrformat=True)
  try:
    area = grd.area_t.where(grd.wet > 0)
  except:
    area = grd.areacello.where(grd.wet > 0)

  try:
    depth = grd.depth_ocean.values
  except:
    depth = grd.deptho.values

  # remove Nan's, otherwise genBasinMasks won't work
  # Get masking for different regions
  depth[np.isnan(depth)] = 0.0
  basin_code = genBasinMasks(grd.geolon.values, grd.geolat.values, depth, xda=True)

  #select a few basins, namely, Global, MedSea,BalticSea,HudsonBay Arctic,
  # Pacific, Atlantic, Indian, Southern, LabSea and BaffinBay
  basins = basin_code.isel(region=[0,4,5,6,7,8,9,10,11,12,13])

  if args.ocean_stats:
    _ds = ocean_stats(args)

  if args.surface:
    #variables = ['SSH','tos','sos','mlotst','oml','speed', 'SSU', 'SSV']
    variables = ['SSH','tos','sos','mlotst','oml','speed']
    xystats(args.native, variables, grd, basins, args)

  if args.forcing:
    variables = ['friver','ficeberg','fsitherm','hfsnthermds','sfdsi', 'hflso',
             'seaice_melt_heat', 'wfo', 'hfds', 'Heat_PmE']
    xystats(args.native, variables, grd, basins, args)

  if args.time_series:
    variables = ['thetaoga','soga','opottempmint','somint']
    _ds = extract_time_series(args.native, variables, area, args)

  print('{} was run successfully!'.format(os.path.basename(__file__)))

  return


def ocean_stats(args):
  '''
   Extract time-series from ocean.stats and ocean.stats.nc.

   Parameters
  ----------

  args : object
    Object with command line options and information about the case.

  Returns
  -------
    Two NetCDF files with time series.

  '''

  header = ["Step", "Day","Truncs", "Energy/Mass",
          "Maximum CFL", "Mean Sea Level",
          "Total Mass", "Mean Salin", "Mean Temp",
         "Frac Mass Err", "Salin Err", "Temp Err"]
  # ocean.stats is not archived, so it should be read from RUNDIR
  df = pd.read_csv(args.rundir+'/ocean.stats',  delimiter=',',
                 usecols=(0,1,2,3,4,5,6,7,8,9,10,11),skiprows=(0,1),
                 names=header)

  # remove characters from each column
  for var in header[3::]:
    df[var] = df[var].str.replace(r'En', '')
#   Not sure why this is not working
#    df[var] = df[var].str.replace(r'[^0-9.E-]+', '').astype(float)
#   Explcitly filter out the remaining text strings
    df[var] = df[var].str.replace(r'CFL', '')
    df[var] = df[var].str.replace(r'SL', '')
    df[var] = df[var].str.replace(r'Me', '')
    df[var] = df[var].str.replace(r'Se', '')
    df[var] = df[var].str.replace(r'Te', '')
    df[var] = df[var].str.replace(r'M', '')
    df[var] = df[var].str.replace(r'S', '')
    df[var] = df[var].str.replace(r'T', '')
    df[var] = df[var].astype(float)

  # rename columns
  new_header = ["Step", "Day","Truncs", "EnergyMass",
            "MaximumCFL", "MeanSeaLevel",
            "TotalMass", "MeanSalin", "MeanTemp",
           "FracMassErr", "SalinErr", "TempErr"]
  units = ['nondim', 'days', 'nondim', 'm2 s-2','nondim',
         'm', 'kg', 'PSU' , 'degC', 'nondim', 'PSU', 'degC']
  for old, new in zip(header, new_header):
    df = df.rename(columns={old:new})


  # create dataarray and write to netCDF file
  data_vars = {}
  for var, unit in zip(new_header,units):
    data_vars.update({var:(('time'), df[var], {"units" : unit})})

  # load ocean.stats.nc
  ds = xr.open_dataset(args.rundir+"/ocean.stats.nc").rename({"Time" : "time"})

  # variables to be added
  variables = [ 'En', 'Ntrunc','Mass', 'Mass_chg', 'Mass_anom', 'max_CFL_trans',
                'max_CFL_lin', 'Salt', 'Salt_chg', 'Salt_anom', 'Heat', 'Heat_chg',
                'Heat_anom', 'age']

  for v in variables:
    data_vars.update({v:(('time'), ds[v].values, {"units" : ds[v].units})})

  data_vars.update({"APE":(("time", "Interface"), ds.APE.values)})
  data_vars.update({"H0":(("time", "Interface"), ds.H0.values)})
  data_vars.update({"KE":(("time", "Layer"), ds.KE.values)})
  data_vars.update({"Mass_lay":(("time", "Layer"), ds.Mass_lay.values)})

  RUN_STARTDATE = cime_xmlquery(args.caseroot, 'RUN_STARTDATE')
  time_units = "days since {}".format(RUN_STARTDATE)
  attrs = {"units": time_units, "calendar" : "noleap"}
  coords={"time": ("time", df["Day"], attrs),
        "Layer" : ("Layer", ds.Layer.values),
        "Interface" : ("Interface", ds.Interface.values)}

  msg = " can be found at https://github.com/NCAR/mom6-tools"
  attrs = {"description": "ocean stats time-series derived from ocean.stats and ocean.stats.nc",
           "casename" : args.casename,
           "caseroot" : args.caseroot,
           "author" : getpass.getuser(),
           "date" : datetime.now().isoformat(),
           "created_using" : os.path.basename(__file__),
           "url" : os.path.basename(__file__) + msg}
  stats = xr.decode_cf(xr.Dataset(data_vars=data_vars, coords=coords, attrs = attrs))
  stats.to_netcdf('ncfiles/{}_ocean.stats.nc'.format(args.casename))

  return stats

def extract_time_series(fname, variables, area, args):
  '''
   Extract time-series and saves annual means.

   Parameters
  ----------

  fname : str
    Name of the file to be processed.

  variables : str
    List of variables to be processed.

  args : object
    Object with command line options and information about the case.

  Returns
  -------
    NetCDF file with annual means.

  '''
  parallel = False
  if args.nw > 1:
    parallel = True
    cluster = NCARCluster()
    cluster.scale(args.nw)
    client = Client(cluster)

  def preprocess(ds):
    ''' Return the dataset with variables'''
    return ds[variables]

  # read forcing files
  startTime = datetime.now()
  print('Reading dataset...')
  ds1 = xr.open_mfdataset(args.OUTDIR+'/'+fname, parallel=parallel)
  # use datetime
  #ds1['time'] = ds1.indexes['time'].to_datetimeindex()
  ds = preprocess(ds1)
  opottempmint = ds['opottempmint']
  somint = ds['somint']
  ds.drop_vars(['opottempmint','somint'])
  opottempmint = (opottempmint*area).sum(['yh','xh'])
  opottempmint.attrs['description'] = 'Global integrated temperature'
  opottempmint.attrs['units'] = 'degC kg'

  somint = (somint*area).sum(['yh','xh'])
  somint.attrs['description'] = 'Global integrated salinity'
  somint.attrs['units'] = 'psu kg'
  ds['opottempmint'] = opottempmint
  ds['somint'] = somint

  print('Time elasped: ', datetime.now() - startTime)

  # add attrs and save
  attrs = {'description': 'Monthly averages of global mean ocean properties.'}
  add_global_attrs(ds,attrs)
  ds.to_netcdf('ncfiles/'+str(args.casename)+'_mon_ave_global_means.nc')
  if parallel:
    # close processes
    print('Releasing workers...\n')
    client.close(); cluster.close()

  return ds

def xystats(fname, variables, grd, basins, args):
  '''
   Compute and plot statistics for 2D variables.

   Parameters
  ----------

  fname : str
    Name of the file to be processed.

  variables : str
    List of variables to be processed.

  grd : OrderedDict
    Dictionary with statistics computed using function myStats_da

  basins : DataArray
   Basins mask to apply. Returns horizontal mean RMSE for each basin provided.
   Basins must be generated by genBasinMasks.

  args : object
    Object with command line options and information about the case.

  Returns
  -------
    Plots min, max, mean, std and rms for variables provided and for different basins.

  '''
  parallel = False
  if args.nw > 1:
    parallel = True
    cluster = NCARCluster()
    cluster.scale(args.nw)
    client = Client(cluster)

  try:
    area = grd.area_t.where(grd.wet > 0)
  except:
    area = grd.areacello.where(grd.wet > 0)

  def preprocess(ds):
    ''' Return the dataset with variables'''
    return ds[variables]

  # read forcing files
  startTime = datetime.now()
  print('Reading dataset...')
  ds1 = xr.open_mfdataset(args.OUTDIR+'/'+fname, parallel=parallel)
  ds = preprocess(ds1)

  # use datetime
  #ds['time'] = ds.indexes['time'].to_datetimeindex()

  print('Time elasped: ', datetime.now() - startTime)

  for var in variables:
    startTime = datetime.now()
    print('\n Processing {}...'.format(var))
    savefig1='PNG/'+args.casename+'_'+str(var)+'_xymean.png'
    savefig2='PNG/'+args.casename+'_'+str(var)+'_stats.png'

    # yearly mean
    ds_var = ds[var]
    stats = myStats_da(ds_var, dims=ds_var.dims[1::], weights=area, basins=basins)
    stats.to_netcdf('ncfiles/'+args.casename+'_'+str(var)+'_stats.nc')
    plot_stats_da(stats, var, ds_var.attrs['units'], save=savefig2)
    ds_var_mean = ds_var.mean(dim='time')
    ds_var_mean.to_netcdf('ncfiles/'+args.casename+'_'+str(var)+'_time_ave.nc')
    dummy = np.ma.masked_invalid(ds_var_mean.values)
    xyplot(dummy, grd.geolon.values, grd.geolat.values, area.values, save=savefig1,
           suptitle=ds_var.attrs['long_name'] +' ['+ ds_var.attrs['units']+']',
           title='Averaged between ' +str(ds_var.time[0].values) + ' and '+ str(ds_var.time[-1].values))

    plt.close()
    print('Time elasped: ', datetime.now() - startTime)

  if parallel:
    # close processes
    print('Releasing workers...\n')
    client.close(); cluster.close()

  return

if __name__ == '__main__':
  main()
