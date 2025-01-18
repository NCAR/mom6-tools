#!/usr/bin/env python

import io, yaml, os
import matplotlib.pyplot as plt
import numpy as np
import warnings, dask, netCDF4, intake
from datetime import datetime, date
import xarray as xr
from ncar_jobqueue import NCARCluster
from dask.distributed import Client
import cftime
import nc_time_axis
import momlevel as ml
import xwavelet as xw
from mom6_tools.MOM6grid import MOM6grid
from mom6_tools.m6toolbox import add_global_attrs
from mom6_tools.m6toolbox import cime_xmlquery
from mom6_tools.m6toolbox import weighted_temporal_mean_vars
from mom6_tools.m6toolbox import geoslice
import matplotlib.pyplot as plt
import yaml, os, intake, pickle


def options():
  try: import argparse
  except: raise Exception('This version of python is not new enough. python 2.7 or newer is required.')
  parser = argparse.ArgumentParser(description='''Script for computing and plotting nino3.4 variability and composite.. \
    Acknowledgment: this script builds on work by John Krasting (NOAA/GFDL). \
    The original version can be found at:  \
    https://github.com/jkrasting/mar/blob/main/src/gfdlnb/notebooks/ocean/ENSO_variability.ipynb.\
    ''')
  parser.add_argument('diag_config_yml_path', type=str, help='''Full path to the yaml file  \
    describing the run and diagnostics to be performed.''')
  parser.add_argument('-sd','--start_date', type=str, default='',
                      help='''Start year to compute averages. Default is to use value set in diag_config_yml_path''')
  parser.add_argument('-ed','--end_date', type=str, default='',
                      help='''End year to compute averages. Default is to use value set in diag_config_yml_path''')
  parser.add_argument('-nw','--number_of_workers',  type=int, default=2,
                      help='''Number of workers to use (default=2).''')
  parser.add_argument('-ys','--year_shift',  type=int, default='0',
                      help="An integer used to shift the time coordinate. For example, for G cases forced with JRA-55 \
                            one might set -ys 1957. Default is 0.")
  parser.add_argument('-debug',   help='''Add priting statements for debugging purposes''',
                      action="store_true")

  cmdLineArgs = parser.parse_args()
  return cmdLineArgs

def main(stream=False):
  # Get options
  args = options()
  nw = args.number_of_workers
  if not os.path.isdir('PNG/ENSO'):
    print('Creating a directory to place figures (PNG/ENSO)... \n')
    os.system('mkdir -p PNG/ENSO')
  if not os.path.isdir('ncfiles'):
    print('Creating a directory to store netcdf files (ncfiles)... \n')
    os.system('mkdir ncfiles')

  # Read in the yaml file
  diag_config_yml = yaml.load(open(args.diag_config_yml_path,'r'), Loader=yaml.Loader)

  caseroot = diag_config_yml['Case']['CASEROOT']
  args.casename = cime_xmlquery(caseroot, 'CASE')
  DOUT_S = cime_xmlquery(caseroot, 'DOUT_S')
  if DOUT_S:
    OUTDIR = cime_xmlquery(caseroot, 'DOUT_S_ROOT')+'/ocn/hist/'
  else:
    OUTDIR = cime_xmlquery(caseroot, 'RUNDIR')

  print('Output directory is:', OUTDIR)
  print('Casename is:', args.casename)
  print('Number of workers to be used:', nw)

  # set avg dates and other params
  avg = diag_config_yml['Avg']
  if not args.start_date : args.start_date = avg['start_date']
  if not args.end_date : args.end_date = avg['end_date']
  args.native = args.casename+diag_config_yml['Fnames']['native']
  args.static = args.casename+diag_config_yml['Fnames']['static']
  args.geom = args.casename+diag_config_yml['Fnames']['geom']
  args.savefigs = True
  args.label = diag_config_yml['Case']['SNAME']
  args.outdir = 'PNG/ENSO/'

  # read grid info
  geom_file = OUTDIR+'/'+args.geom
  if os.path.exists(geom_file):
    grd = MOM6grid(OUTDIR+'/'+args.static, geom_file, xrformat=True)
  else:
    grd = MOM6grid(OUTDIR+'/'+args.static, xrformat=True)

  try:
    depth = grd.depth_ocean
  except:
    depth = grd.deptho

  parallel = False
  if nw > 1:
    parallel = True
    cluster = NCARCluster()
    cluster.scale(nw)
    client = Client(cluster)

  def preprocess(ds):
    ''' Return a dataset desired variables'''
    variables = ['tos']
    return ds[variables]

  print('Reading dataset...')
  startTime = datetime.now()

  ds = xr.open_mfdataset(OUTDIR+'/'+args.native, parallel=parallel, \
                             combine="nested", concat_dim="time", \
                             preprocess=preprocess).chunk({"time": 12})

  print('Time elasped: ', datetime.now() - startTime)

  # Add the latitude, longitude, and areacello
  ds = ds.assign_coords({
    "latitude": (("yh", "xh"), grd.geolat.data),
    "longitude": (("yh", "xh"), grd.geolon.data),
    "areacello": (("yh", "xh"), grd.areacello.fillna(0.).data)
  })

  # Nino3.4 SST
  nino34 = geoslice(ds.tos,y=(-5,5),x=(-170,-120),
                    xcoord="longitude", ycoord="latitude")

  std_dev_model = nino34.std()

  gb = nino34.groupby('time.month')
  nino34_anom = gb - gb.mean(dim='time')
  index_nino34_model = nino34_anom.weighted(nino34.areacello).mean(dim=['yh', 'xh'])

  index_nino34_model_rolling_mean = index_nino34_model.rolling(time=5, center=True).mean()
  normalized_index_nino34_model_rolling_mean = index_nino34_model_rolling_mean / std_dev_model

  nino34 = nino34.weighted(nino34.areacello).mean(("yh","xh"))
  nino34 = nino34.load()
  result_model = xw.Wavelet(nino34, scaled=True)

  # Option to shift by args.year_shift years
  if args.year_shift > 0:
    print('Shifting time coordinate by {}\n'.format(args.year_shift))
    time = normalized_index_nino34_model_rolling_mean.time.data
    shifted_time = [t.replace(year=t.year + args.year_shift) for t in time]
    # Convert back to xarray coordinate if needed
    time_shifted = xr.DataArray(shifted_time, dims=["time"], name="shifted_time")
    normalized_index_nino34_model_rolling_mean['time'] = time_shifted


  x = normalized_index_nino34_model_rolling_mean.time.data
  y1 = normalized_index_nino34_model_rolling_mean.where(normalized_index_nino34_model_rolling_mean >= 0.4).compute().data
  y2 = normalized_index_nino34_model_rolling_mean.where(normalized_index_nino34_model_rolling_mean <= -0.4).compute().data

  #Apply the Conditions to Create the New Data Array
  # Define conditions
  conditions = [
      normalized_index_nino34_model_rolling_mean >= 0.4,
      normalized_index_nino34_model_rolling_mean <= -0.4
  ]
  # Define corresponding values
  values = [1, -1]
  # Apply conditions
  index = np.select(conditions, values, default=0)
  # Create DataArray
  nino34_index = xr.DataArray(
      index,
      coords=[('time',x)],
      name='nino34_index'
  )

  # Add the DataArray to the Dataset
  normalized_index_nino34_model_rolling_mean['nino34_index'] = nino34_index

  if args.savefigs:
    print('Plotting...\n')

    y1 = np.nan_to_num(y1)
    y2 = np.nan_to_num(y2)

    fig = plt.figure(figsize=(12, 6))

    plt.fill_between(
        x,
        y1,
        0.4,
        color='red',
        alpha=0.9,
    )
    plt.fill_between(
        x,
        y2,
        -0.4,
        color='blue',
        alpha=0.9,
    )

    #plt.plot(x, normalized_index_nino34_model_rolling_mean.values, color='black')
    normalized_index_nino34_model_rolling_mean.plot(color='black')
    plt.axhline(0, color='black', lw=0.5)
    plt.axhline(0.4, color='black', linewidth=0.5, linestyle='dotted')
    plt.axhline(-0.4, color='black', linewidth=0.5, linestyle='dotted')
    plt.title('Case {}, Niño 3.4 Index'.format(args.label));
    fname = args.outdir + str(args.casename)+'_nino34_index.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

    fig = result_model.composite()
    fname = args.outdir + str(args.casename)+'_nino34_composite.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

  print('Saving netCDF files...')
  normalized_index_nino34_model_rolling_mean.to_netcdf('ncfiles/'+str(args.casename)+'_nino34_index.nc')
  fname = "ncfiles/" + str(args.casename)+'_nino34_composite.pkl'
  with open(fname, "wb") as file:
    pickle.dump(result_model, file)

  if parallel:
    print('Releasing workers...')
    client.close(); cluster.close()

  print('{} was run successfully!'.format(os.path.basename(__file__)))
  return

def plot_enso_obs(obs, label='oisstv2', basetime="1970-01-01T00:00:00Z"):
  '''Compute and plot enso3.4 index and composite using obs.'''

  # Nino3.4 SST
  nino34_obs = geoslice(obs.sst,y=(-5,5),x=(-170,-120), xcoord="geolon", ycoord="geolat")
  gb = nino34_obs.groupby('time.month')
  nino34_obs_anom = gb - gb.mean(dim='time')
  index_nino34 = nino34_obs_anom.weighted(nino34_obs.areacello).mean(dim=['yh', 'xh'])
  nino34_obs = nino34_obs.weighted(nino34_obs.areacello).mean(("yh","xh"))
  nino34_obs = nino34_obs.load()

  time_values = nino34_obs.time.values  # Extract the time values
  # Convert datetime64 to cftime.DatetimeGregorian
  new_time = [
      cftime.DatetimeGregorian(
          datetime.utcfromtimestamp((t - np.datetime64(basetime)) / np.timedelta64(1, "s")).year,
          datetime.utcfromtimestamp((t - np.datetime64(basetime)) / np.timedelta64(1, "s")).month,
          datetime.utcfromtimestamp((t - np.datetime64(basetime)) / np.timedelta64(1, "s")).day,
          datetime.utcfromtimestamp((t - np.datetime64(basetime)) / np.timedelta64(1, "s")).hour,
          datetime.utcfromtimestamp((t - np.datetime64(basetime)) / np.timedelta64(1, "s")).minute,
          datetime.utcfromtimestamp((t - np.datetime64(basetime)) / np.timedelta64(1, "s")).second,
      )
      for t in time_values
  ]

  # Replace the time coordinate
  nino34_obs = nino34_obs.assign_coords(time=("time", new_time))
  result_obs = xw.Wavelet(nino34_obs, scaled=True)
  fig = result_obs.composite()

  index_nino34_rolling_mean = index_nino34.rolling(time=5, center=True).mean()
  std_dev = nino34_obs.std()
  normalized_index_nino34_rolling_mean = index_nino34_rolling_mean / std_dev
  fig = plt.figure(figsize=(12, 6))
  plt.fill_between(
      normalized_index_nino34_rolling_mean.time.data,
      normalized_index_nino34_rolling_mean.where(
          normalized_index_nino34_rolling_mean >= 0.4
      ).data,
      0.4,
      color='red',
      alpha=0.9,
  )
  plt.fill_between(
      normalized_index_nino34_rolling_mean.time.data,
      normalized_index_nino34_rolling_mean.where(
          normalized_index_nino34_rolling_mean <= -0.4
      ).data,
      -0.4,
      color='blue',
      alpha=0.9,
  )

  normalized_index_nino34_rolling_mean.plot(color='black')
  plt.axhline(0, color='black', lw=0.5)
  plt.axhline(0.4, color='black', linewidth=0.5, linestyle='dotted')
  plt.axhline(-0.4, color='black', linewidth=0.5, linestyle='dotted')
  plt.title('OiSSTv2, Niño 3.4 Index');

  return

if __name__ == '__main__':
  main()

