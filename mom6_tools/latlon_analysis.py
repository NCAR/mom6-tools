#!/usr/bin/env python

# tools for generating MOM6 diagnostics

import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os, yaml
from mom6_tools import m6plot
from mom6_tools.MOM6grid import MOM6grid
from mom6_tools.m6toolbox import weighted_temporal_mean_vars, request_workers
from mom6_tools.m6toolbox import cime_xmlquery

class MyError(Exception):
  """
  Class for error handling
  """
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

def parseCommandLine():
  """
  Parse the command line positional and optional arguments.
  This is the highest level procedure invoked from the very end of the script.
  """
  parser = argparse.ArgumentParser(description=
      '''
      Perform latlon analysis (e.g., time averages and spatial mean) and call plotting functions.
      ''',
  epilog='Written by Gustavo Marques (gmarques@ucar.edu).')

  parser.add_argument('infile', type=str, help='''Path to the file to be processed.
                      It can be either sfc_* or frc_* files (i.e <some_path>/run/case_name.mom6.sfc_*)''')

  parser.add_argument('-case_name', type=str, default='test',
      help='''Case name. Default is test.''')

  parser.add_argument('-static', type=str, default='ocean_static.nc',
      help='''The name of the ocean geometry file. Default is ocean_static.nc''')

  parser.add_argument('-variables', type=str, default=None,
      help='''Tuple containing variables to be averaged and plotted.
      Variable must besepareted by comma (e.g., SST,SSS).
      Default is none, which will process all variables.''')

  parser.add_argument('-year_start', type=int, default=80,
      help='''Start year to compute averages. Default is 80.''')

  parser.add_argument('-year_end', type=int, default=100,
      help='''End year to compute averages. Default is 100.''')

  parser.add_argument('-to_netcdf', help='''Save data into a netCDF file.''',
      action="store_true")

  parser.add_argument('-savefigs', help='''Save figures in a PNG format.''',
      action="store_true")

  parser.add_argument('-time_series', help='''If true, plot time-series of area-averaged statistics''',
      action="store_true")

  optCmdLineArgs = parser.parse_args()
  driver(optCmdLineArgs)

#-- This is where all the action happends, i.e., functions for each diagnostic are called.

def driver(args):
  os.system('mkdir PNG')
  os.system('mkdir ncfiles')

  # Read in the yaml file
  diag_config_yml_path = "diag_config.yml"
  diag_config_yml = yaml.load(open(diag_config_yml_path,'r'), Loader=yaml.Loader)

  caseroot = diag_config_yml['Case']['CASEROOT']
  casename = cime_xmlquery(caseroot, 'CASE')
  DOUT_S = cime_xmlquery(caseroot, 'DOUT_S')
  if DOUT_S:
    OUTDIR = cime_xmlquery(caseroot, 'DOUT_S_ROOT')+'/ocn/hist/'
  else:
    OUTDIR = cime_xmlquery(caseroot, 'RUNDIR')

  print('Output directory is:', OUTDIR)
  print('Casename is:', casename)

  args.static = casename+diag_config_yml['Fnames']['static']
  args.geom =   casename+diag_config_yml['Fnames']['geom']

  # mom6 grid
  # read grid info
  geom_file = OUTDIR+'/'+args.geom
  if os.path.exists(geom_file):
    grd = MOM6grid(OUTDIR+'/'+args.static, geom_file)
  else:
    grd = MOM6grid(OUTDIR+'/'+args.static)

  variables = args.variables.split(',')
  time_mean_latlon(args,grd,variables)

  return

# -- mean surface latlon time series from forcing and surface files
def plot_area_ave_stats(ds, var, args, aspect=[16,9], resolution=576, debug=False):
  # TODO: this can be done in a more elegant way...
  #m6plot.setFigureSize(aspect, resolution, debug=debug)
  f, ax = plt.subplots(5, sharex=True,figsize=(12,10))
  ax[0].plot(ds['time'], ds[var][2,:])
  ax[0].set_title(r'%s, [%s] - Area-averaged statistics'%(var,ds[var].units))
  ax[0].set_ylabel('Mean')
  ax[1].plot(ds['time'], ds[var][0,:])
  ax[1].set_ylabel('Min.')
  ax[2].plot(ds['time'], ds[var][1,:])
  ax[2].set_ylabel('Max.')
  ax[3].plot(ds['time'], ds[var][3,:])
  ax[3].set_ylabel('Std')
  ax[4].plot(ds['time'], ds[var][4,:])
  ax[4].set_ylabel('Rms')
  ax[4].set_xlabel('Year')
  if args.savefigs:
    plt.savefig('PNG/%s_stats.png'%(var))
  else:
    plt.show()

  return

# -- time-average 2D latlon fields and call plotting function
def time_mean_latlon(args, grd, variables=[]):

  if args.nw>1:
    parallel, cluster, client = request_workers(args.nw)

    ds = xr.open_mfdataset(args.infile, \
         parallel=True, data_vars='minimal', chunks={'time': 12},\
         coords='minimal', compat='override')
  else:
    ds = xr.open_mfdataset(args.infile, \
         data_vars='minimal', chunks={'time': 12},\
         coords='minimal', compat='override')

  if len(variables) == 0:
    # plot all 2D varialbles in the dataset
    variables = ds.variables

  ds = ds[variables]
  ds_sel = ds.sel(time=slice(args.start_date,args.end_date))
  ds_ann = weighted_temporal_mean_vars(ds_sel)
  ds1 = ds_ann.mean('time').compute()

  try:
    area = grd.area_t
  except:
    area = grd.areacello

  for var in variables:
    dim = len(ds1[var].shape)
    if dim == 2:
      filename = str('PNG/%s.png' % (var))
      if os.path.isfile(filename):
        print (' \n' + '==> ' + '{} has been saved, moving to the next one ...\n' + ''.format(var))
      else:
        print("About to plot time-average for {} ({})... \n".format(ds[var].long_name, var))
        data = np.ma.masked_invalid(ds1[var].values)
        units = ds[var].attrs['units']

        if args.savefigs:
          m6plot.xyplot( data , grd.geolon, grd.geolat, area=area,
            suptitle=args.case_name,
            title=r'%s, [%s] averaged over years %s-%s'%(var,units,args.start_date,args.end_date),
            extend='both',
            save=filename)
        else:
          m6plot.xyplot( data , grd.geolon, grd.geolat, area=area,
            suptitle=args.case_name,
            title=r'%s, [%s] averaged over years %s-%s'%(var,units,args.start_date,args.end_date),
            extend='both',
            show=True)

    if args.time_series:
      # create Dataset
      dtime = ds1.time.values
      data = np.ma.masked_invalid(ds1[var].values)
      ds_new = create_xarray_dataset(var,units,dtime)
      # loop in time
      for t in range(0,len(dtime)):
        #print ("==> ' + 'step # {} out of {}  ...\n".format(t+1,tm))
        # get stats
        sMin, sMax, mean, std, rms = m6plot.myStats(data[t], area)
        # update Dataset
        ds_new[var][0,t] = sMin; ds_new[var][1,t] = sMax; ds_new[var][2,t] = mean
        ds_new[var][3,t] = std; ds_new[var][4,t] = rms

      # plot
      plot_area_ave_stats(ds_new, var, args)
      #if args.to_netcdf:
      # save in a netcdf file
      #ds.to_netcdf('ncfiles/'+args.case_name+'_stats.nc')
  if args.nw>1:
    client.close(); cluster.close()

  return

# -- create a xarray Dataset given variable, unit and time
def create_xarray_dataset(variable,unit,time):
   ds = xr.Dataset()
   # TODO: fix the time using pandas, # of years are limited, see link below
   # http://pandas-docs.github.io/pandas-docs-travis/timeseries.html#timestamp-limitations
   # It should be days since 0001-01-01 00:00:00
   ds.coords['time'] = time
   stats = ['min','max','mean','std','rms']
   ds.coords['stats'] = stats
   tm = len(time)
   ds[variable] = (('stats', 'time'), np.zeros((len(stats),tm)))
   ds[variable].attrs['units'] = unit

   return ds


# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': parseCommandLine()

