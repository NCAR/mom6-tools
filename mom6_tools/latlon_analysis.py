#!/usr/bin/env python

# tools for generating MOM6 diagnostics

import argparse
import xarray as xr
from netCDF4 import MFDataset, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import m6plot
import warnings
import os
from MOM6grid import MOM6grid

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

#  parser.add_argument('-month', type=str, default='ocean_month__*',
#      help='''The name of monthly mean file(s). Default is ocean_month__*''')

#  parser.add_argument('-month_z_', type=str, default='ocean_month_z__*',
#      help='''The name of monthly mean file(s) remapped to a z coordinate. Default is ocean_month_z__*''')

#  parser.add_argument('-prog', type=str, default='prog__*',
#      help='''The name of prognostic ocean file(s). Default is prog__*''')

#  parser.add_argument('-ndays', type=int, default=2,
#      help='''Number of days to skip when computing time series. Default is 2.''')

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
  # mom6 grid
  grd = MOM6grid(args.static)

  variables = args.variables.split(',')
  # extract mean surface latlon time series from forcing and surface files
  #mean_latlon_time_series(args, grd, ['SSS','SST','MLD_003','SSH','hfds','PRCmE','taux','tauy'])
  # FIXME: SSU and SSV need to be plotted on u and v points, instead of tracer points
  #time_mean_latlon(args,grd, ['SSH','SSS','SST','KPP_OBLdepth','SSU','SSV','hfds','PRCmE','taux','tauy'])
  time_mean_latlon(args,grd,variables)
  #mean_latlon_plot(args,grd,['SSH','SSS','SST','KPP_OBLdepth','SSU','SSV','hfds','PRCmE','taux','tauy'])

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

def check_time_interval(ti,tf,nc):
 ''' Checks if year_start and year_end are within the time interval of the dataset'''
 if ti < nc.time.min() or tf > nc.time.max():
    #raise NameError('Selected start/end years outside the range of the dataset. Please fix that and run again.')
    raise SyntaxError('Selected start/end years outside the range of the dataset. Please fix that and run again.')

 return

# -- time-average 2D latlon fields and call plotting function
def time_mean_latlon(args, grd, variables=[]):
  xr_mfopen_args = {'decode_times':False,
                  'decode_coords':False}

  nc = xr.open_mfdataset(args.infile, **xr_mfopen_args)

  if not nc.time.attrs['calendar'] == 'NOLEAP':
    raise NameError('Only noleap calendars are supported at this moment!')

  # TODO: assign a new variable called time_years
  # convert time in years
  nc['time'] = nc.time/365.

  ti = args.year_start
  tf = args.year_end

  # check if data includes years between ti and tf
  check_time_interval(ti,tf,nc)

  if len(variables) == 0:
    # plot all 2D varialbles in the dataset
    variables = nc.variables

  for var in variables:
    dim = len(nc[var].shape)
    if dim == 3:
      filename = str('PNG/%s.png' % (var))
      if os.path.isfile(filename):
        print (' \n' + '==> ' + '{} has been saved, moving to the next one ...\n' + ''.format(var))
      else:
        print("About to plot time-average for {} ({})... \n".format(nc[var].long_name, var))
        data = np.ma.masked_invalid(nc[var].sel(time=slice(ti,tf)).mean('time').values)
        units = nc[var].attrs['units']

        if args.savefigs:
          m6plot.xyplot( data , grd.geolon, grd.geolat, area=grd.area_t,
            suptitle=args.case_name,
            title=r'%s, [%s] averaged over years %i-%i'%(var,units,ti,tf),
            extend='both',
            save=filename)
        else:
          m6plot.xyplot( data , grd.geolon, grd.geolat, area=grd.area_t,
            suptitle=args.case_name,
            title=r'%s, [%s] averaged over years %i-%i'%(var,units,ti,tf),
            extend='both',
            show=True)

        if args.time_series:
          # create Dataset
          dtime = nc.time.sel(time=slice(ti,tf)).values
          data = np.ma.masked_invalid(nc[var].sel(time=slice(ti,tf)).values)
          ds = create_xarray_dataset(var,units,dtime)
          # loop in time
          for t in range(0,len(dtime)):
            #print ("==> ' + 'step # {} out of {}  ...\n".format(t+1,tm))
            # get stats
            sMin, sMax, mean, std, rms = m6plot.myStats(data[t], grd.area_t)
            # update Dataset
            ds[var][0,t] = sMin; ds[var][1,t] = sMax; ds[var][2,t] = mean
            ds[var][3,t] = std; ds[var][4,t] = rms

          # plot
          plot_area_ave_stats(ds, var, args)
          #if args.to_netcdf:
            # save in a netcdf file
            #ds.to_netcdf('ncfiles/'+args.case_name+'_stats.nc')
  nc.close()
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

