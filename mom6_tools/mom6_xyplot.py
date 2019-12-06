#!/usr/bin/env python

# generates MOM6 diagnostics

import argparse
from netCDF4 import MFDataset, Dataset
import numpy as np
import matplotlib.pyplot as plt
from mom6_tools.m6plot import xyplot
from mom6_tools.MOM6grid import MOM6grid
import warnings
import os

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
      Make lat/lon plots for MOM6.
      ''',
  epilog='Written by Gustavo Marques, Jan. 2018.')

  parser.add_argument('-case_name', type=str, default='test',
      help='''Case name. Default is test.''')

  parser.add_argument('-geometry', type=str, default='ocean_geometry.nc',
      help='''The name of the ocean geometry file. Default is ocean_geometry.nc''')

  parser.add_argument('-outfile', type=str, default='sfc_daily__0000.nc',
      help='''The name of the netCDF file to be used. Default is sfc_daily__0000.nc''')

  parser.add_argument('-variable', type=str, default='SST',
      help='''The name of the variable to be plotted. Default is SST.''')

  parser.add_argument('-start', type=int, default=0,
      help='''Initial time indice to be plotted. Default is 0.''')

  parser.add_argument('-end', type=int, default=-1,
      help='''Final time indice to be plotted. Default is -1.''')

  optCmdLineArgs = parser.parse_args()
  global case_name
  case_name = optCmdLineArgs.case_name
  driver(optCmdLineArgs)

#-- This is where all the action happends, i.e., functions for each diagnostic are called.

def driver(args):
  os.system('mkdir PNG')
  # mom6 grid
  grd = MOM6grid(args.geometry)

  latlon_plot(args,args.outfile,grd,args.variable)

  return

# -- time-mean latlon plot
def latlon_plot(args, ncfile, grd, variable):
  nc = MFDataset(ncfile)
  time = nc.variables['time'][:]/365.
  tm = len(time)
  for var in nc.variables:
    if var == variable:
      print("### About to plot variable {} ### \n".format(var))
      if var == 'SSH':
        clim=[-2,2]
      elif var == 'SSS':
        clim=[30.75,38.0]
      elif var == 'SST':
        clim = [-1.5,31]
      elif var == 'KPP_OBLdepth':
        clim = [0,500]
      elif var == 'MEKE':
        clim = [0,0.3]
      elif var == 'MLD_003':
        clim = [0,2000]

      for t in range(tm):
        filename = str('PNG/%s_%05d.png' % (var,t))
        if os.path.isfile(filename):
          print("File {} already exists! Moving to the next one...\n".format(filename))
        else:
          print ("time index {} of {}".format(t, tm))
          data = nc.variables[var][t,:]
          units = nc.variables[var].units
          #TODO: convert days to date
          xyplot( data , grd.geolon, grd.geolat, area=grd.Ah,
            suptitle=case_name,
            title=r'%s, [%s] - Year: %5.1f'%(var,units,time[t]),
            extend='both',
            clim=clim,
            save=filename)

  nc.close()

  return

# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': parseCommandLine()



