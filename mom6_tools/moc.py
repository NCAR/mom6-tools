#!/usr/bin/env python

import io
import numpy
import matplotlib.pyplot as plt
import os
import sys
import xarray as xr

from mom6_tools import m6plot
from mom6_tools import m6toolbox
from mom6_tools.MOM6grid import MOM6grid

def options():
  try: import argparse
  except: raise Exception('This version of python is not new enough. python 2.7 or newer is required.')
  parser = argparse.ArgumentParser(description='''Script for plotting meridional overturning circulation.''')
  parser.add_argument('infile', type=str, help='''Path to the file(s) to be processed (i.e <some_path>/run/)''')
  parser.add_argument('-l','--label', type=str, default='', help='''Label to add to the plot.''')
  parser.add_argument('-n','--case_name', type=str, default='', help='''Case name.  Default is to read from netCDF file.''')
  parser.add_argument('-savefigs', help='''Save figures in a PNG format.''', action="store_true")
  parser.add_argument('-start_date', type=str, default='0001-01-01', help='''Start year to compute averages. Default 0001-01-01''')
  parser.add_argument('-end_date', type=str, default='0100-12-31',  help='''End year to compute averages. Default 0100-12-31''')

  cmdLineArgs = parser.parse_args()
  return cmdLineArgs
  main(cmdLineArgs)

def main():
  # Get options
  args = options()
  # mom6 grid
  grd = MOM6grid(args.infile+args.case_name+'.mom6.static.nc')
  depth = grd.depth_ocean
  # remote Nan's, otherwise genBasinMasks won't work
  depth[numpy.isnan(depth)] = 0.0
  basin_code = m6toolbox.genBasinMasks(grd.geolon, grd.geolat, depth)

  # load data
  ds = xr.open_mfdataset(args.infile+args.case_name+'.mom6.hm_*.nc', combine='by_coords')
  ti = args.start_date
  tf = args.end_date

  # create a ndarray subclass
  class C(numpy.ndarray): pass

  if 'vmo' in ds.variables:
    varName = 'vmo'; conversion_factor = 1.e-9
  elif 'vh' in ds.variables:
    varName = 'vh'; conversion_factor = 1.e-6
    if 'zw' in ds.variables: conversion_factor = 1.e-9 # Backwards compatible for when we had wrong units for 'vh'
  else: raise Exception('Could not find "vh" or "vmo" in file "%s"'%(args.infile+args.static))

  # selected dates
  ds_var = ds[varName].sel(time=slice(ti, tf))

  # yearly means
  ds_var_yr = ds_var.resample(time="1Y", closed='left', keep_attrs=True).mean(dim='time', keep_attrs=True).load()

  tmp = numpy.ma.masked_invalid(ds_var_yr.mean('time').values)
  tmp = tmp[:].filled(0.)
  VHmod = tmp.view(C)
  VHmod.units = ds[varName].units

  Zmod = m6toolbox.get_z(ds, depth, varName)

  if args.case_name != '':  case_name = args.case_name + ' ' + args.label
  else: case_name = ds.title + ' ' + args.label

  imgbufs = []

  # Global MOC
  m6plot.setFigureSize([16,9],576,debug=False)
  axis = plt.gca()
  cmap = plt.get_cmap('dunnePM')
  z = Zmod.min(axis=-1)
  psiPlot = MOCpsi(VHmod)*conversion_factor
  psiPlot = 0.5 * (psiPlot[0:-1,:]+psiPlot[1::,:])
  yy = grd.geolat_c[:,:].max(axis=-1)+0*z
  ci=m6plot.pmCI(0.,40.,5.)
  plotPsi(yy, z, psiPlot, ci, 'Global MOC [Sv],' + 'averaged between '+ ti + 'and '+ tf )
  plt.xlabel(r'Latitude [$\degree$N]')
  plt.suptitle(case_name)
  plt.gca().invert_yaxis()
  findExtrema(yy, z, psiPlot, max_lat=-30.)
  findExtrema(yy, z, psiPlot, min_lat=25.)
  findExtrema(yy, z, psiPlot, min_depth=2000., mult=-1.)
  objOut = str(case_name)+'_MOC_global.png'
  plt.savefig(objOut)

  # Atlantic MOC
  m6plot.setFigureSize([16,9],576,debug=False)
  cmap = plt.get_cmap('dunnePM')
  m = 0*basin_code; m[(basin_code==2) | (basin_code==4) | (basin_code==6) | (basin_code==7) | (basin_code==8)]=1
  ci=m6plot.pmCI(0.,22.,2.)
  z = (m*Zmod).min(axis=-1)
  psiPlot = MOCpsi(VHmod, vmsk=m*numpy.roll(m,-1,axis=-2))*conversion_factor
  psiPlot = 0.5 * (psiPlot[0:-1,:]+psiPlot[1::,:])
  yy = grd.geolat_c[:,:].max(axis=-1)+0*z
  plotPsi(yy, z, psiPlot, ci, 'Atlantic MOC [Sv],'+ 'averaged between '+ ti + 'and '+ tf )
  plt.xlabel(r'Latitude [$\degree$N]')
  plt.suptitle(case_name)
  plt.gca().invert_yaxis()
  findExtrema(yy, z, psiPlot, min_lat=26.5, max_lat=27.) # RAPID
  findExtrema(yy, z, psiPlot, max_lat=-33.)
  findExtrema(yy, z, psiPlot)
  findExtrema(yy, z, psiPlot, min_lat=5.)
  objOut = str(case_name)+'_MOC_Atlantic.png'
  plt.savefig(objOut,format='png')

  # time-series
  dtime = ds_var_yr.time.values
  amoc_26 = numpy.zeros(len(dtime))
  amoc_45 = numpy.zeros(len(dtime))

  # loop in time
  for t in range(len(dtime)):
    tmp = numpy.ma.masked_invalid(ds_var_yr.sel(time=dtime[t]).values)
    tmp = tmp[:].filled(0.)
    psi = MOCpsi(tmp, vmsk=m*numpy.roll(m,-1,axis=-2))*conversion_factor
    psi = 0.5 * (psi[0:-1,:]+psi[1::,:])
    amoc_26[t] = findExtrema(yy, z, psi, min_lat=26.5, max_lat=27., plot=False)
    amoc_45[t] = findExtrema(yy, z, psi, min_lat=44., max_lat=46., plot=False)

  # create dataarays
  amoc_26_da = xr.DataArray(amoc_26, dims=['time'],
                           coords={'time': dtime})
  amoc_45_da = xr.DataArray(amoc_45, dims=['time'],
                           coords={'time': dtime})

  # load AMOC time series data (5th) cycle used in Danabasoglu et al., doi:10.1016/j.ocemod.2015.11.007
  path = '/glade/p/cesm/omwg/amoc/COREII_AMOC_papers/papers/COREII.variability/data.original/'
  amoc_core_26 = xr.open_dataset(path+'AMOCts.cyc5.26p5.nc')
  # plot
  fig = plt.figure(figsize=(12, 6))
  plt.plot(numpy.arange(len(amoc_26_da.time))+1948.5 ,amoc_26_da.values, color='k', label=case_name, lw=2)
  # core data
  core_mean = amoc_core_26['MOC'].mean(axis=0).data
  core_std = amoc_core_26['MOC'].std(axis=0).data
  plt.plot(amoc_core_26.time,core_mean, 'k', label='CORE II (group mean)', color='#1B2ACC', lw=2)
  plt.fill_between(amoc_core_26.time, core_mean-core_std, core_mean+core_std,
      alpha=0.25, edgecolor='#1B2ACC', facecolor='#089FFF')
  plt.title('AMOC @ 26 $^o$ N', fontsize=16)
  plt.xlabel('Time [years]', fontsize=16); plt.ylabel('Sv', fontsize=16)
  plt.legend(fontsize=14)
  objOut = str(case_name)+'_MOC_26N_time_series.png'
  plt.savefig(objOut,format='png')

  amoc_core_45 = xr.open_dataset(path+'AMOCts.cyc5.45.nc')
  # plot
  fig = plt.figure(figsize=(12, 6))
  plt.plot(numpy.arange(len(amoc_45_da.time))+1948.5 ,amoc_45_da.values, color='k', label=case_name, lw=2)
  # core data
  core_mean = amoc_core_45['MOC'].mean(axis=0).data
  core_std = amoc_core_45['MOC'].std(axis=0).data
  plt.plot(amoc_core_45.time,core_mean, 'k', label='CORE II (group mean)', color='#1B2ACC', lw=2)
  plt.fill_between(amoc_core_45.time, core_mean-core_std, core_mean+core_std,
      alpha=0.25, edgecolor='#1B2ACC', facecolor='#089FFF')
  plt.title('AMOC @ 45 $^o$ N', fontsize=16)
  plt.xlabel('Time [years]', fontsize=16); plt.ylabel('Sv', fontsize=16)
  plt.legend(fontsize=14)
  objOut = str(case_name)+'_MOC_45N_time_series.png'
  plt.savefig(objOut,format='png')
  return

def MOCpsi(vh, vmsk=None):
  """Sums 'vh' zonally and cumulatively in the vertical to yield an overturning stream function, psi(y,z)."""
  shape = list(vh.shape); shape[-3] += 1
  psi = numpy.zeros(shape[:-1])
  if len(shape)==3:
    for k in range(shape[-3]-1,0,-1):
      if vmsk is None: psi[k-1,:] = psi[k,:] - vh[k-1].sum(axis=-1)
      else: psi[k-1,:] = psi[k,:] - (vmsk*vh[k-1]).sum(axis=-1)
  else:
    for n in range(shape[0]):
      for k in range(shape[-3]-1,0,-1):
        if vmsk is None: psi[n,k-1,:] = psi[n,k,:] - vh[n,k-1].sum(axis=-1)
        else: psi[n,k-1,:] = psi[n,k,:] - (vmsk*vh[n,k-1]).sum(axis=-1)
  return psi

def plotPsi(y, z, psi, ci, title='', zval=[0.,-2000.,-6500.]):
  """
  General function to plot the meridional overturning streamfunction, in Sv.


  Parameters
  ----------
  y : 2D numpy array
    Scalar 2D array specifying the latitudes to be plotted.

  z : 2D numpy array
    Scalar 2D array specifying the elevations (depth) to be plotted.

  psi : 2D numpy array
    Meridional overturning streamfunction.

  ci : 1D numpy array
     Contour interval, draw contour lines at the these levels.

  title : str, optional
     The title to place at the top of panel. Default ''.

  zval : 1D numpy array, optional
     Array with 3 values used to set the split vertical scale. Default '[0.,-2000.,-6500.]'.

  Returns
  -------
  """
  cmap = plt.get_cmap('dunnePM')
  plt.contourf(y, z, psi, levels=ci, cmap=cmap, extend='both')
  cbar = plt.colorbar()
  plt.contour(y, z, psi, levels=ci, colors='k')
  plt.gca().set_yscale('splitscale',zval=zval)
  plt.title(title)
  cbar.set_label('[Sv]'); plt.ylabel('Elevation [m]')

def findExtrema(y, z, psi, min_lat=-90., max_lat=90., min_depth=0., mult=1., plot = True):
  psiMax = mult*numpy.amax( mult * numpy.ma.array(psi)[(y>=min_lat) & (y<=max_lat) & (z<-min_depth)] )
  idx = numpy.argmin(numpy.abs(psi-psiMax))
  (j,i) = numpy.unravel_index(idx, psi.shape)
  if plot:
    #plt.plot(y[j,i],z[j,i],'kx',hold=True)
    plt.plot(y[j,i],z[j,i],'kx')
    plt.text(y[j,i],z[j,i],'%.1f'%(psi[j,i]))
  else:
    return psi[j,i]
if __name__ == '__main__':
  main()
