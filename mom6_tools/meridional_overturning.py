#!/usr/bin/env python

import io
import numpy
import m6plot
import m6toolbox
import matplotlib.pyplot as plt
import os
import sys
import xarray as xr

from .latlon_analysis import check_time_interval
from .MOM6grid import MOM6grid

def options():
  try: import argparse
  except: raise Exception('This version of python is not new enough. python 2.7 or newer is required.')
  parser = argparse.ArgumentParser(description='''Script for plotting meridional overturning.''')
#  parser.add_argument('-s','--case_name', type=str, default='', help='''Super-title for experiment.  Default is to read from netCDF file.''')
#  parser.add_argument('-o','--outdir', type=str, default='.', help='''Directory in which to place plots.''')
#  parser.add_argument('-g','--gridspec', type=str, required=True,
#    help='''Directory containing mosaic/grid-spec files (ocean_hgrid.nc and ocean_mask.nc).''')
  parser.add_argument('infile', type=str, help='''Path to the file(s) to be processed (i.e <some_path>/run/)''')
  parser.add_argument('-l','--label', type=str, default='', help='''Label to add to the plot.''')
  parser.add_argument('-n','--case_name', type=str, default='', help='''Case name.  Default is to read from netCDF file.''')
  parser.add_argument('-m','--monthly', type=str, default='.', help='''Monthly-averaged file containing 3D 'uh' and 'vh' ''')
  parser.add_argument('-o','--outdir', type=str, default='.', help='''Directory in which to place plots.''')
  parser.add_argument('-s','--static', type=str, required=True, help='''Name of the MOM6 static file.''')
  parser.add_argument('-savefigs', help='''Save figures in a PNG format.''', action="store_true")
  parser.add_argument('-year_start', type=int, default=80, help='''Start year to compute averages. Default is 80.''')
  parser.add_argument('-year_end', type=int, default=90, help='''End year to compute averages. Default is 100.''')

  cmdLineArgs = parser.parse_args()
  return cmdLineArgs
  main(cmdLineArgs)

def main(stream=False):
  # Get options
  args = options()
  # mom6 grid
  grd = MOM6grid(args.infile+args.static)
  depth = grd.depth_ocean
  # remote Nan's, otherwise genBasinMasks won't work
  depth[numpy.isnan(depth)] = 0.0
  basin_code = m6toolbox.genBasinMasks(grd.geolon, grd.geolat, depth)

  # load data
  ds = xr.open_mfdataset(args.infile+args.monthly,decode_times=False)
  # convert time in years
  ds['time'] = ds.time/365.
  ti = args.year_start
  tf = args.year_end
  # check if data includes years between ti and tf
  check_time_interval(ti,tf,ds)

  # create a ndarray subclass
  class C(numpy.ndarray): pass

  if 'vmo' in ds.variables:
    varName = 'vmo'; conversion_factor = 1.e-9
  elif 'vh' in ds.variables:
    varName = 'vh'; conversion_factor = 1.e-6
    if 'zw' in ds.variables: conversion_factor = 1.e-9 # Backwards compatible for when we had wrong units for 'vh'
  else: raise Exception('Could not find "vh" or "vmo" in file "%s"'%(args.infile+args.static))


  tmp = numpy.ma.masked_invalid(ds[varName].sel(time=slice(ti,tf)).mean('time').data)
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
  z = Zmod.min(axis=-1); psiPlot = MOCpsi(VHmod)*conversion_factor
  #yy = y[1:,:].max(axis=-1)+0*z
  yy = grd.geolat_c[:,:].max(axis=-1)+0*z
  print(z.shape, yy.shape, psiPlot.shape)
  ci=m6plot.pmCI(0.,40.,5.)
  plotPsi(yy, z, psiPlot[1::,:], ci, 'Global MOC [Sv]')
  plt.xlabel(r'Latitude [$\degree$N]')
  plt.suptitle(case_name)
  findExtrema(yy, z, psiPlot, max_lat=-30.)
  findExtrema(yy, z, psiPlot, min_lat=25.)
  findExtrema(yy, z, psiPlot, min_depth=2000., mult=-1.)
  if stream is True: objOut = io.BytesIO()
  else: objOut = args.outdir+'/MOC_global.png'
  plt.savefig(objOut)
  if stream is True: imgbufs.append(objOut)

  # Atlantic MOC
  m6plot.setFigureSize([16,9],576,debug=False)
  cmap = plt.get_cmap('dunnePM')
  m = 0*basin_code; m[(basin_code==2) | (basin_code==4) | (basin_code==6) | (basin_code==7) | (basin_code==8)]=1
  ci=m6plot.pmCI(0.,22.,2.)
  z = (m*Zmod).min(axis=-1); psiPlot = MOCpsi(VHmod, vmsk=m*numpy.roll(m,-1,axis=-2))*conversion_factor
  #yy = y[1:,:].max(axis=-1)+0*z
  yy = grd.geolat_c[:,:].max(axis=-1)+0*z
  plotPsi(yy, z, psiPlot[1::,:], ci, 'Atlantic MOC [Sv]')
  plt.xlabel(r'Latitude [$\degree$N]')
  plt.suptitle(case_name)
  findExtrema(yy, z, psiPlot, min_lat=26.5, max_lat=27.) # RAPID
  findExtrema(yy, z, psiPlot, max_lat=-33.)
  findExtrema(yy, z, psiPlot)
  findExtrema(yy, z, psiPlot, min_lat=5.)
  if stream is True: objOut = io.BytesIO()
  else: objOut = args.outdir+'/MOC_Atlantic.png'
  plt.savefig(objOut,format='png')
  if stream is True: imgbufs.append(objOut)

  if stream is True:
    return imgbufs

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

def plotPsi(y, z, psi, ci, title):
  cmap = plt.get_cmap('dunnePM')
  plt.contourf(y, z, psi, levels=ci, cmap=cmap, extend='both')
  cbar = plt.colorbar()
  plt.contour(y, z, psi, levels=ci, colors='k', hold='on')
  plt.gca().set_yscale('splitscale',zval=[0.,-2000.,-6500.])
  plt.title(title)
  cbar.set_label('[Sv]'); plt.ylabel('Elevation [m]')

def findExtrema(y, z, psi, min_lat=-90., max_lat=90., min_depth=0., mult=1.):
  psiMax = mult*numpy.amax( mult * numpy.ma.array(psi)[(y>=min_lat) & (y<=max_lat) & (z<-min_depth)] )
  idx = numpy.argmin(numpy.abs(psi-psiMax))
  (j,i) = numpy.unravel_index(idx, psi.shape)
  plt.plot(y[j,i],z[j,i],'kx',hold=True)
  plt.text(y[j,i],z[j,i],'%.1f'%(psi[j,i]))

if __name__ == '__main__':
  main()
