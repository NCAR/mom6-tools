#!/usr/bin/env python

import io
import netCDF4
import numpy
import matplotlib.pyplot as plt
import os
import sys
import warnings
import xarray as xr

from mom6_tools import m6plot
from mom6_tools  import m6toolbox
from mom6_tools.MOM6grid import MOM6grid

def options():
  try: import argparse
  except: raise Exception('This version of python is not new enough. python 2.7 or newer is required.')
  parser = argparse.ArgumentParser(description='''Script for plotting plotting poleward heat transport.''')
  parser.add_argument('infile', type=str, help='''Path to the file(s) to be processed (i.e <some_path>/run/)''')
  parser.add_argument('-l','--label', type=str, default='', help='''Label to add to the plot.''')
  parser.add_argument('-n','--case_name', type=str, default='', help='''Case name.  Default is to read from netCDF file.''')
  parser.add_argument('-o','--outdir', type=str, default='.', help='''Directory in which to place plots.''')
  parser.add_argument('-m','--monthly', type=str, default='.', help='''Monthly-averaged file containing 3D 'T_ady_2d' and 'T_diffy_2d''')
  parser.add_argument('-s','--static', type=str, required=True, help='''Name of the MOM6 static file.''')
  parser.add_argument('-savefigs', help='''Save figures in a PNG format.''', action="store_true")
  parser.add_argument('-year_start', type=int, default=80, help='''Start year to compute averages. Default is 80.''')
  parser.add_argument('-year_end', type=int, default=100, help='''End year to compute averages. Default is 100.''')

  cmdLineArgs = parser.parse_args()
  return cmdLineArgs

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
  m6toolbox.check_time_interval(ti,tf,ds)

  # create a ndarray subclass
  class C(numpy.ndarray): pass

  varName = 'T_ady_2d'
  if varName in ds.variables:
    tmp = numpy.ma.masked_invalid(ds[varName].sel(time=slice(ti,tf)).mean('time').data)
    tmp = tmp[:].filled(0.)
    advective = tmp.view(C)
    advective.units = ds[varName].units
  else:
    raise Exception('Could not find "T_ady_2d" in file "%s"'%(args.infile+args.monthly))

  varName = 'T_diffy_2d'
  if varName in ds.variables:
    tmp = numpy.ma.masked_invalid(ds[varName].sel(time=slice(ti,tf)).mean('time').data)
    tmp = tmp[:].filled(0.)
    diffusive = tmp.view(C)
    diffusive.units = ds[varName].units
  else:
    diffusive = None
    warnings.warn('Diffusive temperature term not found. This will result in an underestimation of the heat transport.')

  varName = 'T_lbm_diffy'
  if varName in ds.variables:
    tmp = numpy.ma.masked_invalid(ds[varName].sel(time=slice(ti,tf)).sum('z_l').mean('time').data)
    tmp = tmp[:].filled(0.)
    diffusive = diffusive + tmp.view(C)
  else:
    warnings.warn('Lateral boundary mixing term not found. This will result in an underestimation of the heat transport.')

  plt_heat_transport_model_vs_obs(advective, diffusive, basin_code, grd, args)
  return

def plt_heat_transport_model_vs_obs(advective, diffusive, basin_code, grd, args):
  """Plots model vs obs poleward heat transport for the global, Pacific and Atlantic basins"""
  # Load Observations
  fObs = netCDF4.Dataset('/glade/work/gmarques/cesm/datasets/Trenberth_and_Caron_Heat_Transport.nc')
  #Trenberth and Caron
  yobs = fObs.variables['ylat'][:]
  NCEP = {}; NCEP['Global'] = fObs.variables['OTn']
  NCEP['Atlantic'] = fObs.variables['ATLn'][:]; NCEP['IndoPac'] = fObs.variables['INDPACn'][:]
  ECMWF = {}; ECMWF['Global'] = fObs.variables['OTe'][:]
  ECMWF['Atlantic'] = fObs.variables['ATLe'][:]; ECMWF['IndoPac'] = fObs.variables['INDPACe'][:]

  #G and W
  Global = {}
  Global['lat'] = numpy.array([-30., -19., 24., 47.])
  Global['trans'] = numpy.array([-0.6, -0.8, 1.8, 0.6])
  Global['err'] = numpy.array([0.3, 0.6, 0.3, 0.1])

  Atlantic = {}
  Atlantic['lat'] = numpy.array([-45., -30., -19., -11., -4.5, 7.5, 24., 47.])
  Atlantic['trans'] = numpy.array([0.66, 0.35, 0.77, 0.9, 1., 1.26, 1.27, 0.6])
  Atlantic['err'] = numpy.array([0.12, 0.15, 0.2, 0.4, 0.55, 0.31, 0.15, 0.09])

  IndoPac = {}
  IndoPac['lat'] = numpy.array([-30., -18., 24., 47.])
  IndoPac['trans'] = numpy.array([-0.9, -1.6, 0.52, 0.])
  IndoPac['err'] = numpy.array([0.3, 0.6, 0.2, 0.05,])

  GandW = {}
  GandW['Global'] = Global
  GandW['Atlantic'] = Atlantic
  GandW['IndoPac'] = IndoPac

  if args.case_name != '':  suptitle = args.case_name + ' ' + args.label
  else: suptitle = args.label

  # Global Heat Transport
  plt.figure(figsize=(12,10))
  HTplot = heatTrans(advective,diffusive)
  yy = grd.geolat_c[:,:].max(axis=-1)
  ave_title = ', averaged over years {}-{}'.format(args.year_start, args.year_end)
  plotHeatTrans(yy,HTplot,title='Global Y-Direction Heat Transport [PW]'+ave_title)
  plt.plot(yobs,NCEP['Global'],'k--',linewidth=0.5,label='NCEP')
  plt.plot(yobs,ECMWF['Global'],'k.',linewidth=0.5,label='ECMWF')
  plotGandW(GandW['Global']['lat'],GandW['Global']['trans'],GandW['Global']['err'])
  plt.xlabel(r'Latitude [$\degree$N]',fontsize=10)
  plt.suptitle(suptitle)
  plt.legend(loc=0,fontsize=10)
  annotateObs()
  if diffusive is None: annotatePlot('Warning: Diffusive component of transport is missing.')
  plt.show()
  if args.savefigs:
    objOut = args.outdir+'/'+args.case_name+'_HeatTransport_global.png'
    plt.savefig(objOut)

  # Atlantic Heat Transport
  m = 0*basin_code; m[(basin_code==2) | (basin_code==4) | (basin_code==6) | (basin_code==7) | (basin_code==8)] = 1
  plt.figure(figsize=(12,10))
  HTplot = heatTrans(advective, diffusive, vmask=m*numpy.roll(m,-1,axis=-2))
  yy = grd.geolat_c[:,:].max(axis=-1)
  HTplot[yy<-34] = numpy.nan
  plotHeatTrans(yy,HTplot,title='Atlantic Y-Direction Heat Transport [PW]'+ave_title)
  plt.plot(yobs,NCEP['Atlantic'],'k--',linewidth=0.5,label='NCEP')
  plt.plot(yobs,ECMWF['Atlantic'],'k.',linewidth=0.5,label='ECMWF')
  plotGandW(GandW['Atlantic']['lat'],GandW['Atlantic']['trans'],GandW['Atlantic']['err'])
  plt.xlabel(r'Latitude [$\degree$N]',fontsize=10)
  plt.suptitle(suptitle)
  plt.legend(loc=0,fontsize=10)
  annotateObs()
  if diffusive is None: annotatePlot('Warning: Diffusive component of transport is missing.')
  plt.show()
  if args.savefigs:
    objOut = args.outdir+'/'+args.case_name+'_HeatTransport_global.png'
    plt.savefig(objOut)

  # Indo-Pacific Heat Transport
  m = 0*basin_code; m[(basin_code==3) | (basin_code==5)] = 1
  plt.figure(figsize=(12,10))
  HTplot = heatTrans(advective, diffusive, vmask=m*numpy.roll(m,-1,axis=-2))
  yy = grd.geolat_c[:,:].max(axis=-1)
  HTplot[yy<-34] = numpy.nan
  plotHeatTrans(yy,HTplot,title='Indo-Pacific Y-Direction Heat Transport [PW]'+ave_title)
  plt.plot(yobs,NCEP['IndoPac'],'k--',linewidth=0.5,label='NCEP')
  plt.plot(yobs,ECMWF['IndoPac'],'k.',linewidth=0.5,label='ECMWF')
  plotGandW(GandW['IndoPac']['lat'],GandW['IndoPac']['trans'],GandW['IndoPac']['err'])
  plt.xlabel(r'Latitude [$\degree$N]',fontsize=10)
  annotateObs()
  if diffusive is None: annotatePlot('Warning: Diffusive component of transport is missing.')
  plt.suptitle(suptitle)
  plt.legend(loc=0,fontsize=10)
  plt.show()
  if args.savefigs:
    objOut = args.outdir+'/'+args.case_name+'_HeatTransport_global.png'
    plt.savefig(objOut)

  return

def heatTrans(advective, diffusive=None, vmask=None):
  """Converts vertically integrated temperature advection into heat transport"""
  if diffusive is not None:
    HT = advective[:] + diffusive[:]
  else:
    HT = advective[:]
  if len(HT.shape) == 3:
    HT = HT.mean(axis=0)
  if advective.units == "Celsius meter3 second-1":
    rho0 = 1.035e3
    Cp = 3992.
    HT = HT * (rho0 * Cp)
    HT = HT * 1.e-15  # convert to PW
  elif advective.units == "W":
    HT = HT * 1.e-15
  else:
    print('Unknown units')
  if vmask is not None: HT = HT*vmask
  HT = HT.sum(axis=-1); HT = HT.squeeze() # sum in x-direction
  return HT

def plotHeatTrans(y, HT, title, xlim=(-80,90)):
  plt.plot(y, y*0., 'k', linewidth=0.5)
  plt.plot(y, HT, 'r', linewidth=1.5,label='Model')
  plt.xlim(xlim); plt.ylim(-2.5,3.0)
  plt.title(title)
  plt.grid(True)

def annotatePlot(label):
  fig = plt.gcf()
  #fig.text(0.1,0.85,label)
  fig.text(0.535,0.12,label)

def annotateObs():
  fig = plt.gcf()
  fig.text(0.13,0.85,r"Trenberth, K. E. and J. M. Caron, 2001: Estimates of Meridional Atmosphere and Ocean Heat Transports. J.Climate, 14, 3433-3443.", fontsize=8)
  fig.text(0.13,0.825,r"Ganachaud, A. and C. Wunsch, 2000: Improved estimates of global ocean circulation, heat transport and mixing from hydrographic data.", fontsize=8)
  fig.text(0.13,0.8,r"Nature, 408, 453-457", fontsize=8)

def plotGandW(lat,trans,err):
  low = trans - err
  high = trans + err
  for n in range(0,len(low)):
    if n == 0:
      plt.plot([lat[n],lat[n]], [low[n],high[n]], 'c', linewidth=2.0, label='G&W')
    else:
      plt.plot([lat[n],lat[n]], [low[n],high[n]], 'c', linewidth=2.0)
  plt.scatter(lat,trans,marker='s',facecolor='cyan')

if __name__ == '__main__':
  main()
