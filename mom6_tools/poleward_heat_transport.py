#!/usr/bin/env python

import io, yaml, os
import matplotlib.pyplot as plt
import warnings, dask, numpy, netCDF4
from datetime import datetime, date
import xarray as xr
from mom6_tools.DiagsCase import DiagsCase
from ncar_jobqueue import NCARCluster
from dask.distributed import Client
from mom6_tools import m6plot
from mom6_tools  import m6toolbox
from mom6_tools.MOM6grid import MOM6grid

def options():
  try: import argparse
  except: raise Exception('This version of python is not new enough. python 2.7 or newer is required.')
  parser = argparse.ArgumentParser(description='''Script for plotting poleward heat transport.''')
  parser.add_argument('diag_config_yml_path', type=str, help='''Full path to the yaml file  \
    describing the run and diagnostics to be performed.''')
  parser.add_argument('-v', '--variables', nargs='+', default=['T_ady_2d', 'T_diffy_2d'],
                     help='''Variables to be processed (default=['T_ady_2d', 'T_diffy_2d'])''')
  parser.add_argument('-sd','--start_date',  type=str, default='0001-01-01',
                     help='''Start year to plot (default=0001-01-01)''')
  parser.add_argument('-ed','--end_date',   type=str, default='0100-12-31',
                      help='''Final year to plot (default=0100-12-31)''')
  parser.add_argument('-nw','--number_of_workers',  type=int, default=2,
                      help='''Number of workers to use (default=2).''')
  parser.add_argument('-debug',   help='''Add priting statements for debugging purposes''',
                      action="store_true")

  cmdLineArgs = parser.parse_args()
  return cmdLineArgs

def main(stream=False):
  # Get options
  args = options()
  nw = args.number_of_workers
  if not os.path.isdir('PNG'):
    print('Creating a directory to place figures (PNG)... \n')
    os.system('mkdir PNG')
  if not os.path.isdir('ncfiles'):
    print('Creating a directory to place figures (ncfiles)... \n')
    os.system('mkdir ncfiles')

  # Read in the yaml file
  diag_config_yml = yaml.load(open(args.diag_config_yml_path,'r'), Loader=yaml.Loader)

  # Create the case instance
  dcase = DiagsCase(diag_config_yml['Case'])
  args.case_name = dcase.casename
  args.savefigs = True; args.outdir = 'PNG'
  RUNDIR = dcase.get_value('RUNDIR')
  print('Run directory is:', RUNDIR)
  print('Casename is:', dcase.casename)
  print('Variables to be processed:', args.variables)
  print('Number of workers to be used:', nw)

  # read grid info
  grd = MOM6grid(RUNDIR+'/'+dcase.casename+'.mom6.static.nc')
  depth = grd.depth_ocean
  # remote Nan's, otherwise genBasinMasks won't work
  depth[numpy.isnan(depth)] = 0.0
  basin_code = m6toolbox.genBasinMasks(grd.geolon, grd.geolat, depth)
  parallel, cluster, client = m6toolbox.request_workers(nw)
  print('Reading surface dataset...')
  startTime = datetime.now()
  variables = args.variables
  def preprocess(ds):
    ''' Compute montly averages and return the dataset with variables'''
    return ds[variables].resample(time="1Y", closed='left', \
           keep_attrs=True).mean(dim='time', keep_attrs=True)

  if parallel:
    ds = xr.open_mfdataset(RUNDIR+'/'+dcase.casename+'.mom6.hm_*.nc', \
         parallel=True, data_vars='minimal', chunks={'time': 12},\
         coords='minimal', compat='override', preprocess=preprocess)
  else:
    ds = xr.open_mfdataset(RUNDIR+'/'+dcase.casename+'.mom6.hm_*.nc', \
         data_vars='minimal', coords='minimal', compat='override', \
         preprocess=preprocess)
  print('Time elasped: ', datetime.now() - startTime)

  print('\n Selecting data between {} and {}...'.format(args.start_date, args.end_date))
  startTime = datetime.now()
  ds_sel = ds.sel(time=slice(args.start_date, args.end_date))
  print('Time elasped: ', datetime.now() - startTime)

  print('\n Computing time mean...')
  startTime = datetime.now()
  ds_sel = ds_sel.mean('time').load()
  print('Time elasped: ', datetime.now() - startTime)

  if parallel:
    print('\n Releasing workers...')
    client.close(); cluster.close()

  print('Saving netCDF files...')
  ds_sel.to_netcdf('ncfiles/'+dcase.casename+'_heat_transport.nc')
  # create a ndarray subclass
  class C(numpy.ndarray): pass

  varName = 'T_ady_2d'
  if varName in ds.variables:
    tmp = numpy.ma.masked_invalid(ds_sel[varName].values)
    tmp = tmp[:].filled(0.)
    advective = tmp.view(C)
    advective.units = ds[varName].units
  else:
    raise Exception('Could not find "T_ady_2d" in file "%s"'%(args.infile+args.monthly))

  varName = 'T_diffy_2d'
  if varName in ds.variables:
    tmp = numpy.ma.masked_invalid(ds_sel[varName].values)
    tmp = tmp[:].filled(0.)
    diffusive = tmp.view(C)
    diffusive.units = ds[varName].units
  else:
    diffusive = None
    warnings.warn('Diffusive temperature term not found. This will result in an underestimation of the heat transport.')

  varName = 'T_lbm_diffy'
  if varName in ds.variables:
    tmp = numpy.ma.masked_invalid(ds_sel[varName].sum('z_l').values)
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

  if args.case_name != '':  suptitle = args.case_name
  else: suptitle = ''

  # Global Heat Transport
  plt.figure(figsize=(12,10))
  HTplot = heatTrans(advective,diffusive)
  yy = grd.geolat_c[:,:].max(axis=-1)
  ave_title = ', averaged from {} to {}'.format(args.start_date, args.end_date)
  plotHeatTrans(yy,HTplot,title='Global Y-Direction Heat Transport [PW]'+ave_title)
  plt.plot(yobs,NCEP['Global'],'k--',linewidth=0.5,label='NCEP')
  plt.plot(yobs,ECMWF['Global'],'k.',linewidth=0.5,label='ECMWF')
  plotGandW(GandW['Global']['lat'],GandW['Global']['trans'],GandW['Global']['err'])
  plt.xlabel(r'Latitude [$\degree$N]',fontsize=10)
  plt.suptitle(suptitle)
  plt.legend(loc=0,fontsize=10)
  annotateObs()
  if diffusive is None: annotatePlot('Warning: Diffusive component of transport is missing.')
  if args.savefigs:
    objOut = args.outdir+'/'+args.case_name+'_HeatTransport_global.png'
    plt.savefig(objOut); plt.close()
  else:
    plt.show()
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
  if args.savefigs:
    objOut = args.outdir+'/'+args.case_name+'_HeatTransport_Atlantic.png'
    plt.savefig(objOut); plt.close()
  else:
    plt.show()
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
  if args.savefigs:
    objOut = args.outdir+'/'+args.case_name+'_HeatTransport_IndoPacific.png'
    plt.savefig(objOut); plt.close()
  else:
    plt.show()
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

def plotHeatTrans(y, HT, title, xlim=(-80,90), ylim=(-2.5,3.0)):
  plt.plot(y, y*0., 'k', linewidth=0.5)
  plt.plot(y, HT, 'r', linewidth=1.5,label='Model')
  plt.xlim(xlim); plt.ylim(ylim)
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

