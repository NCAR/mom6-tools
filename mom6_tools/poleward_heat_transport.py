#!/usr/bin/env python

import io, yaml, os
import matplotlib.pyplot as plt
import numpy as np
import warnings, dask, netCDF4
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
  parser.add_argument('-v', '--variables', nargs='+', default=['T_ady_2d', 'T_diffy_2d', 'T_lbd_diffy_2d'],
                     help='''Variables to be processed (default=['T_ady_2d', 'T_diffy_2d', 'T_lbd_diffy_2d'])''')
  parser.add_argument('-sd','--start_date', type=str, default='',
                      help='''Start year to compute averages. Default is to use value set in diag_config_yml_path''')
  parser.add_argument('-ed','--end_date', type=str, default='',
                      help='''End year to compute averages. Default is to use value set in diag_config_yml_path''')
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
  if not os.path.isdir('PNG/HT'):
    print('Creating a directory to place figures (PNG/HT)... \n')
    os.system('mkdir -p PNG/HT')
  if not os.path.isdir('ncfiles'):
    print('Creating a directory to place figures (ncfiles)... \n')
    os.system('mkdir ncfiles')

  # Read in the yaml file
  diag_config_yml = yaml.load(open(args.diag_config_yml_path,'r'), Loader=yaml.Loader)

  # Create the case instance
  dcase = DiagsCase(diag_config_yml['Case'])
  args.case_name = dcase.casename
  DOUT_S = dcase.get_value('DOUT_S')
  if DOUT_S:
    OUTDIR = dcase.get_value('DOUT_S_ROOT')+'/ocn/hist/'
  else:
    OUTDIR = dcase.get_value('RUNDIR')

  args.savefigs = True; args.outdir = 'PNG/HT'
  print('Output directory is:', OUTDIR)
  print('Casename is:', dcase.casename)
  print('Variables to be processed:', args.variables)
  print('Number of workers to be used:', nw)

  # set avg dates
  avg = diag_config_yml['Avg']
  if not args.start_date : args.start_date = avg['start_date']
  if not args.end_date : args.end_date = avg['end_date']

  # read grid info
  grd = MOM6grid(OUTDIR+'/'+dcase.casename+'.mom6.static.nc')
  depth = grd.depth_ocean
  # remote Nan's, otherwise genBasinMasks won't work
  depth[np.isnan(depth)] = 0.0
  basin_code = m6toolbox.genBasinMasks(grd.geolon, grd.geolat, depth)

  parallel = False
  if nw > 1:
    parallel = True
    cluster = NCARCluster()
    cluster.scale(nw)
    client = Client(cluster)

  print('Reading dataset...')
  startTime = datetime.now()
  variables = args.variables

  def preprocess(ds):
    ''' Compute montly averages and return the dataset with variables'''
    for var in variables:
      print('Processing {}'.format(var))
      if var not in ds.variables:
        print('WARNING: ds does not have variable {}. Creating dataarray with zeros'.format(var))
        jm, im = grd.geolat.shape
        tm = len(ds.time)
        da = xr.DataArray(np.zeros((tm, jm, im)), dims=['time','yq','xh'], \
             coords={'yq' : grd.yq, 'xh' : grd.xh, 'time' : ds.time}).rename(var)
        ds = xr.merge([ds, da])
    return ds[variables]

  ds1 = xr.open_mfdataset(OUTDIR+'/'+dcase.casename+'.mom6.hm_*.nc', parallel=parallel)

  # use datetime
  #ds1['time'] = ds1.indexes['time'].to_datetimeindex()

  ds = preprocess(ds1)

  print('Time elasped: ', datetime.now() - startTime)

  print('Selecting data between {} and {}...'.format(args.start_date, args.end_date))
  startTime = datetime.now()
  ds_sel = ds.sel(time=slice(args.start_date, args.end_date))
  print('Time elasped: ', datetime.now() - startTime)

  print('Computing yearly means...')
  startTime = datetime.now()
  ds_sel = ds_sel.resample(time="1Y", closed='left',keep_attrs=True).mean('time',keep_attrs=True)
  print('Time elasped: ', datetime.now() - startTime)

  print('Computing time mean...')
  startTime = datetime.now()
  ds_sel = ds_sel.mean('time').load()
  print('Time elasped: ', datetime.now() - startTime)

  if parallel:
    print('Releasing workers...')
    client.close(); cluster.close()

  varName = 'T_ady_2d'
  print('Saving netCDF files...')
  attrs = {'description': 'Time-mean poleward heat transport by components ', 'units': ds[varName].units,
       'start_date': args.start_date, 'end_date': args.end_date, 'casename': dcase.casename}
  m6toolbox.add_global_attrs(ds_sel,attrs)

  ds_sel.to_netcdf('ncfiles/'+dcase.casename+'_heat_transport.nc')
  # create a ndarray subclass
  class C(np.ndarray): pass

  if varName in ds.variables:
    tmp = np.ma.masked_invalid(ds_sel[varName].values)
    tmp = tmp[:].filled(0.)
    advective = tmp.view(C)
    advective.units = ds[varName].units
  else:
    raise Exception('Could not find "T_ady_2d" in file "%s"'%(args.infile+args.monthly))

  varName = 'T_diffy_2d'
  if varName in ds.variables:
    tmp = np.ma.masked_invalid(ds_sel[varName].values)
    tmp = tmp[:].filled(0.)
    diffusive = tmp.view(C)
    diffusive.units = ds[varName].units
  else:
    diffusive = None
    warnings.warn('Diffusive temperature term not found. This will result in an underestimation of the heat transport.')

  varName = 'T_hbd_diffy_2d'
  if varName in ds.variables:
    tmp = np.ma.masked_invalid(ds_sel[varName].values)
    tmp = tmp[:].filled(0.)
    hbd = tmp.view(C)
    #hbd.units = ds[varName].units
  else:
    hbd = None
    warnings.warn('Quasi-horizontal boundary mixing term not found. This will result in an underestimation of the heat transport.')

  plt_heat_transport_model_vs_obs(advective, diffusive, hbd, basin_code, grd, args)
  return

def plt_heat_transport_model_vs_obs(advective, diffusive, hbd, basin_code, grd, args):
  """Plots model vs obs poleward heat transport for the global, Pacific and Atlantic basins"""
  # Load Observations
  fObs = netCDF4.Dataset('/glade/work/gmarques/cesm/datasets/Trenberth_and_Caron_Heat_Transport.nc')
  # POP JRA-55, 31 year (years 29-59)
  pop = xr.open_dataset('/glade/u/home/bryan/MOM6-modeloutputanalysis/MHT_mean.g210.GIAF_JRA.v13.gx1v7.01.nc')
  # Estimate based on the JRA-55 v1.3 forcing (Tsujino et al, 2019)
  # basin = 0 is Global; basin = 1 is Atlantic and basin = 2 is IndoPacific
  jra = xr.open_dataset('/glade/work/gmarques/cesm/datasets/Heat_transport/jra55fcst_v1_3_annual_1x1/nht_jra55do_v1_3.nc')
  #Trenberth and Caron
  yobs = fObs.variables['ylat'][:]
  NCEP = {}; NCEP['Global'] = fObs.variables['OTn']
  NCEP['Atlantic'] = fObs.variables['ATLn'][:]; NCEP['IndoPac'] = fObs.variables['INDPACn'][:]
  ECMWF = {}; ECMWF['Global'] = fObs.variables['OTe'][:]
  ECMWF['Atlantic'] = fObs.variables['ATLe'][:]; ECMWF['IndoPac'] = fObs.variables['INDPACe'][:]

  #G and W
  Global = {}
  Global['lat'] = np.array([-30., -19., 24., 47.])
  Global['trans'] = np.array([-0.6, -0.8, 1.8, 0.6])
  Global['err'] = np.array([0.3, 0.6, 0.3, 0.1])

  Atlantic = {}
  Atlantic['lat'] = np.array([-45., -30., -19., -11., -4.5, 7.5, 24., 47.])
  Atlantic['trans'] = np.array([0.66, 0.35, 0.77, 0.9, 1., 1.26, 1.27, 0.6])
  Atlantic['err'] = np.array([0.12, 0.15, 0.2, 0.4, 0.55, 0.31, 0.15, 0.09])

  IndoPac = {}
  IndoPac['lat'] = np.array([-30., -18., 24., 47.])
  IndoPac['trans'] = np.array([-0.9, -1.6, 0.52, 0.])
  IndoPac['err'] = np.array([0.3, 0.6, 0.2, 0.05,])

  GandW = {}
  GandW['Global'] = Global
  GandW['Atlantic'] = Atlantic
  GandW['IndoPac'] = IndoPac

  if args.case_name != '':  suptitle = args.case_name
  else: suptitle = ''

  # Global Heat Transport
  plt.figure(figsize=(12,10))
  HTplot = heatTrans(advective,diffusive,hbd)
  yy = grd.geolat_c[:,:].max(axis=-1)
  ave_title = ', averaged from {} to {}'.format(args.start_date, args.end_date)
  plotHeatTrans(yy,HTplot,title='Global Y-Direction Heat Transport [PW]'+ave_title)
  plt.plot(pop.lat_aux_grid.values,pop.MHT_global.values,'orange',linewidth=1,label='POP')
  jra_mean_global = jra.nht[:,0,:].mean('time').values
  jra_std_global = jra.nht[:,0,:].std('time').values
  plt.plot(jra.lat, jra_mean_global,'k', label='JRA-55 v1.3', color='#1B2ACC', lw=1)
  plt.fill_between(jra.lat, jra_mean_global-jra_std_global, jra_mean_global+jra_std_global,
    alpha=0.25, edgecolor='#1B2ACC', facecolor='#089FFF')
  plt.plot(yobs,NCEP['Global'],'k--',linewidth=0.5,label='NCEP')
  plt.plot(yobs,ECMWF['Global'],'k.',linewidth=0.5,label='ECMWF')
  plotGandW(GandW['Global']['lat'],GandW['Global']['trans'],GandW['Global']['err'])
  plt.xlabel(r'Latitude [$\degree$N]',fontsize=10)
  plt.suptitle(suptitle)
  plt.legend(loc=0,fontsize=10)
  annotateObs()
  if diffusive is None: annotatePlot('Warning: Diffusive component of transport is missing.')
  if hbd is None: annotatePlot('Warning: LBD component of transport is missing.')

  if args.savefigs:
    objOut = args.outdir+'/'+args.case_name+'_HeatTransport_global.png'
    plt.savefig(objOut); plt.close()
  else:
    plt.show()
  # Atlantic Heat Transport
  m = 0*basin_code; m[(basin_code==2) | (basin_code==4) | (basin_code==6) | (basin_code==7) | (basin_code==8)] = 1
  plt.figure(figsize=(12,10))
  HTplot = heatTrans(advective, diffusive, hbd, vmask=m*np.roll(m,-1,axis=-2))
  yy = grd.geolat_c[:,:].max(axis=-1)
  HTplot[yy<-34] = np.nan
  plotHeatTrans(yy,HTplot,title='Atlantic Y-Direction Heat Transport [PW]'+ave_title)
  plt.plot(pop.lat_aux_grid.values,pop.MHT_atl.values,'orange',linewidth=1,label='POP')
  jra_mean_atl = jra.nht[:,1,:].mean('time').values
  jra_std_atl = jra.nht[:,1,:].std('time').values
  plt.plot(jra.lat, jra_mean_atl,'k', label='JRA-55 v1.3', color='#1B2ACC', lw=1)
  plt.fill_between(jra.lat, jra_mean_atl-jra_std_atl, jra_mean_atl+jra_std_atl,
    alpha=0.25, edgecolor='#1B2ACC', facecolor='#089FFF')
  plt.plot(yobs,NCEP['Atlantic'],'k--',linewidth=0.5,label='NCEP')
  plt.plot(yobs,ECMWF['Atlantic'],'k.',linewidth=0.5,label='ECMWF')
  plotGandW(GandW['Atlantic']['lat'],GandW['Atlantic']['trans'],GandW['Atlantic']['err'])
  plt.xlabel(r'Latitude [$\degree$N]',fontsize=10)
  plt.suptitle(suptitle)
  plt.legend(loc=0,fontsize=10)
  annotateObs()
  if diffusive is None: annotatePlot('Warning: Diffusive component of transport is missing.')
  if hbd is None: annotatePlot('Warning: LBD component of transport is missing.')
  if args.savefigs:
    objOut = args.outdir+'/'+args.case_name+'_HeatTransport_Atlantic.png'
    plt.savefig(objOut); plt.close()
  else:
    plt.show()
  # Indo-Pacific Heat Transport
  m = 0*basin_code; m[(basin_code==3) | (basin_code==5)] = 1
  plt.figure(figsize=(12,10))
  HTplot = heatTrans(advective, diffusive, hbd, vmask=m*np.roll(m,-1,axis=-2))
  yy = grd.geolat_c[:,:].max(axis=-1)
  HTplot[yy<-34] = np.nan
  plotHeatTrans(yy,HTplot,title='Indo-Pacific Y-Direction Heat Transport [PW]'+ave_title)
  plt.plot(pop.lat_aux_grid.values,(pop.MHT_global-pop.MHT_atl).values,'orange',linewidth=1,label='POP')
  jra_mean_indo = jra.nht[:,2,:].mean('time').values
  jra_std_indo = jra.nht[:,2,:].std('time').values
  plt.plot(jra.lat, jra_mean_indo,'k', label='JRA-55 v1.3', color='#1B2ACC', lw=1)
  plt.fill_between(jra.lat, jra_mean_indo-jra_std_indo, jra_mean_indo+jra_std_indo,
    alpha=0.25, edgecolor='#1B2ACC', facecolor='#089FFF')
  plt.plot(yobs,NCEP['IndoPac'],'k--',linewidth=0.5,label='NCEP')
  plt.plot(yobs,ECMWF['IndoPac'],'k.',linewidth=0.5,label='ECMWF')
  plotGandW(GandW['IndoPac']['lat'],GandW['IndoPac']['trans'],GandW['IndoPac']['err'])
  plt.xlabel(r'Latitude [$\degree$N]',fontsize=10)
  annotateObs()
  if diffusive is None: annotatePlot('Warning: Diffusive component of transport is missing.')
  if hbd is None: annotatePlot('Warning: LBD component of transport is missing.')
  plt.suptitle(suptitle)
  plt.legend(loc=0,fontsize=10)
  if args.savefigs:
    objOut = args.outdir+'/'+args.case_name+'_HeatTransport_IndoPacific.png'
    plt.savefig(objOut); plt.close()
  else:
    plt.show()

  print('{} was run successfully!'.format(os.path.basename(__file__)))

  return

def heatTrans(advective, diffusive=None, hbd=None, vmask=None):
  """Converts vertically integrated temperature advection into heat transport"""
  HT = advective[:]
  if diffusive is not None:
    HT = HT + diffusive[:]
  if hbd is not None:
    HT = HT + hbd[:]
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

