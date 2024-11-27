#!/usr/bin/env python

import io, yaml, os
import matplotlib.pyplot as plt
import numpy as np
import warnings, dask, intake
from datetime import datetime
import xarray as xr
from mom6_tools.m6toolbox import cime_xmlquery
from ncar_jobqueue import NCARCluster
from dask.distributed import Client
from mom6_tools import m6plot
from mom6_tools  import m6toolbox
from mom6_tools.MOM6grid import MOM6grid

def options():
  try: import argparse
  except: raise Exception('This version of python is not new enough. python 2.7 or newer is required.')
  parser = argparse.ArgumentParser(description='''Script for plotting meridional overturning circulation.''')
  parser.add_argument('diag_config_yml_path', type=str, help='''Full path to the yaml file  \
    describing the run and diagnostics to be performed.''')
  #parser.add_argument('-v', '--var', nargs='+', default=['vmo'],
  #                   help='''Variable to be processed (default=['vmo'])''')
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
  main(cmdLineArgs)

def main():
  # Get options
  args = options()

  nw = args.number_of_workers
  if not os.path.isdir('PNG/MOC'):
    print('Creating a directory to place figures (PNG/MOC)... \n')
    os.system('mkdir -p PNG/MOC')
  if not os.path.isdir('ncfiles'):
    print('Creating a directory to place figures (ncfiles)... \n')
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

  args.savefigs = True; args.outdir = 'PNG/MOC/'
  print('Output directory is:', OUTDIR)
  print('Casename is:', args.casename)
  print('Number of workers to be used:', nw)

  # set avg dates
  avg = diag_config_yml['Avg']
  if not args.start_date : args.start_date = avg['start_date']
  if not args.end_date : args.end_date = avg['end_date']

  # file names are provided via yaml
  args.monthly = args.casename+diag_config_yml['Fnames']['z']
  args.sigma2 = args.casename+diag_config_yml['Fnames']['rho2']
  args.static = args.casename+diag_config_yml['Fnames']['static']
  args.geom = args.casename+diag_config_yml['Fnames']['geom']

  # read grid info
  geom_file = OUTDIR+'/'+args.geom
  if os.path.exists(geom_file):
    grd = MOM6grid(OUTDIR+'/'+args.static, geom_file)
  else:
    grd = MOM6grid(OUTDIR+'/'+args.static)

  try:
    depth = grd.depth_ocean
  except:
    depth = grd.deptho
  # remote Nan's, otherwise genBasinMasks won't work
  depth[np.isnan(depth)] = 0.0
  basin_code = m6toolbox.genBasinMasks(grd.geolon, grd.geolat, depth)

  parallel = False
  if nw > 1:
    parallel = True
    cluster = NCARCluster()
    cluster.scale(nw)
    client = Client(cluster)

  print('Reading {} dataset...'.format(args.monthly))
  startTime = datetime.now()
  # load data
  def preprocess(ds):
    variables = ['vmo','vhml','vhGM']
    for v in variables:
      if v not in ds.variables:
        ds[v] = xr.zeros_like(ds.vo)
    return ds[variables]

  ds = xr.open_mfdataset(OUTDIR+'/'+args.monthly, parallel=parallel, preprocess=preprocess)

  # use datetime
  #ds1['time'] = ds1.indexes['time'].to_datetimeindex()

  #ds = preprocess(ds1)

  print('Time elasped: ', datetime.now() - startTime)

  # compute yearly means first since this will be used in the time series
  attrs =  {
         'description': 'Annual mean meridional thickness flux by components ',
         'reduction_method': 'annual mean weighted by days in each month',
         'casename': args.casename
         }
  print('Computing yearly means...')
  startTime = datetime.now()
  ds_ann =  m6toolbox.weighted_temporal_mean_vars(ds,attrs=attrs)
  print('Time elasped: ', datetime.now() - startTime)

  startTime = datetime.now()
  print('Selecting data between {} and {}...'.format(args.start_date, args.end_date))
  ds_sel = ds_ann.sel(time=slice(args.start_date, args.end_date))
  print('Time elasped: ', datetime.now() - startTime)

  print('Computing time mean...')
  startTime = datetime.now()
  ds_mean = ds_sel.mean('time').compute()
  print('Time elasped: ', datetime.now() - startTime)

  # create a ndarray subclass
  class C(np.ndarray): pass
  varName = 'vmo'; conversion_factor = 1.e-9
  tmp = np.ma.masked_invalid(ds_mean[varName].values)
  tmp = tmp[:].filled(0.)
  VHmod = tmp.view(C)
  VHmod.units = ds[varName].units
  Zmod = m6toolbox.get_z(ds, depth, varName) # same here

  if args.casename != '':  casename = args.casename
  else: casename = ''

  # Global MOC
  m6plot.setFigureSize([16,9],576,debug=False)
  axis = plt.gca()
  cmap = plt.get_cmap('dunnePM')
  zg = Zmod.min(axis=-1)
  psiPlot = MOCpsi(VHmod)*conversion_factor
  psiPlot = 0.5 * (psiPlot[0:-1,:]+psiPlot[1::,:])
  yyg = grd.geolat_c[:,:].max(axis=-1)+0*zg
  ci=m6plot.pmCI(0.,40.,5.)
  plotPsi(yyg, zg, psiPlot, ci, 'Global MOC [Sv],' + 'averaged between '+ args.start_date + ' and '+ args.end_date )
  plt.xlabel(r'Latitude [$\degree$N]')
  plt.suptitle(casename)
  findExtrema(yyg, zg, psiPlot, max_lat=-30.)
  findExtrema(yyg, zg, psiPlot, min_lat=25., min_depth=250.)
  findExtrema(yyg, zg, psiPlot, min_depth=2000., mult=-1.)
  plt.gca().invert_yaxis()
  objOut = args.outdir+str(casename)+'_MOC_global.png'
  plt.savefig(objOut)

  if 'zl' in ds:
    zl=ds.zl.values
  elif 'z_l' in ds:
    zl=ds.z_l.values
  else:
    raise ValueError("Dataset does not have vertical coordinate zl or z_l")

  # create dataset to store results
  moc = xr.Dataset(data_vars={ 'moc' :    (('zl','yq'), psiPlot),
                            'amoc' :   (('zl','yq'), np.zeros((psiPlot.shape))),
                            'moc_FFM' :   (('zl','yq'), np.zeros((psiPlot.shape))),
                            'moc_GM' : (('zl','yq'), np.zeros((psiPlot.shape))),
                            'amoc_45' : (('time'), np.zeros((ds_ann.time.shape))),
                            'moc_GM_ACC' : (('time'), np.zeros((ds_ann.time.shape))),
                            'amoc_26' : (('time'), np.zeros((ds_ann.time.shape))) },
                            coords={'zl': zl, 'yq':ds.yq, 'time':ds_ann.time})
  attrs = {'description': 'MOC time-mean sections and time-series', 'units': 'Sv', 'start_date': avg['start_date'],
       'end_date': avg['end_date'], 'casename': args.casename}
  m6toolbox.add_global_attrs(moc,attrs)


  # Atlantic MOC
  m6plot.setFigureSize([16,9],576,debug=False)
  cmap = plt.get_cmap('dunnePM')
  m = 0*basin_code; m[(basin_code==2) | (basin_code==4) | (basin_code==6) | (basin_code==7) | (basin_code==8)]=1
  ci=m6plot.pmCI(0.,22.,2.)
  z = (m*Zmod).min(axis=-1)
  psiPlot = MOCpsi(VHmod, vmsk=m*np.roll(m,-1,axis=-2))*conversion_factor
  psiPlot = 0.5 * (psiPlot[0:-1,:]+psiPlot[1::,:])
  yy = grd.geolat_c[:,:].max(axis=-1)+0*z
  plotPsi(yy, z, psiPlot, ci, 'Atlantic MOC [Sv],'+ 'averaged between '+ args.start_date + ' and '+ args.end_date )
  plt.xlabel(r'Latitude [$\degree$N]')
  plt.suptitle(casename)
  # find range to extract values near the RAPID array
  # this will depend on the grid spacing
  try:
    tmp = findExtrema(yy, z, psiPlot, min_lat=26.5, max_lat=27., min_depth=250., plot=False) # RAPID
    min_lat_rapid = 26.5
    max_lat_rapid = 27.
    max_lat = -33.
  except: # for low-res configurations
    min_lat_rapid = 26.
    max_lat_rapid = 28.
    max_lat = -30.

  findExtrema(yy, z, psiPlot, min_lat=min_lat_rapid, max_lat=max_lat_rapid, min_depth=250.) # RAPID
  findExtrema(yy, z, psiPlot, min_lat=44, max_lat=46., min_depth=250.) # RAPID
  findExtrema(yy, z, psiPlot, max_lat=max_lat)
  findExtrema(yy, z, psiPlot)
  findExtrema(yy, z, psiPlot, min_lat=5.)
  plt.gca().invert_yaxis()
  objOut = args.outdir+str(casename)+'_MOC_Atlantic.png'
  plt.savefig(objOut,format='png')
  moc['amoc'].data = psiPlot

  print('Plotting AMOC profile at 26N...')
  catalog = intake.open_catalog(diag_config_yml['oce_cat'])
  rapid_vertical = catalog["moc-rapid"].to_dask()
  fig, ax = plt.subplots(nrows=1, ncols=1)
  ax.plot(rapid_vertical.stream_function_mar.mean('time'), rapid_vertical.depth, 'k', label='RAPID')
  ax.plot(moc['amoc'].sel(yq=26, method='nearest'), zl, label=casename)
  ax.legend()
  plt.gca().invert_yaxis()
  plt.grid()
  ax.set_xlabel('AMOC @ 26N [Sv]')
  ax.set_ylabel('Depth [m]')
  objOut = args.outdir+str(casename)+'_MOC_profile_26N.png'
  plt.savefig(objOut,format='png')

  print('Computing time series...')
  startTime = datetime.now()
  # time-series
  dtime = ds_ann.time
  amoc_26 = np.zeros(len(dtime))
  amoc_45 = np.zeros(len(dtime))
  moc_GM_ACC = np.zeros(len(dtime))
  if args.debug: startTime = datetime.now()
  # loop in time
  for t in range(len(dtime)):
    tmp = np.ma.masked_invalid(ds_ann[varName][t,:].values)
    tmp = tmp[:].filled(0.)
    # m is still Atlantic ocean
    psi = MOCpsi(tmp, vmsk=m*np.roll(m,-1,axis=-2))*conversion_factor
    psi = 0.5 * (psi[0:-1,:]+psi[1::,:])
    amoc_26[t] = findExtrema(yy, z, psi, min_lat=min_lat_rapid, max_lat=max_lat_rapid, plot=False, min_depth=250.)
    amoc_45[t] = findExtrema(yy, z, psi, min_lat=44., max_lat=46., plot=False, min_depth=250.)
    tmp_GM = np.ma.masked_invalid(ds_ann['vhGM'][t,:].values)
    tmp_GM = tmp_GM[:].filled(0.)
    psiGM = MOCpsi(tmp_GM)*conversion_factor
    psiGM = 0.5 * (psiGM[0:-1,:]+psiGM[1::,:])
    moc_GM_ACC[t] = findExtrema(yyg, zg, psiGM, min_lat=-65., max_lat=-30, mult=-1., plot=False)
  print('Time elasped: ', datetime.now() - startTime)

  # add dataarays to the moc dataset
  moc['amoc_26'].data = amoc_26
  moc['amoc_45'].data = amoc_45
  moc['moc_GM_ACC'].data = moc_GM_ACC

  if parallel:
    print('Releasing workers ...')
    client.close(); cluster.close()

  print('Plotting...')

  # load datasets from oce catalog
  amoc_core_26 = catalog["moc-core2-26p5"].to_dask()
  amoc_pop_26  = catalog["moc-pop-jra-26"].to_dask()
  rapid = m6toolbox.weighted_temporal_mean_vars(catalog["transports-rapid"].to_dask())

  amoc_core_45 = catalog["moc-core2-45"].to_dask()

  amoc_pop_45 = catalog["moc-pop-jra-45"].to_dask()

  # plot
  fig = plt.figure(figsize=(12, 6))
  plt.plot(np.arange(len(moc.time))+1958.5 ,moc['amoc_26'].values, color='k', label=casename, lw=2)
  # core data
  core_mean = amoc_core_26['MOC'].mean(axis=0).data
  core_std = amoc_core_26['MOC'].std(axis=0).data
  plt.plot(amoc_core_26.time,core_mean, 'k', label='CORE II (group mean)', color='#1B2ACC', lw=1)
  plt.fill_between(amoc_core_26.time, core_mean-core_std, core_mean+core_std,
    alpha=0.25, edgecolor='#1B2ACC', facecolor='#089FFF')
  # pop data
  plt.plot(np.arange(len(amoc_pop_26.time))+1958.5 ,amoc_pop_26.AMOC_26n.values, color='r', label='POP', lw=1)
  # rapid
  plt.plot(np.arange(len(rapid.time))+2004.5 ,rapid.moc_mar_hc10.values, color='green', label='RAPID', lw=1)

  plt.title('AMOC @ 26 $^o$ N', fontsize=16)
  plt.ylim(5,20)
  plt.xlim(1948,1958.5+len(moc.time))
  plt.xlabel('Time [years]', fontsize=16); plt.ylabel('Sv', fontsize=16)
  plt.legend(fontsize=13, ncol=2)
  objOut = args.outdir+str(casename)+'_MOC_26N_time_series.png'
  plt.savefig(objOut,format='png')

  # plot
  fig = plt.figure(figsize=(12, 6))
  plt.plot(np.arange(len(moc.time))+1958.5 ,moc['amoc_45'], color='k', label=casename, lw=2)
  # core data
  core_mean = amoc_core_45['MOC'].mean(axis=0).data
  core_std = amoc_core_45['MOC'].std(axis=0).data
  plt.plot(amoc_core_45.time,core_mean, 'k', label='CORE II (group mean)', color='#1B2ACC', lw=2)
  plt.fill_between(amoc_core_45.time, core_mean-core_std, core_mean+core_std,
    alpha=0.25, edgecolor='#1B2ACC', facecolor='#089FFF')
  # pop data
  plt.plot(np.arange(len(amoc_pop_45.time))+1958.5 ,amoc_pop_45.AMOC_45n.values, color='r', label='POP', lw=1)

  plt.title('AMOC @ 45 $^o$ N', fontsize=16)
  plt.ylim(5,20)
  plt.xlim(1948,1958+len(moc.time))
  plt.xlabel('Time [years]', fontsize=16); plt.ylabel('Sv', fontsize=16)
  plt.legend(fontsize=14)
  objOut = args.outdir+str(casename)+'_MOC_45N_time_series.png'
  plt.savefig(objOut,format='png')

  # Submesoscale-induced Global MOC
  class C(np.ndarray): pass
  varName = 'vhml'; conversion_factor = 1.e-9
  tmp = np.ma.masked_invalid(ds_mean[varName].values)
  tmp = tmp[:].filled(0.)
  VHml = tmp.view(C)
  VHml.units = ds[varName].units
  Zmod = m6toolbox.get_z(ds, depth, varName) # same here
  m6plot.setFigureSize([16,9],576,debug=False)
  axis = plt.gca()
  cmap = plt.get_cmap('dunnePM')
  z = Zmod.min(axis=-1); psiPlot = MOCpsi(VHml)*conversion_factor
  psiPlot = 0.5 * (psiPlot[0:-1,:]+psiPlot[1::,:])
  yy = grd.geolat_c[:,:].max(axis=-1)+0*z
  ci=m6plot.pmCI(0.,20.,2.)
  plotPsi(yy, z, psiPlot, ci, 'Global FFH MOC [Sv],' + 'averaged between '+ args.start_date + ' and '+ args.end_date,
          zval=[0.,-400.,-1000.])
  plt.xlabel(r'Latitude [$\degree$N]')
  plt.suptitle(casename)
  plt.gca().invert_yaxis()
  objOut = args.outdir+str(casename)+'_FFH_MOC_global.png'
  plt.savefig(objOut)
  moc['moc_FFM'].data = psiPlot

  # GM-induced Global MOC
  class C(np.ndarray): pass
  varName = 'vhGM'; conversion_factor = 1.e-9
  tmp = np.ma.masked_invalid(ds_mean[varName].values)
  tmp = tmp[:].filled(0.)
  VHGM = tmp.view(C)
  VHGM.units = ds[varName].units
  Zmod = m6toolbox.get_z(ds, depth, varName) # same here
  m6plot.setFigureSize([16,9],576,debug=False)
  axis = plt.gca()
  cmap = plt.get_cmap('dunnePM')
  z = Zmod.min(axis=-1); psiPlot = MOCpsi(VHGM)*conversion_factor
  psiPlot = 0.5 * (psiPlot[0:-1,:]+psiPlot[1::,:])
  yy = grd.geolat_c[:,:].max(axis=-1)+0*z
  ci=m6plot.pmCI(0.,20.,2.)
  plotPsi(yy, z, psiPlot, ci, 'Global GM MOC [Sv],' + 'averaged between '+ args.start_date + ' and '+ args.end_date)
  plt.xlabel(r'Latitude [$\degree$N]')
  plt.suptitle(casename)
  plt.gca().invert_yaxis()
  findExtrema(yy, z, psiPlot, min_lat=-65., max_lat=-30, mult=-1.)
  objOut = args.outdir+str(casename)+'_GM_MOC_global.png'
  plt.savefig(objOut)
  moc['moc_GM'].data = psiPlot

  print('Saving netCDF files...')
  moc.to_netcdf('ncfiles/'+str(casename)+'_MOC.nc')

  print('{} was run successfully!'.format(os.path.basename(__file__)))

  return

def MOCpsi(vh, vmsk=None):
  """Sums 'vh' zonally and cumulatively in the vertical to yield an overturning stream function, psi(y,z)."""
  shape = list(vh.shape); shape[-3] += 1
  psi = np.zeros(shape[:-1])
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
  psiMax = mult*np.amax( mult * np.ma.array(psi)[(y>=min_lat) & (y<=max_lat) & (z<-min_depth)] )
  idx = np.argmin(np.abs(psi-psiMax))
  (j,i) = np.unravel_index(idx, psi.shape)
  if plot:
    #plt.plot(y[j,i],z[j,i],'kx',hold=True)
    plt.plot(y[j,i],z[j,i],'kx')
    plt.text(y[j,i],z[j,i],'%.1f'%(psi[j,i]),color='red', fontsize=12)
  else:
    return psi[j,i]
if __name__ == '__main__':
  main()
