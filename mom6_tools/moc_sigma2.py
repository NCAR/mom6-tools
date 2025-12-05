#!/usr/bin/env python

import io, yaml, os
import matplotlib.pyplot as plt
import numpy as np
import warnings, dask, intake
from datetime import datetime
import xarray as xr
from xgcm import Grid
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
    print('Creating a directory to place output (ncfiles)... \n')
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
  args.sigma2 = args.casename+diag_config_yml['Fnames']['rho2']
  args.static = args.casename+diag_config_yml['Fnames']['static']
  args.geom = args.casename+diag_config_yml['Fnames']['geom']
  args.label = diag_config_yml['Case']['SNAME']

  # read grid info
  geom_file = OUTDIR+'/'+args.geom
  if os.path.exists(geom_file):
    grd = MOM6grid(OUTDIR+'/'+args.static, geom_file)
    grd_xr = MOM6grid(OUTDIR+'/'+args.static, geom_file, xrformat=True)
  else:
    grd = MOM6grid(OUTDIR+'/'+args.static)
    grd_xr = MOM6grid(OUTDIR+'/'+args.static, xrformat=True)

  try:
    depth = grd.depth_ocean
  except:
    depth = grd.deptho

  # remote Nan's, otherwise genBasinMasks won't work
  depth[np.isnan(depth)] = 0.0
  basin_code_xr = m6toolbox.genBasinMasks(grd.geolon, grd.geolat, depth, verbose=False, xda=True)

  # create a grid using xgcm
  coords = {
      'X': {'center': 'xh', 'right': 'xq'},
      'Y': {'center': 'yh', 'right': 'yq'},
  }

  metrics = {
      'X': ["dxt", "dxCu", "dxCv"],
      'Y': ["dyt", "dyCu", "dyCv"]
  }

  grid = Grid(grd_xr, coords=coords, periodic=['X'])

  parallel = False
  if nw > 1:
    parallel = True
    cluster = NCARCluster()
    cluster.scale(nw)
    client = Client(cluster)

  print('Reading {} dataset...'.format(args.sigma2))
  startTime = datetime.now()
  # load data
  def preprocess(ds):
    variables = ['vmo','vhml','vhGM','volcello']
    for v in variables:
      if v not in ds.variables:
        ds[v] = xr.zeros_like(ds.vo)
    return ds[variables]

  ds = xr.open_mfdataset(OUTDIR+'/'+args.sigma2, parallel=parallel, preprocess=preprocess)

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
  h = ds_sel['volcello']/grd_xr.areacello
  vmo  = ds_sel['vmo'].mean(dim='time')
  vhml  = ds_sel['vhml'].mean(dim='time')
  vhGM  = ds_sel['vhGM'].mean(dim='time')
  thk  = h.mean(dim='time')
  print('Time elasped: ', datetime.now() - startTime)

  vmo  = vmo.where(vmo < 1e14)
  vhml  = vhml.where(vhml < 1e14)
  vhGM  = vhGM.where(vhGM < 1e14)
  thk  = thk.where(thk < 1e10)
  zrho = thk.mean(dim='xh').cumsum(dim='rho2_l')

  vmo_xsum = vmo.sum(dim='xh')
  vhml_xsum = vhml.sum(dim='xh')
  vhGM_xsum = vhGM.sum(dim='xh')

  psi      = (vmo_xsum.cumsum(dim='rho2_l') - vmo_xsum.sum(dim='rho2_l'))/1e9 + 0.1
  psi.name = 'meridional-sigma2 overturning'
  # Submeso
  psi_vhml      = (vhml_xsum.cumsum(dim='rho2_l') - vhml_xsum.sum(dim='rho2_l'))/1e9 + 0.1
  psi_vhml.name = 'meridional-sigma2 overturning submeso'
  # Meso
  psi_vhGM      = (vhGM_xsum.cumsum(dim='rho2_l') - vhGM_xsum.sum(dim='rho2_l'))/1e9 + 0.1
  psi_vhGM.name = 'meridional-sigma2 overturning GM'

  # add a depth coordinate to psi, with depth
  # defined by zonal mean of the time mean depth of rho
  psi.coords['depth'] = grid.interp(zrho, 'Y', boundary='extend')
  psi_vhml.coords['depth'] = grid.interp(zrho, 'Y', boundary='extend')
  psi_vhGM.coords['depth'] = grid.interp(zrho, 'Y', boundary='extend')

  psi.load()
  psi_vhml.load()
  psi_vhGM.load()
  zrho.load()

  if args.casename != '':  casename = args.casename
  else: casename = ''

  # Global moc
  levels  = [-40,-35,-30,-25,-20,-15,-10,-5,5,10,15,20,25,30,35,40]
  clevels = [-40,-35,-30,-25,-20,-15,-10,-5,5,10,15,20,25,30,35,40]
  fig, axis = plt.subplots(1,1, figsize=(6.5,4))
  field  = psi
  ycoord = psi['rho2_l']
  xcoord = grd_xr['yq']
  xcoordmesh,ycoordmesh = np.meshgrid(xcoord,ycoord)

  p = plt.contourf(xcoord,ycoord,field,
                   cmap='RdBu_r',
                   levels=levels,
                  )
  plt.contour(xcoordmesh,ycoordmesh,field,clevels,colors='k',linewidths=0.5)
  plt.xlim((-75,70))
  plt.ylim((1029,1037.5))
  plt.gca().invert_yaxis()
  axis.set_ylabel("Sigma2 [kg/m$^3$]",fontsize=8)
  axis.set_xlabel("Latitude",fontsize=8)
  axis.set_facecolor('gray')
  axis.set_title("Case {}, meridional-rho overturning (Sv)".format(args.label),fontsize=10)
  cbar = plt.colorbar(p,pad=0.01,spacing='uniform', extend='both',
                      shrink=0.95,orientation='vertical')
  cbar.set_ticks(clevels)
  objOut = args.outdir+str(casename)+'_MOC_sigma2_global.png'
  plt.savefig(objOut)

  # zrho
  fig, axis = plt.subplots(1,1, figsize=(15,5))
  field = psi
  p = xr.plot.contourf(field, ax=axis, x="yq", y="depth",
                   cmap='RdBu_r',
                   levels=levels,
                   add_colorbar=False
                  )
  cbar = plt.colorbar(p,pad=0.01,spacing='uniform', extend='both',
                      shrink=0.95,orientation='vertical')
  cbar.set_ticks(levels)

  pc=xr.plot.contour(field, ax=axis, x="yq", y="depth",
                    yincrease=False,
                    linewidths=.5,
                    levels=levels,
                    colors='k'
                   )
  plt.xlim((-75,75))
  plt.ylim((0, 4500))
  plt.gca().invert_yaxis()

  axis.set_ylabel("Depth [m]",fontsize=8)
  axis.set_xlabel("Latitude",fontsize=8)
  axis.set_facecolor('gray')
  axis.set_title("Case {}, global meridional-zrho overturning".format(args.label),fontsize=10)
  objOut = args.outdir+str(casename)+'_MOC_zrho_global.png'
  plt.savefig(objOut)

  # create dataset to store results
  moc = xr.Dataset(data_vars={ 'moc' :    (('rho2_l','yq'), psi.data),
                            'amoc' :   (('rho2_l','yq'), np.zeros((psi.shape))),
                            'ipmoc' :   (('rho2_l','yq'), np.zeros((psi.shape))),
                            'moc_FFH' :   (('rho2_l','yq'), np.zeros((psi.shape))),
                            'moc_GM' : (('rho2_l','yq'), np.zeros((psi.shape))),
                            },
                            coords={'rho2_l': ycoord, 'yq':xcoord,
                                    'moc_depth':psi['depth']})

  attrs = {'description': 'MOC sigma2 time-mean sections', 'units': 'Sv', 'start_date': avg['start_date'],
       'end_date': avg['end_date'], 'casename': args.casename}
  m6toolbox.add_global_attrs(moc,attrs)

  # Submesoscale-induced Global MOC
  levels  = [-10,-8,-6,-4,-2,2,4,6,8,10]
  clevels = [-10,-8,-6,-4,-2,2,4,6,8,10]
  fig, axis = plt.subplots(1,1, figsize=(6.5,4))
  field  = psi_vhml
  ycoord = psi_vhml['rho2_l']
  xcoord = grd_xr['yq']
  xcoordmesh,ycoordmesh = np.meshgrid(xcoord,ycoord)

  p = plt.contourf(xcoord,ycoord,field,
                   cmap='RdBu_r',
                   levels=levels,
                  )
  plt.contour(xcoordmesh,ycoordmesh,field,clevels,colors='k',linewidths=0.5)
  plt.xlim((-75,70))
  plt.ylim((1029,1037.5))
  plt.gca().invert_yaxis()
  axis.set_ylabel("Sigma2 [kg/m$^3$]",fontsize=8)
  axis.set_xlabel("Latitude",fontsize=8)
  axis.set_facecolor('gray')
  axis.set_title("Case {}, meridional-rho overturning (Sv) due to vhml".format(args.label),fontsize=10)
  cbar = plt.colorbar(p,pad=0.01,spacing='uniform', extend='both',
                      shrink=0.95,orientation='vertical')
  cbar.set_ticks(clevels)
  objOut = args.outdir+str(casename)+'_MOC_sigma2_global_vhml.png'
  plt.savefig(objOut)
  moc['moc_FFH'].data = psi_vhml.data
  # zrho
  fig, axis = plt.subplots(1,1, figsize=(15,5))
  field = psi_vhml
  p = xr.plot.contourf(field, ax=axis, x="yq", y="depth",
                   cmap='RdBu_r',
                   levels=levels,
                   add_colorbar=False
                  )
  cbar = plt.colorbar(p,pad=0.01,spacing='uniform', extend='both',
                      shrink=0.95,orientation='vertical')
  cbar.set_ticks(levels)

  pc=xr.plot.contour(field, ax=axis, x="yq", y="depth",
                    yincrease=False,
                    linewidths=.5,
                    levels=levels,
                    colors='k'
                   )
  plt.xlim((-75,75))
  plt.ylim((0, 4500))
  plt.gca().invert_yaxis()

  axis.set_ylabel("Depth [m]",fontsize=8)
  axis.set_xlabel("Latitude",fontsize=8)
  axis.set_facecolor('gray')
  axis.set_title("Case {}, global meridional-zrho overturning (Sv) due to vhml".format(args.label),fontsize=10)
  objOut = args.outdir+str(casename)+'_MOC_zrho_global_vhml.png'
  plt.savefig(objOut)

  objOut = args.outdir+str(casename)+'_FFH_MOC_global.png'
  plt.savefig(objOut)
  moc['moc_FFH'].data = psi.data

  # GM-induced Global MOC
  fig, axis = plt.subplots(1,1, figsize=(6.5,4))
  field  = psi_vhGM
  ycoord = psi_vhGM['rho2_l']
  xcoord = grd_xr['yq']
  xcoordmesh,ycoordmesh = np.meshgrid(xcoord,ycoord)

  p = plt.contourf(xcoord,ycoord,field,
                   cmap='RdBu_r',
                   levels=levels,
                  )
  plt.contour(xcoordmesh,ycoordmesh,field,clevels,colors='k',linewidths=0.5)
  plt.xlim((-75,70))
  plt.ylim((1029,1037.5))
  plt.gca().invert_yaxis()
  axis.set_ylabel("Sigma2 [kg/m$^3$]",fontsize=8)
  axis.set_xlabel("Latitude",fontsize=8)
  axis.set_facecolor('gray')
  axis.set_title("Case {}, meridional-rho overturning (Sv) due to vhGM".format(args.label),fontsize=10)
  cbar = plt.colorbar(p,pad=0.01,spacing='uniform', extend='both',
                      shrink=0.95,orientation='vertical')
  cbar.set_ticks(clevels)
  objOut = args.outdir+str(casename)+'_MOC_sigma2_global_vhGM.png'
  plt.savefig(objOut)
  moc['moc_GM'].data = psi_vhGM.data
  # zrho
  fig, axis = plt.subplots(1,1, figsize=(15,5))
  field = psi_vhGM
  p = xr.plot.contourf(field, ax=axis, x="yq", y="depth",
                   cmap='RdBu_r',
                   levels=levels,
                   add_colorbar=False
                  )
  cbar = plt.colorbar(p,pad=0.01,spacing='uniform', extend='both',
                      shrink=0.95,orientation='vertical')
  cbar.set_ticks(levels)

  pc=xr.plot.contour(field, ax=axis, x="yq", y="depth",
                    yincrease=False,
                    linewidths=.5,
                    levels=levels,
                    colors='k'
                   )
  plt.xlim((-75,75))
  plt.ylim((0, 4500))
  plt.gca().invert_yaxis()

  axis.set_ylabel("Depth [m]",fontsize=8)
  axis.set_xlabel("Latitude",fontsize=8)
  axis.set_facecolor('gray')
  axis.set_title("Case {}, global meridional-zrho overturning (Sv) due to vhGM".format(args.label),fontsize=10)
  objOut = args.outdir+str(casename)+'_MOC_zrho_global_vhGM.png'
  plt.savefig(objOut)

  # Indo-Pacific
  atl = basin_code_xr.sel(region='AtlanticOcean') + basin_code_xr.sel(region='Arctic') + \
      basin_code_xr.sel(region='HudsonBay') + basin_code_xr.sel(region='MedSea') + \
      basin_code_xr.sel(region='BlackSea')

  m = basin_code_xr.sel(region='Global') - atl
  vmo  = ds_sel['vmo'].mean(dim='time') * grid.interp(m, 'Y', boundary='extend')
  thk = h.mean(dim='time').where(m != 0)
  vmo  = vmo.where(vmo < 1e14)
  thk  = thk.where(thk < 1e10)
  zrho = thk.mean(dim='xh').cumsum(dim='rho2_l')
  vmo_xsum = vmo.sum(dim='xh')
  psi      = (vmo_xsum.cumsum(dim='rho2_l') - vmo_xsum.sum(dim='rho2_l'))/1e9 + 0.1
  psi.name = 'meridional-sigma2 overturning'
  psi.coords['depth'] = grid.interp(zrho, 'Y', boundary='extend')
  psi.load()
  zrho.load()

  # plot
  # for indopac moc
  levels  = [-40,-35,-30,-25,-20,-15,-10,-5,5,10,15,20,25,30,35,40]
  clevels = [-40,-35,-30,-25,-20,-15,-10,-5,5,10,15,20,25,30,35,40]
  fig, axis = plt.subplots(1,1, figsize=(6.5,4))
  field  = psi
  ycoord = psi['rho2_l']
  xcoord = grd_xr['yq']
  xcoordmesh,ycoordmesh = np.meshgrid(xcoord,ycoord)
  p = plt.contourf(xcoord,ycoord,field,
                   cmap='RdBu_r',
                   levels=levels,
                  )
  plt.contour(xcoordmesh,ycoordmesh,field,clevels,colors='k',linewidths=0.5)
  plt.xlim((-34.5,50))
  plt.ylim((1029.5,1037.))
  plt.gca().invert_yaxis()
  axis.set_ylabel("Potential Density 2000 [kg/m$^3$]",fontsize=8)
  axis.set_xlabel("Latitude",fontsize=8)
  axis.set_facecolor('gray')
  axis.set_title("Case {}, Indo-Pacific meridional-rho overturning (Sv)".format(args.label),fontsize=10)
  cbar = plt.colorbar(p,pad=0.01,spacing='uniform', extend='both',
                      shrink=0.95,orientation='vertical')
  cbar.set_ticks(clevels)
  objOut = args.outdir+str(casename)+'_MOC_sigma2_IndoPacific.png'
  plt.savefig(objOut,format='png')

  #zrho
  fig, axis = plt.subplots(1,1, figsize=(15,5))
  field = psi
  p = xr.plot.contourf(field, ax=axis, x="yq", y="depth",
                   cmap='RdBu_r',
                   levels=levels,
                   add_colorbar=False
                  )
  cbar = plt.colorbar(p,pad=0.01,spacing='uniform', extend='both',
                      shrink=0.95,orientation='vertical')
  cbar.set_ticks(levels)
  pc=xr.plot.contour(field, ax=axis, x="yq", y="depth",
                    yincrease=False,
                    linewidths=.5,
                    levels=levels,
                    colors='k'
                   )
  plt.xlim((-34.5,50))
  plt.ylim((0, 4500))
  plt.gca().invert_yaxis()
  axis.set_ylabel("Depth [m]",fontsize=8)
  axis.set_xlabel("Latitude",fontsize=8)
  axis.set_facecolor('gray')
  axis.set_title("Case {}, Indo-Pacific meridional-zrho overturning".format(args.label),fontsize=10)
  objOut = args.outdir+str(casename)+'_MOC_zrho_IndoPacific.png'
  plt.savefig(objOut,format='png')
  moc['ipmoc'].data = psi.data
  moc = moc.assign_coords({"ipmoc_depth": (["rho2_l","yq"], psi['depth'].data)})

  # Atlantic MOC
  m = basin_code_xr.sel(region='AtlanticOcean') + basin_code_xr.sel(region='Arctic') + \
      basin_code_xr.sel(region='HudsonBay') + basin_code_xr.sel(region='MedSea') + \
      basin_code_xr.sel(region='BlackSea')

  vmo  = ds_sel['vmo'].mean(dim='time') * grid.interp(m, 'Y', boundary='extend')
  thk = h.mean(dim='time').where(m != 0)
  vmo  = vmo.where(vmo < 1e14)
  thk  = thk.where(thk < 1e10)
  zrho = thk.mean(dim='xh').cumsum(dim='rho2_l')
  vmo_xsum = vmo.sum(dim='xh')
  psi      = (vmo_xsum.cumsum(dim='rho2_l') - vmo_xsum.sum(dim='rho2_l'))/1e9 + 0.1
  psi.name = 'meridional-sigma2 overturning'
  psi.coords['depth'] = grid.interp(zrho, 'Y', boundary='extend')
  psi.load()
  zrho.load()
  levels  = [-30,-26,-22,-18,-14,-10,-6,-2,2,6,10,14,18,22,26,30]
  clevels = [-30,-26,-22,-18,-14,-10,-6,-2,2,6,10,14,18,22,26,30]

  # plot
  fig, axis = plt.subplots(1,1, figsize=(6.5,4))
  field  = psi
  ycoord = psi['rho2_l']
  xcoord = grd_xr['yq']
  xcoordmesh,ycoordmesh = np.meshgrid(xcoord,ycoord)

  p = plt.contourf(xcoord,ycoord,field,
                   cmap='RdBu_r',
                   levels=levels,
                  )
  plt.contour(xcoordmesh,ycoordmesh,field,clevels,colors='k',linewidths=0.5)

  plt.xlim((-40,75))
  plt.ylim((1031,1037.5))
  plt.gca().invert_yaxis()

  axis.set_ylabel("Potential Density 2000 [kg/m$^3$]",fontsize=8)
  axis.set_xlabel("Latitude",fontsize=8)
  axis.set_facecolor('gray')
  axis.set_title("Case {}, Atlantic meridional-rho overturning (Sv)".format(args.label),fontsize=10)
  cbar = plt.colorbar(p,pad=0.01,spacing='uniform', extend='both',
                      shrink=0.95,orientation='vertical')
  cbar.set_ticks(clevels)
  objOut = args.outdir+str(casename)+'_MOC_sigma2_Atlantic.png'
  plt.savefig(objOut,format='png')

  # zrho
  fig, axis = plt.subplots(1,1, figsize=(15,5))
  field = psi
  p = xr.plot.contourf(field, ax=axis, x="yq", y="depth",
                   cmap='RdBu_r',
                   levels=levels,
                   add_colorbar=False
                  )
  cbar = plt.colorbar(p,pad=0.01,spacing='uniform', extend='both',
                      shrink=0.95,orientation='vertical')
  cbar.set_ticks(levels)

  pc=xr.plot.contour(field, ax=axis, x="yq", y="depth",
                    yincrease=False,
                    linewidths=.5,
                    levels=levels,
                    colors='k'
                   )
  plt.xlim((-40,75))
  plt.ylim((0, 4500))
  plt.gca().invert_yaxis()

  axis.set_ylabel("Depth [m]",fontsize=8)
  axis.set_xlabel("Latitude",fontsize=8)
  axis.set_facecolor('gray')
  axis.set_title("Case {}, Atlantic meridional-zrho overturning".format(args.label),fontsize=10)
  objOut = args.outdir+str(casename)+'_MOC_zrho_Atlantic.png'
  plt.savefig(objOut,format='png')
  moc['amoc'].data = psi.data
  moc = moc.assign_coords({"amoc_depth": (["rho2_l","yq"], psi['depth'].data)})

  print('Saving netCDF files...')
  moc.to_netcdf('ncfiles/'+str(casename)+'_MOC_sigma2.nc')

  if parallel:
    print('Releasing workers ...')
    client.close(); cluster.close()

  print('{} was run successfully!'.format(os.path.basename(__file__)))

  return

if __name__ == '__main__':
  main()
