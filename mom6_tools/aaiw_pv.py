#!/usr/bin/env python

import io, yaml, os
import matplotlib.pyplot as plt
import numpy as np
import warnings, dask, netCDF4, intake
from datetime import datetime, date
import xarray as xr
from ncar_jobqueue import NCARCluster
from dask.distributed import Client
import momlevel as ml
from mom6_tools import m6plot
from mom6_tools.m6toolbox import genBasinMasks
from mom6_tools.m6toolbox import weighted_temporal_mean_vars
from mom6_tools.m6toolbox import add_global_attrs
from mom6_tools.m6toolbox import cime_xmlquery
from mom6_tools.m6toolbox import geoslice
from mom6_tools.m6toolbox import infer_bounds
from mom6_tools.m6toolbox import standard_grid_area
from mom6_tools.MOM6grid import MOM6grid

def options():
  try: import argparse
  except: raise Exception('This version of python is not new enough. python 2.7 or newer is required.')
  parser = argparse.ArgumentParser(description='''Script for computing and plotting the buoyancy contribution to potential \
    vorticity over the Pacific Sector of the Southern Ocean. \
    Acknowledgment: this script builds on work by John Krasting (NOAA/GFDL). \
    The original version can be found at:  \
    https://github.com/jkrasting/mar/blob/main/src/gfdlnb/notebooks/ocean/AAIW_PV.ipynb.\
    ''')
  parser.add_argument('diag_config_yml_path', type=str, help='''Full path to the yaml file  \
    describing the run and diagnostics to be performed.''')
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
  if not os.path.isdir('PNG/AAIW_PV'):
    print('Creating a directory to place figures (PNG/AAIW_PV)... \n')
    os.system('mkdir -p PNG/AAIW_PV')
  if not os.path.isdir('ncfiles'):
    print('Creating a directory to store netcdf files (ncfiles)... \n')
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

  print('Output directory is:', OUTDIR)
  print('Casename is:', args.casename)
  print('Number of workers to be used:', nw)

  # set avg dates and other params
  avg = diag_config_yml['Avg']
  if not args.start_date : args.start_date = avg['start_date']
  if not args.end_date : args.end_date = avg['end_date']
  args.monthly = args.casename+diag_config_yml['Fnames']['z']
  args.static = args.casename+diag_config_yml['Fnames']['static']
  args.geom = args.casename+diag_config_yml['Fnames']['geom']
  args.savefigs = True
  args.label = diag_config_yml['Case']['SNAME']
  args.outdir = 'PNG/AAIW_PV/'

  # read grid info
  geom_file = OUTDIR+'/'+args.geom
  if os.path.exists(geom_file):
    grd = MOM6grid(OUTDIR+'/'+args.static, geom_file, xrformat=True)
  else:
    grd = MOM6grid(OUTDIR+'/'+args.static, xrformat=True)

  try:
    depth = grd.depth_ocean
  except:
    depth = grd.deptho

  # Coriolis
  coriolis = ml.derived.calc_coriolis(grd.geolat)

  parallel = False
  if nw > 1:
    parallel = True
    cluster = NCARCluster()
    cluster.scale(nw)
    client = Client(cluster)

  def preprocess(ds):
    ''' Return a dataset desired variables'''
    variables = ['thetao', 'so', 'volcello']
    return ds[variables]

  print('Reading dataset...')
  startTime = datetime.now()

  ds = xr.open_mfdataset(OUTDIR+'/'+args.monthly, parallel=parallel, \
                             combine="nested", concat_dim="time", \
                             preprocess=preprocess).chunk({"time": 12})

  print('Time elasped: ', datetime.now() - startTime)

  print('Selecting data between {} and {}...'.format(args.start_date, args.end_date))
  startTime = datetime.now()
  ds_sel = ds.sel(time=slice(args.start_date, args.end_date))
  print('Time elasped: ', datetime.now() - startTime)

  attrs =  {
         'description': 'Annual mean thetao, so and volcello',
         'reduction_method': 'annual mean weighted by days in each month',
         'casename': args.casename
         }

  print('Computing annual means...')
  startTime = datetime.now()
  ds_ann =  weighted_temporal_mean_vars(ds_sel,attrs=attrs)
  print('Time elasped: ', datetime.now() - startTime)

  print('Computing time mean...')
  startTime = datetime.now()
  ds_mean = ds_ann.mean('time').load()
  print('Time elasped: ', datetime.now() - startTime)

  # N2 and PV
  zeta = 0.0
  n2 = ml.derived.calc_n2(ds_mean.thetao, ds_mean.so)
  pv = ml.derived.calc_pv(zeta, coriolis, n2, interp_n2=False, units="cm")
  pv = pv.transpose("z_l", "yh", "xh")
  pv = pv.load()

  # Add the latitude and longitude as new coordinates to the pv DataArray
  pv = pv.assign_coords({
    "latitude": (("yh", "xh"), grd.geolat.data),
    "longitude": (("yh", "xh"), grd.geolon.data)
  })
  ds_mean = ds_mean.assign_coords({
    "latitude": (("yh", "xh"), grd.geolat.data),
    "longitude": (("yh", "xh"), grd.geolon.data)
  })

  levels, colors = ml.util.get_pv_colormap()
  pv = geoslice(pv, x=(-180,-120),y=(-65,0), xcoord="longitude", ycoord="latitude")
  yindex = pv.latitude.mean("xh")

  # Calcualte volume
  volcello = geoslice(ds_mean.volcello, x=(-180,-120),y=(-65,0),
                             xcoord="longitude", ycoord="latitude")
  volume = xr.where(pv > 60.0, volcello, np.nan).sel(z_l=slice(700, None)).sum()
  volume = volume.load()
  print(f"Volume of water with PV > 60 cm-2 s-1: {float(volume/1.0e15)} x 1.0e^15")

  # Make zonal mean plot
  pv = pv.weighted(grd.areacello.fillna(0)).mean("xh")
  pv = pv.transpose("z_l", "yh")

  # plot
  args.label = args.label + ', average between ' + args.start_date + ' and ' + args.end_date
  plot_aaiw_pv(yindex, pv.z_l, pv, levels, colors, args)

  description = 'buoyancy contribution to potential vorticity over the Pacific Sector of the Southern Ocean'
  attrs = {'description': description,
           'unit': 'cm2 s-1',
           'start_date': args.start_date,
           'end_date': args.end_date}
  add_global_attrs(pv,attrs)
  pv = pv.rename('pv')
  print('Saving netCDF files...')
  pv.to_netcdf('ncfiles/'+str(args.casename)+'_AAIW_PV.nc')

  if parallel:
    print('Releasing workers...')
    client.close(); cluster.close()

  print('{} was run successfully!'.format(os.path.basename(__file__)))
  return

def plot_aaiw_pv_obs(dsobs, levels, colors):
  '''Compute and plot the buoyancy contribution to potential \
    vorticity over the Pacific Sector of the Southern Ocean using \
    dsobs.'''

  dsobs = dsobs.sel(xh=slice(180, 240)).sel(yh=slice(None, 0))
  zeta = 0.0
  coriolis = ml.derived.calc_coriolis(dsobs.geolat)
  n2 = ml.derived.calc_n2(dsobs.thetao, dsobs.so)
  pv = ml.derived.calc_pv(zeta, coriolis, n2, interp_n2=False, units="cm")
  pv = pv.transpose("z_l", "yh", "xh")

  # Infer cell bounds
  lat = pv.geolat[:, 0].values
  lon = pv.geolon[0, :].values
  lat_b = infer_bounds(lat)
  lon_b = infer_bounds(lon)

  # Calculate cell area
  area = standard_grid_area(lat_b, lon_b)
  area = xr.DataArray(area, dims=("yh", "xh"), coords={"yh": pv.yh, "xh": pv.xh})

  # Calculate cell volume
  depth = (area * 0.0) + dsobs.z_i[-1]
  dz = ml.derived.calc_dz(dsobs.z_l, dsobs.z_i, depth)
  volcello = dz * area

  # Volume of high-PV water
  volume = xr.where(pv > 60.0, volcello, np.nan).sel(z_l=slice(700, None)).sum()

  print(f"Volume of water with PV > 60 cm-2 s-1: {float(volume)/1.0e15} x 1.0e^15")

  # Take the zonal mean
  pv = pv.weighted(area).mean("xh")

  fig = plt.figure(figsize=(8, 4), dpi=100)
  ax = plt.subplot(1, 1, 1)
  cb = ax.contourf(dsobs.yh, dsobs.z_l, pv, levels=levels, colors=colors)
  cs = ax.contour(
      dsobs.yh, dsobs.z_l, pv, levels=levels, colors=["k"], linewidths=0.4
  )
  ax.set_ylim(0, 1800.0)
  ax.invert_yaxis()

  _ = ax.set_xlabel("Latitude [deegrees]\n(Averaged over 180W to 120W)")
  _ = ax.set_ylabel("Depth [m]")

  ax.hlines(
      700,
      pv.yh.min(),
      pv.yh.max(),
      colors="blue",
      linestyles="dashed",
      linewidths=0.7,
  )

  ax.clabel(cs)

  _ = ax.text(
      0.0,
      1.06,
      r"Buoyancy Contribution to PV:  $(f * N^2)/g$",
      transform=ax.transAxes,
      fontsize=12,
      weight="bold",
  )

  _ = ax.text(
      0.0,
      1.015,
      r"Roemmich and Gilson Gridded Argo Climatology - 2004 to 2018",
      transform=ax.transAxes,
      fontsize=10,
      style="italic",
  )

  plt.colorbar(cb, ticks=[5, 20, 60, 80, 100, 200], label=r"cm$^{-2}$ s$^{-1}$")

  return

def plot_aaiw_pv(y, zl, pv, levels, colors, args):

  fig = plt.figure(figsize=(8, 4), dpi=100)
  ax = plt.subplot(1, 1, 1)
  cb = ax.contourf(y, zl, pv, levels=levels, colors=colors)
  cs = ax.contour(y, zl, pv, levels=levels, colors=["k"], linewidths=0.4)
  ax.set_ylim(0, 1800.0)
  ax.invert_yaxis()

  _ = ax.set_xlabel("Latitude [deegrees]\n(Averaged over 180W to 120W)")
  _ = ax.set_ylabel("Depth [m]")

  ax.hlines(
    700, y.min(), y.max(), colors="blue", linestyles="dashed", linewidths=0.7
  )

  ax.clabel(cs)

  _ = ax.text(
    0.0,
    1.06,
    r"Buoyancy Contribution to PV:  $(f * N^2)/g$",
    transform=ax.transAxes,
    fontsize=12,
    weight="bold",
  )

  _ = ax.text(
    0.0,
    1.015,
    args.label,
    transform=ax.transAxes,
    fontsize=10,
    style="italic",
  )

  plt.colorbar(cb, ticks=[5, 20, 60, 80, 100, 200], label=r"cm$^{-2}$ s$^{-1}$")
  if args.savefigs:
    fname = args.outdir + str(args.casename)+'_AAIW_PV.png'
    plt.savefig(fname, bbox_inches='tight')

if __name__ == '__main__':
  main()

