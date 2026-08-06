#!/usr/bin/env python

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
import warnings, os, yaml, argparse, dask
from datetime import datetime
import gsw
from ncar_jobqueue import NCARCluster
from dask.distributed import Client
from mom6_tools.m6toolbox import cime_xmlquery

warnings.filterwarnings('ignore')

# Observational / topography data path
DWBC_DATA = '/glade/campaign/cgd/oce/datasets/obs/DWBC_26.5N/'

# Transect definition (following Bilo & Johns 2020)
LAT_TRANSECT = 26.5
LON_MIN = -77.0
LON_MAX = -70.0


def options():
  try: import argparse
  except: raise Exception('This version of python is not new enough. python 2.7 or newer is required.')
  parser = argparse.ArgumentParser(description='''Script for plotting the 26.5N DWBC transect
    (mean meridional velocity, 77W-70W) following Bilo & Johns (2020).''')
  parser.add_argument('diag_config_yml_path', type=str, help='''Full path to the yaml file  \
    describing the run and diagnostics to be performed.''')
  parser.add_argument('-sd','--start_date', type=str, default='',
                      help='''Start year to compute averages. Default is to use value set in diag_config_yml_path''')
  parser.add_argument('-ed','--end_date', type=str, default='',
                      help='''End year to compute averages. Default is to use value set in diag_config_yml_path''')
  parser.add_argument('-nw','--number_of_workers',  type=int, default=0,
                      help='''Number of workers to use (default=0, serial).''')
  parser.add_argument('-debug',   help='''Add priting statements for debugging purposes''',
                      action="store_true")
  cmdLineArgs = parser.parse_args()
  return cmdLineArgs


def base_26N_transect(lon_min=-77.0, lon_max=-70.0):
  """Return (fig, ax_lon, ax_dist) with observed topography and station markers."""
  topography = xr.open_dataset(DWBC_DATA + '/topography_26.5N.nc')
  ftopo = interp1d(topography.longitude.values,
                   topography.topography.values, kind='linear')

  stations = xr.open_dataset(DWBC_DATA + '/stations_positions.nc')
  moorings = xr.open_dataset(DWBC_DATA + '/mooring_instrumentation.nc')
  mtopo    = ftopo(moorings.longitude.values)

  fig    = plt.figure(figsize=(10, 7), facecolor='w')
  ax_lon = plt.gca()
  ax_dist = ax_lon.twiny()

  # Stations
  ax_lon.plot(stations.longitude.values,
              np.zeros(stations.longitude.shape[0]),
              'o', mfc='darkorange', markeredgecolor='k', markersize=7, zorder=9)

  # Topography fill
  ax_lon.fill_between(topography.longitude.values, topography.topography,
                      y2=7000, color='k', zorder=10)

  # Mooring positions
  ax_lon.plot(moorings.longitude, mtopo + 120.0,
              '^', mfc='darkorange', markeredgecolor='k', ms=12, zorder=11)

  # Axes layout
  ax_lon.xaxis.tick_bottom()
  ax_dist.xaxis.tick_top()
  ax_dist.set_xlabel('Distance (km)', fontsize=18)
  ax_lon.set_ylabel('Depth (m)', fontsize=22)

  lon_range     = np.arange(-77.0, -69.0, 1.0)
  lon_range_str = [r'%i$^{\circ}$W' % int(np.abs(l)) for l in lon_range]

  ax_lon.set_yticks(range(0, 5500, 500))
  ax_lon.set_xticks(lon_range)
  ax_lon.set_xticklabels(lon_range_str, color='k', fontsize=18)
  ax_dist.set_xticks(range(0, 700, 50))
  ax_dist.tick_params(labelsize=15)
  ax_lon.tick_params(labelsize=18)

  ax_lon.set_ylim(5250, -20)
  ax_lon.set_xlim(lon_min, lon_max)
  ax_dist.set_xlim(0.0, stations.distance.max())

  return fig, ax_lon, ax_dist


def main():
  # Get options
  args = options()
  nw = args.number_of_workers

  os.makedirs('PNG/DWBC', exist_ok=True)
  os.makedirs('ncfiles', exist_ok=True)

  # Read in the yaml file
  diag_config_yml = yaml.load(open(args.diag_config_yml_path,'r'), Loader=yaml.Loader)

  caseroot = diag_config_yml['Case']['CASEROOT']
  casename = cime_xmlquery(caseroot, 'CASE')
  label = diag_config_yml['Case']['SNAME']
  DOUT_S = cime_xmlquery(caseroot, 'DOUT_S')
  if DOUT_S:
    OUTDIR = cime_xmlquery(caseroot, 'DOUT_S_ROOT')+'/ocn/hist/'
  else:
    OUTDIR = cime_xmlquery(caseroot, 'RUNDIR')

  z_stream    = casename + diag_config_yml['Fnames']['z']
  static_file = OUTDIR + '/' + casename + diag_config_yml['Fnames']['static']

  # set avg dates
  avg = diag_config_yml['Avg']
  if not args.start_date : args.start_date = avg['start_date']
  if not args.end_date : args.end_date = avg['end_date']

  print('Casename   :', casename)
  print('OUTDIR     :', OUTDIR)
  print('Date range :', args.start_date, '->', args.end_date)
  print('Stream     :', z_stream)
  print('Number of workers:', nw)

  parallel = False
  if nw > 1:
    parallel = True
    cluster = NCARCluster()
    cluster.scale(nw)
    client = Client(cluster)

  # --- Compute time-mean meridional velocity at 26.5N ---
  lat_transect = LAT_TRANSECT
  lon_min = LON_MIN
  lon_max = LON_MAX

  def preprocess(ds):
    """Keep only vo and select the transect latitude and longitude range."""
    yq_sel = ds.yq.sel(yq=lat_transect, method='nearest').values
    return ds[['vo']].sel(yq=yq_sel, method='nearest').sel(xh=slice(lon_min, lon_max))

  print('Opening z-level files...')
  startTime = datetime.now()
  ds = xr.open_mfdataset(
      OUTDIR + '/' + z_stream,
      parallel=parallel,
      data_vars='minimal',
      coords='minimal',
      compat='override',
      preprocess=preprocess,
  )
  print('Time elapsed:', datetime.now() - startTime)

  # Select date range and compute time mean
  ds_sel = ds.sel(time=slice(args.start_date, args.end_date))
  print('Time steps selected:', ds_sel.time.size)

  print('Computing time mean...')
  startTime = datetime.now()
  vo_mean = ds_sel.vo.mean('time').compute()
  print('Time elapsed:', datetime.now() - startTime)

  print('vo_mean shape (z_l, xh):', vo_mean.shape)
  print('Latitude selected (yq)  :', float(ds.yq))
  print('Longitude range (xh)    :', vo_mean.xh.values[0], '->', vo_mean.xh.values[-1])

  # --- Load model bathymetry along the transect ---
  static = xr.open_dataset(static_file)
  deptho = static['deptho'].sel(yh=lat_transect, method='nearest').sel(xh=slice(lon_min, lon_max))

  # --- Save to netCDF ---
  ds_out = vo_mean.to_dataset(name='vo')
  ds_out['deptho'] = deptho
  ds_out['vo'].attrs = {
      'long_name'   : 'Time-mean meridional velocity at {:.1f}N'.format(lat_transect),
      'units'       : 'm s-1',
      'cell_methods': 'time: mean',
  }
  ds_out['deptho'].attrs = {
      'long_name': 'Sea floor depth',
      'units'    : 'm',
  }
  ds_out.attrs = {
      'description' : 'Time-mean meridional velocity transect at {:.1f}N'.format(lat_transect),
      'casename'    : casename,
      'latitude'    : float(ds.yq),
      'lon_min'     : lon_min,
      'lon_max'     : lon_max,
      'start_date'  : args.start_date,
      'end_date'    : args.end_date,
  }
  outfile = 'ncfiles/{}_vo_mean_{:.1f}N_transect.nc'.format(casename, lat_transect)
  ds_out.to_netcdf(outfile)
  print('Saved:', outfile)

  # --- Colour map matching the observation panels (Bilo & Johns 2020) ---
  boundaries     = np.linspace(-0.3, 0.3, 50)
  cmap_seismic   = plt.cm.get_cmap('seismic', len(boundaries))
  colors         = list(cmap_seismic(np.arange(len(boundaries))))
  cmap_custom    = mpl.colors.ListedColormap(colors, '')
  cmap_custom.set_over(colors[-1])
  cmap_custom.set_under(colors[0])

  kw_color   = dict(vmin=-0.3, vmax=0.3, cmap=cmap_custom, zorder=0)
  kw_contour = dict(levels=[-0.15, -0.1, -0.05, 0.03, 0.1, 0.2, 0.3],
                    colors='k', linewidths=1, linestyles='solid', zorder=1)
  kw_contourz = dict(levels=[0.0],
                     colors='k', linewidths=2.5, linestyles='solid', zorder=1)

  xh  = vo_mean.xh.values
  z_l = vo_mean.z_l.values
  vel = vo_mean.values

  # --- Figure 1: single-panel model transect ---
  fig, ax_lon, ax_dist = base_26N_transect(lon_min=lon_min, lon_max=lon_max)

  cm = ax_lon.pcolormesh(xh, z_l, vel, **kw_color)

  try:
    c  = ax_lon.contour(xh, z_l, vel, **kw_contour)
    cz = ax_lon.contour(xh, z_l, vel, **kw_contourz)
    ax_lon.clabel(c,  fmt='%1.2f', colors='k',     fontsize=13, inline=True, zorder=14)
    ax_lon.clabel(cz, fmt='%i',    colors='beige',  fontsize=15, inline=True, zorder=14)
    [label.set_bbox(dict(edgecolor='none', facecolor='k', pad=0))
     for label in ax_lon.findobj(mpl.text.Text) if label.get_text() == '0']
  except Exception:
    pass

  # Model bathymetry overlay
  ax_lon.fill_between(deptho.xh.values, deptho.values, y2=7000,
                      color='saddlebrown', alpha=0.6, zorder=8, label='Model bathymetry')

  # Colour bar
  cbaxes = fig.add_axes([0.925, 0.115, 0.010, 0.77])
  cb = fig.colorbar(cm, cax=cbaxes, extend='both',
                    ticks=np.arange(-0.3, 0.35, 0.05))
  cb.ax.tick_params(labelsize=13)
  cb.ax.set_title(r'm s$^{-1}$', fontsize=13, x=1.3)

  # Title
  ax_lon.text(
      lon_min + 0.3, 500,
      '{:.1f}N  {}\n{} - {}'.format(lat_transect, label, args.start_date, args.end_date),
      color='beige', fontsize=14,
      bbox=dict(boxstyle='round', facecolor='k'), zorder=12)

  savefig = 'PNG/DWBC/{}_vo_mean_{:.1f}N_transect.png'.format(casename, lat_transect)
  fig.savefig(savefig, bbox_inches='tight', pad_inches=0.05, dpi=150)
  print('Figure saved:', savefig)
  plt.close(fig)

  # --- Figure 2: 4-panel comparison (LADCP, MOCHA, Argo, MOM6) ---
  ladcp  = xr.open_dataset(DWBC_DATA + '/ladcp_statistics.nc')
  mocha  = xr.open_dataset(DWBC_DATA + '/mocha_currentmeter_dwbc_2008_2018.nc')
  argo   = xr.open_dataset(DWBC_DATA + '/argo_vgeos_26.5N.nc')
  argo_depth = -gsw.z_from_p(argo.pressure.values, lat_transect)

  datasets = [
      dict(label='LADCP',
           x=ladcp.longitude.values,  y=ladcp.depth.values,
           v=ladcp.v_mean.values / 100.0,   kind='contourf'),
      dict(label='MOCHA moorings',
           x=mocha.longitude.values, y=mocha.depth.values,
           v=np.nanmean(mocha.v.values, axis=-1).T, kind='contourf'),
      dict(label='Argo geostrophy',
           x=argo.longitude[1:-1].values, y=argo_depth,
           v=argo.v[:, 1:-1].values,      kind='pcolormesh'),
      dict(label='{}\n{} - {}'.format(label, args.start_date, args.end_date),
           x=xh, y=z_l,
           v=vel,                          kind='pcolormesh'),
  ]

  topography = xr.open_dataset(DWBC_DATA + '/topography_26.5N.nc')

  fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor='w',
                           sharex=False, sharey=True)
  axes = axes.flatten()

  for ax, d in zip(axes, datasets):
    if d['kind'] == 'contourf':
      cm = ax.contourf(d['x'], d['y'], d['v'],
                       np.linspace(-0.3, 0.3, 50), **kw_color)
    else:
      cm = ax.pcolormesh(d['x'], d['y'], d['v'], **kw_color)

    try:
      ax.contour(d['x'], d['y'], d['v'], **kw_contour)
      ax.contour(d['x'], d['y'], d['v'], **kw_contourz)
    except Exception:
      pass

    # Topography
    ax.fill_between(topography.longitude.values, topography.topography.values,
                    y2=7000, color='k', zorder=10)

    # Label
    ax.text(0.03, 0.07, d['label'], transform=ax.transAxes,
            color='beige', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='k'), zorder=12)

    ax.set_ylim(5250, -20)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylabel('Depth (m)', fontsize=14)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.tick_params(labelsize=12)

    lon_ticks = np.arange(-77.0, -69.0, 1.0)
    ax.set_xticks(lon_ticks)
    ax.set_xticklabels([r'%i$^{\circ}$W' % int(abs(l)) for l in lon_ticks], fontsize=11)

  # Shared colour bar
  fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.25)
  cbaxes = fig.add_axes([0.91, 0.12, 0.012, 0.76])
  cb = fig.colorbar(cm, cax=cbaxes, extend='both',
                    ticks=np.arange(-0.3, 0.35, 0.05))
  cb.ax.tick_params(labelsize=12)
  cb.ax.set_title(r'm s$^{-1}$', fontsize=12, x=1.5)

  fig.suptitle('Mean meridional velocity at {:.1f}N'.format(lat_transect), fontsize=16)

  savefig4 = 'PNG/DWBC/{}_vo_mean_{:.1f}N_4panel.png'.format(casename, lat_transect)
  fig.savefig(savefig4, bbox_inches='tight', pad_inches=0.05, dpi=150)
  print('Figure saved:', savefig4)
  plt.close(fig)

  if parallel:
    print('\n Releasing workers...')
    client.close(); cluster.close()

  print('{} was run successfully!'.format(os.path.basename(__file__)))

  return


if __name__ == '__main__':
  main()
