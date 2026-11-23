#!/usr/bin/env python

import os, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import xarray as xr
from datetime import datetime
from mom6_tools.jobqueue import add_jobqueue_args, get_cluster, release_workers
from mom6_tools.DiagsCase import DiagsCase
from mom6_tools.m6toolbox import weighted_temporal_mean_vars
from mom6_tools.m6toolbox import add_global_attrs, select_files_by_date
from mom6_tools.m6toolbox import check_vertical_levels

warnings.filterwarnings('ignore')
import gsw

# Circumpolar region plus the five Antarctic sea-ice sectors. Bounds are (west, east) in degrees east; a sector
# holds west <= lon < east (mod 360), so the five sectors tile the circumpolar region exactly.
REGIONS = ['Circumpolar', 'Weddell', 'Indian', 'WestPacific', 'Ross', 'AmundsenBellingshausen']
SECTORS = {'Weddell': (-60., 20.),
           'Indian': (20., 90.),
           'WestPacific': (90., 160.),
           'Ross': (160., -130.),
           'AmundsenBellingshausen': (-130., -60.)}
# per-variable metadata and plot settings
VARS = {'thetao': {'long_name': 'sea water potential temperature', 'units': 'degC',
                   'title': 'θ (°C)', 'dtitle': 'Δθ (°C)', 'cmap': 'RdYlBu_r', 'decimals': 2},
        'so': {'long_name': 'sea water salinity', 'units': 'psu',
               'title': 'S (psu)', 'dtitle': 'ΔS (psu)', 'cmap': 'viridis', 'decimals': 3},
        'sigma2': {'long_name': 'potential density referenced to 2000 dbar (TEOS-10, from 3-D annual-mean '
                                'thetao and so)', 'units': 'kg m-3 - 1000',
                   'title': 'σ2 (kg m⁻³)', 'dtitle': 'Δσ2 (kg m⁻³)', 'cmap': 'cividis_r', 'decimals': 3}}
SIG2_LEVELS = [35.5, 36.0, 36.5, 36.8, 36.9, 37.0, 37.05, 37.1, 37.15]   # sigma2 contours drawn on the sections

def options():
  try: import argparse
  except: raise Exception('This version of python is not new enough. python 2.7 or newer is required.')
  parser = argparse.ArgumentParser(description='''Computes Southern Ocean volume-weighted zonal-mean sections of \
    potential temperature, salinity and sigma2 (TEOS-10, from the 3-D thetao and so), averaged over the averaging window, for the region south of lat_max and the \
    five Antarctic sea-ice sectors (Weddell, Indian, WestPacific, Ross, AmundsenBellingshausen). The section is the \
    mean of the annual means of the years in the window (z-level stream); a year only partly in the window \
    contributes the mean of its available months. Sections are overlaid with sigma2 \
    contours (TEOS-10, from the 3-D thetao and so). Results are saved in OCN_DIAG_ROOT/<casename>_zonal_means.nc \
    and plotted (one figure per region, with theta, salinity and sigma2 panels) in <png_dir>/zonal_means/. \
    If Misc: zonal_mean in diag_config_yml_path points to the <casename>_zonal_means.nc of another case, \
    difference figures (this case minus that case) are also saved, labeled with Misc: zonal_mean_sname.''')
  parser.add_argument('diag_config_yml_path', type=str, help='''Full path to the yaml file  \
    describing the run and diagnostics to be performed.''')
  parser.add_argument('-sd','--start_date', type=str, default='',
                      help='''Start date. Default is the Avg start_date in diag_config_yml_path''')
  parser.add_argument('-ed','--end_date', type=str, default='',
                      help='''End date. Default is the Avg end_date in diag_config_yml_path''')
  parser.add_argument('-lat','--lat_max', type=float, default=-30.,
                      help='''Northern limit (default -30).''')
  parser.add_argument('-nw','--number_of_workers',  type=int, default=2,
                      help='''Number of workers to use (default=2).''')
  parser.add_argument('-debug',   help='''Add priting statements for debugging purposes''',
                      action="store_true")

  add_jobqueue_args(parser)
  cmdLineArgs = parser.parse_args()
  return cmdLineArgs

def in_sector(lon, west, east):
  """True where west <= lon < east, all taken mod 360 (works across the dateline and the 0/360 seam)."""
  x, w, e = np.mod(lon, 360.), np.mod(west, 360.), np.mod(east, 360.)
  if w < e:
    return (x >= w) & (x < e)
  return (x >= w) | (x < e)

def region_masks(lon, lat, wet, lat_max=-30.):
  """
  Region masks [region, y, x] south of lat_max.

  Parameters
  ----------
  lon, lat, wet : 2D arrays
    Longitude (any range), latitude and wet mask (1 = ocean).

  Returns
  -------
  masks : 3D boolean array, regions ordered as REGIONS.
  """
  so = (wet > 0) & (lat < lat_max)
  masks = [so] + [so & in_sector(lon, *SECTORS[r]) for r in REGIONS[1:]]
  masks = np.array(masks)
  # every circumpolar cell must be in exactly one sector
  assert np.array_equal(masks[1:].sum(0), so.astype(int)), 'Sectors do not tile the circumpolar region'
  return masks

def sigma2(thetao, so, z, lon, lat):
  """
  TEOS-10 sigma2 from potential temperature and practical salinity.

  Parameters
  ----------
  thetao, so : 3D arrays [z, y, x]
  z : 1D array of depths (m, positive down)
  lon, lat : 2D arrays

  Returns
  -------
  sigma2 : 3D array (kg m-3 minus 1000)
  """
  p = gsw.p_from_z(-z[:, None, None], lat[None])
  SA = gsw.SA_from_SP(so, p, lon[None], lat[None])
  CT = gsw.CT_from_pt(SA, thetao)
  return gsw.sigma2(SA, CT)

def zonal_mean(x, vol, masks):
  """
  Volume-weighted zonal mean of x [z, y, x] for each mask -> [region, z, y]. NaN where there is no water.
  Also returns the summed weights [region, z, y].
  """
  ok = np.isfinite(x) & (vol > 0)
  w = np.where(ok, vol, 0.)
  xz = np.where(ok, x, 0.)
  out, wsum = [], []
  for m in masks:
    ws = (w * m[None]).sum(-1)
    with np.errstate(invalid='ignore', divide='ignore'):
      out.append(np.where(ws > 0, (xz * w * m[None]).sum(-1) / ws, np.nan))
    wsum.append(ws)
  return np.array(out), np.array(wsum)

def cell_volume(zi, depth, area):
  """Geometric cell volume [z, y, x] from interface depths, bathymetry and cell area."""
  dz = np.clip(np.minimum(zi[1:, None, None], depth[None]) - zi[:-1, None, None], 0., None)
  return dz * area[None]

def main():
  # Get options
  args = options()
  nw = args.number_of_workers

  # Read in the yaml file
  dcase = DiagsCase(args.diag_config_yml_path)
  args.casename = dcase.casename

  # set averaging dates and other params
  if not args.start_date : args.start_date = dcase.start_date
  if not args.end_date : args.end_date = dcase.end_date
  args.monthly = dcase.get_fname('z')
  args.label = dcase.label
  args.savefigs = dcase.savefigs
  args.outdir = dcase.create_png_dir('zonal_means')
  ncfile = os.path.join(dcase.ocn_diag_root, '{}_zonal_means.nc'.format(args.casename))
  # optional comparison case (Misc: zonal_mean / zonal_mean_sname)
  args.zonal_mean = dcase.zonal_mean
  args.zonal_mean_sname = dcase.zonal_mean_sname
  if args.zonal_mean and not os.path.isfile(args.zonal_mean):
    raise FileNotFoundError('Misc: zonal_mean in {} not found: {}'.format(args.diag_config_yml_path,
                                                                         args.zonal_mean))

  print('Casename is:', args.casename)
  print('netCDF file:', ncfile)
  if args.zonal_mean: print('Comparing with:', args.zonal_mean)

  compute(args, dcase, ncfile, nw)
  if args.savefigs:
    plot(args, ncfile)

  print('{} was run successfully!'.format(os.path.basename(__file__)))

def compute(args, dcase, ncfile, nw):
  # read grid info
  grd = dcase.get_grid(xrformat=True)
  try:
    depth = grd.depth_ocean
  except:
    depth = grd.deptho
  # rows with water and south of lat_max (rows are zonal south of the tripolar fold, so this is a contiguous range)
  lat_row = grd.geolat.where(grd.wet > 0).mean('xh').values
  rows = np.where(np.isfinite(lat_row) & (lat_row < args.lat_max))[0]
  ys = slice(int(rows[0]), int(rows[-1]) + 1)
  assert len(rows) == rows[-1] - rows[0] + 1, 'Rows south of lat_max are not contiguous'
  lon = np.nan_to_num(grd.geolon.values[ys])
  lat = np.nan_to_num(grd.geolat.values[ys])
  wet = grd.wet.values[ys]
  area = np.nan_to_num(grd.areacello.values[ys]) * (wet > 0)
  dep = np.nan_to_num(np.asarray(depth)[ys]) * (wet > 0)
  masks = region_masks(lon, lat, wet, args.lat_max)
  lat1 = lat_row[ys]
  print('Rows {}-{}, lat {:.2f} to {:.2f}; wet cells per region:'.format(ys.start, ys.stop - 1, lat1[0], lat1[-1]),
        dict(zip(REGIONS, [int(m.sum()) for m in masks])))

  parallel, cluster, client = get_cluster(nw, args=args,
                                          config=dcase.jobqueue_config)

  # thetao and so are read; sigma2 is computed from them
  save_vars = list(VARS)

  def preprocess(ds):
    ''' Return a dataset desired variables'''
    return ds[['thetao', 'so']].isel(yh=ys)

  print('Reading dataset...')
  startTime = datetime.now()
  files = select_files_by_date(args.monthly, args.start_date, args.end_date)
  print('Opening {} files: {} ... {}'.format(len(files), files[0], files[-1]))
  ds = xr.open_mfdataset(files, parallel=parallel, \
                         combine="nested", concat_dim="time", \
                         preprocess=preprocess, use_cftime=True).chunk({"time": 12})
  print('Time elasped: ', datetime.now() - startTime)

  print('Selecting data between {} and {}...'.format(args.start_date, args.end_date))
  ds_sel = ds.sel(time=slice(args.start_date, args.end_date))
  # average over all years of the window, as in the other scripts: a year only partly in the window
  # contributes the annual mean of its available months (weighted by days in each month)
  years, nmonths = np.unique(ds_sel.time.dt.year.values, return_counts=True)
  partial = {int(y): int(m) for y, m in zip(years, nmonths) if m != 12}
  years = [int(y) for y in years]
  if not years:
    raise ValueError('No model data between {} and {}'.format(args.start_date, args.end_date))
  if partial:
    print('Partial years (year: months):', partial)
  gaps = sorted(set(range(years[0], years[-1] + 1)) - set(years))
  print('Averaging {} years: {}-{}'.format(len(years), years[0], years[-1]),
        '(missing: {})'.format(gaps) if gaps else '')

  z = ds_sel.z_l.values
  with xr.open_dataset(files[0], decode_times=False) as d0:
    zi = d0.z_i.values
  vol = cell_volume(zi, dep, area)

  # time mean = mean of the annual zonal-mean sections, computed one year at a time
  shape = (len(REGIONS), len(z), len(lat1))
  acc = {v: np.zeros(shape) for v in save_vars}
  cnt = {v: np.zeros(shape) for v in save_vars}
  wvol = None
  print('Computing zonal means...')
  startTime = datetime.now()
  for y in years:
    dsy = ds_sel.sel(time=slice('{:04d}-01-01'.format(y), '{:04d}-12-31'.format(y)))
    yr = weighted_temporal_mean_vars(dsy).isel(time=0).load()
    T = yr.thetao.values.astype('f8')
    S = yr.so.values.astype('f8')
    fields = {'thetao': T, 'so': S, 'sigma2': sigma2(T, S, z, lon, lat)}
    for v, x in fields.items():
      sec, ws = zonal_mean(x, vol, masks)
      ok = np.isfinite(sec)
      acc[v] += np.where(ok, sec, 0.)
      cnt[v] += ok
      if wvol is None and v == 'thetao': wvol = ws
    if args.debug: print('  year', y, datetime.now() - startTime)
  out = {}
  for v in save_vars:
    with np.errstate(invalid='ignore', divide='ignore'):
      out[v] = np.where(cnt[v] > 0, acc[v] / cnt[v], np.nan)
    part = ((cnt[v] > 0) & (cnt[v] < len(years))).sum()
    if part: print('  Warning: {}: {} points have water in only some years'.format(v, part))
  print('Time elasped: ', datetime.now() - startTime)

  dso = xr.Dataset(coords={'region': ('region', REGIONS, {'long_name': 'Region name'}),
                           'z_l': ('z_l', z, {'long_name': 'Depth at cell center', 'units': 'm', 'positive': 'down'}),
                           'z_i': ('z_i', zi, {'long_name': 'Depth at cell interface', 'units': 'm', 'positive': 'down'}),
                           'lat': ('lat', lat1, {'long_name': 'Row-mean latitude of wet cells', 'units': 'degrees_north'})})
  dso = dso.assign_coords(
    lon_west=('region', np.array([-180.] + [SECTORS[r][0] for r in REGIONS[1:]]),
              {'long_name': 'Western longitude bound of region (inclusive)', 'units': 'degrees_east'}),
    lon_east=('region', np.array([180.] + [SECTORS[r][1] for r in REGIONS[1:]]),
              {'long_name': 'Eastern longitude bound of region (exclusive, mod 360)', 'units': 'degrees_east'}))
  dims = ('region', 'z_l', 'lat')
  for v in save_vars:
    dso[v] = (dims, out[v].astype('f4'),
              {'long_name': 'Volume-weighted zonal-mean {}, mean of annual means'.format(VARS[v]['long_name']),
               'units': VARS[v]['units']})
  dso['wet_volume'] = (('region', 'z_l', 'lat'), wvol,
                       {'long_name': 'Sum over longitude of the cell volumes used as zonal-mean weights '
                                     '(geometric, from deptho, z_i and areacello; first year)', 'units': 'm3'})
  dso['wet_area'] = (('region', 'lat'), (area[None].astype('f8') * masks).sum(-1),
                     {'long_name': 'Sum over longitude of wet cell area (areacello)', 'units': 'm2'})
  attrs = {'description': 'Southern Ocean volume-weighted zonal-mean sections, averaged over the years '
                          'between start_date and end_date (a partial year contributes the mean of its available '
                          'months): the mean of the annual means (weighted by days in each '
                          'month), computed as the mean of the annual zonal-mean sections. sigma2 is '
                          'computed with TEOS-10 (gsw) from the '
                          '3-D annual-mean thetao and so and then zonally averaged. NaN = no water.',
           'regions': 'Circumpolar = all longitudes south of lat_max; sectors (west <= lon < east, mod 360): '
                      + '; '.join('{} {:g}..{:g}'.format(r, *SECTORS[r]) for r in REGIONS[1:])
                      + '. The five sectors tile the circumpolar region.',
           'lat_definition': 'lat is the row mean of geolat over wet cells (all longitudes); rows south of lat_max',
           'lat_max': args.lat_max,
           'start_date': str(args.start_date), 'end_date': str(args.end_date),
           'year_start': np.int32(years[0]), 'year_end': np.int32(years[-1]), 'nyears': np.int32(len(years)),
           'missing_years': ', '.join(str(y) for y in gaps) or 'none',
           'partial_years': ', '.join('{} ({} months)'.format(y, m) for y, m in partial.items()) or 'none',
           'source': args.monthly,
           'casename': args.casename, 'module': os.path.basename(__file__)}
  add_global_attrs(dso, attrs)
  enc = {v: {'_FillValue': np.float32(np.nan)} for v in save_vars}
  enc.update({v: {'_FillValue': None} for v in ['wet_volume', 'wet_area', 'z_l', 'z_i', 'lat', 'lon_west', 'lon_east']})
  dso.to_netcdf(ncfile, encoding=enc)
  print('Saved', ncfile)

  release_workers(parallel, cluster, client)

def edges(c):
  """Cell edges from centers (midpoints, with the end cells mirrored)."""
  m = 0.5 * (c[1:] + c[:-1])
  return np.concatenate([[2 * c[0] - m[0]], m, [2 * c[-1] - m[-1]]])

def color_levels(x, w, n=16, decimals=3):
  """Color levels bounding n equal-volume bins between the 2nd and 98th volume-weighted percentiles of x."""
  ok = np.isfinite(x) & (w > 0)
  x, w = x[ok], w[ok]
  i = np.argsort(x)
  cw = np.cumsum(w[i])
  lev = np.unique(np.round(np.interp(np.linspace(0.02, 0.98, n + 1), cw / cw[-1], x[i]), decimals))
  if len(lev) < 3:
    lev = np.linspace(x.min(), x.max() + 1e-6, n + 1)
  return lev

def section_figure(ncol):
  """Figure with a row of ncol section panels (shared axes), each with a thin colorbar axis below it."""
  fig = plt.figure(figsize=(4.5 * ncol, 3.6), layout='constrained')
  gs = fig.add_gridspec(2, ncol, height_ratios=[1., 0.06])
  axs = []
  for c in range(ncol):
    axs.append(fig.add_subplot(gs[0, c], sharex=axs[0] if axs else None, sharey=axs[0] if axs else None))
    if c > 0: axs[c].tick_params(labelleft=False)
  cax = [fig.add_subplot(gs[1, c]) for c in range(ncol)]
  return fig, axs, cax

def panel(ax, ds, data, s2, **kw):
  """Section of data on ax (latitude x depth), with sigma2 (s2) contours."""
  pc = ax.pcolormesh(edges(ds.lat.values), ds.z_i.values, data, shading='flat', **kw)
  cs = ax.contour(ds.lat, ds.z_l, s2, levels=SIG2_LEVELS, colors='k', linewidths=0.6)
  ax.clabel(cs, fontsize=5, fmt='%g')
  ax.set_ylim(5500, 0)
  ax.set_xlim(edges(ds.lat.values)[[0, -1]])
  return pc

def plot(args, ncfile):
  with xr.open_dataset(ncfile) as ds:
    ds = ds.load()
  _plot(args, ds)
  if args.zonal_mean:
    with xr.open_dataset(args.zonal_mean) as ds_ref:
      ds_ref = ds_ref.load()
    ds_ref = check_compatible(ds, ds_ref, args.zonal_mean)
    ref_label = args.zonal_mean_sname or ds_ref.attrs.get('casename', os.path.basename(args.zonal_mean))
    plot_diff(args, ds, ds_ref, ref_label)

def _plot(args, ds):
  period = 'years {}-{}'.format(int(ds.attrs['year_start']), int(ds.attrs['year_end']))
  # color levels from south of 50S, so the weakly stratified deep ocean is not washed out
  south = (ds.lat < -50.).values if (ds.lat < -50.).sum() > 2 else np.ones(ds.sizes['lat'], bool)

  # one figure per region; panels from left to right: thetao, so, sigma2
  for reg in ds.region.values:
    d = ds.sel(region=reg)
    fig, axs, cax = section_figure(len(VARS))
    for c, (v, st) in enumerate(VARS.items()):
      # color levels: equal-volume bins between the 2nd and 98th volume-weighted percentiles
      lev = color_levels(d[v].values[:, south], d.wet_volume.values[:, south], decimals=st['decimals'])
      norm = BoundaryNorm(lev, plt.get_cmap(st['cmap']).N, extend='both')
      pc = panel(axs[c], d, d[v], d.sigma2, cmap=st['cmap'], norm=norm)
      axs[c].set_title(st['title'], fontsize=10)
      axs[c].set_xlabel('latitude')
      cb = fig.colorbar(pc, cax=cax[c], orientation='horizontal', extend='both', spacing='uniform',
                        ticks=lev[::2], format='%.{}f'.format(st['decimals']))
      cb.ax.tick_params(labelsize=7)
    axs[0].set_ylabel('depth (m)')
    fig.suptitle('{} ({}): {} zonal mean, {}; contours: σ2'.format(args.casename, args.label, reg, period),
                 fontsize=10, x=0.01, ha='left')
    fname = os.path.join(args.outdir, '{}_zonal_means_{}.png'.format(args.casename, reg))
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print('Saved', fname)

def check_compatible(ds, ds_ref, ref_file):
  """
  Check that ds_ref (read from ref_file) has the variables of ds on the same region/z_l/lat axes.
  Vertical levels are checked with check_vertical_levels (tolerance 1 mm). Returns ds_ref on the
  z_l and lat coordinates of ds, so that differences are taken level by level and row by row.
  """
  missing = [v for v in VARS if v not in ds_ref]
  if missing:
    raise ValueError('{} not in {}'.format(missing, ref_file))
  # vertical levels: same number and depths (up to rounding)
  try:
    check_vertical_levels(ds_ref.z_l, ds.z_l)
  except ValueError as e:
    raise ValueError('Vertical levels of {} do not match this case: {}'.format(ref_file, e))
  for c in ['region', 'lat']:
    a, b = ds[c].values, ds_ref[c].values
    same = a.shape == b.shape and (np.array_equal(a, b) if a.dtype.kind in 'OUS' else np.allclose(a, b))
    if not same:
      raise ValueError('{} differs between this case and {} ({} vs {} points); both must be on the same grid and '
                       'computed with the same -lat'.format(c, ref_file, a.size, b.size))
  return ds_ref.assign_coords(z_l=ds.z_l, lat=ds.lat)

def plot_diff(args, ds, ds_ref, ref_label):
  """
  Difference sections, ds minus ds_ref: one figure per region with thetao, so and sigma2 panels. ds and ds_ref
  are datasets written by compute() on the same region/z_l/lat axes. Symmetric color limits from the
  95th percentile of |difference| in each panel; contours are sigma2 of ds.
  """
  period = 'years {}-{}'.format(int(ds.attrs['year_start']), int(ds.attrs['year_end']))
  ref_period = 'years {}-{}'.format(int(ds_ref.attrs['year_start']), int(ds_ref.attrs['year_end']))
  for reg in ds.region.values:
    d = ds.sel(region=reg)
    fig, axs, cax = section_figure(len(VARS))
    for c, (v, st) in enumerate(VARS.items()):
      dd = d[v] - ds_ref[v].sel(region=reg)
      lim = np.nanpercentile(np.abs(dd.values), 95)
      lim = lim if lim > 0 else 1.
      pc = panel(axs[c], d, dd, d.sigma2, cmap='RdBu_r', vmin=-lim, vmax=lim)
      axs[c].set_title(st['dtitle'], fontsize=10)
      axs[c].set_xlabel('latitude')
      cb = fig.colorbar(pc, cax=cax[c], orientation='horizontal', extend='both',
                        ticks=MaxNLocator(5, symmetric=True))
      cb.ax.tick_params(labelsize=7)
    axs[0].set_ylabel('depth (m)')
    fig.suptitle('{} ({}, {}) minus {} ({}): {} zonal mean; contours: σ2 of {}'.format(
                 args.casename, args.label, period, ref_label, ref_period, reg, args.label),
                 fontsize=10, x=0.01, ha='left')
    fname = os.path.join(args.outdir, '{}_zonal_means_diff_{}.png'.format(args.casename, reg))
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print('Saved', fname)

if __name__ == '__main__':
  main()
