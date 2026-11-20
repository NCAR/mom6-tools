#!/usr/bin/env python

import os, glob, warnings
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
from mom6_tools.m6toolbox import add_global_attrs

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
               'title': 'S (psu)', 'dtitle': 'ΔS (psu)', 'cmap': 'viridis', 'decimals': 3}}
SIG2_LEVELS = [35.5, 36.0, 36.5, 36.8, 36.9, 37.0, 37.05, 37.1, 37.15]   # sigma2 contours drawn on the sections

def options():
  try: import argparse
  except: raise Exception('This version of python is not new enough. python 2.7 or newer is required.')
  parser = argparse.ArgumentParser(description='''Computes Southern Ocean volume-weighted zonal-mean sections of \
    potential temperature or salinity in consecutive n-year blocks, for the \
    region south of lat_max and the five Antarctic sea-ice sectors (Weddell, Indian, WestPacific, Ross, \
    AmundsenBellingshausen). Each block is the mean of the annual means of its years (z-level stream); an incomplete \
    final block is dropped. By default there is a single block spanning the whole averaging window. Sections are \
    overlaid with sigma2 contours (TEOS-10, from the 3-D thetao and so). Results are saved in \
    OCN_DIAG_ROOT/<casename>_so_zonal_means_<var>_<tag>.nc (tag = <n>yr, or full when --nyears is not given) and \
    plotted (each block, and each block minus the first) in <png_dir>/SO_zonal_means/.''')
  parser.add_argument('diag_config_yml_path', type=str, help='''Full path to the yaml file  \
    describing the run and diagnostics to be performed.''')
  parser.add_argument('-v','--variable', type=str, required=True, choices=list(VARS),
                      help='''Name of variable to be processed: thetao or so.''')
  parser.add_argument('-n','--nyears', type=int, default=None,
                      help='''Block length in years (e.g. 10 gives years 1-10, 11-20, ...). Default is a single \
                      block spanning the complete years of the averaging window.''')
  parser.add_argument('-sd','--start_date', type=str, default='',
                      help='''Start date of the first block. Default is the Avg start_date in diag_config_yml_path''')
  parser.add_argument('-ed','--end_date', type=str, default='',
                      help='''End date. Default is the Avg end_date in diag_config_yml_path''')
  parser.add_argument('-lat','--lat_max', type=float, default=-30.,
                      help='''Northern limit (default -30).''')
  parser.add_argument('-plot_only', help='''Only make the figures from an existing netCDF file''',
                      action="store_true")
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

def make_blocks(years, nmonths, n):
  """
  Consecutive n-year blocks starting at years[0]. A block is kept only if all its years have 12 months.

  Returns
  -------
  blocks : list of (first_year, last_year) of the complete blocks
  dropped : list of (first_year, last_year, reason) of the blocks that were dropped
  """
  full = set(int(y) for y, m in zip(years, nmonths) if m == 12)
  y0, y1 = int(years[0]), int(years[-1])
  blocks, dropped = [], []
  for b0 in range(y0, y1 + 1, n):
    yrs = range(b0, b0 + n)
    missing = [y for y in yrs if y not in full]
    if missing:
      dropped.append((b0, b0 + n - 1, 'missing or partial years {}'.format(missing)))
    else:
      blocks.append((b0, b0 + n - 1))
  return blocks, dropped

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
  args.outdir = dcase.create_png_dir('SO_zonal_means')
  # file tag: block length, or 'full' for a single block over the whole window (length known only after reading data)
  args.tag = '{}yr'.format(args.nyears) if args.nyears else 'full'
  ncfile = os.path.join(dcase.ocn_diag_root,
                        '{}_so_zonal_means_{}_{}.nc'.format(args.casename, args.variable, args.tag))

  print('Casename is:', args.casename)
  print('netCDF file:', ncfile)

  if not args.plot_only:
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

  # thetao and so are both read, since the sigma2 contours need them; the selected variable and sigma2 are saved
  var = args.variable
  save_vars = [var, 'sigma2']

  def preprocess(ds):
    ''' Return a dataset desired variables'''
    return ds[['thetao', 'so']].isel(yh=ys)

  print('Reading dataset...')
  startTime = datetime.now()
  ds = xr.open_mfdataset(args.monthly, parallel=parallel, \
                         combine="nested", concat_dim="time", \
                         preprocess=preprocess, use_cftime=True).chunk({"time": 12})
  print('Time elasped: ', datetime.now() - startTime)

  print('Selecting data between {} and {}...'.format(args.start_date, args.end_date))
  ds_sel = ds.sel(time=slice(args.start_date, args.end_date))
  years, nmonths = np.unique(ds_sel.time.dt.year.values, return_counts=True)
  if not args.nyears:
    # single block from the first to the last complete year of the window (partial end years are ignored)
    full = years[nmonths == 12]
    if len(full) == 0:
      raise ValueError('No complete year between {} and {}'.format(args.start_date, args.end_date))
    keep = (years >= full[0]) & (years <= full[-1])
    years, nmonths = years[keep], nmonths[keep]
    args.nyears = int(full[-1] - full[0] + 1)
  blocks, dropped = make_blocks(years, nmonths, args.nyears)
  for b in dropped:
    print('Dropping incomplete block, years {}-{} ({})'.format(*b))
  if not blocks:
    raise ValueError('No complete {}-year block between {} and {}'.format(args.nyears, args.start_date, args.end_date))
  print('{} blocks of {} years:'.format(len(blocks), args.nyears), ', '.join('{}-{}'.format(*b) for b in blocks))

  z = ds_sel.z_l.values
  with xr.open_dataset(sorted(glob.glob(args.monthly))[0], decode_times=False) as d0:
    zi = d0.z_i.values
  vol = cell_volume(zi, dep, area)

  # block means = mean of the annual zonal-mean sections, computed one year at a time
  shape = (len(REGIONS), len(z), len(lat1))
  out = {v: np.full((len(blocks),) + shape, np.nan) for v in save_vars}
  wvol = None
  print('Computing zonal means...')
  startTime = datetime.now()
  for b, (b0, b1) in enumerate(blocks):
    acc = {v: np.zeros(shape) for v in out}
    cnt = {v: np.zeros(shape) for v in out}
    for y in range(b0, b1 + 1):
      dsy = ds_sel.sel(time=slice('{:04d}-01-01'.format(y), '{:04d}-12-31'.format(y)))
      assert dsy.sizes['time'] == 12, 'year {} has {} months'.format(y, dsy.sizes['time'])
      yr = weighted_temporal_mean_vars(dsy).isel(time=0).load()
      T = yr.thetao.values.astype('f8')
      S = yr.so.values.astype('f8')
      fields = {var: T if var == 'thetao' else S, 'sigma2': sigma2(T, S, z, lon, lat)}
      for v, x in fields.items():
        sec, ws = zonal_mean(x, vol, masks)
        ok = np.isfinite(sec)
        acc[v] += np.where(ok, sec, 0.)
        cnt[v] += ok
        if wvol is None and v == var: wvol = ws
      if args.debug: print('  year', y, datetime.now() - startTime)
    for v in out:
      with np.errstate(invalid='ignore', divide='ignore'):
        out[v][b] = np.where(cnt[v] > 0, acc[v] / cnt[v], np.nan)
      part = ((cnt[v] > 0) & (cnt[v] < b1 - b0 + 1)).sum()
      if part: print('  Warning: block {}-{}, {}: {} points have water in only some years'.format(b0, b1, v, part))
    print('  block {}-{} done'.format(b0, b1), datetime.now() - startTime)
  print('Time elasped: ', datetime.now() - startTime)

  dso = xr.Dataset(coords={'block': ('block', np.arange(len(blocks), dtype='i4'),
                                     {'long_name': 'Block index (0 = first block)'}),
                           'region': ('region', REGIONS, {'long_name': 'Region name'}),
                           'z_l': ('z_l', z, {'long_name': 'Depth at cell center', 'units': 'm', 'positive': 'down'}),
                           'z_i': ('z_i', zi, {'long_name': 'Depth at cell interface', 'units': 'm', 'positive': 'down'}),
                           'lat': ('lat', lat1, {'long_name': 'Row-mean latitude of wet cells', 'units': 'degrees_north'})})
  dso = dso.assign_coords(
    block_start=('block', np.array([b[0] for b in blocks], dtype='i4'),
                 {'long_name': 'First model year in block (inclusive)', 'units': 'year'}),
    block_end=('block', np.array([b[1] for b in blocks], dtype='i4'),
               {'long_name': 'Last model year in block (inclusive)', 'units': 'year'}),
    lon_west=('region', np.array([-180.] + [SECTORS[r][0] for r in REGIONS[1:]]),
              {'long_name': 'Western longitude bound of region (inclusive)', 'units': 'degrees_east'}),
    lon_east=('region', np.array([180.] + [SECTORS[r][1] for r in REGIONS[1:]]),
              {'long_name': 'Eastern longitude bound of region (exclusive, mod 360)', 'units': 'degrees_east'}))
  dims = ('block', 'region', 'z_l', 'lat')
  dso[var] = (dims, out[var].astype('f4'),
              {'long_name': 'Volume-weighted zonal-mean {}, block mean of annual means'.format(VARS[var]['long_name']),
               'units': VARS[var]['units']})
  dso['sigma2'] = (dims, out['sigma2'].astype('f4'),
                   {'long_name': 'Volume-weighted zonal-mean potential density referenced to 2000 dbar (TEOS-10, '
                                 'from 3-D annual-mean thetao and so), block mean of annual means',
                    'units': 'kg m-3 - 1000'})
  dso['wet_volume'] = (('region', 'z_l', 'lat'), wvol,
                       {'long_name': 'Sum over longitude of the cell volumes used as zonal-mean weights '
                                     '(geometric, from deptho, z_i and areacello; first year)', 'units': 'm3'})
  dso['wet_area'] = (('region', 'lat'), (area[None].astype('f8') * masks).sum(-1),
                     {'long_name': 'Sum over longitude of wet cell area (areacello)', 'units': 'm2'})
  attrs = {'description': 'Southern Ocean volume-weighted zonal-mean sections in consecutive n-year blocks. Each block '
                          'is the mean of the annual means (weighted by days in each month) of its years, computed as '
                          'the mean of the annual zonal-mean sections. sigma2 (saved for the contours) is '
                          'computed with TEOS-10 (gsw) from the '
                          '3-D annual-mean thetao and so and then zonally averaged. NaN = no water.',
           'regions': 'Circumpolar = all longitudes south of lat_max; sectors (west <= lon < east, mod 360): '
                      + '; '.join('{} {:g}..{:g}'.format(r, *SECTORS[r]) for r in REGIONS[1:])
                      + '. The five sectors tile the circumpolar region.',
           'lat_definition': 'lat is the row mean of geolat over wet cells (all longitudes); rows south of lat_max',
           'nyears': np.int32(args.nyears), 'lat_max': args.lat_max,
           'start_date': str(args.start_date), 'end_date': str(args.end_date),
           'blocks': ', '.join('{}-{}'.format(*b) for b in blocks),
           'dropped_blocks': '; '.join('{}-{} ({})'.format(*b) for b in dropped) if dropped else 'none',
           'source': args.monthly,
           'variable': var,
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

def section_figure(nrow):
  """Figure with a column of nrow section panels (shared axes) and a thin colorbar axis at the bottom."""
  fig = plt.figure(figsize=(6, 1.9 * nrow + 1.3), layout='constrained')
  gs = fig.add_gridspec(nrow + 1, 1, height_ratios=[1.] * nrow + [0.07])
  axs = []
  for r in range(nrow):
    axs.append(fig.add_subplot(gs[r], sharex=axs[0] if axs else None, sharey=axs[0] if axs else None))
    if r < nrow - 1: axs[r].tick_params(labelbottom=False)
  cax = fig.add_subplot(gs[nrow])
  return fig, axs, cax

def plot(args, ncfile):
  with xr.open_dataset(ncfile) as ds:
    _plot(args, ds.load())

def _plot(args, ds):
  v = args.variable
  st = VARS[v]
  n = int(ds.attrs['nyears'])
  late = edges(ds.lat.values)
  zi = ds.z_i.values
  nb = ds.sizes['block']
  lab = ['years {}-{}'.format(int(a), int(b)) for a, b in zip(ds.block_start.values, ds.block_end.values)]

  def panel(ax, data, s2, **kw):
    pc = ax.pcolormesh(late, zi, data, shading='flat', **kw)
    cs = ax.contour(ds.lat, ds.z_l, s2, levels=SIG2_LEVELS, colors='k', linewidths=0.6)
    ax.clabel(cs, fontsize=5, fmt='%g')
    ax.set_ylim(5500, 0)
    ax.set_xlim(late[0], late[-1])
    return pc

  for reg in ds.region.values:
    d = ds.sel(region=reg)
    # absolute sections, color levels shared by all blocks: equal-volume bins between the 2nd and 98th
    # volume-weighted percentiles south of 50S, so the weakly stratified deep ocean is not washed out
    fig, axs, cax = section_figure(nb)
    w = np.broadcast_to(d.wet_volume.values, d[v].shape)
    south = (d.lat < -50.).values if (d.lat < -50.).sum() > 2 else np.ones(d.sizes['lat'], bool)
    lev = color_levels(d[v].values[..., south], w[..., south], decimals=st['decimals'])
    norm = BoundaryNorm(lev, plt.get_cmap(st['cmap']).N, extend='both')
    for b in range(nb):
      pc = panel(axs[b], d[v].isel(block=b), d.sigma2.isel(block=b), cmap=st['cmap'], norm=norm)
      axs[b].set_ylabel('{}\ndepth (m)'.format(lab[b]), fontsize=8)
    axs[0].set_title(st['title'], fontsize=10)
    axs[-1].set_xlabel('latitude')
    cb = fig.colorbar(pc, cax=cax, orientation='horizontal', extend='both', spacing='uniform',
                      ticks=lev[::2], format='%.{}f'.format(st['decimals']))
    cb.ax.tick_params(labelsize=7)
    fig.suptitle('{} ({}): {} zonal means, {}-year blocks; contours: σ2'.format(args.casename, args.label, reg, n),
                 fontsize=10, x=0.01, ha='left')
    fname = os.path.join(args.outdir, '{}_so_zonal_means_{}_{}_{}.png'.format(args.casename, v, reg, args.tag))
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print('Saved', fname)

    # each block minus the first block, symmetric limits shared by all blocks (95th percentile of |difference|)
    if nb < 2:
      print('Only one block, skipping difference figure for', reg)
      continue
    fig, axs, cax = section_figure(nb - 1)
    dd = d[v].isel(block=slice(1, None)) - d[v].isel(block=0)
    lim = np.nanpercentile(np.abs(dd.values), 95)
    lim = lim if lim > 0 else 1.
    for b in range(1, nb):
      pc = panel(axs[b - 1], dd.isel(block=b - 1), d.sigma2.isel(block=b), cmap='RdBu_r', vmin=-lim, vmax=lim)
      axs[b - 1].set_ylabel('{}\ndepth (m)'.format(lab[b]), fontsize=8)
    axs[0].set_title(st['dtitle'], fontsize=10)
    axs[-1].set_xlabel('latitude')
    cb = fig.colorbar(pc, cax=cax, orientation='horizontal', extend='both',
                      ticks=MaxNLocator(5, symmetric=True))
    cb.ax.tick_params(labelsize=7)
    fig.suptitle('{} ({}): {} zonal means, each {}-year block minus {}; contours: σ2 of the block'.format(
                 args.casename, args.label, reg, n, lab[0]), fontsize=10, x=0.01, ha='left')
    fname = os.path.join(args.outdir, '{}_so_zonal_means_diff_{}_{}_{}.png'.format(args.casename, v, reg, args.tag))
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print('Saved', fname)

if __name__ == '__main__':
  main()
