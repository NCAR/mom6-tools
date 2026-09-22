"""
Functions used by the regional diagnostics notebooks in
``mom6_tools/nb_templates/regional_notebooks``.

Every function here also appears, verbatim, in the notebook that uses it, so
that a notebook can be run as a standalone document. The notebooks import the
module copy in the cell where each function is used, so that any single cell
can be run without first running the cell that defines it.

Nothing here reads a notebook global: the grid geometry, the topography, the
GLORYS root and the calendar are all passed in.
"""

import glob
import os
import re
from pathlib import Path

import cmocean.cm  # noqa: F401  -- registers the "cmo.*" colormaps
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xgcm

from mom6_tools.wright_eos import wright_eos


# --- xgcm grid and derived quantities -------------------------------------

XGCM_COORDS = {
    "X": {"center": "xh", "outer": "xq"},
    "Y": {"center": "yh", "outer": "yq"},
    "Z": {"center": "zl", "outer": "zi"},
}


def make_grid(ds, geom, with_metrics=True, xgcm_coords=XGCM_COORDS):
    """Build an xgcm.Grid for a MOM6 regional dataset, as (grid, ds_with_metrics).

    Metrics come from the ocean_geometry file `geom` so that grid.diff /
    grid.interp / grid.derivative / grid.integrate know the cell spacings.
    """
    coords = {ax: pos for ax, pos in xgcm_coords.items() if pos["center"] in ds.dims} # flexible for 3D/2D data
    obj, metrics = ds, None

    if with_metrics:
        for v in ("dxCu", "dyCu", "dxCv", "dyCv", "dxT", "dyT",
                  "dxBu", "dyBu", "Ah", "Aq"):
            if v in geom and set(geom[v].dims) <= set(ds.dims):
                obj = obj.assign_coords({v: geom[v]})
        metrics = {
            ("X",): [v for v in ("dxT", "dxCu", "dxCv", "dxBu") if v in obj.coords],
            ("Y",): [v for v in ("dyT", "dyCu", "dyCv", "dyBu") if v in obj.coords],
            ("X", "Y"): [v for v in ("Ah", "Aq") if v in obj.coords],
        }
        metrics = {k: v for k, v in metrics.items() if v}

    grid = xgcm.Grid(obj, coords=coords, metrics=metrics, padding="extend",
                autoparse_metadata=False)

    return grid, obj


def decode_mom6_time(ds, calendar="proleptic_gregorian"):
    """Decode MOM6 time axes, correcting the calendar that MOM6 mislabels.
    """
    ds = ds.copy()
    for v in ds.variables:
        attrs = dict(ds[v].attrs)
        if "since" in str(attrs.get("units", "")):
            attrs["calendar"] = calendar
            ds[v].attrs = attrs
    return xr.decode_cf(ds)


def relative_vorticity(grid, ds, u="uo", v="vo"):
    """Relative vorticity at the corner (q) point [s-1], the MOM6 discretisation:

        zeta = ( d/dx (v * dyCv) - d/dy (u * dxCu) ) / Aq
    """
    dvdx = grid.diff(ds[v] * ds["dyCv"], "X", padding="fill", fill_value=0.0)
    dudy = grid.diff(ds[u] * ds["dxCu"], "Y", padding="fill", fill_value=0.0)
    zeta = (dvdx - dudy) / ds["Aq"]
    zeta.name = "zeta"
    zeta.attrs = {"long_name": "Relative vorticity", "units": "s-1"}
    return zeta


def layer_depths(e):
    """Layer-centre depths (positive down, m) from interface heights `e`."""
    z = -0.5 * (e.isel(zi=slice(None, -1)).values + e.isel(zi=slice(1, None)).values)
    dims = [d if d != "zi" else "zl" for d in e.dims]
    coords = {k: v for k, v in e.coords.items() if "zi" not in v.dims}
    out = xr.DataArray(z, dims=dims, coords=coords, name="z_center")
    out.attrs = {"long_name": "Layer centre depth", "units": "m", "positive": "down"}
    return out


def density(T, S, p=0.0):
    """In-situ density [kg m-3] from the MOM6 Wright (1997) EOS.

    T in degC, S in psu, p in Pa.  T and S must be xarray objects (wrap a T/S
    mesh in xr.DataArray when contouring sigma0).  Uses mom6_tools.wright_eos.
    """
    rho = xr.apply_ufunc(wright_eos, T, S, p, dask="allowed", keep_attrs=False)
    rho.name = "rho"
    rho.attrs = {"long_name": "In-situ density (Wright 1997 EOS)", "units": "kg m-3"}
    return rho


def sigma0(T, S):
    """Potential density anomaly referenced to 0 dbar [kg m-3]."""
    s = density(T, S, 0.0) - 1000.0
    s.name = "sigma0"
    s.attrs = {"long_name": "Potential density anomaly (sigma-0)", "units": "kg m-3"}
    return s


def save_gif(frame_paths, out_path, duration=350, loop=0, clean = False):
    """Stitch saved PNG frames into an animated GIF with Pillow.
    """
    from PIL import Image

    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE)
              for p in frame_paths]
    out_path = Path(out_path)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=loop, optimize=True)

    if clean:
        for p in frame_paths:
            os.remove(p)

    return out_path


def save_field_gif(field, out_path, subplot_kw, map_kw, figsize, land=None,
                   gridlines_kw=None, cmap="cmo.thermal", coastlines="10m",
                   title=None, dpi=90, duration=100, loop=0, clean=False):
    """One PNG per time record of `field`, stitched into an animated GIF.

    `field` is a map-shaped DataArray with a `time` dim, already masked and
    loaded.  `subplot_kw`, `map_kw`, `figsize`, `land` and `gridlines_kw` are the
    notebook's SUBPLOT, MAP[...], PANEL, LAND and GRID.

    The colour range is taken once over the whole field and the figure size is
    fixed, so every frame comes out on the same scale and the same size.  Frames
    are written beside `out_path`, and `clean` removes them once stitched.
    """
    out_path = Path(out_path)
    vmin, vmax = float(field.min()), float(field.max())
    label = field.name if title is None else title

    frames = []
    for i in range(field.sizes["time"]):
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=subplot_kw)
        field.isel(time=i).plot(ax=ax, **map_kw, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.coastlines(coastlines)
        if land is not None:
            ax.add_feature(land, facecolor="0.85")
        if gridlines_kw is not None:
            ax.gridlines(**gridlines_kw)
        fig.suptitle(f"{label}   {str(field.time.values[i])[:10]}")
        p = out_path.with_name(f"{out_path.stem}_frame_{i:03d}.png")
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        frames.append(p)

    return save_gif(frames, out_path, duration=duration, loop=loop, clean=clean)


# --- Reading an OBC segment file ------------------------------------------
# Everything in forcing_obc_segment_*.nc is named after its segment
# (`temp_segment_001`, `nz_segment_001_temp`, `lon_segment_001`, ...), and the
# boundary runs along whichever of `nx_segment_*` / `ny_segment_*` is longer than
# one point.  `open_obc` strips all of that down to dims (time, depth, n).

OBC_VARS = ("temp", "salt", "u", "v", "eta")


def open_obc(path):
    """Read one OBC segment file as a Dataset with dims (time, depth, n).

    `n` counts points along the boundary; `lon`, `lat` and `depth` come along as
    coordinates.  The attributes record the segment id and its orientation:
    "zonal" = constant latitude, "meridional" = constant longitude, with
    `const_coord` holding that constant value.
    """
    sid = re.search(r"segment_(\d+)\.nc$", path).group(1)
    raw = xr.open_dataset(path).squeeze(drop=True)   # drops the 1-point cross-boundary dim
    along = (f"nx_segment_{sid}" if raw.sizes.get(f"nx_segment_{sid}", 1) > 1
             else f"ny_segment_{sid}")

    def tidy(name):
        """One field, with its segment-specific dim names replaced by n / depth."""
        da = raw[f"{name}_segment_{sid}"]
        return da.rename({along: "n",
                          **{d: "depth" for d in da.dims if d.startswith("nz")}})

    ds = xr.Dataset({v: tidy(v) for v in OBC_VARS}).assign_coords(
        n=np.arange(raw.sizes[along]),
        depth=raw["depth"].values,
        lon=("n", raw[f"lon_segment_{sid}"].values.ravel()),
        lat=("n", raw[f"lat_segment_{sid}"].values.ravel()),
    )

    # The files label their time axis "julian" or "gregorian"; MOM6 reads the
    # dates at face value, so convert date-for-date onto the history calendar.
    ds = ds.convert_calendar("proleptic_gregorian", use_cftime=False)

    zonal = float(np.ptp(ds["lat"].values)) < 1e-6
    ds.attrs.update(segment=sid,
                    orientation="zonal" if zonal else "meridional",
                    const_coord=float(ds["lat"][0] if zonal else ds["lon"][0]))
    return ds


def obc_label(ds):
    """Human-readable label for a segment dataset."""
    key = "lat" if ds.attrs["orientation"] == "zonal" else "lon"
    return f"segment {ds.attrs['segment']} ({key} = {ds.attrs['const_coord']:g})"


def along_name(ds):
    """Name of the coordinate that varies along the segment."""
    return "lon" if ds.attrs["orientation"] == "zonal" else "lat"


def normal_var(ds):
    """The velocity component normal to the segment."""
    return "v" if ds.attrs["orientation"] == "zonal" else "u"


# --- Putting a segment on the model's tracer cells ------------------------
# The boundary points are *supergrid* points, at twice the resolution of the
# model grid: a segment of 2N+1 points spans N tracer cells, and points
# 1, 3, 5, ... sit exactly on the cell centres.  So every other point lines a
# segment up one-for-one with a row (or column) of model h-points, and nothing
# has to be interpolated in space anywhere below.


def to_t_points(seg):
    """Every other boundary point: the segment on the model's tracer cells."""
    out = seg.isel(n=slice(1, None, 2))
    return out.assign_coords(n=np.arange(out.sizes["n"]))


def on_boundary(obj, seg, topog):
    """Slice a t-point field (`topog` itself, or a history stream) along a segment.

    Returns the row (or column) of t-cells that lies along the boundary,
    relabelled with the segment's own `n`, `lon` and `lat` so that it can be
    differenced against the segment directly.  (Those are the boundary-edge
    positions; the cell centres sit half a cell inside.)

    `topog` is the ocean_topog dataset the OBC files were generated against,
    renamed onto the history-file dimension names; its `x`/`y` are what the
    boundary index is looked up in.
    """
    if seg.attrs["orientation"] == "zonal":       # constant latitude -> one yh row
        dim = "yh"
        k = int(np.abs(topog["y"].isel(xh=0) - seg.attrs["const_coord"]).argmin("yh"))
    else:                                         # constant longitude -> one xh column
        dim = "xh"
        k = int(np.abs(topog["x"].isel(yh=0) - seg.attrs["const_coord"]).argmin("xh"))
    along = "xh" if dim == "yh" else "yh"

    out = obj.isel({dim: k}).rename({along: "n"})
    if out.sizes["n"] != seg.sizes["n"]:
        raise ValueError(
            f"{obc_label(seg)} has {seg.sizes['n']} tracer cells but the boundary "
            f"row has {out.sizes['n']} -- does the segment span the whole edge?")
    out.attrs["boundary_line"] = f"{dim} index {k}"
    return out.assign_coords(n=seg["n"].values,
                             lon=("n", seg["lon"].values),
                             lat=("n", seg["lat"].values))


def mask_below_floor(seg, floor):
    """Blank the levels of a segment that lie below the sea floor, and land points.

    The OBC files carry no bathymetry: every level at every point holds a value,
    so on a shallow shelf most of a 50-level GLORYS-derived file is extrapolated
    fill -- small_alaska is 121 m deep but its boundary files run to 5728 m.
    """
    wet = floor["mask"] > 0
    out = seg[["temp", "salt", "u", "v"]].where(wet & (seg["depth"] < floor["depth"]))
    out["eta"] = seg["eta"].where(wet)             # eta is 2-D, so only the land mask
    out.attrs = dict(seg.attrs)
    return out


# --- GLORYS12 access ------------------------------------------------------
# One global daily-mean file per day, ~1.3 GB each, read straight off /gdex.

GLORYS_VARS = ("thetao", "so", "zos")


def glorys_files(dates, root):
    """The GLORYS12 daily-mean file for each 'YYYY-MM-DD' in `dates`, under `root`."""
    files = []
    for d in dates:
        y, m, dd = d.split("-")
        hits = sorted(glob.glob(f"{root}/{y}/*_{y}{m}{dd}_*.nc"))
        if not hits:
            raise FileNotFoundError(f"no GLORYS file for {d} under {root}/{y}")
        files.append(hits[0])
    return files


def open_glorys(dates, geom, root, pad=0.25):
    """Open GLORYS12 daily means over the model's lon/lat box, surface level only.

    `open_mfdataset` reads the coordinates but leaves the fields as dask arrays,
    so the `.sel` below is what decides how much actually comes off disk -- no
    per-file preprocessing needed.  Longitudes are put on the same convention as
    the model grid, which is what makes a dateline-straddling domain work.
    """
    ds = xr.open_mfdataset(glorys_files(dates, root), combine="by_coords", chunks={"time": 1})
    ds = ds[list(GLORYS_VARS)].isel(depth=0, drop=True)

    if float(geom.geolon.max()) > 180.0:            # model grid runs 0..360
        ds = ds.assign_coords(longitude=ds["longitude"] % 360).sortby("longitude")

    return ds.sel(
        longitude=slice(float(geom.geolon.min()) - pad, float(geom.geolon.max()) + pad),
        latitude=slice(float(geom.geolat.min()) - pad, float(geom.geolat.max()) + pad),
    )


# --- Comparison helpers: regridding and error statistics ------------------


def to_model_grid(obj, geom):
    """Sample a GLORYS field (1-D latitude/longitude) onto the model h points.

    `geolon`/`geolat` are 2-D with dims (yh, xh), so passing them as the
    indexers makes `interp` do *pointwise* (advanced) interpolation -- one
    bilinear sample per model cell centre, rather than an outer product.  This
    is how we regrid onto a curvilinear grid without xesmf.
    """
    return (obj.interp(longitude=geom.geolon, latitude=geom.geolat)
               .drop_vars(["longitude", "latitude"], errors="ignore")
               .assign_coords(geolon=geom.geolon, geolat=geom.geolat))


def common_mask(a, b):
    """Restrict a model/observation pair to the cells where both are valid."""
    both = a.notnull() & b.notnull()
    return a.where(both), b.where(both)


def area_stats(mod, obs, area):
    """Area-weighted bias, RMSE and pattern (centred) correlation of `mod` vs `obs`.

    `mod` and `obs` have already been through `common_mask`, so xarray's
    `.weighted()` reductions skip exactly the same cells in both.
    """
    w = area.fillna(0.0)                       # .weighted() rejects NaN weights
    mbar, obar = mod.weighted(w).mean(), obs.weighted(w).mean()
    ma, oa = mod - mbar, obs - obar
    r = ((ma * oa).weighted(w).mean()
         / np.sqrt((ma ** 2).weighted(w).mean() * (oa ** 2).weighted(w).mean()))
    return {"n_cells": int(mod.notnull().sum()),
            "model_mean": float(mbar), "glorys_mean": float(obar),
            "bias": float(mbar - obar),
            "rmse": float(np.sqrt(((mod - obs) ** 2).weighted(w).mean())),
            "pattern_r": float(r)}


def ts_axis(objs, name, dim, n=60):
    """A 1-D axis spanning the combined range of `name` over the datasets `objs`.

    Two of these, on different dims, broadcast against each other into the T-S
    mesh that `sigma0` is contoured on.
    """
    v = np.linspace(min(float(o[name].min()) for o in objs),
                    max(float(o[name].max()) for o in objs), n)
    return xr.DataArray(v, dims=dim, coords={dim: v})
