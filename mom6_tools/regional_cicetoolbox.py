"""
CICE sea-ice and floe-size-distribution (FSD) functions used by the regional
diagnostics notebooks in ``mom6_tools/nb_templates/regional_notebooks``.

Like ``regional_m6toolbox``, every function here also appears, verbatim, in the
notebook that uses it, so a notebook can run as a standalone document.

Data notes:

* The FSD bin dimension on file is ``nf`` (12), and the bin edges are not
  written to the history files. They are hard-coded below from icepack_fsd.F90
  for nfsd=12. ``afsd`` is a density: area fraction in bin k is afsd[k]*dr[k].
* Every ``dafsd_*`` tendency is written WITHOUT the bin-width division, units
  "1/s", so it is summed over bins directly -- never multiplied by dr.
"""

import numpy as np
import xarray as xr


# --- Floe-size bins (icepack_fsd.F90, nfsd=12): 13 edges, metres -----------

FSD_LIMS = np.array([
    6.65000000e-02, 5.31030847e+00, 1.42865861e+01, 2.90576686e+01,
    5.24122136e+01, 8.78691405e+01, 1.39518470e+02, 2.11635752e+02,
    3.08037274e+02, 4.31203059e+02, 5.81277225e+02, 7.55141047e+02,
    9.45812834e+02])
FLOE_RAD_C = 0.5 * (FSD_LIMS[:-1] + FSD_LIMS[1:])   # bin centres, m
FLOE_BINWIDTH = np.diff(FSD_LIMS)                   # dr, m

ICE_EDGE = 0.15      # conventional ice edge (aice)
SMALL_FLOE = 29.0    # m -- the three smallest bins; the break-up end of the spectrum


def open_cice(paths, suffix="", chunks={"time": 1}):
    """Open CICE history files on the MOM6 tracer-grid dimension names.

    CICE and MOM6 share the tracer grid in these regional cases, so nj/ni are
    renamed to yh/xh and the notebook's COORDS / MAP["h"] apply unchanged.
    `suffix` is the stream suffix CICE appends to every field ("_d" for a daily
    h1 stream, "" for a lone monthly stream); it is stripped so the notebook
    can use plain names like "aice".
    """
    ds = xr.open_mfdataset(paths, combine="by_coords", chunks=chunks,
                           decode_timedelta=False)
    if suffix:
        ds = ds.rename({v: v[:-len(suffix)] for v in ds.data_vars if v.endswith(suffix)})
    return ds.rename({"nj": "yh", "ni": "xh"})


def small_floe_fraction(ice):
    """Area fraction in floes smaller than SMALL_FLOE, masked to ice-covered cells."""
    k = np.where(FLOE_RAD_C < SMALL_FLOE)[0]
    dr = xr.DataArray(FLOE_BINWIDTH[k], dims="nf")
    return (ice["afsd"].isel(nf=k) * dr).sum("nf").where(ice["aice"] > ICE_EDGE)


def fracturing_rate(ice):
    """Positive part of the wave FSD tendency, summed over bins [s-1].

    Not weighted by bin width: dafsd_wave is already a per-bin rate (see the
    module notes), and weighting by dr breaks its area conservation.
    """
    d = ice["dafsd_wave"]
    return d.where(d > 0, 0.0).sum("nf")
