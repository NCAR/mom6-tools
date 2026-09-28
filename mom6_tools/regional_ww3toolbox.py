"""
WW3 wave functions used by the regional diagnostics notebooks in
``mom6_tools/nb_templates/regional_notebooks``.

Like ``regional_m6toolbox``, every function here also appears, verbatim, in the
notebook that uses it, so a notebook can run as a standalone document.

Data notes:

* WW3 writes no gridded history unless HIST_OPTION/HIST_N are set, so wave
  fields come from the CMEPS mediator stream ``<case>.cpl.hx.ww3.*.nc``
  (``wavImp_Sw_*`` on ``wavImp_ny``/``wavImp_nx``), which is on the same grid
  as MOM6 and CICE.
* The mediator records are period means stamped at the END of the period, and
  their ``time_bnds`` are degenerate ([t, t]), so the stamps are moved back to
  the middle of each period on open.
* CICE's own ``wave_sig_ht`` can carry ~1e15 artifacts; WAVE_MAX masks them
  when it is used as the fallback.
"""

import xarray as xr


WAVE_MAX = 100.0     # m -- guard for the CICE wave_sig_ht artifact


def open_ww3_mediator(paths, chunks={"time": 1}):
    """Open the CMEPS mediator wave stream on the MOM6 tracer-grid dimension names,
    with each record re-stamped at the middle of its averaging period."""
    wav = xr.open_mfdataset(paths, combine="by_coords", chunks=chunks,
                            decode_timedelta=False)
    wav = wav.rename({"wavImp_ny": "yh", "wavImp_nx": "xh"})
    dt = wav["time"].diff("time").median().values
    return wav.assign_coords(time=wav["time"].values - dt / 2)


def match_to_ice(wav, ice):
    """Average the wave records onto each ice record's time_bounds window.

    Daily waves against monthly ice become monthly means; against daily ice
    each window holds one record. Returned on the ice time axis, so ice*wave
    expressions align.
    """
    bnds = ice["time_bounds"].values
    out = [wav.sel(time=slice(t0, t1)).mean("time") for t0, t1 in bnds]
    return xr.concat(out, dim="time").assign_coords(time=ice["time"].values)


def significant_wave_height(ice, wav):
    """Significant wave height [m]: the mediator's wavImp_Sw_hs_avg when present,
    otherwise CICE's in-ice wave_sig_ht with the artifact masked."""
    if wav is not None and "wavImp_Sw_hs_avg" in wav:
        return wav["wavImp_Sw_hs_avg"]
    h = ice["wave_sig_ht"]
    return h.where(h < WAVE_MAX)
