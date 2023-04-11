import copy
import os
from glob import glob
from pathlib import Path

import dask
import kerchunk
import ujson
import xarray as xr
from kerchunk.combine import MultiZarrToZarr
from kerchunk.netCDF3 import NetCDF3ToZarr


def gen_ref(f, inline_threshold):
    refs = NetCDF3ToZarr(f, inline_threshold=inline_threshold).translate()
    try:
        # avoid empty files for last timestep
        time_ref = refs["refs"]["time/.zarray"]
        # .zarray contains null, assign to allow eval to succeed
        null = 0
        attrs = eval(time_ref)
        if attrs["shape"] == [0]:
            return {}
    except KeyError:
        pass
    return refs


def generate_references_for_stream(
    caseroot: str,
    stream: str,
    merge_static_refs: bool = True,
    inline_threshold: int = 5000,
    existing="skip",
    debug_static=False,
):
    """
    Generate Kerchunk reference JSON files for a single stream.

    Assumptions:
        - JSONs are written to the 'caseroot/jsons' folder.
        - Assumes that output files are under 'caseroot/run'

    Parameters
    ----------
    caseroot: str
        Case root directory. Assumes output is in caseroot/run/
    stream: str
        Stream name.
    merge_static_refs: bool
        If true, merge in the static file. Assumes that caseroot/run/*static*
        will find the file.
    inline_threshold: int
        Threshold, in bytes, below which array values are inlined in the JSON file.
    debug_static: bool
        If True, print the xarray repr for the static file Dataset.
        This aids in debugging

    Returns
    -------
    None
    """

    outfile = f"{caseroot}/run/jsons/{stream}.json"
    casename = caseroot.split("/")[-1]
    if os.path.exists(outfile):
        if existing == "skip":
            return
        if existing == "raise":
            raise OSError(
                "output JSON file exists. Specify existing='overwrite' or existing='skip'."
            )

    if merge_static_refs:
        (staticfile,) = glob(f"{caseroot}/run/{casename}.mom6.static.nc")
        static_refs = gen_ref(staticfile, inline_threshold)

        # The static file with time-invariant variables has a useless `time` dimension.
        # This messes up kerchunk's heuristics.
        # kerchunk.combine.drop returns a function ...
        drop_vars = kerchunk.combine.drop(("time", "Kd_bkgnd", "Kv_bkgnd"))
        static_refs = gen_ref(staticfile, inline_threshold)
        static_refs["refs"] = drop_vars(static_refs["refs"])

        if debug_static:
            ds = open_references_as_xarray(static_refs)
            display(ds)
            ds.close()

    # Get list of files
    flist = sorted(glob(f"{caseroot}/run/*mom6.{stream}_*"))

    if not flist:
        raise OSError(f"No files found for caseroot: {caseroot}, stream: {stream}")

    # generate references in parallel, one  per netcdf file.
    tasks = [dask.delayed(gen_ref)(f, inline_threshold) for f in flist]
    dicts = dask.compute(*tasks)

    # remove empty dicts for empty files
    if not dicts[-1]:
        dicts = dicts[:-1]

    # Combine multiple Zarr references (one per file) to
    # a single aggregate reference file
    mzz = MultiZarrToZarr(
        dicts, inline_threshold=inline_threshold, concat_dims="time"
    ).translate()

    # merge in the static variable references
    # TODO: this deep-copy is necessary because static_refs gets modified in-place otherwise
    if merge_static_refs:
        merged = kerchunk.combine.merge_vars([copy.deepcopy(static_refs), mzz])
    else:
        merged = mzz

    # create the output directory if needed
    Path(f"{caseroot}/run/jsons/").mkdir(parents=True, exist_ok=True)

    # write the JSON
    with open(outfile, "wb") as f:
        f.write(ujson.dumps(merged).encode())


def open_references_as_xarray(refs):
    """nice helper function for experimentation."""
    import fsspec
    import zarr

    fs = fsspec.filesystem("reference", fo=refs)
    mapper = fs.get_mapper()

    f = zarr.open(mapper)
    if len(tuple(f.groups())) > 0:
        ds = xr.open_zarr(mapper, consolidated=False, use_cftime=True)

        raise ValueError(
            "This store contains multiple groups. Use open_datatree instead."
        )

    f.close()

    return ds


def combine_stream_jsons_as_groups(caseroot, streams=None):
    ZARR_GROUP_ENTRY = {".zgroup": '{"zarr_format":2}'}

    import ujson

    newrefs = {}

    if streams is None:
        streams = [
            file.stem for file in (Path(caseroot) / "run" / "jsons").glob("*.json")
        ]

    for stream in streams:
        # read in existing JSON references
        with open(f"{caseroot}/run/jsons/{stream}.json", "rb") as f:
            d = ujson.loads(f.read())

        # Add a new group by renaming the keys
        newrefs.update({f"{stream}/{k}": v for k, v in d["refs"].items()})

    # Add top-level .zgroup entry
    newrefs.update(ZARR_GROUP_ENTRY)

    # This is now the combined dataset
    combined = {"version": 1, "refs": newrefs}

    # write a new reference JSON file
    with open(f"{caseroot}/run/jsons/combined.json", "wb") as f:
        f.write(ujson.dumps(combined).encode())
