import copy
import os
from glob import glob
from pathlib import Path

import dask
import kerchunk
import ujson
from kerchunk.combine import MultiZarrToZarr
from kerchunk.netCDF3 import NetCDF3ToZarr


def generate_references_for_stream(
    caseroot: str,
    stream: str,
    merge_static_refs: bool = True,
    inline_threshold: int = 5000,
    existing="skip",
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

    Returns
    -------
    None
    """

    def gen_ref(f):
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
        return  refs

    outfile = f"{caseroot}/run/jsons/{stream}.json"
    if os.path.exists(outfile):
        if existing == "skip":
            return
        if existing == "raise":
            raise OSError(
                "output JSON file exists. Specify existing='overwrite' or existing='skip'."
            )

    if merge_static_refs:
        (staticfile,) = glob(f"{caseroot}/run/*static*")
        static_refs = gen_ref(staticfile)

        # The static file with time-invariant variables has a useless `time` dimension.
        # This messes up kerchunk's heuristics.
        # kerchunk.combine.drop returns a function ...
        drop_time = kerchunk.combine.drop("time")
        static_refs = drop_time(gen_ref(staticfile))

    # Get list of files
    flist = sorted(glob(f"{caseroot}/run/*mom6.{stream}_*"))

    if not flist:
        raise OSError(f"No files found for caseroot: {caseroot}, stream: {stream}")

    # generate references in parallel, one  per netcdf file.
    tasks = [dask.delayed(gen_ref)(f) for f in flist]
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
