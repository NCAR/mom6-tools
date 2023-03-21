import copy
from glob import glob
from pathlib import Path

import dask.bag
import kerchunk
import ujson
from kerchunk.combine import MultiZarrToZarr
from kerchunk.netCDF3 import NetCDF3ToZarr


def generate_references_for_stream(
    caseroot: str,
    stream: str,
    merge_static_refs: bool = True,
    inline_threshold: int = 5000,
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
        return NetCDF3ToZarr(f, inline_threshold=inline_threshold).translate()

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

    # parallelize generating references using dask.bag
    # Alternatively this could be dask.delayed
    bag = dask.bag.from_sequence(flist, npartitions=len(flist)).map(gen_ref)
    dicts = bag.compute()

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
    with open(f"{caseroot}/run/jsons/{stream}.json", "wb") as f:
        f.write(ujson.dumps(merged).encode())
