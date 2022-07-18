import operator
from functools import reduce

import numpy as np
import xarray as xr


def read_raw_files(paths, debug=False, parallel=False):
    """Reads a list of paths as datasets. Returns a list of datasets."""

    if parallel:
        import dask

        open_ = dask.delayed(xr.open_dataset)
        chunks = {"zl": 40, "zi": 40}
    else:
        open_ = xr.open_dataset
        chunks = None

    raw_dsets = [open_(ff, use_cftime=True, chunks=chunks) for ff in sorted(paths)]

    if parallel:
        raw_dsets = dask.compute(*raw_dsets)

    raw_files_list = [preprocess_mom6_sections(ds) for ds in raw_dsets]

    return raw_files_list


def preprocess_mom6_sections(ds):
    """
    Preprocess function for reading MOM6 field section output.

    Gets rid of dummy dimensions ("xh_sub01") and renames them to
    standard names ("xh")

    Intended use is the ``preprocess`` kwarg of xarray's ``open_mfdataset``.
    """

    # "standard" dimension names
    dims = ["xh", "xq", "yh", "yq"]

    for dim in dims:
        matches = [dsdim for dsdim in ds.dims if dim in dsdim]
        if not matches:
            continue
        if len(matches) > 1:
            [
                np.testing.assert_equal(
                    ds.coords[matches[0]].values, ds.coords[other].values
                )
                for other in matches[1:]
            ]

        ds = ds.rename({matches[0]: dim})
        for match in matches[1:]:
            ds = ds.drop_vars(match).rename({match: dim})

    return ds


def ndimlist(seq):
    """copied from dask."""
    if not isinstance(seq, (list, tuple)):
        return 0
    elif not seq:
        return 1
    else:
        return 1 + ndimlist(seq[0])


def combine_nested(raw_files, concat_dim, debug=False, **kwargs):
    """
    CF-aware combine_nested.


    Parameters
    ----------

    raw_files: list
        Possibly nested list of datasets to concatenate
    concat_dim: str or list of str
        Dimensons to concatenate along. Can be cf-xarray names
        like "X", "longitude" etc.

    Returns
    -------

    Dataset
    """
    assert "join" not in kwargs
    kwargs = dict(coords="minimal", compat="override")

    if isinstance(concat_dim, str):
        concat_dim = (concat_dim,)

    assert ndimlist(raw_files) == len(concat_dim)

    if isinstance(raw_files[0], (list, tuple)):
        raw_files = [combine_nested(seq, concat_dim[1:]) for seq in raw_files]

    assert isinstance(raw_files[0], (xr.Dataset, xr.DataArray))
    return cfconcat(raw_files, concat_dim[0], **kwargs)


def cfconcat(dsets, axis, **kwargs):
    """
    Concat a 1D list of datasets along axis.

    Parameters
    ----------
    dsets: list of Datasets
        A 1D list of xarray Datasets
    axis: str
        A single axis or dimension to concatenate along. Allows
        cf_xarray names like 'X', 'Y' etc. For example axis="X"
        will concatenate along the appropriate "X" axis for the
        variable.

    Returns
    -------
    Dataset
    """

    import cf_xarray

    assert ndimlist(dsets) == 1
    combined = xr.Dataset()
    allvars = set(reduce(operator.add, (list(ds.data_vars) for ds in dsets)))

    for var in allvars:
        # TODO: clean this up
        try:
            var_ = dsets[0][var]
        except KeyError:
            try:
                var_ = dsets[1][var]
            except KeyError:
                raise NotImplementedError(
                    f"Don't know how to combine variables {var} not present in the first two datasets."
                )
        try:
            dims = var_.cf.axes[axis]
        except KeyError:
            combined[var] = var_
            continue

        assert len(dims) == 1
        (dim,) = dims

        arrays = [ds[var] for ds in dsets if var in ds]
        # print(f"{dim}: {xr.concat([array[dim] for array in arrays], dim=dim).data}")

        # At this point we really only want `axis` index values to be different.
        # So set join="exact"
        combined[var] = xr.concat(arrays, dim=dim, join="outer", **kwargs)

    return combined
