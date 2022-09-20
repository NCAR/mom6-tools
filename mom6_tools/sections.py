import operator
from functools import reduce

import numpy as np
import xarray as xr


def read_raw_files(
    paths, debug=False, parallel=False, use_cftime=True, engine="netcdf4", **kwargs
):
    """
    Reads a list of paths as datasets. Returns a list of datasets.

    Parameters
    ----------
    paths: Iterable
        List of paths
    parallel: bool
        If True, parallelze using dask.delayed
    use_cftime: bool
        Passed to xr.open_dataset
    engine: str
        Passed to xr.open_dataset
    **kwargs:
        Passed to xr.open_dataset

    Returns
    -------
    list of Datasets
    """

    if parallel:
        import dask

        open_ = dask.delayed(xr.open_dataset)
        chunks = {"zl": 40, "zi": 40}
    else:
        open_ = xr.open_dataset
        chunks = None

    raw_dsets = [
        open_(ff, use_cftime=use_cftime, engine=engine, chunks=chunks, **kwargs)
        for ff in sorted(paths)
    ]

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

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset

    Returns
    -------
    xarray.Dataset
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


def cfconcat(dsets, axis, join="outer", **kwargs):
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
    import tqdm

    assert ndimlist(dsets) == 1
    combined = xr.Dataset()
    allvars = set(reduce(operator.add, (list(ds.data_vars) for ds in dsets)))

    for var in tqdm.tqdm(allvars):
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

        # I don't like this join="outer"
        combined[var] = xr.concat(arrays, dim=dim, join=join, **kwargs)

    return combined


def visualize_tile(da):
    """
    Draws bounding box for a DataArray

    Parameters
    ----------
    da: xarray.DataArray
        should have variables with X and Y axis attributes

    Returns
    -------
    None
    """
    import cf_xarray
    import matplotlib.pyplot as plt

    x = da.cf["X"]
    y = da.cf["Y"]

    xl, xh = x.min().item(), x.max().item()
    yl, yh = y.min().item(), y.max().item()

    plt.plot(
        [xl, xl, xh, xh, xl],
        [yl, yh, yh, yl, yl],
        lw=2,
    )


def trim_row(raw_files, axis, debug=False):
    """
    Trims overlapping index values along rows of a nested list.

    Parameters
    ----------
    raw_files: list of lists of xarray Datasets
        Nested list of xarray Datasets
    axis : str
        Dataset axis along which to trim overlapping points.
        Passed to .cf.axes
    debug: bool
        If True print labels being dropped.

    Returns
    -------
    list of lists of xarray Datasets
        Datasets in a row do not have overlapping index values
    """
    trimmed = np.zeros(shape=shape(raw_files)).tolist()
    for irow, row in enumerate(raw_files):
        N = len(row)
        for i0, i1 in zip(range(N - 1), range(1, N)):
            ds0 = row[i0]  # [[varname]]
            ds1 = row[i1]  # [[varname]]
            drop_labels = {}
            for dim in ds0.cf.axes[axis]:
                bad = ds0.indexes[dim].intersection(ds1.indexes[dim])
                if not bad.empty:
                    drop_labels[dim] = bad

            if debug:
                print(drop_labels)
            trimmed[irow][i0] = ds0.drop_sel(drop_labels)

            # sanity check
            for dim in ds0.cf.axes[axis]:
                bad = ds1.indexes[dim].intersection(trimmed[irow][i0].indexes[dim])
                assert bad.empty

        trimmed[irow][-1] = raw_files[irow][-1]

    return trimmed


def combine_manual(raw_files, debug=False):
    """
    Manualy combines files like xr.combine_nested.

    The difference is that this function de-duplicates along X, Y axes.
    (the interface points on a tile boundary are written to two files)

    Parameters
    ----------
    raw_files: Iterable
        An Interable of xarray Datasets
    debug : bool
        If True, visualize the inputs using visualize_tile

    Returns
    -------
    xarray.Dataset
    """
    import cf_xarray

    trimmed = trim_row(raw_files, axis="X", debug=debug)
    trimmed = transpose(trim_row(transpose(trimmed), axis="Y", debug=debug))

    if debug:
        [visualize_tile(ds.uo) for ds in trimmed.flat]

    kwargs = dict(coords="minimal", compat="override")

    first = trimmed[0][0]
    combined = xr.Dataset()
    for var in first:
        axes = first[var].cf.axes
        if "X" not in axes and "Y" not in axes:
            continue
        X = axes["X"]
        Y = axes["Y"]

        assert len(X) == 1
        assert len(Y) == 1

        X = X[0]
        Y = Y[0]

        for row in trimmed:
            assert all([ds.indexes[X].is_monotonic for ds in row])
            assert all([ds.indexes[X].is_unique for ds in row])

        combined[var] = xr.concat(
            [xr.concat([ds[var] for ds in row], dim=X, **kwargs) for row in trimmed],
            dim=Y,
            **kwargs,
        )

    return combined


def tile_raw_files(subset, x, y):
    """
    Takes a 1D list (subset) and reshapes as 2D 'y' rows by 'x' columns
    list of lists
    """
    import toolz

    # Don't knopw how to make the object array of Datasets work
    # raw_files = np.empty(len(subset), dtype="O")
    # raw_files[:] = subset
    # raw_files = raw_files.reshape(y, x)

    raw_files = list(toolz.partition(x, subset))
    assert len(raw_files) == y, (len(raw_files), y)
    return raw_files


def ndimlist(seq):
    """ndim for nested lists. copied from dask."""
    if not isinstance(seq, (list, tuple)):
        return 0
    elif not seq:
        return 1
    else:
        return 1 + ndimlist(seq[0])


def shape(seq):
    """shape of nested lists"""
    s = tuple()
    if isinstance(seq[0], (list, tuple)):
        s = shape(seq[0])
    s = (len(seq),) + s
    assert len(s) == ndimlist(seq)
    return s


def transpose(seq):
    """transpose nested lists"""
    assert ndimlist(seq) > 1
    return list(map(list, zip(*seq)))
