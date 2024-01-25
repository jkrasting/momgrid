""" compare.py : module for comparing objects on MOM grids """

__all__ = ["compare", "emplace_arrays"]

import xarray as xr
import momgrid


def compare(arr1, arr2, plot_type="yx", tdim="time", projection=None):
    var = arr1.name
    ds1, ds2 = emplace_arrays(arr1, arr2, plot_type=plot_type, tdim=tdim)

    fig = momgrid.plot.compare_2d(
        ds1[var],
        ds2[var],
        dpi=90,
        singlepanel=True,
        plot_type=plot_type,
        projection=projection,
        clim_diff=(-4.5, 4.5),
    )

    return fig


def emplace_arrays(arr1, arr2, plot_type="yx", tdim="time"):
    assert isinstance(arr1, xr.DataArray) and isinstance(arr2, xr.DataArray)

    ds1 = xr.Dataset({arr1.name: arr1})
    ds2 = xr.Dataset({arr1.name: arr2})

    ds1 = momgrid.Gridset(ds1)
    ds2 = momgrid.Gridset(ds2)

    # If an xy plot is requested, extract the desired depth level
    if plot_type == "yx":
        try:
            ds1.data = momgrid.util.extract_level(ds1.data, level)
            ds2.data = momgrid.util.extract_level(ds2.data, level)
        except:
            pass

    if tdim in ds1.data.dims:
        ds1.data = ds1.data.mean(tdim, keep_attrs=True)

    if tdim in ds2.data.dims:
        ds2.data = ds2.data.mean(tdim, keep_attrs=True)

    # Check if grids are identical and regrid if necessary
    if ds1.model == ds2.model:
        ds1 = ds1.data
        ds2 = ds2.data
    else:
        ds1 = ds1.regrid(resolution=resolution)
        ds2 = ds2.regrid(resolution=resolution)

    if plot_type == "yz":
        ds1 = momgrid.util.x_average_dataset(ds1, region=region)
        ds2 = momgrid.util.x_average_dataset(ds2, region=region)

    return (ds1, ds2)
