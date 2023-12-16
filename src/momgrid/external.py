""" external.py: functions to interface with external packages """

import warnings
import xarray as xr
import xesmf as xe
import pkg_resources as pkgr
from momgrid.util import is_symmetric

__all__ = ["build_regridder_weights", "static_to_xesmf", "woa18_grid"]


def build_regridder_weights(src, dst, periodic=True):
    """Function to generate pre-calculated xesmf weight files"""

    methods = ["bilinear", "nearest_s2d", "nearest_d2s"]

    if (
        set(["lat_b", "lon_b"]).issubset(
            list(src.keys()) + list(src.coords) + list(src.dims)
        )
    ) and (
        set(["lat_b", "lon_b"]).issubset(
            list(dst.keys()) + list(dst.coords) + list(dst.dims)
        )
    ):
        bounds = True
        methods = methods + ["conservative", "conservative_normed", "patch"]
    else:
        bounds = False

    files = []
    for method in methods:
        _ = xe.Regridder(src, dst, method, periodic=periodic)
        _ = _.to_netcdf()
        files.append(_)

    return files


def static_to_xesmf(dset, grid_type="t", filename=None):
    """Function to convert a MOM6 static file to one that can be
    fed into xesmf routines.

    Parameters
    ----------
    dset : xarray.Dataset
        MOM6 static file dataset
    grid_type : str
        Grid type (t,u,v,c), optional. By default "t"

    Returns
    -------
    xarray.Dataset
        Xarray dataset to compatible with xesmf
    """

    assert isinstance(dset, xr.Dataset), "Input must be an xarray dataset."

    if grid_type == "t":
        dsout = xr.Dataset(
            {
                "lat": dset.geolat,
                "lon": dset.geolon,
                "mask": dset.wet,
            }
        )

        if is_symmetric(dset):
            dsout["lat_b"] = dset.geolat_c
            dsout["lon_b"] = dset.geolon_c
        else:
            warnings.warn("Grid is not symmetric, skipping bounds")

    elif grid_type == "u":
        dsout = xr.Dataset(
            {
                "lat": dset.geolat_u,
                "lon": dset.geolon_u,
            }
        )

        if "wet_u" in dset.keys():
            dsout["mask"] = dset.wet_u
        else:
            warnings.warn("Wet mask not present.")

    elif grid_type == "v":
        dsout = xr.Dataset(
            {
                "lat": dset.geolat_v,
                "lon": dset.geolon_v,
            }
        )

        if "wet_v" in dset.keys():
            dsout["mask"] = dset.wet_v
        else:
            warnings.warn("Wet mask not present.")

    elif grid_type == "c":
        dsout = xr.Dataset(
            {
                "lat": dset.geolat_c,
                "lon": dset.geolon_c,
            }
        )

        if "wet_c" in dset.keys():
            dsout["mask"] = dset.wet_c
        else:
            warnings.warn("Wet mask not present.")

    else:
        raise ValueError(f"Unsupported grid type: {grid_type}")

    dsout = dsout.reset_coords(drop=True)

    if filename is not None:
        filename = str(filename)
        dsout.to_netcdf(filename)
        result = filename
    else:
        result = dsout

    return result


def woa18_grid(resolution=0.25):
    """Function to return World Ocean Atlas horizontal grid

    Parameters
    ----------
    resolution : float, optional
        Horizontal resolution (0.25 or 1.0), by deafult 0.25

    Returns
    -------
    xarray.Dataset
    """

    if resolution == 0.25:
        res_str = "025"
    elif resolution == 1.0:
        res_str = "1"
    else:
        raise ValueError(
            f"Unknown resolution: {resolution}. Must be either 0.25 or 1.0"
        )

    fpath = pkgr.resource_filename("momgrid", f"grids/WOA18_{res_str}deg_horiz_grid.nc")

    dset = xr.open_dataset(fpath)

    return dset
