""" external.py: functions to interface with external packages """

import xarray as xr
import pkg_resources as pkgr
from momgrid.util import is_symmetric

__all__ = ["static_to_xesmf", "woa18_grid"]


def static_to_xesmf(dset, grid_type="t"):
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

    # Basic checks
    assert (
        grid_type == "t"
    ), "Only tracer grids are supported (encouraged) for regridding."
    assert isinstance(dset, xr.Dataset), "Input must be an xarray dataset."
    assert is_symmetric(dset), "Static file must be from symmetric memory mode."

    #
    if grid_type == "t":
        dsout = xr.Dataset(
            {
                "lat": dset.geolat,
                "lon": dset.geolon,
                "lat_b": dset.geolat_c,
                "lon_b": dset.geolon_c,
                "mask": dset.wet,
            }
        )
    else:
        dsout = xr.Dataset()

    return dsout


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
