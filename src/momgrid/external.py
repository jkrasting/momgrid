""" external.py: functions to interface with external packages """

import xarray as xr
from momgrid.util import is_symmetric

__all__ = ["static_to_xesmf"]

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
    assert grid_type == "t", "Only tracer grids are supported (encouraged) for regridding."
    assert isinstance(dset, xr.Dataset), "Input must be an xarray dataset."
    assert is_symmetric(dset), "Static file must be from symmetric memory mode."

    #
    if grid_type == "t":
        dsout = xr.Dataset({
            "lat": dset.geolat,
            "lon": dset.geolon,
            "lat_b": dset.geolat_c,
            "lon_b": dset.geolon_c,
            "mask": dset.wet,
        })
    else:
        dsout = xr.Dataset()

    return dsout
    
