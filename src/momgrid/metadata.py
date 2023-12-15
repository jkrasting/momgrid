""" metadata.py - Variable Attribute Definitions """

__all__ = ["add_metadata", "nominal_coord_metadata"]

import xarray as xr


def add_metadata(dset):
    """Function to add missing metadata for static/grid dataset

    Parameters
    ----------
    dset : xarray.Dataset
        Input dataset containing grid variables

    Returns
    -------
    xarray.Dataset
    """

    geolon = "geolon"
    geolat = "geolat"
    areacello = "areacello"
    geolon_u = "geolon_u"
    geolat_u = "geolat_u"
    areacello_u = "areacello_u"
    geolon_v = "geolon_v"
    geolat_v = "geolat_v"
    areacello_v = "areacello_v"
    geolon_c = "geolon_c"
    geolat_c = "geolat_c"
    areacello_c = "areacello_c"
    deptho = "deptho"
    wet = "wet"
    wet_u = "wet_u"
    wet_v = "wet_v"

    attributes = {}

    attributes[areacello] = {
        "units": "m2",
        "long_name": "Ocean Grid-Cell Area",
        "standard_name": "cell_area",
    }

    attributes[areacello_u] = attributes[areacello]
    attributes[areacello_v] = attributes[areacello]
    attributes[areacello_c] = attributes[areacello]

    attributes[deptho] = {
        "units": "m",
        "long_name": "Sea Floor Depth",
        "standard_name": "sea_floor_depth_below_geoid",
    }

    attributes[geolat] = {
        "units": "degrees_north",
        "long_name": "Latitude of tracer (T) points",
        "standard_name": "latitude",
        "axis": "Y",
    }

    attributes[geolat_u] = {
        "units": "degrees_north",
        "comment": "Latitude of zonal velocity (Cu) points",
        "standard_name": "latitude",
        "axis": "Y",
    }

    attributes[geolat_v] = {
        "units": "degrees_north",
        "comment": "Latitude of meridional velocity (Cv) points",
        "standard_name": "latitude",
        "axis": "Y",
    }

    attributes[geolat_c] = {
        "units": "degrees_north",
        "comment": "Latitude of corner (Bu) point",
        "standard_name": "latitude",
        "axis": "Y",
    }

    attributes[geolon] = {
        "units": "degrees_east",
        "comment": "Longitude of tracer (T) points",
        "standard_name": "longitude",
        "axis": "X",
    }

    attributes[geolon_u] = {
        "units": "degrees_east",
        "comment": "Longitude of zonal velocity (Cu) points",
        "standard_name": "longitude",
        "axis": "X",
    }

    attributes[geolon_v] = {
        "units": "degrees_east",
        "comment": "Longitude of meridional velocity (Cv) points",
        "standard_name": "longitude",
        "axis": "X",
    }

    attributes[geolon_c] = {
        "units": "degrees_east",
        "comment": "Longitude of corner (Bu) point",
        "standard_name": "longitude",
        "axis": "X",
    }

    attributes[wet] = {
        "standard_name": "sea_binary_mask",
        "long_name": "Sea Binary Mask 1 = sea, 0 = land at tracer points",
        "units": "1",
    }

    attributes[wet_u] = {
        "standard_name": "sea_binary_mask",
        "long_name": "Sea Binary Mask 1 = sea, 0 = land at (Cu) points",
        "units": "1",
    }

    attributes[wet_v] = {
        "standard_name": "sea_binary_mask",
        "long_name": "Sea Binary Mask 1 = sea, 0 = land at (Cv) points",
        "units": "1",
    }

    for var in dset.keys():
        if var in attributes.keys():
            dset[var].attrs = attributes[var]
            if "lon" in var or "lat" in var:
                dset = dset.set_coords(var)

    return dset


def nominal_coord_metadata(xobj):
    """Function to reset the nominal coordinate / index coordinate metadata

    Parameters
    ----------
    xobj : xarray.Dataset or xarray.DataArray
        Input data containing nominal/index coordinate(s)

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Object with corrected metadata
    """

    ds = xr.Dataset({xobj.name: xobj}) if isinstance(xobj, xr.DataArray) else xobj
    assert isinstance(ds, xr.Dataset)

    for var in list(ds.keys()) + list(ds.coords) + list(ds.dims):
        if var == "xh":
            ds[var].attrs = {
                "units": "1",
                "long_name": "h point nominal x-index",
            }
        elif var == "yh":
            ds[var].attrs = {
                "units": "1",
                "long_name": "h point nominal y-index",
            }
        elif var == "xq":
            ds[var].attrs = {
                "units": "1",
                "long_name": "q point nominal x-index",
            }
        elif var == "yq":
            ds[var].attrs = {
                "units": "1",
                "long_name": "q point nominal y-index",
            }

    if isinstance(xobj, xr.DataArray):
        result = ds[xobj.name]
    else:
        result = ds

    return result
