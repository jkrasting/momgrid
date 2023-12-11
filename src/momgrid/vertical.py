""" vertical.py: routines for various MOM vertical coords/grids """

import numpy as np
import xarray as xr

__all__ = ["analysis_z_35lev_coord"]


def analysis_z_35lev_coord():
    """Returns an xarray-formatted dataset with the 35 z-level coordinate
    used for diagnostics and analysis

    Returns
    -------
        xarray.Dataset
    """

    z_i = np.array(
        [
            0.0,
            5.0,
            15.0,
            25.0,
            40.0,
            62.5,
            87.5,
            112.5,
            137.5,
            175.0,
            225.0,
            275.0,
            350.0,
            450.0,
            550.0,
            650.0,
            750.0,
            850.0,
            950.0,
            1050.0,
            1150.0,
            1250.0,
            1350.0,
            1450.0,
            1625.0,
            1875.0,
            2250.0,
            2750.0,
            3250.0,
            3750.0,
            4250.0,
            4750.0,
            5250.0,
            5750.0,
            6250.0,
            6750.0,
        ]
    )

    z_l = np.array(
        [
            2.5,
            10.0,
            20.0,
            32.5,
            51.25,
            75.0,
            100.0,
            125.0,
            156.25,
            200.0,
            250.0,
            312.5,
            400.0,
            500.0,
            600.0,
            700.0,
            800.0,
            900.0,
            1000.0,
            1100.0,
            1200.0,
            1300.0,
            1400.0,
            1537.5,
            1750.0,
            2062.5,
            2500.0,
            3000.0,
            3500.0,
            4000.0,
            4500.0,
            5000.0,
            5500.0,
            6000.0,
            6500.0,
        ]
    )

    z_l = xr.DataArray(
        z_l,
        dims=("z_l"),
        coords={"z_l": z_l},
        name="z_l",
        attrs={
            "long_name": "Depth at cell center",
            "units": "meters",
            "axis": "Z",
            "positive": "down",
            "edges": "z_i",
        },
    )

    z_i = xr.DataArray(
        z_i,
        dims=("z_i"),
        coords={"z_i": z_i},
        name="z_l",
        attrs={
            "long_name": "Depth at interface",
            "units": "meters",
            "axis": "Z",
            "positive": "down",
        },
    )

    ds = xr.Dataset({"z_l": z_l, "z_i": z_i})

    return ds
