""" geoslice.py : toolkit for selecting a regions """

import itertools
import numpy as np

__all__ = [
    "get_indices_2dcurvilinear",
    "get_indices_1dlatlon",
    "get_indices",
    "geoslice",
    "x_slice",
]


def get_indices_2dcurvilinear(lat_grid, lon_grid, y, x):
    """This function returns the j,i indices for the grid point closest to the input lon(i,j),lat(i,j) coordinates."""
    """It returns the j,i indices."""

    cost = np.fabs((lon_grid - x) ** 2 + (lat_grid - y) ** 2)
    costm = np.where(cost == cost.min())
    j0, i0 = costm[0][0], costm[1][0]
    return j0, i0


def get_indices_1dlatlon(lat_grid, lon_grid, y, x):
    """This function returns the j,i indices for the grid point closest to the input lon(i),lat(j) coordinates."""
    """It returns the j,i indices."""

    lons = np.fabs(np.squeeze(lon_grid) - x)
    lonm = np.where(lons == lons.min())
    lats = np.fabs(np.squeeze(lat_grid) - y)
    latm = np.where(lats == lats.min())
    j0, i0 = latm[0][0], lonm[0][0]
    return j0, i0


def get_indices(lat_grid, lon_grid, y, x):
    """Returns j,i indices of lat_grid and lon grid nearest to x,y"""
    if len(lon_grid.shape) == 1:
        J, I = get_indices_1dlatlon(lat_grid, lon_grid, y, x)
    else:
        J, I = get_indices_2dcurvilinear(lat_grid, lon_grid, y, x)
    return J, I


def geoslice(arr, x=(None, None), y=(None, None), ycoord=None, xcoord=None):
    if xcoord is None:
        xcoord = arr.cf.coordinates["longitude"][0]
    if ycoord is None:
        ycoord = arr.cf.coordinates["latitude"][0]

    xdim = arr[xcoord].dims[-1]
    ydim = arr[ycoord].dims[-2] if len(arr[ycoord].dims) == 2 else arr[ycoord].dims[0]

    combinations = list(itertools.product(y, x))
    xlist = [x[1] for x in combinations if x[1] is not None]
    ylist = [y[0] for y in combinations if y[0] is not None]

    lower_left = (np.min(ylist), np.min(xlist))
    lower_right = (np.min(ylist), np.max(xlist))
    upper_left = (np.max(ylist), np.min(xlist))
    upper_right = (np.max(ylist), np.max(xlist))

    lower_left_ij = get_indices(arr[ycoord], arr[xcoord], *lower_left)
    lower_right_ij = get_indices(arr[ycoord], arr[xcoord], *lower_right)
    upper_left_ij = get_indices(arr[ycoord], arr[xcoord], *upper_left)
    upper_right_ij = get_indices(arr[ycoord], arr[xcoord], *upper_right)

    combinations = [lower_left_ij, lower_right_ij, upper_left_ij, upper_right_ij]
    xlist = [x[1] for x in combinations if x[1] is not None]
    ylist = [y[0] for y in combinations if y[0] is not None]
    xrng = (np.min(xlist), np.max(xlist))
    yrng = (np.min(ylist), np.max(ylist))

    return arr.isel({ydim: slice(*yrng), xdim: slice(*xrng)})


def x_slice(arr, lon_0, xcoord=None, ycoord=None):
    """Function to slice an array at a specific longitude

    Parameters
    ----------
    arr : xarray.core.DataArray
        Input data array
    lon_0 : float
        Longitude to perform slice
    xcoord : str, optional
        Name of x-coordinate otherwise it is inferred, by default None
    ycoord : str, optional
        Name of y-coordinate otherwise it is inferred, by default None

    Returns
    -------
    xarray.core.DataArray
        Sliced data array

    """
    if xcoord is None:
        xcoord = arr.cf.coordinates["longitude"][0]
    if ycoord is None:
        ycoord = arr.cf.coordinates["latitude"][0]

    xdim = arr[xcoord].dims[-1]

    j, i = get_indices(arr[ycoord], arr[xcoord], arr[ycoord].mean(), lon_0)

    result = arr.isel({xdim: i})

    return result
