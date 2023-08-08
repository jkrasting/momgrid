"""util.py : auxillary functions for inferring dataset characteristics"""

__all__ = ["is_hgrid", "is_static", "is_symmetric"]


def is_hgrid(ds):
    """Tests if dataset is an ocean_hgrid.nc file

    Parameters
    ----------
    ds : xarray.core.dataset.Dataset

    Returns
    -------
    bool
        True, if dataset corresponds to an hgrid file, otherwise False
    """

    # an ocean_hgrid.nc file should contain x, y, dx, and dy
    expected = set(["x", "y", "dx", "dy"])

    return expected.issubset(set(ds.variables))


def is_static(ds):
    """Tests if dataset is an ocean_static.nc file

    Parameters
    ----------
    ds : xarray.core.dataset.Dataset

    Returns
    -------
    bool
        True, if dataset corresponds to an ocean static file, otherwise False
    """

    # an ocean_static.nc file should contain at least geolon and geolat
    expected = set(["geolon", "geolat"])

    return expected.issubset(set(ds.variables))


def is_symmetric(ds, xh="xh", yh="yh", xq="xq", yq="yq"):
    """Tests if an dataset is defined on a symmetric grid

    A dataset generated in symmetric memory mode will have dimensionalty
    of `i+1` and `j+1` for the corner points compared to the tracer
    points.

    Parameters
    ----------
    ds : xarray.core.dataset.Dataset
        Input xarray dataset
    xh : str, optional
        Name of x-dimension of tracer points, by default "xh"
    yh : str, optional
        Name of y-dimension of tracer points, by default "yh"
    xq : str, optional
        Name of x-dimension of corner points, by default "xq"
    yq : str, optional
        Name of y-dimension of corner points, by default "yq"

    Returns
    -------
    bool
        True, if dataset has symmetric dimensionality, otherwise False

    """

    xdiff = len(ds[xq]) - len(ds[xh])
    ydiff = len(ds[yq]) - len(ds[yh])

    # Basic validation checks
    assert (
        xdiff == ydiff
    ), "Diffence of tracer and corner points must be identical for x and y dimensions"
    assert xdiff in [0, 1], "Dataset is neither symmetric or non-symmetric"

    return True if xdiff == 1 else False
