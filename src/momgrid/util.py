"""util.py : auxillary functions for inferring dataset characteristics"""

__all__ = [
    "associate_grid_with_data",
    "get_file_type",
    "is_hgrid",
    "is_static",
    "is_symmetric",
    "read_netcdf_from_tar",
    "reset_nominal_coords",
    "verify_dim_lens",
]

import warnings

import os.path
import tarfile
import numpy as np
import xarray as xr
from io import BytesIO

from momgrid.metadata import nominal_coord_metadata


def associate_grid_with_data(grid, data):
    """Function to associate grid metrics with data

    This function accepts a grid object and an xarray data object
    and adds the associated geolon/geolat data to each variable.

    Parameters
    ----------
    grid : xarray.Dataset
        MOMgrid-generated Xarray dataset (using the .to_xarray method)
    data : xarray.Dataset or xarray.DataArray
        MOM6 output data

    Returns
    -------
    xarray.Dataset or xarray.DataArray
    """

    # Define grid point types:
    h_point = ("yh", "xh")
    u_point = ("yh", "xq")
    v_point = ("yq", "xh")
    c_point = ("xq", "yq")

    # variables broken out in case they need to be updated later
    geolon = "geolon"
    geolat = "geolat"
    geolon_u = "geolon_u"
    geolat_u = "geolat_u"
    geolon_v = "geolon_v"
    geolat_v = "geolat_v"
    geolon_c = "geolon_c"
    geolat_c = "geolat_c"

    areacello = "areacello"
    areacello_u = "areacello_u"
    areacello_v = "areacello_v"
    areacello_c = "areacello_c"

    ds = data if isinstance(data, xr.Dataset) else xr.Dataset({data.name: data})

    processed = {}

    for var in ds.keys():
        if set(h_point).issubset(ds[var].dims):
            if verify_dim_lens(ds[var], grid[geolon]):
                processed[var] = ds[var].assign_coords(
                    {geolon: grid[geolon], geolat: grid[geolat]}
                )
                processed[areacello] = grid[areacello]

        elif set(u_point).issubset(ds[var].dims):
            if verify_dim_lens(ds[var], grid[geolon_u]):
                processed[var] = ds[var].assign_coords(
                    {geolon_u: grid[geolon_u], geolat_u: grid[geolat_u]}
                )
            processed[areacello_u] = grid[areacello_u]

        elif set(v_point).issubset(ds[var].dims):
            if verify_dim_lens(ds[var], grid[geolon_v]):
                processed[var] = ds[var].assign_coords(
                    {geolon_v: grid[geolon_v], geolat_v: grid[geolat_v]}
                )
            processed[areacello_v] = grid[areacello_v]

        elif set(c_point).issubset(ds[var].dims):
            if verify_dim_lens(ds[var], grid[geolon_c]):
                processed[var] = ds[var].assign_coords(
                    {geolon_c: grid[geolon_c], geolat_c: grid[geolat_c]}
                )
            processed[areacello_c] = grid[areacello_c]

        else:
            processed[var] = ds[var]

    res = xr.Dataset(processed)
    res.attrs = ds.attrs

    if isinstance(data, xr.DataArray):
        res = res[data.name]

    return res


def get_file_type(fname):
    """Opens a file and determines the file type based on the magic number

    The magic number for NetCDF files is 'CDF\x01' or 'CDF\x02'.
    The magic number for tar files depends on the variant but generally,
    a USTAR tar file starts with "ustar" at byte offset 257 for 5 bytes.

    Parameters
    ----------
    fname : str, path-like
        Input file string
    """

    # make sure file exists
    abspath = os.path.abspath(fname)
    assert os.path.exists(abspath), f"File does not exist: {abspath}"

    # open the file and read the first 512 bytes
    with open(abspath, "rb") as f:
        header = f.read(512)

    # look for the NetCDF magic number
    if (header[0:3] == b"CDF") or (header[1:4] == b"HDF"):
        result = "netcdf"

    # look for the tar file signature
    elif b"ustar" in header[257:262]:
        result = "tar"

    # look for gzipped file
    elif header[0:2] == b"\x1f\x8b":
        result = "tar"

    else:
        result = "unknown"

    return result


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

    if set(["xh", "yh", "xq", "yq"]).issubset(ds.variables):
        xdiff = len(ds[xq]) - len(ds[xh])
        ydiff = len(ds[yq]) - len(ds[yh])

        # Basic validation checks
        assert (
            xdiff == ydiff
        ), "Diffence of tracer and corner points must be identical for x and y dimensions"
        assert xdiff in [0, 1], "Dataset is neither symmetric or non-symmetric"

        result = True if xdiff == 1 else False

    else:
        warnings.warn("Unable to determine if grid is symmetric - assuming False")
        result = False

    return result


def read_netcdf_from_tar(tar_path, netcdf_name):
    """Reads a netcdf file from within a tar file and returns an xarray Dataset

    Parameters
    ----------
    tar_path : str, path-like
        Path to tar file
    netcdf_name : str
        Name of NetCDF file contained within the tar file

    Returns
    -------
        xarray.Dataset
            Dataset object
    """

    with open(tar_path, "rb") as f:
        tar_data = BytesIO(f.read())

    with tarfile.open(fileobj=tar_data, mode="r:*") as tar:
        if (
            netcdf_name not in tar.getnames()
            and f"./{netcdf_name}" not in tar.getnames()
        ):
            raise FileNotFoundError(
                f"The NetCDF file {netcdf_name} was not found in the tar archive."
            )

        effective_name = (
            netcdf_name if netcdf_name in tar.getnames() else f"./{netcdf_name}"
        )

        with tar.extractfile(effective_name) as netcdf_file:
            return xr.open_dataset(BytesIO(netcdf_file.read()))


def reset_nominal_coords(xobj, tracer_dims=("xh", "yh"), velocity_dims=("xq", "yq")):
    """Resets the nominal coordinate values to a monontonic series

    Tracer points are definied on the half integers while the velocity points
    are defined on the full integer points.

    Parameters
    ----------
    xobj : xarray.core.DataArray or xarray.core.Dataset
        Input xarray object
    tracer_dims : tuple, iterable, optional
        Name of tracer dimensions, by default ("xh", "yh")
    velocity_dims : tuple, iterable, optional
        Name of velocity dimensions, by default ("xq", "yq")

    Returns
    -------
        xarray.core.DataArray or xarray.core.Dataset
            Object with reset nominal coordinates
    """

    _xobj = xobj.copy()
    for dim in tracer_dims:
        if dim in _xobj.coords:
            _xobj = _xobj.assign_coords(
                {dim: list(np.arange(0.5, len(_xobj[dim]) + 0.5, 1.0))}
            )

    for dim in velocity_dims:
        if dim in _xobj.coords:
            _xobj = _xobj.assign_coords(
                {dim: list(np.arange(1.0, len(_xobj[dim]) + 1.0, 1.0))}
            )

    _xobj = nominal_coord_metadata(_xobj)

    return _xobj


def verify_dim_lens(var1, var2, verbose=True):
    """Function to test the equality of dimension lengths

    This function determines if the shared dimensions between two
    data arrays are of equal length

    Parameters
    ----------
    var1 : xarray.DataArray
    var2 : xarray.DataArray
    verbose : bool, optional
        Issue warnings if dimensions do not agree, by default True

    Returns
    -------
    bool
    """

    dims = list(set(var1.dims).intersection(set(var2.dims)))
    exception_count = 0
    for dim in dims:
        try:
            assert len(var1[dim]) == len(var2[dim]), (
                f"Different {dim} lengths for {var1.name}: "
                + f"{len(var1[dim])}, {len(var2[dim])} "
                + "Consider symmetric vs. non-symmetric memory "
                + "output vs grid definition."
            )
        except AssertionError as exc:
            warnings.warn(str(exc))
            exception_count += 1
    return exception_count == 0
