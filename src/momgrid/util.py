"""util.py : auxillary functions for inferring dataset characteristics"""

__all__ = [
    "get_file_type",
    "is_hgrid",
    "is_static",
    "is_symmetric",
    "read_netcdf_from_tar",
    "reset_nominal_coords",
]

import os.path
import tarfile
import numpy as np
import xarray as xr
from io import BytesIO


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
    if header[0:3] == b"CDF":
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

    xdiff = len(ds[xq]) - len(ds[xh])
    ydiff = len(ds[yq]) - len(ds[yh])

    # Basic validation checks
    assert (
        xdiff == ydiff
    ), "Diffence of tracer and corner points must be identical for x and y dimensions"
    assert xdiff in [0, 1], "Dataset is neither symmetric or non-symmetric"

    return True if xdiff == 1 else False


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

    return _xobj
