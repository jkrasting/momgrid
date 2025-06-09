"""util.py : auxillary functions for inferring dataset characteristics"""

__all__ = [
    "associate_grid_with_data",
    "extract_level",
    "get_file_type",
    "infer_bounds",
    "is_hgrid",
    "is_static",
    "is_symmetric",
    "load_cached_grid",
    "read_netcdf_from_tar",
    "reset_nominal_coords",
    "standard_grid_area",
    "x_average_dataset",
    "verify_dim_lens",
]

import os
import pickle
import warnings
import cmip_basins

import os.path
import tarfile
import numpy as np
import xarray as xr
from io import BytesIO

from momgrid.metadata import nominal_coord_metadata
from momgrid.geoslice import x_slice


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

    # Define vertical dimensions
    z_layer = "z_l"
    z_interface = "z_i"

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

    deptho = "deptho"
    wet = "wet"

    ds = data if isinstance(data, xr.Dataset) else xr.Dataset({data.name: data})

    # Check that dimensions are identical
    exceptions = []
    for dim in ["xh", "yh", "xq", "yq"]:
        if dim in grid.variables and dim in ds.variables:
            try:
                assert np.array_equal(grid[dim].values, ds[dim].values), dim
            except AssertionError as exc:
                exceptions.append(dim)

    if len(exceptions) > 0:
        raise RuntimeError(
            f"Cannot associate grid to data. Different dims: {exceptions}"
        )

    # Save vertical dimensions to add them later
    vertical_coords = {}
    for x in ["z_l", "z_i"]:
        if x in ds.coords:
            vertical_coords[x] = ds[x]

    processed = {}

    for var in ds.keys():
        if set(h_point).issubset(ds[var].dims):
            if verify_dim_lens(ds[var], grid[geolon]):
                processed[var] = ds[var].assign_coords(
                    {
                        geolon: grid[geolon],
                        geolat: grid[geolat],
                        areacello: grid[areacello],
                        deptho: grid[deptho],
                        wet: grid[wet],
                    }
                )

        elif set(u_point).issubset(ds[var].dims):
            if verify_dim_lens(ds[var], grid[geolon_u]):
                processed[var] = ds[var].assign_coords(
                    {
                        geolon_u: grid[geolon_u],
                        geolat_u: grid[geolat_u],
                        areacello_u: grid[areacello_u],
                    }
                )

        elif set(v_point).issubset(ds[var].dims):
            if verify_dim_lens(ds[var], grid[geolon_v]):
                processed[var] = ds[var].assign_coords(
                    {
                        geolon_v: grid[geolon_v],
                        geolat_v: grid[geolat_v],
                        areacello_v: grid[areacello_v],
                    }
                )

        elif set(c_point).issubset(ds[var].dims):
            if verify_dim_lens(ds[var], grid[geolon_c]):
                processed[var] = ds[var].assign_coords(
                    {
                        geolon_c: grid[geolon_c],
                        geolat_c: grid[geolat_c],
                        areacello_c: grid[areacello_c],
                    }
                )

        else:
            processed[var] = ds[var]

    res = [xr.Dataset({k: v}) for k, v in processed.items()]
    res = xr.merge(res, compat="override")
    res.attrs = ds.attrs

    if len(vertical_coords) > 0:
        for x in vertical_coords.keys():
            res[x] = vertical_coords[x]
            res = res.set_coords({x:vertical_coords[x]})

    if isinstance(data, xr.DataArray):
        res = res[data.name]

    return res


def extract_level(dset, level):
    """Function to extract a depth level from a dataset

    This function is a wrapper for xarray's intrinsic .sel() method
    but uses cfxarray to infer the name of the vertical dimension.

    Parameters
    ----------
    dset : xarray.Dataset
        Input dataset
    level : float
        Depth level to select (m)

    Returns
    -------
    xarray.Dataset
    """

    zaxes = dset.cf.axes["Z"]

    for zax in zaxes:
        dset = dset.sel({zax: level}, method="nearest")

    return dset


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


def infer_bounds(centers, start=None, end=None):
    """Function to infer cell bounds from cell centers

    This function takes a vector of cell centers. Assuming a standard grid,
    the cell bounds are inferred. Optional caps at the start and end can be
    applied if specified.

    Paramters
    ---------
    centers : np.ndarray
        Vector of cell centers
    start : float, optional
        Starting limit of bounds (e.g. -90.), by default None
    end : float, optional
        Starting limit of bounds (e.g. -90.), by default None

    Returns
    -------
    numpy.ndarray
        Vector of cell bounds with a shape of len(centers)+1
    """

    midpoints = (centers[1:] + centers[:-1]) / 2.0
    front = centers[0] - np.abs(centers[0] - midpoints[0])
    end = centers[-1] + np.abs(centers[-1] - midpoints[-1])
    midpoints = np.insert(midpoints, 0, front)
    midpoints = np.append(midpoints, end)
    return np.clip(midpoints, start, end)


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


def load_cached_grid(gridname):
    """Load a cached MOMgrid instance by name

    Parameters
    ----------
    gridname : str
        Name of grid

    Returns
    -------
    momgrid.MOMgrid
        Grid instance
    """

    if "MOMGRID_WEIGHTS_DIR" in os.environ.keys():
        weights_dir = os.environ["MOMGRID_WEIGHTS_DIR"]
    else:
        weights_dir = "./grid_weights"

    try:
        file = open(f"{weights_dir}/{gridname}.pkl", "rb")
        result = pickle.load(file)
        file.close()
    except Exception as exc:
        raise ValueError(f"Unable to load grid: {gridname}")

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


def standard_grid_area(lat_b, lon_b, rad_earth=6371.0e3):
    """Function to calculate the cell areas for a standard grid

    Parameters
    ----------
    lat_b : list or numpy.ndarray
        1-D vector of latitude cell bounds
    lon_b : list or numpy.ndarray
        1-D vector of longitude cell bounds
    rad_earth : float, optional
        Radius of the Earth in meters, by default 6371.0e3

    Returns
    -------
    numpy.ndarray
        2-dimensional array of cell areas
    """

    lat_b = np.array(lat_b)
    lon_b = np.array(lon_b)

    sin_lat_b = np.sin(np.radians(lat_b))

    dy = np.abs(sin_lat_b[1:] - sin_lat_b[0:-1])
    dx = np.abs(lon_b[1:] - lon_b[0:-1])

    dy2d, dx2d = np.meshgrid(dx, dy)

    area = (np.pi / 180.0) * (rad_earth**2) * dy2d * dx2d

    return area


def x_average_dataset(dset, region=None, lon_0=None):
    """Function to average a dataset along the x-dimension

    This function performs a zonal mean on an xarray-dataset

    Parameters
    ----------
    dset : xarray.Dataset
        Input dataset
    region : str, optional
        Basin/region to subset. Options are "atlarc" and "indpac",
        by default None (global)
    lon_0 : float, optional
        Slice along a specific meridian, by default None

    Returns
    -------
    xarray.Dataset
    """

    xcoords = dset.cf.coordinates["longitude"]
    xdims = [dset[x].dims[-1] for x in xcoords]

    ycoords = dset.cf.coordinates["latitude"]

    ydims = []
    for ycoord in ycoords:
        ydim = list(dset[ycoord].dims)

        if len(ydim) > 1:
            ydim = ydim[-2]
            multidim = True
        else:
            ydim = ydim[-1]
            multidim = False

        ydims.append(ydim)

    z_i = dset["z_i"] if "z_i" in dset.keys() else None

    if region is not None:
        result = xr.Dataset()
        basin = {}

        varlist = list(dset.keys()) + ycoords if multidim else list(dset.keys())

        for var in varlist:
            xcoord = dset[var].cf.coordinates["longitude"][0]
            xdim = list(dset[xcoord].dims)[-1]

            ycoord = dset[var].cf.coordinates["latitude"][0]
            ydim = list(dset[ycoord].dims)
            ydim = ydim[-2] if len(ydim) > 1 else ydim[-1]

            if xcoord == "geolon_u":
                wet = "wet_u"
            elif xcoord == "geolon_v":
                wet = "wet_v"
            elif xcoord == "geolon_c":
                wet = "wet_c"
            else:
                wet = "wet"

            dimset = (ydim, xdim)
            if dimset not in basin.keys():
                _dset = xr.Dataset()
                _dset[var] = dset[var]
                _codes = cmip_basins.generate_basin_codes(_dset, xcoord, ycoord, wet)
                regions = {
                    "atlarc": xr.where(
                        (_codes == 2) | (_codes == 4), 1.0, np.nan, keep_attrs=True
                    ),
                    "indpac": xr.where(
                        (_codes == 3) | (_codes == 5), 1.0, np.nan, keep_attrs=True
                    ),
                }
                basin[dimset] = regions
            else:
                regions = basin[dimset]

            result[var] = (dset[var] * regions[region]).mean(xdim)
            result[var].attrs = dset[var].attrs

        result = result.set_coords(ycoords)

    elif lon_0 is not None:
        result = xr.Dataset()
        varlist = list(dset.keys()) + ycoords if multidim else list(dset.keys())
        for var in varlist:
            result[var] = x_slice(dset[var], lon_0)
            result[var].attrs = dset[var].attrs

        result = result.set_coords(ycoords)

    else:
        if multidim:
            dset = dset.reset_coords(ycoords)
            dset = dset.mean(xdims, keep_attrs=True)
            result = dset.set_coords(ycoords)
        else:
            result = dset.mean(xdims, keep_attrs=True)

    mappings = {result[x].dims[0]: x for x in ycoords}

    result["z_i"] = z_i
    result.set_coords("z_i")

    return result


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
