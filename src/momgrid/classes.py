"""classes.py : momgrid object definitions """

__all__ = ["MOMgrid", "Gridset"]

from momgrid.external import woa18_grid

from momgrid.metadata import add_metadata

from momgrid.util import (
    associate_grid_with_data,
    get_file_type,
    is_hgrid,
    is_static,
    is_symmetric,
    read_netcdf_from_tar,
    reset_nominal_coords,
    standard_grid_area,
)

from momgrid.external import build_regridder_weights, static_to_xesmf

import xarray as xr
import numpy as np
import os
import pickle
import warnings


class MOMgrid:
    def __init__(
        self,
        source,
        topog=None,
        depth_var="depth",
        symmetric=True,
        verbose=False,
        hgrid_dtype="float32",
        max_depth=6500.0,
    ):
        """Ocean Model grid object

        Parameters
        ----------
        source : xarray.Dataset or str, path-like
            Xarray dataset or path to grid source, either an ocean_hgrid.nc
            file or an ocean_static.nc file. Optionally, a known configuration
            may also be specified. Known configurations are "OM4" and
            "OM4p5"
        topog : xarray.Dataset or str, path-like, optional
            Xarray dataset or path to ocean topography. If present the wet masks
            corresponding to the grids will be generated, by default None.
        depth_var : str, optional
            Name of the depth variable in the topog file, by default "depth"
        symmetric : bool, optional
            Return metrics compatible with symmetric memory mode,
            by default True
        verbose : bool, optional
            Verbose output, by default False
        hgrid_dtype : str, numpy.dtype
            Precision to use when reconstructing fields from the
            `ocean_hgrid.nc`. The supergrid is natively double precision
            while the ocean static files are single precision,
            by default "float32"
        max_depth : float, optional
            by default 6500.
        """

        # Define names in static file for future flexibility
        geolon = "geolon"
        geolat = "geolat"
        areacello = "areacello"
        dxt = "dxt"
        dyt = "dyt"
        deptho = "deptho"
        wet = "wet"

        # Define names in supergrid file for future flexibility
        xvar = "x"
        yvar = "y"
        areavar = "area"
        dxvar = "dx"
        dyvar = "dy"

        # Specified as kwarg
        depth = depth_var

        # Load source file
        # TODO: add support for a gridspec tar bundle
        if isinstance(source, xr.Dataset):
            ds = source

        elif isinstance(source, str):
            abspath = os.path.abspath(source)
            if os.path.exists(abspath):
                self.source = abspath
                ftype = get_file_type(abspath)
                if ftype == "netcdf":
                    ds = xr.open_dataset(source)
                elif ftype == "tar":
                    ds = read_netcdf_from_tar(abspath, "ocean_hgrid.nc")

                    try:
                        topog = read_netcdf_from_tar(abspath, "ocean_topog.nc")
                    except Exception as _:
                        try:
                            topog = read_netcdf_from_tar(abspath, "topog.nc")
                        except Exception as _:
                            warning.warn(
                                "Unable to find topog in gridspec bundle. "
                                + "Not processing wet masks or deptho"
                            )

                else:
                    raise ValueError(f"Unknown input file type for {abspath}")

            elif source == "OM4":
                self.source = "Default OM4 grid"
                # TODO: implement known grid
                # ds = xr.open_dataset("path_to_known_OM4_hgrid")

            elif source == "OM4p5":
                self.source = "Default OM4p5 grid"
                # TODO: implement known grid
                # ds = xr.open_dataset("path_to_known_OM4p5_hgrid")

            else:
                raise ValueError(f"Unknown source: {source}")

        else:
            raise ValueError(
                "Source must be an xarray dataset, path, or known model config."
            )

        # Load topog file
        if (isinstance(topog, xr.Dataset)) or (topog is None):
            self.topog_source = str(type(topog))
            topog = topog

        elif isinstance(topog, str):
            abspath = os.path.abspath(topog)
            if os.path.exists(abspath):
                self.topog_source = abspath
                topog = xr.open_dataset(topog)
            else:
                raise ValueError(f"Unknown source: {source}")

        else:
            raise ValueError(
                "Source must be an xarray dataset, path, or known model config."
            )

        # TODO: remove this object below; only used in testing
        self.ds = ds

        # Store whether or not the source is a static file or an hgrid file
        self.is_hgrid = is_hgrid(ds)
        self.is_static = is_static(ds)

        # If a static file is defined, test if it is compatible with the symmetric
        # memory mode that was requested
        if self.is_static:
            if is_symmetric(ds) is not symmetric:
                warnings.warn(
                    f"Supplied static file inconsistent with requested memory mode. Adjusting ..."
                )
                self.symmetric = is_symmetric(ds)
            else:
                self.symmetric = symmetric
        else:
            self.symmetric = symmetric

        # If the source information is from the hgrid file, pre-load data
        if self.is_hgrid:
            # TODO: add a hook here to do the downsample

            # x and y values and distances
            x = ds[xvar].values
            dx = ds[dxvar].values
            y = ds[yvar].values
            dy = ds[dyvar].values

            # make cell areas periodic in x and y
            area = ds["area"].values
            area = np.append(area[0, :][None, :], area, axis=0)
            area = np.append(area[:, 0][:, None], area, axis=1)

        # Fetch tracer cell grid metrics
        if self.is_static:
            setattr(self, geolon, ds[geolon].values)
            setattr(self, geolat, ds[geolat].values)
            setattr(self, areacello, ds[areacello].values)
            setattr(self, deptho, ds[deptho].values)
            setattr(self, wet, ds[wet].values)
            setattr(self, dxt, ds[dxt].values)
            setattr(self, dyt, ds[dyt].values)

        elif self.is_hgrid:
            setattr(self, geolon, x[1::2, 1::2].astype(hgrid_dtype))
            setattr(self, dxt, (dx[1::2, ::2] + dx[1::2, 1::2]).astype(hgrid_dtype))

            setattr(self, geolat, y[1::2, 1::2].astype(hgrid_dtype))
            setattr(self, dyt, (dy[::2, 1::2] + dy[1::2, 1::2]).astype(hgrid_dtype))

            _area = area[:-1, :-1]
            _area = (
                _area[::2, ::2]
                + _area[1::2, 1::2]
                + _area[::2, 1::2]
                + _area[1::2, ::2]
            )
            setattr(self, areacello, _area.astype(hgrid_dtype))

            if topog is not None:
                _depth = topog[depth].values
                _depth = np.where(_depth > max_depth, max_depth, _depth)
                _depth = np.where(_depth > 0, _depth, np.nan)
                _wet = np.where(np.isnan(_depth), 0.0, 1.0)

                setattr(self, deptho, _depth.astype(hgrid_dtype))
                setattr(self, wet, _wet.astype(hgrid_dtype))

                # reflect top row about the center
                _wet_padded = np.concatenate((_wet, _wet[-1, :][::-1][None, :]), axis=0)

                _wet_padded = np.concatenate(
                    (_wet_padded, _wet_padded[:, 0][:, None]), axis=1
                )

        # Fetch u-cell grid metrics
        suffix = "_u"
        if self.is_static:
            setattr(self, geolon + suffix, ds[geolon + suffix].values)
            setattr(self, geolat + suffix, ds[geolat + suffix].values)
            setattr(self, areacello + suffix, ds[areacello + "_cu"].values)
            setattr(self, "dxCu", ds["dxCu"].values)
            setattr(self, "dyCu", ds["dyCu"].values)
            setattr(self, wet + suffix, ds[wet + suffix].values)

        elif self.is_hgrid:
            _geolon = x[1::2, ::2]
            _geolon = _geolon if self.symmetric else _geolon[:, 1:]
            setattr(self, geolon + suffix, _geolon.astype(hgrid_dtype))

            _dxCu = dx[1::2, ::2]
            _dxCu = _dxCu + np.roll(dx[1::2, :-1:2], -1, axis=-1)
            _dxCu = np.append(_dxCu[:, 0][:, None], _dxCu, axis=1)
            _dxCu = _dxCu if self.symmetric else _dxCu[:, 1:]
            setattr(self, "dxCu", _dxCu.astype(hgrid_dtype))

            _geolat = y[1::2, ::2]
            _geolat = _geolat if self.symmetric else _geolat[:, 1:]
            setattr(self, geolat + suffix, _geolat.astype(hgrid_dtype))

            _dyCu = dy[::2, 2::2] + dy[1::2, 2::2]
            _dyCu = np.append(_dyCu[:, 0][:, None], _dyCu, axis=1)
            _dyCu = _dyCu if self.symmetric else _dyCu[:, 1:]
            setattr(self, "dyCu", _dyCu.astype(hgrid_dtype))

            _area = area[:-1, :]
            _area = (
                _area[::2, 1::2]
                + _area[1::2, 2::2]
                + _area[::2, 2::2]
                + _area[1::2, 1::2]
            )
            _area = np.append(_area[:, 0][:, None], _area, axis=1)
            _area = _area if self.symmetric else _area[:, 1:]
            setattr(self, areacello + suffix, _area.astype(hgrid_dtype))

            if topog is not None:
                _wet = np.minimum(np.roll(_wet_padded, 1, axis=1), _wet_padded)
                _wet = _wet if not self.symmetric else _wet[0:-1, :]
                setattr(self, wet + suffix, _wet.astype(hgrid_dtype))

        # Fetch v-cell grid metrics
        suffix = "_v"
        if self.is_static:
            setattr(self, geolon + suffix, ds[geolon + suffix].values)
            setattr(self, geolat + suffix, ds[geolat + suffix].values)
            setattr(self, areacello + suffix, ds[areacello + "_cv"].values)
            setattr(self, "dxCv", ds["dxCv"].values)
            setattr(self, "dyCv", ds["dyCv"].values)
            setattr(self, wet + suffix, ds[wet + suffix].values)

        elif self.is_hgrid:
            _geolon = x[::2, 1::2]
            _geolon = _geolon if self.symmetric else _geolon[1:, :]
            setattr(self, geolon + suffix, _geolon.astype(hgrid_dtype))

            _dxCv = dx[2::2, ::2] + dx[2::2, 1::2]
            _dxCv = np.append(_dxCv[0, :][None, :], _dxCv, axis=0)
            _dxCv = _dxCv if self.symmetric else _dxCv[1:, :]
            setattr(self, "dxCv", _dxCv.astype(hgrid_dtype))

            _geolat = y[::2, 1::2]
            _geolat = _geolat if self.symmetric else _geolat[1:, :]
            setattr(self, geolat + suffix, _geolat.astype(hgrid_dtype))

            _dyCv = dy[::2, 2::2] + dy[1::2, 2::2]
            _dyCv = np.append(_dyCv[0, :][None, :], _dyCv, axis=0)
            _dyCv = _dyCv if self.symmetric else _dyCv[1:, :]
            setattr(self, "dyCv", _dyCv.astype(hgrid_dtype))

            _area = area[:, :-1]
            _area = (
                _area[1::2, ::2]
                + _area[2::2, 1::2]
                + _area[1::2, 1::2]
                + _area[2::2, ::2]
            )
            _area = np.append(_area[0, :][None, :], _area, axis=0)
            _area = _area if self.symmetric else _area[1:, :]
            setattr(self, areacello + suffix, _area.astype(hgrid_dtype))

            if topog is not None:
                _wet = np.minimum(np.roll(_wet_padded, 1, axis=0), _wet_padded)
                _wet = _wet if not self.symmetric else _wet[:, 0:-1]
                setattr(self, wet + suffix, _wet.astype(hgrid_dtype))

        # Fetch corner cell grid metrics
        suffix = "_c"
        if self.is_static:
            setattr(self, geolon + suffix, ds[geolon + suffix].values)
            setattr(self, geolat + suffix, ds[geolat + suffix].values)
            setattr(self, areacello + suffix, ds[areacello + "_bu"].values)
            # TODO: setattr(self, wet + suffix, ds[wet + suffix].values)
            # note: dx and dy are not defined in ocean_static.nc files

        elif self.is_hgrid:
            _geolon = x[::2, ::2]
            _geolon = _geolon if self.symmetric else _geolon[1:, 1:]
            setattr(self, geolon + suffix, _geolon.astype(hgrid_dtype))

            _geolat = y[::2, ::2]
            _geolat = _geolat if self.symmetric else _geolat[1:, 1:]
            setattr(self, geolat + suffix, _geolat.astype(hgrid_dtype))

            _area = area
            _area = (
                _area[1::2, 1::2]
                + _area[2::2, 2::2]
                + _area[1::2, 2::2]
                + _area[2::2, 1::2]
            )
            _area = np.append(_area[0, :][None, :], _area, axis=0)
            _area = np.append(_area[:, 0][:, None], _area, axis=1)
            _area = _area if self.symmetric else _area[1:, 1:]
            setattr(self, areacello + suffix, _area.astype(hgrid_dtype))

            # TODO: add wet mask for corner cells

    def to_xarray(self):
        # Define dimension names for future flexibility
        ycenter = "yh"
        xcenter = "xh"
        ycorner = "yq"
        xcorner = "xq"

        # Define names in static file for future flexibility
        geolon = "geolon"
        geolat = "geolat"
        areacello = "areacello"
        dxt = "dxt"
        dyt = "dyt"
        deptho = "deptho"
        wet = "wet"

        tcell = ("", (ycenter, xcenter))
        ucell = ("_u", (ycenter, xcorner))
        vcell = ("_v", (ycorner, xcenter))
        ccell = ("_c", (ycorner, xcorner))

        cell_types = [tcell, ucell, vcell, ccell]

        ds = xr.Dataset()

        for cell_type in cell_types:
            ds[geolon + cell_type[0]] = xr.DataArray(
                getattr(self, geolon + cell_type[0]), dims=cell_type[1]
            )
            ds[geolat + cell_type[0]] = xr.DataArray(
                getattr(self, geolat + cell_type[0]), dims=cell_type[1]
            )
            ds[areacello + cell_type[0]] = xr.DataArray(
                getattr(self, areacello + cell_type[0]), dims=cell_type[1]
            )

            # TODO: The dx/dy names do not follow a predicitable pattern. Need to deal with this
            # ds[dxt+cell_type[0]] = xr.DataArray(getattr(self,dxt+cell_type[0]), dims=cell_type[1])
            # ds[dyt+cell_type[0]] = xr.DataArray(getattr(self,dyt+cell_type[0]), dims=cell_type[1])

        # TODO: Add variable attributes -- long_name, standard_name, units, etc.

        if hasattr(self, "deptho"):
            if self.deptho is not None:
                ds[deptho] = xr.DataArray(getattr(self, deptho), dims=tcell[1])

        for cell_type in cell_types:
            try:
                if hasattr(self, wet + cell_type[0]):
                    ds[wet + cell_type[0]] = xr.DataArray(
                        getattr(self, wet + cell_type[0]), dims=cell_type[1]
                    )
            except:
                # warnings.warn(f"Unable to add wet_{cell_type[0]}")
                pass

        # Promote dimensions to coords
        for coord in ["xh", "yh", "xq", "yq"]:
            ds[coord] = xr.DataArray(ds[coord], dims=(coord), coords={coord: ds[coord]})
        ds = reset_nominal_coords(ds)

        # Add variable metadata
        ds = add_metadata(ds)

        return ds

    def to_xesmf(self, grid_type="t", filename=None):
        return static_to_xesmf(self.to_xarray(), grid_type=grid_type, filename=filename)

    def associate(self, data):
        return associate_grid_with_data(self.to_xarray(), reset_nominal_coords(data))

    def generate_weights(self, dsout, grid_type=["t", "u", "v", "c"], periodic=None):
        grid_type = list(grid_type) if not isinstance(grid_type, list) else grid_type
        symmetric = "sym" if self.symmetric else "nosym"

        if "t" in grid_type:
            _periodic = True if periodic is None else periodic
            dsin = self.to_xesmf(grid_type="t")
            files = build_regridder_weights(dsin, dsout, periodic=_periodic)
            _ = [os.rename(x, f"t_{symmetric}_{x}") for x in files]

        if "u" in grid_type:
            _periodic = False if periodic is None else periodic
            dsin = self.to_xesmf(grid_type="u")
            files = build_regridder_weights(dsin, dsout, periodic=_periodic)
            _ = [os.rename(x, f"u_{symmetric}_{x}") for x in files]

        if "v" in grid_type:
            _periodic = True if periodic is None else periodic
            dsin = self.to_xesmf(grid_type="v")
            files = build_regridder_weights(dsin, dsout, periodic=_periodic)
            _ = [os.rename(x, f"v_{symmetric}_{x}") for x in files]

        if "c" in grid_type:
            _periodic = False if periodic is None else periodic
            dsin = self.to_xesmf(grid_type="c")
            files = build_regridder_weights(dsin, dsout, periodic=_periodic)
            _ = [os.rename(x, f"c_{symmetric}_{x}") for x in files]

        return "Finished generating weights."


class Gridset:
    def __init__(self, dset, ignore=None):
        """Combination class of MOM grid object and a model data set

        Parameters
        ----------
        dset : xarray.Dataset, str (path-like), or list (paths)
            Model dataset object, path to NetCDF file, or list of files
        ignore : str or list
            List of variables to ignore when processing the dataset
        """

        # Open dataset if an xarray Dataset object is not supplied
        if not isinstance(dset, xr.Dataset):
            dset = xr.open_mfdataset(dset, use_cftime=True)

        # Filter variables supplied by the ignore kwarg
        if ignore is not None:
            ignore = [ignore] if not isinstance(ignore, list) else ignore
            ignore = [x for x in ignore if x in dset.keys()]
            dset = dset.drop_vars(ignore)

        # Get dimension lengths. Note the MOM6 conventions of
        # xh, yh, xq, and yq are used below
        self.xh = len(dset["xh"]) if "xh" in dset.dims else 0
        self.yh = len(dset["yh"]) if "yh" in dset.dims else 0
        self.xq = len(dset["xq"]) if "xq" in dset.dims else 0
        self.yq = len(dset["yq"]) if "yq" in dset.dims else 0

        # Construct a list of tuples of the grids inferred from the
        # supplied input dataset
        self.grid = [
            (self.yh, self.xh),
            (self.yh, self.xq),
            (self.yq, self.xh),
            (self.yq, self.xq),
        ]

        # Infer the model type based on the grid dimensions

        if any(x in [(1080, 1441), (1081, 1440), (1081, 1441)] for x in self.grid):
            self.model = "om4_sym"
            self._periodic = True

        elif any(x in [(576, 721), (577, 720), (577, 721)] for x in self.grid):
            self.model = "esm4_sym"
            self._periodic = True

        # Note that if a dataset contains ONLY variables in the tracer grid,
        # it would be better to use a symmetric version of the grid since it
        # correctly includes the corner cell data
        elif any(x in [(1080, 1440)] for x in self.grid):
            self.model = "om4_nonsym"
            self._periodic = True

        # Note that if a dataset contains ONLY variables in the tracer grid,
        # it would be better to use a symmetric version of the grid since it
        # correctly includes the corner cell data
        elif any(x in [(576, 720)] for x in self.grid):
            self.model = "esm4_nonsym"
            self._periodic = True

        elif any(
            x in [(1161, 1440), (1161, 1441), (1162, 1440), (1162, 1441)]
            for x in self.grid
        ):
            self.model = "om5"
            self._periodic = True

        elif any(
            x in [(845, 775), (845, 776), (846, 775), (846, 776)] for x in self.grid
        ):
            self.model = "nwa12"
            self._periodic = False

        elif any(x in [(320, 360)] for x in self.grid):
            self.model = "spearmed_nonsym"
            self._periodic = True

        else:
            self.model = None
            self._periodic = None

        # Load a cached copy of the grid object for the specific model.
        # This cached copy is significantly faster than calculating each time.
        # However, it might be a good idea to allow run-time calculation
        # as an option in the future for new or unsupported model configs.

        if "MOMGRID_WEIGHTS_DIR" in os.environ.keys():
            weights_dir = os.environ["MOMGRID_WEIGHTS_DIR"]
        else:
            weights_dir = "./grid_weights"

        if self.model is not None:
            try:
                file = open(f"{weights_dir}/{self.model}.pkl", "rb")
                self.grid = pickle.load(file)
                file.close()
            except Exception as exc:
                raise ValueError(f"Unable to load grid: {self.model}")
        else:
            raise ValueError("Unable to infer model from input data.")

        # Associate the 2D geolat/geolon as coordinate variables
        self.data = self.grid.associate(dset)

        # Drop singleton dimensions and coordinates
        self.data = self.data.squeeze()

        # Re-apply metadata for coordinate variables
        self.coord_attrs = {}
        for coord in self.data.coords:
            self.coord_attrs[coord] = self.data[coord].attrs

    def shape_string(self, grid_type):
        """Constructs xESMF string for cached grid weights"""
        if grid_type == "t":
            shape = f"{self.yh}x{self.xh}"
        elif grid_type == "u":
            shape = f"{self.yh}x{self.xq}"
        elif grid_type == "v":
            shape = f"{self.yq}x{self.xh}"
        elif grid_type == "c":
            shape = f"{self.yq}x{self.xq}"
        else:
            shape = ""

        return shape

    def grid_type(self, dims):
        """Infers grid type based on dimension names"""
        if set(("yh", "xh")).issubset(dims):
            grid_type = "t"
        elif set(("yh", "xq")).issubset(dims):
            grid_type = "u"
        elif set(("yq", "xh")).issubset(dims):
            grid_type = "v"
        elif set(("yq", "xq")).issubset(dims):
            grid_type = "c"
        else:
            grid_type = ""

        return grid_type

    def periodic(self, grid_type):
        """Returns False for u and c grids"""
        return False if grid_type in ["u", "c"] else self._periodic

    def regrid_var(self, var, method="bilinear", resolution=1.0):
        """Regrids a variable using xESMF

        Parameters
        ----------
        var : str
            Variable name
        method : str, optional
            xESMF regridding method, by default "bilinear"
        resolution : float
            Target WOA18 grid resolution (1.0 or 0.25), by default 0.25

        Returns
        -------
        xarray.DataArray
        """

        # Import deliberately added here. (Did not want to make the package
        # dependent on xESMF -- just needed for regridding)

        import xesmf as xe

        # Determine grid type
        grid_type = self.grid_type(self.data[var].dims)
        symmetric = "sym" if self.grid.symmetric else "nosym"

        # Determine if grid is periodic
        if method not in ["conservative"]:
            periodic = "_peri" if self.periodic(grid_type) else ""
        else:
            periodic = ""

        # Constuct xESMF shape string
        shape = self.shape_string(grid_type)

        # Setup output grid

        if resolution == 1.0:
            grid_dst = woa18_grid(resolution=1.0)
            dims_dst = "180x360"

        elif resolution == 0.25:
            grid_dst = woa18_grid(resolution=0.25)
            dims_dst = "720x1440"

        # Construct full name of the saved regridder weights
        regridder = f"{grid_type}_{symmetric}_{method}_{shape}_{dims_dst}{periodic}"

        if "MOMGRID_WEIGHTS_DIR" in os.environ.keys():
            weights_dir = os.environ["MOMGRID_WEIGHTS_DIR"]
        else:
            weights_dir = "./grid_weights"

        # Check if a cached version of the regridder exists. If not, load it
        if not hasattr(self, regridder):
            setattr(
                self,
                regridder,
                xe.Regridder(
                    self.grid.to_xesmf(grid_type=grid_type),
                    grid_dst,
                    method,
                    weights=f"{weights_dir}/{regridder}.nc",
                ),
            )

        # Regrid the variable using the regridder
        _regridder = getattr(self, regridder)
        result = _regridder(self.data[var])

        # Set variable name, attributes, and coordinates
        result.name = var
        result.attrs = {**result.attrs, **self.data[var].attrs}
        for coord in result.coords:
            if coord in self.data[var].coords:
                if len(result.coords[coord].attrs) == 0:
                    result[coord].attrs = self.coord_attrs[coord]

        return result

    def regrid(self, method="bilinear", resolution=1.0):
        """Loop over all variables and regrid"""

        # Create an empty dataset to hold the results
        dsout = xr.Dataset()

        # Loop over variables
        for var in list(self.data.keys()):
            try:
                dsout[var] = self.regrid_var(var, method=method, resolution=resolution)

            # Warn if regridding failed
            except Exception as exc:
                warnings.warn(str(exc))
                pass

        # Add back in the mask and the bounds
        woa_grid = woa18_grid(resolution=resolution)
        for var in ["lat_b", "lon_b", "mask"]:
            dsout[var] = woa_grid[var]
            dsout = dsout.set_coords(var)

        # Calculate the cell area and set attributes
        dsout["areacello"] = xr.DataArray(
            standard_grid_area(dsout["lat_b"], dsout["lon_b"]),
            dims=("lat", "lon"),
        )

        dsout["areacello"].attrs = {
            "units": "m2",
            "long_name": "Ocean Grid-Cell Area",
            "standard_name": "cell_area",
        }

        # Make cell area a coordinate variable
        dsout = dsout.set_coords("areacello")

        if "z_l" in dsout.coords:
            if "z_i" not in dsout.coords and "z_i" in self.data.keys():
                dsout = dsout.assign_coords({"z_i": self.data["z_i"]})

        return dsout

    def subset(self, varlist=None):
        """Function to subset the data"""

        # Make a copy of the dataset object
        dset = self.data

        if varlist is not None:
            varlist = [varlist] if isinstance(varlist, str) else varlist
            assert isinstance(varlist, list), "varlist must be a str or list"
            dset = [dset[x] for x in varlist if x in dset.keys()]
            dset = xr.Dataset({x.name: x for x in dset})

            if "z_i" not in dset.coords and "z_i" in self.data.keys():
                dset = dset.assign_coords({"z_i": self.data["z_i"]})

        # Reassign the result to the object
        self.data = dset

        # Return a list of the retained variables
        return list(self.data.keys())

    def __repr__(self):
        # TODO: write useful information about the object
        return str(self.data)

    def __str__(self):
        # TODO: write useful information about the object
        return self.data
