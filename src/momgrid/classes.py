"""classes.py : momgrid object definitions """

__all__ = ["MOMgrid"]

from momgrid.util import (
    get_file_type,
    is_hgrid,
    is_static,
    is_symmetric,
    read_netcdf_from_tar,
)

import xarray as xr
import numpy as np
import os
import warnings


class MOMgrid:
    def __init__(
        self,
        source,
        topog=None,
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
        depth = "depth"

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
            setattr(self, dxt, ds[dxt].values)
            setattr(self, dyt, ds[dyt].values)
            setattr(self, deptho, ds[deptho].values)
            setattr(self, wet, ds[wet].values)

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
            if hasattr(self, wet + cell_type[0]):
                ds[wet + cell_type[0]] = xr.DataArray(
                    getattr(self, wet + cell_type[0]), dims=cell_type[1]
                )

        return ds
