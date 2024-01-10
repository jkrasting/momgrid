""" plot.py - plotting routines """

import warnings
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from . import VerticalSplitScale

from momgrid.geoslice import geoslice
from momgrid.util import infer_bounds

__all__ = [
    "add_stats_box",
    "generate_cmap_and_norm",
    "round_step",
    "update_stats",
    "compare_2d",
]


def add_stats_box(
    ax,
    xloc,
    yloc,
    label,
    min_val,
    max_val,
    mean_val=None,
    rms_val=None,
    ha="left",
    sig_digits=5,
):
    """Function to add a formatted stats box to an axis

    If both mean_val and rms_val are supplied, a 2x2 box will be returned

    Parameters
    ----------
    ax : matplot.pyplot.axis
        Axis object to add a stats box
    xloc : float
        x-location of box (0.0 to 1.0)
    yloc : float
        y-location of box (0.0 to 1.0)
    label : str
        String title for box
    min_val : float
        Minimum value
    max_val : float
        Maximum value
    mean_val : float, optional
        Mean value, by default None
    rms_val : float, optional
        RMS difference / error value, by default None
    ha : string, optional
        Horizontal alignment, by default "left"
    sig_digits : int, optional
        Number of significant digits to report, by default 5
    """

    # Creating the text string with rounded values

    # Construct the text string based on the supplied values
    # Rounding is to 5 significant digits
    if mean_val is not None and rms_val is not None:
        textstr = (
            f"{label}:\n"
            f"min = {min_val:.{sig_digits}g}   max = {max_val:.{sig_digits}g}\n"
            f"mean = {mean_val:.{sig_digits}g}   RMSDiff = {rms_val:.{sig_digits}g}"
        )

    else:
        textstr = (
            f"{label}:\n"
            f"min = {min_val:.{sig_digits}g}\n"
            f"max = {max_val:.{sig_digits}g}"
        )

    # Define the style properties for the box
    props = dict(
        boxstyle="round,pad=0.5", edgecolor="black", linewidth=1.5, facecolor="white"
    )

    # Add the box and text to the axis object
    ax.text(
        xloc,
        yloc,
        textstr,
        transform=ax.transAxes,
        fontsize=8,
        ha=ha,
        verticalalignment="top",
        bbox=props,
    )


def round_step(step):
    """Function to round step interval to a more intuituve value

    Rounds the step to the nearest value ending in 0, 2, or 5.

    Parameters
    ----------
    step : float
       Value of step interval to be rounded

    Returns
    -------
    float
        Rounded step value
    """

    # Get the order of magnitude
    power = 10 ** np.floor(np.log10(step))

    rounded_step = np.round(step / power, 1) * power

    # Adjust to end in 0, 2, or 5
    last_digit = int(str(rounded_step)[-1])
    if last_digit < 2:
        return np.floor(rounded_step / power) * power
    elif last_digit < 5:
        return np.floor(rounded_step / power) * power + 2 * power / 10
    elif last_digit < 7:
        return np.floor(rounded_step / power) * power + 5 * power / 10
    else:
        return np.floor(rounded_step / power) * power + power


def generate_cmap_and_norm(
    min_val,
    max_val,
    levels,
    cmap_name="viridis",
    white_center=False,
    centered_on_zero=False,
):
    """Function to generate a colormap with specific values

    Generates a discrete colormap and normalization for a specified
    range and number of levels, with an option to center the colormap
    on zero and have equal step sizes ending in 0, 2, or 5.

    Parameters
    ----------
    min_val : float
        Minimum value of the range.
    max_val : float
        Maximum value of the range.
    levels : int
        Number of intervals or levels
        (must be odd if centered_on_zero is True).
    cmap_name : str, optional
        Name of an existing colormap, by default "viridis"
    white_center : bool
        Whether to set the middle interval color to white,
        by default False
    centered_on_zero : bool
        Whether to center the middle interval on zero,
        by default False

    Returns
    -------
    cmap : ListedColormap
    norm : BoundaryNorm instance
    """

    # Check if requested levels is consisted with options
    if centered_on_zero and levels % 2 == 0:
        raise ValueError("Levels must be an odd number when centering on zero.")

    # If centering on zero, adjust the min and max to be symmetrical around zero
    if centered_on_zero:
        max_abs_val = max(abs(min_val), abs(max_val))
        min_val, max_val = -max_abs_val, max_abs_val

    # Calculate the step size
    step = (max_val - min_val) / (levels - 1)

    # Round the step to the nearest number ending in 0, 2, or 5
    step = round_step(step)

    # Adjust min and max values to fit the new step size
    if centered_on_zero:
        # Ensure that min_val is a multiple of the step size away from zero
        min_val = -((levels - 1) // 2) * step
        max_val = ((levels - 1) // 2) * step
    else:
        min_val = np.floor(min_val / step) * step
        max_val = min_val + step * (levels - 1)

    # Generate the values and boundaries
    values = np.linspace(min_val, max_val, levels)
    boundaries = [val - step / 2 for val in values] + [max_val + step / 2]

    base_cmap = plt.get_cmap(cmap_name)
    colors = base_cmap(np.linspace(0, 1, levels))

    if white_center and centered_on_zero:
        # Set the middle interval color to white
        mid_index = (levels - 1) // 2
        colors[mid_index] = (1, 1, 1, 1)

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    return cmap, norm


def update_stats(
    var1,
    var2,
    diffvar,
    weights,
    ydim,
    xdim,
    ordinate,
    abscissa,
    singlepanel,
    plot_type,
    projection,
    splitscale,
    ax4=None,
    ax5=None,
    ax8=None,
):
    """
    Function update statistics, calculate RMS, and plot zonal/meridonal
    RMS panels

    Parameters
    ----------
    var1 : xarray.DataArray
        Input data variable (A)
    var2 : xarray.DataArray
        Input data variable (B)
    diffvar : xarray.DataArray
        Difference array (A minus B)
    weights : xarray.DataArray
        Weights array to use for area averaging
    ydim : str
        Name of y-dimension (image-relative, not geocentric)
    xdim : str
        Name of x-dimension (image-relative, not geocentric)
    ordinate : str
        Name of y-coordinate (image-relative, not geocentric)
    abscissa : str
        Name of x-coordinate (image-relative, not geocentric)
    singlepanel : bool
        Difference plot only if True, otherwise three-panel format
    plot_type : str
        Either "yx" or "yz"
    projection : None or cartopy.crs.Projection
        Map projection
    splitscale : float
        Split vertical scale (yz plots only)
    ax4 : matplotlib.axes.Axes, optional
        RHS RMS plot
    ax5 : matplotlib.axes.Axes, optional
        Bottom RMS plot
    ax8 : matplotlib.axes.Axes, optional
        Stats area axis
    """
    # Calculate RMSE
    var_se = diffvar**2
    var_rmse_xave = np.sqrt(var_se.weighted(weights).mean(xdim))
    var_rmse_yave = np.sqrt(var_se.weighted(weights).mean(ydim))
    rms = float(np.sqrt(var_se.weighted(weights).mean((ydim, xdim))))

    # Plot RMS "wings" only in three-panel mode
    if not singlepanel:
        if ax4 is not None:
            # Generate y nominal coordinates
            if len(diffvar[ordinate].shape) == 2:
                nominal_y = diffvar[ordinate].mean(xdim)
            else:
                nominal_y = diffvar[ordinate]

            # Plot the zonal (yx) or depth (yz) mean RMS difference
            if projection is None:
                ax4.plot(var_rmse_xave, nominal_y, color="k")
                ax4.set_ylim(nominal_y.min(), nominal_y.max())
                ax4.text(
                    0.5,
                    1.01,
                    "RMSE",
                    ha="center",
                    fontsize=8,
                    transform=ax4.transAxes,
                )
                if plot_type == "yz":
                    if splitscale is not None:
                        ax4.set_yscale("splitscale", zval=splitscale)
                        ax4.axhline(
                            y=splitscale[1],
                            color="black",
                            linewidth=0.5,
                            linestyle="dashed",
                        )
                    else:
                        ax4.invert_yaxis()
                ax4.set_yticks([])
            else:
                ax4.axis("off")

        # Plot the meridional (yx) or zonal (yz) mean RMS difference
        if ax5 is not None:
            # Generate x nominal coordinates
            if len(diffvar[ordinate].shape) == 2:
                nominal_x = diffvar[abscissa].mean(ydim)
            else:
                nominal_x = diffvar[abscissa]

            # Plot the meridional mean RMS difference
            if projection is None:
                ax5.plot(nominal_x, var_rmse_yave, color="k")
                ax5.set_xlim(nominal_x.min(), nominal_x.max())
                ax5.set_xticks([])
                ax5.yaxis.set_label_position("right")
                ax5.yaxis.tick_right()
            else:
                ax5.axis("off")

    # Add A & B stats boxes if in three panel mode
    if not singlepanel:
        add_stats_box(ax8, 0.0, 0.5, "Dataset A", float(var1.min()), float(var1.max()))
        add_stats_box(ax8, 0.15, 0.5, "Dataset B", float(var2.min()), float(var2.max()))

    # Add difference plot stats box
    add_stats_box(
        ax8,
        0.6,
        0.5,
        "Difference",
        float(diffvar.min()),
        float(diffvar.max()),
        float(diffvar.weighted(weights).mean((ydim, xdim))),
        rms,
        ha="left",
    )


def create_event_handler(
    ax,
    var1,
    var2,
    diffvar,
    weights,
    ydim,
    xdim,
    ordinate,
    abscissa,
    singlepanel,
    plot_type,
    projection,
    splitscale,
    ax4=None,
    ax5=None,
    ax8=None,
    event="zoom",
):
    def on_zoom(event):
        if event.inaxes == ax:  # Check if the event was on the correct axes
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            var1_sub = geoslice(var1, x=xlim, y=ylim)
            var2_sub = geoslice(var2, x=xlim, y=ylim)
            diffvar_sub = geoslice(diffvar, x=xlim, y=ylim)
            weights_sub = geoslice(weights, x=xlim, y=ylim)

            # Update the title or perform other actions based on the zoom
            # ax.set_title(f"Zoomed Range: x={xlim}, y={ylim}")

            # Clear ax8 and add new statistical information
            _ = ax8.clear()
            _ = ax8.axis("off")
            update_stats(
                var1_sub,
                var2_sub,
                diffvar_sub,
                weights_sub,
                ydim,
                xdim,
                ordinate,
                abscissa,
                singlepanel,
                plot_type,
                projection,
                splitscale,
                ax4=ax4,
                ax5=ax5,
                ax8=ax8,
            )

            plt.draw()

    def on_draw(event):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        var1_sub = geoslice(var1, x=xlim, y=ylim)
        var2_sub = geoslice(var2, x=xlim, y=ylim)
        diffvar_sub = geoslice(diffvar, x=xlim, y=ylim)
        weights_sub = geoslice(weights, x=xlim, y=ylim)

        # Update the title or perform other actions based on the zoom
        # ax.set_title(f"Zoomed Range: x={xlim}, y={ylim}")

        # Clear ax8 and add new statistical information
        _ = ax8.clear()
        _ = ax8.axis("off")

        if projection is None:
            update_stats(
                var1_sub,
                var2_sub,
                diffvar_sub,
                weights_sub,
                ydim,
                xdim,
                ordinate,
                abscissa,
                singlepanel,
                plot_type,
                projection,
                splitscale,
                ax4=None,
                ax5=None,
                ax8=ax8,
            )

        plt.draw()

    if event == "zoom":
        return on_zoom
    if event == "draw":
        return on_draw


def compare_2d(
    var1,
    var2,
    label1=None,
    label2=None,
    xrng=None,
    yrng=None,
    levels=17,
    clim=None,
    clim_diff=None,
    cmap1="viridis",
    cmap2="RdBu_r",
    stats=True,
    plot_type=None,
    splitscale=None,
    projection=None,
    singlepanel=False,
    dpi=150,
    topright_label="",
):
    """Compare two datasets on a set of matplotlib subplots.

    The function assumes that the two input arrays are on the same
    horizontal (xy) grid.

    Parameters
    ----------
    var1 : xarray.DataArray
        First dataset for comparison.
    var2 : xarray.DataArray
        Second dataset for comparison.
    label1 : str, optional
        Label for the first dataset.
    label2 : str, optional
        Label for the second dataset.
    xrng : tuple, optional
        Range for x-axis slicing (min, max).
    yrng : tuple, optional
        Range for y-axis slicing (min, max).
    levels : int, optional
        Number of levels for color mapping.
    clim : tuple, optional
        Color limit for the first dataset's plot.
    clim_diff : tuple, optional
        Color limit for the difference plot.
    cmap1 : str, optional
        Colormap for the first dataset's plot.
    cmap2 : str, optional
        Colormap for the difference plot.
    stats : bool, optional
        Calculate bias and RMS difference, by default True
    plot_type : str, optional
        Either "yx" or "yz", value inferred if missing
    splitscale : float, optional
        Depth value for vertical split scale, by default None
    projection : cartopy.crs.Projection, optional
        Cartopy projection, by default None
    dpi : float
        Dots per inch (dpi) value to use for image
    topright_label :

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object with the plots.
    """

    # Set the overall figure size, maintain a 16:9 aspect ratio, and set dpi
    fig = plt.figure(figsize=(16, 9), dpi=dpi, facecolor="white")

    # Create a grid layout.
    grid = plt.GridSpec(16, 16, figure=fig)

    # Define a facecolor for the subplots. This color will be used to fill in
    # missing values and be the deafult land color
    facecolor = "gray"

    # Define the subplot axes
    if singlepanel:
        # Used for a single-panel difference plot
        ax0 = fig.add_subplot(grid[0, :])  # Header
        ax3 = fig.add_subplot(
            grid[1:12, 1:16], facecolor=facecolor, projection=projection
        )  # Right
        ax4 = None
        ax5 = None
        ax7 = fig.add_subplot(grid[13:14, :])  # Right Colorbar
        ax8 = fig.add_subplot(grid[14:16, :])  # Info panel

    else:
        # The structure of the three-panel plot:
        #   The left side has plots of the two input variables that are stacked
        #   vertically. The right side has a difference plot that is larger
        #   in order to emphasize the difference between the two variables

        ax0 = fig.add_subplot(grid[0, :])  # Header
        ax1 = fig.add_subplot(
            grid[1:8, :6], facecolor=facecolor, projection=projection
        )  # Left Top
        ax2 = fig.add_subplot(
            grid[8:15, :6], facecolor=facecolor, projection=projection
        )  # Left Bottom
        ax3 = fig.add_subplot(
            grid[1:12, 6:15], facecolor=facecolor, projection=projection
        )  # Right
        ax4 = fig.add_subplot(grid[1:12, 15:16])  # Zonal Means
        ax5 = fig.add_subplot(grid[12:13, 6:15])  # Meridional Means
        ax6 = fig.add_subplot(grid[15:16, :6])  # Left Colorbar
        ax7 = fig.add_subplot(grid[13:14, 6:15])  # Right Colorbar
        ax8 = fig.add_subplot(grid[14:16, 6:])  # Info panel

    # Now that the projections are initialized, set the supplied cartopy
    # projection to PlateCarree to maximize plotting compatibility
    if projection is not None:
        import cartopy.crs as ccrs

        projection = ccrs.PlateCarree()

    # Turn off the bounding boxes for the header and footer space
    _ = ax0.axis("off")
    _ = ax8.axis("off")

    # Infer the names of the grid coordinates and dimensions
    if plot_type is None:
        _var = var1.reset_coords(drop=True)
        if "longitude" in _var.cf.coordinates:
            plot_type = "yx"
        elif "vertical" in _var.cf.coordinates:
            plot_type = "yz"
        else:
            raise ValueError("Unable to determine if this yx or yz plot")

    # The `_var` object is used below to infer coordinates. The `var1`
    # and `var2` objects have already been checked that they are identical
    # at this point

    _var = var1

    # The plots are 2-dimensional. Determine the coordinate variable names
    # that should be used for plotting
    if plot_type == "yx":
        abscissa = _var.cf.coordinates["longitude"][0]
        ordinate = _var.cf.coordinates["latitude"][0]

    elif plot_type == "yz":
        abscissa = _var.cf.coordinates["latitude"][0]
        ordinate = _var.cf.coordinates["vertical"][0]

    # Determine the plot-relative (not grid-relative) x and y dimension names
    ydim = _var.dims[-2]
    xdim = _var.dims[-1]

    # If a yz-plot is requested, determine the bottom level to use if setting
    # a "splitscale" level
    if plot_type == "yz":
        if splitscale is not None:
            zbot = float(min(_var[ydim].max(), 6500))
            splitscale = [zbot, splitscale, 0.0]

    # Attempt to determine the metadata from the first variable
    var_name = var1.name
    long_name = var1.long_name if "long_name" in var1.attrs.keys() else ""
    units = var1.units if "units" in var1.attrs.keys() else ""

    # Add annotations at the top of the figure for the variable metadata
    _ = ax0.text(
        0.0, 0.8, var_name, weight="bold", fontsize=16, transform=ax0.transAxes
    )
    _ = ax0.text(
        0.0,
        0.0,
        long_name,
        style="italic",
        fontsize=12,
        transform=ax0.transAxes,
    )

    # Set string labels for the input datasets, also define formatting
    # properties for the label boxes
    label1 = "" if label1 is None else f" {str(label1)}"
    label2 = "" if label2 is None else f" {str(label2)}"
    props = dict(
        boxstyle="round,pad=0.3", edgecolor="black", linewidth=1.5, facecolor="white"
    )

    # Drop portions if coordinates are NaNs. This can happen especially in yz plots
    # when selecting a domain (e.g. "atlarc")  The masking/averaging can introduce
    # NaNs in the coordinate variable
    var1 = var1.where(~var1[abscissa].isnull(), drop=True)
    var2 = var2.where(~var2[abscissa].isnull(), drop=True)

    # Subset a region of the variables if requested
    if isinstance(xrng, tuple) and isinstance(yrng, tuple):
        var1 = geoslice(var1, x=xrng, y=yrng)
        var2 = geoslice(var2, x=xrng, y=yrng)
        geosliced = True
    else:
        geosliced = False

    if not singlepanel:
        # The left-hand upper and lower panels are plotted in three-panel mode;
        # these panels are turned off if only a difference plot is requested.

        # Figure out the colorbar range based on data from both variables
        if clim is None:
            _ = np.concatenate((var1, var2))
            clim = (np.nanpercentile(_, 2), np.nanpercentile(_, 98))
        cmap, norm = generate_cmap_and_norm(*clim, levels, cmap_name=cmap1)

        # Top left panel - Dataset "A"
        if projection is not None:
            cb1 = ax1.pcolormesh(
                var1[abscissa],
                var2[ordinate],
                var1.values,
                cmap=cmap,
                norm=norm,
                transform=projection,
            )
            ax1.coastlines(linewidth=0.5)

        else:
            cb1 = ax1.pcolormesh(
                var1[abscissa], var2[ordinate], var1.values, cmap=cmap, norm=norm
            )

        ax1.text(
            0.05,
            0.05,
            f"A.{label1}",
            transform=ax1.transAxes,
            fontsize=8,
            ha="left",
            bbox=props,
        )

        # Bottom left panel - Dataset "B"
        if projection is not None:
            cb2 = ax2.pcolormesh(
                var2[abscissa],
                var2[ordinate],
                var2.values,
                cmap=cmap,
                norm=norm,
                transform=projection,
            )
            ax2.coastlines(linewidth=0.5)

        else:
            cb2 = ax2.pcolormesh(
                var2[abscissa], var2[ordinate], var2.values, cmap=cmap, norm=norm
            )

        ax2.text(
            0.05,
            0.05,
            f"B.{label2}",
            transform=ax2.transAxes,
            fontsize=8,
            ha="left",
            bbox=props,
        )

        # Add the colorbar
        fig.colorbar(cb1, cax=ax6, orientation="horizontal", extend="both", label=units)

    # Right Panel - Difference plot
    # Calculate a difference field
    # Determine range and colorbar for difference plot
    # Then plot the field and add annotations

    diffvar = var1 - var2

    if clim_diff is None:
        clim_diff = (np.nanpercentile(diffvar, 2), np.nanpercentile(diffvar, 98))

    try:
        levels_m1 = levels - 1 if levels % 2 == 0 else levels
        cmap, norm = generate_cmap_and_norm(
            *clim_diff,
            levels_m1,
            cmap_name=cmap2,
            centered_on_zero=True,
            white_center=True,
        )
    except Exception as exc:
        warnings.warn(str(exc))
        cmap = cmap2
        norm = None

    if projection is not None:
        cb3 = ax3.pcolormesh(
            diffvar[abscissa],
            diffvar[ordinate],
            diffvar.values,
            cmap=cmap,
            norm=norm,
            transform=projection,
        )
        ax3.coastlines(linewidth=0.5)

        if geosliced:
            ax3.set_extent([*xrng, *yrng], crs=projection)

    else:
        cb3 = ax3.pcolormesh(
            diffvar[abscissa],
            diffvar[ordinate],
            diffvar.values,
            cmap=cmap,
            norm=norm,
        )

    ax3.text(
        0.05,
        0.05,
        "Difference (A minus B)",
        transform=ax3.transAxes,
        fontsize=12,
        ha="left",
        bbox=props,
    )

    ax3.text(
        0.98,
        1.03,
        topright_label,
        transform=ax3.transAxes,
        fontsize=10,
        style="italic",
        ha="right",
    )

    # Add the colorbar for the difference plot
    fig.colorbar(cb3, cax=ax7, orientation="horizontal", extend="both", label=units)

    axes = [ax3] if singlepanel else [ax1, ax2, ax3]

    if plot_type == "yz":
        for ax in axes:
            if splitscale is not None:
                ax.set_yscale("splitscale", zval=splitscale)
                ax.axhline(
                    y=splitscale[1], color="black", linewidth=0.5, linestyle="dashed"
                )
            else:
                ax.invert_yaxis()

        plt.subplots_adjust(wspace=2.0, hspace=1.5)

    else:
        # Remove ticks and labels
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.subplots_adjust(wspace=0.7, hspace=0.8)

    if plot_type == "yz":
        zbounds = infer_bounds(diffvar[ordinate].values, 0, None)
        dz = zbounds[1:] - zbounds[0:-1]

        ybounds = infer_bounds(diffvar[abscissa].values, -90, 90)
        ybounds = np.radians(ybounds)

        upper_y = ybounds[1:]
        lower_y = ybounds[0:-1]

        dy = (
            2.0
            * 6371.0e3
            * np.arcsin(np.sqrt(np.sin((upper_y - lower_y) / 2.0) ** 2.0))
        )

        weights = xr.DataArray(
            dz[:, np.newaxis] * dy, dims=diffvar.dims, coords=diffvar.coords
        )

    else:
        # Infer the cell area name
        area = var1.cf.standard_names["cell_area"][0]
        weights = var1[area]

    if stats:
        update_stats(
            var1,
            var2,
            diffvar,
            weights,
            ydim,
            xdim,
            ordinate,
            abscissa,
            singlepanel,
            plot_type,
            projection,
            splitscale,
            ax4=ax4,
            ax5=ax5,
            ax8=ax8,
        )

    else:
        if not singlepanel:
            ax4.axis("off")
            ax5.axis("off")

    if singlepanel and plot_type == "yx":
        draw_handler = create_event_handler(
            ax3,
            var1,
            var2,
            diffvar,
            weights,
            ydim,
            xdim,
            ordinate,
            abscissa,
            singlepanel,
            plot_type,
            projection,
            splitscale,
            ax4=ax4,
            ax5=ax4,
            ax8=ax8,
            event="draw",
        )

        if projection is not None:
            fig.canvas.mpl_connect("button_release_event", draw_handler)

        else:
            fig.canvas.mpl_connect("draw_event", draw_handler)

    return fig
