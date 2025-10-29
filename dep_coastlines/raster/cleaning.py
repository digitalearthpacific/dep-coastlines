import numpy as np
import rioxarray as rx
import xarray as xr
import xrspatial as xs
from rasterio.warp import transform_bounds
from scipy.ndimage import gaussian_filter
from skimage.measure import label
from xarray import DataArray, Dataset, apply_ufunc

from dep_coastlines.grid import remote_aoi_raster_path


def smooth_gaussian(da: DataArray, sigma: float = 0.799) -> DataArray:
    """Apply a 3x3 gaussian kernel to a DataArray.

    The convolution ignores nan values by using the kernel values for
    the non-nan pixels only, and scaling the divisor accordingly. A sigma
    value of 0.799 represents a "classic" 3x3 kernel with a center weight
    of approximately 1/4, side weights of approximately 1/8, and corner weights of
    approximately 1/16. Lower values more heavily weight the center.

    Edge pixels (those without a full 3x3 window) will be set to NaN.

    Args:
        da: A 2-d DataArray with dimensions `"x"` & `"y"`.
        sigma: Value for the sigma parameter as passed to
            :py:func:`scipy.ndimage.gaussian_filter`.

    Returns:
        The input DataArray with the smoothing applied.
    """

    def kernel(sigma: float):
        """Returns a 2D Gaussian kernel array."""
        input = np.zeros((3, 3))
        input[1, 1] = 1
        return gaussian_filter(input, sigma)

    weights = DataArray(kernel(sigma), dims=("xw", "yw"))
    total = (
        da.fillna(0)
        .rolling(x=3, y=3, center=True)
        .construct(x="xw", y="yw")
        .dot(weights)
    )
    divisor = (
        (~da.isnull())
        .astype(int)
        .rolling(x=3, y=3, center=True)
        .construct(x="xw", y="yw")
        .dot(weights)
    )
    return (total / divisor).where(~da.isnull())


# def find_disconnected_areas(core, candidate):


def remove_disconnected_land(
    certain_land: DataArray, candidate_land: DataArray
) -> DataArray:
    """Remove pixels from `candidate_land` that are not connected to `certain_land`.

    The implementation is to first identify unique land regions and use them as zones,
    calculating the maximum value of certain_land (i.e. 1 or 0) within each. Regions
    with a max of 1 are kept.

    This is similar to but more restrictive than :func:`coastlines.vector.temporal_masking`.

    Args:
        certain_land: A boolean DataArray where true values represent known land
            areas.
        candidate_land: A boolean DataArray where true values represent locations
            we think could be land, if it is connected to known land areas.

    Returns:
        `candidate_land` with any pixels not connected to `certain_land` removed.
    """

    if "year" in candidate_land.dims and candidate_land.year.size > 1:
        return candidate_land.groupby("year").map(
            lambda candidate_year: remove_disconnected_land(
                certain_land, candidate_year.squeeze(drop=True)
            )
        )

    zones = apply_ufunc(
        label, candidate_land, None, 0, dask="parallelized", kwargs=dict(connectivity=1)
    )

    connected_or_not = xs.zonal_stats(
        zones, certain_land.astype("int8"), stats_funcs=["max"]
    )
    connected_zones = connected_or_not["zone"][connected_or_not["max"] == 1]
    return candidate_land.where(zones.isin(connected_zones)) == 1


def find_inland_areas(water_bool_da, ocean_bool_da) -> DataArray:
    ocean_10 = ocean_bool_da.astype("int8").compute()

    def _find_inland_2d(bool_da_2d: DataArray) -> DataArray:
        water_zones = xr.full_like(bool_da_2d, 0, dtype="int16")
        water_zones.values = label(bool_da_2d.astype("int8"), background=0)
        location_by_zone = xs.zonal_stats(
            water_zones.where(water_zones > 0), ocean_10, stats_funcs=["max"]
        )
        inland_zones = location_by_zone["zone"][location_by_zone["max"] == 0]
        return water_zones.isin(inland_zones)

    # Can't do this in chunks because the labels would be repeated across chunks.
    # but could parallelize across years I think
    return water_bool_da.groupby("year").apply(
        lambda da: _find_inland_2d(da.squeeze(drop=True))
    )


def fill_with_nearby_dates(xarr: DataArray | Dataset) -> DataArray | Dataset:
    """Fill missing values with those from the prior or following year.

    If both adjacent years are missing or null, the output will be null.
    If both adjacent years have data, the earlier year's is used.

    Args:
        xarr: An input xarray object with a "year" dimension, which is a string
            representation of a 4-digit year.

    Returns:
        The input with null values filled with values from adjacent years. If `xarr`
        is a :class:`DataArray` object, the same is returned.
    """

    def fill(da: DataArray) -> DataArray:
        output = da.to_dataset("year")
        for year in da.year.values:
            output[year] = da.sel(year=year)
            intyear = int(year)
            years = [
                str(y) for y in [intyear + 1, intyear - 1] if str(y) in da.year.values
            ]
            for inner_year in years:
                output[year] = output[year].where(
                    ~output[year].isnull(), output[inner_year]
                )
        return output.to_array(dim="year")

    return xarr.apply(fill) if isinstance(xarr, Dataset) else fill(xarr)
