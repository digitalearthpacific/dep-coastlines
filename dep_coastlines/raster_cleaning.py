"""These are functions needed for raster cleaning, similar to coastlines.raster.
So far everything is inspired / taken from the Digital Earth Austalia 
(https://github.com/GeoscienceAustralia/dea-coastlines) and
Digital Earth Africa (https://github.com/digitalearthafrica/deafrica-coastlines)
coastline work, so much credit to those projects / authors for much of what 
is below.

We are not using those functions directly at this point because of efforts
dask-enable some things that were not previously dask enabled (see, for
instance, temporal_masking), and I am sure there will be modifications / 
additions to the workflow which are study area specific to the Pacific.
"""

import numpy as np
from rasterio.warp import transform_bounds
import rioxarray as rx
from skimage.measure import label
from scipy.ndimage import gaussian_filter
import xarray as xr
import xrspatial as xs
from xarray import apply_ufunc, DataArray, Dataset

from dep_coastlines.grid import remote_aoi_raster_path

BooleanDataArray = DataArray


def smooth_gaussian(da: DataArray, sigma: float = 0.799) -> DataArray:
    """Apply a 3x3 gaussian kernel to the input. The convolution ignores nan values
    by using the kernel values for the non-nan pixels only, and scaling the
    divisor accordingly.

    A sigma value of 0.799 represents a "classic" 3x3 kernel with a center weight
    of approximately 1/4, side weights of approximately 1/8, and corner weights of
    approximately 1/16. Lower values more heavily weight the center.
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


def remove_disconnected_land(
    certain_land: BooleanDataArray, candidate_land: BooleanDataArray
):
    """Remove pixels from `candidate_land` that are not connected to `certain_land`.
    The algorithm is to first identify unique land regions and use them as zones,
    calculating the maximum value of certain_land (i.e. 1 or 0) within each. Regions
    with a max of 1 are kept.

    This is similar to, but more restrictive than the `coastlines.vector.temporal_masking`
    function.
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


def fill_with_nearby_dates(xarr: DataArray | Dataset) -> DataArray:
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


def small_areas(bool_da: DataArray, min_size_in_pixels: int = 55) -> DataArray:
    def _remove_2d(bool_da_2d: DataArray) -> DataArray:
        # For now, ok doing this in chunks, but could remove some bigger areas
        # if they span chunks
        zones = xr.apply_ufunc(label, bool_da_2d, None, 0, dask="parallelized")
        size_by_zone = xs.zonal_stats(
            zones, bool_da_2d.astype(int), stats_funcs=["sum"]
        )
        small_zones = size_by_zone["zone"][size_by_zone["sum"] < min_size_in_pixels]
        return bool_da_2d.where(zones.isin(small_zones) == 1, False)

    return bool_da.groupby("year").apply(lambda da: _remove_2d(da.squeeze(drop=True)))


def ephemeral_areas(bool_da: DataArray) -> DataArray:
    """Dask-enabled version of coastlines.vector.temporal_masking from
    the deafrica-coastlines (and dea-coastlines) work. I renamed it so it
    was clear what the return was.
    """

    # Create a temporal mask by identifying true pixels with a direct
    # spatial connection (e.g. contiguous) to true pixels in either the
    # previous or subsequent timestep.

    # This is used to clean up noisy land pixels (e.g. caused by clouds,
    # white water, sensor issues), as these pixels typically occur
    # randomly with no relationship to the distribution of land in
    # neighbouring timesteps. True land, however, is likely to appear
    # in proximity to land before or after the specific timestep.

    def _temporal_masking_2d(da_year):
        zones = xr.apply_ufunc(label, da_year, None, 0, dask="parallelized")

        # neighbours is 1 if the pixel was True in the prior or next year
        neighbours = (
            (
                bool_da.shift(year=-1, fill_value=False)
                | bool_da.shift(year=1, fill_value=False)
            )
            .sel(year=da_year.year)
            .astype("int8")
        )

        # a zone with any pixel which was True in the year before after will
        # have value = 1
        location_by_zone = xs.zonal_stats(
            zones, neighbours.astype("int8"), stats_funcs=["max"]
        )

        stable_zones = location_by_zone["zone"][location_by_zone["max"] == 1]
        return zones.isin(stable_zones)

    return ~bool_da.groupby("year").apply(
        lambda da: _temporal_masking_2d(da.squeeze(drop=True))
    )


def load_gadm_land(ds: Dataset) -> DataArray:
    # This is a rasterized version of gadm. It seems better than any ESA product
    # at defining land (see for instance Vanuatu).
    land = rx.open_rasterio(f"s3://{remote_aoi_raster_path}", chunks=True)
    bounds = list(transform_bounds(ds.rio.crs, land.rio.crs, *ds.rio.bounds()))
    return land.rio.clip_box(*bounds).squeeze().rio.reproject_match(ds).astype(bool)
