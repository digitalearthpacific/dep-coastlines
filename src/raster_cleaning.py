"""
These are functions needed for raster cleaning, similar to coastlines.raster.
So far everything is inspired / taken from the Digital Earth Austalia 
(https://github.com/GeoscienceAustralia/dea-coastlines) and
Digital Earth Africa (https://github.com/digitalearthafrica/deafrica-coastlines)
coastline work, so much credit to those projects / authors for much of what 
is below.

We are not using those functions directly at this point because I wanted to 
dask-enable some things that were not previously dask enabled (see, for
instance, temporal_masking), and I am sure there will be modifications / 
additions to the workflow which are study area specific to the Pacific.
"""

from pathlib import Path

import flox.xarray  # <- not positive this is in the planetary computer image
from geopandas import GeoDataFrame
from retry import retry
from shapely import make_valid
from skimage.measure import label
from skimage.morphology import binary_dilation, disk
import xarray as xr
from xarray import DataArray, Dataset


def get_coastal_mask(areas: GeoDataFrame) -> GeoDataFrame:
    land_plus = areas.buffer(1000)
    land_minus = areas.buffer(-1000)
    return make_valid(land_plus.difference(land_minus).unary_union)


def temporal_masking(ds: DataArray) -> DataArray:
    """Dask-enabled version of coastlines.vector.temporal_masking from
    the deafrica-coastlines (and dea-coastlines) work."""

    # We could use xrspatial.zonal.regions but it's not xarray or dask aware yet
    zones = xr.apply_ufunc(label, ds, None, 0, dask="parallelized")

    neighbours = (
        ds.shift(year=-1, fill_value=False) | ds.shift(year=1, fill_value=False)
    ).astype("int8")

    # deafrica code does the following steps in a different way. I chose this
    # because it is less memory intensive and seems to work (betterish at least)
    # with dask.

    # I got out of memory errors with this
    # zone_maxes = stats(zones, neighbours, stats_funcs=["max"])

    # xarray docs say that flox is used by default if it is loaded, I did
    # not find that to be the case. If it did work you could just do
    # zone_maxes = neighbours.groupby(zones).max()
    return flox.xarray.xarray_reduce(neighbours, by=zones, func="max")


@retry(tries=20, delay=10)
def contours_preprocess(
    yearly_ds: Dataset,
    gapfill_ds: Dataset,
    water_index: str = "nir08",
    index_threshold: float = 128.0,
    mask_temporal: bool = True,
) -> Dataset | DataArray:
    # Remove low obs pixels and replace with 3-year gapfill
    combined_ds = yearly_ds.where(yearly_ds["count"] > 5, gapfill_ds)

    # Set any pixels with only one observation to NaN, as these
    # are extremely vulnerable to noise
    combined_ds = combined_ds.where(yearly_ds["count"] > 1)

    # Apply water index threshold and re-apply nodata values
    nodata = combined_ds[water_index].isnull()
    thresholded_ds = combined_ds[water_index] < index_threshold
    thresholded_ds = thresholded_ds.where(~nodata)

    # The threshold here should checked out a bit more. It varies for Australia
    # and Africa, and for the shortertime period we are starting with (i.e.
    # 2013-2023) a stray year could me amplified if other years are missing
    # all_time_land = thresholded_ds.mean(dim="year") > 1 / 3.0

    if mask_temporal:
        # Create a temporal mask by identifying land pixels with a direct
        # spatial connection (e.g. contiguous) to land pixels in either the
        # previous or subsequent timestep.

        # This is used to clean up noisy land pixels (e.g. caused by clouds,
        # white water, sensor issues), as these pixels typically occur
        # randomly with no relationship to the distribution of land in
        # neighbouring timesteps. True land, however, is likely to appear
        # in proximity to land before or after the specific timestep.

        # Compute temporal mask
        temporal_mask = temporal_masking(thresholded_ds == 1)

        # Set any pixels outside mask to 0 to represent water
        thresholded_ds = thresholded_ds.where(temporal_mask, 0)

    thresholded_ds = xr.apply_ufunc(
        binary_dilation,
        thresholded_ds.to_dataset("year"),
        disk(5),
        dask="allowed",
    ).to_array("year")
    return combined_ds[water_index].where(thresholded_ds)  # == 1?
