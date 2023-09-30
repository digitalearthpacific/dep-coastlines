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

from typing import Union

import flox.xarray  # <- not positive this is in the planetary computer image
from geopandas import GeoDataFrame
import numpy as np
import odc.algo
from retry import retry
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_bounds
import rioxarray as rx
from shapely import make_valid
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk
import xarray as xr
import xrspatial as xs
from xarray import DataArray, Dataset


def get_coastal_mask(areas: GeoDataFrame) -> GeoDataFrame:
    land_plus = areas.buffer(1000)
    land_minus = areas.buffer(-1000)
    return make_valid(land_plus.difference(land_minus).unary_union)


def remove_small_areas(bool_ds: DataArray, min_size_in_pixels: int = 55) -> DataArray:
    def _remove_2d(bool_ds_2d: DataArray) -> DataArray:
        zones = xr.apply_ufunc(label, bool_ds_2d, None, 0, dask="parallelized")
        size_by_zone = xs.zonal_stats(
            zones, bool_ds_2d.astype(int), stats_funcs=["sum"]
        )
        big_zones = size_by_zone["zone"][size_by_zone["sum"] >= min_size_in_pixels]
        return bool_ds_2d.where(zones.isin(big_zones) == 1)

    return xr.concat(
        [_remove_2d(bool_ds.sel(year=year)) for year in bool_ds.year], dim="year"
    )


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


def load_esa_water_land(ds: Dataset) -> DataArray:
    # This is from http://maps.elie.ucl.ac.be/CCI/viewer/download.php
    # See Lamarche, C.; Santoro, M.; Bontemps, S.; Dâ€™Andrimont, R.; Radoux, J.; Giustarini, L.; Brockmann, C.; Wevers, J.; Defourny, P.; Arino, O. Compilation and Validation of SAR and Optical Data Products for a Complete and Global Map of Inland/Ocean Water Tailored to the Climate Modeling Community. Remote Sens. 2017, 9, 36. https://doi.org/10.3390/rs9010036
    input_path = "https://deppcpublicstorage.blob.core.windows.net/output/src/ESACCI-LC-L4-WB-Ocean-Land-Map-150m-P13Y-2000-v4.0-8859.tif"

    # In theory we could just use the `crs` arg of clip_box but that wasn't working
    # for me with CRS 8859. This did.

    # Also, if you're wondering why I reprojected to CRS 8859 it's because I couldn't
    # reproject from 4326 for tiles that crossed the antimeridian.
    # If you're wondering why I set the CRS manually it's because it's not being
    # read from teh asset for some reason.
    water_land = rx.open_rasterio(input_path, chunks=True).rio.write_crs(8859)
    bounds = list(transform_bounds(ds.rio.crs, water_land.rio.crs, *ds.rio.bounds()))
    return water_land.rio.clip_box(*bounds).squeeze().rio.reproject_match(ds)


@retry(RasterioIOError, tries=10, delay=3)
def contours_preprocess(
    yearly_ds: Dataset,
    gapfill_ds: Dataset,
    water_index: str = "nir08",
    index_threshold: float = 0,
    masking_index: str = "nir08",
    mask_temporal: bool = True,
    mask_esa_water_land: bool = True,
    remove_tiny_areas: bool = True,
) -> Union[Dataset, DataArray]:
    # Remove low obs pixels and replace with 3-year gapfill
    combined_ds = yearly_ds.where(yearly_ds["count"] > 5, gapfill_ds)

    # Set any pixels with only one observation to NaN, as these
    # are extremely vulnerable to noise
    combined_ds = combined_ds.where(combined_ds["count"] > 1)

    # Identify and remove water noise. Basically areas which mndwi says are land
    # which nir08 thinks are probably not, and esa says are not too
    esa_water_land = load_esa_water_land(yearly_ds)
    esa_ocean = esa_water_land == 0
    # usual cutoff is -1280, I relax a bit to soften the impact on true coastal
    # areas.
    water_noise = (combined_ds.mndwi < 0) & (combined_ds.nir08 > -800) & (esa_ocean)
    # The choice to be made is whether to simply mask out the water areas, or
    # recode. The recoding is not the best, and we should probably mask out when
    # we are certain we are only getting noise. Otherwise we remove some usable areas.
    thats_water = 100
    combined_ds = combined_ds.where(~water_noise, thats_water)

    # Apply water index threshold and re-apply nodata values
    # Here we use both the thresholded nir08 and mndwi to define land. They must
    # both agree. This helps to remove both water noise from mndwi and surf
    # (and other) artifacts from nir08. (I'll also add, in general, nir08 is more
    # dependable at identifying land than mndwi.)
    land_mask = combined_ds[masking_index] < index_threshold
    other_land_mask = combined_ds["mndwi"] < 0.0
    land_mask = land_mask.where(other_land_mask, 0)

    nodata = combined_ds[masking_index].isnull()
    land_mask = land_mask.where(~nodata)

    # The threshold here should checked out a bit more. It varies for Australia
    # and Africa, and for the shortertime period we are starting with (i.e.
    # 2013-2023) a stray year could me amplified if other years are missing
    # all_time_land = land_mask.mean(dim="year") > (1 / 4.0)
    # land_mask = land_mask.where(all_time_land, 0)

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
        temporal_mask = temporal_masking(land_mask == 1)

        # Set any pixels outside mask to 0 to represent water
        land_mask = land_mask.where(temporal_mask, 0)

    if mask_esa_water_land:
        close_to_coast = odc.algo.mask_cleanup(esa_ocean, [("erosion", 10)])
        land_mask = land_mask.where(close_to_coast)

    if remove_tiny_areas:
        # This was created mainly for surf artifacts in nir08, and may not be
        # needed if using e.g. mndwi
        land_mask = land_mask.where(remove_small_areas(land_mask == 1) == 1, 0)

    land_mask = odc.algo.mask_cleanup(land_mask == 1, [("dilation", 3)])
    return combined_ds[water_index].where(land_mask)
