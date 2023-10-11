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

# import flox.xarray  # <- not positive this is in the planetary computer image
import odc.algo
from retry import retry
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_bounds
import rioxarray as rx
from skimage.measure import label
import xarray as xr
import xrspatial as xs
from xarray import DataArray, Dataset


@retry(RasterioIOError, tries=10, delay=3)
def contours_preprocess(
    yearly_ds: Dataset,
    gapfill_ds: Dataset,
    water_index: str = "mndwi",
    masking_index: str = "mndwi",
    masking_threshold: float = 0,
    mask_ephemeral_land: bool = True,
    mask_ephemeral_water: bool = True,
    mask_esa_water_land: bool = True,
    mask_nir: bool = True,
    remove_tiny_areas: bool = True,
    remove_inland_water: bool = True,
    remove_water_noise: bool = True,
) -> Union[Dataset, DataArray]:
    # Remove low obs pixels and replace with 3-year gapfill
    yearly_ds = yearly_ds.where(yearly_ds["count"] > 5, gapfill_ds)

    # Set any pixels with only one observation to NaN, as these
    # are extremely vulnerable to noise
    yearly_ds = yearly_ds.where(yearly_ds["count"] > 1)

    # Apply water index threshold and re-apply nodata values
    # Here we use both the thresholded nir08 and mndwi to define land. They must
    # both agree. This helps to remove both water noise from mndwi and surf
    # (and other) artifacts from nir08. (I'll also add, in general, nir08 is more
    # dependable at identifying land than mndwi.)
    land_mask = yearly_ds[masking_index] < masking_threshold

    gadm_land = load_gadm_land(yearly_ds)
    gadm_core_land = odc.algo.mask_cleanup(gadm_land, mask_filters=[("erosion", 15)])
    full_land_mask = land_mask | gadm_core_land

    if mask_nir:
        nir08_threshold = -1280.0
        nir08_land = yearly_ds["nir08"] < nir08_threshold
        land_mask = land_mask & nir08_land

    nodata = yearly_ds[water_index].isnull()
    analysis_mask = land_mask.where(~nodata, False)
    analysis_mask = odc.algo.mask_cleanup(analysis_mask, mask_filters=[("dilation", 3)])

    if mask_esa_water_land:
        esa_water_land = load_esa_water_land(yearly_ds)
        esa_ocean = esa_water_land == 0
        close_to_coast = ~odc.algo.mask_cleanup(esa_ocean, [("erosion", 5)])
        analysis_mask = analysis_mask & close_to_coast

    if mask_ephemeral_land:
        ephemeral_land = ephemeral_areas(full_land_mask)
        analysis_mask = analysis_mask & ~ephemeral_land

    if mask_ephemeral_water:
        ephemeral_water = ephemeral_areas(~full_land_mask)
        analysis_mask = analysis_mask & ~ephemeral_water

    gadm_land = load_gadm_land(yearly_ds)
    if remove_inland_water:
        gadm_ocean = ~gadm_land
        inland_water = find_inland_areas(~land_mask, gadm_ocean)
        analysis_mask = analysis_mask & ~inland_water

    if remove_tiny_areas:
        analysis_mask = analysis_mask & ~small_areas(analysis_mask)

    if remove_water_noise:
        # Identify and remove water noise. Basically areas which mndwi says are land
        # which nir08 thinks are probably not, and esa says are not too
        # usual cutoff is -1280, I relax a bit to soften the impact on true coastal
        # areas.
        esa_water_land = load_esa_water_land(yearly_ds)
        esa_ocean = esa_water_land == 0
        water_noise = (yearly_ds.mndwi < 0) & (yearly_ds.nir08 > -800) & (esa_ocean)
        # The choice to be made is whether to simply mask out the water areas, or
        # recode. The recoding is not the best, and we should probably mask out when
        # we are certain we are only getting noise. Otherwise we remove some usable areas.
        # thats_water = 100
        # yearly_ds = yearly_ds.where(~water_noise, thats_water)
        analysis_mask = analysis_mask & ~water_noise

    all_time_land = analysis_mask.mean(dim="year") >= 0.15
    all_time_land = all_time_land | gadm_core_land

    inland = odc.algo.mask_cleanup(all_time_land, mask_filters=[("erosion", 15)])
    deep_ocean = odc.algo.mask_cleanup(~all_time_land, mask_filters=[("erosion", 15)])

    analysis_mask = analysis_mask & ~inland & ~deep_ocean

    return yearly_ds[water_index].where(analysis_mask)


def find_inland_areas(water_bool_da, ocean_bool_da) -> DataArray:
    def _find_inland_2d(bool_da_2d: DataArray) -> DataArray:
        water_zones = xr.full_like(bool_da_2d, 0, dtype="int16")
        water_zones.values = label(bool_da_2d.astype("int8"), background=0)
        location_by_zone = xs.zonal_stats(
            water_zones, ocean_bool_da.astype("int8").compute(), stats_funcs=["max"]
        )
        inland_zones = location_by_zone["zone"][location_by_zone["max"] == 0]
        return water_zones.isin(inland_zones)

    # Can't do this in chunks because the labels would be repeated across chunks.
    # but could parallelize across years I think
    return water_bool_da.groupby("year").apply(_find_inland_2d)


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

    return bool_da.groupby("year").apply(_remove_2d)


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

    return ~bool_da.groupby("year").apply(_temporal_masking_2d)


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


def load_gadm_land(ds: Dataset) -> DataArray:
    # This is a rasterized version of gadm. It seems better than any ESA product
    # at defining land (see for instance Vanuatu).
    input_path = "https://deppcpublicstorage.blob.core.windows.net/output/aoi/aoi.tif"
    land = rx.open_rasterio(input_path, chunks=True).rio.write_crs(8859)
    bounds = list(transform_bounds(ds.rio.crs, land.rio.crs, *ds.rio.bounds()))
    return land.rio.clip_box(*bounds).squeeze().rio.reproject_match(ds).astype(bool)
