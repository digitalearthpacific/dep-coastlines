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

from rasterio.warp import transform_bounds
import rioxarray as rx
from skimage.measure import label
import xarray as xr
import xrspatial as xs
from xarray import DataArray, Dataset


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


def load_gadm_land(ds: Dataset) -> DataArray:
    # This is a rasterized version of gadm. It seems better than any ESA product
    # at defining land (see for instance Vanuatu).
    input_path = "https://deppcpublicstorage.blob.core.windows.net/output/aoi/aoi.tif"
    land = rx.open_rasterio(input_path, chunks=True).rio.write_crs(8859)
    bounds = list(transform_bounds(ds.rio.crs, land.rio.crs, *ds.rio.bounds()))
    return land.rio.clip_box(*bounds).squeeze().rio.reproject_match(ds).astype(bool)
