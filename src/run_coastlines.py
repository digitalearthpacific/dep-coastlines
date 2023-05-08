from pathlib import Path

from typing import Callable, Union

from dask.distributed import Client, Lock
from dea_tools.spatial import subpixel_contours
from geopandas import GeoDataFrame
import rasterio
import rioxarray
from xarray import DataArray, Dataset

# local submodules
from coastlines.raster import pixel_tides, tide_cutoffs, export_annual_gapfill
from coastlines.vector import contours_preprocess, coastal_masking
from dep_tools.Processor import Processor
from dep_tools.utils import make_geocube_dask

from constants import STORAGE_AOI_PREFIX
from landsat_utils import item_collection_for_pathrow, mask_clouds


def mndwi(xr: DataArray) -> DataArray:
    # modified normalized differential water index is just a normalized index
    # like NDVI, with different bands
    green = xr.sel(band="green")
    swir = xr.sel(band="swir16")
    #    return xrspatial.multispectral.ndvi(green, swir).rename("mndwi")
    mndwi = (green - swir) / (green + swir)
    return mndwi.rename("mndwi")


def filter_by_cutoffs(
    ds: Dataset,
    tide_cutoff_min: Union[int, float, DataArray],
    tide_cutoff_max: Union[int, float, DataArray],
) -> Dataset:
    """
    coastline.raster.load_tidal_subset that doesn't load
    """
    # Determine what pixels were acquired in selected tide range, and
    # drop time-steps without any relevant pixels to reduce data to load
    tide_bool = (ds.tide_m >= tide_cutoff_min) & (ds.tide_m <= tide_cutoff_max)
    ds = ds.sel(time=tide_bool.sum(dim=["x", "y"]) > 0)

    # Apply mask
    return ds.where(tide_bool)


def coastlines_by_year(
    xr: DataArray,
) -> DataArray:  # , land_areas: GeoDataFrame) -> DataArray:
    #    xr = xr.rio.clip(land_areas.boundary.buffer(1000), all_touched=True, from_disk=True)

    working_ds = mndwi(xr).to_dataset()

    tides_lowres = pixel_tides(working_ds, resample=False).transpose("time", "y", "x")
    working_ds["tide_m"] = tides_lowres.rio.reproject_match(
        working_ds, rasterio.enums.Resampling.bilinear
    )

    tide_cutoff_min, tide_cutoff_max = tide_cutoffs(
        working_ds, tides_lowres, tide_centre=0.0
    )

    # Should also filter by land here!
    working_ds = filter_by_cutoffs(working_ds, tide_cutoff_min, tide_cutoff_max).drop(
        "tide_m"
    )

    # filter by land

    # okay this is contrived, really we just need the buffer, as
    # we don't yet have an inland water layer (but could possibly use wofs??)

    year_ds = working_ds.sel(time="2015")
    median_ds = year_ds.median(dim="time", keep_attrs=True)
    median_ds["stdev"] = year_ds.mndwi.std(dim="time", keep_attrs=True)
    median_ds["count"] = year_ds.mndwi.count(dim="time", keep_attrs=True).astype(
        "int16"
    )

    working_da = working_ds.to_array().squeeze(drop=True)
    median_ds["mndwi_3year"] = working_da.median(dim="time", keep_attrs=True)
    median_ds["stdev_3year"] = working_da.std(dim="time", keep_attrs=True)
    median_ds["count_3year"] = working_da.count(dim="time", keep_attrs=True).astype(
        "int16"
    )
    breakpoint()
    return median_ds

    # geodata ds (see coastlines.vector) specifies value of 0 for ocean,
    # 1 for mainland and 2 for island, but I'm not sure 1 vs 2 is being
    # used for anything
    like = (
        working_ds.mndwi.isel(time=0)
        .squeeze(drop=True)
        .assign_coords(dict(value=1))
        .expand_dims("value")
    )

    land_areas[["value"]] = 1
    coastal_da = make_geocube_dask(land_areas, ["value"], like, fill=0).sel(value=1)

    year_ds = working_ds.groupby("time.year")
    yearly_ds = year_ds.median()
    yearly_ds["stdev"] = year_ds.std(dim="time").to_array().squeeze(drop=True)
    yearly_ds["count"] = (
        year_ds.count(dim="time").to_array().squeeze(drop=True).astype("int16")
    )

    gapfill_ds = working_ds.median(dim="time", keep_attrs=True)
    gapfill_ds["stdev"] = working_ds.mndwi.std(dim="time", keep_attrs=True)
    gapfill_ds["count"] = working_ds.mndwi.count(dim="time", keep_attrs=True).astype(
        "int16"
    )

    # I believe contours_preprocess needs these loaded
    # (well I got errors from apply_ufunc when it wasn't)
    yearly_ds = yearly_ds.load()
    gapfill_ds = gapfill_ds.load()
    coastal_da = coastal_da.load()

    # masked_ds, certainty_masks = contours_preprocess(
    # get all intermediates for debugging
    (
        masked_ds,
        certainty_masks,
        all_time,
        all_time_clean,
        river_mask,
        ocean_da,
        coastal_mask,
        inland_mask,
        thresholded_ds,
        annual_mask,
    ) = contours_preprocess(
        yearly_ds.load(),
        gapfill_ds.load(),
        water_index="mndwi",
        index_threshold=0.0,
        coastal_da=coastal_da.load(),
        debug=True,
    )

    # issues above with annual_mask, so try this
    combined_ds = yearly_ds.where(yearly_ds["count"] > 5, gapfill_ds)
    best_guess = combined_ds.mndwi.where(coastal_mask)

    return subpixel_contours(
        da=best_guess,
        min_vertices=10,
        dim="year",
    ).set_index("year")


def run_processor(
    year: int,
    scene_processor: Callable,
    **kwargs,
):
    processor = Processor(year, scene_processor, **kwargs)
    # cluster = GatewayCluster(worker_cores=1, worker_memory=8)
    # cluster.scale(400)
    #        with cluster.get_client() as client:
    with Client() as client:
        print(client.dashboard_link)
        processor.process_by_scene()


if __name__ == "__main__":
    STORAGE_AOI_PREFIX = Path(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/"
    )
    aoi_by_pathrow_file = STORAGE_AOI_PREFIX / "aoi_split_by_landsat_pathrow.gpkg"

    run_processor(
        year=2015,
        scene_processor=coastlines_by_year,
        dataset_id="coastlines",
    )
