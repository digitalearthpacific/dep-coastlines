import os
from typing import Callable, List, Union

from azure.storage.blob import ContainerClient
from dask.distributed import Client, Lock
from dask_gateway import GatewayCluster
from dea_tools.spatial import subpixel_contours
import geopandas as gpd
from pandas import Timestamp
import rasterio
import rioxarray
from xarray import DataArray, Dataset

# local submodules
from coastlines.raster import (
    tidal_composite,
    pixel_tides,
    tide_cutoffs,
    export_annual_gapfill,
)
from coastlines.vector import contours_preprocess, coastal_masking
from dep_tools.Processor import Processor
from dep_tools.utils import make_geocube_dask


def mndwi(xr: DataArray) -> DataArray:
    # modified normalized differential water index is just a normalized index
    # like NDVI, with different bands
    green = xr.sel(band="green")
    swir = xr.sel(band="swir16")
    #    return xrspatial.multispectral.ndvi(green, swir).rename("mndwi")
    mndwi = (green - swir) / (green + swir)
    return mndwi.rename("mndwi")


def ndwi(xr: DataArray) -> DataArray:
    green = xr.sel(band="green")
    nir = xr.sel(band="nir08")
    ndwi = (green - nir) / (green + nir)
    return ndwi.rename("ndwi")


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


def load_tides(
    path,
    row,
    dataset_id: str = "tpxo_lowres",
    container_name: str = "output",
) -> DataArray:
    print(os.environ["AZURE_STORAGE_ACCOUNT"])
    print(os.environ["AZURE_STORAGE_SAS_TOKEN"])

    da = rioxarray.open_rasterio(
        f"https://deppcpublicstorage.blob.core.windows.net/output/{dataset_id}/{dataset_id}_{path}_{row}.tif",
        #        f"/vsiaz/{container_name}/{dataset_id}/{dataset_id}_{path}_{row}.tif",
    )
    return da.assign_coords(
        # this is the original data type produced by pixel_tides
        band=[Timestamp(t) for t in da.attrs["long_name"]]
    ).rename(band="time")


def coastlines_by_year(xr: DataArray, area) -> Dataset:
    working_ds = mndwi(xr).to_dataset()
    working_ds["ndwi"] = ndwi(xr)

    tides_lowres = load_tides(area["PATH"].values[0], area["ROW"].values[0])
    breakpoint()
    working_ds["tide_m"] = tides_lowres.rio.reproject_match(
        working_ds, rasterio.enums.Resampling.bilinear
    )

    tide_cutoff_min, tide_cutoff_max = tide_cutoffs(
        working_ds, tides_lowres, tide_centre=0.0
    )

    working_ds = filter_by_cutoffs(working_ds, tide_cutoff_min, tide_cutoff_max).drop(
        "tide_m"
    )

    # This taken from tidal_composites (I would use it directly but it
    # sets different nodata values which our writer can't handle,
    # and adds a year dimension (likewise)
    median_ds = working_ds.median(dim="time", keep_attrs=True)
    median_ds["count"] = working_ds.mndwi.count(dim="time", keep_attrs=True).astype(
        "int16"
    )
    median_ds["stdev"] = working_ds.mndwi.std(dim="time", keep_attrs=True)

    return median_ds


def run_processor(
    scene_processor: Callable,
    dataset_id: str,
    year: str,
    **kwargs,
) -> None:
    processor = Processor(scene_processor, dataset_id, year, **kwargs)
    cluster = GatewayCluster(worker_cores=1, worker_memory=8)
    cluster.scale(400)
    with cluster.get_client() as client:
        # with Client() as client:
        print(client.dashboard_link)
        processor.process_by_scene()


if __name__ == "__main__":
    aoi_by_tile = gpd.read_file(
        "/tmp/src/data/coastline_split_by_pathrow.gpkg"
        #        "data/coastline_split_by_pathrow.gpkg"
    ).set_index(["PATH", "ROW"], drop=False)

    year = "2016"
    run_processor(
        year=year,
        scene_processor=coastlines_by_year,
        dataset_id="coastlines",
        aoi_by_tile=aoi_by_tile,
        convert_output_to_int16=False,
        send_area_to_scene_processor=True,
    )
