from typing import Callable, Dict, Union

from dask.distributed import Client, Lock
from dea_tools.spatial import subpixel_contours
import geopandas as gpd
import numpy as np
import rasterio
import rioxarray
from xarray import DataArray, Dataset

# local submodules
from coastlines.raster import tidal_composite, pixel_tides, tide_cutoffs, export_annual_gapfill
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


def coastlines_by_year(xr: DataArray, pixel_tides_kwargs : Dict = dict()) -> Dataset:
    working_ds = mndwi(xr).to_dataset()
    working_ds["ndwi"] = ndwi(xr)

    tides_lowres = pixel_tides(working_ds, resample=False, **pixel_tides_kwargs).transpose("time", "y", "x")
    working_ds["tide_m"] = tides_lowres.rio.reproject_match(
        working_ds, rasterio.enums.Resampling.bilinear
    )

    tide_cutoff_min, tide_cutoff_max = tide_cutoffs(
        working_ds, tides_lowres, tide_centre=0.0
    )

    working_ds = filter_by_cutoffs(working_ds, tide_cutoff_min, tide_cutoff_max).drop("tide_m")

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
    year: int,
    scene_processor: Callable,
    **kwargs,
) -> None:
    processor = Processor(year, scene_processor, **kwargs)
    # cluster = GatewayCluster(worker_cores=1, worker_memory=8)
    # cluster.scale(400)
    #        with cluster.get_client() as client:
    with Client() as client:
        print(client.dashboard_link)
        processor.process_by_scene()


if __name__ == "__main__":
    pixel_tides_kwargs = dict(
            model="TPXO9-atlas-v5", 
#            directory="../coastlines-local/tidal-models/"
            )
    aoi_by_tile = gpd.read_file("data/coastline_split_by_pathrow.gpkg").set_index(["PATH", "ROW"])

    year = 2015
    run_processor(
        year=year,
        scene_processor=coastlines_by_year,
        scene_processor_kwargs=dict(pixel_tides_kwargs=pixel_tides_kwargs),
        dataset_id="coastlines",
        aoi_by_tile=aoi_by_tile,
        convert_output_to_int16=False
    )
