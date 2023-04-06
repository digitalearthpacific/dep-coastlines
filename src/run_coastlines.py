from typing import Callable, Union

from dask.distributed import Client
from dask_gateway import GatewayCluster
import geopandas as gpd
from pandas import DatetimeIndex, Timestamp
import rioxarray
from xarray import DataArray, Dataset

from dep_tools.Processor import Processor


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
    da = rioxarray.open_rasterio(
        f"https://deppcpublicstorage.blob.core.windows.net/output/{dataset_id}/{dataset_id}_{path}_{row}.tif",
        chunks=True,
    )
    return da.assign_coords(
        # this is the original data type produced by pixel_tides
        band=[Timestamp(t) for t in da.attrs["long_name"]]
    ).rename(band="time")


def tide_cutoffs_dask(
    ds: Dataset, tides_lowres: DataArray, tide_centre=0.0, resampling="linear"
) -> tuple[DataArray, DataArray]:
    """A replacement for coastlines.tide_cutoffs that is dask enabled"""
    # Calculate min and max tides
    tide_min = tides_lowres.min(dim="time")
    tide_max = tides_lowres.max(dim="time")

    # Identify cutoffs
    tide_cutoff_buffer = (tide_max - tide_min) * 0.25
    tide_cutoff_min = tide_centre - tide_cutoff_buffer
    tide_cutoff_max = tide_centre + tide_cutoff_buffer

    # Reproject into original geobox
    tide_cutoff_min = tide_cutoff_min.interp(
        x=ds.coords["x"].values, y=ds.coords["y"].values, method=resampling
    )

    tide_cutoff_max = tide_cutoff_max.interp(
        x=ds.coords["x"].values, y=ds.coords["y"].values, method=resampling
    )

    return tide_cutoff_min, tide_cutoff_max


def coastlines_by_year(xr: DataArray, area) -> Dataset:
    working_ds = mndwi(xr).to_dataset()
    working_ds["ndwi"] = ndwi(xr)

    tides_lowres = load_tides(area["PATH"].values[0], area["ROW"].values[0])
    # The deafrica-coastlines code uses rio.reproject_match, but it is not dask
    # enabled. However, we shouldn't need the reprojection, so we can use
    # (the dask enabled) DataArray.interp instead. Note that the default resampler
    # ("linear") is equivalent to bilinear.
    working_ds["tide_m"] = tides_lowres.interp(
        dict(x=working_ds.coords["x"].values, y=working_ds.coords["y"].values)
    )

    tide_cutoff_min, tide_cutoff_max = tide_cutoffs_dask(
        working_ds, tides_lowres, tide_centre=0.0
    )

    working_ds = filter_by_cutoffs(working_ds, tide_cutoff_min, tide_cutoff_max).drop(
        "tide_m"
    )

    # This taken from tidal_composites (I would use it directly but it
    # sets different nodata values which our writer can't handle,
    # and adds a year dimension (likewise)
    #    median_ds = working_ds.median(dim="time", keep_attrs=True)
    #    median_ds["count"] = working_ds.mndwi.count(dim="time", keep_attrs=True).astype(
    #        "int16"
    #    )
    #    median_ds["stdev"] = working_ds.mndwi.std(dim="time", keep_attrs=True)
    median_ds = working_ds.resample(time="1Y").median(keep_attrs=True)
    median_ds["count"] = (
        working_ds.mndwi.resample(time="1Y").count(keep_attrs=True).astype("int16")
    )
    median_ds["stdev"] = working_ds.mndwi.resample(time="1Y").std(keep_attrs=True)
    median_ds = median_ds.assign_coords(
        time=[f"y{t.year}" for t in DatetimeIndex(median_ds.time)]
    )

    # return median_ds.ndwi.to_dataset("time")
    return median_ds


def run_processor(
    scene_processor: Callable,
    dataset_id: str,
    **kwargs,
) -> None:
    processor = Processor(scene_processor, dataset_id, **kwargs)
    cluster = GatewayCluster(worker_cores=1, worker_memory=8)
    cluster.scale(50)
    with cluster.get_client() as client:
        # with Client() as client:
        print(client.dashboard_link)
        processor.process_by_scene()


if __name__ == "__main__":
    aoi_by_tile = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
        # "/tmp/src/data/coastline_split_by_pathrow.gpkg"
        #        "data/coastline_split_by_pathrow.gpkg"
    ).set_index(["PATH", "ROW"], drop=False)

    # year = "2016"
    run_processor(
        scene_processor=coastlines_by_year,
        dataset_id="coastlines",
        # year=year,
        aoi_by_tile=aoi_by_tile,
        convert_output_to_int16=False,
        send_area_to_scene_processor=True,
    )
