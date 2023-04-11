from typing import Callable, Tuple, Union

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
    tides_lowres,
    tide_cutoff_min: Union[int, float, DataArray],
    tide_cutoff_max: Union[int, float, DataArray],
) -> Dataset:
    """
    coastline.raster.load_tidal_subset that doesn't load
    """
    # Determine what pixels were acquired in selected tide range, and
    # drop time-steps without any relevant pixels to reduce data to load
    # tide_bool = (ds.tide_m >= tide_cutoff_min) & (ds.tide_m <= tide_cutoff_max)
    # Changing this to use the lowres tides, since it's causing some memory spikes
    breakpoint()
    tide_bool = (tides_lowres >= tide_cutoff_min) & (tides_lowres <= tide_cutoff_max)

    # This step loads tide_bool in memory so if you are getting memory spikes,
    # or if you have overwrite=False and you're trying to fill in some missing
    # outputs and it's taking a while, this is probably the reason.
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
        f"https://deppcpublicstorage.blob.core.windows.net/{container_name}/{dataset_id}/{dataset_id}_{path}_{row}.tif",
        chunks=True,
    )

    time_strings = da.attrs["long_name"]
    band_names = (
        # this is the original data type produced by pixel_tides
        [Timestamp(t) for t in time_strings]
        # In case there's only one
        if isinstance(time_strings, Tuple)
        else [Timestamp(time_strings)]
    )

    return da.assign_coords(band=band_names).rename(band="time").drop_duplicates(...)


def tide_cutoffs_dask(
    ds: Dataset, tides_lowres: DataArray, tide_centre=0.0, resampling="linear"
) -> Tuple[DataArray, DataArray]:
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
    # Possible we should do this in Processor.py, need to think through
    # whether there is a case where we _would_ want duplicates
    xr = xr.drop_duplicates(...)
    working_ds = mndwi(xr).to_dataset()
    working_ds["ndwi"] = ndwi(xr)

    tides_lowres = load_tides(area["PATH"].values[0], area["ROW"].values[0])
    # Filter out times that are not in the tidal data. Basically because I have
    # been caching the times, we may miss very recent readings (like here it is
    # April 10 and I don't have tides for March 30 or April 7 Landsat data.
    working_ds = working_ds.sel(
        time=working_ds.time[working_ds.time.isin(tides_lowres.time)]
    )
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

    working_ds = filter_by_cutoffs(
        working_ds, tides_lowres, tide_cutoff_min, tide_cutoff_max
    ).drop("tide_m")

    # We filtered out all the data
    if len(working_ds.time) == 0:
        return None

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
        time=[f"{t.year}" for t in DatetimeIndex(median_ds.time)]
    )

    # return median_ds.ndwi.to_dataset("time")
    return median_ds


def run_processor(
    scene_processor: Callable,
    dataset_id: str,
    **kwargs,
) -> None:
    processor = Processor(scene_processor, dataset_id, **kwargs)
    try:
        cluster = GatewayCluster(worker_cores=1, worker_memory=8)
        cluster.scale(400)
        with cluster.get_client() as client:
            print(client.dashboard_link)
            processor.process_by_scene()
    except ValueError:
        with Client() as client:
            print(client.dashboard_link)
            processor.process_by_scene()


if __name__ == "__main__":
    aoi_by_tile = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
    ).set_index(["PATH", "ROW"], drop=False)
    aoi_by_tile = aoi_by_tile[aoi_by_tile["PR"] == "076064"]

    run_processor(
        scene_processor=coastlines_by_year,
        dataset_id="coastlines",
        aoi_by_tile=aoi_by_tile,
        convert_output_to_int16=True,
        send_area_to_scene_processor=True,
        split_output_by_year=True,
        split_output_by_variable=False,
        overwrite=False,
    )
