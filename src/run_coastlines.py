from typing import Callable, Tuple, Union

from dask.distributed import Client
from dask_gateway import GatewayCluster
import geopandas as gpd
from pandas import DatetimeIndex, Timestamp
import rioxarray
from retry import retry
from xarray import DataArray, Dataset

from dep_tools.Processor import Processor
from dep_tools.utils import scale_and_offset


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


def awei(xr: DataArray) -> DataArray:
    green = xr.sel(band="green")
    swir1 = xr.sel(band="swir16")
    swir2 = xr.sel(band="swir22")
    nir = xr.sel(band="nir08")

    awei = 4 * (green - swir2) - (0.25 * nir + 2.75 * swir1)
    return awei.rename("awei")


def normalized_ratio(band1: DataArray, band2: DataArray) -> DataArray:
    return (band1 - band2) / (band1 + band2)


def wofs(tm_da: DataArray) -> DataArray:
    # lX indicates a left path from node X
    # rX indicates a right
    # dX is just the logic for _that_ node
    tm = tm_da.to_dataset("band")
    tm["ndi52"] = normalized_ratio(tm.swir16, tm.green)
    tm["ndi43"] = normalized_ratio(tm.nir08, tm.red)
    tm["ndi72"] = normalized_ratio(tm.swir22, tm.green)

    d1 = tm.ndi52 <= -0.01
    l2 = d1 & (tm.blue <= 2083.5)
    d3 = tm.swir22 <= 323.5

    l3 = l2 & d3
    w1 = l3 & (tm.ndi43 <= 0.61)

    r3 = l2 & ~d3
    d5 = tm.blue <= 1400.5
    d6 = tm.ndi72 <= -0.23
    d7 = tm.ndi43 <= 0.22
    w2 = r3 & d5 & d6 & d7

    w3 = r3 & d5 & d6 & ~d7 & (tm.blue <= 473.0)

    w4 = r3 & d5 & ~d6 & (tm.blue <= 379.0)
    w7 = r3 & ~d5 & (tm.ndi43 <= -0.01)

    d11 = tm.ndi52 <= 0.23
    l13 = ~d1 & d11 & (tm.blue <= 334.5) & (tm.ndi43 <= 0.54)
    d14 = tm.ndi52 <= -0.12

    w5 = l13 & d14
    r14 = l13 & ~d14
    d15 = tm.red <= 364.5

    w6 = r14 & d15 & (tm.blue <= 129.5)
    w8 = r14 & ~d15 & (tm.blue <= 300.5)

    w10 = (
        ~d1
        & ~d11
        & (tm.ndi52 <= 0.32)
        & (tm.blue <= 249.5)
        & (tm.ndi43 <= 0.45)
        & (tm.red <= 364.5)
        & (tm.blue <= 129.5)
    )

    water = w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 | w10
    return water.where(tm.red.notnull(), float("nan"))


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
    tide_bool = (tides_lowres >= tide_cutoff_min) & (tides_lowres <= tide_cutoff_max)

    # This step loads tide_bool in memory so if you are getting memory spikes,
    # or if you have overwrite=False and you're trying to fill in some missing
    # outputs and it's taking a while, this is probably the reason.
    ds = ds.sel(time=tide_bool.sum(dim=["x", "y"]) > 0)

    # Apply mask to high res data
    tide_bool_highres = (ds.tide_m >= tide_cutoff_min) & (ds.tide_m <= tide_cutoff_max)
    return ds.where(tide_bool_highres)


@retry(tries=20, delay=10)
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

    chunks = dict(x=ds.chunks["x"], y=ds.chunks["y"])

    # Reproject into original geobox
    tide_cutoff_min = tide_cutoff_min.interp(
        x=ds.coords["x"].values, y=ds.coords["y"].values, method=resampling
    ).chunk(chunks)

    tide_cutoff_max = tide_cutoff_max.interp(
        x=ds.coords["x"].values, y=ds.coords["y"].values, method=resampling
    ).chunk(chunks)

    return tide_cutoff_min, tide_cutoff_max


def coastlines_by_year(xr: DataArray, area) -> Dataset:
    # Possible we should do this in Processor.py, need to think through
    # whether there is a case where we _would_ want duplicates
    xr = xr.drop_duplicates(...)
    working_ds = mndwi(xr).to_dataset()
    working_ds["ndwi"] = ndwi(xr)
    working_ds["awei"] = awei(xr)

    # recaling for the wofs algorithm
    l1_scale = 0.0001
    l1_rescale = 1.0 / l1_scale
    xr = scale_and_offset(xr, scale=[l1_rescale])
    working_ds["wofs"] = wofs(xr)

    tides_lowres = load_tides(area["PATH"].values[0], area["ROW"].values[0])
    # Filter out times that are not in the tidal data. Basically because I have
    # been caching the times, we may miss very recent readings (like here it is
    # April 10 and I don't have tides for March 30 or April 7 Landsat data.
    working_ds = working_ds.sel(
        time=working_ds.time[working_ds.time.isin(tides_lowres.time)]
    )

    # Now filter out tide times that are not in the working_ds
    tides_lowres = tides_lowres.sel(
        time=tides_lowres.time[tides_lowres.time.isin(working_ds.time)]
    )

    # The deafrica-coastlines code uses rio.reproject_match, but it is not dask
    # enabled. However, we shouldn't need the reprojection, so we can use
    # (the dask enabled) DataArray.interp instead. Note that the default resampler
    # ("linear") is equivalent to bilinear.
    working_ds["tide_m"] = tides_lowres.interp(
        dict(x=working_ds.coords["x"].values, y=working_ds.coords["y"].values)
    )
    working_ds = working_ds.unify_chunks()

    tide_cutoff_min, tide_cutoff_max = tide_cutoffs_dask(
        working_ds, tides_lowres, tide_centre=0.0
    )

    working_ds = filter_by_cutoffs(
        working_ds, tides_lowres, tide_cutoff_min, tide_cutoff_max
    ).drop("tide_m")

    # In case we filtered out all the data
    if len(working_ds.time) == 0:
        return None

    median_ds = working_ds.resample(time="1Y").median(keep_attrs=True)
    median_ds["wofs"] = working_ds.wofs.resample(time="1Y").mean(keep_attrs=True)
    median_ds["count"] = (
        working_ds.mndwi.resample(time="1Y").count(keep_attrs=True).astype("int16")
    )
    median_ds["stdev"] = working_ds.mndwi.resample(time="1Y").std(keep_attrs=True)
    median_ds = median_ds.assign_coords(
        time=[f"{t.year}" for t in DatetimeIndex(median_ds.time)]
    )

    return median_ds.squeeze()


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

    run_processor(
        scene_processor=coastlines_by_year,
        dataset_id="coastlines",
        aoi_by_tile=aoi_by_tile,
        convert_output_to_int16=True,
        send_area_to_scene_processor=True,
        # year="2015",
        split_output_by_year=True,
        split_output_by_variable=False,
        overwrite=False,
        output_value_multiplier=1000,
    )
