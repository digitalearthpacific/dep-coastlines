from typing import Callable, Literal

from dask.distributed import Client
from dask_gateway import GatewayCluster
import geopandas as gpd
from numpy import unique
from numpy.lib.stride_tricks import sliding_window_view
from pandas import DatetimeIndex
import rioxarray
from retry import retry
from xarray import concat, DataArray, Dataset

from dep_tools.Processor import run_processor
from dep_tools.utils import scale_and_offset

from .tide_utils filter_tides


def mndwi(xr: DataArray) -> DataArray:
    # modified normalized differential water index is just a normalized index
    # like NDVI, with different bands
    mndwi = normalized_ratio(xr.sel(band="green"), xr.sel(band="swir16"))
    return mndwi.rename("mndwi")


def ndwi(xr: DataArray) -> DataArray:
    ndwi = normalized_ratio(xr.sel(band="green"), xr.sel(band="nir08"))
    return ndwi.rename("ndwi")


def awei(xr: DataArray) -> DataArray:
    green = xr.sel(band="green")
    swir1 = xr.sel(band="swir16")
    swir2 = xr.sel(band="swir22")
    nir = xr.sel(band="nir08")

    awei = 4 * (green - swir2) - (0.25 * nir + 2.75 * swir1)
    return awei.rename("awei")


def wofs(tm_da: DataArray) -> DataArray:
    # First, rescale to what the wofs model expects
    l1_scale = 0.0001
    l1_rescale = 1.0 / l1_scale
    tm_da = scale_and_offset(tm_da, scale=[l1_rescale])
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


def normalized_ratio(band1: DataArray, band2: DataArray) -> DataArray:
    return (band1 - band2) / (band1 + band2)



def coastlines_by_year(
    xr: Dataset, area, composite_type: str = Literal["annual", "three_year"]
) -> Dataset:
    # Possible we should do this in Processor.py, need to think through
    # whether there is a case where we _would_ want duplicates
    xr = xr.drop_duplicates(...)
    working_ds = mndwi(xr).to_dataset()
    working_ds["ndwi"] = ndwi(xr)
    working_ds["awei"] = awei(xr)

    working_ds["wofs"] = wofs(xr)

    working_ds = filter_by_tides(working_ds, area["PATH"].values[0], area["ROW"].values[0])

    # In case we filtered out all the data
    if len(working_ds.time) == 0:
        return None

    if composite_type == "annual":
        median_ds = working_ds.resample(time="1Y").median(keep_attrs=True)
        median_ds["count"] = (
            working_ds.mndwi.resample(time="1Y").count(keep_attrs=True).astype("int16")
        )
        median_ds["stdev"] = working_ds.mndwi.resample(time="1Y").std(keep_attrs=True)
        median_ds = median_ds.assign_coords(
            time=[f"{t.year}" for t in DatetimeIndex(median_ds.time)]
        )
        return median_ds.squeeze()
    elif composite_type == "three_year":
        # three year composites
        three_year_sets = sliding_window_view(
            unique(xr.time.astype("datetime64[Y]").values), 3
        )
        three_year_medians = list()
        for three_year_set in three_year_sets:
            this_working_ds = working_ds.sel(
                time=slice(min(three_year_set), max(three_year_set))
            )

            # Or another value; there is some more filtering in coastlines.vector
            if len(this_working_ds.time) < 1:
                continue
            this_median = this_working_ds.median("time", keep_attrs=True)
            this_median["count"] = this_working_ds.mndwi.count(
                "time", keep_attrs=True
            ).astype("int16")
            this_median["stdev"] = this_working_ds.mndwi.std("time", keep_attrs=True)
            this_median = this_median.assign_coords(
                time=three_year_set[1].astype("datetime64[Y]").astype("str")
            )
            three_year_medians.append(this_median)
        return concat(three_year_medians, "time")
    else:
        raise ValueError(f"{composite_type} is not a valid value for `composite_type`")


if __name__ == "__main__":
    aoi_by_tile = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
    ).set_index(["PATH", "ROW"], drop=False)[85:170]

    run_processor(
        scene_processor=coastlines_by_year,
        dataset_id="coastlines-composite",
        aoi_by_tile=aoi_by_tile,
        convert_output_to_int16=True,
        send_area_to_scene_processor=True,
        scene_processor_kwargs=dict(composite_type="three_year"),
        split_output_by_year=True,
        split_output_by_variable=False,
        overwrite=False,
        output_value_multiplier=1000,
    )
