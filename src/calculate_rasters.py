import os
from statistics import median
from typing import List, Literal

from azure.storage.blob import ContainerClient
import geopandas as gpd
from numpy import unique
from numpy.lib.stride_tricks import sliding_window_view
from pandas import DatetimeIndex
import rioxarray
from xarray import concat, DataArray, Dataset

from dep_tools.Processor import run_processor

from tide_utils import filter_by_tides
from water_indices import mndwi, ndwi, awei, wofs


def coastlines_by_year(
    xr: DataArray, area, composite_type: str = Literal["annual", "three_year"]
) -> Dataset:
    # Possible we should do this in Processor.py, need to think through
    # whether there is a case where we _would_ want duplicates
    xr = xr.drop_duplicates(...)
    xr = filter_by_tides(xr, area["PATH"].values[0], area["ROW"].values[0])

    working_ds = mndwi(xr).to_dataset()
    working_ds["ndwi"] = ndwi(xr)
    working_ds["awei"] = awei(xr)

    working_ds["wofs"] = wofs(xr)
    working_ds["nir08"] = xr.sel(band="nir08")

    # In case we filtered out all the data
    if not "time" in working_ds or len(working_ds.time) == 0:
        return None

    return create_composite(working_ds, composite_type)


def create_composite(
    working_ds: Dataset, composite_type: str = Literal["all", "annual", "three_year"]
) -> Dataset:
    # working_ds has readings for all times.
    # returns a dataset resampled to the specified cadence. Note that
    # three year composites are rolling.

    #    if composite_type == "annual":
    #        median_ds = working_ds.resample(time="1Y").median(keep_attrs=True)
    #        breakpoint()
    #        median_ds["wofs"] = working_ds.wofs.resample(time="1Y").mean(keep_attrs=True)
    #
    #        median_ds["count"] = (
    #            working_ds.mndwi.resample(time="1Y").count(keep_attrs=True).astype("int16")
    #        )
    #        median_ds["stdev"] = working_ds.nir08.resample(time="1Y").std(keep_attrs=True)
    #        median_ds = median_ds.assign_coords(
    #            time=[f"{t.year}" for t in DatetimeIndex(median_ds.time)]
    #        )
    #        return median_ds.squeeze()
    #    elif composite_type == "three_year":
    # using the same code as for the single year composite doesn't work,
    # since using e.g. .resample(time="3Y") creates non-overlapping
    # 3 year composites.

    # First create a list of 3 year time periods like
    # [[2000, 2001, 2002], [2001, 2002, 2003], [2002, 2003, 2004]]
    if composite_type == "all":
        output = working_ds.median("time", keep_attrs=True)
        output["wofs"] = working_ds.wofs.mean("time", keep_attrs=True)
        output["count"] = working_ds.mndwi.count("time", keep_attrs=True).astype(
            "int16"
        )
        output["stdev"] = working_ds.mndwi.std("time", keep_attrs=True)
        return output
    else:
        years = unique(DatetimeIndex(working_ds.time).year)
        sets = (
            sliding_window_view(years, 3)
            if composite_type == "three_year"
            else [[y] for y in years]
        )
        # three_year_sets = sliding_window_view(years, 3)
        # three_year_medians = list()
        medians = list()

        # Then select the values for the years in each composite and summarise
        #    for three_year_set in three_year_sets:
        for time_set in sets:
            this_working_ds = working_ds.sel(
                time=slice(str(min(time_set)), str(max(time_set)))
            )

            # Or another value; there is some more filtering in coastlines.vector
            # Here it is mostly to prevent errors when trying to write an
            # empty array
            if len(this_working_ds.time) < 1:
                continue
            this_median = this_working_ds.median("time", keep_attrs=True)
            this_median["wofs"] = this_working_ds.wofs.mean("time", keep_attrs=True)
            this_median["count"] = this_working_ds.mndwi.count(
                "time", keep_attrs=True
            ).astype("int16")
            this_median["stdev"] = this_working_ds.mndwi.std("time", keep_attrs=True)
            this_median = this_median.assign_coords(time=str(median(time_set)))
            medians.append(this_median)
        return concat(medians, "time").squeeze().unify_chunks()


#    else:
#        raise ValueError(f"{composite_type} is not a valid value for `composite_type`")


if __name__ == "__main__":
    aoi_by_tile = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
    ).set_index(["PATH", "ROW"], drop=False)
    aoi_by_tile = aoi_by_tile.loc[aoi_by_tile.index == (84, 55),]

    storage_account = os.environ["AZURE_STORAGE_ACCOUNT"]
    container_name = "output"
    credential = os.environ["AZURE_STORAGE_SAS_TOKEN"]
    container_client = ContainerClient(
        f"https://{storage_account}.blob.core.windows.net",
        container_name=container_name,
        credential=credential,
    )

    run_processor(
        scene_processor=coastlines_by_year,
        dataset_id="coastlines",
        n_workers=80,
        year="2014",
        container_client=container_client,
        aoi_by_tile=aoi_by_tile,
        convert_output_to_int16=True,
        send_area_to_scene_processor=True,
        scene_processor_kwargs=dict(composite_type="all"),
        #        split_output_by_year=True,
        split_output_by_variable=False,
        overwrite=True,
        output_value_multiplier=1000,
    )
