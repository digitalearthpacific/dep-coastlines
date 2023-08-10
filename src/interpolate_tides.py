"""
Calculates low-resolution tide rasters for all areas and times using 
functionality from the Digital Earth Austalia project. In the DEA and DEAfrica
coastline processing this is included with other raster processing but 
1) For this project all the water index, etc. calculations are dask-enabled,
   while this is not.
2) It has separate prerequisites that other pieces due to the tidal calculation
   packages (see below).
3) This is the most static part of the workflow in that once the areas are
   set, the tidal calculations can be "one and done" for the most part (unless
   you wish to revisit for more recent data at a later time). No reason to redo
   this every time if testing out new water indices / cloud masking etc.

Tidal processing should be done before anything else, as results are needed for
filtering input landsat data before water index calculation.

This can be run locally with tidal model data in ../coastlines-local (see below) as
it does not actually pull any planetary computer data and only writes
small-ish (5km resolution) data to blob storage. It has low memory requirements
and takes just a few hours to run for the full area and all times. It could be
modified to run in kbatch but requires a docker image with the large tidal models
embedded.

TODO: If revisiting this file, consider abstracting some of the constant values
set in the main script body and using typer.
"""

from ast import literal_eval
from typing import Union

from dask.distributed import Client, Lock
import geopandas as gpd
from pandas import DataFrame
from xarray import DataArray, Dataset


from azure_logger import CsvLogger
from dep_tools.runner import run_by_area
from dep_tools.processors import Processor
from dep_tools.loaders import LandsatOdcLoader
from dep_tools.utils import get_container_client

from dep_tools.writers import AzureXrWriter

# from dep_tools.writers2 import LocalXrWriter
from tide_utils import fill_and_interp, tide_cutoffs_dask, load_tides


class TideProcessor(Processor):
    def process(self, xr: DataArray, area) -> Dataset:
        tides_lowres = load_tides(area.index[0])
        working_ds = (
            xr.isel(band=0)
            .drop_duplicates(...)
            .sel(time=xr.time[xr.time.isin(tides_lowres.time)])
            .to_dataset()
        )

        # Now filter out tide times that are not in the ds
        tides_lowres = tides_lowres.sel(
            time=tides_lowres.time[tides_lowres.time.isin(working_ds.time)]
        )

        tides_lowres.coords["time"] = tides_lowres.coords["time"].astype("str")

        tide_cutoff_min, tide_cutoff_max = tide_cutoffs_dask(
            tides_lowres, tide_centre=0.0
        )

        tides_highres = fill_and_interp(tides_lowres, working_ds).to_dataset("time")
        tide_cutoff_min = fill_and_interp(tide_cutoff_min, working_ds)
        tide_cutoff_max = fill_and_interp(tide_cutoff_max, working_ds)

        # Do this  _after_ interpolation
        if area is not None:
            tides_highres = tides_highres.rio.clip(
                area.to_crs(tides_highres.rio.crs).geometry,
                all_touched=True,
                from_disk=True,
            )
        return tides_highres


def get_log_path(
    prefix: str, dataset_id: str, version: str, datetime: Union[str, None] = None
) -> str:
    return (
        f"{prefix}/{dataset_id}/logs/{dataset_id}_{version}_{datetime.replace('/', '_')}_log.csv"
        if datetime is not None
        else f"{prefix}/{dataset_id}/logs/{dataset_id}_{version}_log.csv"
    )


def filter_by_log(df: DataFrame, log: DataFrame) -> DataFrame:
    # Need to decide if this is where we do this. I want to keep the logger
    # fairly generic. Suppose we could subclass it.
    log = log.set_index("index")
    log.index = [literal_eval(i) for i in log.index]

    # Need to filter by errors

    return df[~df.index.isin(log.index)]


def main(datetime: str, version: str, client=None) -> None:
    dataset_id = "tpxo_highres"
    prefix = f"coastlines/{version}"

    aoi_by_tile = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
    ).set_index(["PATH", "ROW"], drop=False)

    loader = LandsatOdcLoader(
        datetime=datetime,
        dask_chunksize=dict(band=1, time=1, x=1024, y=1024),
        odc_load_kwargs=dict(fail_on_error=False),
    )

    processor = TideProcessor(send_area_to_processor=True)

    writer = AzureXrWriter(
        dataset_id=dataset_id,
        year=datetime,
        prefix=prefix,
        convert_to_int16=True,
        overwrite=False,
        extra_attrs=dict(dep_version=version),
    )

    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=get_log_path(prefix, dataset_id, version, datetime),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )

    aoi_by_tile = filter_by_log(aoi_by_tile, logger.parse_log())

    run_by_area(
        areas=aoi_by_tile,
        loader=loader,
        processor=processor,
        writer=writer,
        logger=logger,
    )


if __name__ == "__main__":
    for year in range(1984, 2024):
        main(str(year), "9Aug2023")
