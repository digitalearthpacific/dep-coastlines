"""Calculates low-resolution tide rasters for all areas and times using 
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

from itertools import product
import json
import sys

from dask.distributed import Client
import geopandas as gpd
import planetary_computer
import pystac_client
from xarray import DataArray, Dataset

from dea_tools.coastal import pixel_tides

from azure_logger import CsvLogger, filter_by_log
from dep_tools.loaders2 import LandsatPystacSearcher, OdcLoader, SearchLoader
from dep_tools.namers import DepItemPath
from dep_tools.processors import Processor
from dep_tools.runner import run_by_area
from dep_tools.task import ErrorCategoryAreaTask, MultiAreaTask
from dep_tools.utils import get_container_client
from dep_tools.writers import DsWriter

from grid import grid
from writer import DaWriter


class TideProcessor(Processor):
    def process(self, xr: Dataset) -> Dataset:
        working_ds = xr.red.drop_duplicates(...)

        tides_lowres = pixel_tides(
            working_ds,
            resample=False,
            model="TPXO9-atlas-v5",
            directory="../coastlines-local/tidal-models/",
            resolution=4980,
        ).transpose("time", "y", "x")

        tides_lowres.coords["time"] = tides_lowres.coords["time"].astype("str")
        return tides_lowres.to_dataset("time")


def get_ids(datetime, version, dataset_id, retry_errors=True) -> list:
    namer = DepItemPath(
        sensor="ls",
        dataset_id="coastlines/tpx09",
        version=version,
        time=datetime.replace("/", "_"),
    )

    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=namer.log_path(),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )
    return filter_by_log(grid, logger.parse_log(), retry_errors).index.to_list()


def get_years_from_datetime(datetime):
    years = datetime.split("-")
    if len(years) == 2:
        years = range(int(years[0]), int(years[1]) + 1)
    elif len(years) > 2:
        ValueError(f"{datetime} is not a valid value for --datetime")
    return years


def print_tasks(datetime, version, limit, no_retry_errors, dataset_id):
    ids = get_ids(datetime, version, dataset_id, not no_retry_errors)
    params = [
        {
            "region-code": region[0][0],
            "region-index": region[0][1],
            "datetime": region[1],
        }
        for id in ids
    ]

    if limit is not None:
        params = params[0 : int(limit)]

    json.dump(params, sys.stdout)


def run(task_id: str | list[str], datetime: str, version: str, dataset_id: str) -> None:
    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    searcher = LandsatPystacSearcher(client=client, datetime=datetime)
    stacloader = OdcLoader(
        fail_on_error=False,
        chunks=dict(band=1, time=1, x=1024, y=1024),
        groupby="solar_day",
    )
    loader = SearchLoader(searcher, stacloader)

    processor = TideProcessor(send_area_to_processor=False)
    namer = DepItemPath(
        sensor="ls",
        dataset_id=dataset_id,
        version=version,
        time=datetime.replace("/", "_"),
        zero_pad_numbers=True,
    )

    writer = DaWriter(itempath=namer, driver="COG", blocksize=16)

    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=namer.log_path(),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )

    if isinstance(task_id, list):
        MultiAreaTask(
            task_id, grid, ErrorCategoryAreaTask, loader, processor, writer, logger
        ).run()
    else:
        ErrorCategoryAreaTask(
            task_id, grid.loc[[task_id]], loader, processor, writer, logger
        ).run()


if __name__ == "__main__":
    datetime = "1984/2023"
    version = "0.6.0"
    dataset_id = "coastlines/tpx09"
    task_ids = get_ids(datetime, version, dataset_id)
    with Client():
        run(task_ids, datetime, version, dataset_id)
