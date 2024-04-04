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

import planetary_computer
import pystac_client
from xarray import Dataset

from dea_tools.coastal import pixel_tides

from azure_logger import CsvLogger
from dep_tools.loaders import SearchLoader
from dep_tools.searchers import LandsatPystacSearcher
from dep_tools.namers import DepItemPath
from dep_tools.processors import Processor
from dep_tools.task import ErrorCategoryAreaTask, MultiAreaTask
from dep_tools.utils import get_container_client

from dep_coastlines.ProjOdcLoader import ProjOdcLoader
from grid import grid
from task_utils import get_ids
from writer import DaWriter


class TideProcessor(Processor):
    def process(self, xr: Dataset, area) -> Dataset:
        working_ds = xr.rio.clip(
            area.to_crs(xr.red.rio.crs).geometry, all_touched=True, from_disk=True
        ).red.drop_duplicates(...)

        tides_lowres = pixel_tides(
            working_ds,
            resample=False,
            model="TPXO9-atlas-v5",
            directory="../coastlines-local/tidal-models/",
            resolution=4980,
            parallel=False,
        ).transpose("time", "y", "x")

        tides_lowres.coords["time"] = tides_lowres.coords["time"].astype("str")
        return tides_lowres.to_dataset("time")


def run(task_id: str | list[str], datetime: str, version: str, dataset_id: str) -> None:
    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    searcher = LandsatPystacSearcher(
        search_intersecting_pathrows=True, client=client, datetime=datetime
    )
    stacloader = ProjOdcLoader(
        fail_on_error=False,
        chunks=dict(band=1, time=1, x=1024, y=1024),
        bands=["red"],
    )
    loader = SearchLoader(searcher, stacloader)

    processor = TideProcessor(send_area_to_processor=True)
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
            task_id,
            grid,
            ErrorCategoryAreaTask,
            loader,
            processor,
            writer,
            logger,
        ).run()
    else:
        ErrorCategoryAreaTask(
            task_id, grid.loc[[task_id]], loader, processor, writer, logger
        ).run()


if __name__ == "__main__":
    datetime = "1984/2023"
    version = "0.7.0"
    dataset_id = "coastlines/tpxo9"
    task_ids = get_ids(datetime, version, dataset_id)
    run(task_ids, datetime, version, dataset_id)
