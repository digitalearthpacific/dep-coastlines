"""Calculates low-resolution tide rasters for all areas and times using 
functionality from the Digital Earth Austalia project. In the DEA and DEAfrica
coastline processing this is included with other raster processing but 
in our work
1) It has separate prerequisites that other pieces due to the tidal calculation
   packages (see below).
2) This is the most static part of the workflow in that once the areas are
   set, the tidal calculations can be "one and done" for the most part (unless
   you wish to revisit for more recent data at a later time). No reason to redo
   this every time if testing out new water indices / cloud masking etc.

Tidal processing should be done before anything else, as results are needed for
filtering input landsat data before water index calculation.

This can be run locally with tidal model data in ../coastlines-local (see below) as
it does not actually pull any remote data and only writes
small-ish (5km resolution) data to cloud storage. It has low memory requirements
and takes a reasonable time to run for the full area and all times. It could be
modified to run in the cloud but would require a docker image with the large tidal models
embedded.
"""

import planetary_computer as pc
from xarray import Dataset

from dea_tools.coastal import pixel_tides

from cloud_logger import CsvLogger, S3Handler
from dep_tools.loaders import SearchLoader
from dep_tools.searchers import LandsatPystacSearcher
from dep_tools.namers import DepItemPath
from dep_tools.processors import Processor
from dep_tools.task import ErrorCategoryAreaTask, MultiAreaTask
from dep_tools.aws import write_to_s3

from dep_coastlines.config import BUCKET
from grid import grid
from task_utils import get_ids
from dep_coastlines.io import PreprocessWriter, ProjOdcLoader, OdcMemoryWriter, S3Writer


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
        return tides_lowres  # .to_dataset("time")


def run(task_id: str | list[str], datetime: str, version: str, dataset_id: str) -> None:
    searcher = LandsatPystacSearcher(
        search_intersecting_pathrows=True,
        catalog="https://planetarycomputer.microsoft.com/api/stac/v1",
        datetime=datetime,
    )
    stacloader = ProjOdcLoader(
        fail_on_error=False,
        chunks=dict(band=1, time=1, x=1024, y=1024),
        bands=["red"],
        url_patch=pc.sign,
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

    memory_writer = OdcMemoryWriter(block_size=16)
    s3_writer = S3Writer(itempath=namer, bucket=BUCKET)
    writer = PreprocessWriter(pre_processor=memory_writer, writer=s3_writer)

    # writer = CompositeWriter(
    #    itempath=namer, driver="COG", blocksize=16, writer=write_to_s3, bucket="dep-cl"
    # )

    logger = CsvLogger(
        name=dataset_id,
        path=f"{BUCKET}/{namer.log_path()}",
        overwrite=False,
        header="time|index|status|paths|comment\n",
        cloud_handler=S3Handler,
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
    version = "0.8.0"
    dataset_id = "coastlines/interim/tpxo9"
    task_ids = get_ids(datetime, version, dataset_id)
    run(task_ids, datetime, version, dataset_id)
