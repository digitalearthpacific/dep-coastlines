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

from pathlib import Path

from boto3 import setup_default_session
from dask.distributed import Client
import typer
from xarray import Dataset

from dea_tools.coastal import pixel_tides

from dep_tools.searchers import LandsatPystacSearcher
from dep_tools.processors import Processor
from dep_tools.task import MultiAreaTask, StacTask

from dep_coastlines.common import coastlineLogger, coastlineItemPath, LOCAL_AWS_PROFILE
from dep_coastlines.io import ProjOdcLoader, CompositeWriter
from dep_coastlines.grid import grid
from dep_coastlines.task_utils import get_ids


class TideProcessor(Processor):
    def __init__(
        self, tide_directory: Path | str = "../coastlines-local/tidal-models", **kwargs
    ):
        super().__init__(**kwargs)
        self._tide_directory = tide_directory

    def process(self, xr: Dataset, area) -> Dataset:
        a_variable = list(xr.keys())[0]
        working_ds = xr.rio.clip(
            area.to_crs(xr[a_variable].rio.crs).geometry,
            all_touched=True,
            from_disk=True,
        )[a_variable].drop_duplicates(...)

        tides_lowres = pixel_tides(
            working_ds,
            resample=False,
            model="TPXO9-atlas-v5",
            directory=self._tide_directory,
            resolution=4980,
            parallel=False,
        ).transpose("time", "y", "x")

        tides_lowres.coords["time"] = tides_lowres.coords["time"].astype("str")
        return tides_lowres  # .to_dataset("time")


def run(
    task_id: str | list[str],
    datetime: str,
    version: str,
    dataset_id: str,
    setup_auth: bool = False,
) -> None:
    if setup_auth:
        setup_default_session(profile_name=LOCAL_AWS_PROFILE)

    searcher = LandsatPystacSearcher(
        search_intersecting_pathrows=True,
        catalog="https://earth-search.aws.element84.com/v1",
        datetime=datetime,
    )
    loader = ProjOdcLoader(
        fail_on_error=False,
        chunks=dict(band=1, time=1, x=1024, y=1024),
        bands=["red"],
    )

    processor = TideProcessor(send_area_to_processor=True)
    namer = coastlineItemPath(dataset_id, version, datetime)

    writer = CompositeWriter(itempath=namer, driver="COG", blocksize=16)

    logger = coastlineLogger(namer, dataset_id=dataset_id, setup_auth=setup_auth)

    if isinstance(task_id, list):
        MultiAreaTask(
            ids=task_id,
            areas=grid,
            task_class=StacTask,
            searcher=searcher,
            loader=loader,
            processor=processor,
            writer=writer,
            logger=logger,
        ).run()
    else:
        StacTask(
            task_id,
            grid.loc[[task_id]],
            searcher=searcher,
            loader=loader,
            processor=processor,
            writer=writer,
            logger=logger,
        ).run()


def main(setup_auth: bool = False):
    datetime = "1984/2023"
    version = "0.8.0"
    dataset_id = "coastlines/interim/tpxo9"
    task_ids = get_ids(datetime, version, dataset_id, setup_auth=setup_auth)
    with Client():
        run(task_ids, datetime, version, dataset_id, setup_auth=setup_auth)


if __name__ == "__main__":
    typer.run(main)
