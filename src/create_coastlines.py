"""Perform annual coastline and rates-of-change extraction from mosaics."""

from typing import Annotated, Tuple

import boto3
from dask.distributed import Client
from dep_tools.task import ErrorCategoryAreaTask
from odc.stac import configure_s3_access
from typer import Option, run

from dep_coastlines.common import coastlineItemPath, coastlineLogger
from dep_coastlines.config import MOSAIC_VERSION
from dep_coastlines.grid import buffered_grid as GRID
from dep_coastlines.io import CoastlineWriter, MultiyearMosaicLoader
from dep_coastlines.vector import Cleaner

DATASET_ID = "coastlines/interim/coastlines"


def main(
    column: Annotated[str, Option()],
    row: Annotated[str, Option()],
    version: Annotated[str, Option()],
    start_year: Annotated[int, Option()] = 1984,
    end_year: Annotated[int, Option()] = 2024,
    water_index: str = "twndwi",
):
    configure_s3_access(cloud_defaults=True, requester_pays=True)
    boto3.setup_default_session()
    with Client():
        process_id(
            (int(column), int(row)),
            version=version,
            start_year=start_year,
            end_year=end_year,
            water_index=water_index,
        )


def process_id(
    task_id: Tuple | list[Tuple] | None,
    dataset_id=DATASET_ID,
    version: str = "0.8.0",
    start_year: int = 1984,
    end_year: int = 2024,
    water_index="twndwi",
) -> None:
    namer = coastlineItemPath(dataset_id, version, time=f"{start_year}/{end_year}")
    logger = coastlineLogger(namer, dataset_id=dataset_id)

    loader = MultiyearMosaicLoader(
        start_year=start_year,
        end_year=end_year,
        version=MOSAIC_VERSION,
    )
    processor = Cleaner(
        water_index=water_index,
        send_area_to_processor=True,
        initial_year=str(start_year),
        baseline_year=str(end_year),
    )
    writer = CoastlineWriter(
        namer,
        extra_attrs=dict(dep_version=version),
    )

    ErrorCategoryAreaTask(
        task_id, GRID.loc[[task_id]], loader, processor, writer, logger
    ).run()


if __name__ == "__main__":
    run(main)
