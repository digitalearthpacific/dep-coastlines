"""Calculates water indices for the given areas and times and saves to s3."""

from datetime import datetime
from typing import Tuple

import boto3
import geopandas as gpd
from dask.distributed import Client
from dep_tools.exceptions import EmptyCollectionError, NoOutputError
from dep_tools.namers import GenericItemPath
from dep_tools.processors import XrPostProcessor
from dep_tools.searchers import LandsatPystacSearcher
from dep_tools.stac_utils import use_alternate_s3_href
from dep_tools.task import AwsStacTask
from dep_tools.writers import AwsDsCogWriter
from odc.geo.geobox import AnchorEnum
from odc.stac import configure_s3_access
from pystac_client import Client as PystacClient
from rasterio.errors import RasterioIOError, WarpOperationError
from retry import retry
from typer import run

from dep_coastlines.common import (
    coastlineItemPath,
    coastlineLogger,
)
from dep_coastlines.config import MOSAIC_DATASET_ID
from dep_coastlines.grid import buffered_grid as grid
from dep_coastlines.io import ProjOdcLoader
from dep_coastlines.raster import MosaicProcessor
from dep_coastlines.tide_utils import tides_for_items
from dep_coastlines.time_utils import (
    composite_from_years,
    parse_datetime,
    years_from_yearstring,
)


def process_id(
    task_id: Tuple | list[Tuple] | None,
    version: str,
    datetime: str = "2020/2024",
    dataset_id: str = MOSAIC_DATASET_ID,
    load_before_write: bool = True,
    fail_on_read_error: bool = True,
) -> None:
    """Calculate annual and three-year tide-corrected mosaics.

    Args:
        task_id: The DE Pacific tile id for the given task.
        version: The version of the output mosaic data.
        datetime: A string of the form "year" or "year1/year2" which indicates
            the year or the start and end year of the desired mosaics.
        dataset_id: The dataset identifier for the output.
        load_before_write: Should the output dataset be loaded into memory
            before writing? Passed to :class:`dep_tools.loaders.OdcLoader`.
        fail_on_read_error: Should the process fail on a read error? Passed
            as the `"fail_on_error"` argument to :func:`odc.stac.load`.
    """
    loader = ProjOdcLoader(
        chunks=dict(band=1, time=1, x=8192, y=8192),
        resampling={"qa_pixel": "nearest", "*": "cubic"},
        fail_on_error=fail_on_read_error,
        bands=["qa_pixel", "nir08", "swir16", "swir22", "red", "blue", "green"],
        clip_to_area=False,
        dtype="float32",
        anchor=AnchorEnum.CENTER,
    )

    area = grid.loc[[task_id]]

    namer = coastlineItemPath(dataset_id, version, datetime)
    logger = coastlineLogger(namer, dataset_id=dataset_id)
    post_processor = XrPostProcessor(
        convert_to_int16=False,
        extra_attrs=dict(dep_version=version),
    )

    writer = AwsDsCogWriter(
        itempath=namer, overwrite=False, load_before_write=load_before_write
    )

    try:
        paths = _process_all_years(
            all_time=datetime,
            area=area,
            itempath=namer,
            post_processor=post_processor,
            id=task_id,
            loader=loader,
            writer=writer,
            logger=logger,
        )
    except Exception as e:
        logger.error([task_id, "error", f'"{e}"'])
        raise e

    logger.info([task_id, "complete", paths])


def _process_all_years(
    area: gpd.GeoDataFrame, all_time: str, itempath: GenericItemPath, **kwargs
):
    client = PystacClient.open(
        "https://landsatlook.usgs.gov/stac-server",
        modifier=use_alternate_s3_href,
    )

    items = LandsatPystacSearcher(
        search_intersecting_pathrows=True,
        datetime=all_time,
        client=client,
        collections=["landsat-c2l2-sr"],
    ).search(area)

    tides = tides_for_items(items, area)
    breakpoint()
    processor = MosaicProcessor(
        tides,
        mask_clouds=True,
        scale_and_offset=True,
        send_area_to_processor=True,
    )

    paths = []
    years = composite_from_years(parse_datetime(all_time.replace("/", "_")), [1, 3])
    for year in years:
        print(year)
        configure_s3_access(cloud_defaults=True, requester_pays=True)
        itempath.time = year.replace("/", "_")
        try:
            paths += _run_single_task(
                AwsStacTask(
                    itempath=itempath,
                    area=area,
                    processor=processor,
                    searcher=ItemFilterer(items, year),
                    **kwargs,
                )
            )
        except (EmptyCollectionError, NoOutputError):
            continue

    return paths


@retry(
    (WarpOperationError, RasterioIOError),
    tries=5,
    delay=60,
    backoff=2,
    max_delay=480,
)
def _run_single_task(task) -> str | list[str]:
    return task.run()


class ItemFilterer:
    """A "searcher" which filters a list of items by the given year."""

    def __init__(self, items, year):
        self._items = []
        for item in items:
            format_string = (
                "%Y-%m-%dT%H:%M:%S.%fZ"
                if "." in item.properties["datetime"]
                else "%Y-%m-%dT%H:%M:%SZ"
            )
            item_year = str(
                datetime.strptime(item.properties["datetime"], format_string).year
            )

            if item_year in years_from_yearstring(year, "/"):
                self._items.append(item)

        if len(self._items) == 0:
            raise EmptyCollectionError

    def search(self, *_):
        return self._items


def main(
    row: str,
    column: str,
    version: str,
    load_before_write: bool = True,
    fail_on_read_error: bool = True,
):
    configure_s3_access(cloud_defaults=True, requester_pays=True)
    boto3.setup_default_session()
    with Client():
        process_id(
            (int(column), int(row)),
            version,
            load_before_write=load_before_write,
            fail_on_read_error=fail_on_read_error,
        )


if __name__ == "__main__":
    run(main)
