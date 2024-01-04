"""This was just for visualization / debugging sake, and outputs
are not used in a technical way for the coastline model. This is
hopefully made obsolete by other annual mosaicking effors (like
geomedian).
"""
from dask.distributed import Client
from dask_gateway import GatewayCluster
import planetary_computer
import pystac_client
from xarray import Dataset

from azure_logger import CsvLogger, filter_by_log
from dep_tools.loaders2 import LandsatPystacSearcher, OdcLoader, SearchLoader
from dep_tools.namers import DepItemPath
from dep_tools.processors import LandsatProcessor
from dep_tools.stac_utils import set_stac_properties
from dep_tools.task import ErrorCategoryAreaTask, MultiAreaTask
from dep_tools.utils import get_container_client
from dep_tools.writers import DsWriter

from grid import grid
from xarray import DataArray, Dataset


class MosaicProcessor(LandsatProcessor):
    def process(self, xr: DataArray, area) -> Dataset | None:
        xr = super().process(xr.rio.clip(area.to_crs(xr.red.rio.crs).geometry))
        working_ds = xr[["red", "blue", "green", "nir08", "swir16", "swir22"]]

        # In case we filtered out all the data
        if not "time" in working_ds or len(working_ds.time) == 0:
            return None

        results = working_ds.median("time", keep_attrs=True).astype("uint16")
        return set_stac_properties(xr, results)


def get_ids(logger, retry_errors=True) -> list:
    return filter_by_log(grid, logger.parse_log(), retry_errors).index.to_list()


def run(task_id: str | list[str], datetime: str, namer, logger) -> None:
    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    searcher = LandsatPystacSearcher(client=client, datetime=datetime)
    stacloader = OdcLoader(
        fail_on_error=False,
        chunks=dict(band=1, time=1, x=4096, y=4096),
        groupby="solar_day",
        resampling={"qa_pixel": "nearest", "*": "cubic"},
    )
    loader = SearchLoader(searcher, stacloader)

    processor = MosaicProcessor(
        scale_and_offset=False,
        send_area_to_processor=True,
    )

    writer = DsWriter(itempath=namer, output_nodata=0)

    if isinstance(task_id, list):
        MultiAreaTask(
            task_id,
            grid,
            ErrorCategoryAreaTask,
            loader,
            processor,
            writer,
            logger,
            fail_on_error=False,
        ).run()
    else:
        ErrorCategoryAreaTask(
            task_id,
            grid.loc[[task_id]],
            loader,
            processor,
            writer,
            logger,
        ).run()


if __name__ == "__main__":
    datetime = "2023"
    version = "0.6.0"
    dataset_id = "coastlines/annual-mosaic"
    namer = DepItemPath(
        sensor="ls",
        dataset_id=dataset_id,
        version=version,
        time=datetime,
    )

    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=namer.log_path(),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )
    task_ids = get_ids(logger)
    # cluster = GatewayCluster(worker_cores=1, worker_memory=8)
    # cluster.scale(100)
    # with cluster.get_client() as client:
    with Client():
        #    print(client.dashboard_link)
        run(task_ids, datetime, namer, logger)
