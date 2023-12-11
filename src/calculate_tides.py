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
from xarray import DataArray, Dataset

from dea_tools.coastal import pixel_tides

from azure_logger import CsvLogger, filter_by_log
from dep_tools.loaders import LandsatOdcLoader
from dep_tools.namers import DepItemPath
from dep_tools.processors import Processor
from dep_tools.runner import run_by_area
from dep_tools.utils import get_container_client

from dep_tools.writers import DsWriter


class TideProcessor(Processor):
    def process(self, xr: DataArray, area) -> Dataset:
        working_ds = xr.isel(band=0).to_dataset().drop_duplicates(...)

        tides_lowres = (
            pixel_tides(
                working_ds,
                resample=False,
                model="TPXO9-atlas-v5",
                directory="../coastlines-local/tidal-models/",
                resolution=4980,
            ).transpose("time", "y", "x")
            # For 5km resolution, this is reasonable chunking
            .chunk(time=1, x=1, y=1)
        )

        tides_lowres.coords["time"] = tides_lowres.coords["time"].astype("str")
        return tides_lowres


def get_ids(datetime, version, dataset_id, retry_errors)
    grid = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
    ).set_index(["PATH", "ROW"])

    namer = DepItemPath(
        sensor="ls", dataset_id="coastlines/tpx09", version=version, time=datetime
    )

    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=namer.log_path(),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )
    grid = filter_by_log(grid, logger.parse_log(), retry_errors)
    years = get_years_from_datetime(datetime)
    return product(grid.index, years)


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


def run(task_id: str | list[str], datetime: str, version: str) -> None:
    dataset_id = "coastlines/tpx09"

    loader = LandsatOdcLoader(
        datetime=datetime,
        dask_chunksize=dict(band=1, time=1, x=1024, y=1024),
        odc_load_kwargs=dict(fail_on_error=False),
    )

    processor = TideProcessor(send_area_to_processor=True)

    writer = DsWriter(
        itempath=namer,
        overwrite=False,
        convert_to_int16=False,
        extra_attrs=dict(dep_version=version),
        output_nodata=0,  # <- check this
    )

    logger = CsvLogger( name=dataset_id, container_client=get_container_client(),
        path=namer.log_path(),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )

    if isinstance(task_id, list):
        MultiTask(task_id, 
                  grid,
        ErrorCategoryAreaTask,
                  loader, processor,writer,logger).run()
    else:
        ErrorCategoryAreaTask(task_id, grid.loc[[task_id]], loader, processor, writer, logger).run()






if __name__ == "__main__":
    with Client() as client:
        main("1984/2023", "0.6.0", client)
