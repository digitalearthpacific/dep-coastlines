"""Calculates water indices for the given areas and times and saves to blob storage.
As of this writing we are focusing on the use of the nir band for all coastline
work, but the other bands are here for legacy sake and possible use later.

This is best run using kbatch (see calculate_water_indices.yml) with a single
year or group of years (e.g. 2013/2015). We previously tried to run this
variously using all years / all composite years , separately for each variable, 
etc. but the long running / task heavy processes often failed in practice.

Each year took an hour or two to run, so if you start multiple
processes you can calculate for all years within a day or so.

"""
from typing import Union, Iterable, Tuple

from dask.distributed import Client
from odc.algo import erase_bad
import planetary_computer
import pystac_client
from xarray import Dataset, DataArray

from azure_logger import CsvLogger
from dep_tools.landsat_utils import cloud_mask
from dep_tools.loaders import OdcLoader, SearchLoader
from dep_tools.namers import DepItemPath
from dep_tools.searchers import LandsatPystacSearcher
from dep_tools.stac_utils import set_stac_properties
from dep_tools.processors import LandsatProcessor
from dep_tools.task import ErrorCategoryAreaTask, MultiAreaTask
from dep_tools.utils import get_container_client
from dep_tools.writers import DsWriter
from tide_utils import filter_by_tides, TideLoader

from task_utils import get_ids
from grid import test_grid


def mask_clouds_by_day(
    xr: DataArray | Dataset,
    filters: Iterable[Tuple[str, int]] | None = None,
    keep_ints: bool = False,
) -> DataArray | Dataset:
    mask = cloud_mask(xr, filters)
    mask.coords["day"] = mask.time.dt.floor("1D")
    mask_by_day = mask.groupby("day").max().sel(day=mask.day)

    if keep_ints:
        return erase_bad(xr, mask_by_day)
    else:
        return xr.where(~mask_by_day)


class NirProcessor(LandsatProcessor):
    def process(self, xr: Dataset, area) -> Union[Dataset, None]:
        # Check this for misalignment
        xr = mask_clouds_by_day(
            super().process(xr).drop_duplicates(...), keep_ints=True
        )

        tide_namer = DepItemPath(
            sensor="ls",
            dataset_id="coastlines/tpxo9",
            version="0.6.2",
            time="1984_2023",
            zero_pad_numbers=True,
        )
        tide_loader = TideLoader(tide_namer)
        xr = filter_by_tides(xr, area.index[0], tide_loader)

        # In case we filtered out all the data
        if not "time" in xr.coords or len(xr.time) == 0:
            return None

        # TODO:
        # ✔ check cloud mask merging,
        # calc geomad,
        # make all_times mosaic and
        # tier 1 only mosaic, etc
        # ✔ set stac properties
        # check CRS
        # Any way to not load so many times? In the writer maybe?
        # ✔ clip to area

        xr.coords["time"] = xr.time.dt.floor("1D")

        # or mean or median or whatever
        xr = xr.groupby("time").first()
        output = xr.nir08.median("time", keep_attrs=True).to_dataset()
        output["count"] = xr.nir08.count("time", keep_attrs=True).astype("int16")
        output["nir_stdev"] = xr.nir08.std("time", keep_attrs=True)
        return set_stac_properties(xr, output)


def run(task_id: str | list[str], datetime: str, version: str, dataset_id: str) -> None:
    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    searcher = LandsatPystacSearcher(
        search_intersecting_pathrows=True,
        client=client,
        datetime=datetime,
        # pystac_client_search_kwargs=dict(query=["landsat:collection_category=T1"]),
        # exclude_platforms=["landsat-7"],
    )
    stacloader = OdcLoader(
        datetime=datetime,
        chunks=dict(band=1, time=1, x=4096, y=4096),
        resampling={"qa_pixel": "nearest", "*": "cubic"},
        fail_on_error=False,
        bands=["qa_pixel", "nir08"],
        clip_to_area=True,
        dtype="float32",
    )
    loader = SearchLoader(searcher, stacloader)

    processor = NirProcessor(
        mask_clouds=False, scale_and_offset=False, send_area_to_processor=True
    )

    namer = DepItemPath(
        sensor="ls",
        dataset_id=dataset_id,
        version=version,
        time=datetime.replace("/", "_"),
        zero_pad_numbers=True,
    )
    writer = DsWriter(
        itempath=namer,
        overwrite=False,
        extra_attrs=dict(dep_version=version),
    )
    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=namer.log_path(),
        overwrite=True,
        header="time|index|status|paths|comment\n",
    )

    if isinstance(task_id, list):
        MultiAreaTask(
            task_id,
            test_grid,
            ErrorCategoryAreaTask,
            loader,
            processor,
            writer,
            logger,
        ).run()
    else:
        ErrorCategoryAreaTask(
            task_id, test_grid.loc[[task_id]], loader, processor, writer, logger
        ).run()


if __name__ == "__main__":
    datetime = "2018"
    version = "0.6.0"
    dataset_id = "coastlines/nir08-corrected"
    task_ids = get_ids(datetime, version, dataset_id, grid=test_grid)

    with Client() as client:
        run(task_ids, datetime, version, dataset_id)
