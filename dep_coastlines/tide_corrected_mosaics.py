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
from typing import Iterable, Tuple, Annotated

from dask.distributed import Client
from odc.geo.geobox import AnchorEnum
import planetary_computer
import pystac_client
from typer import Option, Typer
from xarray import Dataset, DataArray

from azure_logger import CsvLogger
from dep_tools.landsat_utils import cloud_mask
from dep_tools.loaders import SearchLoader
from dep_tools.namers import DepItemPath
from dep_tools.searchers import LandsatPystacSearcher
from dep_tools.stac_utils import set_stac_properties
from dep_tools.processors import LandsatProcessor
from dep_tools.task import ErrorCategoryAreaTask, MultiAreaTask
from dep_tools.utils import get_container_client
from dep_tools.writers import DsWriter

from dep_coastlines.water_indices import mndwi, ndwi, nirwi
from dep_coastlines.ProjOdcLoader import ProjOdcLoader
from dep_coastlines.tide_utils import filter_by_tides, TideLoader
from dep_coastlines.task_utils import get_ids, bool_parser
from dep_coastlines.grid import test_buffered_grid as test_grid

DATASET_ID = "coastlines/mosaics-corrected"
app = Typer()


def mask_clouds_by_day(
    xr: DataArray | Dataset,
    filters: Iterable[Tuple[str, int]] | None = None,
) -> DataArray | Dataset:
    mask = cloud_mask(xr, filters)
    mask.coords["day"] = mask.time.dt.floor("1D")
    mask_by_day = mask.groupby("day").max().sel(day=mask.day)

    if isinstance(xr, DataArray):
        return xr.where(~mask_by_day, xr.rio.nodata)
    else:
        for variable in xr:
            xr[variable] = xr[variable].where(~mask_by_day, xr[variable].rio.nodata)
        return xr


def mad(da, median_da):
    return abs(da - median_da).median(dim="time")


class MosaicProcessor(LandsatProcessor):
    def process(self, xr: Dataset, area) -> Dataset | None:
        # Do the cloud mask first before scale and offset are done
        # When masking by day is stable, move to LandsatProcessor
        xr = super().process(mask_clouds_by_day(xr)).drop_duplicates(...)

        tide_namer = DepItemPath(
            sensor="ls",
            dataset_id="coastlines/tpxo9",
            version="0.7.0",
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
        # ✓ calc geomad,
        # make all_times mosaic and
        # tier 1 only mosaic, etc
        # ✔ set stac properties
        # ✔ check CRS
        # ✓ Any way to not load so many times? In the writer maybe?
        # ✔ clip to area

        xr.coords["time"] = xr.time.dt.floor("1D")

        # or mean or median or whatever
        xr = xr.groupby("time").first().drop_vars(["qa_pixel"])
        xr["mndwi"] = mndwi(xr)
        xr["ndwi"] = ndwi(xr)
        cutoff = 0.128 if self.scale_and_offset else 1280.0
        xr["nirwi"] = nirwi(xr, cutoff)
        xr["meanwi"] = (xr.ndwi + xr.nirwi) / 2
        output = xr.median("time", keep_attrs=True)
        output_mad = mad(xr, output).astype("float32")

        output_mad = output_mad.rename(
            dict((variable, variable + "_mad") for variable in output_mad)
        )
        output = output.merge(output_mad)
        output["count"] = xr.nir08.count("time").fillna(0).astype("int16")
        output["nir08_stdev"] = xr.nir08.std("time", keep_attrs=True).astype("float32")
        output["meanwi_stdev"] = xr.meanwi.std("time", keep_attrs=True).astype(
            "float32"
        )

        output["nirwi_stdev"] = xr.nirwi.std("time", keep_attrs=True).astype("float32")

        output["mndwi_stdev"] = xr.mndwi.std("time", keep_attrs=True).astype("float32")

        output["ndwi_stdev"] = xr.ndwi.std("time", keep_attrs=True).astype("float32")

        scalers = [
            key
            for key in output.keys()
            if not (key.endswith("stdev") or key.endswith("mad") or key == "count")
        ]
        output[scalers] = (output[scalers] * 10_000).astype("int16")

        return set_stac_properties(xr, output).chunk(dict(x=2048, y=2048))


def run(
    task_id: Tuple | list[Tuple] | None,
    datetime: str,
    version: str,
    dataset_id: str = DATASET_ID,
    load_before_write: bool = False,
) -> None:
    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    searcher = LandsatPystacSearcher(
        search_intersecting_pathrows=True,
        client=client,
        datetime=datetime,
    )
    stacloader = ProjOdcLoader(
        datetime=datetime,
        chunks=dict(band=1, time=1, x=8192, y=8192),
        resampling={"qa_pixel": "nearest", "*": "cubic"},
        fail_on_error=False,
        bands=["qa_pixel", "nir08", "swir16", "swir22", "red", "blue", "green"],
        clip_to_area=True,
        dtype="float32",
        anchor=AnchorEnum.CENTER,  # see relevant issue for why this is needed
    )
    loader = SearchLoader(searcher, stacloader)

    processor = MosaicProcessor(
        mask_clouds=True, scale_and_offset=True, send_area_to_processor=True
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
        convert_to_int16=False,
        overwrite=False,
        load_before_write=load_before_write,
        extra_attrs=dict(dep_version=version),
        #        write_multithreaded=False,
    )
    logger = CsvLogger(
        name=DATASET_ID,
        container_client=get_container_client(),
        path=namer.log_path(),
        overwrite=False,
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


@app.command()
def process_id(
    datetime: Annotated[str, Option()],
    version: Annotated[str, Option()],
    row: Annotated[str, Option()],
    column: Annotated[str, Option()],
    load_before_write: Annotated[str, Option(parser=bool_parser)] = "False",
):
    with Client(memory_limit="16GiB"):
        run(
            (int(row), int(column)),
            datetime,
            version,
            load_before_write=load_before_write,
        )


@app.command()
def process_all_ids(
    datetime: Annotated[str, Option()],
    version: Annotated[str, Option()],
    dataset_id=DATASET_ID,
    # If run in argo this needs to be changed but I should not ever do that
    overwrite_log: Annotated[bool, Option()] = False,
    load_before_write: Annotated[str, Option(parser=bool_parser)] = "False",
):
    task_ids = get_ids(
        datetime, version, dataset_id, grid=test_grid, delete_existing_log=overwrite_log
    )

    with Client(memory_limit="16GiB"):
        run(
            task_ids,
            datetime,
            version,
            dataset_id,
            load_before_write=load_before_write,
        )


if __name__ == "__main__":
    app()
