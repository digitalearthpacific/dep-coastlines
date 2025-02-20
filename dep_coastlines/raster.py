"""Calculates water indices for the given areas and times and saves to s3.
"""

from datetime import datetime
from typing import Iterable, Tuple, Annotated

import boto3
from dask.distributed import Client
from odc.geo.geobox import AnchorEnum
from odc.stac import configure_s3_access
from pystac_client import Client as PystacClient
from rasterio.errors import RasterioIOError, WarpOperationError
from retry import retry
from typer import Option, run
from xarray import Dataset, DataArray

from dep_tools.exceptions import EmptyCollectionError, NoOutputError
from dep_tools.landsat_utils import cloud_mask
from dep_tools.processors import LandsatProcessor, XrPostProcessor
from dep_tools.searchers import LandsatPystacSearcher
from dep_tools.task import AwsStacTask
from dep_tools.utils import scale_to_int16
from dep_tools.writers import AwsDsCogWriter

from dep_coastlines.common import (
    coastlineItemPath,
    coastlineLogger,
    use_alternate_s3_href,
)
from dep_coastlines.config import MOSAIC_DATASET_ID
from dep_coastlines.grid import buffered_grid as grid
from dep_coastlines.io import ProjOdcLoader
from dep_coastlines.tide_utils import filter_by_tides, tides_for_items
from dep_coastlines.time_utils import (
    composite_from_years,
    parse_datetime,
    years_from_yearstring,
)
from dep_coastlines.task_utils import bool_parser
from dep_coastlines.water_indices import twndwi

client = PystacClient.open(
    "https://landsatlook.usgs.gov/stac-server",
    modifier=use_alternate_s3_href,
)


def process_all_years(area, all_time: str, itempath, **kwargs):
    years = composite_from_years(parse_datetime(all_time.replace("/", "_")), [1, 3])
    task_class = AwsStacTask
    items = LandsatPystacSearcher(
        search_intersecting_pathrows=True,
        datetime=all_time,
        client=client,
        collections=["landsat-c2l2-sr"],
    ).search(area)
    tides = tides_for_items(items, area)
    processor = MosaicProcessor(
        tides,
        mask_clouds=True,
        scale_and_offset=True,
        send_area_to_processor=True,
    )

    @retry(
        (WarpOperationError, RasterioIOError),
        tries=5,
        delay=60,
        backoff=2,
        max_delay=480,
    )
    def _run_single_task(task) -> str | list[str]:
        return task.run()

    paths = []
    for year in years:
        print(year)
        configure_s3_access(cloud_defaults=True, requester_pays=True)
        itempath.time = year.replace("/", "_")
        try:
            paths += _run_single_task(
                task_class(
                    itempath=itempath,
                    processor=processor,
                    searcher=ItemFilterer(items, year),
                    area=area,
                    **kwargs,
                )
            )
        except (EmptyCollectionError, NoOutputError):
            continue

    return paths


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


class MosaicProcessor(LandsatProcessor):
    def __init__(self, all_tides, **kwargs):
        super().__init__(**kwargs)
        self._tides = all_tides

    def process(self, xr: Dataset, area) -> Dataset | None:
        xr = xr.rio.clip(
            area.to_crs(xr.rio.crs).geometry,
            all_touched=True,
            from_disk=True,
            drop=True,
        )
        # Do the cloud mask first before scale and offset are done
        # When masking by day is stable, move to LandsatProcessor
        xr = super().process(mask_clouds_by_day(xr)).drop_duplicates(...)
        self._tides["time"] = self._tides.time.astype(xr.time.dtype)
        xr = filter_by_tides(xr, self._tides)

        # In case we filtered out all the data
        if not "time" in xr.coords or len(xr.time) == 0:
            raise NoOutputError

        # Limit to one reading per day. This can be accomplished by
        # using groupby="solarday" when loading, but I discovered that landsat
        # masks are not consistent between images (see `mask_clouds_by_day`).
        xr.coords["time"] = xr.time.dt.floor("1D")
        xr = xr.groupby("time").first().drop_vars(["qa_pixel"])

        # This is the nir cutoff for water / land established in
        # https://doi.org/10.1186/s42834-019-0016-5
        cutoff = 0.128 if self.scale_and_offset else 1280.0
        xr["twndwi"] = twndwi(xr, nir_cutoff=cutoff)
        output = xr.median("time", keep_attrs=True)
        output_mad = mad(xr, output).astype("float32")

        output_mad = output_mad.rename(
            dict((variable, variable + "_mad") for variable in output_mad)
        )
        output = output.merge(output_mad)
        output["count"] = xr.nir08.count("time").fillna(0).astype("int16")
        output["twndwi_stdev"] = xr.twndwi.std("time", keep_attrs=True).astype(
            "float32"
        )

        scalers = [
            key
            for key in output.keys()
            if not (key.endswith("stdev") or key.endswith("mad") or key == "count")
        ]
        output[scalers] = scale_to_int16(
            output[scalers], output_multiplier=10_000, output_nodata=-32767
        )

        return output


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


def process_id(
    task_id: Tuple | list[Tuple] | None,
    version: str,
    datetime: str = "1984/2024",
    dataset_id: str = MOSAIC_DATASET_ID,
    load_before_write: bool = True,
    fail_on_read_error: bool = True,
) -> None:
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
        paths = process_all_years(
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


def main(
    row: Annotated[str, Option()],
    column: Annotated[str, Option()],
    version: Annotated[str, Option()],
    load_before_write: Annotated[str, Option(parser=bool_parser)] = "True",
    fail_on_read_error: Annotated[str, Option(parser=bool_parser)] = "True",
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
