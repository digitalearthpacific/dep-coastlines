"""Calculates water indices for the given areas and times and saves to blob storage.
"""

from typing import Iterable, Tuple, Annotated

import boto3
from dask.distributed import Client
from odc.geo.geobox import AnchorEnum
from odc.stac import configure_s3_access
from typer import Option, run
from xarray import Dataset, DataArray

from dep_tools.exceptions import EmptyCollectionError, NoOutputError
from dep_tools.landsat_utils import cloud_mask
from dep_tools.namers import DepItemPath
from dep_tools.processors import LandsatProcessor, XrPostProcessor
from dep_tools.searchers import LandsatPystacSearcher
from dep_tools.stac_utils import set_stac_properties
from dep_tools.task import MultiAreaTask, AwsStacTask
from dep_tools.utils import scale_to_int16
from dep_tools.writers import AwsDsCogWriter

from dep_coastlines.common import coastlineItemPath, coastlineLogger
from dep_coastlines.grid import buffered_grid as grid
from dep_coastlines.io import ProjOdcLoader, TideLoader
from dep_coastlines.calculate_tides import TideProcessor
from dep_coastlines.tide_utils import filter_by_tides
from dep_coastlines.time_utils import composite_from_years, parse_datetime
from dep_coastlines.task_utils import bool_parser
from dep_coastlines.water_indices import twndwi

DATASET_ID = "coastlines/interim/mosaic"

from urllib3 import Retry

from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO

retry = Retry(
    total=20, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None
)
stac_api_io = StacApiIO(max_retries=retry)
client = Client.open("https://earth-search.aws.element84.com/v1", stac_io=stac_api_io)


class MultiYearTask:
    def __init__(self, all_time: str, itempath, searcher, **kwargs):
        self._years = composite_from_years(
            parse_datetime(all_time.replace("/", "_")), [1, 3]
        )
        self._itempath = itempath
        self._searcher = searcher
        self._kwargs = kwargs
        self._task_class = AwsStacTask

    def run(self):
        paths = []
        for year in self._years:
            self._itempath.time = year.replace("/", "_")
            # Re-search instead of filtering the dates. We could do the
            # latter if the searches take too much time
            year_searcher = LandsatPystacSearcher(
                search_intersecting_pathrows=True,
                catalog="https://earth-search.aws.element84.com/v1",
                datetime=year,
            )
            year_searcher._client = client
            try:
                paths += self._task_class(
                    itempath=self._itempath,
                    searcher=year_searcher,
                    **self._kwargs,
                ).run()
            except (EmptyCollectionError, NoOutputError):
                continue
        return paths


def load_tides(area, full_time="1984/2023"):
    items = LandsatPystacSearcher(
        search_intersecting_pathrows=True,
        catalog="https://earth-search.aws.element84.com/v1",
        datetime=full_time,
    ).search(area)

    ds = ProjOdcLoader(
        fail_on_error=False,
        chunks=dict(band=1, time=1, x=1024, y=1024),
        bands=["red"],
    ).load(items, area)

    return TideProcessor(
        send_area_to_processor=True, tide_directory="data/raw/tidal_models"
    ).process(ds, area)


class MosaicProcessor(LandsatProcessor):
    def __init__(self, area, full_time="1984/2023", **kwargs):
        super().__init__(**kwargs)
        self._tides = load_tides(area, full_time)

    def process(self, xr: Dataset, area) -> Dataset | None:
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

        return output.chunk(dict(x=2048, y=2048))


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
    datetime: str = "1984/2023",
    dataset_id: str = DATASET_ID,
    load_before_write: bool = False,
) -> None:
    searcher = LandsatPystacSearcher(
        search_intersecting_pathrows=True,
        catalog="https://earth-search.aws.element84.com/v1",
        datetime=datetime,
    )
    searcher._client = client
    loader = ProjOdcLoader(
        chunks=dict(band=1, time=1, x=8192, y=8192),
        resampling={"qa_pixel": "nearest", "*": "cubic"},
        fail_on_error=False,
        bands=["qa_pixel", "nir08", "swir16", "swir22", "red", "blue", "green"],
        clip_to_area=True,
        dtype="float32",
        anchor=AnchorEnum.CENTER,  # see relevant issue for why this is needed
    )

    area = grid.loc[[task_id]]
    processor = MosaicProcessor(
        mask_clouds=True, scale_and_offset=True, send_area_to_processor=True, area=area
    )

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
        paths = MultiYearTask(
            all_time=datetime,
            itempath=namer,
            searcher=searcher,
            post_processor=post_processor,
            id=task_id,
            area=area,
            loader=loader,
            processor=processor,
            writer=writer,
            logger=logger,
        ).run()
    except Exception as e:
        logger.error([task_id, "error", e])
        raise e

    logger.info([task_id, "complete", paths])


def main(
    row: Annotated[str, Option()],
    column: Annotated[str, Option()],
    version: Annotated[str, Option()],
    load_before_write: Annotated[str, Option(parser=bool_parser)] = "True",
):
    configure_s3_access(cloud_defaults=True, requester_pays=True)
    boto3.setup_default_session()
    with Client(memory_limit="16GiB"):
        process_id(
            (int(column), int(row)),
            version,
            load_before_write=load_before_write,
        )


if __name__ == "__main__":
    run(main)
