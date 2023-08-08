"""
Calculates water indices for the given areas and times and saves to blob storage.
As of this writing we are focusing on the use of the nir band for all coastline
work, but the other bands are here for legacy sake and possible use later.

This is best run using kbatch (see calculate_water_indices.yml) with a single
year or group of years (e.g. 2013/2015). We previously tried to run this
variously using all years / all composite years , separately for each variable, 
etc. but the long running / task heavy processes often failed in practice.

Each year took an hour or two to run, so if you start multiple
processes you can calculate for all years within a day or so.

"""
from ast import literal_eval

import geopandas as gpd
from dask.distributed import Client
from pandas import DataFrame
from rasterio.enums import Resampling
from xarray import DataArray, Dataset

from azure_logger import CsvLogger
from dep_tools.runner import run_by_area_dask, run_by_area
from dep_tools.loaders import LandsatOdcLoader, LandsatStackLoader
from dep_tools.processors import LandsatProcessor
from dep_tools.utils import get_container_client
from dep_tools.writers import AzureXrWriter
from tide_utils import filter_by_tides


class NirProcessor(LandsatProcessor):
    def process(self, xr: DataArray, area) -> Dataset:
        xr = super().process(xr)
        xr = xr.drop_duplicates(...).sel(band="nir08")
        working_ds = filter_by_tides(
            xr, area["PATH"].values[0], area["ROW"].values[0], area
        ).to_dataset(name="nir08")

        # In case we filtered out all the data
        if not "time" in working_ds or len(working_ds.time) == 0:
            return None

        working_ds.coords["time"] = working_ds.time.dt.floor("1D")

        # or mean or median or whatever
        working_ds = working_ds.groupby("time").first()

        output = working_ds.median("time", keep_attrs=True)
        output["count"] = working_ds.nir08.count("time", keep_attrs=True).astype(
            "int16"
        )
        output["stdev"] = working_ds.nir08.std("time", keep_attrs=True)
        return output.unify_chunks()


def get_log_path(prefix: str, dataset_id: str, version: str, datetime: str) -> str:
    return f"{prefix}/{dataset_id}/logs/{dataset_id}_{version}_{datetime.replace('/', '_')}_log.csv"


def main(datetime: str, version: str) -> None:
    aoi_by_tile = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
    ).set_index(["PATH", "ROW"], drop=False)

    dataset_id = "nir08"
    prefix = f"coastlines/{version}"

    #    loader = LandsatOdcLoader(
    #        datetime=datetime,
    #        dask_chunksize=dict(band=1, time=1, x=1024, y=1024),
    #        odc_load_kwargs=dict(
    #            resampling={"qa_pixel": "nearest", "*": "cubic"},
    #            fail_on_error=False,
    #            bands=["qa_pixel", "nir08"],
    #        ),
    #    )

    loader = LandsatStackLoader(
        datetime=datetime,
        dask_chunksize=1024,
        exclude_platforms=["landsat-7"],
        resamplers_and_assets=[
            {"resampler": Resampling.nearest, "assets": ["qa_pixel"]},
            {"resampler": Resampling.cubic, "assets": ["nir08"]},
        ],
    )

    processor = NirProcessor(send_area_to_processor=True)
    writer = AzureXrWriter(
        dataset_id=dataset_id,
        year=datetime,
        prefix=prefix,
        convert_to_int16=True,
        overwrite=False,
        output_value_multiplier=10000,
        extra_attrs=dict(dep_version=version),
    )
    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=get_log_path(prefix, dataset_id, version, datetime),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )

    aoi_by_tile = filter_by_log(aoi_by_tile, logger.parse_log())

    #    with Client():
    #        run_by_area(
    #            areas=aoi_by_tile,
    #            loader=loader,
    #            processor=processor,
    #            writer=writer,
    #            logger=logger,
    #        )

    run_by_area_dask(
        areas=aoi_by_tile,
        loader=loader,
        processor=processor,
        writer=writer,
        logger=logger,
        worker_memory=16,
    )


def filter_by_log(df: DataFrame, log: DataFrame) -> DataFrame:
    # Need to decide if this is where we do this. I want to keep the logger
    # fairly generic. Suppose we could subclass it.
    log = log.set_index("index")
    log.index = [literal_eval(i) for i in log.index]

    # Need to filter by errors

    return df[~df.index.isin(log.index)]


if __name__ == "__main__":
    main("2014/2016", "3Aug2023")
