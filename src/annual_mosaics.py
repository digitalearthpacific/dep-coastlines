from typing import Union

import geopandas as gpd
from numpy.lib.stride_tricks import sliding_window_view
from xarray import DataArray, Dataset

from azure_logger import CsvLogger, get_log_path, filter_by_log
from dep_tools.runner import run_by_area_dask, run_by_area
from dep_tools.loaders import LandsatOdcLoader
from dep_tools.processors import LandsatProcessor
from dep_tools.utils import get_container_client
from dep_tools.writers import AzureXrWriter
from tide_utils import filter_by_tides


class MosaicProcessor(LandsatProcessor):
    def process(self, xr: DataArray, area) -> Union[Dataset, None]:
        xr = super().process(xr).drop_duplicates(...)
        xr = filter_by_tides(xr, area.index[0], area)
        working_ds = xr.to_dataset("band")[
            ["red", "blue", "green", "nir08", "swir16", "swir22"]
        ]

        # In case we filtered out all the data
        if not "time" in working_ds or len(working_ds.time) == 0:
            return None

        working_ds.coords["time"] = xr.time.dt.floor("1D")

        working_ds = working_ds.groupby("time").first()
        return working_ds.median("time", keep_attrs=True)


def main(datetime: str, version: str) -> None:
    aoi_by_tile = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
    ).set_index(["PATH", "ROW"], drop=False)

    dataset_id = "annual-mosaic"
    prefix = f"coastlines/{version}"

    loader = LandsatOdcLoader(
        datetime=datetime,
        dask_chunksize=dict(band=1, time=1, x=4096, y=4096),
        odc_load_kwargs=dict(
            resampling={"qa_pixel": "nearest", "*": "cubic"},
            fail_on_error=False,
        ),
        pystac_client_search_kwargs=dict(query=["landsat:collection_category=T1"]),
        exclude_platforms=["landsat-7"],
    )

    processor = MosaicProcessor(
        scale_and_offset=False,
        send_area_to_processor=True,
        dilate_mask=True,
    )

    writer = AzureXrWriter(
        dataset_id=dataset_id,
        year=datetime,
        prefix=prefix,
        overwrite=False,
        output_value_multiplier=1,
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

    run_by_area_dask(
        areas=aoi_by_tile,
        loader=loader,
        processor=processor,
        writer=writer,
        logger=logger,
        n_workers=20,
        worker_memory=16,
    )


if __name__ == "__main__":
    single_years = list(range(2013, 2024))
    composite_years = [f"{y[0]}/{y[2]}" for y in sliding_window_view(single_years, 3)]
    all_years = single_years + composite_years

    for year in all_years:
        main(str(year), "0_3_4")
