"""
This is a work-in-progress script to "post-process" water index data before
vectorizing. The functionality here is part of the vectorization code from the
DEA and DEAfrica work, but it is primarily raster based, hence the name.
The actual vectorization (using subpixel_contours) may or may not belong here, 
so things may move around / get renamed some in the coming weeks.

Please refer to raster_cleaning.py for specific functions.
"""

from joblib import load
from typing import Tuple, Annotated

from dask.distributed import Client
from dea_tools.classification import predict_xr
from dea_tools.spatial import subpixel_contours
from geopandas import GeoDataFrame
from typer import Option, Typer
from xarray import Dataset, concat

from azure_logger import CsvLogger
from dep_tools.namers import DepItemPath
from dep_tools.processors import Processor
from dep_tools.task import MultiAreaTask, ErrorCategoryAreaTask
from dep_tools.utils import get_container_client, write_to_local_storage

from MosaicLoader import DeluxeMosaicLoader
from raster_cleaning import contours_preprocess
from grid import test_grid
from task_utils import get_ids
from writer import CoastlineWriter


app = Typer()
DATASET_ID = "coastlines/coastlines"


def fill_with_nearest_later_date(xr):
    output = xr.to_dataset("year")
    for year in xr.year.values:
        output[year] = xr.sel(year=year)
        for inner_year in xr.year.values:
            if int(inner_year) <= int(year):
                continue
            output[year] = output[year].where(
                ~output[year].isnull(), output[inner_year]
            )
    return output.to_array(dim="year")


def mask_noisy_water(xr, mask, water_index):
    mean_water_value = dict(mndwi=0.415616, ndwi=0.788478)
    NOISY_WATER_CODE = 7
    return xr.where(mask != NOISY_WATER_CODE, mean_water_value[water_index])


def mask_surf(xr, mask):
    SURF_CODE = 4
    clean_water_nir08 = 56.536145
    return xr.where(mask != SURF_CODE, clean_water_nir08)


class Cleaner(Processor):
    def __init__(
        self,
        water_index: str = "nir08",
        index_threshold: float = -1280.0,
    ):
        super().__init__()
        self.index_threshold = index_threshold
        self.water_index = water_index
        self.mask_model = load("data/dirty_water_shrunk_1Feb.joblib")

    def _calculate_mask(self, input):
        training_columns = [
            "blue",
            "green",
            "nir08",
            "red",
            "swir16",
            "nir08_dev",
            "swir16_dev",
            "swir22_dev",
            "blue_all",
            "green_all",
            "nir08_all",
            "red_all",
            "mndwi",
            "ndwi",
        ]
        masks = []
        for year in input.year:
            year_mask = predict_xr(
                self.mask_model, input.sel(year=year)[training_columns], clean=True
            ).Predictions
            year_mask.coords["year"] = year
            masks.append(year_mask)
        return concat(masks, dim="year")

    def process(self, input: Dataset | list[Dataset]) -> Tuple[Dataset, GeoDataFrame]:
        # TODO
        # save mask somehow

        masks = []
        CLOUD_CODE = 6
        if isinstance(input, list):
            this_mask = self._calculate_mask(input[0])
            masks.append(this_mask)
            ultimate_mask = this_mask
            output = input[0].where(this_mask != CLOUD_CODE)
            for ds in input[1:]:
                ds = ds.sel(year=ds.year[ds.year.isin(output.year)])
                this_mask = self._calculate_mask(ds)
                masks.append(this_mask)
                missings = output.isnull()
                output = output.where(~missings, ds.where(this_mask != CLOUD_CODE))
                ultimate_mask = ultimate_mask.where(~missings.nir08, this_mask)
        else:
            mask = self._calculate_mask(input)
            masks.append(mask)
            output = input.where(mask != CLOUD_CODE)
            ultimate_mask = mask

        if self.water_index in ["mndwi", "ndwi"]:
            output = mask_noisy_water(output, ultimate_mask, self.water_index)

        output = output.where(output["count"] > 1)[self.water_index]
        output = fill_with_nearest_later_date(output)

        if self.water_index == "nir08":
            output = mask_surf(output, ultimate_mask)
            output *= -1

        combined_gdf = subpixel_contours(
            output,
            dim="year",
            z_values=[self.index_threshold],
            min_vertices=10,
        )
        combined_gdf.year = combined_gdf.year.astype(int)

        output["year"] = output.year.astype(str)
        return output.to_dataset("year"), combined_gdf


def run(
    task_id: Tuple | list[Tuple] | None, dataset_id=DATASET_ID, version: str = "0.6.0"
) -> None:
    start_year = 1999
    end_year = 2023
    namer = DepItemPath(
        sensor="ls",
        dataset_id=dataset_id,
        version=version,
        time=f"{start_year}_{end_year}",
        zero_pad_numbers=True,
    )

    loader = DeluxeMosaicLoader(
        start_year=start_year, end_year=end_year, years_per_composite=[1, 3, 5]
    )
    processor = Cleaner(water_index="nir08", index_threshold=-1280.0)
    writer = CoastlineWriter(
        namer,
        extra_attrs=dict(dep_version=version),
        writer=write_to_local_storage,
    )
    logger = CsvLogger(
        name=dataset_id,
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
    row: Annotated[str, Option()],
    column: Annotated[str, Option()],
):
    with Client():
        run((int(row), int(column)))


@app.command()
def process_all_ids(
    version: Annotated[str, Option()],
    overwrite_log: Annotated[bool, Option()] = False,
    dataset_id=DATASET_ID,
    datetime="1999/2023",
):
    task_ids = get_ids(
        datetime, version, dataset_id, grid=test_grid, delete_existing_log=overwrite_log
    )

    with Client():
        run(task_ids, dataset_id=dataset_id, version=version)


if __name__ == "__main__":
    app()
