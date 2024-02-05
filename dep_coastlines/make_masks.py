from joblib import load
from typing import Tuple, Annotated

from dask.distributed import Client
from dea_tools.classification import predict_xr
from dea_tools.spatial import subpixel_contours
from geopandas import GeoDataFrame
from numpy import isin, mean
from numpy.lib.stride_tricks import sliding_window_view
from typer import Option, Typer
from xarray import Dataset, concat

from azure_logger import CsvLogger
from dep_tools.loaders import Loader
from dep_tools.namers import DepItemPath
from dep_tools.processors import Processor
from dep_tools.task import MultiAreaTask, ErrorCategoryAreaTask
from dep_tools.utils import get_container_client, write_to_local_storage

from MosaicLoader import DeluxeMosaicLoader
from grid import test_grid
from task_utils import get_ids
from writer import CompositeWriter


app = Typer()
DATASET_ID = "coastlines/mask"

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


class MaskMaker(Processor):
    def __init__(self, model=load("data/dirty_water_shrunk_1Feb.joblib")):
        super().__init__()
        self._model = model

    def process(self, input):
        masks = []
        for year in input.year:
            input_ds = input.sel(year=year)[training_columns]
            year_mask = predict_xr(
                self._model,
                input_ds,
                clean=True,
            ).Predictions.where(~input_ds.red.isnull())
            year_mask.coords["year"] = year
            masks.append(year_mask)
        output = concat(masks, dim="year").to_dataset("year")
        del output.attrs["grid_mapping"]  # tsk tsk
        return output


def run(
    task_id: Tuple | list[Tuple] | None, version="0.6.0", dataset_id=DATASET_ID
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
        start_year=start_year, end_year=end_year, years_per_composite=1
    )
    processor = MaskMaker()
    writer = CompositeWriter(
        namer,
        driver="COG",
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
        run(task_ids, version=version, dataset_id=dataset_id)


if __name__ == "__main__":
    app()
