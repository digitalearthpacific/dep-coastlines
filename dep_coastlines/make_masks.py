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

from MosaicLoader import MultiyearMosaicLoader
from grid import test_grid
from task_utils import get_ids
from writer import CompositeWriter


app = Typer()
DATASET_ID = "coastlines/mask"

training_columns = [
    "blue",
    "blue_mad",
    "count",
    "green",
    "green_mad",
    "nir08",
    "nir08_mad",
    "nir08_stdev",
    "red",
    "red_mad",
    "swir16",
    "swir16_mad",
    "swir22",
    "swir22_mad",
    "blue_all",
    "green_all",
    "nir08_all",
    "red_all",
    "swir16_all",
    "swir22_all",
]


class MaskMaker(Processor):
    def __init__(self, model=load("data/cleaning_model.joblib")):
        super().__init__()
        self._model = model

    def process(self, input):
        all_time = input.median(dim="year").compute()
        all_time = all_time.rename({k: k + "_all" for k in all_time.keys()})
        masks = []
        for year in input.year:
            input_ds = input.sel(year=year).merge(
                all_time.chunk(dict(x=input.chunks["x"], y=input.chunks["y"]))
            )[training_columns]
            year_mask = predict_xr(
                self._model,
                input_ds,
                clean=True,
            ).Predictions.astype("int8")
            year_mask.coords["year"] = year
            masks.append(year_mask)
        output = concat(masks, dim="year").to_dataset("year")
        del output.attrs["grid_mapping"]  # tsk tsk
        return output


def run(task_id: Tuple | list[Tuple] | None, dataset_id=DATASET_ID) -> None:
    version = "0.6.0"
    start_year = 1999
    end_year = 2023
    namer = DepItemPath(
        sensor="ls",
        dataset_id=dataset_id,
        version=version,
        time=f"{start_year}_{end_year}",
        zero_pad_numbers=True,
    )

    loader = MultiyearMosaicLoader(start_year, end_year, years_per_composite=1)
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
        run(task_ids)


if __name__ == "__main__":
    app()
