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
from numpy import mean
from numpy.lib.stride_tricks import sliding_window_view
from typer import Option, Typer
from xarray import Dataset, concat

from azure_logger import CsvLogger
from dep_tools.loaders import Loader
from dep_tools.namers import DepItemPath
from dep_tools.processors import Processor
from dep_tools.task import MultiAreaTask, ErrorCategoryAreaTask
from dep_tools.utils import get_container_client

from MosaicLoader import MosaicLoader
from water_indices import mndwi, ndwi, wofs
from raster_cleaning import contours_preprocess
from grid import test_grid
from task_utils import get_ids
from writer import CoastlineWriter


app = Typer()
DATASET_ID = "coastlines/coastlines"


def _set_year_to_middle_year(xr: Dataset) -> Dataset:
    edge_years = [y.split("/") for y in xr.year.to_numpy()]
    middle_years = [int(mean([int(y[0]), int(y[1])])) for y in edge_years]
    xr["year"] = middle_years
    return xr


def get_datetimes(start_year, end_year, years_per_composite):
    # nearly duplicated in task_utils, should probably refactor
    # yeah, just switch to get_composite_datetime and make years separate
    assert years_per_composite % 2 == 1
    year_buffer = int((years_per_composite - 1) / 2)
    years = range(int(start_year) - year_buffer, int(end_year) + 1 + year_buffer)
    if years_per_composite > 1:
        years = [
            f"{y[0]}/{y[years_per_composite - 1]}"
            for y in sliding_window_view(list(years), years_per_composite)
        ]
    return [str(y) for y in years]


class MultiyearMosaicLoader(Loader):
    def __init__(self, start_year, end_year, years_per_composite=1):
        super().__init__()
        self._start_year = start_year
        self._end_year = end_year
        self._years_per_composite = years_per_composite
        self._container_client = get_container_client()

    def load(self, area) -> Tuple[Dataset, Dataset]:
        dss = []
        for datetime in get_datetimes(
            self._start_year, self._end_year, self._years_per_composite
        ):
            itempath = DepItemPath(
                sensor="ls",
                dataset_id="coastlines/mosaics-corrected",
                version="0.6.0",
                time=datetime.replace("/", "_"),
                zero_pad_numbers=True,
            )
            loader = MosaicLoader(
                itempath=itempath, container_client=self._container_client
            )
            dss.append(
                loader.load(area.index.values[0]).assign_coords({"year": datetime})
            )

        output = concat(dss, dim="year")
        if self._years_per_composite > 1:
            output = _set_year_to_middle_year(output)

        # output["mndwi"] = mndwi(output)
        # output["ndwi"] = ndwi(output)
        # output["wofs"] = wofs(output)
        return output


class Cleaner(Processor):
    def __init__(
        self,
        water_index: str = "nir08",
        index_threshold: float = -1280.0,
        masking_index: str = "mndwi",
        masking_threshold: float = 0,
    ):
        super().__init__()
        self.index_threshold = index_threshold
        self.water_index = water_index
        self.masking_index = masking_index
        self.masking_threshold = masking_threshold
        self.mask_model = load("data/cleaning_model.joblib")

    def _calculate_mask(self, input):
        masks = []
        for year in input.year:
            year_mask = predict_xr(
                self.mask_model, input.sel(year=year), clean=True
            ).Predictions.astype(bool)
            year_mask.coords["year"] = year
            masks.append(year_mask)
        return concat(masks, dim="year")

    def process(self, input: Tuple[Dataset, Dataset]) -> Tuple[Dataset, GeoDataFrame]:
        # yearly_ds, composite_ds = input

        # thresholding for nir band is the opposite direction of
        # all other indices, so we multiply by negative 1.
        #        if "nir08" in yearly_ds:
        #            yearly_ds["nir08"] = yearly_ds.nir08 * -1
        #            composite_ds["nir08"] = composite_ds.nir08 * -1

        #        combined_ds = contours_preprocess(
        #            yearly_ds,
        #            composite_ds,
        #            water_index=self.water_index,
        #            masking_index=self.masking_index,
        #            masking_threshold=self.masking_threshold,
        #            mask_nir=True,
        #            mask_ephemeral_land=True,
        #            mask_ephemeral_water=False,
        #            mask_esa_water_land=True,
        #            remove_tiny_areas=False,
        #            remove_inland_water=False,
        #            remove_water_noise=False,
        #        )

        # TODO
        # save mask somehow
        # fill in empty from other composites

        mask = self._calculate_mask(input)
        output = input[self.water_index].where(~mask)

        if self.water_index == "nir08":
            output *= -1

        combined_gdf = subpixel_contours(
            output, dim="year", z_values=[self.index_threshold], min_vertices=3
        )
        combined_gdf.year = combined_gdf.year.astype(int)

        output["year"] = output.year.astype(str)
        return output.to_dataset("year"), combined_gdf


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

    loader = MultiyearMosaicLoader(start_year, end_year, years_per_composite=5)
    processor = Cleaner()
    writer = CoastlineWriter(
        namer,
        overwrite=True,
        extra_attrs=dict(dep_version=version),
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
