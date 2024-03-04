"""
This is a work-in-progress script to "post-process" water index data before
vectorizing. The functionality here is part of the vectorization code from the
DEA and DEAfrica work, but it is primarily raster based, hence the name.
The actual vectorization (using subpixel_contours) may or may not belong here, 
so things may move around / get renamed some in the coming weeks.

Please refer to raster_cleaning.py for specific functions.
"""

from joblib import load
import operator
from pathlib import Path
from typing import Tuple, Annotated, Callable

from dask.distributed import Client
from dea_tools.classification import predict_xr
from dea_tools.spatial import subpixel_contours
from geopandas import GeoDataFrame
from odc.algo import mask_cleanup
from typer import Option, Typer
from xarray import Dataset, concat, apply_ufunc
import xrspatial as xs
from skimage.measure import label

from azure_logger import CsvLogger
from dep_tools.namers import DepItemPath
from dep_tools.processors import Processor
from dep_tools.task import MultiAreaTask, ErrorCategoryAreaTask
from dep_tools.utils import get_container_client, write_to_local_storage

from dep_coastlines.MosaicLoader import MultiyearMosaicLoader
from dep_coastlines.raster_cleaning import (
    load_gadm_land,
    find_inland_areas,
)
from dep_coastlines.grid import test_grid
from dep_coastlines.mask_model import SavedModel
from dep_coastlines.task_utils import get_ids
from dep_coastlines.writer import CoastlineWriter


app = Typer()
DATASET_ID = "coastlines/coastlines"


def remove_disconnected_land(certain_land, candidate_land):
    if candidate_land.year.size > 1:
        return candidate_land.groupby("year").map(
            lambda candidate_year: remove_disconnected_land(
                certain_land, candidate_year
            )
        )

    zones = apply_ufunc(
        label, candidate_land, None, 0, dask="parallelized", kwargs=dict(connectivity=1)
    )
    connected_or_not = xs.zonal_stats(
        zones, certain_land.astype("int8"), stats_funcs=["max"]
    )
    connected_zones = connected_or_not["zone"][connected_or_not["max"] == 1]
    return candidate_land.where(zones.isin(connected_zones)) == 1


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


class ModelPredictor:
    def __init__(self, model: SavedModel):
        self.model = model
        self.codes = self.model.codes.groupby(self.model.response_column).first()

    def code_for_name(self, name):
        return (
            self.model.codes.reset_index()
            .set_index("code")
            .loc[name, self.model.response_column]
        )

    def calculate_mask(self, input):
        masks = []
        for year in input.year:
            year_mask = predict_xr(
                self.model.model,
                input.sel(year=year)[self.model.predictor_columns],
                clean=True,
            ).Predictions
            year_mask.coords["year"] = year
            masks.append(year_mask)
        return concat(masks, dim="year")

    def apply_mask(self, input):
        # masks = []
        cloud_code = self.code_for_name("cloud")
        if isinstance(input, list):
            this_mask = self.calculate_mask(input[0])
            # masks.append(this_mask)
            # ultimate_mask = this_mask
            output = input[0].where(this_mask != cloud_code, drop=False)
            for ds in input[1:]:
                ds = ds.sel(year=ds.year[ds.year.isin(output.year)])
                this_mask = self.calculate_mask(ds)
                # masks.append(this_mask)
                missings = output.isnull()
                output = output.where(
                    ~missings, ds.where(this_mask != cloud_code, drop=False)
                )
        #                ultimate_mask = ultimate_mask.where(~missings.nir08, this_mask)
        else:
            mask = self.calculate_mask(input)
            # masks.append(mask)
            output = input.where(mask != cloud_code)
        #            ultimate_mask = mask
        return output


def calculate_consensus_land(ds):
    return (ds.nir08_all > 1280.0) & (ds.mndwi_all < 0) & (ds.ndwi_all < 0)


class Cleaner(Processor):
    def __init__(
        self,
        water_index: str = "meanwi",
        index_threshold: float = 0,
        comparison: Callable = operator.lt,
        number_of_expansions: int = 8,
        model_file: Path = Path(__file__).parent / "../data/model_26Feb.joblib",
    ):
        super().__init__()
        self.index_threshold = index_threshold
        self.water_index = water_index
        #        model_dict = load(model_file)
        #        model = SavedModel(
        #            model=model_dict["model"],
        #            training_data=model_dict["training_data"],
        #            predictor_columns=model_dict["predictor_columns"],
        #            response_column=model_dict["response_column"],
        #            codes=model_dict["codes"],
        #        )

        self.model = ModelPredictor(load(model_file))
        # self.model = ModelPredictor(model)
        self.comparison = comparison
        self.number_of_expansions = number_of_expansions

    def land(self, output):
        return self.comparison(output[self.water_index], self.index_threshold)

    def water(self, output):
        return output[self.water_index] >= self.index_threshold

    def expand_analysis_zone(self, analysis_zone, output):
        # Only expand where there's an edge that's land. Do it multiple times
        # to fill between larger areas
        # say in Funafuti or Majiro. Later we will fill one last time with
        # water to ensure lines are closed.
        for _ in range(self.number_of_expansions):
            analysis_zone = analysis_zone | mask_cleanup(
                self.land(output.where(analysis_zone)),
                mask_filters=[("dilation", 1)],
            )
        return analysis_zone

    def process(self, input: Dataset | list[Dataset]) -> Tuple[Dataset, GeoDataFrame]:
        breakpoint()
        output = self.model.apply_mask(input)

        obvious_water = 0.5
        output = output[[self.water_index]].compute()
        output[self.water_index] = fill_with_nearest_later_date(
            output[self.water_index]
        )

        candidate_land = self.land(output)
        consensus_land = calculate_consensus_land(input[0].isel(year=0)).compute()
        # Connected areas are contiguous zones that are connected in some way to
        # the consensus areas. This ensures that all edges of these (on the basis of nir)
        # are included
        connected_areas = remove_disconnected_land(consensus_land, candidate_land)
        # erosion of 2 here borks e.g. funafuti but is needed for e.g.
        # shoreline of tongatapu
        # maybe only erode areas not in consensus land?
        # This works for tongatapu but not funafuti
        # analysis_zone = mask_cleanup(connected_areas,
        #                           mask_filters=[("erosion", 2), ("dilation",2)])

        # Basically, NIR says it's land but none of the others say there is
        # even neighboring land
        # This _could_ eliminate some areas, we will have to see. If that
        # happens we can consider
        # making a larger kernel. OR, we could use the surf detector from the
        # mask model,
        # if that's the only class that causes issue.
        # I did this because of bands of surf off the south side of Tongatapu
        # that were connected
        # in a single place to the mainland.
        no_connected_neighbors = xs.focal.mean(consensus_land) == 0
        suspicious_connected_areas = candidate_land & no_connected_neighbors
        analysis_zone = connected_areas & ~suspicious_connected_areas
        analysis_zone = self.expand_analysis_zone(analysis_zone, output)

        gadm_land = load_gadm_land(output)
        # consensus land may have inland water, but gadm doesn't.
        # Also, consensus land will have masked areas as False rather
        # than nan. Neither of these should matter because gadm doesn't have
        # these issues. I bring in conensus land basically to fix the areas
        # near shoreline that gadm may miss.
        land = gadm_land | consensus_land
        ocean = mask_cleanup(~land, mask_filters=[("erosion", 2)])
        inland_areas = find_inland_areas(self.water(output), ocean)
        output = (
            output[self.water_index]
            .where(analysis_zone, obvious_water)
            .where(~inland_areas)
        )
        output = output.groupby("year").map(xs.focal.mean)
        combined_gdf = subpixel_contours(
            output,
            dim="year",
            z_values=[self.index_threshold],
            min_vertices=5,
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

    loader = MultiyearMosaicLoader(
        start_year=start_year,
        end_year=end_year,
        years_per_composite=[1, 3],
        version="0.6.0.3",
    )
    processor = Cleaner()
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
    version: Annotated[str, Option()],
):
    with Client():
        run((int(row), int(column)), version=version)


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
