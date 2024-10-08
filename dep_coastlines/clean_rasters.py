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
from coastlines.vector import (
    all_time_stats,
    annual_movements,
    calculate_regressions,
    certainty_masking,
    contour_certainty,
    points_on_line,
    region_atttributes,
)
from dea_tools.spatial import subpixel_contours
from geopandas import GeoDataFrame, read_file
import geohash
from numpy import isfinite
from odc.algo import mask_cleanup
from retry import retry
from typer import Option, Typer
from xarray import DataArray, Dataset
import xrspatial as xs

from azure_logger import CsvLogger
from dep_tools.namers import DepItemPath
from dep_tools.processors import Processor
from dep_tools.task import (
    EmptyCollectionError,
    MultiAreaTask,
    ErrorCategoryAreaTask,
    NoOutputError,
)
from dep_tools.utils import get_container_client

from dep_coastlines.io import CoastlineWriter, MultiyearMosaicLoader
from dep_coastlines.cloud_model.predictor import ModelPredictor
from dep_coastlines.cloud_model.fit_model import SavedModel
from dep_coastlines.raster_cleaning import (
    load_gadm_land,
    find_inland_areas,
    fill_with_nearby_dates,
    remove_disconnected_land,
    smooth_gaussian,
)
from dep_coastlines.grid import buffered_grid as GRID
from dep_coastlines.task_utils import get_ids


app = Typer()
DATASET_ID = "coastlines/coastlines"

BooleanDataArray = DataArray


def calculate_consensus_land(ds: Dataset) -> BooleanDataArray:
    """Returns true for areas for which the all-years medians of mndwi,
    ndwi and nirwi are less than zero. (nirwi
    is negative where the nir08 band is greater than 0.128.)"""
    return (ds.nirwi_all < 0) & (ds.mndwi_all < 0) & (ds.ndwi_all < 0)


class Cleaner(Processor):
    def __init__(
        self,
        water_index: str = "meanwi",
        index_threshold: float = 0,
        comparison: Callable = operator.lt,
        number_of_expansions: int = 8,
        baseline_year: str = "2023",
        model_file=Path(__file__).parent / "cloud_model/full_model_0-7-0-4.joblib",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index_threshold = index_threshold
        self.water_index_name = water_index
        self.baseline_year = baseline_year

        self.model = ModelPredictor(load(model_file))
        self.comparison = comparison
        self.number_of_expansions = number_of_expansions

    def land(self, output):
        return self.comparison(output[self.water_index_name], self.index_threshold)

    def water(self, output):
        return output[self.water_index_name] >= self.index_threshold

    def expand_analysis_zone(self, analysis_zone, output, return_max_cap: bool = False):
        # Only expand where there's an edge that's land. Do it multiple times
        # to fill between larger areas. Later we will fill one last time with
        # water to ensure lines are closed.
        def expand_once(analysis_zone):
            return analysis_zone | mask_cleanup(
                self.land(output.where(analysis_zone)),
                mask_filters=[("dilation", 2)],
            )

        for _ in range(self.number_of_expansions):
            analysis_zone = expand_once(analysis_zone)

        if return_max_cap:
            last_expansion = expand_once(analysis_zone)
            max_cap = last_expansion & ~analysis_zone
            return analysis_zone, max_cap

        return analysis_zone

    def add_attributes(self):
        eez = read_file(
            "https://pacificdata.org/data/dataset/964dbebf-2f42-414e-bf99-dd7125eedb16/resource/dad3f7b2-a8aa-4584-8bca-a77e16a391fe/download/country_boundary_eez.geojson"
        )
        these_areas = (
            eez.to_crs(3832)
            .clip(self.coastlines.buffer(250).to_crs(3832).total_bounds)
            .to_crs(self.water_index.rio.crs)
        )
        self.coastlines = region_atttributes(
            self.coastlines.set_index("year"),
            these_areas,
            attribute_col="ISO_Ter1",
            rename_col="eez_territory",
        )
        self.roc_points = region_atttributes(
            self.roc_points,
            these_areas,
            attribute_col="ISO_Ter1",
            rename_col="eez_territory",
        )

    def points(self, contours, water_index):
        water_index["year"] = water_index.year.astype(int)
        contours.year = contours.year.astype(str)
        contours = contours.set_index("year")

        baseline_year = contours.index.astype(int).max().astype(str)
        points_gdf = points_on_line(contours, baseline_year, distance=30)
        if points_gdf is not None and len(points_gdf) > 0:
            points_gdf = annual_movements(
                points_gdf,
                contours,
                water_index.to_dataset(name=self.water_index_name),
                self.baseline_year,
                self.water_index_name,
                max_valid_dist=5000,
            )
        points_gdf = calculate_regressions(points_gdf, contours)

        stats_list = ["valid_obs", "valid_span", "sce", "nsm", "max_year", "min_year"]
        points_gdf[stats_list] = points_gdf.apply(
            lambda x: all_time_stats(x, initial_year=1999), axis=1
        )

        points_gdf["certainty"] = "good"
        # for now, leaving out "offshore_islands" category, as most of the study
        # area would be in there. However, consider (if available) breakdowns
        # like "atolls", "volcanic", etc

        # These would need to be mapped.
        # points_gdf.loc[
        #    rocky_shoreline_flag(points_gdf, geomorphology_gdf), "certainty"
        # ] = "likely rocky coastline"

        # Other things to consider:
        # tidal flats
        # reefs

        points_gdf.loc[points_gdf.rate_time.abs() > 200, "certainty"] = (
            "extreme value (> 200 m)"
        )

        points_gdf.loc[points_gdf.angle_std > 30, "certainty"] = (
            "high angular variability"
        )
        points_gdf.loc[points_gdf.valid_obs < 15, "certainty"] = (
            "insufficient observations"
        )

        # Generate a geohash UID for each point and set as index
        uids = (
            points_gdf.geometry.to_crs("EPSG:4326")
            .apply(lambda x: geohash.encode(x.y, x.x, precision=10))
            .rename("uid")
        )
        points_gdf = points_gdf.set_index(uids)

        contours = contours.reset_index()
        contours.year = contours.year.astype(int)
        return points_gdf

    def process(
        self, input: Dataset | list[Dataset], area
    ) -> Tuple[Dataset, GeoDataFrame, GeoDataFrame | None]:
        output, mask = self.model.apply_mask(input)
        output = fill_with_nearby_dates(output)

        variation_var = self.water_index_name + "_mad"
        variables_to_keep = [self.water_index_name, variation_var, "count"]
        output = output[variables_to_keep].compute()

        an_input = input[0] if isinstance(input, list) else input
        consensus_land = calculate_consensus_land(an_input.isel(year=0)).compute()

        candidate_land = self.land(output)
        # Connected are contiguous zones that are connected in some way to
        # the consensus areas. This ensures that all edges of these are included
        connected_areas = remove_disconnected_land(consensus_land, candidate_land)
        no_connected_neighbors = xs.focal.mean(consensus_land) == 0
        suspicious_connected_areas = candidate_land & no_connected_neighbors
        # So we don't expand disconnected land
        analysis_zone = connected_areas & ~suspicious_connected_areas

        analysis_zone, max_cap = self.expand_analysis_zone(analysis_zone, output, True)
        # To remove disconnected land we expanded into
        # land must be expanded for algo to work
        # this needs to return connected land and water
        candidate_land = self.land(output.where(analysis_zone))
        connected_areas = remove_disconnected_land(consensus_land, candidate_land)
        # don't remove suspicous areas because expanded land may not be within
        # 1 cell of consensus land
        disconnected_areas = self.land(output.where(analysis_zone)) & ~connected_areas
        analysis_zone = analysis_zone & ~disconnected_areas
        obvious_water = 0.5

        gadm_land = load_gadm_land(output)
        # consensus land may have inland water, but gadm doesn't.
        # Also, consensus land will have masked areas as False rather
        # than nan. Neither of these should matter because gadm doesn't have
        # these issues. I bring in consensus land basically to fix the areas
        # near shoreline that gadm may miss.
        land = gadm_land | consensus_land
        ocean = mask_cleanup(~land, mask_filters=[("erosion", 2)])

        # basically to capture land outside the buffer that would otherwise
        # link inland water to perceived ocean
        # The amount here is linked to the buffer value in the grid
        core_land = mask_cleanup(gadm_land, mask_filters=[("erosion", 60)])

        self.water_index = (
            output[self.water_index_name]
            .where(analysis_zone | core_land)
            .where(~max_cap, obvious_water)
            .groupby("year")
            .map(smooth_gaussian, sigma=0.4)
            .rio.write_crs(output.rio.crs)
        )

        # All this logic is to ensure areas on the landward side of the analysis
        # buffer aren't coded as water
        obvious_land = -0.5
        water = ~self.land(
            self.water_index.where(
                ~(self.water_index.isnull() & (core_land | consensus_land)),
                obvious_land,
            ).to_dataset(name=self.water_index_name)
        )

        inland_water = find_inland_areas(water, ocean)
        water_index = self.water_index.where(~inland_water).where(
            lambda wi: isfinite(wi)
        )

        self.coastlines = subpixel_contours(
            water_index,
            dim="year",
            z_values=[self.index_threshold],
            min_vertices=5,
        )

        if len(self.coastlines) == 0:
            raise NoOutputError("no coastlines created; water index may be empty")

        certainty_masks = certainty_masking(output, variation_var)
        self.coastlines = contour_certainty(
            self.coastlines.set_index("year"), certainty_masks
        ).reset_index()
        self.roc_points = self.points(self.coastlines, output[self.water_index_name])

        self.add_attributes()

        water_index["year"] = water_index.year.astype(str)
        return (
            water_index.to_dataset("year"),
            mask,
            self.coastlines,
            self.roc_points,
        )


def run(
    task_id: Tuple | list[Tuple] | None,
    dataset_id=DATASET_ID,
    version: str = "0.7.0",
    water_index="twndwi",
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
        version="0.7.0.4",
    )
    processor = Cleaner(water_index=water_index, send_area_to_processor=True)
    writer = CoastlineWriter(
        namer,
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
            GRID,
            ErrorCategoryAreaTask,
            loader,
            processor,
            writer,
            logger,
        ).run()
    else:
        ErrorCategoryAreaTask(
            task_id, GRID.loc[[task_id]], loader, processor, writer, logger
        ).run()


@app.command()
# @retry(tries=3)
def process_id(
    row: Annotated[str, Option()],
    column: Annotated[str, Option()],
    version: Annotated[str, Option()],
    water_index: str = "twndwi",
):
    with Client():
        try:
            run((int(row), int(column)), version=version, water_index=water_index)
        except (NoOutputError, EmptyCollectionError):
            pass


@app.command()
def process_all_ids(
    version: Annotated[str, Option()],
    overwrite_log: Annotated[bool, Option()] = False,
    water_index: str = "twndwi",
    dataset_id=DATASET_ID,
    datetime="1999/2023",
):
    task_ids = get_ids(
        datetime, version, dataset_id, grid=GRID, delete_existing_log=overwrite_log
    )

    with Client():
        run(task_ids, dataset_id=dataset_id, version=version, water_index=water_index)


if __name__ == "__main__":
    app()
