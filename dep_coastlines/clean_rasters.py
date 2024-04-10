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
    points_on_line,
    annual_movements,
    calculate_regressions,
    contour_certainty,
    region_atttributes,
)
from dea_tools.classification import predict_xr
from dea_tools.spatial import subpixel_contours
from geopandas import GeoDataFrame, read_file
import numpy as np
from odc.algo import mask_cleanup
from typer import Option, Typer
from xarray import DataArray, Dataset, concat, apply_ufunc
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
from dep_coastlines.vector import certainty_masking
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


def fill_nearby(xr):
    def fill(da):
        output = da.to_dataset("year")
        for year in da.year.values:
            output[year] = da.sel(year=year)
            intyear = int(year)
            years = [
                str(y)
                for y in [intyear + 1, intyear - 1, intyear + 2, intyear - 2]
                if str(y) in da.year.values
            ]
            for inner_year in years:
                output[year] = output[year].where(
                    ~output[year].isnull(), output[inner_year]
                )
        return output.to_array(dim="year")

    return xr.apply(fill) if isinstance(xr, Dataset) else fill(xr)


def fill_with_nearest_later_date(xr):
    def fill_da(da):
        output = da.to_dataset("year")
        for year in da.year.values:
            output[year] = da.sel(year=year)
            for inner_year in da.year.values:
                if int(inner_year) <= int(year):
                    continue
                output[year] = output[year].where(
                    ~output[year].isnull(), output[inner_year]
                )
        return output.to_array(dim="year")

    return xr.apply(fill_da) if isinstance(xr, Dataset) else fill_da(xr)


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
        cloud_code = self.code_for_name("cloud")
        if isinstance(input, list):
            this_mask = self.calculate_mask(input[0])
            output = input[0].where(this_mask != cloud_code, drop=False)
            for ds in input[1:]:
                ds = ds.sel(year=ds.year[ds.year.isin(output.year)])
                this_mask = self.calculate_mask(ds)
                missings = (output.isnull()) | (output["count"] <= 4)
                output = output.where(
                    ~missings, ds.where(this_mask != cloud_code, drop=False)
                )
        else:
            mask = self.calculate_mask(input)
            output = input.where(mask != cloud_code)
        return output


def calculate_consensus_land(ds):
    # return (ds.nir08_all > 1280.0) & (ds.mndwi_all < 0) & (ds.ndwi_all < 0)
    return (ds.nirwi_all < 0) & (ds.mndwi_all < 0) & (ds.ndwi_all < 0)
    # return (ds.nir08 > 1280.0) & (ds.mndwi < 0) & (ds.ndwi < 0)


def convolve(da):
    # This is a convolution with a gaussian kernel that ignores NAs.
    weights = DataArray([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dims=("xw", "yw"))
    total = (
        da.fillna(0)
        .rolling(x=3, y=3, center=True)
        .construct(x="xw", y="yw")
        .dot(weights)
    )
    divisor = (
        (~da.isnull())
        .astype(int)
        .rolling(x=3, y=3, center=True)
        .construct(x="xw", y="yw")
        .dot(weights)
    )
    return (total / divisor).where(~da.isnull())


class Cleaner(Processor):
    def __init__(
        self,
        water_index: str = "meanwi",
        index_threshold: float = 0,
        comparison: Callable = operator.lt,
        number_of_expansions: int = 4,
        baseline_year: str = "2023",
        model_file=Path(__file__).parent / "full_model_9Apr2024.joblib",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index_threshold = index_threshold
        self.water_index = water_index
        self.baseline_year = baseline_year

        self.model = ModelPredictor(load(model_file))
        self.comparison = comparison
        self.number_of_expansions = number_of_expansions

    def land(self, output):
        return self.comparison(output[self.water_index], self.index_threshold)

    def water(self, output):
        return output[self.water_index] >= self.index_threshold

    def expand_analysis_zone(self, analysis_zone, output, return_max_cap: bool = False):
        # Only expand where there's an edge that's land. Do it multiple times
        # to fill between larger areas
        # say in Funafuti or Majiro. Later we will fill one last time with
        # water to ensure lines are closed.
        def expand_once(analysis_zone):
            return analysis_zone | mask_cleanup(
                self.land(output.where(analysis_zone)),
                mask_filters=[("dilation", 1)],
            )

        for _ in range(self.number_of_expansions):
            analysis_zone = expand_once(analysis_zone)

        if return_max_cap:
            last_expansion = expand_once(analysis_zone)
            max_cap = last_expansion & ~analysis_zone
            return analysis_zone, max_cap

        return analysis_zone

    def points(self, contours, water_index):
        water_index["year"] = water_index.year.astype(int)
        contours.year = contours.year.astype(str)
        contours = contours.set_index("year")

        points_gdf = points_on_line(contours, self.baseline_year, distance=30)
        if points_gdf is not None and len(points_gdf) > 0:
            points_gdf = annual_movements(
                points_gdf,
                contours,
                water_index.to_dataset(name=self.water_index),
                self.baseline_year,
                self.water_index,
                max_valid_dist=5000,
            )
        points_gdf = calculate_regressions(points_gdf, contours)

        stats_list = ["valid_obs", "valid_span", "sce", "nsm", "max_year", "min_year"]
        points_gdf[stats_list] = points_gdf.apply(
            lambda x: all_time_stats(x, initial_year=1999), axis=1
        )
        contours = contours.reset_index()
        contours.year = contours.year.astype(int)
        return points_gdf

    def process(
        self, input: Dataset | list[Dataset], area
    ) -> Tuple[Dataset, GeoDataFrame, GeoDataFrame | None]:
        output = self.model.apply_mask(input)
        output = output.where(output["count"] > 4)
        output = fill_nearby(output)
        variation_var = self.water_index + "_mad"
        variables_to_keep = [self.water_index, variation_var, "count"]

        output = output[variables_to_keep].compute()

        candidate_land = self.land(output)
        an_input = input[0] if isinstance(input, list) else input
        consensus_land = calculate_consensus_land(an_input.isel(year=0)).compute()
        # Connected are contiguous zones that are connected in some way to
        # the consensus areas. This ensures that all edges of these are included
        connected_areas = remove_disconnected_land(consensus_land, candidate_land)
        no_connected_neighbors = xs.focal.mean(consensus_land) == 0
        suspicious_connected_areas = candidate_land & no_connected_neighbors
        analysis_zone = connected_areas & ~suspicious_connected_areas
        analysis_zone, max_cap = self.expand_analysis_zone(analysis_zone, output, True)
        obvious_water = 0.5

        gadm_land = load_gadm_land(output)
        # consensus land may have inland water, but gadm doesn't.
        # Also, consensus land will have masked areas as False rather
        # than nan. Neither of these should matter because gadm doesn't have
        # these issues. I bring in consensus land basically to fix the areas
        # near shoreline that gadm may miss.
        land = gadm_land | consensus_land
        ocean = mask_cleanup(~land, mask_filters=[("erosion", 2)])
        inland_areas = find_inland_areas(self.water(output), ocean)

        water_index = (
            output[self.water_index]
            .where(analysis_zone | land, obvious_water)
            .where(~inland_areas)
            .groupby("year")
            .map(convolve)
            .rio.write_crs(output.rio.crs)
        )

        coastlines = subpixel_contours(
            water_index,
            dim="year",
            z_values=[self.index_threshold],
            min_vertices=5,
        )

        certainty_masks = certainty_masking(output, variation_var)
        coastlines = contour_certainty(
            coastlines.set_index("year"), certainty_masks
        ).reset_index()  # .set_index("year")

        # Taking this out for now as these data are not open
        # eez = read_file("data/src/global_ffa_spc_sla_pol_-180-180_mar2023.zip").to_crs( water_index.rio.crs)
        # these_areas = eez  # .clip(coastlines.to_crs(4326).total_bounds).to_crs( water_index.rio.crs)
        # these_areas.geometry = these_areas.geometry.buffer(250)
        roc_points = self.points(coastlines, water_index)
        #        coastlines = region_atttributes(
        #            coastlines.set_index("year"),
        #            these_areas,
        #            attribute_col="TERRITORY1",
        #            rename_col="eez_territory",
        #        )
        #        roc_points = region_atttributes(
        #            roc_points,
        #            these_areas,
        #            attribute_col="TERRITORY1",
        #            rename_col="eez_territory",
        #        )

        water_index["year"] = water_index.year.astype(str)
        return (
            water_index.to_dataset("year"),  # .rio.clip(area_proj.geometry),
            coastlines,
            roc_points,
        )


def run(
    task_id: Tuple | list[Tuple] | None,
    dataset_id=DATASET_ID,
    version: str = "0.6.0",
    water_index="meanwi",
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
        version="0.7.0.1",
    )
    processor = Cleaner(water_index=water_index, send_area_to_processor=True)
    writer = CoastlineWriter(
        namer,
        extra_attrs=dict(dep_version=version),
        #        writer=write_to_local_storage,
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
    water_index: str = "meanwi",
):
    with Client():
        run((int(row), int(column)), version=version, water_index=water_index)


@app.command()
def process_all_ids(
    version: Annotated[str, Option()],
    overwrite_log: Annotated[bool, Option()] = False,
    water_index: str = "meanwi",
    dataset_id=DATASET_ID,
    datetime="1999/2023",
):
    task_ids = get_ids(
        datetime, version, dataset_id, grid=test_grid, delete_existing_log=overwrite_log
    )

    with Client():
        run(task_ids, dataset_id=dataset_id, version=version, water_index=water_index)


if __name__ == "__main__":
    app()
