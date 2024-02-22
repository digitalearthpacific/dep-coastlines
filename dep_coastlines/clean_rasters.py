"""
This is a work-in-progress script to "post-process" water index data before
vectorizing. The functionality here is part of the vectorization code from the
DEA and DEAfrica work, but it is primarily raster based, hence the name.
The actual vectorization (using subpixel_contours) may or may not belong here, 
so things may move around / get renamed some in the coming weeks.

Please refer to raster_cleaning.py for specific functions.
"""

from distributed import connect
from joblib import load
from typing import Tuple, Annotated

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

from MosaicLoader import MultiyearMosaicLoader
from raster_cleaning import contours_preprocess, load_gadm_land, find_inland_areas
from grid import test_grid
from task_utils import get_ids
from writer import CoastlineWriter


app = Typer()
DATASET_ID = "coastlines/coastlines"


def remove_disconnected_land(certain_land, candidate_land):
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


def mask_noisy_water(xr, mask, water_index):
    mean_water_value = dict(mndwi=0.415616, ndwi=0.788478)
    NOISY_WATER_CODE = 7
    return xr.where(mask != NOISY_WATER_CODE, mean_water_value[water_index])


def mask_surf(xr, mask, filler):
    SURF_CODE = 4
    return xr.where(mask != SURF_CODE, filler)


def mask_bare_terrain(xr, mask):
    BARE_TERRAIN_CODE = 8
    return xr.where(mask != BARE_TERRAIN_CODE)


class Cleaner(Processor):
    def __init__(
        self,
        water_index: str = "nir08",
        index_threshold: float = -1280.0,
    ):
        super().__init__()
        self.index_threshold = index_threshold
        self.water_index = water_index
        model_obj = load("data/model_20Feb.joblib")
        self.mask_model = model_obj["model"]
        self.training_columns = model_obj["predictor_columns"]
        self.response_column = model_obj["response_column"]
        codes = model_obj["codes"]
        self.model_codes = codes.groupby(self.response_column).first()

    def code_for_name(self, name):
        return (
            self.model_codes.reset_index()
            .set_index("code")
            .loc[name, self.response_column]
        )

    def _calculate_mask(self, input):
        masks = []
        for year in input.year:
            year_mask = predict_xr(
                self.mask_model, input.sel(year=year)[self.training_columns], clean=True
            ).Predictions
            year_mask.coords["year"] = year
            masks.append(year_mask)
        return concat(masks, dim="year")

    def process(self, input: Dataset | list[Dataset]) -> Tuple[Dataset, GeoDataFrame]:
        # masks = []
        CLOUD_CODE = self.code_for_name("cloud")
        if isinstance(input, list):
            this_mask = self._calculate_mask(input[0])
            # masks.append(this_mask)
            #            ultimate_mask = this_mask
            output = input[0].where(this_mask != CLOUD_CODE)
            for ds in input[1:]:
                ds = ds.sel(year=ds.year[ds.year.isin(output.year)])
                this_mask = self._calculate_mask(ds)
                # masks.append(this_mask)
                missings = output.isnull()
                output = output.where(~missings, ds.where(this_mask != CLOUD_CODE))
        #                ultimate_mask = ultimate_mask.where(~missings.nir08, this_mask)
        else:
            mask = self._calculate_mask(input)
            # masks.append(mask)
            output = input.where(mask != CLOUD_CODE)
        #            ultimate_mask = mask

        all_time = output.median(dim="year")
        consensus = (
            (output.nir08_all > 1280.0) & (output.mndwi_all < 0) & (output.ndwi_all < 0)
        )
        output = output[["nir08", "mndwi", "ndwi", "meanwi", "nirwi"]]

        # Connected areas are contiguous zones that are connected in some way to
        # the consensus areas. This ensures that all edges of these (on the basis of nir)
        # are included
        import operator

        band = "meanwi"
        cutoff = 0
        obvious_water = 0.5
        comparison = operator.lt

        def land(output):
            return comparison(output[band], cutoff)

        def water(output):
            return ~land(output)

        output[band] = fill_with_nearest_later_date(output[band])
        candidate = land(output)
        connected_areas = candidate.groupby("year").map(
            lambda candidate_year: remove_disconnected_land(consensus, candidate_year)
        )
        # erosion of 2 here borks e.g. funafuti but is needed for e.g. shoreline of tongatapu
        # maybe only erode areas not in consensus land?
        # This works for tongatapu but not funafuti
        # analysis_zone = mask_cleanup(connected_areas, mask_filters=[("erosion", 2), ("dilation",2)])

        # Basically, NIR says it's land but none of the others say there is even neighboring land
        # This _could_ eliminate some areas, we will have to see. If that happens we can consider
        # making a larger kernel. OR, we could use the surf detector from the mask model,
        # if that's the only class that causes issue.
        # I did this because of bands of surf off the south side of Tongatapu that were connected
        # in a single place to the mainland.
        no_connected_neighbors = xs.focal.mean(consensus) == 0
        # suspicious_connected_areas = candidate & no_connected_neighbors
        analysis_zone = connected_areas & ~no_connected_neighbors
        # Only expand where there's an edge that's land. Do it multiple times to fill between larger areas
        # say in Funafuti or Majiro. Later we will fill one last time with water to ensure lines are closed.
        number_of_expansions = 8
        for _ in range(number_of_expansions):
            analysis_zone = analysis_zone | mask_cleanup(
                land(output.where(analysis_zone)),
                mask_filters=[("dilation", 1)],
            )

        gadm_land = load_gadm_land(output)
        gadm_ocean = mask_cleanup(~gadm_land, mask_filters=[("erosion", 2)])
        inland_areas = find_inland_areas(water(output), gadm_ocean)
        output = output[band].where(analysis_zone).where(~inland_areas)
        combined_gdf = subpixel_contours(
            output,
            dim="year",
            z_values=[cutoff],
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
        start_year=start_year, end_year=end_year, years_per_composite=[1, 3]
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
