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

import boto3
from coastlines.vector import (
    all_time_stats,
    annual_movements,
    calculate_regressions,
    contour_certainty,
    points_on_line,
    region_atttributes,
)
from dask.distributed import Client
from dea_tools.spatial import subpixel_contours
from geopandas import GeoDataFrame
import geohash
from numpy import isfinite
from odc.algo import mask_cleanup
from odc.stac import configure_s3_access
from pyogrio import read_dataframe
from typer import Option, run
from xarray import DataArray, Dataset
import xrspatial as xs

from dep_tools.processors import Processor
from dep_tools.task import (
    ErrorCategoryAreaTask,
    AwsStacTask,
    EmptyCollectionError,
    NoOutputError,
)

from dep_coastlines.common import coastlineItemPath, coastlineLogger
from dep_coastlines.config import MOSAIC_VERSION
from dep_coastlines.cloud_model.fit_model import SavedModel  # needed for load
from dep_coastlines.cloud_model.predictor import ModelPredictor
from dep_coastlines.io import CoastlineWriter, MultiyearMosaicLoader
from dep_coastlines.raster_cleaning import (
    load_gadm_land,
    find_inland_areas,
    fill_with_nearby_dates,
    remove_disconnected_land,
    smooth_gaussian,
)
from dep_coastlines.grid import buffered_grid as GRID


DATASET_ID = "coastlines/interim/coastlines"

BooleanDataArray = DataArray


def calculate_consensus_land(ds: Dataset) -> BooleanDataArray:
    """Returns true for areas for which the all-years medians of mndwi,
    ndwi and nirwi are less than zero. (nirwi
    is negative where the nir08 band is greater than 0.128.)"""
    return (ds.nirwi_all < 0) & (ds.mndwi_all < 0) & (ds.ndwi_all < 0)


class Cleaner(Processor):
    def __init__(
        self,
        water_index: str = "twndwi",
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
        # gpd.read_file got a 403
        eez = read_dataframe(
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
        points_gdf = calculate_regressions(points_gdf)  # , contours)

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
            .map(smooth_gaussian)
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

        certainty_masks = certainty_masking(
            output.rename({variation_var: "stdev"}), stdev_threshold=0.3
        )
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


from dea_tools.spatial import subpixel_contours, xr_vectorize
import numpy as np
from rasterio.features import sieve
from skimage.morphology import (
    dilation,
    disk,
)
import xarray as xr


# The following 2 functions copied from dea coastlines solely for the
# .squeeze to accomodate xarray updates
def _create_mask(raster_mask, sieve_size, crs):
    """
    Clean and dilate an annual raster produced by `certainty_masking`,
    then vectorize into a dictionary of vector features that are
    taken as an input by `contour_certainty`.
    """

    # Clean mask by sieving to merge small areas of pixels into
    # their neighbours.
    sieved = xr.apply_ufunc(sieve, raster_mask, sieve_size)

    # Apply greyscale dilation to expand masked pixels and
    # err on the side of overclassifying certainty issues
    dilated = xr.apply_ufunc(dilation, sieved, disk(3))

    # Vectorise
    vector_mask = xr_vectorize(
        dilated,
        crs=crs,
        attribute_col="certainty",
    )

    # Dissolve column, fix geometry and rename classes
    vector_mask = vector_mask.dissolve("certainty")
    vector_mask["geometry"] = vector_mask.geometry.buffer(0)
    vector_mask = vector_mask.rename(
        {0: "good", 1: "unstable data", 2: "insufficient data"}
    )

    return (raster_mask.year.item(), vector_mask)


def certainty_masking(yearly_ds, obs_threshold=5, stdev_threshold=0.3, sieve_size=128):
    """
    Generate annual vector polygon masks containing information
    about the certainty of each extracted shoreline feature.
    These masks are used to assign each shoreline feature with
    important certainty information to flag potential issues with
    the data.

    Parameters:
    -----------
    yearly_ds : xarray.Dataset
        An `xarray.Dataset` containing annual DEA Coastlines
        rasters.
    obs_threshold : int, optional
        The minimum number of post-gapfilling Landsat observations
        required for an extracted shoreline to be considered good
        quality. Annual shorelines based on low numbers of
        observations can be noisy due to the influence of
        environmental noise like unmasked cloud, sea spray, white
        water etc. Defaults to 5.
    stdev_threshold : float, optional
        The maximum MNDWI standard deviation required for a
        post-gapfilled Landsat observation to be considered good
        quality. Annual shorelines based on MNDWI with a high
        standard deviation represent unstable data, which can
        indicate that the tidal modelling process did not adequately
        remove the influence of tide. For more information,
        refer to Bishop-Taylor et al. 2021
        (https://doi.org/10.1016/j.rse.2021.112734).
        Defaults to 0.3.
    sieve_size : int, optional
        To reduce the complexity of the output masks, they are
        first cleaned using `rasterio.features.sieve` to replace
        small areas of pixels with the values of their larger
        neighbours. This parameter sets the minimum polygon size
        to retain in this process. Defaults to 128.

    Returns:
    --------
    vector_masks : dictionary of geopandas.GeoDataFrames
        A dictionary with year (as an str) as the key, and vector
        data as a `geopandas.GeoDataFrame` for each year in the
        analysis.
    """

    from concurrent.futures import ProcessPoolExecutor
    from itertools import repeat

    # Identify problematic pixels
    high_stdev = yearly_ds["stdev"] > stdev_threshold
    low_obs = yearly_ds["count"] < obs_threshold

    # Create raster mask with values of 0 for good data, values of
    # 1 for unstable data, and values of 2 for insufficient data.
    raster_mask = high_stdev.where(~low_obs, 2).astype(np.int16)

    # Process in parallel
    with ProcessPoolExecutor() as executor:
        # Apply func in parallel, repeating params for each iteration
        groups = [group.squeeze() for (i, group) in raster_mask.groupby("year")]
        to_iterate = (
            groups,
            *(repeat(i, len(groups)) for i in [sieve_size, yearly_ds.odc.crs]),
        )
        vector_masks = dict(executor.map(_create_mask, *to_iterate), total=len(groups))

    return vector_masks


def process_id(
    task_id: Tuple | list[Tuple] | None,
    dataset_id=DATASET_ID,
    version: str = "0.8.0",
    water_index="twndwi",
) -> None:
    start_year = 1999
    end_year = 2023
    namer = coastlineItemPath(dataset_id, version, time=f"{start_year}_{end_year}")
    logger = coastlineLogger(namer, dataset_id=dataset_id)

    loader = MultiyearMosaicLoader(
        start_year=start_year,
        end_year=end_year,
        years_per_composite=[1, 3],
        version=MOSAIC_VERSION,
    )
    processor = Cleaner(water_index=water_index, send_area_to_processor=True)
    writer = CoastlineWriter(
        namer,
        extra_attrs=dict(dep_version=version),
    )

    ErrorCategoryAreaTask(
        task_id, GRID.loc[[task_id]], loader, processor, writer, logger
    ).run()


def main(
    row: Annotated[str, Option()],
    column: Annotated[str, Option()],
    version: Annotated[str, Option()],
    water_index: str = "twndwi",
):
    configure_s3_access(cloud_defaults=True, requester_pays=True)
    boto3.setup_default_session()
    with Client():
        process_id((int(column), int(row)), version=version, water_index=water_index)


if __name__ == "__main__":
    run(main)
