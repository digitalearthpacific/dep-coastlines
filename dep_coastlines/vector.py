"""
This is a work-in-progress script to "post-process" water index data before
vectorizing. The functionality here is part of the vectorization code from the
DEA and DEAfrica work, but it is primarily raster based, hence the name.
The actual vectorization (using subpixel_contours) may or may not belong here,
so things may move around / get renamed some in the coming weeks.

Please refer to raster_cleaning.py for specific functions.
"""

from typing import Tuple

import geohash
import geopandas as gpd
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
from dep_tools.processors import Processor
from dep_tools.task import (
    NoOutputError,
)
from geopandas import GeoDataFrame
from joblib import load
from numpy import isfinite
from odc.algo import mask_cleanup
from pyogrio import read_dataframe
from xarray import Dataset

from dep_coastlines.cloud_model.predictor import ModelPredictor
from dep_coastlines.config import CLOUD_MODEL_FILE
from dep_coastlines.raster.cleaning import (
    fill_with_nearby_dates,
    load_gadm_land,
    remove_disconnected_areas,
    smooth_gaussian,
)


def calculate_roc_stats(ratesofchange_gdf, initial_year, minimum_valid_observations=15):
    stats_list = ["valid_obs", "valid_span", "sce", "nsm", "max_year", "min_year"]
    ratesofchange_gdf[stats_list] = ratesofchange_gdf.apply(
        lambda x: all_time_stats(x, initial_year=initial_year), axis=1
    )

    ratesofchange_gdf["certainty"] = "good"
    # for now, leaving out "offshore_islands" category, as most of the study
    # area would be in there. However, consider (if available) breakdowns
    # like "atolls", "volcanic", etc

    # These would need to be mapped.
    # ratesofchange_gdf.loc[
    #    rocky_shoreline_flag(ratesofchange_gdf, geomorphology_gdf), "certainty"
    # ] = "likely rocky coastline"

    # Other things to consider:
    # tidal flats
    # reefs

    ratesofchange_gdf.loc[ratesofchange_gdf.rate_time.abs() > 200, "certainty"] = (
        "extreme value (> 200 m)"
    )

    ratesofchange_gdf.loc[ratesofchange_gdf.angle_std > 30, "certainty"] = (
        "high angular variability"
    )
    ratesofchange_gdf.loc[
        ratesofchange_gdf.valid_obs < minimum_valid_observations, "certainty"
    ] = "insufficient observations"
    return ratesofchange_gdf


# def calculate_consensus_land(ds: Dataset) -> DataArray:
#     """Returns true for areas for which the all-years medians of mndwi,
#     ndwi and nirwi are less than zero. (nirwi
#     is negative where the nir08 band is greater than 0.128.)"""
#     return ds.twndwi_all < 0
#     # return (ds.nirwi_all < 0) & (ds.mndwi_all < 0) & (ds.ndwi_all < 0)


def calculate_rates_of_change(
    contours, water_index, water_index_name, baseline_year, initial_year
):
    """Calculate rates of change points from coastline contours and water index."""
    water_index["year"] = water_index.year.astype(int)
    contours.year = contours.year.astype(str)
    contours = contours.set_index("year")

    baseline_year = contours.index.astype(int).max().astype(str)
    # Define points at 30-m intervals
    points_gdf = points_on_line(contours, baseline_year, distance=30)
    if points_gdf is not None and len(points_gdf) > 0:
        points_gdf = annual_movements(
            points_gdf,
            contours,
            water_index.to_dataset(name=water_index_name),
            baseline_year,
            water_index_name,
            max_valid_dist=5000,
        )
    points_gdf = calculate_regressions(points_gdf)
    points_gdf = calculate_roc_stats(points_gdf, initial_year)

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


def expand_analysis_zone(
    analysis_zone,
    water_index,
    number_of_expansions: int = 64,
    return_max_cap: bool = False,
):
    # Only expand where there's an edge that's land. Do it multiple times
    # to fill between larger areas. Later we will fill one last time with
    # water to ensure lines are closed.
    # We do 2 because the last expansion needs 2 to fill in corners.
    def expand_once(analysis_zone):
        return analysis_zone | mask_cleanup(
            water_index.where(analysis_zone) < 0,
            mask_filters=[("dilation", 2)],
        )

    for _ in range(number_of_expansions):
        analysis_zone = expand_once(analysis_zone)

    if return_max_cap:
        last_expansion = expand_once(analysis_zone)
        max_cap = last_expansion & ~analysis_zone
        return analysis_zone, max_cap

    return analysis_zone


def add_attributes(coastlines, roc_points):
    """Add other attributes to coastlines and rates of change points.

    Currently the only attribute added is the three-letter country code
    for the economic exclusion zone within which the feature falls.
    """
    eez = gpd.read_file(
        "https://dep-public-staging.s3.us-west-2.amazonaws.com/aoi/country_boundary_eez.gpkg"
    )
    these_areas = (
        eez.to_crs(3832)
        .clip(coastlines.buffer(250).to_crs(3832).total_bounds)
        .to_crs(coastlines.crs)
    )
    coastlines = region_atttributes(
        coastlines.set_index("year"),
        these_areas,
        attribute_col="ISO_Ter1",
        rename_col="eez_territory",  # pyright: ignore[reportArgumentType]
    )
    roc_points = region_atttributes(
        roc_points,
        these_areas,
        attribute_col="ISO_Ter1",
        rename_col="eez_territory",  # pyright: ignore[reportArgumentType]
    )
    return coastlines, roc_points


class Cleaner(Processor):
    send_area_to_processor = False

    def __init__(
        self,
        water_index: str = "twndwi",
        initial_year: str = "1999",
        baseline_year: str = "2023",
        model_file=CLOUD_MODEL_FILE,
    ):
        super().__init__()
        self.water_index_name = water_index
        self.initial_year = initial_year  # for points
        self.baseline_year = baseline_year  #
        self.model = ModelPredictor(load(model_file))

    def process(
        self, input: list[Dataset]
    ) -> Tuple[Dataset, GeoDataFrame, GeoDataFrame | None]:
        """

        Args:
            input: An :class:`xarray.Dataset` with coordinates "x", "y" and
                "year" and (minimally) the variable corresponding to
                the value of `water_index`, `<water_index>_mad`, `count`,
                `nirwi_all`, `mndwi_all`, & `ndwi_all`. It also must contain
                any variables used in the model contained in `model_file`.

        Returns:


        Raises:
            NoOutputError: If there is determined to be no land in the area.
        """
        # Apply cloud mask
        output, mask = self.model.apply_mask(input)

        # Fill missing data
        output = fill_with_nearby_dates(output)

        # filter variables and load data into memory
        variation_var = self.water_index_name + "_mad"
        variables_to_keep = [self.water_index_name, variation_var, "count"]
        output = output[variables_to_keep].compute()

        # Remove any infinite values from water index and smooth
        output[self.water_index_name] = (
            output[self.water_index_name]
            .where(lambda wi: isfinite(wi))
            .groupby("year")
            .map(smooth_gaussian)
        )

        # Smooth variation var too
        # output[variation_var] = (
        #     output[variation_var].groupby("year").map(smooth_gaussian)
        # )
        #

        # Since we are operating using a buffer around the gadm boundary (see grid.py),
        # there are a lot of inland areas which we need to make sure are coded as land.
        # This is most important when we are detecting inland water below.
        gadm_land = load_gadm_land(output)

        # Erode gadm land significantly to ensure coastline which may be mismapped
        # is not included.
        core_land = mask_cleanup(gadm_land, mask_filters=[("erosion", 60)])

        # Define consensus land across all years
        # consensus_land = calculate_consensus_land(input[0].isel(year=0)).compute()
        consensus_land = (
            output[self.water_index_name].median(dim="year", skipna=True) < 0
        ) | core_land

        # Define candidate land as areas identified as land in this year's data.
        # This contains land, but also likely some areas misidentified due to
        # clouds that were not masked, etc. These areas will be further filtered.
        candidate_land = output[self.water_index_name] < 0

        # Identify land areas connected in some way to consensus areas.
        connected_areas = remove_disconnected_areas(consensus_land, candidate_land)

        # Define consensus land for each year as known land areas or places
        # connected to it.
        annual_consensus_land = core_land | connected_areas

        # Next, expand the analysis zone to include a collar of water, so
        # the coastline extraction can work. This is accomplished by iteratively
        # dilating land using `expand_analysis_zone`.
        analysis_zone, max_cap = expand_analysis_zone(
            analysis_zone=annual_consensus_land,
            water_index=output[self.water_index_name],
            return_max_cap=True,
        )

        # Mask areas outside the analysis zone
        self.water_index = output[self.water_index_name].where(analysis_zone)
        # obvious_water = 0.5
        # .where(~max_cap, obvious_water) .groupby("year")
        #            .map(smooth_gaussian)
        #    .rio.write_crs(output.rio.crs)
        # )

        # Now remove inland water, defined as places not connected to the ocean.

        # First create an annual water map. It is places where the water index
        # is non-negative, but also places where it is null that we know aren't land.
        # annual_water = (self.water_index > 0) | ~core_land & self.water_index.isnull()
        consensus_water = ~(consensus_land | gadm_land)
        annual_water = (
            self.water_index > 0
        ) | consensus_water & self.water_index.isnull()

        # Define core ocean areas as places that are not always land, eroded by 60-m
        # all_time_land = core_land | consensus_land
        core_ocean = mask_cleanup(consensus_water, mask_filters=[("erosion", 2)])

        # Define ocean for each year by removing water that is not connected to ocean
        annual_ocean = remove_disconnected_areas(core_ocean, annual_water)

        # Define inland water for each year as places defined as water that are
        # not ocean.
        annual_inland_water = annual_water & ~annual_ocean

        # Mask out inland water
        self.water_index = self.water_index.where(~annual_inland_water)

        # inland_water = find_inland_areas(water, ocean)
        # inland_water = find_disconnected_areas(ocean, water)
        #
        # Perform coastline delineation.
        coastlines = subpixel_contours(
            self.water_index,
            dim="year",
            min_vertices=5,
        )

        if len(coastlines) == 0:
            raise NoOutputError("no coastlines created; water index may be empty")

        certainty_masks = certainty_masking(
            output.rename({variation_var: "stdev"}), stdev_threshold=0.3
        )
        coastlines = contour_certainty(
            coastlines.set_index("year"), certainty_masks
        ).reset_index()
        roc_points = calculate_rates_of_change(
            contours=coastlines,
            water_index=output[self.water_index_name],
            water_index_name=self.water_index_name,
            baseline_year=self.baseline_year,
            initial_year=self.initial_year,
        )

        coastlines, roc_points = add_attributes(coastlines, roc_points)

        # water_index["year"] = water_index.year.astype(str)
        self.water_index["year"] = self.water_index.year.astype(str)
        return (
            # water_index.to_dataset("year"),
            self.water_index.to_dataset("year"),
            mask,
            coastlines,
            roc_points,
        )
