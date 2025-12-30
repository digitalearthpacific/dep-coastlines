"""
This is a work-in-progress script to "post-process" water index data before
vectorizing. The functionality here is part of the vectorization code from the
DEA and DEAfrica work, but it is primarily raster based, hence the name.
The actual vectorization (using subpixel_contours) may or may not belong here,
so things may move around / get renamed some in the coming weeks.

Please refer to raster_cleaning.py for specific functions.
"""

from pathlib import Path
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
from xarray import DataArray, Dataset

from dep_coastlines.cloud_model.predictor import ModelPredictor
from dep_coastlines.config import CLOUD_MODEL_FILE
from dep_coastlines.raster.cleaning import (
    define_analysis_zone,
    map_inland_water,
    fill_with_nearby_dates,
    load_gadm_land,
    load_land_additions,
    remove_disconnected_areas,
    smooth_gaussian,
)


class Cleaner(Processor):
    send_area_to_processor = False

    def __init__(
        self,
        water_index: str = "twndwi",
        initial_year: int = 1999,
        model_file: Path | str | None = CLOUD_MODEL_FILE,
    ):
        """Initialize a :class:`Processor` to clean mosaics and derive coastlines.

        Args:
            water_index: The name of the water index variable to use.
            initial_year: Passed to :func:`all_time_stats`, used to calculate
                `nsm` variable in rates of change calculates.
            model_file: An optional file containing a :class:`BaseEstimator`
                object, used to perform cloud masking on the annual mosaics.
        """
        super().__init__()
        self.water_index_name = water_index
        self.initial_year = initial_year  # for points
        self.model = (
            ModelPredictor(load(model_file)) if model_file is not None else None
        )

    def process(
        self, input: list[Dataset]
    ) -> Tuple[Dataset, GeoDataFrame, GeoDataFrame | None]:
        """Clean the input mosaics and perform coastline extraction.

        Args:
            input: An :class:`xarray.Dataset` with coordinates "x", "y" and
                "year" and (minimally) the variable named by `water_index`,
                `<water_index>_mad`, `count`, `nirwi`, `mndwi`, & `ndwi`.
                It also must contain any variables used in the model
                contained in `model_file`.

        Returns:
            A tuple containing the (masked) water index, coastlines, and
            rates of change points.

        Raises:
            NoOutputError: If there is determined to be no land in the area.
        """
        # Apply cloud mask
        if self.model is not None:
            output, _ = self.model.apply_mask(input)
        else:
            output = input

        # Fill missing data
        output = fill_with_nearby_dates(output)

        # Remove variables we no longer need and load data into memory.
        variation_var = self.water_index_name + "_mad"
        variables_to_keep = set(
            [
                self.water_index_name,
                variation_var,
                "count",
                "ndwi",
                "mndwi",
                "nirwi",
            ]
        )
        output = output[variables_to_keep].compute()

        # Remove any infinite values from water index and apply smoothing.
        water_index = (
            output[self.water_index_name]
            .where(lambda wi: isfinite(wi))
            .groupby("year")
            .map(smooth_gaussian)
            .rio.write_crs(output[self.water_index_name].rio.crs)
        )

        # Since we are operating using a buffer around the gadm boundary (see grid.py),
        # there are a lot of inland areas which we need to make sure are coded as land.
        # This is most important when we are detecting inland water below.
        gadm_land = load_gadm_land(output)

        # places missed likely because they only appeared in portions
        # of the time period (like e.g. volcanos).
        land_additions = load_land_additions(water_index)

        # Erode gadm land significantly to ensure land-ocean boundaries which are
        # mismapped (include too much land) is not included.
        core_land = (
            mask_cleanup(gadm_land, mask_filters=[("erosion", 60)]) | land_additions
        )

        # Define consensus land across all years by using all water indices
        consensus_land = (
            (output.ndwi.median(dim="year", skipna=True) < 0)
            & (output.mndwi.median(dim="year", skipna=True) < 0)
            & (output.nirwi.median(dim="year", skipna=True) < 0)
        ) | core_land

        # Define candidate land as areas identified as land in this year's data.
        # This contains land, but also likely some areas misidentified due to
        # clouds that were not masked, etc. These areas will be further filtered.
        candidate_land = water_index < 0

        # Identify land areas connected in some way to consensus areas.
        connected_areas = remove_disconnected_areas(consensus_land, candidate_land)

        # Define consensus land for each year as known land areas or places
        # connected to it.
        annual_consensus_land = core_land | connected_areas

        # Next, expand the analysis zone to include a collar of water, so
        # the coastline extraction can work. This is accomplished by iteratively
        # dilating land.
        analysis_zone = define_analysis_zone(
            core_land=annual_consensus_land,
            water_index=water_index,
        )

        # Mask areas outside the analysis zone
        water_index = water_index.where(analysis_zone)

        # Now remove inland water, defined as places not connected to the ocean.

        # First, define consensus ocean as the inverse of a liberal definition of land.
        # We use gadm as it doesn't generally include inland water bodies,
        # (though sometimes it includes rivers (PNG & Fiji) as they are connected
        # to the ocean, and will be handled during certainty masking)
        # We also include consensus land, which may include areas not in gadm.
        consensus_ocean = ~(consensus_land | gadm_land)
        annual_inland_water = map_inland_water(consensus_ocean, water_index)
        water_index = water_index.where(~annual_inland_water)

        # Perform coastline delineation.
        coastlines = subpixel_contours(
            water_index,
            dim="year",
            min_vertices=5,
        )

        if len(coastlines) == 0:
            raise NoOutputError("no coastlines created; water index may be empty")

        # Now post-process coastlines. Add certainty attributes, calculate
        # rates of change, and add any other attributes.
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
            initial_year=self.initial_year,
        )

        coastlines, roc_points = add_attributes(coastlines, roc_points)

        water_index["year"] = water_index.year.astype(str)
        water_index = water_index.to_dataset("year")
        return (
            water_index,
            coastlines,
            roc_points,
        )


def calculate_roc_stats(
    ratesofchange: gpd.GeoDataFrame,
    initial_year: int,
    minimum_valid_observations: int = 15,
):
    """Calculate rates of change statistics.

    This is a wrapper around :func:`coastlines.vector.all_time_stats` that
    also calculates certainty.

    Args:
        ratesofchange: Rates of change points.
        initial_year: Passed to :func:`vector.coastlines.all_time_stats`.
        minimum_valid_observations: The minimum number of observations
            needed for valid statistics.

    Returns:
        The rates of change points with additional columns "valid_obs",
        "valid_span", "sce", "nsm", "max_year", "min_year", and "certainty".
    """
    stats_list = ["valid_obs", "valid_span", "sce", "nsm", "max_year", "min_year"]
    ratesofchange[stats_list] = ratesofchange.apply(
        lambda x: all_time_stats(x, initial_year=initial_year), axis=1
    )

    # Initialize certainty column.
    ratesofchange["certainty"] = "good"
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

    ratesofchange.loc[ratesofchange.rate_time.abs() > 200, "certainty"] = (
        "extreme value (> 200 m)"
    )

    ratesofchange.loc[ratesofchange.angle_std > 30, "certainty"] = (
        "high angular variability"
    )
    ratesofchange.loc[
        ratesofchange.valid_obs < minimum_valid_observations, "certainty"
    ] = "insufficient observations"
    return ratesofchange


def calculate_rates_of_change(
    contours: gpd.GeoDataFrame,
    water_index: DataArray,
    water_index_name: str,
    initial_year: int,
) -> gpd.GeoDataFrame:
    """Calculate rates of change points for the given contours.

    This is a wrapper around :func:`coastlines.annual_movements`. The most
    recent year in the water_index "year" is passed as the baseline year.

    Args:
        contours: Coastlines for all years.
        water_index: The water index used to derive coastlines. This is used
            to determine directionality of rates of change points.
        water_index_name: The name of the water index.
        initial_year:

    Returns:
        Rates of change points.

    """
    water_index["year"] = water_index.year.astype(int)
    contours.year = contours.year.astype(str)
    contours = contours.set_index("year")  # pyright: ignore[reportAssignmentType]

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


def add_attributes(
    coastlines: gpd.GeoDataFrame, roc_points: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
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
    output_coastlines = region_atttributes(
        coastlines.set_index("year"),
        these_areas,
        attribute_col="ISO_Ter1",
        rename_col="eez_territory",  # pyright: ignore[reportArgumentType]
    )
    output_roc_points = region_atttributes(
        roc_points,
        these_areas,
        attribute_col="ISO_Ter1",
        rename_col="eez_territory",  # pyright: ignore[reportArgumentType]
    )
    return output_coastlines, output_roc_points
