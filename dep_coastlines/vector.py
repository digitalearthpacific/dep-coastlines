"""
This is a work-in-progress script to "post-process" water index data before
vectorizing. The functionality here is part of the vectorization code from the
DEA and DEAfrica work, but it is primarily raster based, hence the name.
The actual vectorization (using subpixel_contours) may or may not belong here,
so things may move around / get renamed some in the coming weeks.

Please refer to raster_cleaning.py for specific functions.
"""

import operator
from typing import Callable, Tuple

import geohash
import xrspatial as xs
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
from xarray import DataArray, Dataset

from dep_coastlines.cloud_model.predictor import ModelPredictor
from dep_coastlines.config import CLOUD_MODEL_FILE
from dep_coastlines.raster.cleaning import (
    fill_with_nearby_dates,
    find_inland_areas,
    load_gadm_land,
    remove_disconnected_land,
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


def calculate_consensus_land(ds: Dataset) -> DataArray:
    """Returns true for areas for which the all-years medians of mndwi,
    ndwi and nirwi are less than zero. (nirwi
    is negative where the nir08 band is greater than 0.128.)"""
    return (ds.nirwi_all < 0) & (ds.mndwi_all < 0) & (ds.ndwi_all < 0)


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


class Cleaner(Processor):

    def __init__(
        self,
        water_index: str = "twndwi",
        index_threshold: float = 0,
        comparison: Callable = operator.lt,
        number_of_expansions: int = 64,
        initial_year: str = "1999",
        baseline_year: str = "2023",
        model_file=CLOUD_MODEL_FILE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index_threshold = index_threshold
        self.water_index_name = water_index
        self.initial_year = initial_year  # for points
        self.baseline_year = baseline_year  #

        self.model = ModelPredictor(load(model_file))
        self.comparison = comparison
        self.number_of_expansions = number_of_expansions
        self.send_area_to_processor = False

    def land(self, output):
        return self.comparison(output[self.water_index_name], self.index_threshold)

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
        """Add other attributes to coastlines and rates of change points.

        Currently the only attribute added is the three-letter country code
        for the economic exclusion zone within which the feature falls.
        """
        # Using read_dataframe because gpd.read_file returns a 403 error.
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

    def process(
        self, input: Dataset | list[Dataset]
    ) -> Tuple[Dataset, GeoDataFrame, GeoDataFrame | None]:
        """

        Args:
            input: An :class:`xarray.Dataset` with coordinates "x", "y" and
                "year" variables:

        Returns:


        Raises:
            NoOutputError: If there is determined to be no land in the area.
        """
        breakpoint()
        # Apply cloud mask
        output, mask = self.model.apply_mask(input)

        # Fill missing data
        output = fill_with_nearby_dates(output)

        # filter variables and load data into memory
        variation_var = self.water_index_name + "_mad"
        variables_to_keep = [self.water_index_name, variation_var, "count"]
        output = output[variables_to_keep].compute()

        an_input = input[0] if isinstance(input, list) else input
        # Consensus land are places that are identified as land by
        # multiple water indices across all-time mosaics. These are areas
        # that we are almost certain contain land. They likely exclude
        # land areas in some years, so should not be considered complete.
        consensus_land = calculate_consensus_land(an_input.isel(year=0)).compute()

        # Define candidate land as areas identified as land in this year's data.
        # This contains land, but also likely some areas misidentified due to
        # clouds that were not masked, etc. These areas will be further filtered.
        candidate_land = self.land(output)

        # Identify candidate areas connected in some way to consensus areas.
        # This ensures that we are including true bounds of known land areas
        # for _this_ year, based on our target water index & data.
        connected_areas = remove_disconnected_land(consensus_land, candidate_land)

        # This identifies consensus land area that are a single pixel in size.
        no_connected_neighbors = xs.focal.mean(consensus_land) == 0

        # Candidate land that is on top of consensus land of only a pixel in size
        # may be false positives.
        suspicious_connected_areas = candidate_land & no_connected_neighbors

        # Since we have only defined land areas to this point, we need to expand our
        # analysis to include bounding water so coastline delineation via
        # the marching squares algorithm can correctly function.
        # To do this, we first filter "connected areas" to remove suspicous ones.
        analysis_zone = connected_areas & ~suspicious_connected_areas

        # Next, we expand the analysis zone to include a collar of water. This is
        # accomplished by iteratively dilating land only by 2 pixels.
        # See `expand_analysis_zone` for details.
        analysis_zone, max_cap = self.expand_analysis_zone(analysis_zone, output, True)

        # Redefine candidate land based on the new analysis zone.
        candidate_land = self.land(output.where(analysis_zone))
        # Now again remove disconnected areas which we may have expanded into.
        connected_areas = remove_disconnected_land(consensus_land, candidate_land)
        # don't remove suspicous areas because expanded land may not be within
        # 1 cell of consensus land
        disconnected_areas = candidate_land & ~connected_areas
        analysis_zone = analysis_zone & ~disconnected_areas
        if not analysis_zone.any():
            raise NoOutputError(
                "Analysis zone is empty, there may be no land detected in this area"
            )

        # Since we are operating using a buffer around the gadm boundary (see grid.py),
        # there are a lot of inland areas which we need to make sure are coded as land.
        # Much of this becomes necessary because we are dealing with bool data (things
        # can only be True/False, not null).
        gadm_land = load_gadm_land(output)

        # basically to capture land outside the buffer that would otherwise
        # link inland water to perceived ocean
        # The amount here is linked to the buffer value in the grid
        core_land = mask_cleanup(gadm_land, mask_filters=[("erosion", 60)])

        obvious_water = 0.5
        # Do final masking of the water index.
        # 1. Only perform coastline extraction over defined analysis zone
        #    and core land.
        # 2. Areas that may have been at the limit of expansion in need to be
        #    collared by water.
        # 3. Apply gaussian smoothing. This is done here rather than during loading
        #    using e.g. a cubic convolution to control smoothing over null areas.
        #    (smooth gaussian handles nans).
        self.water_index = (
            output[self.water_index_name]
            .where(analysis_zone | core_land)
            .where(~max_cap, obvious_water)
            .groupby("year")
            .map(smooth_gaussian)
            .rio.write_crs(output.rio.crs)
        )

        # Next, we need to code inland areas which lie outside our analysis
        # buffer as land.
        obvious_land = -0.5
        water = ~self.land(
            self.water_index.where(
                ~(self.water_index.isnull() & (core_land | consensus_land)),
                obvious_land,
            ).to_dataset(name=self.water_index_name)
        )

        # consensus land may have inland water, but gadm doesn't.
        # Also, consensus land will have masked areas as False rather
        # than nan. Neither of these should matter because gadm doesn't have
        # these issues. I bring in consensus land basically to fix the areas
        # near shoreline that gadm may miss.
        land = gadm_land | consensus_land
        ocean = mask_cleanup(~land, mask_filters=[("erosion", 2)])
        inland_water = find_inland_areas(water, ocean)
        # Remove some infinite values that may be present in the water index.
        water_index = self.water_index.where(~inland_water).where(
            lambda wi: isfinite(wi)
        )

        # Perform coastline delineation.
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
        self.roc_points = calculate_rates_of_change(
            contours=self.coastlines,
            water_index=output[self.water_index_name],
            water_index_name=self.water_index_name,
            baseline_year=self.baseline_year,
            initial_year=self.initial_year,
        )

        self.add_attributes()

        water_index["year"] = water_index.year.astype(str)
        return (
            water_index.to_dataset("year"),
            mask,
            self.coastlines,
            self.roc_points,
        )
