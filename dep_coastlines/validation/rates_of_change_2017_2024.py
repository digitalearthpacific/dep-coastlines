from coastlines.vector import calculate_regressions
import geopandas as gpd

from dep_coastlines.config import NEXT_GEN_COASTLINES_OUTPUT
from dep_coastlines.continental import generate_hotspots
from dep_coastlines.vector import calculate_roc_stats


def calculate_roc_some_years(roc_points, start_year: int = 2017, end_year: int = 2024):
    breakpoint()
    these_points = roc_points[
        [f"dist_{year}" for year in range(start_year, end_year + 1)]
        + ["angle_mean", "angle_std", "geometry"]
    ]
    roc_gdf = calculate_regressions(these_points)
    return calculate_roc_stats(
        roc_gdf, start_year, minimum_valid_observations=start_year + 1 - end_year
    )


def main():
    roc_points = gpd.read_file(
        NEXT_GEN_COASTLINES_OUTPUT,
        layer="rates_of_change",
        engine="pyogrio",
        use_arrow=True,
    )
    roc_points_2017_2024 = calculate_roc_some_years(roc_points)
    roc_points_2017_2024.to_file("data/validation/roc_2017_to_2024.gpkg")

    shorelines = gpd.read_file(
        NEXT_GEN_COASTLINES_OUTPUT,
        layer="shorelines_annual",
        engine="pyogrio",
        use_arrow=True,
    )
    shorelines = shorelines[shorelines.year >= 2017].set_index("year")
    generate_hotspots(5_000, shorelines, roc_points_2017_2024, 2024).to_file(
        "data/validation/hotspots_5000m_2017_to_2024.gpkg"
    )


if __name__ == "__main__":
    main()
