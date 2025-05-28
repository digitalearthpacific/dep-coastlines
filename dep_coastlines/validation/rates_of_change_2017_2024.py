from coastlines.vector import calculate_regressions
import geopandas as gpd

from dep_coastlines.config import NEXT_GEN_COASTLINES_OUTPUT


def calculate_roc_some_years(roc_points, start_year: int = 2017, end_year: int = 2024):
    these_points = roc_points[
        [f"dist_{year}" for year in range(start_year, end_year + 1)]
        + ["angle_mean", "angle_std", "geometry"]
    ]
    return calculate_regressions(these_points)


def main():
    roc_points = gpd.read_file(
        NEXT_GEN_COASTLINES_OUTPUT,
        layer="rates_of_change",
        engine="pyogrio",
        use_arrow=True,
    )
    calculate_roc_some_years(roc_points).to_file(
        "data/validation/roc_2017_to_2024.gpkg"
    )


if __name__ == "__main__":
    main()
