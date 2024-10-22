# Adapted from coastlines.validation
import warnings

from coastlines.validation import deacl_val_stats
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from dep_coastlines.config import CURRENT_COASTLINES_OUTPUT


def prep_validation_lines(validation_lines, transects):
    if validation_lines.crs != transects.crs:
        warnings.warn(
            "Reprojection validation lines to transect CRS, may compromise data"
        )
    validation_lines = validation_lines.to_crs(transects.crs)
    intersect_gdf = transects.copy()
    intersect_gdf["val_point"] = transects.intersection(
        validation_lines.geometry.unary_union
    )
    intersect_gdf = intersect_gdf.loc[intersect_gdf.val_point.geom_type == "Point"]

    # Add measurement metadata
    intersect_gdf[["start_x", "start_y"]] = intersect_gdf.apply(
        lambda x: pd.Series(x.geometry.coords[0]), axis=1
    )
    intersect_gdf[["end_x", "end_y"]] = intersect_gdf.apply(
        lambda x: pd.Series(x.geometry.coords[1]), axis=1
    )
    # Not sure about these names, they were 0_dist etc
    intersect_gdf["val_dist"] = intersect_gdf.apply(
        lambda x: Point(x.start_x, x.start_y).distance(x["val_point"]), axis=1
    )
    intersect_gdf[["val_x", "val_y"]] = intersect_gdf.apply(
        lambda x: pd.Series(x.val_point.coords[0][0:2]), axis=1
    )
    return intersect_gdf


def load_coastlines(validation_lines):
    return gpd.read_file(
        CURRENT_COASTLINES_OUTPUT,
        layer="shorelines_annual",
        bbox=validation_lines.buffer(100).total_bounds.tolist(),
    )


def validate(validation_lines, transects):
    validation_lines = prep_validation_lines(validation_lines, transects).assign(
        year=2003
    )
    coastlines = load_coastlines(validation_lines)

    results_df = validation_lines.merge(
        coastlines, on="year", suffixes=("_val", "_depcl")
    )

    results_df["intersect"] = results_df.apply(
        lambda x: x.geometry_val.intersection(x.geometry_depcl), axis=1
    )
    results_df = results_df[
        results_df.apply(lambda x: x.intersect.type == "Point", axis=1)
    ]
    results_df[f"depcl_x"] = gpd.GeoSeries(results_df["intersect"]).x
    results_df[f"depcl_y"] = gpd.GeoSeries(results_df["intersect"]).y

    # For each row, compute distance between origin and intersect
    results_df[f"depcl_dist"] = results_df.apply(
        lambda x: x.intersect.distance(Point(x.start_x, x.start_y)), axis=1
    )

    results_df["error_m"] = results_df.val_dist - results_df.depcl_dist
    stats = deacl_val_stats(results_df.val_dist, results_df, depcl_dist)
    breakpoint()


if __name__ == "__main__":
    validation_lines = gpd.read_file("data/validation/validation_test.gpkg").assign(
        year=2003
    )
    transects = gpd.read_file("data/src/validation_transects.gpkg")
    validate(validation_lines, transects)
