# Adapted from coastlines.validation
import warnings

from coastlines.validation import deacl_val_stats, generate_transects
import geopandas as gpd
import pandas as pd
from pandas.core.apply import generate_apply_looper
from shapely.geometry import Point, MultiPoint

from dep_coastlines.config import CURRENT_COASTLINES_OUTPUT, VECTOR_DATASET_ID


def prep_validation_lines(validation_lines, transects):
    if validation_lines.crs != transects.crs:
        warnings.warn(
            "Reprojecting validation lines to transect CRS, may compromise data"
        )
    validation_lines = validation_lines.to_crs(transects.crs)
    intersect_gdf = transects.copy()
    years = validation_lines.year.unique()
    if len(years) > 1:
        warnings.warn(
            "More than one year present in validation lines, only keeping the first"
        )
    intersect_gdf["year"] = years[0]
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

    intersect_gdf["val_dist"] = intersect_gdf.apply(
        lambda x: Point(x.start_x, x.start_y).distance(x["val_point"]), axis=1
    )
    intersect_gdf[["val_x", "val_y"]] = intersect_gdf.apply(
        lambda x: pd.Series(x.val_point.coords[0][0:2]), axis=1
    )

    return intersect_gdf


def load_coastlines(validation_lines, year: int | None = None):
    coastlines = gpd.read_file(
        CURRENT_COASTLINES_OUTPUT,
        layer="shorelines_annual",
        bbox=validation_lines.buffer(100).total_bounds.tolist(),
    )
    if year:
        coastlines = coastlines[coastlines.year == year]
    return coastlines


def validate(validation_lines, transects):
    # Generate transects from the validation lines
    transects_geom = generate_transects(
        validation_lines.geometry.unary_union, length=40
    )
    transects = gpd.GeoDataFrame(geometry=transects_geom, crs=validation_lines.crs)

    # Generate transects from the coastlines
    coastlines = load_coastlines(validation_lines, year=validation_lines.year[0])
    #    transects_geom = generate_transects(coastlines.geometry.unary_union)
    #    transects = gpd.GeoDataFrame(geometry=transects_geom, crs=validation_lines.crs)
    validation_lines = prep_validation_lines(validation_lines, transects)

    # For now, merge geometry (good and uncertain)
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
    from shapely.geometry import LineString

    results_df["val_depcl_line"] = results_df.apply(
        lambda x: LineString([(x.val_x, x.val_y), (x.depcl_x, x.depcl_y)]), axis=1
    )

    return results_df


from shapely import buffer
from shapely.geometry import LineString
from shapely.ops import nearest_points
from coastlines.vector import points_on_line
from coastlines.validation import perpendicular_line

from dep_coastlines.grid import buffered_grid as grid
from dep_coastlines.common import coastlineItemPath
from dep_coastlines.config import (
    VECTOR_VERSION,
    VECTOR_DATASET_ID,
    VECTOR_DATETIME,
    HTTPS_PREFIX,
)

import xarray as xr
import rioxarray as rx
from rioxarray.merge import merge_arrays


def load_coastlines_raster_for_geometry(geometry, year):

    # get index of grid cells for geometry
    indices = grid[grid.intersects(geometry.unary_union)].index
    itemPath = coastlineItemPath(
        dataset_id=VECTOR_DATASET_ID, version=VECTOR_VERSION, time=VECTOR_DATETIME
    )

    def load_file(path, year):
        da = rx.open_rasterio(path, chunks=True)
        da_year = da.isel(band=da.attrs["long_name"].index(str(year)))
        da_year.attrs["long_name"] = None
        return da_year

    if len(indices) > 1:
        return merge_arrays(
            [
                load_file(f"{HTTPS_PREFIX}/{itemPath.path(item_id)}", year)
                for item_id in indices
            ]
        )
    return load_file(f"{HTTPS_PREFIX}/{itemPath.path(indices[0])}", year)


def shortest_line_from_lines_to_point(point, lines):
    return LineString([p.coords[0] for p in nearest_points(lines, point)])


import numpy as np
from shapely import Geometry


def line_along_transect(
    transect: Geometry, coastlines: Geometry, validation_lines: Geometry
):
    coastlines_on_transect = transect.intersection(coastlines)
    validation_lines_on_transect = transect.intersection(validation_lines)

    if coastlines_on_transect.is_empty or validation_lines_on_transect.is_empty:
        return LineString()

    if isinstance(coastlines_on_transect, Point) and isinstance(
        validation_lines_on_transect, Point
    ):
        coastline_point = coastlines_on_transect
        validation_point = validation_lines_on_transect

    if isinstance(coastlines_on_transect, MultiPoint) or isinstance(
        validation_lines_on_transect, MultiPoint
    ):
        coastlines_points = (
            [coastlines_on_transect]
            if isinstance(coastlines_on_transect, Point)
            else coastlines_on_transect.geoms
        )

        validation_lines_points = (
            [validation_lines_on_transect]
            if isinstance(validation_lines_on_transect, Point)
            else validation_lines_on_transect.geoms
        )

        index_of_closest_pair = np.argmin(
            [
                cp.distance(vp)
                for cp in coastlines_points
                for vp in validation_lines_points
            ]
        )
        print(f"cl: {len(coastlines_points)}")
        print(f"vl: {len(validation_lines_points)}")
        print(index_of_closest_pair)
        #        if len(coastlines_points) == 1: breakpoint()
        coastline_point = coastlines_points[
            index_of_closest_pair // len(validation_lines_points)
        ]
        validation_point = validation_lines_points[
            index_of_closest_pair % len(validation_lines_points)
        ]

    return LineString([coastline_point, validation_point])


#    coastlines_point = coastlines_on_transect if len(coastlines_on_transect) == 0 else closest_point_to


def generate_lines_of_difference(validation_lines, coastlines):
    aoi = gpd.read_file("data/aoi.gpkg", mask=validation_lines)
    year = validation_lines.year[0]

    #    inner_line = gpd.GeoDataFrame(
    #        geometry=[aoi.to_crs(3832).geometry.buffer(-40).unary_union.boundary], crs=3832
    #    )
    #    starting_points = points_on_line(inner_line, 0)
    #    inner_line = inner_line.geometry.unary_union
    coastlines_geom = coastlines.geometry.unary_union
    transects = generate_transects(coastlines_geom, interval=100)
    lines = gpd.GeoDataFrame(
        geometry=transects.apply(
            line_along_transect,
            coastlines=coastlines_geom,
            validation_lines=validation_lines.geometry.unary_union,
        ),
        crs=validation_lines.crs,
    )

    #    starting_points = points_on_line(
    #        gpd.GeoDataFrame(geometry=[coastlines_geom], crs=3832), 0
    #    ).clip(validation_lines.geometry.unary_union.buffer(100, cap_style="flat"))
    #
    #    # starting_points = points_on_line(validation_lines.set_index("year"), year)
    #    lines = gpd.GeoDataFrame(
    #        geometry=starting_points.geometry.apply(
    #            shortest_line_from_lines_to_point,
    #            # lines=coastlines_geom,
    #            lines=validation_lines.geometry.unary_union,
    #        ),
    #        crs=validation_lines.crs,
    #    )

    # Try to use an inner line (based on negatively buffered gadm) to
    # determine direction. Issue here was it was impossible to find a buffer
    # distance that covered enough range (so that the inner line was landward
    # of either other line) and also didn't eliminate narrow peninsulas
    #    def distance_to_start_point(line_of_difference, inner_land_line=inner_line):
    #        return shortest_line_from_lines_to_point(
    #            Point(line_of_difference.coords[0]), inner_land_line
    #        ).length
    #
    #    def distance_to_end_point(line_of_difference, inner_land_line=inner_line):
    #        return shortest_line_from_lines_to_point(
    #            Point(line_of_difference.coords[1]), inner_land_line
    #        ).length
    #
    #    def signed_difference(line_of_difference, inner_land_line=inner_line):
    #        distance_to_start_point = shortest_line_from_lines_to_point(
    #            Point(line_of_difference.coords[0]), inner_land_line
    #        ).length
    #        distance_to_end_point = shortest_line_from_lines_to_point(
    #            Point(line_of_difference.coords[1]), inner_land_line
    #        ).length
    #        multiplier = -1 if distance_to_start_point > distance_to_end_point else 1
    #        return line_of_difference.length * multiplier
    #
    #    lines["val_dist"] = lines.geometry.apply(distance_to_start_point)
    #    lines["depcl_dist"] = lines.geometry.apply(distance_to_end_point)
    #    lines["diff"] = lines.geometry.apply(signed_difference)
    import numpy as np
    from rasterio.enums import Resampling

    def direction(line, coastlines_raster):
        return (
            -1
            if np.diff(
                coastlines_raster.sel(
                    pd.DataFrame.from_records(
                        line.coords, columns=("x", "y")
                    ).to_xarray(),
                    method="nearest",
                )
            )
            < 0
            else 1
        )

    coastlines_raster = load_coastlines_raster_for_geometry(
        lines.geometry, 2022
    ).rio.reproject(3832, resolution=5, resampling=Resampling.bilinear)
    lines["direction"] = lines.geometry.apply(
        direction, coastlines_raster=coastlines_raster
    )
    lines["length"] = lines.geometry.apply(lambda line: line.length)
    lines["diff"] = lines.direction * lines.length

    return lines


if __name__ == "__main__":
    files = [
        {
            "path": "data/validation/validation_lines/nrcs_ortho/17Mar2025/guam_vivid_2022_30cm_SK.gpkg",
            # TODO: Get year from metadata (may differ from date of release)
            "year": 2022,
        },
        {
            "path": "data/validation/validation_lines/nrcs_ortho/17Mar2025/american_samoa_vivid_2022_30cm_lines_KR.gpkg",
            "year": 2022,
        },
        {
            "path": "data/validation/validation_lines/nrcs_ortho/17Mar2025/american_samoa_vivid_2022_30cm_mw.gpkg",
            "year": 2022,
        },
        {
            "path": "data/validation/validation_lines/nrcs_ortho/17Mar2025/cnmi_vivid_2022_30cm_ELB_110325.gpkg",
            "year": 2022,
        },
        {
            "path": "data/validation/validation_lines/nrcs_ortho/marshall_islands_vivid_2023_30cm_Jesse.gpkg",
            "year": 2023,
        },
    ]

    transects = gpd.read_file("data/src/validation_transects.gpkg")
    stats = []
    for file in files:
        validation_lines = (
            gpd.read_file(file["path"]).assign(year=file["year"]).to_crs(3832)
        )
        coastlines = load_coastlines(validation_lines, year=file["year"])
        # stats.append(validate(validation_lines, transects))
        stats.append(generate_lines_of_difference(validation_lines, coastlines))
    breakpoint()
