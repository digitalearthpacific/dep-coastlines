from pathlib import Path

from coastlines.validation import generate_transects
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.enums import Resampling
from shapely.geometry import Point, MultiPoint
from shapely import Geometry
from shapely.geometry import LineString
from shapely.ops import nearest_points

from util import load_coastlines, load_coastlines_raster_for_geometry


def load_validation_data():
    files = [
        {
            "path": "data/validation/validation_lines/nrcs_ortho/21Apr2025/guam_vivid_2022_30cm_SK.gpkg",
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
        {"path": "data/validation/validation_lines/WV12DEC13015047.gpkg", "year": 2012},
    ]
    validation_lines = pd.concat(
        [
            gpd.read_file(file["path"])
            .assign(year=file["year"], set_id=Path(file["path"]).stem)
            .to_crs(3832)
            for file in files
        ]
    ).to_crs(3832)

    s2_validation_lines = gpd.read_file("data/validation/s2_validation_lines_2.gpkg")
    return pd.concat([validation_lines, s2_validation_lines])


def shortest_line_from_lines_to_point(point, lines):
    return LineString([p.coords[0] for p in nearest_points(lines, point)])


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
        coastline_point = coastlines_points[
            index_of_closest_pair // len(validation_lines_points)
        ]
        validation_point = validation_lines_points[
            index_of_closest_pair % len(validation_lines_points)
        ]

    return LineString([coastline_point, validation_point])


def generate_lines_of_difference(validation_lines, coastlines):
    print(validation_lines.set_id.iloc[0])
    coastlines_geom = coastlines.geometry.union_all()
    transects = generate_transects(coastlines_geom, interval=30)
    lines = gpd.GeoDataFrame(
        geometry=transects.apply(
            line_along_transect,
            coastlines=coastlines_geom,
            validation_lines=validation_lines.geometry.union_all(),
        ),
        crs=validation_lines.crs,
    )
    lines = lines[~lines.geometry.is_empty]

    coastlines_raster = load_coastlines_raster_for_geometry(
        lines.geometry, validation_lines.year.iloc[0]
    ).rio.reproject(3832, resolution=10, resampling=Resampling.bilinear)

    def direction(line, coastlines_raster):
        return (
            1
            if np.diff(
                coastlines_raster.sel(
                    pd.DataFrame.from_records(
                        line.coords, columns=("x", "y")
                    ).to_xarray(),
                    method="nearest",
                )
            )
            < 0
            else -1
        )

    lines["direction"] = lines.geometry.apply(
        direction, coastlines_raster=coastlines_raster
    )
    lines["length"] = lines.geometry.apply(lambda line: line.length)
    lines["diff"] = lines.direction * lines.length

    return lines


def main():
    validation_lines = load_validation_data()
    all_lines = (
        validation_lines.groupby(["set_id", "year"]).apply(
            lambda set: generate_lines_of_difference(
                set, load_coastlines(set, year=set.year.iloc[0], use_next_gen=True)
            )
        )
        # add set_id back as a column
        .reset_index()
    )
    output_file = "data/validation/annual_lines_of_difference.gpkg"
    all_lines.to_file(output_file)


if __name__ == "__main__":
    main()
