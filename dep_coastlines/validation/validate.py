from coastlines.validation import deacl_val_stats, generate_transects
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.enums import Resampling
import rioxarray as rx
from rioxarray.merge import merge_arrays
from shapely.geometry import Point, MultiPoint
from shapely import Geometry
from shapely.geometry import LineString
from shapely.ops import nearest_points

from dep_coastlines.grid import buffered_grid as grid
from dep_coastlines.common import coastlineItemPath
from dep_coastlines.config import (
    VECTOR_VERSION,
    VECTOR_DATASET_ID,
    VECTOR_DATETIME,
    HTTPS_PREFIX,
    CURRENT_COASTLINES_OUTPUT,
    NEXT_GEN_COASTLINES_OUTPUT,
)


def load_coastlines(
    geom,
    layer: str = "shorelines_annual",
    year: int | None = None,
    buffer: float = 100,
    use_next_gen: bool = False,
):
    buffered_geom = geom.buffer(buffer) if buffer is not None else geom
    coastlines_file = (
        NEXT_GEN_COASTLINES_OUTPUT if use_next_gen else CURRENT_COASTLINES_OUTPUT
    )
    coastlines = gpd.read_file(
        coastlines_file,
        layer=layer,
        bbox=(
            buffered_geom.total_bounds.tolist()
            if isinstance(buffered_geom, gpd.GeoDataFrame)
            else buffered_geom.bounds
        ),
        engine="pyogrio",
        use_arrow=True,
    )
    if year:
        coastlines = coastlines[coastlines.year == year]
    return coastlines


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
    lines = lines[~lines.geometry.is_empty]

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
    lines_of_difference = []
    for file in files:
        validation_lines = (
            gpd.read_file(file["path"]).assign(year=file["year"]).to_crs(3832)
        )
        coastlines = load_coastlines(validation_lines, year=file["year"])
        lines_of_difference.append(
            generate_lines_of_difference(validation_lines, coastlines)
        )

    output_file = "data/validation/lines_of_difference.gpkg"
    all_lines = pd.concat(lines_of_difference)
    all_lines.to_file(output_file)

    stats = deacl_val_stats(np.zeros(len(all_lines)), all_lines["diff"])
    print(stats)
