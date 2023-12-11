from pathlib import Path

import geopandas as gpd
from shapely import make_valid

from dep_grid.grid import grid, PACIFIC_EPSG

grid_file = "data/coastlines_grid.gpkg"

if not Path(grid_file).exists():
    aoi = gpd.read_file("data/aoi.gpkg")

    # A buffer of the exterior line created weird interior gaps,
    # (See https://github.com/locationtech/jts/issues/876)
    # so I buffer the polygons by a positive then negative buffer
    # and take the difference. I do a tiny amount here so I can remove
    # the boundary between PNG & Indonesia without pulling out any of
    # western PNG.
    tiny_buffer = 0.0001
    aoi_buffer = aoi.buffer(tiny_buffer)
    aoi_negative_buffer = aoi.buffer(-tiny_buffer)
    aoi = aoi_buffer.difference(aoi_negative_buffer)

    # remove border of PNG & Indonesia
    indonesia = gpd.read_file(
        f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_IDN.gpkg"
    )
    indonesia.geometry = indonesia.buffer(0.01)
    aoi = gpd.GeoDataFrame(geometry=aoi).overlay(indonesia, how="difference")

    # Approx 2km at equator, other CRSes did not work (horizontal stripes that
    # spanned the globe).
    # Note this will give a warning about geographic buffers being incorrect,
    # but I checked and this is fine
    buffer_distance_dd = 0.018018018018018018
    coast_buffer = make_valid(
        aoi.buffer(buffer_distance_dd - tiny_buffer).to_crs(PACIFIC_EPSG).unary_union
    )

    full_grid = grid(return_type="GeoSeries")

    coastline_grid = full_grid.intersection(coast_buffer)
    coastline_grid = gpd.GeoDataFrame(
        geometry=coastline_grid[~coastline_grid.geometry.is_empty]
    )
    coastline_grid.reset_index(names=["column", "row"]).to_file(
        "data/coastlines_grid.gpkg"
    )

grid = gpd.read_file(grid_file).set_index(["column", "row"])
# pathrows = gpd.read_file(
#    "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip"
# )
#
# intersection = pathrows.intersection(coast_buffer)
#
# pathrows.geometry = intersection
# output = pathrows[~pathrows.is_empty]
#
# output.to_file("data/coastline_split_by_pathrow.gpkg")
