from pathlib import Path

import geopandas as gpd
from shapely import make_valid

from dep_grid.grid import grid, PACIFIC_EPSG
from dep_tools.azure import blob_exists
from dep_tools.utils import write_to_blob_storage

grid_blob_path = "aoi/coastline_grid.gpkg"
url_prefix = "https://deppcpublicstorage.blob.core.windows.net/output/"
grid_url = url_prefix + grid_blob_path

if not blob_exists(grid_blob_path):
    aoi = gpd.read_file(url_prefix + "aoi/aoi.gpkg")

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
    ).reset_index(names=["column", "row"])
    coastline_grid["id"] = coastline_grid.apply(
        lambda r: f"{str(r.column).zfill(3)}{str(r.row).zfill(3)}", axis=1
    )
    write_to_blob_storage(
        coastline_grid,
        grid_blob_path,
        write_args=dict(driver="GPKG", layer_name="coastline_grid"),
    )


grid = gpd.read_file(grid_url).set_index(["column", "row"])
test_tiles = [
    (61, 19),
    (53, 47),
    (48, 14),
    (68, 15),
    (68, 16),
    (62, 30),
    (49, 49),
    (60, 20),
]

test_grid = grid.loc[test_tiles]
