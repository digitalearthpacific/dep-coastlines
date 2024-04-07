from pathlib import Path

import geopandas as gpd
from shapely import make_valid

from dep_grid.grid import grid, PACIFIC_EPSG
from dep_tools.azure import blob_exists
from dep_tools.utils import write_to_blob_storage

url_prefix = "https://deppcpublicstorage.blob.core.windows.net/output/"

grid_blob_path = "aoi/coastline_grid.gpkg"
local_grid_blob_path = Path(__file__).parent / "../data/coastline_grid.gpkg"
grid_url = url_prefix + grid_blob_path

buffered_grid_blob_path = "aoi/buffered_coastline_grid.gpkg"
local_buffered_grid_blob_path = (
    Path(__file__).parent / "../data/buffered_coastline_grid.gpkg"
)
buffered_grid_url = url_prefix + buffered_grid_blob_path

OVERWRITE = False


def get_best_zone(gdf) -> gpd.GeoDataFrame:
    # Using the "most common" zone among loaded landsat data works fine except
    # in some years in some areas the most common zone differs, due to data
    # availability. Here we determine which zone, has the greatest area
    # of the buffered gadm for each cell.
    #
    # Alternatively, we would need to determine which landsat _scene_ has the
    # most area within the buffered gadm for each cell, then determine the
    # projection of that scene. I can't find such a crosswalk (a table of
    # which projection each landsat scene has).
    utm_zones = (
        gpd.read_file("data/World_UTM_Grid_-8777149898303524843.gpkg")[
            # bad stuff here I think from reprojection
            lambda x: x["ZONE"]
            != 25
        ]
        .to_crs(3832)
        .dissolve(by="ZONE")
        .reset_index()
        # makes e.g. 55 32655
        .assign(epsg=lambda d: "326" + d.ZONE.astype("Int64").astype(str).str.zfill(2))
    )

    zone_lookup = (
        gdf.overlay(utm_zones, how="intersection")
        .assign(area=lambda r: r.geometry.area)
        .sort_values("area", ascending=False)
        # drops the second by default, e.g. the lower value(s)
        .drop_duplicates(["row", "column"])
        .set_index(["row", "column"])
    ).epsg

    return gdf.set_index(["row", "column"]).join(zone_lookup).reset_index()


def make_grid(buffer=None) -> gpd.GeoDataFrame:
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
    if buffer is not None:
        # join style 2 should make corners stay corners
        full_grid = full_grid.buffer(buffer, join_style=2)

    coastline_grid = full_grid.intersection(coast_buffer)
    coastline_grid = gpd.GeoDataFrame(
        geometry=coastline_grid[~coastline_grid.geometry.is_empty]
    ).reset_index(names=["column", "row"])
    coastline_grid["id"] = coastline_grid.apply(
        lambda r: f"{str(r.row).zfill(3)}{str(r.column).zfill(3)}", axis=1
    )
    coastline_grid["tile_id"] = coastline_grid.apply(
        lambda r: f"{str(r.row)},{str(r.column)}", axis=1
    )

    return get_best_zone(coastline_grid)


if not blob_exists(grid_blob_path) or OVERWRITE:
    coastline_grid = make_grid()
    write_to_blob_storage(
        coastline_grid,
        grid_blob_path,
        driver="GPKG",
        layer_name="coastline_grid",
    )
    coastline_grid.to_file(local_grid_blob_path)

if not blob_exists(buffered_grid_blob_path) or OVERWRITE:
    coastline_grid = make_grid(250)
    write_to_blob_storage(
        coastline_grid,
        buffered_grid_blob_path,
        driver="GPKG",
        layer_name="coastline_grid",
    )
    coastline_grid.to_file(local_buffered_grid_blob_path)


grid = gpd.read_file(grid_url).set_index(["row", "column"])
buffered_grid = gpd.read_file(buffered_grid_url).set_index(["row", "column"])

test_tiles = [
    (11, 123),
    (14, 50),
    (15, 50),
    (15, 51),
    (16, 67),
    (16, 71),
    (17, 67),
    (17, 71),
    (18, 68),
    (19, 63),
    (19, 64),
    (19, 65),
    (19, 66),
    (19, 67),
    (20, 62),
    (20, 63),
    (20, 64),
    (20, 65),
    (20, 66),
    (20, 67),
    (21, 62),
    (21, 63),
    (21, 64),
    (21, 65),
    (21, 66),
    (21, 67),
    (22, 63),
    (22, 64),
    (22, 65),
    (22, 66),
    (22, 67),
    (23, 66),
    (24, 68),
    (25, 68),
    (26, 70),
    (27, 62),
    (29, 65),
    (31, 64),
    (31, 65),
    (32, 23),
    (32, 24),
    (33, 24),
    (49, 55),
    (49, 56),
    (51, 51),
    (52, 51),
    (20, 108),
    (28, 98),
    (30, 100),
    (35, 93),
    (35, 94),
    (36, 72),
    (36, 75),
    (36, 76),
    (36, 95),
    (37, 72),
    (37, 75),
    (37, 76),
    (37, 95),
    (38, 61),
    (38, 62),
    (38, 75),
    (38, 76),
    (39, 60),
    (40, 53),
    (40, 59),
    (40, 60),
    (40, 61),
    (41, 58),
    (41, 59),
    (42, 57),
    (42, 58),
    (42, 59),
    (43, 57),
    (43, 58),
    (43, 92),
    (44, 58),
    (44, 92),
    (45, 57),
    (46, 89),
    (46, 90),
    (47, 88),
]

test_grid = grid.loc[test_tiles]
test_buffered_grid = buffered_grid.loc[test_tiles]
