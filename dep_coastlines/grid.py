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
        lambda r: f"{str(r.column).zfill(3)}{str(r.row).zfill(3)}", axis=1
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
    coastline_grid.to_file(grid_blob_path)

if not blob_exists(buffered_grid_blob_path) or OVERWRITE:
    coastline_grid = make_grid(250)
    write_to_blob_storage(
        coastline_grid,
        buffered_grid_blob_path,
        driver="GPKG",
        layer_name="coastline_grid",
    )
    coastline_grid.to_file(buffered_grid_blob_path)


grid = gpd.read_file(grid_url).set_index(["column", "row"])
buffered_grid = gpd.read_file(buffered_grid_url).set_index(["column", "row"])
test_tiles = [
    (62, 30),  # Tuvalu
    (23, 31),  # PNG
    (53, 47),
    (68, 15),  # Tongatapu
    (68, 16),  # Tongatapu
    (49, 49),
    (48, 14),
    (118, 11),  # Pitcairn
    (60, 20),  # Fiji
    (61, 19),
    (60, 26),
    (64, 16),
    (65, 17),
    (61, 18),
    (63, 18),
    (64, 18),
    (65, 18),
    (60, 19),
    (62, 19),
    (63, 19),
    (64, 19),
    (65, 19),
    (59, 20),
    (61, 20),
    (62, 20),
    (63, 20),
    (64, 20),
    (60, 21),
    (61, 21),
    (62, 21),
    (63, 21),
    (64, 21),
    (63, 22),
    (64, 22),
    (59, 26),
    (62, 27),  # Fiji ^^^
    (67, 25),  # WF
    (65, 24),
    (65, 23),  # ^^
]

test_grid = grid.loc[test_tiles]
test_buffered_grid = buffered_grid.loc[test_tiles]
