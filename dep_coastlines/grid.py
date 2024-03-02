from pathlib import Path

import geopandas as gpd
from shapely import make_valid

from dep_grid.grid import grid, PACIFIC_EPSG
from dep_tools.azure import blob_exists
from dep_tools.utils import write_to_blob_storage

grid_blob_path = "aoi/coastline_grid.gpkg"
local_grid_blob_path = Path(__file__).parent / "../data/coastline_grid.gpkg"
url_prefix = "https://deppcpublicstorage.blob.core.windows.net/output/"
grid_url = url_prefix + grid_blob_path


def get_best_zone(gdf):
    # Using the "most common" zone among loaded landsat data works fine except
    # in some years in some areas the most common zone differs, due to data
    # availability. Here we determine which zone, has the greatest amoung area
    # of the buffered gadm for each cell.
    #
    # Alternatively, we would need to determine which landsat _scene_ has the
    # most area within the buffered gadm for each cell, then determine the
    # projection of that scene. I can't find such a crosswalk (a table of
    # which projection each landsat scene has).
    utm_zones = (
        gpd.read_file("../data/World_UTM_Grid_-8777149898303524843.gpkg")[
            # bad stuff here I think from reprojection
            lambda x: x["ZONE"]
            != 25
        ]
        .to_crs(3832)
        .dissolve(by="ZONE")
        .reset_index()
        # makes e.g. 55 32655
        .assign(ZONE=lambda d: "326" + d.ZONE.astype("Int64").astype(str))
        .set_index(["row", "column"])
    )

    zone_lookup = (
        grid.overlay(utm_zones, how="intersection")
        .assign(area=lambda r: r.geometry.area)
        .sort_values("area", ascending=False)
        # drops the second by default, e.g. the lower value(s)
        .drop_duplicates(["row", "column"])
        .set_index(["row", "column"])
    ).ZONE

    return gdf.join(zone_lookup)


if not blob_exists(grid_blob_path):
    # if not Path(grid_blob_path).exists:
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

    # coastline_grid = get_best_zone(coastline_grid)

    write_to_blob_storage(
        coastline_grid,
        grid_blob_path,
        write_args=dict(driver="GPKG", layer_name="coastline_grid"),
    )


grid = gpd.read_file(grid_url).set_index(["column", "row"])
# grid = gpd.read_file(grid_blob_path).set_index(["column", "row"])
test_tiles = [
    (53, 47),
    (61, 19),
    (68, 15),  # Tongatapu
    (68, 16),  # Tongatapu
    (62, 30),  # Tuvalu
    (49, 49),
    (48, 14),
    (60, 20),
    (118, 11),  # Pitcairn
    (23, 31),  # PNG
]

test_grid = grid.loc[test_tiles]
