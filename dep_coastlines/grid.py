from pathlib import Path

import geopandas as gpd
from osgeo import gdal, gdalconst
import pandas as pd
from shapely import make_valid
from s3fs import S3FileSystem

from dep_tools.grids import grid, PACIFIC_EPSG, gadm
from dep_tools.utils import fix_winding

import dep_coastlines.config as config

OVERWRITE = False


def assign_crs(gdf) -> gpd.GeoDataFrame:
    # Using the "most common" zone among loaded landsat data works fine except
    # in some years in some areas the most common zone differs, due to data
    # availability. Here we determine which zone, has the greatest area of
    # the buffered gadm for each cell.
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
        .to_crs(config.OUTPUT_CRS)
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


def remove_inland_borders(aoi):
    # The only inland border is between PNG & Indonesia.

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
    return gpd.GeoDataFrame(geometry=aoi).overlay(indonesia, how="difference")


def full_aoi() -> gpd.GeoDataFrame:
    padm = gadm()
    hawaii = (
        gpd.read_file(
            "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_USA.gpkg",
            layer="ADM_ADM_1",
        )
        .set_index("ISO_1")
        .loc[["US-HI"]]
        .reset_index()
    )
    return pd.concat([padm, hawaii]).dissolve()[["geometry"]]


def make_grid(grid_buffer=250) -> gpd.GeoDataFrame:
    aoi = full_aoi()

    aoi = remove_inland_borders(aoi)

    # Buffer the country boundaries enough to be sure to capture the coastline.
    # Approx 2km at equator, other CRSes did not work (horizontal stripes that
    # spanned the globe).
    # Note this will give a warning about geographic buffers being incorrect,
    # but I checked and this is fine
    buffer_distance_dd = 0.018018018018018018
    tiny_buffer = 0.0001
    coast_buffer = make_valid(
        aoi.buffer(buffer_distance_dd - tiny_buffer).to_crs(PACIFIC_EPSG).unary_union
    )

    full_grid = grid(return_type="GeoSeries")
    # limit to cells which overlap the aoi (this prevents including cells
    # that are just buffer)
    full_grid = full_grid.iloc[
        full_grid.to_crs(4326).sindex.query(
            aoi.geometry.iloc[0], predicate="intersects"
        )
    ]

    # This buffer is for grid cells. Adding overlap prevents disjoint edges
    # between tiles.
    if grid_buffer is not None:
        # join style 2 should make corners stay corners
        full_grid = full_grid.buffer(grid_buffer, join_style=2)

    coastline_grid = full_grid.intersection(coast_buffer)
    coastline_grid = gpd.GeoDataFrame(
        geometry=coastline_grid[~coastline_grid.geometry.is_empty]
    ).reset_index(names=["column", "row"])
    coastline_grid["id"] = coastline_grid.apply(
        lambda r: f"{str(r.column).zfill(3)}{str(r.row).zfill(3)}", axis=1
    )
    coastline_grid["tile_id"] = coastline_grid.apply(
        lambda r: f"{str(r.column)},{str(r.row)}", axis=1
    )
    coastline_grid["geometry"] = coastline_grid.geometry.apply(fix_winding)
    return assign_crs(coastline_grid)


remote_fs = S3FileSystem(anon=True)

buffered_grid_name = "buffered_coastline_grid.gpkg"
local_buffered_grid_blob_path = (
    Path(__file__).parent / f"../data/raw/{buffered_grid_name}"
)
buffered_grid_bucket_path = (
    f"{config.BUCKET}/dep_ls_coastlines/raw/{buffered_grid_name}"
)

aoi_raster_name = "coastlines_aoi.tif"
local_aoi_raster_path = Path(__file__).parent / f"../data/raw/{aoi_raster_name}"
remote_aoi_raster_path = f"{config.BUCKET}/dep_ls_coastlines/raw/{aoi_raster_name}"

if not remote_fs.exists(buffered_grid_bucket_path) or OVERWRITE:
    aoi = full_aoi()

    opts = gdal.RasterizeOptions(
        creationOptions=dict(
            COMPRESS="CCITTFAX4", TILED="YES", NBITS=1, SPARSE_OK="TRUE"
        ),
        outputType=gdalconst.GDT_Byte,
        xRes=30,
        yRes=30,
        burnValues=[1],
    )
    gdal.Rasterize(
        local_aoi_path, gdal.OpenEx(aoi.to_crs(PACIFIC_EPSG).to_json()), options=opts
    )
    rw_fs = S3FileSystem(anon=False)
    rw_fs.put_file(local_aoi_path, remote_aoi_path)


if not remote_fs.exists(buffered_grid_bucket_path) or OVERWRITE:
    rw_fs = S3FileSystem(anon=False)
    coastline_grid = make_grid(250)
    coastline_grid.to_file(local_buffered_grid_blob_path)
    rw_fs.put_file(local_buffered_grid_blob_path, buffered_grid_bucket_path)


buffered_grid = gpd.read_file(
    remote_fs.open(f"s3://{buffered_grid_bucket_path}")
).set_index(["column", "row"])


_test_tiles = [
    (123, 11),
    (50, 14),
    (50, 15),
    (51, 15),
    (67, 16),
    (71, 16),
    (67, 17),
    (71, 17),
    (68, 18),
    (63, 19),
    (64, 19),
    (65, 19),
    (66, 19),
    (67, 19),
    (62, 20),
    (63, 20),
    (64, 20),
    (65, 20),
    (66, 20),
    (67, 20),
    (62, 21),
    (63, 21),
    (64, 21),
    (65, 21),
    (66, 21),
    (67, 21),
    (63, 22),
    (64, 22),
    (65, 22),
    (66, 22),
    (67, 22),
    (66, 23),
    (68, 24),
    (68, 25),
    (70, 26),
    (62, 27),
    (65, 29),
    (64, 31),
    (65, 31),
    (23, 32),
    (24, 32),
    (24, 33),
    (55, 49),
    (56, 49),
    (51, 51),
    (51, 52),
    (108, 20),
    (98, 28),
    (100, 30),
    (93, 35),
    (94, 35),
    (72, 36),
    (75, 36),
    (76, 36),
    (95, 36),
    (72, 37),
    (75, 37),
    (76, 37),
    (95, 37),
    (61, 38),
    (62, 38),
    (75, 38),
    (76, 38),
    (60, 39),
    (53, 40),
    (59, 40),
    (60, 40),
    (61, 40),
    (58, 41),
    (59, 41),
    (57, 42),
    (58, 42),
    (59, 42),
    (57, 43),
    (58, 43),
    (92, 43),
    (58, 44),
    (92, 44),
    (57, 45),
    (89, 46),
    (90, 46),
    (88, 47),
]
test_buffered_grid = buffered_grid.loc[_test_tiles]
