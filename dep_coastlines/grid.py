"""Build and load the analysis grid.

The grid and the area of interest raster are stored on S3 under
`config.GRID_VERSION`. Loading the grid reads it from there. To build both
after changing the area of interest, bump `config.GRID_VERSION` and run
`python -m dep_coastlines.grid`.
"""

from functools import cache
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
import rioxarray as rx
from dep_tools.grids import PACIFIC_EPSG, gadm, grid
from dep_tools.utils import fix_winding
from osgeo import gdal, gdalconst
from s3fs import S3FileSystem
from shapely import make_valid
from shapely.geometry import box
from xarray import DataArray

import dep_coastlines.config as config

_grid_folder = (
    f"{config.BUCKET}/dep_ls_coastlines/raw/grid/"
    f"{config.GRID_VERSION.replace('.', '-')}"
)

buffered_grid_name = "buffered_coastline_grid.gpkg"
buffered_grid_bucket_path = f"{_grid_folder}/{buffered_grid_name}"

aoi_raster_name = "coastlines_aoi.tif"
remote_aoi_raster_path = f"{_grid_folder}/{aoi_raster_name}"


@cache
def buffered_grid() -> gpd.GeoDataFrame:
    """Load the analysis grid, indexed by (column, row).

    The grid is read from S3 once and every call returns that same object.
    Don't modify it in place (e.g. adding columns or `inplace=True`), or the
    change will show up for every other caller. Use `.copy()` first if needed.

    Raises:
        FileNotFoundError: If the grid is not on S3. The grid is never
            rebuilt without being asked, so existing outputs always match
            the grid they were made with.
    """
    _check_on_s3(buffered_grid_bucket_path)
    # set_index keeps the GeoDataFrame at runtime, but pyright types its
    # result as a plain DataFrame
    return cast(
        gpd.GeoDataFrame,
        gpd.read_file(
            S3FileSystem(anon=True).open(f"s3://{buffered_grid_bucket_path}")
        ).set_index(["column", "row"]),
    )


def aoi_raster() -> DataArray:
    """Open the area of interest raster (1 for land) from S3.

    Raises:
        FileNotFoundError: If the raster is not on S3.
    """
    _check_on_s3(remote_aoi_raster_path)
    return cast(
        DataArray, rx.open_rasterio(f"s3://{remote_aoi_raster_path}", chunks=True)
    )


def build() -> None:
    """Build the area of interest raster and the grid and upload them to S3.

    Both are built from the same area of interest, so they always match.
    This needs S3 write access.

    Raises:
        FileExistsError: If the grid already exists for
            `config.GRID_VERSION`. Outputs were made with that version, so
            it must not change.
    """
    # The grid is uploaded last, so it only exists once a build has finished.
    # A raster left by a failed upload is overwritten instead of blocking.
    if S3FileSystem(anon=True).exists(buffered_grid_bucket_path):
        raise FileExistsError(
            f"Grid version {config.GRID_VERSION} already exists at "
            f"s3://{buffered_grid_bucket_path}. "
            "Bump config.GRID_VERSION to build a new one."
        )

    # Build both locally first, so a failure leaves nothing half-built on S3
    aoi = _full_aoi()
    local_aoi_raster_path = _write_aoi_raster(aoi)
    local_grid_path = _write_grid(aoi)

    # Upload the grid last; its presence marks the version as complete
    write_fs = S3FileSystem(anon=False)
    write_fs.put_file(local_aoi_raster_path, remote_aoi_raster_path)
    write_fs.put_file(local_grid_path, buffered_grid_bucket_path)


def debug_grid() -> gpd.GeoDataFrame:
    """Select tiles which have been useful in debugging outputs."""
    test_tiles = [
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
    return buffered_grid().loc[test_tiles]


def make_grid(
    outline: gpd.GeoDataFrame, grid_buffer: int | float = 250
) -> gpd.GeoDataFrame:
    """Create the analysis grid.

    The output is a GeoDataFrame which contains an approximately
    2-km buffer around the land-water boundary of the area of interest.

    Args:
        outline: A thin band along the coast, as returned by
            `_coastline_outline()`, with any land borders removed.
        grid_buffer: An amount in meters by which to buffer each grid cell.
            Doing this maintains consistency in coastlines across tile
            bounds.

    Returns:
        The analysis grid. It is indexed by the DE Pacific grid index
        (see :func:`dep_tools.grids.grid` ) in a (column, row) format.
    """
    # Buffer the country boundaries enough to be sure to capture the coastline.
    # Approx 2km at equator, other CRSes did not work (horizontal stripes that
    # spanned the globe).
    # Note this will give a warning about geographic buffers being incorrect,
    # but I checked and this is fine
    buffer_distance_dd = 0.018018018018018018
    tiny_buffer = 0.0001
    coast_buffer = make_valid(
        outline.buffer(buffer_distance_dd - tiny_buffer)
        .to_crs(PACIFIC_EPSG)
        .unary_union
    )

    full_grid = cast(gpd.GeoSeries, grid(return_type="GeoSeries"))
    # limit to cells which overlap the outline (this prevents including cells
    # that are just buffer)
    full_grid = full_grid.iloc[
        full_grid.to_crs(4326).sindex.query(
            outline.geometry.iloc[0], predicate="intersects"
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
    return _assign_crs(cast(gpd.GeoDataFrame, coastline_grid))


# Cached because the raster is opened once per tile. A failed check raises,
# and raised errors are not cached, so it is retried on the next call.
@cache
def _check_on_s3(path: str) -> None:
    """Raise a clear error if a file built by `build()` is missing."""
    if not S3FileSystem(anon=True).exists(path):
        raise FileNotFoundError(
            f"Nothing at s3://{path}. Run `python -m dep_coastlines.grid` "
            f"to build grid version {config.GRID_VERSION}."
        )


def _coastline_outline(aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Turn land polygons into a thin band along their edges.

    Args:
        aoi: The country boundaries.

    Returns:
        A thin outline around each polygon.
    """
    # A buffer of the exterior line created weird interior gaps,
    # (See https://github.com/locationtech/jts/issues/876)
    # so I buffer the polygons by a positive then negative buffer
    # and take the difference. I do a tiny amount here so I can remove
    # the boundary between PNG & Indonesia without pulling out any of
    # western PNG.
    tiny_buffer = 0.0001
    return gpd.GeoDataFrame(
        geometry=aoi.buffer(tiny_buffer).difference(aoi.buffer(-tiny_buffer))
    )


def _remove_png_border(outline: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove the land border between Papua New Guinea and Indonesia.

    It is the only land border in the area. Removing it prevents
    unnecessary processing of areas which don't contain coasts.

    Args:
        outline: The output of `_coastline_outline()`.

    Returns:
        The outline without the border.
    """
    indonesia = gpd.read_file(
        f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_IDN.gpkg"
    )
    indonesia.geometry = indonesia.buffer(0.01)
    return outline.overlay(indonesia, how="difference")


def _assign_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign an appropriate CRS to each grid cell.

    :func:`odc.stac.load` can determine the best output grid for a given
    set of STAC items based on which CRS is most common among them. The
    issue is the most common CRS can vary across years for the same location
    since an area can contain multiple landsat tiles (in different projections)
    and the most common one can differ. Here we determine which UTM zone has the
    greatest area of the buffered gadm for each cell.

    Alternatively, we would need to determine which Landsat tile has the
    most area within the buffered gadm for each cell, then determine the
    projection of that tile. I couldn't find such a crosswalk (a table of
    which projection each landsat scene has).

    Args:
        gdf: The coastlines grid.

    Returns:
        The input with an additional column `"epsg"` which contains the
        EPSG code for the identified "best" UTM zone.
    """
    utm_zones = _utm_zones()

    zone_lookup = (
        gdf.overlay(utm_zones, how="intersection")
        .assign(area=lambda r: r.geometry.area)
        .sort_values("area", ascending=False)
        # drops the second by default, e.g. the lower value(s)
        .drop_duplicates(["row", "column"])
        .set_index(["row", "column"])
    ).epsg

    return cast(
        gpd.GeoDataFrame,
        gdf.set_index(["row", "column"]).join(zone_lookup).reset_index(),
    )


def _utm_zones() -> gpd.GeoDataFrame:
    """Make the 60 UTM zones, with the EPSG code Landsat uses for each.

    Each zone is a 6 degree band of longitude, with zone 1 starting at 180 W
    (see https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system).
    The irregular zones around Norway are left out, since they are outside
    the area of interest. Landsat Collection 2 uses northern hemisphere codes
    (326xx) everywhere, so southern ones (327xx) are not used.
    """
    # Zone 25 ends at 30 W, opposite the center of the output CRS (150 E), so
    # it wraps around the whole map when reprojected. It's in the Atlantic,
    # so it's left out.
    zone_numbers = [zone for zone in range(1, 61) if zone != 25]
    # UTM is only defined from 80 S to 84 N
    zones = [
        box(-180 + (zone - 1) * 6, -80, -180 + zone * 6, 84) for zone in zone_numbers
    ]
    return gpd.GeoDataFrame(
        dict(epsg=[f"326{zone:02d}" for zone in zone_numbers]),
        geometry=zones,
        crs=4326,
    ).to_crs(config.OUTPUT_CRS)


def _full_aoi() -> gpd.GeoDataFrame:
    """The full area of interest for the coastlines product.

    This includes the output of `gadm()` and anything returned
    by `_osm_land_additions()`.
    """
    return pd.concat([gadm(), _osm_land_additions()]).dissolve()[["geometry"]]


def _osm_land_additions() -> gpd.GeoDataFrame:
    """Load land that GADM is missing from the Overture Maps copy of OSM land.

    GADM leaves out some small islands (for example southern Jaluit Atoll and
    Kili Island in the Marshall Islands, see issue #58). Without them, those
    areas fall outside the processing grid and are never mapped. To add
    another area, append it to `areas`.

    The Overture release is pinned in `config.OVERTURE_RELEASE` so a rebuild
    gives the same land each time. Overture only keeps recent releases, so a
    rebuild fails if the pinned one has been removed.

    This function does not accommodate areas that cross the antimeridian.
    """
    # name: (xmin, ymin, xmax, ymax) in EPSG:4326
    areas = {
        "Jaluit Atoll, MHL": (169.40, 5.70, 169.80, 6.35),
        "Kili Island, MHL": (169.08, 5.60, 169.16, 5.68),
    }
    overture_bucket = "overturemaps-us-west-2/release"
    release = config.OVERTURE_RELEASE
    anon_fs = S3FileSystem(anon=True)
    if not anon_fs.exists(f"{overture_bucket}/{release}"):
        raise FileNotFoundError(
            f"Overture release {release} is no longer available. "
            "Update config.OVERTURE_RELEASE to a current release."
        )
    additions = pd.concat(
        [
            # Only the `land` class of the `land` subtype is kept. These are
            # the polygons built from OSM coastlines. Other classes (island,
            # islet, archipelago) overlap them, and reefs are not land.
            gpd.read_parquet(
                f"{overture_bucket}/{release}/theme=base/type=land/",
                filesystem=anon_fs,
                bbox=bounds,
                columns=["geometry"],
                filters=[("subtype", "==", "land"), ("class", "==", "land")],
            ).assign(name=name)
            for name, bounds in areas.items()
        ]
    )
    additions["source"] = f"OpenStreetMap via Overture Maps {release}, ODbL"
    # GeoParquet labels the data OGC:CRS84, which is the same as EPSG:4326 in
    # lon/lat order, but geopandas won't concatenate the two.
    return additions.set_crs(4326, allow_override=True)


def _write_aoi_raster(aoi: gpd.GeoDataFrame) -> Path:
    """Rasterize the area of interest at 30 m and return the local path."""
    local_aoi_raster_path = Path(__file__).parent / f"../data/raw/{aoi_raster_name}"
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
        local_aoi_raster_path,
        gdal.OpenEx(aoi.to_crs(PACIFIC_EPSG).to_json()),
        options=opts,
    )
    return local_aoi_raster_path


def _write_grid(aoi: gpd.GeoDataFrame) -> Path:
    """Build the buffered analysis grid and return the local path."""
    local_grid_path = Path(__file__).parent / f"../data/raw/{buffered_grid_name}"
    outline = _remove_png_border(_coastline_outline(aoi))
    make_grid(outline, 250).to_file(local_grid_path)
    return local_grid_path


if __name__ == "__main__":
    build()
