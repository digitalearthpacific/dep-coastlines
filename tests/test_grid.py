import geopandas as gpd
import pandas as pd
import pytest
from dep_tools.grids import PACIFIC_EPSG, gadm
from shapely.geometry import box

from dep_coastlines.grid import (
    _full_aoi,
    _osm_land_additions,
    buffered_grid,
    debug_grid,
)


def test_buffered_grid_loads():
    grid = buffered_grid()
    assert not grid.empty
    assert grid.index.names == ["column", "row"]
    assert {"id", "tile_id", "epsg"} <= set(grid.columns)
    assert grid.crs == PACIFIC_EPSG
    assert grid.epsg.notna().all()


def test_debug_tiles_are_in_grid():
    # Raises a KeyError if any debugging tile is missing from the grid
    assert not debug_grid().empty


def test_aoi_excludes_hawaii():
    # Covers the main and northwestern Hawaiian islands. A plain latitude
    # cutoff would also catch the northern Mariana Islands.
    hawaii = gpd.GeoSeries([box(-179, 18.5, -154, 29)], crs=4326)
    # Compare in the Pacific CRS so the antimeridian doesn't split shapes
    aoi = _full_aoi().to_crs(PACIFIC_EPSG)
    assert not aoi.intersects(hawaii.to_crs(PACIFIC_EPSG).iloc[0]).any()


@pytest.fixture(scope="module")
def mhl_gadm() -> gpd.GeoDataFrame:
    # Use the same GADM copy the grid is built from
    padm = gadm()
    return padm.loc[padm.GID_0 == "MHL"]


@pytest.fixture(scope="module")
def osm_additions() -> gpd.GeoDataFrame:
    return _osm_land_additions()


@pytest.fixture(scope="module")
def missing_land_points(osm_additions) -> gpd.GeoSeries:
    """One point on Kili Island and one on the largest southern Jaluit islet."""
    kili = osm_additions.loc[osm_additions.name == "Kili Island, MHL"]
    jaluit = osm_additions.loc[osm_additions.name == "Jaluit Atoll, MHL"]
    southern_jaluit = jaluit.loc[jaluit.representative_point().y < 6.0]
    largest_southern = southern_jaluit.loc[
        [southern_jaluit.to_crs(PACIFIC_EPSG).area.idxmax()]
    ]
    return pd.concat([kili, largest_southern]).representative_point()


def test_gadm_is_missing_the_land(mhl_gadm, missing_land_points):
    gadm_land = mhl_gadm.union_all()
    assert not missing_land_points.within(gadm_land).any()


def test_osm_additions_cover_the_land(mhl_gadm, osm_additions, missing_land_points):
    combined_land = pd.concat([mhl_gadm, osm_additions]).union_all()
    assert missing_land_points.within(combined_land).all()
