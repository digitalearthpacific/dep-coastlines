import geopandas as gpd
import pandas as pd
import pytest
from dep_tools.grids import PACIFIC_EPSG, gadm
from shapely.geometry import box

from dep_coastlines.grid import _coastline_outline, _osm_land_additions, make_grid


@pytest.fixture(scope="module")
def mhl_gadm() -> gpd.GeoDataFrame:
    # Use the same GADM copy the grid is built from
    padm = gadm()
    return padm.loc[padm.GID_0 == "MHL"]


def test_make_grid(mhl_gadm):
    # The Marshall Islands have no land borders, so the PNG border step
    # (and its download) isn't needed
    grid = make_grid(_coastline_outline(mhl_gadm.dissolve()))
    assert not grid.empty
    assert {"column", "row", "id", "tile_id", "epsg"} <= set(grid.columns)
    assert grid.crs == PACIFIC_EPSG
    assert grid.epsg.notna().all()


def test_gadm_excludes_hawaii():
    # Covers the main and northwestern Hawaiian islands. A plain latitude
    # cutoff would also catch the northern Mariana Islands. The OSM
    # additions are only in the Marshall Islands, so GADM alone decides this.
    hawaii = gpd.GeoSeries([box(-179, 18.5, -154, 29)], crs=4326)
    # Compare in the Pacific CRS, so the antimeridian doesn't split shapes
    padm = gadm().to_crs(PACIFIC_EPSG)
    assert not padm.intersects(hawaii.to_crs(PACIFIC_EPSG).iloc[0]).any()


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
