import geopandas as gpd
import pandas as pd
import pytest

from dep_coastlines.grid import _osm_land_additions


@pytest.fixture(scope="module")
def mhl_gadm() -> gpd.GeoDataFrame:
    return gpd.read_file(
        "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_MHL.gpkg",
        layer="ADM_ADM_0",
    )


@pytest.fixture(scope="module")
def osm_additions() -> gpd.GeoDataFrame:
    return _osm_land_additions()


@pytest.fixture(scope="module")
def missing_land_points(osm_additions) -> gpd.GeoSeries:
    """One point on Kili Island and one on the largest southern Jaluit islet."""
    kili = osm_additions.loc[osm_additions.name == "Kili Island, MHL"]
    jaluit = osm_additions.loc[osm_additions.name == "Jaluit Atoll, MHL"]
    southern_jaluit = jaluit.loc[jaluit.representative_point().y < 6.0]
    largest_southern = southern_jaluit.loc[[southern_jaluit.to_crs(3832).area.idxmax()]]
    return pd.concat([kili, largest_southern]).representative_point()


def test_gadm_is_missing_the_land(mhl_gadm, missing_land_points):
    gadm_land = mhl_gadm.union_all()
    assert not missing_land_points.within(gadm_land).any()


def test_osm_additions_cover_the_land(mhl_gadm, osm_additions, missing_land_points):
    combined_land = pd.concat([mhl_gadm, osm_additions]).union_all()
    assert missing_land_points.within(combined_land).all()
