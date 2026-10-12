from eo_tides.model import model_tides
import geopandas as gpd
import pandas as pd
from pyproj import CRS
import pyTMD.io
from shapely.geometry import box
import xarray as xr

from dep_coastlines.config import TIDE_MODEL
from dep_coastlines.tide_utils import tides_for_area

TIDE_DIRECTORY = "data/raw/tidal_models"


class TestTideUtils:
    area = gpd.GeoDataFrame(
        dict(epsg=[32660]), geometry=[box(178, -18, 179, -17)], crs=4326
    )
    datetime = "2021/2022"

    def test_tides_for_area(self):
        tides = tides_for_area(self.area, self.datetime)
        assert isinstance(tides, xr.DataArray)
        assert tides.dims == ("time", "x", "y")
        assert tides.odc.crs == CRS(self.area.epsg[0])

    def test_tide_model_is_ocean_tide(self):
        # The load tide model also runs without error, but is only a few cm
        # (see issue #59)
        model = pyTMD.io.model(TIDE_DIRECTORY, format="netcdf").elevation(TIDE_MODEL)
        assert model.variable == "tide_ocean"

        # Jaluit, where the ocean tide range is well over 1 m
        tides = model_tides(
            x=169.6,
            y=5.9,
            time=pd.date_range("2021-01-01", periods=8, freq="3h"),
            model=TIDE_MODEL,
            directory=TIDE_DIRECTORY,
        )
        assert tides.tide_height.max() - tides.tide_height.min() > 1.0


#    def test_tides_for_items(self):
