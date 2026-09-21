import geopandas as gpd
from pyproj import CRS
from shapely.geometry import box
import xarray as xr

from dep_coastlines.tide_utils import tides_for_area


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


#    def test_tides_for_items(self):
