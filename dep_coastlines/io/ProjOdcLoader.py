import geopandas as gpd
from pystac import Item, ItemCollection

from dep_tools.loaders import OdcLoader


class ProjOdcLoader(OdcLoader):
    """An OdcLoader which allows custom projections for each load. An integer
    EPSG code must be available in the 'EPSG' column of the supplied area.
    
    For context, see 
    See https://github.com/digitalearthpacific/dep-coastlines/issues/34
    """
    def load(self, items: list[Item] | ItemCollection, areas: gpd.GeoDataFrame):
        if not "epsg" in areas.columns:
            raise ValueError("An `epsg` column must exist in the supplied area and represent a valid integer EPSG code.")
        self._kwargs["crs"] = int(areas.iloc[0].epsg)
        return super().load(items, areas)
