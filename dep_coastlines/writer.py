from typing import Tuple

from geopandas import GeoDataFrame
from xarray import DataArray

from dep_tools.namers import DepItemPath
from dep_tools.utils import write_to_blob_storage
from dep_tools.writers import Writer


class CompositeWriter(Writer):
    def __init__(self, itempath: DepItemPath, **kwargs):
        self._itempath = itempath
        self.kwargs = kwargs

    def write(
        self, da: DataArray | GeoDataFrame, item_id: str | list, ext: str = ".tif"
    ) -> str | list:
        path = self._itempath.path(item_id, ext=ext)
        write_to_blob_storage(da, path=path, write_args=self.kwargs)
        return path


DaWriter = CompositeWriter


class CoastlineWriter(Writer):
    def __init__(self, itempath: DepItemPath, **kwargs):
        # **kwargs are for e.g. overwrite, which applies to both
        self._rasterWriter = CompositeWriter(itempath, driver="COG", **kwargs)
        self._vectorWriter = CompositeWriter(itempath, driver="GPKG", **kwargs)

    def write(self, daAndgdf: Tuple[DataArray, GeoDataFrame], item_id: str | list):
        self._rasterWriter.write(daAndgdf[0], item_id)
        self._vectorWriter.kwargs["layer"] = f"lines_{item_id}"
        self._vectorWriter.write(daAndgdf[1], item_id, ".gpkg")
