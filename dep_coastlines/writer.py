from typing import Tuple

from geopandas import GeoDataFrame
from xarray import Dataset

from dep_tools.namers import DepItemPath
from dep_tools.utils import write_to_blob_storage
from dep_tools.writers import Writer


class CompositeWriter(Writer):
    def __init__(self, itempath: DepItemPath, writer=write_to_blob_storage, **kwargs):
        self._itempath = itempath
        self._writer = writer
        self.kwargs = kwargs

    def write(
        self, ds: Dataset | GeoDataFrame, item_id: str | list, ext: str = ".tif"
    ) -> str | list:
        path = self._itempath.path(item_id, ext=ext)
        self._writer(ds, path=path, write_args=self.kwargs)
        return path


DaWriter = CompositeWriter


class CoastlineWriter(Writer):
    def __init__(self, itempath: DepItemPath, **kwargs):
        # **kwargs are for e.g. overwrite, which applies to both
        self._rasterWriter = CompositeWriter(itempath, driver="COG", **kwargs)
        self._vectorWriter = CompositeWriter(itempath, driver="GPKG", **kwargs)

    def write(self, output: Tuple[Dataset, GeoDataFrame, GeoDataFrame], item_id: str):
        self._rasterWriter.write(output[0], item_id)
        self._vectorWriter.kwargs["layer"] = f"lines_{item_id}"
        self._vectorWriter.write(output[1], item_id, ".gpkg")
        self._vectorWriter.kwargs["layer"] = f"roc_{item_id}"
        self._vectorWriter.write(output[2], item_id, "_roc.gpkg")
