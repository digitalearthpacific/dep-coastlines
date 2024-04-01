from typing import Tuple

from coastlines.vector import vector_schema
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
        water_index, contours, rates_of_change = output
        self._rasterWriter.write(water_index, item_id)
        self._vectorWriter.kwargs["layer"] = f"lines_{item_id}"
        contour_schema = vector_schema(contours)
        # Not sure why they reset the index in vector_schema, but we don't need this
        contour_schema.pop("index", None)
        self._vectorWriter.kwargs["schema"] = dict(
            properties=contour_schema,
            geometry=["MultiLineString", "LineString"],
        )
        self._vectorWriter.write(
            contours,
            item_id,
            ".gpkg",
        )
        if rates_of_change is not None:
            self._vectorWriter.kwargs["layer"] = f"roc_{item_id}"
            roc_schema = vector_schema(rates_of_change)
            roc_schema.pop("index", None)
            self._vectorWriter.kwargs["schema"] = dict(
                properties=roc_schema, geometry="Point"
            )
            self._vectorWriter.write(rates_of_change, item_id, "_roc.gpkg")
