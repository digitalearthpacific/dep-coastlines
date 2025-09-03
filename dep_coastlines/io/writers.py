from typing import Tuple

from coastlines.vector import vector_schema
from geopandas import GeoDataFrame
from xarray import Dataset

from dep_tools.aws import write_to_s3
from dep_tools.namers import S3ItemPath
from dep_tools.writers import Writer


class CoastlineWriter(Writer):
    def __init__(self, itempath, **kwargs):
        # **kwargs are for e.g. overwrite, which applies to both
        self._rasterWriter = _CompositeWriter(itempath, use_odc_writer=False, **kwargs)
        self._vectorWriter = _CompositeWriter(itempath, driver="GPKG", **kwargs)

    def write(
        self, output: Tuple[Dataset, Dataset, GeoDataFrame, GeoDataFrame], item_id: str
    ):
        water_index, mask, contours, rates_of_change = output
        self._rasterWriter.write(water_index, item_id)
        #        self._rasterWriter.write(mask.Predictions, item_id, ext="_mask_prediction.tif")
        #        self._rasterWriter.write(
        #            mask.Probabilities, item_id, ext="_mask_probabilities.tif"
        #        )
        self._vectorWriter.kwargs["layer"] = f"lines_{item_id}"
        self._vectorWriter.kwargs["engine"] = "fiona"
        contour_schema = vector_schema(contours)
        contour_schema["eez_territory"] = "str:3"
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
            # This is too short in the deafrica schema
            roc_schema["outl_time"] = "str:120"
            roc_schema["eez_territory"] = "str:3"
            roc_schema.pop("index", None)
            self._vectorWriter.kwargs["schema"] = dict(
                properties=roc_schema, geometry="Point"
            )

            self._vectorWriter.write(rates_of_change, item_id, "_roc.gpkg")


class _CompositeWriter(Writer):
    def __init__(self, itempath: S3ItemPath, **kwargs):
        self._itempath = itempath
        self._writer = write_to_s3
        self.kwargs = kwargs

    def write(
        self, data: Dataset | GeoDataFrame, item_id: str | list, ext: str = ".tif"
    ) -> str | list:
        path = self._itempath.path(item_id, ext=ext)
        self._writer(data, path=path, bucket=self._itempath.bucket, **self.kwargs)
        return path
