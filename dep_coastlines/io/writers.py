from typing import Tuple

from coastlines.vector import vector_schema
from geopandas import GeoDataFrame
from xarray import Dataset

from dep_tools.aws import write_to_s3
from dep_tools.processors import Processor
from dep_tools.namers import S3ItemPath
from dep_tools.writers import Writer


# class PreprocessWriter(Writer):
#    def __init__(self, pre_processor: Processor, writer: Writer, **kwargs):
#        self._pre_processor = pre_processor
#        self._writer = writer
#        self._kwargs = kwargs
#
#    def write(self, data, item_id):
#        output = self._pre_processor.process(data)
#        return self._writer.write(output, item_id, **self._kwargs)
#
#
# from odc.geo.xr import to_cog
#
#
# class OdcMemoryWriter(Processor):
#    def __init__(self, **kwargs):
#        self._kwargs = kwargs
#
#    def process(self, d):
#        return to_cog(d, **self._kwargs)
#
#
# import boto3
# from botocore.client import BaseClient
# from dep_tools.aws import s3_dump
#
#
# class S3Writer(Writer):
#    def __init__(
#        self,
#        itempath: DepItemPath,
#        bucket: str,
#        client: BaseClient | None = None,
#        **kwargs,
#    ):
#        self._itempath = itempath
#        self._bucket = bucket
#        if client is None:
#            client = boto3.client("s3")
#        self._client = client
#        self._kwargs = kwargs
#
#    def write(self, d, item_id):
#        path = self._itempath.path(item_id, ext=".tif")
#
#        s3_dump(d, bucket=self._bucket, key=path, client=self._client)


class CompositeWriter(Writer):
    def __init__(self, itempath: S3ItemPath, **kwargs):
        self._itempath = itempath
        self._writer = write_to_s3
        self.kwargs = kwargs

    def write(
        self, ds: Dataset | GeoDataFrame, item_id: str | list, ext: str = ".tif"
    ) -> str | list:
        path = self._itempath.path(item_id, ext=ext)
        self._writer(ds, path=path, bucket=self._itempath.bucket, **self.kwargs)
        return path


DaWriter = CompositeWriter


class CoastlineWriter(Writer):
    def __init__(self, itempath, **kwargs):
        # **kwargs are for e.g. overwrite, which applies to both
        self._rasterWriter = CompositeWriter(itempath, use_odc_writer=False, **kwargs)
        self._vectorWriter = CompositeWriter(itempath, driver="GPKG", **kwargs)

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
