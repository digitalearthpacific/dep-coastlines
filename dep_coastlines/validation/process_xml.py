from pathlib import Path
from xml.etree import ElementTree as ET

from dea_tools.coastal import pixel_tides
import numpy as np
from odc.geo import Resolution
from odc.geo.geom import BoundingBox
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros
import pandas as pd


def _get_da(record):
    bbox = BoundingBox(
        record["min_x"], record["min_y"], record["max_x"], record["max_y"]
    )
    # All projections are utm, all datums are wgs84. If we process another set,
    # this may need to change
    proj_string = f"+proj=utm +datum=WGS84 +zone={record['utm_zone']}"
    if record["utm_hemi"] == "S":
        proj_string += " +south"

    resolution = Resolution(record["res_x"], record["res_y"])
    da = xr_zeros(
        GeoBox.from_bbox(bbox, proj_string, resolution=resolution), chunks=(1000, 1000)
    )
    breakpoint()


def make_tides(record):
    da = _get_da(record)
    tides = pixel_tides(da, resample=False)
    breakpoint()


records = []
for xml_file in Path("dep_coastlines/validation/imagery_xml").glob("*"):
    mpp = {
        e.tag: e.text
        for e in ET.parse(xml_file).getroot().findall("./IMD/MAP_PROJECTED_PRODUCT/")
    }
    if not "EARLIESTACQTIME" in mpp or not "MAPZONE" in mpp:
        print(xml_file)  # 6 files
        continue

    record = dict(
        path=xml_file,
        time=np.datetime64(mpp["EARLIESTACQTIME"]),
        datum=mpp["DATUMNAME"],
        proj=mpp["MAPPROJNAME"],
        utm_zone=mpp["MAPZONE"],
        utm_hemi=mpp["MAPHEMI"],
        min_x=float(mpp["ULX"]),
        max_x=float(mpp["URX"]),
        min_y=float(mpp["LLY"]),
        max_y=float(mpp["ULY"]),
        res_x=float(mpp["COLSPACING"]),
        res_y=float(mpp["ROWSPACING"]),
    )
    records.append(record)
    make_tides(record)


pd.DataFrame.from_records(records).to_csv("dep_coastlines/validation/xml_summary.csv")
