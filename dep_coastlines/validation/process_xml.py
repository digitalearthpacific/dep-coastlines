from pathlib import Path
from xml.etree import ElementTree as ET

from dea_tools.coastal import pixel_tides
import geopandas as gpd
import numpy as np
from odc.geo import Resolution
from odc.geo.geom import BoundingBox, box, Geometry
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros
import xarray as xr

from dep_tools.namers import DepItemPath
from dep_coastlines.tide_utils import filter_by_tides, TideLoader


def _get_da(record, crs):
    resolution = Resolution(30, 30)

    geobox = GeoBox.from_geopolygon(
        Geometry(record.geometry, crs=3832).to_crs(crs), resolution=resolution
    )
    return xr_zeros(
        geobox,
        chunks=(1000, 1000),
    ).expand_dims(time=[record["time"]])


def make_tides(record, crs):
    da = _get_da(record, crs)
    return pixel_tides(
        da,
        resample=False,
        model="TPXO9-atlas-v5",
        directory="../coastlines-local/tidal-models/",
        resolution=30,
        parallel=False,
        extrapolate=False,  # greatly speeds up tide calcs
    )


def prep_xml():
    records = []
    for xml_file in Path("dep_coastlines/validation/imagery_xml").glob("*"):
        root = ET.parse(xml_file).getroot()
        mpp = {e.tag: e.text for e in root.findall("./IMD/MAP_PROJECTED_PRODUCT/")}
        if not "EARLIESTACQTIME" in mpp or not "MAPZONE" in mpp:
            print(xml_file)  # 6 files
            continue

        record = dict(
            image_id=root.findall("./IMD/PRODUCTORDERID")[0].text,
            path=str(xml_file),
            time=np.datetime64(mpp["EARLIESTACQTIME"]),
            datum=mpp["DATUMNAME"],
            proj=mpp["MAPPROJNAME"],
            utm_zone=mpp["MAPZONE"],
            utm_hemi=mpp["MAPHEMI"],
            xmin=float(mpp["ULX"]),
            ymin=float(mpp["LLY"]),
            xmax=float(mpp["URX"]),
            ymax=float(mpp["ULY"]),
            res_x=float(mpp["COLSPACING"]),
            res_y=float(mpp["ROWSPACING"]),
        )

        proj_string = f"+proj=utm +datum=WGS84 +zone={record['utm_zone']}"
        if record["utm_hemi"] == "S":
            proj_string += " +south"

        record["geometry"] = (
            box(
                record["xmin"],
                record["ymin"],
                record["xmax"],
                record["ymax"],
                crs=proj_string,
            )
            .to_crs(3832)
            .geom
        )

        records.append(record)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=3832)


def load_tides(gdf):
    tide_namer = DepItemPath(
        sensor="ls",
        dataset_id="coastlines/tpxo9",
        version="0.7.0",
        time="1984_2023",
        zero_pad_numbers=True,
    )
    tide_loader = TideLoader(tide_namer)
    from dep_coastlines.tide_utils import tide_cutoffs_dask

    mins = []
    maxes = []
    for _, row in gdf.iterrows():
        amin, amax = tide_cutoffs_dask(tide_loader.load((row.row, row.column)))
        mins.append(amin.expand_dims(id=1))
        maxes.append(amax.expand_dims(id=1))

    return xr.concat(mins, dim="id").mean(dim="id"), xr.concat(maxes, dim="id").mean(
        dim="id"
    )


def calculate_valid_areas(gdf):
    # 1 load existing tides
    tide_cutoff_min_lr, tide_cutoff_max_lr = load_tides(gdf)
    # 2 calculate tides for here
    this_tide = make_tides(gdf.iloc[0], tide_cutoff_min_lr.odc.crs)
    # 3 see if within cutoff and where
    valid_areas = (this_tide > tide_cutoff_min_lr) & (this_tide < tide_cutoff_max_lr)
    valid_areas.load()
    # 4 write output

    if valid_areas.values.any():
        fpath = f"data/validation/valid_areas_{gdf.iloc[0]['image_id']}_8May_1.tif"
        # 6 return 1/0 if any valid places
        valid_areas.rio.write_crs(this_tide.rio.crs).astype(int).rio.to_raster(fpath)
    # 7 make / find transect locations


if __name__ == "__main__":
    image_boxes = prep_xml()
    coastline_grid = gpd.read_file("data/coastline_grid.gpkg")
    coastal_areas_in_boxes = image_boxes.overlay(coastline_grid, how="intersection")
    coastal_areas_in_boxes.groupby("image_id").apply(calculate_valid_areas)
    breakpoint()
