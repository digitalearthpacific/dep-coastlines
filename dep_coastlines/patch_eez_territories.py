import os
import geopandas as gpd
from pathlib import Path
from coastlines.vector import region_atttributes
from pyogrio import read_dataframe

eez = read_dataframe(
    "https://pacificdata.org/data/dataset/964dbebf-2f42-414e-bf99-dd7125eedb16/resource/dad3f7b2-a8aa-4584-8bca-a77e16a391fe/download/country_boundary_eez.geojson"
)


def pitch(gdf):
    if "eez_territory_1" in gdf.columns:
        return gdf.rename(columns=dict(eez_territory_2="eez_territory")).drop(
            columns=["eez_territory_1"]
        )
    elif "eez_territory_left" in gdf.columns:
        return gdf.rename(columns=dict(eez_territory_right="eez_territory")).drop(
            columns=["eez_territory_left"]
        )
    return gdf


def patch(gdf):
    breakpoint()
    these_areas = (
        eez.to_crs(3832).clip(gdf.buffer(500).to_crs(3832).total_bounds).to_crs(gdf.crs)
    )
    if "year" in gdf.columns:
        gdf = gdf.set_index("year")
    return region_atttributes(
        gdf,
        these_areas,
        attribute_col="ISO_Ter1",
        rename_col="eez_territory",
    )


for gpkg in Path("data/interim/vector/0-7-0-55").rglob("*.gpkg"):
    print(gpkg)
    output = pitch(gpd.read_file(gpkg))
    os.remove(gpkg)
    output.to_file(gpkg)
