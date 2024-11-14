from pathlib import Path
import requests

from dea_tools.spatial import xr_vectorize
import geopandas as gpd
from pandas import DatetimeIndex
from timezonefinder import TimezoneFinder

from dep_coastlines.validation.util import load_tides, make_tides
from dep_coastlines.grid import buffered_grid as grid

OUTPUT_DIR = Path("data/validation/valid_areas/nrcs_ortho")


def load_metadata(name):
    crs = 4326
    params = {
        "where": "1=1",
        "outFields": "*",
        "outSr": crs,
        "f": "geojson",
    }
    url = f"https://nrcsgeoservices.sc.egov.usda.gov/arcgis/rest/services/ortho_imagery/{name}_metadata/MapServer/0"
    resp = requests.get(f"{url}/query", params=params)
    gjson = resp.json()

    features = gpd.GeoDataFrame.from_features(gjson, crs=crs)
    return features.overlay(grid.to_crs(features.crs).reset_index(), how="intersection")


def set_timestamp(gdf):
    centroid = gdf.geometry.unary_union.centroid
    timezone = TimezoneFinder().timezone_at(lng=centroid.x, lat=centroid.y)
    gdf["time_start"] = (
        DatetimeIndex(gdf.acq_date + " 1030", tz=timezone).tz_convert("UTC")
        # pixel_tides or xarray needs the "tz=blah blach" bit removed
        # from a Timestamp item or makes time a big integer
        .tz_localize(None)
    )
    gdf["time_end"] = (
        DatetimeIndex(gdf.acq_date + " 1130", tz=timezone)
        .tz_convert("UTC")
        .tz_localize(None)
    )
    return gdf


def calculate_valid_areas(gdf):
    tide_cutoff_min_lr, tide_cutoff_max_lr = load_tides(gdf)
    gdf["time"] = gdf.time_start
    this_early_tide = make_tides(gdf.iloc[[0]], tide_cutoff_min_lr.odc.crs)
    gdf["time"] = gdf.time_end
    this_late_tide = make_tides(gdf.iloc[[0]], tide_cutoff_min_lr.odc.crs)
    tide_cutoff_min = tide_cutoff_min_lr.rio.reproject_match(this_early_tide)
    tide_cutoff_max = tide_cutoff_max_lr.rio.reproject_match(this_early_tide)
    valid_areas = (
        (this_early_tide > tide_cutoff_min)
        & (this_early_tide < tide_cutoff_max)
        & (this_late_tide > tide_cutoff_min)
        & (this_late_tide < tide_cutoff_max)
    )
    print(gdf.iloc[0].catalog_id)
    valid_areas_gdf = xr_vectorize(valid_areas.rio.write_crs(this_early_tide.rio.crs))
    valid_areas_gdf = valid_areas_gdf[valid_areas_gdf.attribute == 1]
    return gdf.overlay(valid_areas_gdf.to_crs(gdf.crs), how="intersection")


def process_layer(name):
    m = load_metadata(name)
    m = set_timestamp(m)
    return m.groupby("catalog_id").apply(calculate_valid_areas).reset_index(drop=True)


if __name__ == "__main__":
    names = [
        "american_samoa_vivid_2022_30cm",
        "cnmi_vivid_2022_30cm",
        "guam_vivid_2022_30cm",
        #        "guam_vivid",
        "marshall_islands_vivid2021",
        "marshall_islands_vivid_2023_30cm",
        "micronesia2021",
        "micronesia_vivid_2023_30cm",
        "palau_vivid",
        "palau_vivid_2022_30cm",
    ]
    for name in names:
        output_file = OUTPUT_DIR / f"{name}.gpkg"
        if not output_file.exists():
            output = process_layer(name)
            output.to_file(output_file)
        else:
            print(f"skipping {output_file}")
