from pathlib import Path
import requests

from dea_tools.spatial import xr_vectorize
import geopandas as gpd
import pandas as pd
from timezonefinder import TimezoneFinder
from xarray import concat

from dep_coastlines.config import CURRENT_COASTLINES_OUTPUT
from dep_coastlines.tide_utils import tide_cutoffs_lr, tides_for_area
from dep_coastlines.validation.util import make_tides
from dep_coastlines.grid import buffered_grid as grid

VALID_AREAS_OUTPUT_DIR = Path("data/validation/valid_areas/nrcs_ortho")
VALIDATION_LINES_OUTPUT_DIR = Path("data/validation/validation_lines/nrcs_ortho")
COASTLINES = gpd.read_file(CURRENT_COASTLINES_OUTPUT)

OVERWRITE = True


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
        pd.DatetimeIndex(gdf.acq_date + " 1030", tz=timezone).tz_convert("UTC")
        # pixel_tides or xarray needs the "tz=blah blach" bit removed
        # from a Timestamp item or makes time a big integer
        .tz_localize(None)
    )
    gdf["time_end"] = (
        pd.DatetimeIndex(gdf.acq_date + " 1130", tz=timezone)
        .tz_convert("UTC")
        .tz_localize(None)
    )
    return gdf


def areas_near_mean_tide(gdf):
    mins = []
    maxes = []
    for _, row in gdf.iterrows():
        agdf = gpd.GeoDataFrame([row], crs=gdf.crs)
        all_time = pd.date_range(start="1984", end="2025", freq="16d").tolist()
        tides = make_tides(agdf, crs=3832, time=all_time)
        amin, amax = tide_cutoffs_lr(tides)
        mins.append(amin.expand_dims(id=1))
        maxes.append(amax.expand_dims(id=1))

    # I assume that areas won't overlap, but we might need to consider that
    tide_cutoff_min_lr = concat(mins, dim="id").mean(dim="id")
    tide_cutoff_max_lr = concat(maxes, dim="id").mean(dim="id")
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
    breakpoint()
    valid_areas_gdf = xr_vectorize(valid_areas.rio.write_crs(this_early_tide.rio.crs))
    valid_areas_gdf = valid_areas_gdf[valid_areas_gdf.attribute == 1]
    return gdf.overlay(valid_areas_gdf.to_crs(gdf.crs), how="intersection")


def areas_with_coastlines(gdf, coastlines=COASTLINES):
    if len(gdf) > 0:
        year = pd.DatetimeIndex(gdf.acq_date)[0].year
        return (
            gpd.sjoin(
                gdf.to_crs(coastlines.crs),
                coastlines[coastlines.year == year],
                how="left",
            )
            .drop_duplicates("catalog_id")
            .to_crs(gdf.crs)
        )

    return gdf


def calculate_valid_areas(gdf):
    valid_areas = areas_near_mean_tide(gdf)
    return areas_with_coastlines(valid_areas)


def process_layer(name):
    m = load_metadata(name)
    m = set_timestamp(m)
    good_row = m[m["catalog_id"] == "1040010084663300"]
    calculate_valid_areas(good_row)
    return m.groupby("catalog_id").apply(calculate_valid_areas).reset_index(drop=True)


if __name__ == "__main__":
    names = [
        # "american_samoa_vivid_2022_30cm",
        # "cnmi_vivid_2022_30cm",
        # "guam_vivid_2022_30cm",
        #        "guam_vivid",
        # "marshall_islands_vivid2021",
        "marshall_islands_vivid_2023_30cm",
        "micronesia2021",
        "micronesia_vivid_2023_30cm",
        "palau_vivid",
        "palau_vivid_2022_30cm",
    ]
    for name in names:
        output_file = VALID_AREAS_OUTPUT_DIR / f"{name}.gpkg"
        if not output_file.exists() or OVERWRITE:
            output = process_layer(name)
            output.to_file(output_file)
            lines_file = VALIDATION_LINES_OUTPUT_DIR / f"{name}.gpkg"
            empty_lines_gdf = gpd.GeoDataFrame(
                columns=["geometry", "digitiser_name"],
                crs=output.crs,
            )
            empty_lines_gdf.to_file(
                lines_file,
                layer=name,
                driver="GPKG",
                schema=dict(
                    geometry="LineString", properties=dict(digitiser_name="str:40")
                ),
                engine="fiona",
            )
        else:
            print(f"skipping {output_file}")
