from coastlines.vector import annual_movements, calculate_regressions
import geopandas as gpd
import pandas as pd
from rasterio.enums import Resampling
import xarray as xr

from dep_coastlines.config import NEXT_GEN_COASTLINES_OUTPUT as coastlines_file
from dep_coastlines.vector import calculate_rates_of_change
from util import load_coastlines_raster_for_geometry


def compare_roc(validation_contours):
    print(validation_contours.set_id.iloc[0])
    aoi = validation_contours.geometry.buffer(
        distance=100, cap_style="flat"
    ).unary_union.buffer(distance=-30)

    shorelines_gdf = gpd.read_file(
        coastlines_file,
        layer="shorelines_annual",
        bbox=aoi.bounds,
        engine="pyogrio",
        use_arrow=True,
    ).clip(aoi)
    # need to sort here or all_time_stats doesn't always get all data
    validation_shorelines = shorelines_gdf[shorelines_gdf.year >= 2017].sort_values(
        by="year"
    )

    water_for_contours = (
        xr.concat(
            [
                load_coastlines_raster_for_geometry(
                    validation_contours, year
                ).assign_coords(year=year)
                for year in validation_contours.year
            ],
            dim="year",
        )
        .to_dataset(name="twndwi")
        .fillna(1)
        .rio.reproject(3832, resolution=5, resamplng=Resampling.bilinear)
    )

    points_gdf = calculate_rates_of_change(
        validation_shorelines,
        water_index=water_for_contours.twndwi,
        water_index_name="twndwi",
        initial_year=2017,
    ).reset_index()

    val_points_gdf = annual_movements(
        points_gdf[["uid", "geometry"]],
        validation_contours.set_index("year"),
        water_for_contours,
        2024,
        "twndwi",
        max_valid_dist=5000,
    )
    common_columns = points_gdf.columns.intersection(val_points_gdf.columns)
    val_points_gdf = val_points_gdf[common_columns]
    val_points_gdf = calculate_regressions(val_points_gdf)

    return pd.concat(
        [
            points_gdf.add_prefix("cl_").set_index("cl_geometry"),
            val_points_gdf.add_prefix("val_").set_index("val_geometry"),
        ],
        axis=1,
    ).assign(
        # TODO: these aren't e.g. "mae" but rather abs error
        mae=lambda x: abs(x.cl_rate_time - x.val_rate_time),
        mse=lambda x: (x.cl_rate_time - x.val_rate_time) ** 2,
    )


def main():
    df = (
        gpd.read_file("data/validation/s2_validation_lines_2.gpkg")
        .to_crs(3832)
        .groupby("set_id")
        .apply(
            lambda lines: compare_roc(
                validation_contours=lines.dissolve(by="year").reset_index()
            )
        )
    )
    gpd.GeoDataFrame(df.reset_index(), geometry="level_1").to_file(
        "data/validation/rates_of_change_validation.gpkg"
    )


if __name__ == "__main__":
    main()
