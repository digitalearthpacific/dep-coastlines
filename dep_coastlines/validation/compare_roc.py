from coastlines.vector import annual_movements, calculate_regressions
import geopandas as gpd
import pandas as pd
import plotnine as pn
from rasterio.enums import Resampling
from shapely import box
from sklearn.metrics import r2_score
import xarray as xr

from validate import load_coastlines, load_coastlines_raster_for_geometry


def compare_roc(validation_contours):
    aoi = box(*validation_contours.geometry.total_bounds)
    points_gdf = load_coastlines(
        aoi, layer="rates_of_change", buffer=-20, use_next_gen=True
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

    val_points_gdf = annual_movements(
        points_gdf[["uid", "geometry"]],
        validation_contours.set_index("year"),
        water_for_contours,
        2024,
        "twndwi",
        max_valid_dist=5000,
    )
    common_columns = points_gdf.columns.intersection(val_points_gdf.columns)
    points_gdf = points_gdf[common_columns]
    points_gdf = calculate_regressions(points_gdf)
    val_points_gdf = val_points_gdf[common_columns]
    val_points_gdf = calculate_regressions(val_points_gdf)

    return pd.concat(
        [
            points_gdf.add_prefix("cl_").set_index("cl_geometry"),
            val_points_gdf.add_prefix("val_").set_index("val_geometry"),
        ],
        axis=1,
    ).assign(
        mae=lambda x: abs(x.cl_rate_time - x.val_rate_time),
        rmse=lambda x: ((x.cl_rate_time - x.val_rate_time) ** 2) ** 0.5,
    )


if __name__ == "__main__":
    test_validation_contours = (
        gpd.read_file("data/validation/s2_validation_test_2.gpkg")
        .dissolve(by="year")
        .reset_index()
    )
    d = compare_roc(validation_contours=test_validation_contours)
    gpd.GeoDataFrame(d.reset_index(), geometry="index").to_file("d2.gpkg")
    r2 = r2_score(d.val_rate_time, d.cl_rate_time)
    mae = d.mae.mean()
    rmse = d.rmse.mean()

    plot = (
        pn.ggplot(d, pn.aes(x=d.val_rate_time, y=d.cl_rate_time))
        + pn.geom_point()
        + pn.theme_bw()
        + pn.labs(x="Validation ROC", y="DEP Coastlines ROC")
        + pn.annotate(
            "text", label=f"r2 = {r2:.2f}\nmae={mae:.2f}\nrmse={rmse:.2f}", x=-15, y=25
        )
    )
    plot.save("plot2.png", dpi=300)
