"""Prepares the training data for the cloud model."""

import geopandas as gpd
import pandas as pd
import xarray as xr

from dep_coastlines.io import MosaicLoader
from dep_coastlines.common import coastlineItemPath
from dep_coastlines.config import MOSAIC_DATASET_ID, MOSAIC_VERSION

TRAINING_DATA_FILE = f"data/cloud_model_training_data_{MOSAIC_VERSION}.csv"


def main():
    input_file = f"data/cloud_model_training_points.gpkg"
    training_points = gpd.read_file(input_file)
    data = load_point_values(training_points)
    data.reset_index(drop=True).to_csv(TRAINING_DATA_FILE)


def load_point_values(points: gpd.GeoDataFrame) -> pd.DataFrame:
    """Extract Landsat values at training points.

    Args:
        points: A :class:`geopandas.GeoDataFrame` with columns "code",
            "time", "row", and "column". "code" indicates the name of the
            class represented (e.g. "cloud"), "time" is of the format
            `<year>`, `<year1>/<year2>` `<year1>,<year2>` or "all", representing
            a single year, a range of years, a list of years, or all years.
            A value of "all" is currently converted to "2020/2022".

    Returns:
        The input points with additional columns added.

    """
    return (
        _split_multiyears(_cast_all(points))
        .groupby("time")
        .apply(_pull_data_for_datetime)
    )


def _split_multiyears(
    df: pd.DataFrame, time_column: str = "time", splitter: str = ","
) -> pd.DataFrame:
    """Splits a list of years and/or year ranges into multiple rows."""
    # Some columns were entered as comma separated values, e.g. 1997,1999/2001
    df[time_column] = df[time_column].str.split(splitter)
    df = df.explode(time_column)
    df[time_column] = df[time_column].str.strip()
    return df


def _cast_all(df, time_column="time"):
    """Convert a value of "all" to "2020/2022"."""
    # Some time values were entered as "all", could be _all_ possible years,
    # for now just make it the biggest mosaic
    df.loc[df[time_column] == "all", time_column] = "2020/2022"
    return df


def _pull_data_for_datetime(
    df: pd.DataFrame,
    mosaic_dataset_id: str = MOSAIC_DATASET_ID,
    mosaic_version: str = MOSAIC_VERSION,
) -> pd.DataFrame:
    """Extract dataset values for a points at a single time."""
    itempath = coastlineItemPath(
        dataset_id=mosaic_dataset_id,
        version=mosaic_version,
        time=df.time.iloc[0].replace("/", "_"),
    )
    loader = MosaicLoader(itempath=itempath)

    def _pull_data_for_cell(group):
        ds = loader.load(group.set_index(["column", "row"]))
        print(f"{df.time.iloc[0]}|{group.column.iloc[0]}|{group.row.iloc[0]}")
        return add_image_values(group, ds)

    return df.groupby(["column", "row"]).apply(_pull_data_for_cell)


def add_image_values(pts: gpd.GeoDataFrame, image: xr.Dataset) -> gpd.GeoDataFrame:
    """Add the values of the image at each point location to the input GeoDataFrame"""
    pts_proj = pts.to_crs(image.rio.crs)
    # a DataArray with x & y coords
    pts_da = pts_proj.assign(x=pts_proj.geometry.x, y=pts_proj.geometry.y).to_xarray()

    # get a dataframe or series (for a single point)
    # I subset by .keys() or otherwise the coords get passed as columns, which
    # can wreak havoc since there is sometimes a time coord which clashes with the
    # point columns.
    pt_values_i = (
        image.sel(pts_da[["x", "y"]], method="nearest").squeeze().compute().to_pandas()
    )[image.keys()]

    if isinstance(pt_values_i, pd.Series):
        pt_values_i = pt_values_i.to_frame().transpose()
        pt_values_i.index = pts.index

    return pd.concat([pts, pt_values_i], axis=1).to_crs(4326)


if __name__ == "__main__":
    main()
