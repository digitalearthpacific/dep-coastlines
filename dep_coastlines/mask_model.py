import geopandas as gpd
import pandas as pd
import xarray as xr

from dep_coastlines.MosaicLoader import DeluxeMosaicLoader
from dep_tools.namers import DepItemPath
from dep_tools.utils import get_container_client


def split_multiyears(df, time_column="time", splitter=","):
    # Some columns were entered as comma separated values, e.g. 1997,1999/2001
    df[time_column] = df[time_column].str.split(splitter)
    df = df.explode(time_column)
    df[time_column] = df[time_column].str.strip()
    return df


def cast_all(df, time_column="time"):
    # Some time values were entered as "all", could be _all_ possible years,
    # for now just make it the big mosaic
    df.loc[df[time_column] == "all", time_column] = "1999/2023"
    return df


def add_image_values(pts: gpd.GeoDataFrame, image) -> gpd.GeoDataFrame:
    """Add the values of the image at each point location to the input GeoDataFrame"""
    # Get values for each of the image bands at each of the points.
    pts_proj = pts.to_crs(image.rio.crs)
    # a DataArray with x & y coords
    pts_da = pts_proj.assign(x=pts_proj.geometry.x, y=pts_proj.geometry.y).to_xarray()

    # a dataframe or series (for a single point)
    pt_values_i = (
        image.sel(pts_da[["x", "y"]], method="nearest").squeeze().compute().to_pandas()
    )

    if isinstance(pt_values_i, pd.Series):
        pt_values_i = pt_values_i.to_frame().transpose()
        pt_values_i.index = pts.index

    output = pd.concat([pts, pt_values_i], axis=1).to_crs(4326)
    return output


def pull_data_for_datetime(df):
    print(df)
    itempath = DepItemPath(
        sensor="ls",
        dataset_id="coastlines/mosaics-corrected",
        version="0.6.0",
        time=df.time.iloc[0].replace("/", "_"),
        zero_pad_numbers=True,
    )
    loader = DeluxeMosaicLoader(itempath=itempath)

    def _pull_data_for_cell(group):
        ds = loader.load(group.set_index(["row", "column"]))
        return add_image_values(group, ds)

    output = df.groupby(["row", "column"]).apply(_pull_data_for_cell)
    return output


def prep_training_data():
    (
        cast_all(split_multiyears(gpd.read_file("data/training_data.gpkg")))
        .groupby("time")
        .apply(pull_data_for_datetime)
        .to_csv("data/training_data_with_features_1Feb2024.csv", index=False)
    )


if __name__ == "__main__":
    prep_training_data()
