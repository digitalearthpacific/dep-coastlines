import geopandas as gpd
import pandas as pd

from dep_coastlines.config import CURRENT_COASTLINES_OUTPUT


def add_nearest_IslandID(roc, atoll_lines):
    nearest_lines = atoll_lines.sindex.nearest(
        roc.geometry, max_distance=1000, return_distance=True
    )
    points_within_distance = roc.iloc[nearest_lines[0][0, :], :].reset_index()
    points_within_distance.loc[:, "distance"] = nearest_lines[1]
    points_within_distance.loc[:, "IslandID"] = (
        atoll_lines.iloc[nearest_lines[0][1, :], :].reset_index().IslandID
    )
    return points_within_distance


def merge_dep_and_kenchel_roc():
    roc = gpd.read_file(
        CURRENT_COASTLINES_OUTPUT, layer="rates_of_change", engine="pyogrio"
    )
    atoll_lines = gpd.read_file("data/validation/Kenchetal2024_COMMSENV.zip").to_crs(
        3832
    )
    points_near_island = add_nearest_IslandID(roc, atoll_lines)
    island_roc_summary = points_near_island.groupby("IslandID").mean("rate_time")

    atoll_summary_from_paper = pd.read_excel(
        "data/validation/Pacific Atoll Island Change Analysis Dataset_2024.xlsx"
        # Not all these are the same, FYI
    ).set_index("Atoll Island ID")
    atoll_summary_from_paper["mean_rate_paper"] = (
        atoll_summary_from_paper["No. of Accretion Transects"]
        * atoll_summary_from_paper["Average Accretion (m/yr)"]
        + atoll_summary_from_paper["No. of Erosion Transects"]
        * atoll_summary_from_paper["Average Erosion (m/yr)"]
        + atoll_summary_from_paper["No. of Transects With No Change"]
        * atoll_summary_from_paper["Average of No Change (m/yr)"]
    ) / atoll_summary_from_paper["No. of Shoreline Transcts Analysed"]
    return island_roc_summary.join(atoll_summary_from_paper, on="IslandID")


if __name__ == "__main__":
    summary = merge_dep_and_kenchel_roc()
