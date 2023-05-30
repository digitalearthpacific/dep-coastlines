from pathlib import Path

import geopandas as gpd

from dea_tools.spatial import subpixel_contours

from utils import download_files_for_df, load_local_data
from raster_cleaning import get_coastal_mask, contours_preprocess


aoi = gpd.read_file("data/aoi_split_by_landsat_pathrow.gpkg")
tonga = aoi[aoi.NAME_0 == "Tonga"].to_crs(8859)
local_dir = Path("data/tonga")
dataset_ids = ["coastlines"]

start_year = 2014
end_year = 2022
download_files_for_df(tonga, dataset_ids, local_dir, start_year, end_year)
land_zone = get_coastal_mask(tonga)


yearly_ds = load_local_data(
    local_dir, "coastlines", range(start_year, end_year), land_zone
)[["nir08", "count"]]
composite_years = [f"{year-1}_{year+1}" for year in range(start_year, end_year)]
composite_ds = load_local_data(local_dir, "coastlines", composite_years, land_zone)[
    ["nir08", "count"]
]

# Multiply by -1 since the nir08 thresholding is in the opposite direction
# as the index thresholding
yearly_ds["nir08"] = yearly_ds.nir08 * -1
composite_ds["nir08"] = composite_ds.nir08 * -1
water_index = "nir08"
index_threshold = -128.0

composite_ds["year"] = range(start_year, end_year)
combined_ds = contours_preprocess(
    yearly_ds,
    composite_ds,
    water_index=water_index,
    index_threshold=index_threshold,
    mask_temporal=True,
)

combined_ds.rio.to_raster("combined_ds_tm.tif", driver="COG")
subpixel_contours(da=combined_ds, dim="year", z_values=index_threshold).to_file(
    "test.gpkg"
)
