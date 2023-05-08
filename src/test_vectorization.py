import os
from pathlib import Path
from typing import Dict, List

from azure.storage.blob import ContainerClient
import geopandas as gpd
from geopandas import GeoDataFrame
import numpy as np
from osgeo import gdal
from pandas import DataFrame
import rioxarray as rx
from shapely import make_valid
import xarray as xr
from xarray import Dataset

from dea_tools.spatial import subpixel_contours


def container_client(
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"],
    container_name: str = "output",
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
) -> ContainerClient:
    return ContainerClient(
        f"https://{storage_account}.blob.core.windows.net",
        container_name=container_name,
        credential=credential,
    )


def download_blob(
    container_client: ContainerClient,
    dataset: str,
    year: int,
    path: str,
    row: str,
    local_dir: Path,
) -> None:
    remote_path = f"{dataset}/{year}/{dataset}_{year}_{path}_{row}.tif"
    local_path = f"{local_dir}/{dataset}_{year}_{path}_{row}.tif"
    blob_client = container_client.get_blob_client(remote_path)
    if blob_client.exists() and not Path(local_path).exists():
        with open(local_path, "wb") as dst:
            download_stream = blob_client.download_blob()
            dst.write(download_stream.readall())


def contours_preprocess(
    yearly_ds: Dataset,
    gapfill_ds: Dataset,
    landsat_ds: Dataset,
) -> Dataset:
    breakpoint()
    # Remove low obs pixels and replace with 3-year gapfill
    yearly_ds = yearly_ds.where(yearly_ds["count"] > 5, gapfill_ds)

    # Set any pixels with only one observation to NaN, as these
    # are extremely vulnerable to noise
    yearly_ds = yearly_ds.where(yearly_ds["count"] > 1)

    # Apply water index threshold and re-apply nodata values
    thresholded_ds = yearly_ds[water_index] < index_threshold
    nodata = yearly_ds[water_index].isnull()
    thresholded_ds = thresholded_ds.where(~nodata)
    pass


def download_for_df(df: DataFrame, dataset_ids: List, local_dir: Path) -> None:
    cc = container_client()
    for ds in dataset_ids:
        for i, row in df.iterrows():
            for year in range(2013, 2024):
                download_blob(cc, ds, year, row.PATH, row.ROW, local_dir)


def get_coastal_mask(areas: GeoDataFrame) -> GeoDataFrame:
    land_plus = areas.buffer(100)
    land_minus = areas.buffer(-500)
    return make_valid(land_plus.difference(land_minus).unary_union)


def load_data(local_dir: str, dataset_ids: List, years, mask: GeoDataFrame) -> Dict:
    # initialize output
    output = {}
    for id in dataset_ids:
        output[id] = []

    for year in years:
        for id in dataset_ids:
            files = [str(f) for f in Path(local_dir).glob(f"{id}_{year}*.tif")]
            vrt_file = f"{local_dir}/{id}_{year}.vrt"
            # gdal.BuildVRT(vrt_file, files)

            # BuildVRT doesn't store band names (as far as I can tell)
            # so let's get them from the first file
            test_da = rx.open_rasterio(files[0], chunks=True)
            band_names = list(test_da.attrs["long_name"])
            da = (
                rx.open_rasterio(vrt_file, chunks=True)
                .assign_coords(band=band_names)
                .rio.write_crs(8859)
                .rio.clip([mask], from_disk=True)
                .astype("float32")
            )
            da = da.where(da != -32767, np.nan)

            da["time"] = year
            output[id].append(da.to_dataset("band"))

            # for name, index in bands.items():
            #    all_years = xr.concat(year_das, "time")
            #    subpixel_contours(all_years.sel(band=index), [0.128]).to_file(
            #        f"data/tonga/coastlines_nir_{name}.gpkg"
            #    )
    return {k: xr.concat(v, dim="time") for k, v in output.items()}


aoi = gpd.read_file("data/aoi_split_by_landsat_pathrow.gpkg")
tonga = aoi[aoi.NAME_0 == "Tonga"].to_crs(8859)
local_dir = "data/tonga"
dataset_ids = ["coastlines", "coastlines-composite", "landsat-mosaic"]

land_zone = get_coastal_mask(tonga)


time_serieses = load_data("data/tonga", dataset_ids, range(2013, 2016), land_zone)

contours_preprocess(*time_serieses.values())
