import os
from pathlib import Path

from azure.storage.blob import ContainerClient
import geopandas as gpd
from geopandas import GeoDataframe
import numpy as np
from osgeo import gdal
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
    container_client: ContainerClient, dataset: str, year: int, path: str, row: str
) -> None:
    remote_path = f"{dataset}/{year}/{dataset}_{year}_{path}_{row}.tif"
    local_path = f"data/tonga/{dataset}_{year}_{path}_{row}.tif"
    blob_client = container_client.get_blob_client(remote_path)
    if blob_client.exists() and not Path(local_path).exists():
        with open(local_path, "wb") as dst:
            download_stream = blob_client.download_blob()
            dst.write(download_stream.readall())


def contours_preprocess(
    yearly_ds: Dataset, gapfill_ds: Dataset, water_index: str, index_threshold: float
) -> Dataset:
    pass


def download_for_df(df: Dataframe) -> None:
    cc = container_client()
    for ds in ["coastlines", "coastlines-composite", "landsat-mosaic"]:
        for i, row in df.iterrows():
            for year in range(2013, 2024):
                download_blob(cc, ds, year, row.PATH, row.ROW)


def get_coastal_mask(areas: GeoDataframe) -> GeoDataframe:
    land_plus = areas.buffer(100)
    land_minus = areas.buffer(-500)
    return make_valid(land_plus.difference(land_minus).unary_union)


aoi = gpd.read_file("data/aoi_split_by_landsat_pathrow.gpkg")
tonga = aoi[aoi.NAME_0 == "Tonga"].to_crs(8859)

land_zone = get_coastal_mask(tonga)


year_das = []
for year in range(2013, 2024):
    files = [str(f) for f in Path("data/tonga").glob(f"landsat-mosaic_{year}*.tif")]
    vrt_file = f"data/tonga/coastlines_nir_{year}.vrt"
    gdal.BuildVRT(vrt_file, files)
    # bands = dict(mdnwi=1, ndwi=2, awei=3, wofs=4)
    bands = dict(nir08=4)
    da = (
        rx.open_rasterio(vrt_file, chunks=True)
        .rio.write_crs(8859)
        .rio.clip([land_zone])
        .astype("float32")
    )
    da = da.where(da != -32767, np.nan)

    for name, index in bands.items():
        path = f"data/tonga/coastlines_{year}_{name}.gpkg"
        #        subpixel_contours(da.sel(band=index), [0.128]).to_file(path)
        print(path)

    da["time"] = year
    year_das.append(da)

for name, index in bands.items():
    all_years = xr.concat(year_das, "time")
    subpixel_contours(all_years.sel(band=index), [0.128]).to_file(
        f"data/tonga/coastlines_nir_{name}.gpkg"
    )
