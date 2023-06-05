from pathlib import Path
from typing import List, Union

from geopandas import GeoDataFrame
from osgeo import gdal
import numpy as np
from pandas import DataFrame
from rasterio.errors import RasterioIOError
from retry import retry
import rioxarray as rx
import xarray as xr

from dep_tools.utils import download_blob, get_blob_path, get_container_client


@retry(tries=20, delay=10)
def load_blobs(dataset_id: str, path, row, years, **kwargs) -> Union[xr.Dataset, None]:
    # Some obvious overlap between this and `load_local_data` that should be
    # cleaned up
    prefix = Path("https://deppcpublicstorage.blob.core.windows.net/output")
    dss = list()
    for year in years:
        blob_path = prefix / get_blob_path(dataset_id, year, path, row)
        try:
            da = rx.open_rasterio(blob_path, **kwargs)
        except RasterioIOError:
            return None

        band_names = list(da.attrs["long_name"])
        da = da.assign_coords(band=band_names).rio.write_crs(8859).astype("float32")
        da = da.where(da != -32767, np.nan)

        da["year"] = year
        dss.append(da.to_dataset("band"))
    return xr.concat(dss, dim="year")


def load_local_data(
    local_dir: Path, dataset_id: str, years, mask: GeoDataFrame
) -> xr.Dataset:
    output = list()
    for year in years:
        files = [str(f) for f in local_dir.glob(f"{dataset_id}_{year}*.tif")]
        vrt_file = local_dir / f"{dataset_id}_{year}.vrt"
        gdal.BuildVRT(str(vrt_file), files)

        # BuildVRT doesn't store band names (as far as I can tell)
        # so let's get them from the first file
        test_da = rx.open_rasterio(files[0], chunks=True)
        band_names = list(test_da.attrs["long_name"])
        da = (
            rx.open_rasterio(vrt_file, chunks=2048)
            .assign_coords(band=band_names)
            .rio.write_crs(8859)
            .rio.clip([mask], from_disk=True)
            .astype("float32")
        )
        da = da.where(da != -32767, np.nan)

        da["year"] = year
        output.append(da.to_dataset("band"))

    return xr.concat(output, dim="year")


def download_files_for_df(
    df: DataFrame, dataset_ids: List, local_dir: Path, start_year: int, end_year: int
) -> None:
    cc = get_container_client()
    for ds in dataset_ids:
        for i, row in df.iterrows():
            for year in range(start_year, end_year + 1):
                download_blob(cc, ds, year, row.PATH, row.ROW, local_dir)
                if year > start_year and year < end_year:
                    download_blob(
                        cc, ds, f"{year-1}_{year+1}", row.PATH, row.ROW, local_dir
                    )
