import io
import os
from pathlib import Path
from typing import Dict, List, Union

import azure.storage.blob
from dask.distributed import Client, Lock
import geopandas as gpd
from geocube.api.core import make_geocube
import numpy as np
from osgeo import gdal
import osgeo_utils.gdal2tiles
import rasterio
import rioxarray
from tqdm import tqdm
import xarray as xr
from xarray import DataArray


def scale_and_offset(
    da: xr.DataArray, scale: List[float] = [1], offset: float = 0
) -> xr.DataArray:
    """Apply the given scale and offset to the given DataArray"""
    return da * scale + offset


def make_geocube_dask(
    df: gpd.GeoDataFrame, measurements: List[str], like: xr.DataArray, **kwargs
):
    """Dask-enabled geocube.make_geocube. Not completely implemented."""

    def rasterize_block(block):
        return (
            make_geocube(df, measurements=measurements, like=block, **kwargs)
            .to_array(measurements[0])
            .assign_coords(block.coords)
        )

    like = like.rename(dict(zip(["band"], measurements)))
    return like.map_blocks(rasterize_block, template=like)


def write_to_blob_storage(
    xr: DataArray,
    path: Union[str, Path],
    write_args: Dict,
    #    output_scale: List = [1.0],
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"],
    container_name: str = "output",
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
) -> None:
    container_client = azure.storage.blob.ContainerClient(
        f"https://{storage_account}.blob.core.windows.net",
        container_name=container_name,
        credential=credential,
    )

    with io.BytesIO() as buffer:
        xr.rio.to_raster(buffer, **write_args)
        buffer.seek(0)
        blob_client = container_client.get_blob_client(path)
        blob_client.upload_blob(buffer, overwrite=True)


def copy_to_blob_storage(
    local_path: Path,
    remote_path: Path,
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"],
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
    container_name: str = "output",
) -> None:
    container_client = azure.storage.blob.ContainerClient(
        f"https://{storage_account}.blob.core.windows.net",
        container_name=container_name,
        credential=credential,
    )

    with open(local_path, "rb") as src:
        blob_client = container_client.get_blob_client(str(remote_path))
        blob_client.upload_blob(src, overwrite=True)


def scale_to_int16(
    da: DataArray, output_multiplier: int, output_nodata: int
) -> DataArray:
    """Multiply the given DataArray by the given multiplier and convert to
    int16 data type, with the given nodata value"""
    return (
        np.multiply(da, output_multiplier)
        .where(da.notnull(), output_nodata)
        .astype("int16")
        .rio.write_nodata(output_nodata)
        .rio.write_crs(da.rio.crs)
    )


def raster_bounds(raster_path: Path) -> List:
    """Returns the bounds for a raster file at the given path"""
    with rasterio.open(raster_path) as t:
        return list(t.bounds)


def gpdf_bounds(gpdf: gpd.GeoDataFrame) -> List[float]:
    """Returns the bounds for the give GeoDataFrame, and makes sure
    it doesn't cross the antimeridian."""
    bbox = gpdf.to_crs("EPSG:4326").bounds.values[0]
    # Or the opposite!
    bbox_crosses_antimeridian = bbox[0] < 0 and bbox[2] > 0
    if bbox_crosses_antimeridian:
        # This may be overkill, but nothing else was really working
        bbox[0] = -179.9999999999
        bbox[2] = 179.9999999999
    return bbox


def build_vrt(
    prefix: str,
    bounds: List,
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"],
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
    container_name: str = "output",
) -> Path:
    container_client = azure.storage.blob.ContainerClient(
        f"https://{storage_account}.blob.core.windows.net",
        container_name=container_name,
        credential=credential,
    )
    blobs = [
        f"/vsiaz/{container_name}/{blob.name}"
        for blob in container_client.list_blobs()
        if blob.name.startswith(prefix)
    ]

    local_prefix = Path(prefix).stem
    vrt_file = f"data/{local_prefix}.vrt"
    gdal.BuildVRT(vrt_file, blobs, outputBounds=bounds)
    return Path(vrt_file)


def create_tiles(
    color_file: str,
    prefix: str,
    bounds: List,
    remake_mosaic: bool = True,
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"],
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
    container_name: str = "output",
):
    if remake_mosaic:
        with Client() as local_client:
            mosaic_scenes(
                prefix=prefix,
                bounds=bounds,
                client=local_client,
                storage_account=storage_account,
                credential=credential,
                container_name=container_name,
                scale_factor=1.0 / 1000,
                overwrite=remake_mosaic,
            )
    dst_vrt_file = f"data/{Path(prefix).stem}_rgb.vrt"
    gdal.DEMProcessing(
        dst_vrt_file,
        str(_mosaic_file(prefix)),
        "color-relief",
        colorFilename=color_file,
        addAlpha=True,
    )
    dst_name = f"data/tiles/{prefix}"
    os.makedirs(dst_name, exist_ok=True)
    max_zoom = 11
    # First arg is just a dummy so the second arg is not removed (see gdal2tiles code)
    # I'm using 512 x 512 tiles so there's fewer files to copy over. likewise
    # for -x
    osgeo_utils.gdal2tiles.main(
        [
            "gdal2tiles.py",
            "--tilesize=512",
            "--processes=4",
            f"--zoom=0-{max_zoom}",
            "-x",
            dst_vrt_file,
            dst_name,
        ]
    )

    for local_path in tqdm(Path(dst_name).rglob("*")):
        if local_path.is_file():
            remote_path = Path("tiles") / "/".join(local_path.parts[4:])
            copy_to_blob_storage(
                local_path, remote_path, storage_account, credential, container_name
            )
            local_path.unlink()


def _local_prefix(prefix: str) -> str:
    return Path(prefix).stem


def _mosaic_file(prefix: str) -> str:
    return f"data/{_local_prefix(prefix)}.tif"


def mosaic_scenes(
    prefix: str,
    bounds: List,
    client: Client,
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"],
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
    container_name: str = "output",
    scale_factor: float = None,
    overwrite: bool = True,
) -> None:

    mosaic_file = _mosaic_file(prefix)
    if not Path(mosaic_file).is_file() or overwrite:
        vrt_file = build_vrt(
            prefix, bounds, storage_account, credential, container_name
        )
        vrt_file = f"data/{_local_prefix(prefix)}.vrt"
        rioxarray.open_rasterio(vrt_file, chunks=True).rio.to_raster(
            mosaic_file,
            compress="LZW",
            predictor=2,
            lock=Lock("rio", client=client),
        )

        if scale_factor is not None:
            with rasterio.open(mosaic_file, "r+") as dst:
                dst.scales = (scale_factor,)
