from pathlib import Path

import rioxarray as rx
import xarray as xr

from dep_tools.exceptions import EmptyCollectionError
from dep_tools.utils import (
    get_blob_path,
    blob_exists,
)


def load_blob(dataset_id, item_id, prefix, year=None, **kwargs) -> xr.DataArray:
    blob_path = get_blob_path(dataset_id, item_id, prefix, year)
    az_prefix = Path("https://deppcpublicstorage.blob.core.windows.net/output")
    blob_url = az_prefix / blob_path

    if not blob_exists(blob_path):
        raise EmptyCollectionError()

    return rx.open_rasterio(blob_url, **kwargs)
