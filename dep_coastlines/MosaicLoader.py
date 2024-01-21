from azure.storage.blob import ContainerClient
import rioxarray as rx
import xarray as xr

from dep_tools.azure import list_blob_container
from dep_tools.loaders import Loader
from dep_tools.namers import DepItemPath


# see the old utils.py for improvements in here


class MosaicLoader(Loader):
    def __init__(self, itempath: DepItemPath, container_client: ContainerClient):
        self._itempath = itempath
        self._container_client = container_client

    def load(self, item_id):
        dss = [
            _load_single_path(path)
            for path in list_blob_container(
                self._container_client,
                prefix=self._itempath._folder(item_id)
                + "/",  # Add trailing slash or we get e.g. 1999_2001
                suffix=".tif",
            )
        ]

        # Necessary I think b/c we must have switched writers
        return xr.merge([ds.rio.reproject_match(dss[0]) for ds in dss])


def _load_single_path(path) -> xr.Dataset:
    prefix = "/home/jesse/Projects/D4D/dep-coastlines/data/"

    ds = rx.open_rasterio(prefix + path, chunks=True, masked=True)
    return ds.squeeze().to_dataset(name=ds.attrs["long_name"])
