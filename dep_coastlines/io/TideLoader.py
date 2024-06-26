from pandas import Timestamp
from xarray import DataArray

from dep_tools.loaders import Loader
from dep_tools.namers import DepItemPath


class TideLoader(Loader):
    def __init__(
        self,
        itempath: DepItemPath,
        storage_account: str = "deppcpublicstorage",
        container_name: str = "output",
    ):
        self._itempath = itempath
        self._storage_account = storage_account
        self._container_name = container_name

    def load(self, item_id) -> DataArray:
        da = rx.open_rasterio(
            f"https://{self._storage_account}.blob.core.windows.net/{self._container_name}/{self._itempath.path(item_id)}",
            chunks={"x": 1, "y": 1},
        )
        time_strings = da.attrs["long_name"]
        band_names = (
            # this is the original data type produced by pixel_tides
            [Timestamp(t) for t in time_strings]
            # In case there's only one
            if isinstance(time_strings, Tuple)
            else [Timestamp(time_strings)]
        )

        return (
            da.assign_coords(band=("band", band_names))
            .rename(band="time")
            .drop_duplicates(...)
            .rio.write_nodata(float("nan"))
        )
