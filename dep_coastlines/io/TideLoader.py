from pandas import Timestamp
import rioxarray as rx
from typing import Tuple
from xarray import DataArray

from dep_tools.loaders import Loader
from dep_tools.namers import DepItemPath

from dep_coastlines.common import TIDES_NAMER
from dep_coastlines.config import HTTPS_PREFIX


class TideLoader(Loader):
    def __init__(
        self, itempath: DepItemPath = TIDES_NAMER, https_prefix: str = HTTPS_PREFIX
    ):
        self._itempath = itempath
        self._https_prefix = https_prefix

    def load(self, item_id) -> DataArray:
        da = rx.open_rasterio(
            f"{self._https_prefix}/{self._itempath.path(item_id)}",
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
