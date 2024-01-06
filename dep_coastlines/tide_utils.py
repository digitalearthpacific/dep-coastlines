from typing import Tuple

from pandas import Timestamp
from retry import retry
import rioxarray as rx
from xarray import DataArray, Dataset


from dep_tools.loaders import Loader
from dep_tools.namers import DepItemPath


def tides_lowres(da: Dataset, item_id, tide_loader: Loader) -> DataArray:
    tides_lowres = tide_loader.load(item_id)
    da = da.sel(time=da.time[da.time.isin(tides_lowres.time)])

    # Now filter out tide times that are not in the ds
    tides_lowres = tides_lowres.sel(
        time=tides_lowres.time[tides_lowres.time.isin(da.time)]
    ).chunk(dict(x=-1, y=-1, time=1))

    # For some areas, some tidal data does not actual extend into
    # the tidal zone. This fills these values with the nearest value.
    # This is perhaps not the ideal approach (but neither is using
    # 5-km tidal data!), an alternative might be to fill the values
    # with a different tidal dataset, or at least fill / average
    # values among them.
    # tides_lowres = apply_ufunc(
    #    lambda da: fillnodata(da, ~np.isnan(da), max_search_distance=2),
    #    tides_lowres,
    #    dask="parallelized",
    # )
    return tides_lowres


@retry(tries=10, delay=3)
def filter_by_tides(ds: Dataset, item_id, tide_loader: Loader, area=None) -> Dataset:
    """Remove out of range tide values from a given dataset."""
    tides_lr = tide_loader.load(item_id)

    tide_cutoff_min, tide_cutoff_max = tide_cutoffs_dask(tides_lr, tide_centre=0.0)

    tides_lr = tides_lr.sel(time=ds.time[ds.time.isin(tides_lr.time)])

    # Now filter out tide times that are not in the ds
    tides_lr = tides_lr.sel(time=tides_lr.time[tides_lr.time.isin(ds.time)])

    tide_bool_lr = (tides_lr >= tide_cutoff_min) & (tides_lr <= tide_cutoff_max)

    # Filter to times that have _any_ tides within the range
    ds = ds.sel(time=tide_bool_lr.sum(dim=["x", "y"]) > 0)

    # Filter tides again
    tides_lr = tides_lr.sel(time=tides_lr.time[tides_lr.time.isin(ds.time)])

    tide_cutoff_min_hr = tide_cutoff_min.interp(x=ds.x, y=ds.y)
    tide_cutoff_max_hr = tide_cutoff_max.interp(x=ds.x, y=ds.y)

    tides_hr = tides_lr.chunk(time=1).interp(x=ds.x, y=ds.y)

    tide_bool_hr = (tides_hr >= tide_cutoff_min_hr) & (tides_hr <= tide_cutoff_max_hr)

    # Apply mask, and load in corresponding tide masked data
    return ds.where(tide_bool_hr)


def tide_cutoffs_dask(
    tides_lowres: DataArray, tide_centre=0.0
) -> Tuple[DataArray, DataArray]:
    """A replacement for coastlines.tide_cutoffs that is dask enabled"""
    # Calculate min and max tides
    tide_min = tides_lowres.min(dim="time")
    tide_max = tides_lowres.max(dim="time")

    # Identify cutoffs
    tide_cutoff_buffer = (tide_max - tide_min) * 0.25
    tide_cutoff_min = tide_centre - tide_cutoff_buffer
    tide_cutoff_max = tide_centre + tide_cutoff_buffer

    return tide_cutoff_min, tide_cutoff_max


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
