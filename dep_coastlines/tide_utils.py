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


def tides_highres(da: Dataset, item_id, tide_loader: Loader) -> DataArray:
    return (
        tides_lowres(da, item_id, tide_loader)
        .chunk(time=1, x=1, y=1)
        .interp(x=da.x, y=da.y)
        .chunk(x=128, y=128)
    )


@retry(tries=10, delay=3)
def filter_by_tides(ds: Dataset, item_id, tide_loader: Loader, area=None) -> Dataset:
    """Remove out of range tide values from a given dataset."""
    tides = tides_highres(ds, item_id, tide_loader)

    # Just be aware if you rerun that the cutoffs will likely change ever so slightly
    # as new data are added.
    # Perhaps not enough to change the "quality" data readings though.
    tide_cutoff_min, tide_cutoff_max = tide_cutoffs_dask(tides, tide_centre=0.0)
    tide_bool = (tides >= tide_cutoff_min) & (tides <= tide_cutoff_max)
    ds = ds.sel(time=tide_bool.sum(dim=["x", "y"]) > 0)

    # Apply mask, and load in corresponding tide masked data
    return ds.where(tide_bool)


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

    def load(self, item_id):
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
