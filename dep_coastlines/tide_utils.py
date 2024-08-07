from typing import Tuple

from retry import retry
from xarray import DataArray, Dataset

from dep_tools.loaders import Loader


@retry(tries=10, delay=3)
def filter_by_tides(ds: Dataset, item_id, tide_loader: Loader, area=None) -> Dataset:
    """Remove out of range tide values from a given dataset."""
    tides_lr = tide_loader.load(item_id)

    tide_cutoff_min, tide_cutoff_max = tide_cutoffs_dask(tides_lr, tide_centre=0.0)

    tides_lr = tides_lr.sel(time=ds.time[ds.time.isin(tides_lr.time)])

    # Now filter out tide times that are not in the ds
    tides_lr = tides_lr.sel(time=tides_lr.time[tides_lr.time.isin(ds.time)])

    tide_bool_lr = (tides_lr >= tide_cutoff_min) & (tides_lr <= tide_cutoff_max)

    ds = ds.sel(time=ds.time[ds.time.isin(tides_lr.time)])
    # Filter to times that have _any_ tides within the range.
    # (this will load lr data into memory)
    ds = ds.sel(time=tide_bool_lr.sum(dim=["x", "y"]) > 0)

    # Filter tides again, now that there are fewer times
    tides_lr = tides_lr.sel(time=tides_lr.time[tides_lr.time.isin(ds.time)])

    tide_cutoff_min_hr = tide_cutoff_min.interp(x=ds.x, y=ds.y)
    tide_cutoff_max_hr = tide_cutoff_max.interp(x=ds.x, y=ds.y)

    tides_hr = tides_lr.chunk(time=1).interp(x=ds.x, y=ds.y)

    # This will load cutoff arrays into memory
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
