from typing import Tuple, Union

from pandas import Timestamp
from retry import retry
import rioxarray as rx
from xarray import DataArray, Dataset

def filter_by_tides(ds: Dataset, path: str, row:str) -> Dataset:
    """Remove out of range tide values from a given dataset."""
    # TODO: add pathrow to dataset?
    # TODO: add kwargs for functions below as needed

    tides_lowres = load_tides(path, row)
    # Filter out times that are not in the tidal data. Basically because I have
    # been caching the times, we may miss very recent readings (like here it is
    # April 10 and I don't have tides for March 30 or April 7 Landsat data.
    ds = ds.sel(
        time=ds.time[ds.time.isin(tides_lowres.time)]
    )

    # Now filter out tide times that are not in the ds
    tides_lowres = tides_lowres.sel(
        time=tides_lowres.time[tides_lowres.time.isin(ds.time)]
    )

    # The deafrica-coastlines code uses rio.reproject_match, but it is not dask
    # enabled. However, we shouldn't need the reprojection, so we can use
    # (the dask enabled) DataArray.interp instead. Note that the default resampler
    # ("linear") is equivalent to bilinear.
    ds["tide_m"] = tides_lowres.interp(
        dict(x=ds.coords["x"].values, y=ds.coords["y"].values)
    )
    ds = ds.unify_chunks()

    tide_cutoff_min, tide_cutoff_max = tide_cutoffs_dask(
        ds, tides_lowres, tide_centre=0.0
    )

    return filter_by_cutoffs(
        ds, tides_lowres, tide_cutoff_min, tide_cutoff_max
    ).drop("tide_m")


def filter_by_cutoffs(
    ds: Dataset,
    tides_lowres: DataArray,
    tide_cutoff_min: Union[int, float, DataArray],
    tide_cutoff_max: Union[int, float, DataArray],
) -> Dataset:
    """
    coastline.raster.load_tidal_subset that doesn't load
    """
    # Determine what pixels were acquired in selected tide range, and
    # drop time-steps without any relevant pixels to reduce data to load
    tide_bool = (ds.tide_m >= tide_cutoff_min) & (ds.tide_m <= tide_cutoff_max)
    # Changing this to use the lowres tides, since it's causing some memory spikes
    tide_bool = (tides_lowres >= tide_cutoff_min) & (tides_lowres <= tide_cutoff_max)

    # This step loads tide_bool in memory so if you are getting memory spikes,
    # or if you have overwrite=False and you're trying to fill in some missing
    # outputs and it's taking a while, this is probably the reason.
    ds = ds.sel(time=tide_bool.sum(dim=["x", "y"]) > 0)

    # Apply mask to high res data
    tide_bool_highres = (ds.tide_m >= tide_cutoff_min) & (ds.tide_m <= tide_cutoff_max)
    return ds.where(tide_bool_highres)


# Retry is here for network issues, if timeout, etc. when running via
# kbatch, it will bring down the whole process.
@retry(tries=50, delay=10)
def load_tides(
    path:str,
    row:str,
    storage_account: str = "deppcpublicstorage",
    dataset_id: str = "tpxo_lowres",
    container_name: str = "output",
) -> DataArray:
    """Loads previously calculated tide data (via src/calculate_tides.py)"""
    da = rx.open_rasterio(
        f"https://{storage_account}.blob.core.windows.net/{container_name}/{dataset_id}/{dataset_id}_{path}_{row}.tif",
        chunks=True,
    )

    time_strings = da.attrs["long_name"]
    band_names = (
        # this is the original data type produced by pixel_tides
        [Timestamp(t) for t in time_strings]
        # In case there's only one
        if isinstance(time_strings, Tuple)
        else [Timestamp(time_strings)]
    )

    return da.assign_coords(band=band_names).rename(band="time").drop_duplicates(...)


def tide_cutoffs_dask(
    ds: Dataset, tides_lowres: DataArray, tide_centre=0.0, resampling="linear"
) -> Tuple[DataArray, DataArray]:
    """A replacement for coastlines.tide_cutoffs that is dask enabled"""
    # Calculate min and max tides
    tide_min = tides_lowres.min(dim="time")
    tide_max = tides_lowres.max(dim="time")

    # Identify cutoffs
    tide_cutoff_buffer = (tide_max - tide_min) * 0.25
    tide_cutoff_min = tide_centre - tide_cutoff_buffer
    tide_cutoff_max = tide_centre + tide_cutoff_buffer

    chunks = dict(x=ds.chunks["x"], y=ds.chunks["y"])

    # Reproject into original geobox
    tide_cutoff_min = tide_cutoff_min.interp(
        x=ds.coords["x"].values, y=ds.coords["y"].values, method=resampling
    ).chunk(chunks)

    tide_cutoff_max = tide_cutoff_max.interp(
        x=ds.coords["x"].values, y=ds.coords["y"].values, method=resampling
    ).chunk(chunks)

    return tide_cutoff_min, tide_cutoff_max
