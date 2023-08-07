from typing import Tuple, Union

from pandas import Timestamp
from retry import retry
import rioxarray as rx
from xarray import DataArray, Dataset


@retry(tries=10, delay=3)
def filter_by_tides(da: DataArray, path: str, row: str, area=None) -> DataArray:
    """Remove out of range tide values from a given dataset."""
    # TODO: add pathrow to dataset?
    # TODO: add kwargs for functions below as needed
    tides_lowres = load_tides(path, row)

    # perhaps should be done in load_tides
    if area is not None:
        tides_lowres = tides_lowres.rio.clip(
            area.to_crs(tides_lowres.rio.crs).geometry, all_touched=True, from_disk=True
        )

    # Filter out times that are not in the tidal data. Basically because I have
    # been caching the times, we may miss very recent readings (like here it is
    # April 10 and I don't have tides for March 30 or April 7 Landsat data.
    #
    # Just be aware if youi rerun that the cutoffs will likely change ever so slightly.
    # Perhaps not enough to change the "quality" data readings though.
    ds = da.sel(time=da.time[da.time.isin(tides_lowres.time)]).to_dataset()

    # Now filter out tide times that are not in the ds
    tides_lowres = tides_lowres.sel(
        time=tides_lowres.time[tides_lowres.time.isin(ds.time)]
    )

    ds["tide_m"] = fill_and_interp(tides_lowres, ds)

    ds = ds.unify_chunks()

    tide_cutoff_min, tide_cutoff_max = tide_cutoffs_dask(
        ds, tides_lowres, tide_centre=0.0
    )

    # obv should not be hardcoded, either fix here or in fill_and_interp
    tide_cutoff_min.chunk(512)
    tide_cutoff_max.chunk(512)

    tide_bool_highres = (ds.tide_m >= tide_cutoff_min) & (ds.tide_m <= tide_cutoff_max)

    # The squeeze is because there is a "variable" dim added when converted to ds
    return ds.where(tide_bool_highres).drop("tide_m").to_array().squeeze(drop=True)


# Retry is here for network issues, if timeout, etc. when running via
# kbatch, it will bring down the whole process.
@retry(tries=10, delay=3)
def load_tides(
    path: str,
    row: str,
    storage_account: str = "deppcpublicstorage",
    dataset_id: str = "tpxo_lowres",
    container_name: str = "output",
) -> DataArray:
    """Loads previously calculated tide data (via src/calculate_tides.py)"""
    da = rx.open_rasterio(
        f"https://{storage_account}.blob.core.windows.net/{container_name}/coastlines/{dataset_id}/{dataset_id}_{path}_{row}.tif",
        chunks=True,
    )
    print(f"hey hey: {da.rio.crs}")

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
    )


def fill_and_interp(xr_to_interp, xr_to_interp_to, na=float("nan")):
    # The deafrica-coastlines code uses rio.reproject_match, but it is not dask
    # enabled. However, we shouldn't need the reprojection, so we can use
    # (the dask enabled) DataArray.interp instead. Note that the default resampler
    # ("linear") is equivalent to bilinear.

    # Interpolate_na is probably the closest to what we need, but it does
    # e.g. smooth across narrow islands where tides may differ on each side.
    # So the question is which method? Nearest makes some sense here, but
    # so does linear.

    return (
        xr_to_interp.rio.write_nodata(na)
        .map_blocks(lambda xr: xr.rio.interpolate_na("nearest"), template=xr_to_interp)
        .interp(
            dict(
                x=xr_to_interp_to.coords["x"].values,
                y=xr_to_interp_to.coords["y"].values,
            ),
        )
    )


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

    tide_cutoff_min = fill_and_interp(tide_cutoff_min, ds)
    tide_cutoff_max = fill_and_interp(tide_cutoff_max, ds)

    return tide_cutoff_min, tide_cutoff_max
