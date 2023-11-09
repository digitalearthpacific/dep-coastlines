from typing import Tuple

import dask
import numpy as np
from pandas import Timestamp
from retry import retry
import rioxarray as rx
from scipy.interpolate import griddata
from xarray import DataArray, concat, apply_ufunc


from rasterio.fill import fillnodata


def tides_highres(da: DataArray, item_id, area=None) -> DataArray:
    tides_lowres = load_tides(item_id)
    da = da.sel(time=da.time[da.time.isin(tides_lowres.time)])

    # Now filter out tide times that are not in the ds
    tides_lowres = tides_lowres.sel(
        time=tides_lowres.time[tides_lowres.time.isin(da.time)]
    ).chunk(dict(x=-1, y=-1, time=1))

    tides_lowres = apply_ufunc(
        lambda da: fillnodata(da, ~np.isnan(da), max_search_distance=2),
        tides_lowres,
        dask="parallelized",
    )

    return tides_lowres.chunk(time=1, x=1, y=1).interp(x=da.x, y=da.y)


@retry(tries=10, delay=3)
def filter_by_tides(da: DataArray, item_id, area=None) -> DataArray:
    """Remove out of range tide values from a given dataset."""
    tides_lowres = load_tides(item_id)

    # Just be aware if you rerun that the cutoffs will likely change ever so slightly
    # as new data are added.
    # Perhaps not enough to change the "quality" data readings though.
    tide_cutoff_min, tide_cutoff_max = tide_cutoffs_dask(tides_lowres, tide_centre=0.0)

    # Filter out times that are not in the tidal data. Basically because I have
    # been caching the times, we may miss very recent readings (like here it is
    # April 10 and I don't have tides for March 30 or April 7 Landsat data.
    ds = da.sel(time=da.time[da.time.isin(tides_lowres.time)]).to_dataset()

    # Now filter out tide times that are not in the ds
    tides_lowres = tides_lowres.sel(
        time=tides_lowres.time[tides_lowres.time.isin(ds.time)]
    ).chunk(dict(x=1, y=1))

    sm = tides_lowres.interp(x=ds.x, y=ds.y)
    ds["tide_m"] = tides_lowres.interp(x=ds.x, y=ds.y)
    return ds
    from rasterio.fill import fillnodata

    # memory spike from this
    values = tides_lowres.to_numpy()
    mask = ~np.isnan(values)
    tides_lowres.values = fillnodata(values, mask)

    tide_bool = (tides_lowres >= tide_cutoff_min) & (tides_lowres <= tide_cutoff_max)

    valid_count_by_time = tide_bool.sum(dim=["x", "y"]).values
    ds = ds.sel(time=valid_count_by_time > 0)

    return ds.to_array().squeeze(drop=True)

    total_cells = len(tides_lowres.x) * len(tides_lowres.y)
    times_to_test = (valid_count_by_time > 0) & (valid_count_by_time < total_cells)

    ds_to_test = ds.sel(time=times_to_test)

    if len(ds_to_test.time) > 0:
        all_good_ds = ds.sel(time=valid_count_by_time == total_cells)
        tide_test = tides_lowres.sel(time=times_to_test)
        tide_highres = tide_test.interp(dict(x=ds.x.values, y=ds.y.values))
        tide_min_highres = tide_cutoff_min.interp(dict(x=ds.x.values, y=ds.y.values))
        tide_max_highres = tide_cutoff_max.interp(dict(x=ds.x.values, y=ds.y.values))

        tide_bool_highres = (tide_highres > tide_min_highres) & (
            tide_highres < tide_max_highres
        )
        test_ds = ds_to_test.where(tide_bool_highres)
        ds = concat([all_good_ds, test_ds], dim="time")
    else:
        ds = ds.sel(time=valid_count_by_time > 0)

    return ds.to_array().squeeze(drop=True)


#    # interp_to = get_mask(area, ds.isel(time=0))
#    a_band = ds.isel(time=0).to_array().squeeze()
#    # effectively replaces all values with constant
#    unmasked_band = a_band.where(0, 1)
#    interp_to = unmasked_band.rio.clip(area.to_crs(unmasked_band.rio.crs).geometry)
#    #    ds["tide_m"] = fill_and_interp(tides_lowres, interp_to)
#    #    tide_cutoff_min = fill_and_interp(tide_cutoff_min, interp_to)
#    #    tide_cutoff_max = fill_and_interp(tide_cutoff_max, interp_to)
#    #
#    #    tide_bool_highres = (ds.tide_m >= tide_cutoff_min) & (ds.tide_m <= tide_cutoff_max)
#    tide_bool_highres = fill_and_interp(tides_lowres, interp_to).chunk(
#        time=1, x=4096, band=4096
#    )
#
#    return ds.where(tide_bool_highres).to_array().squeeze(drop=True)


def fill_and_interp(xr_to_interp, xr_to_interp_to):
    # The deafrica-coastlines code uses rio.reproject_match, but it is not dask
    # enabled. However, we shouldn't need the reprojection, so we can use
    # (the dask enabled) DataArray.interp instead. Note that the default resampler
    # ("linear") is equivalent to bilinear.

    # Interpolate_na is probably the closest to what we need, but it does
    # e.g. smooth across narrow islands where tides may differ on each side.
    # So the question is which method? Nearest makes some sense here, but
    # so does linear.

    # this is slow. dilate? rasterio.fill.fillnodata?
    filled_input = xr_to_interp

    from rasterio.fill import fillnodata

    # memory spike from this
    values = filled_input.to_numpy()
    mask = ~np.isnan(values)
    filled_input.values = fillnodata(values, mask, 5)

    #    filled_input = xr_to_interp.rio.write_nodata(float("nan")).rio.interpolate_na(
    #        "nearest"
    #    )
    xr_to_interp_1d = filled_input.stack(z=("y", "x"))
    points = xr_to_interp_1d.z.to_numpy().tolist()
    values = np.transpose(xr_to_interp_1d.values.tolist())

    xr_to_interp_to_1d = xr_to_interp_to.stack(z=("y", "x"))

    xr_to_interp_to_1d_sparse = xr_to_interp_to_1d.dropna("z")

    new_points = xr_to_interp_to_1d_sparse.z.to_numpy().tolist()
    new_data = griddata(points, values, new_points)

    output_sparse = xr_to_interp_to_1d_sparse.expand_dims(time=xr_to_interp.time)
    output_sparse.values = new_data.transpose()

    tide_min_sparse = output_sparse.min(dim="time")
    tide_max_sparse = output_sparse.max(dim="time")

    tide_cutoff_buffer = (tide_max_sparse - tide_min_sparse) * 0.25
    tide_cutoff_min = 0 - tide_cutoff_buffer
    tide_cutoff_max = 0 + tide_cutoff_buffer

    output_bool_sparse = (output_sparse >= tide_cutoff_min) & (
        output_sparse <= tide_cutoff_max
    )

    output_bool_sparse = output_bool_sparse.unstack(fill_value=False)

    #    output_full = (
    #        xr_to_interp_to_1d.expand_dims(time=xr_to_interp.time)
    #        .astype(bool)
    #        .unstack(fill_value=False)
    #    )

    #    breakpoint()
    #    _, output_mask = align(
    #        output_full, output_bool_sparse, join="left", fill_value=False
    #    )
    return output_bool_sparse


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


@retry(tries=10, delay=3)
def load_tides(
    item_id,
    storage_account: str = "deppcpublicstorage",
    dataset_id: str = "tpxo_lowres",
    container_name: str = "output",
) -> DataArray:
    """Loads previously calculated tide data (via src/calculate_tides.py)"""
    suffix = "_".join([str(i) for i in item_id])
    da = rx.open_rasterio(
        f"https://{storage_account}.blob.core.windows.net/{container_name}/coastlines/{dataset_id}/{dataset_id}_{suffix}.tif",
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
