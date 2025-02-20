from pathlib import Path
from typing import Tuple

from dea_tools.coastal import pixel_tides
from dep_tools.searchers import LandsatPystacSearcher
from odc.geo.geobox import AnchorEnum
from pystac_client import Client as PystacClient
from xarray import DataArray, Dataset

from dep_coastlines.io import ProjOdcLoader
from dep_coastlines.common import use_alternate_s3_href


def tides_for_area(area, datetime="1984/2024", **kwargs):
    client = PystacClient.open(
        "https://landsatlook.usgs.gov/stac-server",
        modifier=use_alternate_s3_href,
    )
    items = LandsatPystacSearcher(
        search_intersecting_pathrows=True,
        datetime=datetime,
        client=client,
        collections=["landsat-c2l2-sr"],
    ).search(area)
    return tides_for_items(items, area, **kwargs)


def tides_for_items(items, area, **kwargs):
    ds = ProjOdcLoader(
        chunks=dict(band=1, time=1, x=8192, y=8192),
        resampling={"qa_pixel": "nearest", "*": "cubic"},
        fail_on_error=False,
        bands=["red"],
        clip_to_area=True,
        dtype="float32",
        anchor=AnchorEnum.CENTER,
    ).load(items, area)
    breakpoint()

    return tides_lowres(ds, **kwargs)


def tides_lowres(xr: Dataset, tide_directory="data/raw/tidal_models") -> Dataset:
    tides_lowres = pixel_tides(
        xr,
        resample=False,
        model="FES2022",
        directory=tide_directory,
        resolution=3300,
        parallel=False,  # not doing parallel since it seemed to be slower
        extrapolate=True,  # we are likely using the fes2022 extrapolated tide
        # data but there are still some areas which are mis-masked as lakes
    ).transpose("time", "y", "x")

    tides_lowres.coords["time"] = tides_lowres.coords["time"].astype("str")
    return tides_lowres


def filter_by_tides(ds: Dataset, tides_lr: DataArray) -> Dataset:
    """Remove out of range tide values from a given dataset."""
    tide_cutoff_min, tide_cutoff_max = tide_cutoffs_lr(tides_lr, tide_centre=0.0)

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


def tide_cutoffs_lr(
    tides_lowres: DataArray, tide_centre=0.0
) -> Tuple[DataArray, DataArray]:
    """A replacement for coastlines.tide_cutoffs that is a little memory
    friendlier"""
    # Calculate min and max tides
    tide_min = tides_lowres.min(dim="time")
    tide_max = tides_lowres.max(dim="time")

    # Identify cutoffs
    tide_cutoff_buffer = (tide_max - tide_min) * 0.25
    tide_cutoff_min = tide_centre - tide_cutoff_buffer
    tide_cutoff_max = tide_centre + tide_cutoff_buffer
    breakpoint()

    return tide_cutoff_min, tide_cutoff_max
