from typing import Iterable, Tuple

import geopandas as gpd
from dep_tools.exceptions import NoOutputError
from dep_tools.landsat_utils import cloud_mask
from dep_tools.processors import LandsatProcessor
from dep_tools.utils import scale_to_int16
from xarray import DataArray, Dataset

from dep_coastlines.tide_utils import filter_by_tides
from dep_coastlines.water_indices import twndwi


class MosaicProcessor(LandsatProcessor):
    """Create a tide-filtered annual mosaic.

    Args:
        all_tides: All tides for the target area.
    """

    def __init__(self, all_tides: Dataset, **kwargs):
        super().__init__(**kwargs)
        self._tides = all_tides

    def process(self, xr: Dataset, area: gpd.GeoDataFrame) -> Dataset | None:
        # Clip the input to the given area, to reduce computation size
        # for tiles with limited coastal areas.
        xr = xr.rio.clip(
            area.to_crs(xr.rio.crs).geometry,
            all_touched=True,
            from_disk=True,
            drop=True,
        )
        # Do the cloud mask first before scale and offset are done
        xr = super().process(mask_clouds_by_day(xr)).drop_duplicates(...)
        self._tides["time"] = self._tides.time.astype(xr.time.dtype)
        xr = filter_by_tides(xr, self._tides)

        # In case we filtered out all the data
        if not "time" in xr.coords or len(xr.time) == 0:
            raise NoOutputError

        # Limit to one reading per day. This can be accomplished by
        # using groupby="solarday" when loading, but I discovered that landsat
        # masks are not consistent between images (see `mask_clouds_by_day`).
        xr.coords["time"] = xr.time.dt.floor("1D")
        xr = xr.groupby("time").first().drop_vars(["qa_pixel"])

        # This is the nir cutoff for water / land established in
        # https://doi.org/10.1186/s42834-019-0016-5
        cutoff = 0.128 if self.scale_and_offset else 1280.0
        xr["twndwi"] = twndwi(xr, nir_cutoff=cutoff)
        output = xr.median("time", keep_attrs=True)
        output_mad = mad(xr, output).astype("float32")

        output_mad = output_mad.rename(
            dict((variable, variable + "_mad") for variable in output_mad)
        )
        output = output.merge(output_mad)
        output["count"] = xr.nir08.count("time").fillna(0).astype("int16")
        output["twndwi_stdev"] = xr.twndwi.std("time", keep_attrs=True).astype(
            "float32"
        )

        # Scale individual bands from 0-10,000 except for the "count" bands and
        # the bands which end with "stdev" or "mad".
        scalers = [
            key
            for key in output.keys()
            if not (
                str(key).endswith("stdev") or str(key).endswith("mad") or key == "count"
            )
        ]
        output[scalers] = scale_to_int16(
            output[scalers], output_multiplier=10_000, output_nodata=-32767
        )

        return output


def mask_clouds_by_day(
    xr: DataArray | Dataset,
    filters: Iterable[Tuple[str, int]] | None = None,
) -> DataArray | Dataset:
    """Combine cloud masks for the same day and mask Landsat imagery.

    Though overlapping landsat images for the same day were likely taken
    an instant apart, cloud masks are not always consistent between them.
    This function combines overlapping masks for the same day by masking
    out areas where any scenes show clouds among overlapping images.

    Args:
        xr: A Dataset with a "qa_pixel" variable, a "time" dimension, and likely
            multiple days of data.
        filters: Passed to :func:`dep_tools.landsat_utils.cloud_mask`.

    Returns:
        The input data, with clouds masked out. Masked areas are given the nodata
        value (accessed via `.rio.nodata`).
    """
    # Get the cloud mask for all
    mask = cloud_mask(xr, filters)
    mask.coords["day"] = mask.time.dt.floor("1D")
    mask_by_day = mask.groupby("day").max().sel(day=mask.day)

    if isinstance(xr, DataArray):
        return xr.where(~mask_by_day, xr.rio.nodata)
    else:
        for variable in xr:
            xr[variable] = xr[variable].where(~mask_by_day, xr[variable].rio.nodata)
        return xr


def mad(da: DataArray, median_da: DataArray) -> DataArray:
    """Calculate median absolute deviation (MAD).

    Args:
        da: A DataArray with a "time" dimension.
        median_da: The median of that array, over time.

    Returns:
        The median of the absolute deviations of the inputs along the time band.
    """
    return abs(da - median_da).median(dim="time")
