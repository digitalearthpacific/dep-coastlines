from xarray import DataArray, Dataset

from dep_tools.utils import scale_and_offset


def mndwi(xr: Dataset) -> DataArray:
    # modified normalized differential water index is just a normalized index
    # like NDVI, with different bands
    mndwi = normalized_ratio(xr.green, xr.swir16)
    return mndwi.rename("mndwi")


def ndwi(xr: Dataset) -> DataArray:
    ndwi = normalized_ratio(xr.green, xr.nir08)
    return ndwi.rename("ndwi")


def ndvi(xr: Dataset) -> DataArray:
    ndvi = normalized_ratio(xr.nir08, xr.red)
    return ndvi.rename("ndvi")


def wndwi(xr: Dataset, alpha: float = 0.5) -> DataArray:
    # Alpha ranges from 0 to 1, with higher values indicating
    # greater influence of nir08.
    # See https://doi.org/10.1080/01431161.2017.1341667
    wndwi = (xr.green - alpha * xr.nir08 - (1 - alpha) * xr.swir16) / (
        xr.green + alpha * xr.nir08 + (1 - alpha) * xr.swir16
    )
    return wndwi.rename("wndwi")


def twndwi(xr: Dataset, alpha: float = 0.5, nir_cutoff: float = 0.128) -> DataArray:
    return (1 - alpha) * mndwi(xr) + alpha * tndwi(xr, nir_cutoff)


def nirwi(xr: Dataset, cutoff: float = 0.128) -> DataArray:
    # Magic cutoff is from https://doi.org/10.1186/s42834-019-0016-5
    # I make it an "index" so
    # 1. Directionality is the same as other indices and
    # 2. The scales are similarly comprable
    return normalized_ratio(cutoff, xr.nir08).rename("nirwi")


def tndwi(xr: Dataset, cutoff: float = 0.128) -> DataArray:
    green = xr.green.where((xr.nir08 >= cutoff) & (xr.green < cutoff), cutoff)
    return normalized_ratio(green, xr.nir08)


def stndwi(xr: Dataset, cutoff: float = 0.128) -> DataArray:
    values_are_high = (xr.green > cutoff) & (xr.nir08 > cutoff)
    land = xr.green < xr.nir08
    super_green = xr.green.where(values_are_high | land, cutoff)
    return normalized_ratio(super_green, xr.nir08)


def tmndwi(xr: Dataset, cutoff: float = 0.081) -> DataArray:
    green = xr.green.where((xr.swir16 >= cutoff) & (xr.green < cutoff), cutoff)
    return normalized_ratio(green, xr.swir16)


def awei(xr: Dataset) -> DataArray:
    awei = 4 * (xr.green - xr.swir22) - (0.25 * xr.nir08 + 2.75 * xr.swir16)
    return awei.rename("awei")


def normalized_ratio(band1: DataArray | float, band2: DataArray | float) -> DataArray:
    return (band1 - band2) / (band1 + band2)
