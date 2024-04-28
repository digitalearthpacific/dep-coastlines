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


def twndwi(xr: Dataset, alpha: float = 0.5) -> DataArray:
    return (1 - alpha) * mndwi(xr) + alpha * tndwi(xr)

    # Alpha ranges from 0 to 1, with higher values indicating
    # greater influence of nir08.
    # See https://doi.org/10.1080/01431161.2017.1341667
    cutoff = 0.128
    # green = xr.green.where(xr.green > cutoff, cutoff)
    green = xr.green.where((xr.nir08 >= cutoff) & (xr.green < cutoff), cutoff)
    # green_for_nir = xr.green.where((xr.nir08 >= cutoff) & (xr.green < cutoff), cutoff)
    # nir_term = green_for_nir - alpha * xr.nir08
    # swir_term = xr.green - (1 - alpha) * xr.swir16
    # wndwi = nir_term + swir_term / (
    wndwi = (green - alpha * xr.nir08 - (1 - alpha) * xr.swir16) / (
        green + alpha * xr.nir08 + (1 - alpha) * xr.swir16
    )
    return wndwi.rename("twndwi")


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


def tmndwi(xr: Dataset, cutoff: float = 0.1) -> DataArray:
    big_green = xr.green.where(xr.green > cutoff, cutoff)
    return normalized_ratio(big_green, xr.swir16)


def awei(xr: Dataset) -> DataArray:
    awei = 4 * (xr.green - xr.swir22) - (0.25 * xr.nir08 + 2.75 * xr.swir16)
    return awei.rename("awei")


def wofs(tm: Dataset) -> DataArray:
    # First, rescale to what the wofs model expects
    # (input values should be scaled, not raw int)
    l1_scale = 0.0001
    l1_rescale = 1.0 / l1_scale
    tm = scale_and_offset(tm, scale=[l1_rescale])
    # lX indicates a left path from node X
    # rX indicates a right
    # dX is just the logic for _that_ node
    tm["ndi52"] = normalized_ratio(tm.swir16, tm.green)
    tm["ndi43"] = normalized_ratio(tm.nir08, tm.red)
    tm["ndi72"] = normalized_ratio(tm.swir22, tm.green)

    d1 = tm.ndi52 <= -0.01
    l2 = d1 & (tm.blue <= 2083.5)
    d3 = tm.swir22 <= 323.5

    l3 = l2 & d3
    w1 = l3 & (tm.ndi43 <= 0.61)

    r3 = l2 & ~d3
    d5 = tm.blue <= 1400.5
    d6 = tm.ndi72 <= -0.23
    d7 = tm.ndi43 <= 0.22
    w2 = r3 & d5 & d6 & d7

    w3 = r3 & d5 & d6 & ~d7 & (tm.blue <= 473.0)

    w4 = r3 & d5 & ~d6 & (tm.blue <= 379.0)
    w7 = r3 & ~d5 & (tm.ndi43 <= -0.01)

    d11 = tm.ndi52 <= 0.23
    l13 = ~d1 & d11 & (tm.blue <= 334.5) & (tm.ndi43 <= 0.54)
    d14 = tm.ndi52 <= -0.12

    w5 = l13 & d14
    r14 = l13 & ~d14
    d15 = tm.red <= 364.5

    w6 = r14 & d15 & (tm.blue <= 129.5)
    w8 = r14 & ~d15 & (tm.blue <= 300.5)

    w10 = (
        ~d1
        & ~d11
        & (tm.ndi52 <= 0.32)
        & (tm.blue <= 249.5)
        & (tm.ndi43 <= 0.45)
        & (tm.red <= 364.5)
        & (tm.blue <= 129.5)
    )

    water = w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 | w10
    return water.where(tm.red.notnull(), float("nan"))


def normalized_ratio(band1: DataArray | float, band2: DataArray | float) -> DataArray:
    return (band1 - band2) / (band1 + band2)
