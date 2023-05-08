from xarray import DataArray

from dep_tools.utils import scale_and_offset


def mndwi(xr: DataArray) -> DataArray:
    # modified normalized differential water index is just a normalized index
    # like NDVI, with different bands
    mndwi = normalized_ratio(xr.sel(band="green"), xr.sel(band="swir16"))
    return mndwi.rename("mndwi")


def ndwi(xr: DataArray) -> DataArray:
    ndwi = normalized_ratio(xr.sel(band="green"), xr.sel(band="nir08"))
    return ndwi.rename("ndwi")


def awei(xr: DataArray) -> DataArray:
    green = xr.sel(band="green")
    swir1 = xr.sel(band="swir16")
    swir2 = xr.sel(band="swir22")
    nir = xr.sel(band="nir08")

    awei = 4 * (green - swir2) - (0.25 * nir + 2.75 * swir1)
    return awei.rename("awei")


def wofs(tm_da: DataArray) -> DataArray:
    # First, rescale to what the wofs model expects
    l1_scale = 0.0001
    l1_rescale = 1.0 / l1_scale
    tm_da = scale_and_offset(tm_da, scale=[l1_rescale])
    # lX indicates a left path from node X
    # rX indicates a right
    # dX is just the logic for _that_ node
    tm = tm_da.to_dataset("band")
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


def normalized_ratio(band1: DataArray, band2: DataArray) -> DataArray:
    return (band1 - band2) / (band1 + band2)
