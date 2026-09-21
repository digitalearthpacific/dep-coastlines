"""Water indices for use in discriminating water and land."""

from xarray import DataArray, Dataset


def mndwi(xr: Dataset) -> DataArray:
    """Calculate modified normalized differental water index (MNDWI).

    MNDWI is calculated as (green - swir16) / (green + swir16).

    Args:
        xr: An :class:`xarray.Dataset` with variables "green" and
            "swir16".

    Returns: A :class:`xarray.DataArray` named "mndwi".
    """
    mndwi = _normalized_ratio(xr.green, xr.swir16)
    return mndwi.rename("mndwi")


def ndwi(xr: Dataset) -> DataArray:
    """Calculate normalized differental water index (NDWI).

    NDWI is calculated as (green - nir08) / (green + nir08).

    Args:
        xr: An :class:`xarray.Dataset` with variables "green" and
            "nir08".

    Returns: A :class:`xarray.DataArray` named "ndwi".
    """
    return _normalized_ratio(xr.green, xr.nir08).rename("ndwi")


def twndwi(xr: Dataset, alpha: float = 0.5, nir_cutoff: float = 0.128) -> DataArray:
    """Calculate a water-index as a weighted average of MNDWI and thresholded nir.

    Guo et al. (2017) devised a water index (WNDWI) based on a weighted average of
    modified normalized differential water index (MNDWI) and the
    normalized differential water index (NDWI). This function implements a similar
    weighted index, but replacing NDWI with a thresholded water index based on
    Mondejar and Tongco (2017). This water index has been extensively tested
    against others and has proven more impervious to noisy values in water near coastal
    areas than other indices and combinations.

    For more information, see: https://doi.org/10.1080/01431161.2017.1341667
    and https://doi.org/10.1080/01431161.2017.1341667

    Args:
        xr: An xarray dataset with variables "green", "swir16", and "nir08",
            containing the green, shortwave infrared, and near-infrared Landsat
            bands.
        alpha: The weighting value, between zero and one. A value of 0.5 indicates
            equal weighting of MNDWI and the thresholded nir index, with lower values
            more heavily weighting MNDWI.
        nir_cutoff:
            The cutoff used to threshold the near-infrared band.

    Returns:
        The calculated water index.

    """
    green = xr.green.where(
        (xr.nir08 >= nir_cutoff) & (xr.green < nir_cutoff), nir_cutoff
    )
    tndwi = _normalized_ratio(green, xr.nir08)
    return (1 - alpha) * mndwi(xr) + alpha * tndwi


def nirwi(xr: Dataset, cutoff: float = 0.128) -> DataArray:
    """Calculates a water index by thresholding the near-infrared band.

    Mondejar and Tongco (2019) defined a water index for Landsat 8 data
    by applying a threshold to the near-infrared band. This function
    implements that operation as a normalized ratio, to maintain
    the same directionality and scale as MNDWI & NDWI, with negative
    values indicating land and positive water. The index is calculated as

    (cutoff - near infrared band) / (cutoff + near infrared band)

    See https://doi.org/10.1186/s42834-019-0016-5 for more details on the
    cutoff value

    Args:
        xr: A dataset with a variable named `nir08` containing the fractional
            (0-1) near-infrared Landsat (or other similar band).
        cutoff: The cutoff value, above which is land.

    Returns:
        The water index. Negative values indicate land, positive water.
    """
    return _normalized_ratio(cutoff, xr.nir08).rename("nirwi")


def _normalized_ratio(band1: DataArray | float, band2: DataArray | float) -> DataArray:
    """Calculate the normalized ratio of 2 arrays."""
    return (band1 - band2) / (band1 + band2)
