import geopandas as gpd
import numpy as np
import rioxarray as rx
import xarray as xr
import xrspatial as xs
from geocube.api.core import make_geocube
from odc.algo import mask_cleanup
from rasterio.warp import transform_bounds
from scipy.ndimage import gaussian_filter
from shapely.geometry import box
from skimage.measure import label
from xarray import DataArray, Dataset, apply_ufunc

from dep_coastlines.grid import remote_aoi_raster_path as AOI_RASTER_PATH


def define_analysis_zone(
    core_land: DataArray,
    water_index: DataArray,
    number_of_expansions: int = 64,
    return_max_cap: bool = False,
) -> DataArray | tuple[DataArray, DataArray]:
    """Define the analysis zone for coastline extraction.

    Common water indices applied to Landsat data are sensitive to noise at
    low values, which can create false positive land detections in ocean. This
    pattern is often noticable is ocean areas near coasts, particularly around
    small islands. This function is used to define an analysis zone around
    known land areas which includes as little ocean as possible (some ocean
    is needed to perform coastline delineation correctly). This is
    accomplished by iteratively expanding land areas only, as defined by the
    water index.

    Args:
        core_land: The starting land mask for each year. This should be as
            conservative as possible, while including all islands, etc.
        water_index: The water index used to determine whether a pixel is land or
            water. As is convention, negative values indicate land, positive water.
        number_of_expansions: The number of dilations to perform when extending land
            areas.
        return_max_cap: Depending on the number of expansions and how conservatively
            defined core land is, the mask may not expand into water in all places.
            This returns areas which would continue to expand if expansion continued.
    Returns:
        An analysis mask.

    """
    # Only expand where there's an edge that's land. Do it multiple times
    # to fill between larger areas.
    # We do 2 because the last expansion needs 2 to fill in corners.
    analysis_zone = core_land.copy()

    def expand_once(analysis_zone):
        return analysis_zone | mask_cleanup(
            water_index.where(analysis_zone) < 0,
            mask_filters=[("dilation", 2)],
        )

    for _ in range(number_of_expansions):
        analysis_zone = expand_once(analysis_zone)

    if return_max_cap:
        last_expansion = expand_once(analysis_zone)
        max_cap = last_expansion & ~analysis_zone
        return analysis_zone, max_cap

    return analysis_zone


def smooth_gaussian(da: DataArray, sigma: float = 0.799) -> DataArray:
    """Apply a 3x3 gaussian kernel to a DataArray.

    The convolution ignores nan values by using the kernel values for
    the non-nan pixels only, and scaling the divisor accordingly. A sigma
    value of 0.799 represents a "classic" 3x3 kernel with a center weight
    of approximately 1/4, side weights of approximately 1/8, and corner weights of
    approximately 1/16. Lower values more heavily weight the center.

    Edge pixels (those without a full 3x3 window) will be set to NaN.

    Args:
        da: A 2-d DataArray with dimensions `"x"` & `"y"`.
        sigma: Value for the sigma parameter as passed to
            :py:func:`scipy.ndimage.gaussian_filter`.

    Returns:
        The input DataArray with the smoothing applied.
    """

    def kernel(sigma: float):
        """Returns a 2D Gaussian kernel array."""
        input = np.zeros((3, 3))
        input[1, 1] = 1
        return gaussian_filter(input, sigma)

    weights = DataArray(kernel(sigma), dims=("xw", "yw"))
    total = (
        da.fillna(0)
        .rolling(x=3, y=3, center=True)
        .construct(x="xw", y="yw")
        .dot(weights)
    )
    divisor = (
        (~da.isnull())
        .astype(int)
        .rolling(x=3, y=3, center=True)
        .construct(x="xw", y="yw")
        .dot(weights)
    )
    return (total / divisor).where(~da.isnull())


def remove_disconnected_areas(core_areas: DataArray, candidate_areas: DataArray):
    """Remove pixels from `candidate_areas` that are not connected to `core_areas`.

    The implementation is to first identify unique candidate regions and use them
    as zones, calculating the maximum value of core_areas (i.e. 1 or 0) within each.
    Regions with a max of 1 are kept.

    This is similar to but more restrictive than :func:`coastlines.vector.temporal_masking`.

    Args:
        core_areas: A boolean DataArray where true values represent known areas.
        candidate_areas: A boolean DataArray where true values represent locations
            we think could be areas, if connected to core areas.

    Returns:
        `candidate_areas` with any pixels not connected to `core_areas` removed.
    """

    if "year" in candidate_areas.dims and candidate_areas.year.size > 1:
        return candidate_areas.groupby("year").map(
            lambda candidate_year: remove_disconnected_land(
                core_areas, candidate_year.squeeze(drop=True)
            )
        )

    zones = apply_ufunc(
        label,
        candidate_areas,
        None,
        0,
        dask="parallelized",
        kwargs=dict(connectivity=1),
    )

    connected_or_not = xs.zonal_stats(
        zones, core_areas.astype("int8"), stats_funcs=["max"]
    )
    connected_zones = connected_or_not["zone"][connected_or_not["max"] == 1]
    return candidate_areas.where(zones.isin(connected_zones)) == 1


def remove_disconnected_land(
    certain_land: DataArray, candidate_land: DataArray
) -> DataArray:
    """Remove pixels from `candidate_land` that are not connected to `certain_land`.

    The implementation is to first identify unique land regions and use them as zones,
    calculating the maximum value of certain_land (i.e. 1 or 0) within each. Regions
    with a max of 1 are kept.

    This is similar to but more restrictive than :func:`coastlines.vector.temporal_masking`.

    Args:
        certain_land: A boolean DataArray where true values represent known land
            areas.
        candidate_land: A boolean DataArray where true values represent locations
            we think could be land, if it is connected to known land areas.

    Returns:
        `candidate_land` with any pixels not connected to `certain_land` removed.
    """

    if "year" in candidate_land.dims and candidate_land.year.size > 1:
        return candidate_land.groupby("year").map(
            lambda candidate_year: remove_disconnected_land(
                certain_land, candidate_year.squeeze(drop=True)
            )
        )

    zones = apply_ufunc(
        label, candidate_land, None, 0, dask="parallelized", kwargs=dict(connectivity=1)
    )

    connected_or_not = xs.zonal_stats(
        zones, certain_land.astype("int8"), stats_funcs=["max"]
    )
    connected_zones = connected_or_not["zone"][connected_or_not["max"] == 1]
    return candidate_land.where(zones.isin(connected_zones)) == 1


def map_inland_water(consensus_ocean: DataArray, water_index: DataArray) -> DataArray:
    """Map inland water bodies based on their lack of connection to ocean.

    Args:
        consensus_ocean: A boolean DataArray indicating areas that are definitely
            ocean all the time.
        water_index: The continuous water index for each year, where positive
            values indicate water and negative indicate land. Has dims x, y &
            year.

    Returns:
        A boolean DataArray with dims x, y and year indiciating areas of inland
        water for each year.
    """
    # First create an annual water map. It is places where the water index
    # is non-negative, but also places where it is null that we know aren't land.
    annual_water = (water_index >= 0) | consensus_ocean & water_index.isnull()

    # Define core ocean areas as places that are not always land, eroded by 60-m
    core_ocean = mask_cleanup(consensus_ocean, mask_filters=[("erosion", 2)])

    # Define ocean for each year by removing water that is not connected to ocean
    annual_ocean = remove_disconnected_areas(core_ocean, annual_water)

    return annual_water & ~annual_ocean


def fill_with_nearby_dates(xarr: DataArray | Dataset) -> DataArray | Dataset:
    """Fill missing values with those from the prior or following year.

    If both adjacent years are missing or null, the output will be null.
    If both adjacent years have data, the earlier year's is used.

    Args:
        xarr: An input xarray object with a "year" dimension, which is a string
            representation of a 4-digit year.

    Returns:
        The input with null values filled with values from adjacent years. If `xarr`
        is a :class:`DataArray` object, the same is returned.
    """

    def fill(da: DataArray) -> DataArray:
        output = da.to_dataset("year")
        for year in da.year.values:
            output[year] = da.sel(year=year)
            intyear = int(year)
            years = [
                str(y) for y in [intyear + 1, intyear - 1] if str(y) in da.year.values
            ]
            for inner_year in years:
                output[year] = output[year].where(
                    ~output[year].isnull(), output[inner_year]
                )
        return output.to_array(dim="year")

    return xarr.apply(fill) if isinstance(xarr, Dataset) else fill(xarr)


def load_gadm_land(ds: Dataset | DataArray) -> DataArray:
    # This is a rasterized version of gadm. It seems better than any ESA product
    # at defining land (see for instance Vanuatu).
    land = rx.open_rasterio(f"s3://{AOI_RASTER_PATH}", chunks=True)
    bounds = list(transform_bounds(ds.rio.crs, land.rio.crs, *ds.rio.bounds()))
    return land.rio.clip_box(*bounds).squeeze().rio.reproject_match(ds).astype(bool)


def load_land_additions(da: DataArray) -> Dataset:

    additions = (
        gpd.read_file("data/land_areas_to_add.gpkg")
        .to_crs(da.rio.crs)
        .clip(box(*da.rio.bounds()))
    )
    additions["one"] = 1
    return make_geocube(additions, like=da, measurements=["one"]).one == 1
