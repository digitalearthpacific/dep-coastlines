import numpy as np
from rasterio.features import sieve
from skimage.morphology import dilation, disk
from dea_tools.spatial import xr_vectorize

# Taken from the latest deafrica code for now, needed to remove the "transform=" arg from xr_vectorize since it is redundant with the current version of dea-tools.

# should fork, issue, etc. when I have a chance


def certainty_masking(yearly_ds, obs_threshold=5, stdev_threshold=0.25, sieve_size=128):
    """
    Generate annual vector polygon masks containing information
    about the certainty of each extracted shoreline feature.
    These masks are used to assign each shoreline feature with
    important certainty information to flag potential issues with
    the data.

    Parameters:
    -----------
    yearly_ds : xarray.Dataset
        An `xarray.Dataset` containing annual DE Africa Coastlines
        rasters.
    obs_threshold : int, optional
        The minimum number of post-gapfilling Landsat observations
        required for an extracted shoreline to be considered good
        quality. Annual shorelines based on low numbers of
        observations can be noisy due to the influence of
        environmental noise like unmasked cloud, sea spray, white
        water etc. Defaults to 5.
    stdev_threshold : float, optional
        The maximum MNDWI standard deviation required for a
        post-gapfilled Landsat observation to be considered good
        quality. Annual shorelines based on MNDWI with a high
        standard deviation represent unstable data, which can
        indicate that the tidal modelling process did not adequately
        remove the influence of tide. For more information,
        refer to BIshop-Taylor et al. 2021
        (https://doi.org/10.1016/j.rse.2021.112734).
        Defaults to 0.25.
    sieve_size : int, optional
        To reduce the complexity of the output masks, they are
        first cleaned using `rasterio.features.sieve` to replace
        small areas of pixels with the values of their larger
        neighbours. This parameter sets the minimum polygon size
        to retain in this process. Defaults to 128.

    Returns:
    --------
    vector_masks : dictionary of geopandas.GeoDataFrames
        A dictionary with year (as an str) as the key, and vector
        data as a `geopandas.GeoDataFrame` for each year in the
        analysis.
    """

    # Identify problematic pixels
    high_stdev = yearly_ds["stdev"] > stdev_threshold
    low_obs = yearly_ds["count"] < obs_threshold

    # Create raster mask with values of 0 for good data, values of
    # 1 for unstable data, and values of 2 for insufficient data.
    # Clean this by sieving to merge small areas of pixels into
    # their neighbours
    raster_mask = (
        high_stdev.where(~low_obs, 2)
        .groupby("year")
        .apply(lambda x: sieve(x.values.astype(np.int16), size=sieve_size))
    )

    # Apply greyscale dilation to expand masked pixels to err on
    # the side of overclassifying certainty issues
    raster_mask = raster_mask.groupby("year").apply(
        lambda x: dilation(x.values, disk(3))
    )

    # Loop through each mask and vectorise
    vector_masks = {}
    for i, arr in raster_mask.groupby("year"):
        vector_mask = xr_vectorize(
            arr,
            crs=yearly_ds.geobox.crs,
            # transform=yearly_ds.geobox.affine,
            attribute_col="certainty",
        )

        # Dissolve column and fix geometry
        vector_mask = vector_mask.dissolve("certainty")
        vector_mask["geometry"] = vector_mask.geometry.buffer(0)

        # Rename classes and add to dict
        vector_mask = vector_mask.rename(
            {0: "good", 1: "unstable data", 2: "insufficient data"}
        )
        vector_masks[str(i)] = vector_mask

    return vector_masks
