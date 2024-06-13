import numpy as np
from rasterio.features import sieve
from skimage.morphology import dilation, disk
from dea_tools.spatial import xr_vectorize

# Taken from the latest deafrica code for now, needed to remove the "transform=" arg from xr_vectorize since it is redundant with the current version of dea-tools.

# should fork, issue, etc. when I have a chance


def certainty_masking(
    yearly_ds, variation_var, obs_threshold=5, variation_threshold=0.3, sieve_size=128
):
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

    high_variation = yearly_ds[variation_var] > variation_threshold
    low_obs = yearly_ds["count"] < obs_threshold

    # Create raster mask with values of 0 for good data, values of
    # 1 for unstable data, and values of 2 for insufficient data.
    # Clean this by sieving to merge small areas of pixels into
    # their neighbours
    raster_mask = (
        high_variation.where(~low_obs, 2)
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


# for this, I had to
# 1. change the interp to sel to stop nans from propagating
# 2. fill nans with water so the directionality would function correctly
# The latter may need some revisiting
def annual_movements(
    points_gdf, contours_gdf, yearly_ds, baseline_year, water_index, max_valid_dist=1000
):
    """
    For each rate of change point along the baseline annual coastline,
    compute the distance to the nearest point on all neighbouring annual
    coastlines and add this data as new fields in the dataset.

    Distances are assigned a directionality (negative = located inland,
    positive = located sea-ward) by sampling water index values from the
    underlying DEA Coastlines rasters to determine if a coastline was
    located in wetter or drier terrain than the baseline coastline.

    Parameters:
    -----------
    points_gdf : geopandas.GeoDataFrame
        A `geopandas.GeoDataFrame` containing rates of change points.
    contours_gdf : geopandas.GeoDataFrame
        A `geopandas.GeoDataFrame` containing annual coastlines.
    yearly_ds : xarray.Dataset
        An `xarray.Dataset` containing annual DEA CoastLines rasters.
    baseline_year : string
        A string giving the year used as the baseline when generating
        the rates of change points dataset. This is used to load DEA
        CoastLines water index rasters to calculate change
        directionality.
    water_index : string
        A string giving the water index used in the analysis. This is
        used to load DEA CoastLines water index rasters to calculate
        change directionality.
    max_valid_dist : int or float, optional
        Any annual distance greater than this distance will be set
        to `np.nan`.

    Returns:
    --------
    points_gdf : geopandas.GeoDataFrame
        A `geopandas.GeoDataFrame` containing rates of change points
        with added 'dist_*' attribute columns giving the distance to
        each annual coastline from the baseline. Negative values
        indicate that an annual coastline was located inland of the
        baseline; positive values indicate the coastline was located
        towards the ocean.
    """

    def _point_interp(points, array, **kwargs):
        points_gs = gpd.GeoSeries(points)
        x_vals = xr.DataArray(points_gs.x, dims="z")
        y_vals = xr.DataArray(points_gs.y, dims="z")
        return array.interp(x=x_vals, y=y_vals, **kwargs)

    def _point_value(points, array, **kwargs):
        points_da = points.get_coordinates().to_xarray()
        return array.sel(points_da, method="nearest", **kwargs)

    # Get array of water index values for baseline time period
    water = 1
    yearly_ds_unmasked = yearly_ds.where(~np.isnan(yearly_ds), water)
    baseline_array = yearly_ds_unmasked[water_index].sel(year=int(baseline_year))

    # Copy baseline point geometry to new column in points dataset
    points_gdf["p_baseline"] = points_gdf.geometry

    # Years to analyse
    years = contours_gdf.index.unique().values

    # Iterate through all comparison years in contour gdf
    for comp_year in years:
        # Set comparison contour
        comp_contour = contours_gdf.loc[[comp_year]].geometry.iloc[0]

        # Find nearest point on comparison contour, and add these to points dataset
        points_gdf[f"p_{comp_year}"] = points_gdf.apply(
            lambda x: nearest_points(x.p_baseline, comp_contour)[1], axis=1
        )

        # Compute distance between baseline and comparison year points and add
        # this distance as a new field named by the current year being analysed
        distances = points_gdf.apply(
            lambda x: x.geometry.distance(x[f"p_{comp_year}"]), axis=1
        )

        # Set any value over X m to NaN, and drop any points with
        # less than 50% valid observations
        points_gdf[f"dist_{comp_year}"] = distances.where(distances < max_valid_dist)

        # Extract comparison array containing water index values for the
        # current year being analysed
        comp_array = yearly_ds_unmasked[water_index].sel(year=int(comp_year))

        # Sample water index values for baseline and comparison points
        points_gdf["index_comp_p1"] = _point_value(points_gdf["p_baseline"], comp_array)
        points_gdf["index_baseline_p2"] = _point_value(
            points_gdf[f"p_{comp_year}"], baseline_array
        )

        # Compute change directionality (positive = located towards the
        # ocean; negative = located inland)
        points_gdf["loss_gain"] = np.where(
            points_gdf.index_baseline_p2 > points_gdf.index_comp_p1, 1, -1
        )

        # Ensure NaNs are correctly propagated (otherwise, X > NaN
        # will return False, resulting in an incorrect land-ward direction)
        is_nan = points_gdf[["index_comp_p1", "index_baseline_p2"]].isna().any(axis=1)
        points_gdf["loss_gain"] = points_gdf["loss_gain"].where(~is_nan)

        # Multiply distance to set change to negative, positive or NaN
        points_gdf[f"dist_{comp_year}"] = (
            points_gdf[f"dist_{comp_year}"] * points_gdf.loss_gain
        )

        # Calculate compass bearing from baseline to comparison point;
        # first we need our points in lat-lon
        lat_lon = points_gdf[["p_baseline", f"p_{comp_year}"]].apply(
            lambda x: gpd.GeoSeries(x, crs=points_gdf.crs).to_crs("EPSG:4326")
        )

        geodesic = pyproj.Geod(ellps="WGS84")
        bearings = geodesic.inv(
            lons1=lat_lon.iloc[:, 0].values.x,
            lats1=lat_lon.iloc[:, 0].values.y,
            lons2=lat_lon.iloc[:, 1].values.x,
            lats2=lat_lon.iloc[:, 1].values.y,
        )[0]

        # Add bearing as a new column after first restricting
        # angles between 0 and 180 as we are only interested in
        # the overall axis of our points e.g. north-south
        points_gdf[f"bearings_{comp_year}"] = bearings % 180

    # Calculate mean and standard deviation of angles
    points_gdf["angle_mean"] = (
        points_gdf.loc[:, points_gdf.columns.str.contains("bearings_")]
        .apply(lambda x: circmean(x, high=180), axis=1)
        .round(0)
        .astype(int)
    )
    points_gdf["angle_std"] = (
        points_gdf.loc[:, points_gdf.columns.str.contains("bearings_")]
        .apply(lambda x: circstd(x, high=180), axis=1)
        .round(0)
        .astype(int)
    )

    # Keep only required columns
    to_keep = points_gdf.columns.str.contains("dist|geometry|angle")
    points_gdf = points_gdf.loc[:, to_keep]
    points_gdf = points_gdf.assign(**{f"dist_{baseline_year}": 0.0})
    points_gdf = points_gdf.round(2)

    return points_gdf
