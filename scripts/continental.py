#!/usr/bin/env python
# coding: utf-8

# This code combines individual datasets into continental DE Pacific
# Coastlines layers:
#
#     * Combines output shorelines and rates of change statistics point
#       vectors into single continental datasets
#     * Aggregates this data to produce moving window hotspot datasets
#       that summarise coastal change at regional and continental scale.
#     * Writes outputs to GeoPackage and zipped shapefiles
#
# Code adapted from
# https://github.com/digitalearthafrica/deafrica-coastlines/blob/main/coastlines/continental.py
# by Jesse Anderson for Digital Earth Pacific.
# Original authors: Robbi Bishop-Taylor, Alex Leith, Indiphile Ngqambuza
# Major changes:
#    * Read and write to s3
#    * Create pmtiles output using tippecanoe
#

import os
from pathlib import Path
import sys

import click
from coastlines.utils import configure_logging
from coastlines.vector import points_on_line, change_regress, vector_schema
from dep_tools.utils import shift_negative_longitudes
import fiona
import geohash as gh
import geopandas as gpd
import pandas as pd
from s3fs import S3FileSystem
from shapely.geometry.point import Point

from dep_coastlines.common import coastlineItemPath
from dep_coastlines.config import (
    BUCKET,
    HTTPS_PREFIX,
    VECTOR_DATASET_ID,
    VECTOR_DATETIME,
    VECTOR_VERSION,
)

STYLES_FILE = "dep_coastlines/styles.csv"


def calculate_roc_stats(ratesofchange, initial_year, minimum_valid_observations=15):
    """Recalculate rates of change statistics at continental scale.

    Thin wrapper around coastlines.vector.all_time_stats that avoids
    importing dep_coastlines.vector (which pulls in heavy C extensions
    that cause segfaults in long-running processes).
    """
    from coastlines.vector import all_time_stats

    stats_list = ["valid_obs", "valid_span", "sce", "nsm", "max_year", "min_year"]
    ratesofchange[stats_list] = ratesofchange.apply(
        lambda x: all_time_stats(x, initial_year=initial_year), axis=1
    )
    ratesofchange["certainty"] = "good"
    ratesofchange.loc[ratesofchange.rate_time.abs() > 200, "certainty"] = (
        "extreme value (> 200 m)"
    )
    ratesofchange.loc[ratesofchange.angle_std > 30, "certainty"] = (
        "high angular variability"
    )
    ratesofchange.loc[
        ratesofchange.valid_obs < minimum_valid_observations, "certainty"
    ] = "insufficient observations"
    return ratesofchange


def wms_fields(gdf):
    """
    Calculates several addition fields required
    for the WMS/TerriaJS Coastlines visualisation.

    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        The input rate of change points vector dataset.

    Returns:
    --------
    A `pandas.DataFrame` containing additional fields
    with a "wms_*" prefix.
    """

    return pd.DataFrame(
        dict(
            wms_abs=gdf.rate_time.abs(),
            wms_conf=gdf.se_time * 1.96,
            wms_grew=gdf.rate_time < 0,
            wms_retr=gdf.rate_time > 0,
            wms_sig=gdf.sig_time <= 0.01,
            wms_good=gdf.certainty == "good",
        )
    )


@click.command()
@click.option(
    "--vector-version",
    type=str,
    required=True,
    default=VECTOR_VERSION,
    help="A unique string proving a name that was used "
    "for output vector directories and files. This is used "
    "to identify the tiled annual shoreline and rates of "
    "change layers that will be combined into continental-"
    "scale layers.",
)
@click.option(
    "--continental-version",
    type=str,
    help="A unique string proving a name that will be used "
    "for output continental-scale layers. This allows "
    "multiple versions of continental-scale layers to be "
    "generated from the same input vector data, e.g. for "
    "testing different hotspot of coastal change summary "
    "layers. If not provided, this will default to the "
    'string provided to "--vector-version".',
)
@click.option(
    "--skip-merge",
    is_flag=True,
    default=False,
    help="Skip the slow S3 merge steps for shorelines and rates of "
    "change. Assumes the GeoPackage already contains these layers "
    "from a previous run.",
)
@click.option(
    "--hotspots/--no-hotspots",
    is_flag=True,
    default=True,
    help="Whether to generate continental-scale hotspots of "
    "coastal change summary layer.",
)
@click.option(
    "--hotspots-radius",
    default=[15000, 5000, 1000],
    multiple=True,
    help="The distance (in metres) used to generate coastal "
    "change hotspots summary layers. This controls the spacing "
    "of each summary point, and the radius used to aggregate "
    "rates of change statistics around each point. "
    "The default generates three hotspot layers with radii "
    "15000 m, 5000 m and 1000 m. To specify multiple custom "
    "radii, repeat this argument, e.g. "
    "`--hotspots-radius 1000 --hotspots-radius 5000`.",
)
@click.option(
    "--baseline-year",
    type=int,
    default=2024,
    help="The annual shoreline used to generate the hotspot "
    "summary points. This is typically the most recent "
    "annual shoreline in the dataset.",
)
@click.option(
    "--include-styles/--no-include-styles",
    is_flag=True,
    default=True,
    help="Set this to indicate whether to include styles " "in output Geopackage file.",
)
@click.option(
    "--include-tiles/--no-include-tiles",
    is_flag=True,
    default=True,
    help="Set this to indicate whether to produce tiles.",
)
@click.option(
    "--filter-to-pacific/--no-filter-to-pacific",
    is_flag=True,
    default=False,
    help="Filter out data north of 18.5° latitude to exclude Hawaii.",
)
def continental_cli(
    vector_version,
    continental_version,
    skip_merge,
    hotspots,
    hotspots_radius,
    baseline_year,
    include_styles,
    include_tiles,
    filter_to_pacific,
):
    log = configure_logging("Continental layers and hotspots generation")

    # If no continental version is provided, copy this from vector
    # version
    if continental_version is None:
        continental_version = vector_version
    output_dir = Path(f"data/processed/{continental_version}")
    output_dir.mkdir(exist_ok=True, parents=True)
    log.info(f"Writing data to {output_dir}")

    # Output path for geopackage and zipped shapefiles
    OUTPUT_GPKG = output_dir / f"dep_ls_coastlines_{continental_version}.gpkg"
    OUTPUT_SHPS = output_dir / f"dep_ls_coastlines_{continental_version}.shp.zip"

    #################
    # Merge vectors #
    #################

    if not skip_merge:
        def input_paths(vector_version):
            vectorItemPath = coastlineItemPath(
                dataset_id=VECTOR_DATASET_ID, version=vector_version, time=VECTOR_DATETIME
            )
            fs = S3FileSystem(anon=False)
            shoreline_paths = " ".join(
                [
                    f"'/vsicurl?max_retry=3&url={HTTPS_PREFIX}/{path.split('/',1)[1]}'"
                    for path in fs.glob(
                        f"{vectorItemPath.bucket}/{vectorItemPath._folder_prefix}/**/*{vectorItemPath.time}.gpkg"
                    )
                    if path.endswith("gpkg")
                ]
            )
            ratesofchange_paths = " ".join(
                [
                    f"'/vsicurl?max_retry=3&url={HTTPS_PREFIX}/{path.split('/',1)[1]}'"
                    for path in fs.glob(
                        f"{vectorItemPath.bucket}/{vectorItemPath._folder_prefix}/**/*{vectorItemPath.time}_roc.gpkg"
                    )
                    if path.endswith("gpkg")
                ]
            )
            return shoreline_paths, ratesofchange_paths

        # Setup input and output file paths
        shoreline_paths, ratesofchange_paths = input_paths(vector_version)

        log.info(f"Found {len(shoreline_paths.split())} shoreline files to merge")
        log.info(f"Found {len(ratesofchange_paths.split())} rates of change files to merge")

        # If shapefile zip exists, delete it first
        if OUTPUT_SHPS.exists():
            OUTPUT_SHPS.unlink()

        # Combine annual shorelines into a single continental layer
        from tempfile import NamedTemporaryFile

        TMP_GPKG = NamedTemporaryFile(suffix=".gpkg").name
        os.system(
            f"ogrmerge.py -o "
            f"{TMP_GPKG} {shoreline_paths} "
            f"-single -overwrite_ds -t_srs epsg:3832 "
            f"-nln shorelines_annual"
        )
        os.system(
            f"ogr2ogr {OUTPUT_GPKG} {TMP_GPKG} "
            f"-dialect sqlite -nln shorelines_annual "
            f'-sql "select * from shorelines_annual order by year"'
        )
        log.info("Merging annual shorelines complete")

        # Combine rates of change stats points into single continental layer
        os.system(
            f"ogrmerge.py "
            f"-o {OUTPUT_GPKG} {ratesofchange_paths} "
            f"-single -update -t_srs epsg:3832 "
            f"-nln rates_of_change"
        )
        log.info("Merging rates of change points complete")

    else:
        log.info("Skipping merge steps, using existing GeoPackage")

    ################################
    # Load GeoPackage vector data  #
    ################################

    log.info("Loading continental vector data from GeoPackage")
    try:
        ratesofchange_gdf = gpd.read_file(
            OUTPUT_GPKG, layer="rates_of_change", engine="pyogrio", use_arrow=True
        ).set_index("uid")

        ratesofchange_gdf = calculate_roc_stats(
            ratesofchange_gdf,
            initial_year=int(VECTOR_DATETIME[:4]),
            minimum_valid_observations=18,
        )

        shorelines_gdf = gpd.read_file(
            OUTPUT_GPKG, layer="shorelines_annual", engine="pyogrio", use_arrow=True
        ).set_index("year")

    except (fiona.errors.DriverError, ValueError):
        raise FileNotFoundError(
            "Continental-scale annual shoreline and rates of "
            "change layers are required. Run without --skip-merge first."
        )

    if filter_to_pacific:
        max_y = 2125000  # ~18.5°N in EPSG:3832
        roc_mask = ratesofchange_gdf.geometry.y <= max_y
        n_roc_removed = (~roc_mask).sum()
        ratesofchange_gdf = ratesofchange_gdf[roc_mask]

        sl_mask = shorelines_gdf.geometry.bounds["maxy"] <= max_y
        n_sl_removed = (~sl_mask).sum()
        shorelines_gdf = shorelines_gdf[sl_mask]

        log.info(
            f"Filtered to Pacific: removed {n_roc_removed} rates of change "
            f"points and {n_sl_removed} shorelines north of {max_y} m "
            f"(EPSG:3832)"
        )

    #####################
    # Generate hotspots #
    #####################

    # Generate hotspot points that provide regional/continental summary
    # of hotspots of coastal erosion and growth
    if hotspots:
        log.info("Generating continental hotspots")

        ######################
        # Calculate hotspots #
        ######################

        for i, radius in enumerate(hotspots_radius):
            # Extract hotspot points
            log.info(f"Calculating {radius} m hotspots")
            hotspots_gdf = points_on_line(
                shorelines_gdf,
                index=baseline_year,
                distance=int(radius / 2),
            )

            # Create polygon windows by buffering points
            buffered_gdf = hotspots_gdf[["geometry"]].copy()
            buffered_gdf["geometry"] = buffered_gdf.buffer(radius)

            # Spatial join rate of change points to each polygon
            hotspot_grouped = (
                ratesofchange_gdf.loc[
                    ratesofchange_gdf.certainty == "good",
                    ratesofchange_gdf.columns.str.contains("dist_|geometry"),
                ]
                .sjoin(buffered_gdf, predicate="within")
                .drop(columns=["geometry"])
                .groupby("index_right")
            )

            # Aggregate/summarise values by taking median of all points
            # within each buffered polygon
            hotspot_values = hotspot_grouped.median().round(2)

            # Extract year from distance columns (remove "dist_")
            x_years = hotspot_values.columns.str.replace("dist_", "").astype(int)

            # Compute coastal change rates by linearly regressing annual
            # movements vs. time
            rate_out = hotspot_values.apply(
                lambda row: change_regress(
                    y_vals=row.values.astype(float), x_vals=x_years, x_labels=x_years
                ),
                axis=1,
            )

            # Add rates of change back into dataframe
            hotspot_values[
                ["rate_time", "incpt_time", "sig_time", "se_time", "outl_time"]
            ] = rate_out

            # Join aggregated values back to hotspot points after
            # dropping unused columns (regression intercept)
            hotspots_gdf = hotspots_gdf.join(hotspot_values.drop("incpt_time", axis=1))

            # Add hotspots radius attribute column
            hotspots_gdf["radius_m"] = radius

            # Initialise certainty column with good values
            hotspots_gdf["certainty"] = "good"

            # Identify any points with insufficient observations and flag these as
            # uncertain. We can obtain a sensible threshold by dividing the
            # hotspots radius by 30 m along-shore rates of change point distance)
            hotspots_gdf["n"] = hotspot_grouped.size()
            hotspots_gdf["n"] = hotspots_gdf["n"].fillna(0)
            hotspots_gdf.loc[hotspots_gdf.n < (radius / 30), "certainty"] = (
                "insufficient points"
            )

            # Generate a geohash UID for each point and set as index
            uids = (
                hotspots_gdf.geometry.to_crs("EPSG:4326")
                .apply(lambda x: gh.encode(x.y, x.x, precision=11))
                .rename("uid")
            )
            hotspots_gdf = hotspots_gdf.set_index(uids)

            # Export hotspots to file, incrementing name for each layer
            try:
                # Export to geopackage
                layer_name = f"hotspots_zoom_{range(0, 10)[i + 1]}"
                hotspots_gdf.to_file(
                    OUTPUT_GPKG,
                    layer=layer_name,
                    schema={
                        "properties": vector_schema(hotspots_gdf),
                        "geometry": "Point",
                    },
                    engine="fiona",  # needed for schema to work
                )

                # Add additional WMS fields and add to shapefile
                hotspots_gdf = pd.concat(
                    [hotspots_gdf, wms_fields(gdf=hotspots_gdf)], axis=1
                )
                hotspots_gdf.to_file(
                    OUTPUT_SHPS,
                    layer=f"coastlines_{continental_version}_{layer_name}",
                    schema={
                        "properties": vector_schema(hotspots_gdf),
                        "geometry": "Point",
                    },
                    engine="fiona",
                )

            except ValueError as e:
                log.exception(f"Failed to generate hotspots with error: {e}")
                sys.exit(1)

        log.info("Writing hotspots complete")

    else:
        log.info("Not writing hotspots...")

    ############################
    # Export zipped shapefiles #
    ############################

    # Add rates of change points to shapefile zip
    ratesofchange_shp = pd.concat(
        [ratesofchange_gdf, wms_fields(gdf=ratesofchange_gdf)], axis=1
    )

    ratesofchange_shp.to_file(
        OUTPUT_SHPS,
        layer=f"coastlines_{continental_version}_rates_of_change",
        schema={
            "properties": vector_schema(ratesofchange_shp),
            "geometry": "Point",
        },
        engine="fiona",
    )

    log.info("Writing rates of change points to zipped shapefiles complete")

    # Add annual shorelines to shapefile zip
    shorelines_gdf.to_file(
        OUTPUT_SHPS,
        layer=f"coastlines_{continental_version}_shorelines_annual",
        schema={
            "properties": vector_schema(shorelines_gdf),
            "geometry": ["MultiLineString", "LineString"],
        },
        engine="fiona",
    )

    log.info("Writing annual shorelines to zipped shapefiles complete")

    #########################
    # Add GeoPackage styles #
    #########################

    if include_styles:
        styles = gpd.read_file(STYLES_FILE)
        # Need to add fake geometry to write to a geopackage. This appears
        # to be due to a recent change in geopandas.
        styles_gpdf = gpd.GeoDataFrame(
            styles, geometry=[Point(0, 0) for _ in range(len(styles))], crs=3832
        )
        styles_gpdf.to_file(OUTPUT_GPKG, layer="layer_styles")
        log.info("Writing styles to GeoPackage file complete")

    else:
        log.info("Not writing styles to GeoPackage")

    if include_tiles:
        log.info("Generating tiles with tippecanoe")
        OUTPUT_TILES = (
            Path(output_dir) / f"dep_ls_coastlines_{continental_version}.pmtiles"
        )
        build_tiles(OUTPUT_GPKG, OUTPUT_TILES)

    upload_dir(output_dir, continental_version)


def build_tiles(output_gpkg: Path, output_file: Path) -> None:
    layers = {
        layer_name: gpd.read_file(
            output_gpkg, layer=layer_name, engine="pyogrio", use_arrow=True
        )
        for layer_name in fiona.listlayers(output_gpkg)
        if layer_name != "layer_styles"
    }
    tippecanoe_layers = ""
    for name, gdf in layers.items():
        gdf = gdf.to_crs(4326)
        gdf["geometry"] = gdf.geometry.apply(shift_negative_longitudes)
        output_geojson_path = output_file.parent / f"{output_file.stem}_{name}.geojson"
        output_pmtile_path = output_file.parent / f"{output_file.stem}_{name}.pmtiles"
        gdf.to_file(output_geojson_path)
        tippecanoe_layers += f"{output_pmtile_path} "
        roc_opts = " -y sig_time -y rate_time -y certainty"
        opts = dict(
            hotspots_zoom_1=f"-B 0 {roc_opts}",
            hotspots_zoom_2=f"-B 4 {roc_opts}",
            hotspots_zoom_3=f"-B 7 {roc_opts}",
            rates_of_change=f"-B 10 {roc_opts} -y se_time",
            shorelines_annual="-y year -y certainty",
        )[name]
        os.system(
            f"tippecanoe {opts} -pi -z13 -f -o {output_pmtile_path} -L {name}:{output_geojson_path}"
        )

    os.system(f"tile-join -f -pk -o {output_file} {tippecanoe_layers}")


def upload_dir(local_dir, version):
    s3 = S3FileSystem()
    s3_path = f"{BUCKET}/dep_ls_coastlines/processed/{version}/"
    s3.put(local_dir, s3_path, recursive=True)


if __name__ == "__main__":
    continental_cli()
