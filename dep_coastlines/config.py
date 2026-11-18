"""Configuration information"""
import os

BUCKET = "dep-public-staging"
"""str: The name of the output bucket used for interim data (mosaics and coastline tiles)."""


OUTPUT_CRS = 3832
"""str | int: The CRS of the output data."""

_region = "us-west-2"
HTTPS_PREFIX = f"https://{BUCKET}.s3.{_region}.amazonaws.com"
"""str: The https domain for output data."""

AOI_URL = f"{HTTPS_PREFIX}/aoi/aoi.gpkg"
"""str: The web url to the AOI."""

CURRENT_COASTLINES_OUTPUT = "data/processed/0-7-0-54/dep_ls_coastlines_0-7-0-54.gpkg"
NEXT_GEN_COASTLINES_OUTPUT = "data/processed/0.8.2/dep_ls_coastlines_0.8.2.gpkg"

GRID_VERSION = "0.9"
"""str: The version of the analysis grid and area of interest raster. It
follows the minor version of the product. A built version is never changed,
so changing the area of interest means a new version."""

MOSAIC_VERSION = "0.8.1"
MOSAIC_DATASET_ID = "coastlines/interim/mosaic"

CLOUD_MODEL_VERSION = "0.8.1"
CLOUD_MODEL_FILE = (
    f"dep_coastlines/cloud_model/cloud_model_{CLOUD_MODEL_VERSION}.joblib"
)

VECTOR_VERSION = "0.8.4dev4"
VECTOR_DATASET_ID = "coastlines/interim/coastlines"
VECTOR_DATETIME = "1999/2025"

TIDE_MODEL = "FES2022_extrapolated"
"""str: The pyTMD tide model name. The extrapolated FES2022 ocean tide fills
gaps along coastlines that the standard version leaves empty."""

OVERTURE_RELEASE = "2026-10-21.0"
"""str: The Overture Maps release that missing land is read from when the
grid is rebuilt. Overture only keeps recent releases, so this may need to be
moved forward before a rebuild."""

STAC_CATALOG_URL = os.getenv("STAC_CATALOG_URL", "https://landsatlook.usgs.gov/stac-server")
STAC_COLLECTIONS = os.getenv("STAC_COLLECTIONS", "landsat-c2l2-sr").split(",")
