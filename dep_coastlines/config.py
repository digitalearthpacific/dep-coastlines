"""Configuration information"""

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

MOSAIC_VERSION = "0.8.1"
MOSAIC_DATASET_ID = "coastlines/interim/mosaic"

CLOUD_MODEL_VERSION = "0.8.1"
CLOUD_MODEL_FILE = (
    f"dep_coastlines/cloud_model/cloud_model_{CLOUD_MODEL_VERSION}.joblib"
)

VECTOR_VERSION = "0.8.4dev4"
VECTOR_DATASET_ID = "coastlines/interim/coastlines"
VECTOR_DATETIME = "1999/2024"
