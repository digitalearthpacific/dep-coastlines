BUCKET = "dep-public-staging"
OUTPUT_CRS = 3832
_region = "us-west-2"
HTTPS_PREFIX = f"https://{BUCKET}.s3.{_region}.amazonaws.com"
AOI_URL = f"{HTTPS_PREFIX}/aoi/aoi.gpkg"
CURRENT_COASTLINES_OUTPUT = "data/processed/0-7-0-54/dep_ls_coastlines_0-7-0-54.gpkg"
NEXT_GEN_COASTLINES_OUTPUT = "data/processed/0.8.2/dep_ls_coastlines_0.8.2.gpkg"

MOSAIC_VERSION = "0.8.1"
MOSAIC_DATASET_ID = "coastlines/interim/mosaic"

CLOUD_MODEL_VERSION = "0.8.0"
CLOUD_MODEL_FILE = (
    f"dep_coastlines/cloud_model/cloud_model_{CLOUD_MODEL_VERSION}.joblib"
)

VECTOR_VERSION = "0-8-1"
VECTOR_DATASET_ID = "coastlines/interim/coastlines"
VECTOR_DATETIME = "1984/2024"
