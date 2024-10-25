from common import coastlineItemPath

BUCKET = "dep-public-staging"
OUTPUT_CRS = 3832
_region = "us-west-2"
HTTPS_PREFIX = f"https://{BUCKET}.s3.{_region}.amazonaws.com"
AOI_URL = f"{HTTPS_PREFIX}/aoi/aoi.gpkg"
CURRENT_COASTLINES_OUTPUT = "data/processed/0-7-0-54/dep_ls_coastlines_0-7-0-54.gpkg"

TIDES_VERSION = "0.8.0"
TIDES_DATASET_ID = "coastlines/interim/tidal_models/fes2022b"
TIDES_DATETIME = "1984/2023"
TIDES_NAMER = coastlineItemPath(TIDES_DATASET_ID, TIDES_VERSION, TIDES_DATETIME)
