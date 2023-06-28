#!/usr/bin/env python3

import shutil
from pathlib import Path

import geopandas as gpd
import requests
from shapely import make_valid

PNG_INLAND_PATHROWS = [
    "100064",
    "100065",
    "099064",
    "099065",
    "098064",
]


def download_file(url, local_file):
    # From: https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    with requests.get(url, stream=True) as r:
        with open(local_file, "wb") as f:
            shutil.copyfileobj(r.raw, f)


pathrows_file = Path("data/WRS2_descending_0.zip")
aoi = Path("data/aoi.gpkg")

if not aoi.exists():
    print(f"AOI file {aoi} does not exist. Run src/aoi/aoi.py first.")
    exit(1)

aoi = gpd.read_file("data/aoi.geojson")

if not pathrows_file.exists():
    # Cache the zipped file
    print("Downloading pathrows data...")
    url = "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip"
    download_file(url, pathrows_file)
else:
    print("Using cached pathrows data...")

print("Loading pathrows...")
pathrows = gpd.read_file(pathrows_file)

# Clean up the polygons
aoi_simplified = make_valid(aoi.simplify(0.01).unary_union)

# Select the pathrows that intersect the AOI using a spatial index
print("Running the intersection")
pathrows = pathrows.iloc[pathrows.sindex.query(aoi_simplified, predicate="intersects")]

# Drop the PNG inland pathrows
print("Dropping PNG inland pathrows")
pathrows = pathrows[~pathrows["PR"].isin(PNG_INLAND_PATHROWS)]

print(f"Writing {len(pathrows)} pathrows to file.")
pathrows.to_file("data/coastal_pathrows.geojson")
