#!/usr/bin/env python3
"""Extracts the AOI from GADM data and saves it as a geojson file."""
import io
import zipfile
from pathlib import Path

import geopandas as gpd
import requests

countries = [
    "American Samoa",
    "Cook Islands",
    "Fiji",
    "French Polynesia",
    "Guam",
    "Kiribati",
    "Marshall Islands",
    "Micronesia",
    "Nauru",
    "New Caledonia",
    "Niue",
    "Northern Mariana Islands",
    "Palau",
    "Papua New Guinea",
    "Pitcairn Islands",
    "Solomon Islands",
    "Samoa",
    "Tokelau",
    "Tonga",
    "Tuvalu",
    "Vanuatu",
    "Wallis and Futuna",
]

gadm_dir = Path("data/gadm36")
in_file = gadm_dir / "gadm36_levels.gpkg"


gadm_dir.mkdir(parents=True, exist_ok=True)

if not in_file.exists():
    print("Downloading GADM data...")

    url = "https://biogeo.ucdavis.edu/data/gadm3.6/gadm36_levels_gpkg.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("data/gadm36")
else:
    print("Using cached GADM data...")


# Extracting geojson
print("Reading level 0 layer from the file")
all_polys = gpd.read_file(in_file, layer="level0").query("NAME_0 in @countries").dissolve("NAME_0")

print(f"Found {len(all_polys)} country shapes. Writing to file.")
all_polys.to_file("data/aoi.gpkg")
