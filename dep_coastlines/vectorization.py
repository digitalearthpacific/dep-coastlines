"""
Simple example script for vectorization from local files.
"""
from pathlib import Path

import rioxarray as rx

from dea_tools.spatial import subpixel_contours


if __name__ == "__main__":
    files = [str(f) for f in Path("data/clean-nir").glob("*.tif")]
    for file in files:
        subpixel_contours(
            rx.open_rasterio(file).rio.write_crs(8859), dim="band", z_values=[-128.0]
        ).to_file(f"{file}.gpkg")
        print(file)
