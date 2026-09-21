import geopandas as gpd

from coastlines.validation import generate_transects
from dep_coastlines.config import AOI_URL, OUTPUT_CRS
from dep_coastlines.grid import buffered_grid


def generate_dep_transects(output_file):
    aoi = gpd.read_file(AOI_URL).to_crs(OUTPUT_CRS)
    (
        gpd.GeoDataFrame(
            geometry=generate_transects(aoi.geometry.boundary.unary_union),
            crs=OUTPUT_CRS,
        )
        .clip(buffered_grid)
        .to_file(output_file)
    )


if __name__ == "__main__":
    generate_dep_transects("data/src/validation_transects.gpkg")
