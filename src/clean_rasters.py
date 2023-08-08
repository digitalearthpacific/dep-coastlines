"""
This is a work-in-progress script to "post-process" water index data before
vectorizing. The functionality here is part of the vectorization code from the
DEA and DEAfrica work, but it is primarily raster based, hence the name.
The actual vectorization (using subpixel_contours) may or may not belong here, 
so things may move around / get renamed some in the coming weeks.

Please refer to raster_cleaning.py for specific functions.
"""
from dask.distributed import Client
from dask_gateway import GatewayCluster
import geopandas as gpd
import rioxarray as rx

from dea_tools.spatial import subpixel_contours
from dep_tools.utils import blob_exists, write_to_blob_storage

from raster_cleaning import contours_preprocess
from utils import load_blobs
from dep_tools.utils import get_blob_path


def clean_rasters(
    overwrite_images: bool = False, overwrite_lines: bool = False
) -> None:
    aoi = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
    ).set_index(["PATH", "ROW"], drop=False)

    version = "3Aug2023"
    prefix = f"coastlines/{version}"
    start_year = 2014
    end_year = 2023

    index_threshold = -1280.0
    # While this functionality is similar to dep_tools.Processor, we don't
    # do a stac search / load so it doesn't fit entirely.
    for index, _ in aoi.iterrows():
        output_path = get_blob_path("nir08-clean", index, prefix)
        print(output_path)
        if not blob_exists(output_path) or overwrite_images:
            # If using local data, could use load_local_data instead. They
            # may be combined soon.
            yearly_ds = load_blobs(
                "nir08", index, prefix, range(start_year, end_year), chunks=True
            )

            # some scenes have no data
            if yearly_ds is None:
                continue

            yearly_ds = yearly_ds[["nir08", "count"]]

            composite_years = [
                f"{year-1}_{year+1}" for year in range(start_year, end_year)
            ]
            composite_ds = load_blobs(
                "nir08", index, prefix, composite_years, chunks=True
            )[["nir08", "count"]]

            # thresholding for nir band is the opposite direction of
            # all other indices, so we multiply by negative 1.
            yearly_ds["nir08"] = yearly_ds.nir08 * -1
            composite_ds["nir08"] = composite_ds.nir08 * -1
            water_index = "nir08"

            composite_ds["year"] = range(start_year, end_year)
            combined_ds = contours_preprocess(
                yearly_ds,
                composite_ds,
                water_index=water_index,
                index_threshold=index_threshold,
                mask_temporal=True,
            )
            # Some strange thing happening where year needs to be numeric for
            # contours preprocess, but we need to make it a string here or
            # rioxarray has problems writing it ("numpy.int64 has no attribute
            # encode")
            combined_ds["year"] = combined_ds.year.astype("str")

            write_to_blob_storage(
                combined_ds.to_dataset("year"),
                path=output_path,
                write_args=dict(driver="COG"),
                overwrite=overwrite_images,
            )
        else:
            combined_ds = rx.open_rasterio(f"/vsiaz/output/{output_path}", chunks=True)

            combined_ds = combined_ds.rename({"band": "year"}).assign_coords(
                year=list(combined_ds.attrs["long_name"])
            )

        lines_path = get_blob_path("lines", index, prefix, ext="gpkg")
        if not blob_exists(lines_path) or overwrite_lines:
            combined_gdf = subpixel_contours(
                combined_ds, dim="year", z_values=[index_threshold]
            )
            write_to_blob_storage(
                combined_gdf,
                lines_path,
                write_args=dict(driver="GPKG", layer=f"lines_{index}"),
                overwrite=overwrite_lines,
            )


if __name__ == "__main__":
    overwrite_images = True
    overwrite_lines = True
    try:
        cluster = GatewayCluster(worker_cores=1, worker_memory=8)
        cluster.scale(100)
        with cluster.get_client() as client:
            print(client.dashboard_link)
            clean_rasters(overwrite_images, overwrite_lines)
    except ValueError:
        with Client() as client:
            print(client.dashboard_link)
            clean_rasters(overwrite_images, overwrite_lines)
