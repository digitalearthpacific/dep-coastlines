"""
This is a work-in-progress script to "post-process" water index data before
vectorizing. The functionality here is part of the vectorization code from the
DEA and DEAfrica work, but it is primarily raster based, hence the name.
I'm still on the fence about whether to include the actual vectorization
here (using subpixel_contours), so things may move around / get renamed some
in the coming weeks.

Please refer to raster_cleaning.py for specific functions.
"""
from dask.distributed import Client
from dask_gateway import GatewayCluster
import geopandas as gpd

from dea_tools.spatial import subpixel_contours
from dep_tools.utils import blob_exists, write_to_blob_storage

from raster_cleaning import contours_preprocess
from utils import load_blobs


def clean_rasters():
    aoi = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
    )
    start_year = 2014
    end_year = 2022

    # While this functionality is similar to dep_tools.Processor, we don't
    # do a stac search / load so it doesn't fit entirely.
    for _, r in aoi.iterrows():
        path = r.PATH
        row = r.ROW
        output_path = f"clean-nir/clean_nir_{path}_{row}.tif"
        if not blob_exists(output_path):
            # If using local data, could use load_local_data instead. They
            # may be combined soon.
            yearly_ds = load_blobs(
                "coastlines", path, row, range(start_year, end_year), chunks=True
            )

            # some scenes have no data
            if yearly_ds is None:
                continue

            yearly_ds = yearly_ds[["nir08", "count"]]

            composite_years = [
                f"{year-1}_{year+1}" for year in range(start_year, end_year)
            ]
            composite_ds = load_blobs(
                "coastlines", path, row, composite_years, chunks=True
            )[["nir08", "count"]]

            yearly_ds["nir08"] = yearly_ds.nir08 * -1
            composite_ds["nir08"] = composite_ds.nir08 * -1
            water_index = "nir08"
            index_threshold = -128.0

            composite_ds["year"] = range(start_year, end_year)
            combined_ds = contours_preprocess(
                yearly_ds,
                composite_ds,
                water_index=water_index,
                index_threshold=index_threshold,
                mask_temporal=True,
            )

            write_to_blob_storage(
                combined_ds,
                path=output_path,
                write_args=dict(driver="COG"),
                overwrite=False,
            )

            combined_gdf = subpixel_contours(combined_ds, dim="year", z_values=[-128.0])
            write_to_blob_storage(
                combined_gdf, f"clean-nir/clean_nir_{path}_{row}.gpkg", overwrite=False
            )


if __name__ == "__main__":
    try:
        cluster = GatewayCluster(worker_cores=1, worker_memory=8)
        cluster.scale(100)
        with cluster.get_client() as client:
            print(client.dashboard_link)
            clean_rasters()
    except ValueError:
        with Client() as client:
            print(client.dashboard_link)
            clean_rasters()
