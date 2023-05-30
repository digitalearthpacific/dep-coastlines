from dask.distributed import Client
from dask_gateway import GatewayCluster
import geopandas as gpd

from dep_tools.utils import blob_exists, write_to_blob_storage

from raster_cleaning import contours_preprocess
from utils import load_blobs


def clean_rasters():
    aoi = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
    )
    start_year = 2014
    end_year = 2022

    for _, r in aoi.iterrows():
        path = r.PATH
        row = r.ROW
        output_path = f"clean-nir/clean_nir_{path}_{row}.tif"
        if not blob_exists(output_path):
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
