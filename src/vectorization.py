from pathlib import Path

from dask.distributed import Client
from osgeo import gdal
import rioxarray as rx

from dea_tools.spatial import subpixel_contours


if __name__ == "__main__":
    files = [str(f) for f in Path("data/clean-nir").glob("*.tif")]
    for file in files:
        subpixel_contours(
            rx.open_rasterio(file).rio.write_crs(8859), dim="band", z_values=[-128.0]
        ).to_file(f"{file}.gpkg")
        print(file)
    breakpoint()

    vrt_file = "data/clean-nit.vrt"
    gdal.BuildVRT(vrt_file, files)

    breakpoint()
    da = rx.open_rasterio(vrt_file, chunks=True).rio.write_crs(8859)
    da["band"] = range(2014, 2022)

    #    with Client() as client:
    #        print(client.dashboard_link)
    subpixel_contours(da=da, dim="band", z_values=[-128.0]).to_file("test.gpkg")
