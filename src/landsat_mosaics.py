from typing import Union

import geopandas as gpd
from pandas import DatetimeIndex
from xarray import Dataset

from dep_tools.Processor import run_processor

from tide_utils import filter_by_tides

def landsat_mosaics(ds: Dataset, area) -> Union[Dataset, None]:
    bands = ["blue", "green", "red", "nir08", "swir16", "swir22"]
    ds = ds.drop_duplicates(...).sel(band=bands).to_dataset("band")
    ds = filter_by_tides(ds, area["PATH"].values[0], area["ROW"].values[0])

    if len(ds.time) == 0:
        return None

    median_ds = ds.resample(time="1Y").median()
    median_ds = median_ds.assign_coords(time=[f"{t.year}" for t in DatetimeIndex(median_ds.time)])
    return median_ds.squeeze()



if __name__ == "__main__":
    aoi_by_tile = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
        ).set_index(["PATH", "ROW"], drop=False)[170:255]

    run_processor(
        scene_processor=landsat_mosaics,
        dataset_id="landsat-mosaic",
        aoi_by_tile=aoi_by_tile,
        convert_output_to_int16=False,
        send_area_to_scene_processor=True,
        split_output_by_year=True,
        split_output_by_variable=False,
        overwrite=False,
    )
