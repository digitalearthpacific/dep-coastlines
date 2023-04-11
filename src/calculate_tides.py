from typing import Dict

import geopandas as gpd
from xarray import DataArray, Dataset

# local submodules
from dea_tools.coastal import pixel_tides
from dep_tools.Processor import Processor


def calculate_tides(xr: DataArray, pixel_tides_kwargs: Dict = dict()) -> Dataset:
    working_ds = xr.isel(band=0).to_dataset().drop_duplicates(...)

    tides_lowres = (
        pixel_tides(working_ds, resample=False, **pixel_tides_kwargs)
        .transpose("time", "y", "x")
        .to_dataset("time")
    )

    # Possible workflow using map_blocks
    #    working_ds = working_ds.chunk(dict(time=270, x=256, y=256))
    #    pixel_tides_kwargs.update(dict(resample=False))
    # Problem here was making the template (consider the res, etc)
    #    tides_lowres = working_ds.map_blocks(pixel_tides, kwargs=pixel_tides_kwargs)
    #    tides_lowres =  pixel_tides(working_ds, resample=False, **pixel_tides_kwargs).transpose("time", "y", "x").to_dataset("time")

    # date bands are type pd.Timestamp, need to change them to string
    tides_lowres = tides_lowres.rename(
        dict(zip(tides_lowres.keys(), [str(k) for k in tides_lowres.keys()]))
    )

    return tides_lowres


if __name__ == "__main__":
    pixel_tides_kwargs = dict(
        model="TPXO9-atlas-v5", directory="../coastlines-local/tidal-models/"
    )
    aoi_by_tile = gpd.read_file("data/coastline_split_by_pathrow.gpkg").set_index(
        ["PATH", "ROW"]
    )

    Processor(
        scene_processor=calculate_tides,
        dataset_id="tpxo_lowres",
        scene_processor_kwargs=dict(pixel_tides_kwargs=pixel_tides_kwargs),
        aoi_by_tile=aoi_by_tile,
        convert_output_to_int16=False,
    ).process_by_scene()
