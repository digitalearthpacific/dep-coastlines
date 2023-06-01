"""
Calculates water indices for the given areas and times and saves to blob storage.
As of this writing we are focusing on the use of the nir band for all coastline
work, but the other bands are here for legacy sake and possible use later.

This is best run using kbatch (see calculate_water_indices.yml) with a single
year or group of years (e.g. 2013/2015). We previously tried to run this
variously using all years / all composite years , separately for each variable, 
etc. but the long running / task heavy processes often failed in practice.

Each year took an hour or two to run, so if you start multiple
processes you can calculate for all years within a day or so.

TODO: Should probably not call this "coastlines".
"""

from azure.storage.blob import ContainerClient
import geopandas as gpd
import rioxarray
from xarray import DataArray, Dataset

from dep_tools.Processor import run_processor
from tide_utils import filter_by_tides
from water_indices import mndwi, ndwi, awei, wofs


def water_indices(xr: DataArray, area) -> Dataset:
    # Possible we should do this in Processor.py, need to think through
    # whether there is a case where we _would_ want duplicates
    xr = xr.drop_duplicates(...)
    xr = filter_by_tides(xr, area["PATH"].values[0], area["ROW"].values[0])

    working_ds = mndwi(xr).to_dataset()
    working_ds["ndwi"] = ndwi(xr)
    working_ds["awei"] = awei(xr)

    working_ds["wofs"] = wofs(xr)
    working_ds["nir08"] = xr.sel(band="nir08")

    # In case we filtered out all the data
    if not "time" in working_ds or len(working_ds.time) == 0:
        return None

    output = working_ds.median("time", keep_attrs=True)
    output["wofs"] = working_ds.wofs.mean("time", keep_attrs=True)
    output["count"] = working_ds.mndwi.count("time", keep_attrs=True).astype("int16")
    output["stdev"] = working_ds.mndwi.std("time", keep_attrs=True)
    return output


if __name__ == "__main__":
    aoi_by_tile = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
    ).set_index(["PATH", "ROW"], drop=False)

    run_processor(
        scene_processor=water_indices,
        dataset_id="coastlines",
        n_workers=80,
        year="2014",
        aoi_by_tile=aoi_by_tile,
        convert_output_to_int16=True,
        send_area_to_scene_processor=True,
        overwrite=True,
        output_value_multiplier=1000,
    )
