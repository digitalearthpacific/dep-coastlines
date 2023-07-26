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

import geopandas as gpd
from xarray import DataArray, Dataset


from dep_tools.Processor import run_processor
from tide_utils import filter_by_tides
from water_indices import mndwi, ndwi, awei, wofs


def nir(xr: DataArray, area) -> Dataset:
    xr = xr.drop_duplicates(...).sel(band="nir08")
    working_ds = filter_by_tides(
        xr, area["PATH"].values[0], area["ROW"].values[0]
    ).to_dataset(name="nir08")

    # In case we filtered out all the data
    if not "time" in working_ds or len(working_ds.time) == 0:
        return None

    working_ds.coords["time"] = working_ds.time.dt.floor("1D")

    # or mean or median or whatever
    working_ds = working_ds.groupby("time").first()

    output = working_ds.median("time", keep_attrs=True)
    output["count"] = working_ds.nir08.count("time", keep_attrs=True).astype("int16")
    output["stdev"] = working_ds.nir08.std("time", keep_attrs=True)
    return output


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

    working_ds.coords["time"] = xr.time.dt.floor("1D")

    # or mean or median or whatever
    working_ds = working_ds.groupby("time").first()
    output = working_ds.median("time", keep_attrs=True)
    output["wofs"] = working_ds.wofs.mean("time", keep_attrs=True)
    output["count"] = working_ds.mndwi.count("time", keep_attrs=True).astype("int16")
    output["stdev"] = working_ds.mndwi.std("time", keep_attrs=True)
    return output


if __name__ == "__main__":
    aoi_by_tile = (
        gpd.read_file(
            "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
        )
        .set_index(["PATH", "ROW"], drop=False)
        .query("PATH == 82 & ROW == 74")
    )

    run_processor(
        scene_processor=nir,
        dataset_id="nir08",
        worker_memory=16,
        prefix="coastlines",
        year="2017",
        aoi_by_tile=aoi_by_tile,
        stac_loader_kwargs=dict(
            epsg="native", resampling={"qa_pixel": "nearest", "*": "cubic"}
        ),
        dask_chunksize=dict(band=1, time=1, x=2048, y=2048),
        convert_output_to_int16=True,
        send_area_to_scene_processor=True,
        overwrite=True,
        output_value_multiplier=10000,
        extra_attrs=dict(dep_version="19Jul2023"),
    )
