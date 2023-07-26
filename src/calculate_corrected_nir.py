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

"""

import geopandas as gpd
from xarray import DataArray, Dataset


from azure_logger import CsvLogger
from dep_tools.runner import run
from dep_tools.loaders import LandsatOdcLoader
from dep_tools.processor import LandsatProcessor
from dep_tools.utils import get_container_client
from dep_tools.writers import AzureXrWriter
from tide_utils import filter_by_tides


class NirProcessor(LandsatProcessor):
    def process(self, xr: DataArray, area) -> Dataset:
        xr = super().process(xr)
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
        output["count"] = working_ds.nir08.count("time", keep_attrs=True).astype(
            "int16"
        )
        output["stdev"] = working_ds.nir08.std("time", keep_attrs=True)
        return output


if __name__ == "__main__":
    aoi_by_tile = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
    ).set_index(["PATH", "ROW"], drop=False)

    prefix = "coastlines"
    dataset_id = "nir08"
    year = "2021"
    version = "19Jul2023"

    loader = LandsatOdcLoader(
        datetime=year,
        dask_chunksize=dict(band=1, time=1, x=2048, y=2048),
        odc_load_kwargs=dict(resampling={"qa_pixel": "nearest", "*": "cubic"}),
    )
    processor = NirProcessor(send_area_to_processor=True)
    writer = AzureXrWriter(
        dataset_id=dataset_id,
        year=year,
        prefix=prefix,
        convert_to_int16=True,
        overwrite=True,
        output_value_multiplier=10000,
        extra_attrs=dict(dep_version=version),
    )
    logger = CsvLogger(
        name="nir08",
        container_client=get_container_client(),
        path=f"{prefix}/{dataset_id}/{year}/log_{version}.csv",
        header="time| index| status\n",
    )

    run(aoi_by_tile, loader, processor, writer, logger, worker_memory=16)
