"""Calculates low-resolution tide rasters for all areas and times using 
functionality from the Digital Earth Austalia project. In the DEA and DEAfrica
coastline processing this is included with other raster processing but 
1) For this project all the water index, etc. calculations are dask-enabled,
   while this is not.
2) It has separate prerequisites that other pieces due to the tidal calculation
   packages (see below).
3) This is the most static part of the workflow in that once the areas are
   set, the tidal calculations can be "one and done" for the most part (unless
   you wish to revisit for more recent data at a later time). No reason to redo
   this every time if testing out new water indices / cloud masking etc.

Tidal processing should be done before anything else, as results are needed for
filtering input landsat data before water index calculation.

This can be run locally with tidal model data in ../coastlines-local (see below) as
it does not actually pull any planetary computer data and only writes
small-ish (5km resolution) data to blob storage. It has low memory requirements
and takes just a few hours to run for the full area and all times. It could be
modified to run in kbatch but requires a docker image with the large tidal models
embedded.

TODO: If revisiting this file, consider abstracting some of the constant values
set in the main script body and using typer.
"""

from ast import literal_eval
from typing import Union

from dask.distributed import Client, Lock
import geopandas as gpd
from pandas import DataFrame
from xarray import DataArray, Dataset

from dea_tools.coastal import pixel_tides

from azure_logger import CsvLogger, get_log_path, filter_by_log
from dep_tools.runner import run_by_area
from dep_tools.processors import Processor
from dep_tools.loaders import LandsatOdcLoader
from dep_tools.utils import get_container_client

from dep_tools.writers import AzureXrWriter

# from dep_tools.writers2 import LocalXrWriter
from tide_utils import fill_and_interp, tide_cutoffs_dask


class TideProcessor(Processor):
    def process(self, xr: DataArray, area) -> Dataset:
        working_ds = xr.isel(band=0).to_dataset().drop_duplicates(...)

        tides_lowres = (
            pixel_tides(
                working_ds,
                resample=False,
                model="TPXO9-atlas-v5",
                directory="../coastlines-local/tidal-models/",
                resolution=4980,
            )
            .transpose("time", "y", "x")
            .chunk(time=1)
        )

        tides_lowres.coords["time"] = tides_lowres.coords["time"].astype("str")

        #        tide_cutoff_min, tide_cutoff_max = tide_cutoffs_dask(
        #            tides_lowres, tide_centre=0.0
        #        )

        a_band = working_ds.isel(time=0).to_array().squeeze()
        # effectively replaces all values with constant
        unmasked_band = a_band.where(0, 1)
        interp_to = unmasked_band.rio.clip(area.to_crs(unmasked_band.rio.crs).geometry)

        output = fill_and_interp(tides_lowres, interp_to)

        output.coords["time"] = output.time.astype(str)
        return output.to_dataset("time").astype("int8")

        # Do this  _after_ interpolation
        #        if area is not None:
        #            tides_highres = tides_highres.rio.clip(
        #                area.to_crs(tides_highres.rio.crs).geometry,
        #                all_touched=True,
        #                from_disk=True,
        #            )
        # return tides_highres


def main(datetime: str, version: str, client) -> None:
    dataset_id = "tpx09"
    prefix = f"coastlines/{version}"

    aoi_by_tile = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
    ).set_index(["PATH", "ROW"])

    loader = LandsatOdcLoader(
        datetime=datetime,
        dask_chunksize=dict(band=1, time=1, x=1024, y=1024),
        odc_load_kwargs=dict(fail_on_error=False),
    )

    processor = TideProcessor(send_area_to_processor=True)

    writer = AzureXrWriter(
        dataset_id=dataset_id,
        prefix=prefix,
        convert_to_int16=False,
        overwrite=False,
        extra_attrs=dict(dep_version=version),
        output_nodata=0
        #        write_kwargs=dict(
        #            tiled=True, windowed=True
        #        ),  # lock=Lock("rio", client=client)),
    )

    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=get_log_path(prefix, dataset_id, version, datetime),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )

    aoi_by_tile = filter_by_log(aoi_by_tile, logger.parse_log())

    run_by_area(
        areas=aoi_by_tile,
        loader=loader,
        processor=processor,
        writer=writer,
        logger=logger,
    )


if __name__ == "__main__":
    with Client() as client:
        main("2013/2023", "0.6.0", client)
