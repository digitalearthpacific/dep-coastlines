"""
Calculates low-resolution tide rasters for all areas and times using 
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

I ran this locally with tidal model data in ../coastlines-local (see below) as
it does not actually pull any planetary computer data and only writes
small-ish (5km resolution) data to blob storage. It has low memory requirements
and takes just a few hours to run for the full area and all times. It could be
modified to run in kbatch but requires a docker image with the large tidal models
embedded.

TODO: If revisiting this file, consider abstracting some of the constant values
set in the main script body and using typer.
"""

from typing import Dict

import geopandas as gpd
from xarray import DataArray, Dataset

from dea_tools.coastal import pixel_tides
from dep_tools.Processor import run_processor
from dep_tools.utils import get_container_client


def calculate_tides(xr: DataArray, pixel_tides_kwargs: Dict = dict()) -> Dataset:
    working_ds = xr.isel(band=0).to_dataset().drop_duplicates(...)

    tides_lowres = (
        pixel_tides(working_ds, resample=False, **pixel_tides_kwargs)
        .transpose("time", "y", "x")
        .to_dataset("time")
    )

    # date bands are type pd.Timestamp, need to change them to string
    # Watch as (apparently) older versions of rioxarray do not write the
    # band names (times) as `long_name` attribute on output files. Probably
    # worth checking the first few outputs to see.
    tides_lowres = tides_lowres.rename(
        dict(zip(tides_lowres.keys(), [str(k) for k in tides_lowres.keys()]))
    )

    return tides_lowres


if __name__ == "__main__":
    pixel_tides_kwargs = dict(
        model="TPXO9-atlas-v5", directory="../coastlines-local/tidal-models/"
    )
    aoi_by_tile = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
    ).set_index(["PATH", "ROW"])

    run_processor(
        scene_processor=calculate_tides,
        dataset_id="tpxo_lowres",
        container_client=get_container_client(),
        scene_processor_kwargs=dict(pixel_tides_kwargs=pixel_tides_kwargs),
        aoi_by_tile=aoi_by_tile,
        convert_output_to_int16=False,
        overwrite=True,
    )
