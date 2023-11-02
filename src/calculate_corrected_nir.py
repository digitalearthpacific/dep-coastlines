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
from typing import Union

import geopandas as gpd
from numpy import isnan
from numpy.lib.stride_tricks import sliding_window_view
from xarray import DataArray, Dataset

from azure_logger import CsvLogger, get_log_path, filter_by_log
from dep_tools.runner import run_by_area_dask, run_by_area
from dep_tools.loaders import LandsatOdcLoader
from dep_tools.processors import LandsatProcessor
from dep_tools.utils import get_container_client
from dep_tools.writers import AzureXrWriter
from tide_utils import filter_by_tides, tides_highres
from water_indices import mndwi, ndwi


class NirProcessor(LandsatProcessor):
    def process(self, xr: DataArray, area) -> Union[Dataset, None]:
        xr = super().process(xr).drop_duplicates(...)
        # xr = filter_by_tides(xr, area.index[0], area)

        working_ds = mndwi(xr).to_dataset()
        working_ds["ndwi"] = ndwi(xr)
        working_ds["nir08"] = xr.sel(band="nir08")
        nir = working_ds.nir08.chunk(time=-1, x=256, y=256)
        thr = tides_highres(xr, area.index[0], area).chunk(time=-1, x=256, y=256)

        def lr(x, m, b):
            return m * x + b

        bad = (~isnan(nir)).sum(dim="time") <= 1
        # dask needs bools like this loaded
        bad_z = bad.stack(z=("y", "x")).compute()
        nir_z = nir.stack(z=("y", "x")).where(~bad_z, drop=True)
        thr_z = thr.stack(z=("y", "x")).where(~bad_z, drop=True)

        nir_z.curvefit(thr_z, func=lr, reduce_dims="time", skipna=True).unstack().sel(
            param="b"
        ).curvefit_coefficients.rio.to_raster("fit2.tif", driver="COG")
        breakpoint()

        fit = nir.where(~bad, drop=True).curvefit(
            thr.where(~bad, drop=True),
            func=lr,
            reduce_dims="time",
            errors="ignore",
            skipna=True,
        )

        # In case we filtered out all the data
        if not "time" in working_ds or len(working_ds.time) == 0:
            return None

        working_ds.coords["time"] = xr.time.dt.floor("1D")

        # or mean or median or whatever
        working_ds = working_ds.groupby("time").first()
        output = working_ds.median("time", keep_attrs=True)
        output["count"] = working_ds.nir08.count("time", keep_attrs=True).astype(
            "int16"
        )
        output["nir_stdev"] = working_ds.nir08.std("time", keep_attrs=True)
        return output


def main(datetime: str, version: str) -> None:
    aoi_by_tile = (
        gpd.read_file(
            "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
        )
        .set_index(["PATH", "ROW"], drop=False)
        .query("PATH == 74 & ROW == 72")
    )

    dataset_id = "water-indices"
    prefix = f"coastlines/{version}"

    loader = LandsatOdcLoader(
        datetime=datetime,
        dask_chunksize=dict(band=1, time=1, x=4096, y=4096),
        odc_load_kwargs=dict(
            resampling={"qa_pixel": "nearest", "*": "cubic"},
            fail_on_error=False,
            bands=["qa_pixel", "nir08", "green", "swir16"],
        ),
        pystac_client_search_kwargs=dict(query=["landsat:collection_category=T1"]),
        exclude_platforms=["landsat-7"],
    )

    processor = NirProcessor(
        send_area_to_processor=True,
        dilate_mask=True,
    )

    writer = AzureXrWriter(
        dataset_id=dataset_id,
        year=datetime,
        prefix=prefix,
        convert_to_int16=True,
        overwrite=False,
        output_value_multiplier=10000,
        extra_attrs=dict(dep_version=version),
    )
    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=get_log_path(prefix, dataset_id, version, datetime),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )
    # aoi_by_tile = filter_by_log(aoi_by_tile, logger.parse_log())

    run_by_area_dask(
        areas=aoi_by_tile,
        loader=loader,
        processor=processor,
        writer=writer,
        logger=logger,
        n_workers=20,
        worker_memory=16,
    )


if __name__ == "__main__":
    single_years = list(range(1999, 2012))
    composite_years = [f"{y[0]}/{y[2]}" for y in sliding_window_view(single_years, 3)]
    all_years = single_years + composite_years

    main("2018", "0-5-0")
