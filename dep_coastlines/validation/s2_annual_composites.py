from itertools import product
import json
from pathlib import Path
import sys
from typing import Annotated

from dask.distributed import Client as DaskClient
from dep_tools.aws import object_exists, write_to_s3
from dep_tools.s2_utils import mask_clouds
from eo_tides import pixel_tides as eo_pixel_tides
import geopandas as gpd
from numpy.dtypes import DateTime64DType
import odc.stac
import pandas as pd
from pystac_client import Client
import rioxarray
from shapely import box
import typer


from dep_coastlines.tide_utils import filter_by_tides, tide_cutoffs_lr, tides_lowres
from dep_coastlines.validation.util import make_tides

CHUNK_SIZE = 8192

OUTPUT_DIR = Path("data/validation/s2_mosaics")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

app = typer.Typer()


def load_s2(s2_tile, year):
    s2_catalog = "https://earth-search.aws.element84.com/v1"
    client = Client.open(s2_catalog)

    s2_items = client.search(
        datetime=year,
        collections=["sentinel-2-l2a"],
        query={"grid:code": {"eq": f"MGRS-{s2_tile.lstrip('0')}"}},
    ).items()
    return odc.stac.load(
        s2_items,
        bands=["red", "green", "blue", "scl"],
        chunks=dict(x=CHUNK_SIZE, y=CHUNK_SIZE),
        anchor="center",
    )


def load_ocm(s2_tile, year):
    ocm_catalog = "https://stac.staging.digitalearthpacific.io"
    ocm_client = Client.open(ocm_catalog)

    ocm_items = ocm_client.search(
        collections=["dep_s2_ocm"],
        datetime=year,
        query={"grid:code": {"eq": f"MGRS-{s2_tile.lstrip('0')}"}},
    ).items()
    return odc.stac.load(
        ocm_items, chunks=dict(x=CHUNK_SIZE, y=CHUNK_SIZE), anchor="center"
    )


def all_time_tides(ds):
    all_time = pd.date_range(start="1984", end="2025", freq="16d").tolist()
    gdf = gpd.GeoDataFrame(geometry=[box(*ds.rio.bounds())], crs=ds.rio.crs)
    return make_tides(gdf, crs=ds.rio.crs, time=all_time)


def filter_by_tides(ds):

    all_tides = eo_pixel_tides(
        ds,
        pd.date_range(start="1984", end="2025", freq="16d").tolist(),
        resample=False,
        resolution=3300,
        directory="data/raw/tidal_models",
        model="FES2022_load",
    )
    tide_cutoff_min, tide_cutoff_max = tide_cutoffs_lr(all_tides)
    tide_cutoff_min_hr = (
        tide_cutoff_min.interp(x=ds.x, y=ds.y)
        .compute()
        .chunk(x=CHUNK_SIZE, y=CHUNK_SIZE)
    )
    tide_cutoff_max_hr = (
        tide_cutoff_max.interp(x=ds.x, y=ds.y)
        .compute()
        .chunk(x=CHUNK_SIZE, y=CHUNK_SIZE)
    )
    ds_tides = eo_pixel_tides(
        ds,
        dask_chunks=(CHUNK_SIZE, CHUNK_SIZE),
        dask_compute=False,
        directory="data/raw/tidal_models",
        model="FES2022_load",
    )

    tide_bool = (ds_tides >= tide_cutoff_min_hr) & (ds_tides <= tide_cutoff_max_hr)
    tide_bool["time"] = tide_bool.time.astype(DateTime64DType)
    return ds.where(tide_bool)


@app.command()
def create_mosaic(tile: str, year: str):
    bucket = "dep-public-staging"
    path = f"dep_ls_coastlines/validation/s2_mosaics/{tile}_{year}.tif"
    if not object_exists(bucket=bucket, key=path):
        s2 = load_s2(tile, year)
        ocm = load_ocm(tile, year)

        s2 = filter_by_tides(s2)

        # output_path = OUTPUT_DIR / f"{tile}_{year}.tif"
        output = (
            mask_clouds(s2)
            .where((ocm.mask == 0) & (s2 != 0))[["red", "green", "blue"]]
            .median(dim="time")
        )

        write_to_s3(
            output,
            path=path,
            bucket=bucket,
            use_odc_writer=False,
        )


def bool_parser(raw: str):
    return False if raw == "False" else True


@app.command()
def tiles_and_years(
    output_json: Annotated[str, typer.Option(parser=bool_parser)] = False,
):
    selected_tiles = [
        "59NLG",
        "56MKU",
        "01KGU",
        "07LFL",
        "01KHV",
        "56MQU",
        "60KWF",
        "59NLG",
        "54LYR",
        "01KFS",
        "55MEM",
        "57LXK",
        "57NVH",
        "58KEB",
        "58KHF",
        "59NQB",
        "60KXF",
        "60LYR",
    ]
    years = range(2017, 2025)
    output = product(selected_tiles, years)
    return (
        json.dump([{"tile": tile, "year": year} for tile, year in output], sys.stdout)
        if output_json
        else output
    )


def main():
    for tile, year in tiles_and_years():
        create_mosaic(tile, str(year))


if __name__ == "__main__":
    with DaskClient():
        app()
