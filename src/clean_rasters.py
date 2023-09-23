"""
This is a work-in-progress script to "post-process" water index data before
vectorizing. The functionality here is part of the vectorization code from the
DEA and DEAfrica work, but it is primarily raster based, hence the name.
The actual vectorization (using subpixel_contours) may or may not belong here, 
so things may move around / get renamed some in the coming weeks.

Please refer to raster_cleaning.py for specific functions.
"""


from typing import Tuple

from dea_tools.spatial import subpixel_contours
import geopandas as gpd
from numpy import mean
from xarray import Dataset, where

from azure_logger import CsvLogger, get_log_path
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.loaders import Loader
from dep_tools.processors import Processor
from dep_tools.runner import run_by_area_dask
from dep_tools.utils import (
    blob_exists,
    write_to_blob_storage,
    get_blob_path,
    get_container_client,
)
from dep_tools.writers import Writer

from raster_cleaning import contours_preprocess
from utils import load_blobs


def _set_year_to_middle_year(xr: Dataset) -> Dataset:
    edge_years = [y.split("_") for y in xr.year.to_numpy()]
    middle_years = [int(mean([int(y[0]), int(y[1])])) for y in edge_years]
    xr["year"] = middle_years
    return xr


class CompositeLoader(Loader):
    def __init__(self, prefix, start_year, end_year, dataset, **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix
        self.start_year = start_year
        self.end_year = end_year
        self.dataset = dataset

    def load(self, area) -> Tuple[Dataset, Dataset]:
        yearly_ds = load_blobs(
            self.dataset,
            area.index[0],
            self.prefix,
            range(self.start_year, self.end_year),
            chunks=True,
        )

        #        yearly_mosaic = load_blobs(
        #            "annual-mosaic",
        #            area.index[0],
        #            "coastlines/0_3_4",
        #            range(self.start_year, self.end_year),
        #            chunks=True,
        #        )
        #        yearly_ds["gss"] = yearly_mosaic.green - yearly_mosaic.swir16
        #        yearly_ds["gns"] = yearly_mosaic.green - yearly_mosaic.nir08

        if yearly_ds is None:
            raise EmptyCollectionError()

        composite_years = [
            f"{year-1}_{year+1}" for year in range(self.start_year, self.end_year)
        ]
        composite_ds = load_blobs(
            self.dataset, area.index[0], self.prefix, composite_years, chunks=True
        )

        #        composite_mosaic = load_blobs(
        #            "annual-mosaic",
        #            area.index[0],
        #            "coastlines/0_3_4",
        #            composite_years,
        #            chunks=True,
        #        )
        #        composite_ds["gss"] = composite_mosaic.green - composite_mosaic.swir16
        #        composite_ds["gns"] = composite_mosaic.green - composite_mosaic.nir08
        #
        composite_ds = _set_year_to_middle_year(composite_ds)

        composite_ds = composite_ds.where(
            composite_ds.year.isin(yearly_ds.year), drop=True
        )
        yearly_ds = yearly_ds.where(yearly_ds.year.isin(composite_ds.year), drop=True)

        if len(yearly_ds.year) == 0:
            raise EmptyCollectionError()

        return (yearly_ds, composite_ds)


class Cleaner(Processor):
    def __init__(
        self,
        water_index: str = "nir08",
        masking_index: str = "nir08",
        index_threshold: float = -1280.0,
        masking_threshold: float = -1280.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index_threshold = index_threshold
        self.water_index = water_index
        self.masking_index = masking_index
        self.masking_threshold = masking_threshold

    def process(self, input: Tuple[Dataset, Dataset]) -> Dataset:
        yearly_ds, composite_ds = input

        # thresholding for nir band is the opposite direction of
        # all other indices, so we multiply by negative 1.
        if "nir08" in yearly_ds:
            yearly_ds["nir08"] = yearly_ds.nir08 * -1
            composite_ds["nir08"] = composite_ds.nir08 * -1

        combined_ds = contours_preprocess(
            yearly_ds,
            composite_ds,
            water_index=self.water_index,
            index_threshold=self.masking_threshold,
            masking_index=self.masking_index,
            mask_temporal=True,
            mask_esa_water_land=True,
        )

        # We need to make it a string here or
        # rioxarray has problems writing it ("numpy.int64 has no attribute
        # encode")
        combined_ds["year"] = combined_ds.year.astype(str)

        combined_gdf = subpixel_contours(
            combined_ds, dim="year", z_values=[self.index_threshold]
        )
        combined_gdf.year = combined_gdf.year.astype(int)

        return (combined_ds.to_dataset("year"), combined_gdf)


class CleanedWriter(Writer):
    def __init__(self, dataset_id: str, prefix: str, overwrite: bool = False) -> None:
        self.dataset_id = dataset_id
        self.prefix = prefix
        self.overwrite = overwrite

    def write(self, output, item_id):
        xr, gdf = output
        output_path = get_blob_path(self.dataset_id, item_id, self.prefix)
        if not blob_exists(output_path) or self.overwrite:
            write_to_blob_storage(
                xr,
                path=output_path,
                write_args=dict(driver="COG"),
                overwrite=self.overwrite,
            )

        lines_path = get_blob_path(
            f"{self.dataset_id}-lines", item_id, self.prefix, ext="gpkg"
        )
        if not blob_exists(lines_path) or self.overwrite:
            write_to_blob_storage(
                gdf,
                lines_path,
                write_args=dict(driver="GPKG", layer=f"lines_{item_id}"),
                overwrite=self.overwrite,
            )


def main(
    water_index="nir08",
    threshold=-1280.0,
    masking_index="nir08",
    masking_threshold=-1280.0,
) -> None:
    aoi = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
    ).set_index(["PATH", "ROW"], drop=False)

    input_dataset = "water-indices"
    input_version = "4Sep2023"
    input_prefix = f"coastlines/{input_version}"

    output_dataset = f"{water_index}-clean"
    output_version = "0-3-11"
    prefix = f"coastlines/{output_version}"
    start_year = 2014
    end_year = 2023

    loader = CompositeLoader(input_prefix, start_year, end_year, input_dataset)
    processor = Cleaner(
        water_index=water_index,
        index_threshold=threshold,
        masking_index=masking_index,
        masking_threshold=masking_threshold,
    )
    writer = CleanedWriter(output_dataset, prefix, overwrite=True)
    logger = CsvLogger(
        name=output_dataset,
        container_client=get_container_client(),
        path=get_log_path(
            prefix, output_dataset, output_version, f"{start_year}_{end_year}"
        ),
        overwrite=True,
        header="time|index|status|paths|comment\n",
    )

    aoi = logger.filter_by_log(aoi)

    run_by_area_dask(
        areas=aoi,
        loader=loader,
        processor=processor,
        writer=writer,
        logger=logger,
    )


if __name__ == "__main__":
    main(
        water_index="mndwi",
        threshold=0,
        masking_index="nir08",
        masking_threshold=-1280.0,
    )
