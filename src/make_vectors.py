from dea_tools.spatial import subpixel_contours
import geopandas as gpd
from xarray import DataArray

from azure_logger import CsvLogger, get_log_path
from dep_tools.loaders import Loader
from dep_tools.processors import Processor
from dep_tools.runner import run_by_area
from dep_tools.utils import (
    blob_exists,
    write_to_blob_storage,
    get_blob_path,
    get_container_client,
)
from dep_tools.writers import Writer

from utils import load_blob


class CleanLoader(Loader):
    def __init__(self, prefix, dataset_id, **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix
        self.dataset_id = dataset_id

    def load(self, area) -> DataArray:
        da = load_blob(self.dataset_id, area.index[0], self.prefix)

        year = (
            [int(da.attrs["long_name"])]
            if len(da.band) == 1
            else [int(y) for y in da.attrs["long_name"]]
        )

        return da.rename({"band": "year"}).assign_coords(year=year)


class Vectorizer(Processor):
    def __init__(self, index_threshold: float = -1280.0, **kwargs):
        super().__init__(**kwargs)
        self.index_threshold = index_threshold

    def process(self, input: DataArray) -> gpd.GeoDataFrame:
        output = subpixel_contours(input, dim="year", z_values=[self.index_threshold])
        output.year = output.year.astype(int)
        return output


class VectorWriter(Writer):
    def __init__(self, dataset_id: str, prefix: str, overwrite: bool = False) -> None:
        self.dataset_id = dataset_id
        self.prefix = prefix
        self.overwrite = overwrite

    def write(self, output, item_id):
        lines_path = get_blob_path("lines", item_id, self.prefix, ext="gpkg")
        if not blob_exists(lines_path) or self.overwrite:
            write_to_blob_storage(
                output,
                lines_path,
                write_args=dict(driver="GPKG", layer=f"lines_{item_id}"),
                overwrite=self.overwrite,
            )


def main() -> None:
    aoi = gpd.read_file(
        "https://deppcpublicstorage.blob.core.windows.net/output/aoi/coastline_split_by_pathrow.gpkg"
    ).set_index(["PATH", "ROW"], drop=False)

    dataset_id = "lines"
    start_year = 2014
    end_year = 2022

    input_version = "10Aug2023"
    input_prefix = f"coastlines/{input_version}"

    output_version = "11Aug2023"
    prefix = f"coastlines/{output_version}"

    index_threshold = -1280.0

    loader = CleanLoader(input_prefix, dataset_id="nir08-clean")
    processor = Vectorizer(index_threshold=index_threshold)
    writer = VectorWriter(dataset_id, prefix, overwrite=False)
    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=get_log_path(
            prefix, dataset_id, output_version, f"{start_year}_{end_year}"
        ),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )

    aoi = logger.filter_by_log(aoi)

    run_by_area(
        areas=aoi,
        loader=loader,
        processor=processor,
        writer=writer,
        logger=logger,
    )


if __name__ == "__main__":
    main()
