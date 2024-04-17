from pathlib import Path
from statistics import mode
import warnings

from azure.storage.blob import ContainerClient
from numpy import mean
from numpy.lib.stride_tricks import sliding_window_view
import odc.geo.xr
import rioxarray as rx
import xarray as xr

from dep_tools.azure import list_blob_container
from dep_tools.loaders import Loader
from dep_tools.namers import DepItemPath
from dep_tools.utils import get_container_client

from dep_coastlines.water_indices import tndwi


# see the old utils.py for improvements in here
def _set_year_to_middle_year(xr: xr.Dataset) -> xr.Dataset:
    edge_years = [y.split("/") for y in xr.year.to_numpy()]
    middle_years = [str(int(mean([int(y[0]), int(y[1])]))) for y in edge_years]
    xr["year"] = middle_years
    return xr


def get_datetimes(start_year, end_year, years_per_composite):
    # nearly duplicated in task_utils, should probably refactor
    # yeah, just switch to get_composite_datetime and make years separate
    assert years_per_composite % 2 == 1
    year_buffer = int((years_per_composite - 1) / 2)
    years = range(int(start_year) - year_buffer, int(end_year) + 1 + year_buffer)
    if years_per_composite > 1:
        years = [
            f"{y[0]}/{y[years_per_composite - 1]}"
            for y in sliding_window_view(list(years), years_per_composite)
        ]
    return [str(y) for y in years]


def unify_crses(dss):
    crses = [ds.odc.crs.to_epsg() for ds in dss]
    if len(set(crses)) > 1:
        most_common_crs = mode(crses)
        most_common_geobox = dss[crses.index(most_common_crs)].odc.geobox
        for i, ds in enumerate(dss):
            if ds.odc.crs.to_epsg() != most_common_crs:
                dss[i] = ds.odc.reproject(most_common_geobox)
    return dss


class MultiyearMosaicLoader(Loader):
    def __init__(
        self,
        start_year,
        end_year,
        years_per_composite: list[int] | int = 1,
        version: str = "0.6.0",
    ):
        super().__init__()
        self._start_year = start_year
        self._end_year = end_year
        self._version = version
        if isinstance(years_per_composite, list):
            if len(years_per_composite) == 1:
                self._years_per_composite = years_per_composite[0]
            else:
                self._years_per_composite = years_per_composite
                self._years_per_composite.sort()
        else:
            self._years_per_composite = years_per_composite  # int

        self._container_client = get_container_client()

    def load_composite_set(self, area, years_per_composite) -> xr.Dataset:
        dss = []
        for datetime in get_datetimes(
            self._start_year, self._end_year, years_per_composite
        ):
            itempath = DepItemPath(
                sensor="ls",
                dataset_id="coastlines/mosaics-corrected",
                version=self._version,
                time=datetime.replace("/", "_"),
                zero_pad_numbers=True,
            )
            loader = MosaicLoader(
                itempath=itempath, container_client=self._container_client
            )
            ds = loader.load(area)
            if ds is not None:
                dss.append(ds.assign_coords({"year": datetime}))

        output = xr.concat(dss, dim="year")
        if years_per_composite > 1:
            output = _set_year_to_middle_year(output)

        return output

    def load(self, area) -> xr.Dataset | list[xr.Dataset]:
        if not isinstance(self._years_per_composite, list):
            return add_deviations(
                self.load_composite_set(area, self._years_per_composite)
            )
        else:
            composite_sets = [
                self.load_composite_set(area, years_per_composite)
                for years_per_composite in self._years_per_composite
            ]
            # all_time = composite_sets[0].median(dim="year").compute()

            return [
                add_deviations(composite_set)  # , all_time)
                for composite_set in composite_sets
            ]


class MosaicLoader(Loader):
    def __init__(
        self,
        itempath: DepItemPath,
        container_client: ContainerClient | None = None,
        add_deviations: bool = False,
    ):
        self._itempath = itempath
        self._container_client = (
            container_client if container_client is not None else get_container_client()
        )
        self._add_deviations = add_deviations

    def load(self, area):
        if self._add_deviations:
            all_time = (
                MultiyearMosaicLoader(
                    start_year=1999, end_year=2023, version=self._itempath.version
                )
                .load(area)
                .median(dim="year")
            )

        item_id = area.index.to_numpy()[0]
        blobs = list(
            list_blob_container(
                self._container_client,
                prefix=self._itempath._folder(item_id)
                + "/",  # Add trailing slash or we get e.g. 1999_2001
                suffix=".tif",
            )
        )

        if len(blobs) > 0:
            output = xr.merge([_load_single_path(path) for path in blobs])
            output[
                [
                    k
                    for k in output.keys()
                    if not (k.endswith("mad") or k.endswith("stdev") or k == "count")
                ]
            ] /= 10_000
            output["tndwi"] = tndwi(output)
            output["tmndwi"] = tmndwi(output)

            return add_deviations(output, all_time) if self._add_deviations else output
        else:
            message = "No items in folder " + self._itempath._folder(item_id)
            warnings.warn(message)
            return None


def add_deviations(xr, all_time=None):
    if all_time is None:
        all_time = xr.median(dim="year")
    deviation = xr - all_time
    deviation = deviation.rename({k: k + "_dev" for k in list(deviation.keys())})
    all_time = all_time.rename({k: k + "_all" for k in list(all_time.keys())})
    return xr.merge(deviation).merge(all_time).chunk(xr.chunks)


def _load_single_path(path) -> xr.Dataset:
    prefix = "data/"
    if not Path(prefix + path).exists():
        prefix = "https://deppcpublicstorage.blob.core.windows.net/output/"

    ds = rx.open_rasterio(prefix + path, chunks=True, masked=True)
    return ds.squeeze().to_dataset(name=ds.attrs["long_name"])
