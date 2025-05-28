import warnings

from numpy import mean
from numpy.lib.stride_tricks import sliding_window_view
import odc.geo.xr
from odc.geo.geobox import AnchorEnum
import odc.stac
from pystac import Item
import xarray as xr

from dep_tools.loaders import Loader
from dep_tools.namers import DepItemPath

from dep_coastlines.common import coastlineItemPath
from dep_coastlines.config import MOSAIC_DATASET_ID, MOSAIC_VERSION, HTTPS_PREFIX
from dep_coastlines.water_indices import twndwi, mndwi, ndwi, nirwi


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


class MultiyearMosaicLoader(Loader):
    def __init__(
        self,
        start_year,
        end_year,
        years_per_composite: list[int] | int = [1, 3],
        version: str = MOSAIC_VERSION,
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

    def load_composite_set(self, area, years_per_composite) -> xr.Dataset:
        dss = []
        for datetime in get_datetimes(
            self._start_year, self._end_year, years_per_composite
        ):
            itempath = coastlineItemPath(
                dataset_id=MOSAIC_DATASET_ID,
                version=self._version,
                time=datetime.replace("/", "_"),
            )
            loader = MosaicLoader(itempath=itempath)
            ds = loader.load(area)
            if ds is not None:
                dss.append(ds.assign_coords({"year": datetime}))

        output = xr.concat(dss, dim="year")
        # Check before you change this!
        output["twndwi"] = twndwi(output)
        output["mndwi"] = mndwi(output)
        output["ndwi"] = ndwi(output)
        output["nirwi"] = nirwi(output)
        if years_per_composite > 1:
            output = _set_year_to_middle_year(output)

        return output

    def load(self, area) -> xr.Dataset | list[xr.Dataset]:
        if not isinstance(self._years_per_composite, list):
            return _add_deviations(
                self.load_composite_set(area, self._years_per_composite)
            )
        else:
            composite_sets = [
                self.load_composite_set(area, years_per_composite)
                for years_per_composite in self._years_per_composite
            ]

            return [_add_deviations(composite_set) for composite_set in composite_sets]


class MosaicLoader(Loader):
    def __init__(
        self,
        itempath: DepItemPath,
        add_deviations: bool = False,
    ):
        self._itempath = itempath
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
        stac_path = self._itempath.stac_path(item_id)
        try:
            stac_item = Item.from_file(f"{HTTPS_PREFIX}/{stac_path}")
        except Exception as e:
            warnings.warn("error from when loading stac item: {}".format(e))
            return None

        output = odc.stac.load([stac_item], chunks=dict(x=2048, y=2048)).squeeze()

        output[
            [
                k
                for k in output.keys()
                if not (k.endswith("mad") or k.endswith("stdev") or k == "count")
            ]
        ] /= 10_000

        return _add_deviations(output, all_time) if self._add_deviations else output


def _set_year_to_middle_year(ds: xr.Dataset) -> xr.Dataset:
    """For an xarray Dataset with a "year" coordinate with format YYYY/YYYY+2
    (e.g. 2021/2023), the year is set to the midpoint (2022 in this example)
    """
    edge_years = [y.split("/") for y in ds.year.to_numpy()]
    middle_years = [str(int(mean([int(y[0]), int(y[1])]))) for y in edge_years]
    ds["year"] = middle_years
    return ds


def _add_deviations(xr, all_time=None):
    if all_time is None:
        all_time = xr.median(dim="year")
    deviation = xr - all_time
    deviation = deviation.rename({k: k + "_dev" for k in list(deviation.keys())})
    all_time = all_time.rename({k: k + "_all" for k in list(all_time.keys())})
    return xr.merge(deviation).merge(all_time).chunk(xr.chunks)
