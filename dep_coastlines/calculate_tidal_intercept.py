"""
This is a (currently) experimental alternative which leverages all available
landsat data, rather than filtering out extreme tide values. Briefly, it
determines the pixel value of the landsat band or derived index by fitting
a linear regression where the tide height is the independent variable and
the landsat value the dependent variable. 
"""
from dask.distributed import Client
from typing import Union

# import geopandas as gpd
# from numpy import isnan
# from numpy.lib.stride_tricks import sliding_window_view
import planetary_computer
import pystac_client
from xarray import DataArray, Dataset

from azure_logger import CsvLogger
from dep_tools.loaders import (
    Loader,
    OdcLoader,
    SearchLoader,
)
from dep_tools.searchers import LandsatPystacSearcher
from dep_tools.namers import DepItemPath
from dep_tools.processors import LandsatProcessor
from dep_tools.stac_utils import set_stac_properties
from dep_tools.task import ErrorCategoryAreaTask, MultiAreaTask
from dep_tools.utils import get_container_client
from dep_tools.writers import DsWriter

from calculate_tides import get_ids
from grid import grid
from tide_utils import TideLoader, tides_highres, filter_by_tides


class InterceptProcessor(LandsatProcessor):
    def __init__(self, tide_loader: Loader, **kwargs):
        super().__init__(**kwargs)
        self._tide_loader = tide_loader

    def process(self, xr: DataArray, area) -> Union[Dataset, None]:
        xr = super().process(xr)

        # No chunking along time dimension to be able to use curvefit
        nir = xr.nir08.chunk(time=-1)
        thr = tides_highres(xr, area.index[0], self._tide_loader).chunk(time=-1)

        # dask needs bools like this loaded to use .where
        count = nir.count("time", keep_attrs=True).compute().astype("uint16")
        # This is not the final determination (that's done in cleaning)
        # Just here because curvefit can't fit only 1 point
        # unfittable = count <= 1

        # The fitting is (somewhat) computationally expensive. So as to
        # not fit places we don't need to, stack the x & y then filter
        # out the "unfittable" areas above -> places that are either all nan
        # of only have one usable value. This would all work without this step,
        # just take longer (you'd just use nir and thr below).

        # bad_z = unfittable.stack(z=("y", "x"))
        # nir_z = nir.stack(z=("y", "x")).where(~bad_z, drop=True)
        # thr_z = thr.stack(z=("y", "x")).where(~bad_z, drop=True)

        def lr(x, m, b):
            return m * x + b

        output = (
            nir.curvefit(thr, func=lr, reduce_dims="time")
            # nir_z.curvefit(thr_z, func=lr, reduce_dims="time")
            # .unstack()
            # this is necessary or output is shifted because of gaps in
            # coordinate values due to filtering
            # .reindex(nir.indexes, fill_value=float("nan"))
            .curvefit_coefficients.to_dataset("param").rio.write_crs(xr.rio.crs)
        )

        output["count"] = count
        # output["median"] = xr.nir08.median("time")
        tide_filtered = filter_by_tides(xr, area.index[0], self._tide_loader)
        output["median"] = tide_filtered.nir08.median("time")
        return set_stac_properties(xr, output)


def run(task_id: str | list[str], datetime: str, version: str, dataset_id: str) -> None:
    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    searcher = LandsatPystacSearcher(client=client, datetime=datetime)
    stacloader = OdcLoader(
        clip_to_area=True,
        chunks=dict(band=1, time=1, x=4096, y=4096),
        resampling={"qa_pixel": "nearest", "*": "cubic"},
        fail_on_error=False,
        bands=["qa_pixel", "nir08"],
        groupby="solar_day",
    )
    loader = SearchLoader(searcher, stacloader)

    tide_namer = DepItemPath(
        sensor="ls",
        dataset_id="coastlines/tpxo9",
        version="0.6.0",
        time="1984_2023",
        zero_pad_numbers=True,
    )
    tide_loader = TideLoader(tide_namer)

    processor = InterceptProcessor(
        send_area_to_processor=True, mask_clouds=True, tide_loader=tide_loader
    )
    namer = DepItemPath(
        sensor="ls",
        dataset_id=dataset_id,
        version=version,
        time=datetime.replace("/", "_"),
        zero_pad_numbers=True,
    )

    writer = DsWriter(itempath=namer, extra_attrs=dict(dep_version=version))
    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=namer.log_path(),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )

    if isinstance(task_id, list):
        MultiAreaTask(
            task_id,
            grid,
            ErrorCategoryAreaTask,
            loader,
            processor,
            writer,
            logger,
        ).run()
    else:
        ErrorCategoryAreaTask(
            task_id, grid.loc[[task_id]], loader, processor, writer, logger
        ).run()


if __name__ == "__main__":
    datetime = "2017"
    version = "0.6.0"
    dataset_id = "coastlines/nir08_tidal_intercept"
    with Client(memory_limit="16GiB"):
        task_ids = get_ids(datetime, version, dataset_id)
        run(task_ids, datetime, version, dataset_id)
