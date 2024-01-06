import json
import sys

from azure_logger import CsvLogger, filter_by_log
from dep_tools.namers import DepItemPath
from dep_tools.utils import get_container_client

from grid import test_grid as GRID


def get_ids(datetime, version, dataset_id, retry_errors=True, grid=GRID) -> list:
    namer = DepItemPath(
        sensor="ls",
        dataset_id=dataset_id,
        version=version,
        time=datetime.replace("/", "_"),
    )

    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=namer.log_path(),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )
    return filter_by_log(grid, logger.parse_log(), retry_errors).index.to_list()


def get_years_from_datetime(datetime):
    years = datetime.split("-")
    if len(years) == 2:
        years = range(int(years[0]), int(years[1]) + 1)
    elif len(years) > 2:
        ValueError(f"{datetime} is not a valid value for --datetime")
    return years


def print_tasks(datetime, version, limit, no_retry_errors, dataset_id):
    ids = get_ids(datetime, version, dataset_id, not no_retry_errors)
    params = [
        {
            "region-code": region[0][0],
            "region-index": region[0][1],
            "datetime": region[1],
        }
        for id in ids
    ]

    if limit is not None:
        params = params[0 : int(limit)]

    json.dump(params, sys.stdout)
