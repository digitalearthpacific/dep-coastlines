"""Print a list of ids which need to be processed."""

import json
import sys
from typing import Annotated

from cloud_logger import filter_by_log
from dep_tools.parsers import bool_parser, datetime_parser
from dep_tools.stac_utils import remove_items_with_existing_stac
from typer import Option, run

from dep_coastlines.common import (
    coastlineItemPath,
    coastlineLogger,
)
from dep_coastlines.grid import buffered_grid as GRID

def print_ids(
    dataset_id: Annotated[str, Option()],
    version: Annotated[str, Option()],
    datetime: Annotated[str, Option(parser=datetime_parser)],
    retry_errors: Annotated[str, Option(parser=bool_parser)] = "True",
    overwrite_logs: Annotated[str, Option(parser=bool_parser)] = "False",
    filter_using_log: Annotated[str, Option(parser=bool_parser)] = "True",
    filter_existing_stac_items: Annotated[str, Option(parser=bool_parser)] = "False",
):
    """Print a list of tile ids and years to process.

    Args:
        dataset_id: The dataset id.
        version: The output data version.
        datetime: A string of the form <year> or <year 1>_<year 2>, representing
         the year or years to process. Parsed by :func:`dep_tools.parsers.parse_datetime`.
        retry_errors: If `filter_using_log` is `True`, whether to retry tasks that were
            previously logged as errors.
        overwrite_logs: Whether to delete the log before processing.
        filter_using_log: Whether to filter tasks to run using the log.
        filter_existing_stac_items: Whether to filter tasks to run by excluding those
            which already have an output STAC Item stored as json in the output location.

    Returns: Nothing. A list of dictionaries with keys `column`, `row`, and `datetime` is
        sent to stdout.
    """
    params = [{"column": id[0], "row": id[1], "datetime": year} for year in datetime for id in 
        _get_ids(
            datetime=year,
            version=version,
            dataset_id=dataset_id,
            retry_errors=bool(retry_errors),
            delete_existing_log=bool(overwrite_logs),
            filter_using_log=bool(filter_using_log),
            filter_existing_stac_items=bool(filter_existing_stac_items),
        )]

    json.dump(params, sys.stdout)


def _get_ids(
    datetime,
    version,
    dataset_id,
    retry_errors=True,
    grid=GRID,
    delete_existing_log: bool = False,
    filter_existing_stac_items: bool = False,
    filter_using_log: bool = True,
) -> list:
    namer = coastlineItemPath(dataset_id=dataset_id, version=version, time=datetime)
    logger = coastlineLogger(
        namer,
        dataset_id=dataset_id,
        delete_existing_log=delete_existing_log,
    )

    if filter_existing_stac_items:
        grid = remove_items_with_existing_stac(grid, namer)

    if filter_using_log:
        grid = filter_by_log(grid, logger.parse_log(), retry_errors)

    return grid.index.to_list()


if __name__ == "__main__":
    run(print_ids)
