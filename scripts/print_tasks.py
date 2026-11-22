"""Print a list of ids which need to be processed."""

import json
import sys
from typing import Annotated, cast

import pandas as pd
from cloud_logger import filter_by_log
from dep_tools.parsers import bool_parser, datetime_parser
from dep_tools.stac_utils import remove_items_with_existing_stac
from typer import Option, run

from dep_coastlines.common import (
    coastlineItemPath,
    coastlineLogger,
)
from dep_coastlines.grid import buffered_grid


def print_ids(
    dataset_id: Annotated[str, Option()],
    version: Annotated[str, Option()],
    datetime: Annotated[str, Option(parser=datetime_parser)],
    retry_errors: Annotated[str, Option(parser=bool_parser)] = "True",
    overwrite_logs: Annotated[str, Option(parser=bool_parser)] = "False",
    filter_using_log: Annotated[str, Option(parser=bool_parser)] = "True",
    filter_existing_stac_items: Annotated[str, Option(parser=bool_parser)] = "False",
):
    """Print a list of tile ids and time periods to process.

    Boolean options are false only when given the exact string "False"; any
    other value is true.

    Args:
        dataset_id: The dataset id.
        version: The output data version.
        datetime: The time period(s) to process, parsed by
            :func:`dep_tools.parsers.datetime_parser`. One of:

            - ``<year>``: one task per tile for that year.
            - ``<year 1>_<year 2>``: one task per tile for each year in the
              inclusive range.
            - Any other string, such as ``2000/2024``: one task per tile, with
              the string passed through unchanged as a single time period.
        retry_errors: If `filter_using_log` is true, whether to retry tasks that
            were previously logged as errors.
        overwrite_logs: Whether to delete the log for each time period before
            processing. The log is deleted even if `filter_using_log` is false.
        filter_using_log: Whether to filter tasks to run using the log.
        filter_existing_stac_items: Whether to filter tasks to run by excluding
            those which already have an output STAC Item stored as json in the
            output location.

    Returns: Nothing. A JSON list of objects of the form
        ``{"column": int, "row": int, "datetime": str}`` is written to stdout.
    """
    params = _task_params(
        dataset_id=dataset_id,
        version=version,
        periods=cast(list[str], datetime),
        retry_errors=cast(bool, retry_errors),
        overwrite_logs=cast(bool, overwrite_logs),
        filter_using_log=cast(bool, filter_using_log),
        filter_existing_stac_items=cast(bool, filter_existing_stac_items),
    )
    json.dump(params, sys.stdout)


def _task_params(
    dataset_id: str,
    version: str,
    periods: list[str],
    retry_errors: bool = True,
    overwrite_logs: bool = False,
    filter_using_log: bool = True,
    filter_existing_stac_items: bool = False,
) -> list[dict[str, int | str]]:
    """Build the task parameters for each tile and time period.

    See :func:`print_ids` for a description of the arguments.

    Returns: A list of dictionaries with keys `column`, `row`, and `datetime`.
    """
    return [
        {"column": tile_id[0], "row": tile_id[1], "datetime": period}
        for period in periods
        for tile_id in _get_ids(
            datetime=period,
            version=version,
            dataset_id=dataset_id,
            retry_errors=retry_errors,
            delete_existing_log=overwrite_logs,
            filter_using_log=filter_using_log,
            filter_existing_stac_items=filter_existing_stac_items,
        )
    ]


def _get_ids(
    datetime: str,
    version: str,
    dataset_id: str,
    retry_errors: bool = True,
    grid: pd.DataFrame | None = None,
    delete_existing_log: bool = False,
    filter_existing_stac_items: bool = False,
    filter_using_log: bool = True,
) -> list[tuple[int, int]]:
    """Get the ids of the tiles to process for a single time period.

    Tiles with an existing STAC Item are removed first (if
    `filter_existing_stac_items` is true), then tiles are filtered using the
    log (if `filter_using_log` is true).

    Args:
        datetime: A single time period, e.g. "2024" or "2000/2024".
        version: The output data version.
        dataset_id: The dataset id.
        retry_errors: Whether to keep tiles previously logged as errors.
        grid: The grid to select tiles from, indexed by (column, row). Defaults
            to :func:`dep_coastlines.grid.buffered_grid`.
        delete_existing_log: Whether to delete the log before reading it.
        filter_existing_stac_items: Whether to remove tiles which already have
            an output STAC Item.
        filter_using_log: Whether to remove tiles using the log.

    Returns: The (column, row) ids of the tiles to process.
    """
    tiles: pd.DataFrame = buffered_grid() if grid is None else grid
    namer = coastlineItemPath(dataset_id=dataset_id, version=version, time=datetime)
    logger = coastlineLogger(
        namer,
        dataset_id=dataset_id,
        delete_existing_log=delete_existing_log,
    )

    if filter_existing_stac_items:
        tiles = remove_items_with_existing_stac(tiles, namer)

    if filter_using_log:
        # parse_log is typed as possibly returning None, but it is just
        # pd.read_csv and never does.
        log = cast(pd.DataFrame, logger.parse_log())
        tiles = filter_by_log(tiles, log, retry_errors)

    return tiles.index.to_list()


if __name__ == "__main__":
    run(print_ids)
