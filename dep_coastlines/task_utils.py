from typing import Annotated, Optional
import json
import sys

from typer import Typer, Option
from cloud_logger import filter_by_log
from dep_tools.stac_utils import remove_items_with_existing_stac

from dep_coastlines.common import (
    coastlineItemPath,
    coastlineLogger,
    int_or_none,
    cs_list_of_ints,
    bool_parser,
)
from dep_coastlines.grid import buffered_grid as GRID
from dep_coastlines.time_utils import parse_datetime, composite_from_years

app = Typer()


def get_ids(
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


@app.command()
def print_ids(
    dataset_id: Annotated[str, Option()],
    version: Annotated[str, Option()],
    datetime: Annotated[str, Option()],
    years_per_composite: Annotated[str, Option(parser=cs_list_of_ints)] = "1",
    limit: Optional[str] = Option(None, parser=int_or_none),
    retry_errors: Annotated[str, Option(parser=bool_parser)] = "True",
    overwrite_logs: Annotated[str, Option(parser=bool_parser)] = "False",
    filter_using_log: Annotated[str, Option(parser=bool_parser)] = "True",
    filter_existing_stac_items: Annotated[str, Option(parser=bool_parser)] = "False",
):
    params = []
    # All the casting here is just to appease the linter; the parsers handle
    # the actual conversions.
    for year in composite_from_years(parse_datetime(datetime), years_per_composite):
        ids = get_ids(
            datetime=year,
            version=version,
            dataset_id=dataset_id,
            retry_errors=bool(retry_errors),
            delete_existing_log=bool(overwrite_logs),
            filter_using_log=bool(filter_using_log),
            filter_existing_stac_items=bool(filter_existing_stac_items),
        )

        params += [{"column": id[0], "row": id[1], "datetime": year} for id in ids]

    if limit is not None:
        params = params[0 : int(limit)]

    json.dump(params, sys.stdout)


if __name__ == "__main__":
    app()
