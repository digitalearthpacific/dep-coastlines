import json
import sys

from typing import Annotated, Optional

from typer import Typer, Option

from azure_logger import CsvLogger, filter_by_log
from dep_tools.namers import DepItemPath
from dep_tools.utils import get_container_client

from dep_coastlines.grid import test_grid as GRID
from time_utils import parse_datetime, composite_from_years

app = Typer()


def get_ids(
    datetime,
    version,
    dataset_id,
    retry_errors=True,
    grid=GRID,
    delete_existing_log: bool = False,
) -> list:
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
        overwrite=delete_existing_log,
        header="time|index|status|paths|comment\n",
    )
    return filter_by_log(grid, logger.parse_log(), retry_errors).index.to_list()


def int_or_none(raw: str) -> Optional[int]:
    return None if raw == "None" else int(raw)


def cs_list_of_ints(raw: str) -> list[int] | int:
    return [int(s) for s in raw.split(",")] if "," in raw else int(raw)


def bool_parser(raw: str):
    return False if raw == "False" else True


@app.command()
def print_ids(
    dataset_id: Annotated[str, Option()],
    version: Annotated[str, Option()],
    datetime: Annotated[str, Option()],
    years_per_composite: Annotated[str, Option(parser=cs_list_of_ints)] = "1",
    limit: Optional[str] = Option(None, parser=int_or_none),
    # Would be better to make this an Optional[bool] but argo can't do that
    retry_errors: Annotated[str, Option(parser=bool)] = "True",
    overwrite_logs: Annotated[str, Option(parser=bool)] = "False",
):
    ids = get_ids(
        datetime=datetime,
        version=version,
        dataset_id=dataset_id,
        retry_errors=retry_errors,
        delete_existing_log=overwrite_logs,
    )

    years = composite_from_years(parse_datetime(datetime), years_per_composite)
    params = [
        {"row": id[0], "column": id[1], "datetime": year}
        for id in ids
        for year in years
    ]

    if limit is not None:
        params = params[0 : int(limit)]

    json.dump(params, sys.stdout)


if __name__ == "__main__":
    app()
