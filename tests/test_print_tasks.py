import json

import print_tasks
import pytest
from typer import Typer
from typer.testing import CliRunner


def test_task_params_groups_by_period():
    params = print_tasks._task_params(
        periods=["2000", "2001"],
        get_ids=lambda period: [(10, 20), (11, 21)],
    )
    assert params == [
        {"column": 10, "row": 20, "datetime": "2000"},
        {"column": 11, "row": 21, "datetime": "2000"},
        {"column": 10, "row": 20, "datetime": "2001"},
        {"column": 11, "row": 21, "datetime": "2001"},
    ]


def test_task_params_asks_for_each_period_once():
    seen = []

    def get_ids(period):
        seen.append(period)
        return []

    assert print_tasks._task_params(["2000", "2001"], get_ids) == []
    assert seen == ["2000", "2001"]


# The command-line tests below run the real command, so they replace
# _get_ids, the only step that reads from S3.


@pytest.fixture
def get_ids_calls(monkeypatch) -> list[tuple[str, dict]]:
    calls = []

    def fake_get_ids(datetime, **kwargs):
        calls.append((datetime, kwargs))
        return [(1, 2)]

    monkeypatch.setattr(print_tasks, "_get_ids", fake_get_ids)
    return calls


def _invoke(*args: str) -> list[dict]:
    app = Typer()
    app.command()(print_tasks.print_ids)
    result = CliRunner().invoke(app, ["--dataset-id", "ds", "--version", "v1", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_print_ids_expands_year_range(get_ids_calls):
    _invoke("--datetime", "2000_2002")
    assert [period for period, _ in get_ids_calls] == ["2000", "2001", "2002"]


def test_print_ids_passes_slash_range_through(get_ids_calls):
    _invoke("--datetime", "2000/2024")
    assert [period for period, _ in get_ids_calls] == ["2000/2024"]


def test_print_ids_defaults(get_ids_calls):
    _invoke("--datetime", "2000")
    assert get_ids_calls[0][1] == {
        "version": "v1",
        "dataset_id": "ds",
        "retry_errors": True,
        "delete_existing_log": False,
        "filter_using_log": True,
        "filter_existing_stac_items": False,
    }


def test_print_ids_booleans_are_false_only_for_exact_false(get_ids_calls):
    _invoke(
        "--datetime",
        "2000",
        "--retry-errors",
        "False",
        "--filter-using-log",
        "false",
        "--overwrite-logs",
        "yes",
    )
    kwargs = get_ids_calls[0][1]
    assert kwargs["retry_errors"] is False
    # Only the exact string "False" is false
    assert kwargs["filter_using_log"] is True
    # --overwrite-logs is passed on as delete_existing_log
    assert kwargs["delete_existing_log"] is True


def test_print_ids_writes_json(get_ids_calls):
    assert _invoke("--datetime", "2000") == [
        {"column": 1, "row": 2, "datetime": "2000"}
    ]
