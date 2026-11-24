from datetime import datetime

import pytest
from create_mosaic import ItemFilterer, _run_single_task
from dep_tools.exceptions import EmptyCollectionError
from pystac import Item


def _item(id: str, dt: datetime | None) -> Item:
    # pystac requires start/end datetimes when datetime is None
    props = (
        {}
        if dt is not None
        else {
            "start_datetime": "2000-01-01T00:00:00Z",
            "end_datetime": "2000-12-31T00:00:00Z",
        }
    )
    return Item(id, None, None, dt, props)


@pytest.fixture
def items() -> list[Item]:
    return [_item(f"i{year}", datetime(year, 6, 1)) for year in range(1999, 2004)]


def _ids(filterer: ItemFilterer) -> list[str]:
    return [item.id for item in filterer.search()]


def test_filter_single_year(items):
    assert _ids(ItemFilterer(items, "2003")) == ["i2003"]


def test_filter_year_range_is_inclusive(items):
    assert _ids(ItemFilterer(items, "2000/2002")) == ["i2000", "i2001", "i2002"]


def test_filter_skips_items_without_datetime(items):
    items.append(_item("no_datetime", None))
    assert _ids(ItemFilterer(items, "2000")) == ["i2000"]


def test_filter_raises_when_nothing_matches(items):
    with pytest.raises(EmptyCollectionError):
        ItemFilterer(items, "2010")


class _FakeTask:
    def __init__(self, result):
        self._result = result

    def run(self):
        return self._result


def test_run_single_task_wraps_a_single_path():
    assert _run_single_task(_FakeTask("a.tif")) == ["a.tif"]


def test_run_single_task_keeps_a_list_of_paths():
    assert _run_single_task(_FakeTask(["a.tif", "b.tif"])) == ["a.tif", "b.tif"]
