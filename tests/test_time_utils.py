from dep_coastlines.time_utils import composite_from_years


def test_single_year_composites_are_unchanged():
    assert composite_from_years(["2000", "2001"]) == ["2000", "2001"]


def test_three_year_composites_are_centred_on_each_year():
    assert composite_from_years(["2000", "2001"], 3) == ["1999/2001", "2000/2002"]


def test_list_of_sizes_concatenates_composites_in_order():
    assert composite_from_years(["2000", "2001"], [1, 3]) == [
        "2000",
        "2001",
        "1999/2001",
        "2000/2002",
    ]
