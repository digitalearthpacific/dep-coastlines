def parse_datetime(datetime: str) -> list[str]:
    if "," in datetime:
        datetime_sets = datetime.split(",")
        output = []
        for a_datetime in datetime_sets:
            output += _years_from_datetime(a_datetime)
    else:
        output = _years_from_datetime(datetime)

    # remove duplicates
    return list(set(output))


def composite_from_years(years: list, years_per_composite: list | int = 1) -> list[str]:
    if years_per_composite == 1:
        return years

    if isinstance(years_per_composite, list):
        output = []
        for a_years_per_composite in years_per_composite:
            output += composite_from_years(years, a_years_per_composite)
        return output

    assert years_per_composite % 2 == 1
    year_buffer = int((years_per_composite - 1) / 2)
    return [f"{int(year) - year_buffer}/{int(year) + year_buffer}" for year in years]


def _years_from_datetime(datetime: str) -> list[str]:
    if "_" in datetime:
        years = datetime.split("_")
        if len(years) == 2:
            years = range(int(years[0]), int(years[1]) + 1)
        elif len(years) > 2:
            ValueError(f"{datetime} is not a valid value for --datetime")
        return [str(y) for y in years]
    else:
        return [datetime]
