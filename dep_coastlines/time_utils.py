def composite_from_years(
    years: list[str], years_per_composite: list | int = 1
) -> list[str | int]:
    """Convert a list of years to a list of multi-year composites.

    Args:
        years: A list of years.
        years_per_composite: A single integer or a list of integers, representing
            how many years to include in each composite. Must be a positive odd
            integer.

    Returns: If `years_per_composite` is 1, then the input list is returned
        unaltered. If a larger integer, then a multi-year range including the
        year as the center point of a range containing that number of years.
        If a list of integers, then a list of all composites created from all
        years and values. Output ranges are of the form <year 1>/<year 2>,
        to be passed to e.g. a pystac search.
    """
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
