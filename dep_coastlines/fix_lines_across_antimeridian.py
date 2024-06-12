import geopandas as gpd
from typer import run

from dep_tools.utils import shift_negative_longitudes


def fix_lines_across_antimeridian(
    input_file: str, output_file: str, input_kwargs=dict(), output_kwargs=dict()
):
    breakpoint()
    d = gpd.read_file(input_file, **input_kwargs).to_crs(4326)
    d["geometry"] = d.geometry.apply(shift_negative_longitudes)
    d.to_file(output_file, **output_kwargs)


if __name__ == "__main__":
    run(fix_lines_across_antimeridian)
