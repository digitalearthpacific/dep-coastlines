from pathlib import Path

import lzma
import typer
import xarray as xr


def clip_tides(input_dir: Path, output_dir: Path) -> None:
    for path in input_dir.glob("*/*.xz"):
        with lzma.open(path) as nc:
            print(path)
            src = xr.open_dataset(nc)
            output = src.where(
                (src.lat > -28) & (src.lat < 21) & (src.lon > 100) & (src.lon < 235),
                drop=True,
            )
            encoding = {var: {"zlib": True} for var in output}
            full_output_dir = output_dir / path.parent.relative_to(input_dir)
            full_output_dir.mkdir(parents=True, exist_ok=True)
            output.to_netcdf(
                f"{full_output_dir}/{path.with_suffix('').stem}.nc",
                encoding=encoding,
                engine="h5netcdf",
            )


if __name__ == "__main__":
    typer.run(clip_tides)
