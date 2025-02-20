from pathlib import Path
import warnings

import boto3
import lzma
import typer
import xarray as xr

from dep_coastlines.config import BUCKET


def clip_tide_data(input_dir: Path, output_dir: Path, copy_to_s3: bool = False) -> None:
    for path in input_dir.glob("*/*.xz"):
        with lzma.open(path) as nc:
            print(path)
            src = xr.open_dataset(nc, engine="h5netcdf")
            output = src.where(
                (src.lat > -28) & (src.lat < 24) & (src.lon > 130) & (src.lon < 236),
                drop=True,
            )
            subdir = path.parent.relative_to(input_dir)
            output_filename = f"{path.with_suffix('').stem}.nc"

            full_output_dir = output_dir / subdir
            full_output_dir.mkdir(parents=True, exist_ok=True)

            output_path = f"{full_output_dir}/{output_filename}"
            remote_path = f"dep_ls_coastlines/raw/tidal_models/fes2022b/{subdir}/{output_filename}"

            encoding = {var: _fix_encoding(src[var].encoding) for var in src}
            output.to_netcdf(
                output_path,
                encoding=encoding,
                engine="h5netcdf",
            )

            if copy_to_s3:
                client = boto3.client("s3")
                client.upload_file(output_path, Bucket=BUCKET, Key=remote_path)


def _fix_encoding(encoding: dict) -> dict:
    good_keys = ["shuffle", "zlib", "complevel", "dtype", "_FillValue"]
    return {k: v for k, v in encoding.items() if k in good_keys}


def write_urls(url_file_path):
    client = boto3.client("s3")
    with open(url_file_path, "w") as dst:
        for o in client.list_objects(
            Bucket=BUCKET, Prefix="dep_ls_coastlines/raw/tidal_models/fes2022b"
        )["Contents"]:
            if not "non_extrapolated" in o["Key"]:
                dst.write(f"https://{BUCKET}.s3.us-west-2.amazonaws.com/{o['Key']}\n")


def main(
    input_dir: Path,
    output_dir: Path = Path("data/raw/tidal_models/fes2022b"),
    copy_to_s3: bool = True,
    write_urls_to_file: bool = True,
    url_file_path="data/tide_data_urls.txt",
):
    clip_tide_data(input_dir=input_dir, output_dir=output_dir, copy_to_s3=copy_to_s3)
    if write_urls_to_file:
        if not copy_to_s3:
            warnings.warn("writing urls to file, but copy_to_s3 not specified!")
        write_urls(url_file_path)


if __name__ == "__main__":
    typer.run(main)
