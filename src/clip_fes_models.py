from pathlib import Path
import tarfile
from typing import Iterable

import geopandas as gpd
import rioxarray
import xarray as xr


def clip_tidal_data(xr: xr.Dataset, geometries: Iterable) -> xr.Dataset:
    # Do this in 2 steps rather than using rio.clip directly because some of
    # the variables in the tidal data do not have both spatial dimensions
    clipped_dims = (
        xr.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)["phase"]
        .rio.write_crs(4326)
        .rio.clip(geometries)
    )

    return xr.sel(lon=clipped_dims.lon, lat=clipped_dims.lat)


def clip_tidal_folder(
    input_folder: Path, clip_geometry: Iterable, output_folder: Path
) -> None:
    for f in input_folder.rglob("*.nc"):
        input_ds = xr.open_dataset(f)
        output_ds = clip_tidal_data(input_ds, clip_geometry)
        output_subfolder = output_folder / Path(*f.parts[2:-1])
        output_subfolder.mkdir(parents=True, exist_ok=True)
        output_path = output_subfolder / Path(f.name)
        output_ds.to_netcdf(output_path)
        print(output_path)


if __name__ == "__main__":
    fes_folder = Path("data/fes2014")
    aoi_file = "../DigitalEarthPacific/data/aoi.gpkg"
    aoi_geom = gpd.read_file(aoi_file).unary_union
    clip_tidal_folder(fes_folder, aoi_geom, "data/fes2014_clipped")
