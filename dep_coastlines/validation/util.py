import geopandas as gpd
import netCDF4  # see note in calculate_tides.py
from odc.geo import Resolution
from odc.geo.geom import Geometry
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros
from retry import retry
import rioxarray as rx
from rioxarray.merge import merge_arrays

from dep_coastlines.common import coastlineItemPath
from dep_coastlines.config import (
    VECTOR_VERSION,
    VECTOR_DATASET_ID,
    VECTOR_DATETIME,
    HTTPS_PREFIX,
    CURRENT_COASTLINES_OUTPUT,
    NEXT_GEN_COASTLINES_OUTPUT,
)
from dep_coastlines.grid import buffered_grid as GRID
from dep_coastlines.tide_utils import tides_lowres, tide_cutoffs_lr


@retry(tries=3)
def load_coastlines(
    geom,
    layer: str = "shorelines_annual",
    year: int | None = None,
    buffer: float = 100,
    use_next_gen: bool = False,
):
    buffered_geom = geom.buffer(buffer) if buffer is not None else geom
    coastlines_file = (
        NEXT_GEN_COASTLINES_OUTPUT if use_next_gen else CURRENT_COASTLINES_OUTPUT
    )
    coastlines = gpd.read_file(
        coastlines_file,
        layer=layer,
        bbox=(
            buffered_geom.total_bounds.tolist()
            if isinstance(buffered_geom, gpd.GeoDataFrame)
            else buffered_geom.union_all().bounds
        ),
        engine="pyogrio",
        use_arrow=True,
    )
    if year:
        coastlines = coastlines[coastlines.year == year]
    return coastlines


@retry(tries=3)
def load_coastlines_raster_for_geometry(geometry, year, grid=GRID):
    # get index of grid cells for geometry
    indices = grid[grid.intersects(geometry.union_all())].index
    itemPath = coastlineItemPath(
        dataset_id=VECTOR_DATASET_ID, version=VECTOR_VERSION, time=VECTOR_DATETIME
    )

    def load_file(path, year):
        da = rx.open_rasterio(path, chunks=True)
        da_year = da.isel(band=da.attrs["long_name"].index(str(year)))
        da_year.attrs["long_name"] = None
        return da_year

    if len(indices) > 1:
        return merge_arrays(
            [
                load_file(f"{HTTPS_PREFIX}/{itemPath.path(item_id)}", year)
                for item_id in indices
            ]
        )
    return load_file(f"{HTTPS_PREFIX}/{itemPath.path(indices[0])}", year)


def _get_da(gdf, crs=None, time=None):
    assigned_crs = gdf.crs if crs is None else crs
    resolution = Resolution(3300, 3300)

    geobox = GeoBox.from_geopolygon(
        Geometry(gdf.iloc[0].geometry, crs=gdf.crs).to_crs(assigned_crs),
        resolution=resolution,
    )
    assigned_time = time if time is not None else [gdf.iloc[0]["time"]]
    return xr_zeros(
        geobox,
        chunks=(1000, 1000),
    ).expand_dims(time=assigned_time)


def make_tides(gdf, crs=None, time=None):
    return tides_lowres(_get_da(gdf, crs, time)).squeeze(drop=True)
