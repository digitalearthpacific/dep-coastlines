from dea_tools.coastal import pixel_tides
import netCDF4  # see note in calculate_tides.py
from odc.geo import Resolution
from odc.geo.geom import Geometry
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros
from xarray import concat

from dep_coastlines.tide_utils import tides_lowres, tide_cutoffs_lr


def load_tides(gdf):
    tide_loader = TideLoader(TIDES_NAMER)
    mins = []
    maxes = []
    for _, row in gdf.iterrows():
        amin, amax = tide_cutoffs_lr(tide_loader.load((row.column, row.row)))
        mins.append(amin.expand_dims(id=1))
        maxes.append(amax.expand_dims(id=1))

    # I assume that areas won't overlap, but we might need to consider that
    return concat(mins, dim="id").mean(dim="id"), concat(maxes, dim="id").mean(dim="id")


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
