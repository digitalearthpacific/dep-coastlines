from dask.dataframe import multi
import geopandas as gpd
from shapely.geometry import Point


def linestring_to_points(linestring, mls_id, ls_id):
    return [
        {"mls_id": mls_id, "ls_id": ls_id, "pt_id": i, "geometry": Point(coords)}
        for i, coords in enumerate(linestring.coords)
    ]


def multilinestring_to_points(multilinestring, id):
    return [
        linestring_to_points(linestring, id, i)
        for i, linestring in enumerate(multilinestring.geoms)
    ]


def gpd_to_points(gpdf):
    for _, row in gpdf.iterrows():
        for linestring_id, linestring in enumerate(row.geometry.geoms):
            for coord_index, coords in enumerate(linestring.coords):
                yield {
                    "year": row["year"],
                    "ls_id": linestring_id,
                    "pt_id": coord_index,
                    "geometry": Point(coords),
                }


x = gpd.read_file("mndwi-clean-lines_70_75.gpkg")

points = [multilinestring_to_points(mls, i) for i, mls in enumerate(x.geometry)]

flat_points = sum(sum(points, []), [])
o = gpd.GeoDataFrame.from_records(flat_points)
breakpoint()

# xr = rioxarray.open_rasterio("mask.tif", chunks=True)
# extrct = xr.sel(dict(x=o.geometry.x, y=o.geometry.y))
