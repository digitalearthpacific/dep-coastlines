import geopandas as gpd
from shapely import make_valid

# We use 8859 so distances will be in meters
aoi = gpd.read_file("data/aoi.gpkg")

# Approx 2km at equator, other CRSes did not work (horizontal stripes that
# spanned the globe).
# Note this will give a warning about geographic buffers being incorrect,
# but I checked and this is fine
buffer_distance_dd = 0.018018018018018018

# A buffer of the exterior line created weird interior gaps,
# so I buffer the polygons by a positive then negative buffer
# and take the difference
aoi_buffer = aoi.buffer(buffer_distance_dd)
aoi_negative_buffer = aoi.buffer(-buffer_distance_dd)

coast_buffer = make_valid(aoi_buffer.difference(aoi_negative_buffer).unary_union)

pathrows = gpd.read_file(
    "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip"
)

intersection = pathrows.intersection(coast_buffer)

pathrows.geometry = intersection
output = pathrows[~pathrows.is_empty]

output.to_file("data/coastline_split_by_pathrow.gpkg")
