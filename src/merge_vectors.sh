#!/usr/bin/env bash

version=$1
az_src="coastlines/$version/lines"
dst=data/$version
output=data/$version/coastlines_$version.gpkg
output_geojson=data/$version/coastlines_$version.geojson
tile_folder=data/tiles/$version

mkdir -p $dst
az storage fs directory download -f output -s $az_src -d $dst --recursive 
ogrmerge.py -o tmp.gpkg $dst/lines/*.gpkg -field_strategy Union -single -nln coastlines -overwrite_ds
# Fixes feature misordering across files
ogr2ogr $output -dialect sqlite tmp.gpkg -nln coastlines -sql "select year as year, st_union(geom) as geom from coastlines group by year"
rm tmp.gpkg

python src/fix_lines_across_antimeridian.py $output $output_geojson


mkdir -p $tile_folder
tippecanoe -e $tile_folder/coastlines -z 12 -l coastlines $output_geojson --force

azcopy copy $tile_folder/coastlines https://deppcpublicstorage.blob.core.windows.net/output/tiles/$version/?$AZURE_STORAGE_SAS_TOKEN --from-to=LocalBlob --blob-type BlockBlob --content-type application/vnd.mapbox-vector-tile --content-encoding gzip --recursive

