#!/usr/bin/env bash

version=$1
clean_src=$2
prefix=$1/$2
az_src="coastlines/$prefix-lines"
dst=data/$version
output=data/$version/$clean_src-$version.gpkg
output_geojson=data/$version/$clean_src-$version.geojson
tile_folder=data/tiles/$clean_src-$version

mkdir -p $dst
az storage fs directory download -f output -s $az_src -d $dst --recursive 
ogrmerge.py -o tmp.gpkg $dst/$clean_src-lines/*.gpkg -field_strategy Union -single -nln coastlines -overwrite_ds
# Fixes feature misordering across files
ogr2ogr $output -dialect sqlite tmp.gpkg -nln coastlines -sql "select year as year, st_union(geom) as geom from coastlines group by year"
rm tmp.gpkg

python src/fix_lines_across_antimeridian.py $output $output_geojson


mkdir -p $tile_folder
tippecanoe -e $tile_folder/coastlines -z 12 -l coastlines $output_geojson --force

azcopy copy $tile_folder/coastlines https://deppcpublicstorage.blob.core.windows.net/output/tiles/$prefix/?$AZURE_STORAGE_SAS_TOKEN --from-to=LocalBlob --blob-type BlockBlob --content-type application/vnd.mapbox-vector-tile --content-encoding gzip --recursive

