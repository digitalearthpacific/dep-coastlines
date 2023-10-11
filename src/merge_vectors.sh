#!/usr/bin/env bash

version=$1
clean_src=$2
prefix=$1/$2
az_src="coastlines/$prefix-lines"
dst=data/$version

gpkg_epsg=3832
output=data/$version/coastlines_${version}_$gpkg_epsg.gpkg
output_geojson=data/$version/coastlines_$version.geojson
output_4326=data/$version/coastlines_${version}_4326.gpkg
tile_folder=data/tiles/coastlines-$version

mkdir -p $dst
# az storage fs directory download -f output -s $az_src -d $dst --recursive 
#ogrmerge.py -o tmp.gpkg -t_srs $dst/$clean_src-lines/*.gpkg -field_strategy Union -single -nln coastlines -overwrite_ds -t_srs EPSG:$gpkg_epsg
ogrmerge.py -o data/$version/coastlines_${version}_${gpkg_epsg}_exploded.gpkg -t_srs $dst/$clean_src-lines/*.gpkg -field_strategy Union -single -nln coastlines -overwrite_ds -t_srs EPSG:$gpkg_epsg
## Fixes feature misordering across files
#ogr2ogr $output -dialect sqlite tmp.gpkg -nln coastlines -sql "select year as year, st_union(geom) as geom from coastlines group by year"
#rm tmp.gpkg

#python src/fix_lines_across_antimeridian.py $output $output_4326
#
#ogr2ogr $output_geojson $output_4326
#
#azcopy copy $output https://deppcpublicstorage.blob.core.windows.net/output/coastlines/$version/?$AZURE_STORAGE_SAS_TOKEN --from-to=LocalBlob --blob-type BlockBlob 
#
#azcopy copy $output_geojson https://deppcpublicstorage.blob.core.windows.net/output/coastlines/$version/?$AZURE_STORAGE_SAS_TOKEN --from-to=LocalBlob --blob-type BlockBlob 
##
#mkdir -p $tile_folder
#tippecanoe -e $tile_folder/coastlines -z 12 -l coastlines $output_geojson --force
#
#azcopy copy $tile_folder/coastlines https://deppcpublicstorage.blob.core.windows.net/output/tiles/$prefix/?$AZURE_STORAGE_SAS_TOKEN --from-to=LocalBlob --blob-type BlockBlob --content-type application/vnd.mapbox-vector-tile --content-encoding gzip --recursive
