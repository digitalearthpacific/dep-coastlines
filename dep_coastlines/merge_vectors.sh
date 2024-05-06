#!/usr/bin/env bash

version=$1

gpkg_epsg=3832
ODIR=dep_ls_coastlines/coastlines
output=${ODIR}/coastlines_${version}_$gpkg_epsg.gpkg
output_geojson=${ODIR}/coastlines_$version.geojson
output_4326=${ODIR}/coastlines_${version}_4326.gpkg
output_exploded=data/$version/coastlines_${version}_exploded.gpkg
tile_folder=data/tiles/coastlines-$version

roc_output=${ODIR}/coastlines_${version}_roc_$gpkg_epsg.gpkg

INPUTS=`find $ODIR/$version/ -type f -name "*3.gpkg"`
#az storage fs directory download -f output -s $az_src -d $dst --recursive 
ogrmerge.py -o tmp.gpkg $INPUTS -field_strategy Union -single -nln coastlines -overwrite_ds -t_srs EPSG:$gpkg_epsg

# Fixes feature misordering across files
ogr2ogr $output -dialect sqlite tmp.gpkg -nln coastlines -sql "select year, certainty, st_union(geom) as geom from coastlines group by year, certainty"
# rm tmp.gpkg

python dep_coastlines/fix_lines_across_antimeridian.py $output $output_4326

ogr2ogr $output_geojson $output_4326

INPUTS=`find $ODIR/$version/ -type f -name "*roc.gpkg"`
ogrmerge.py -o $roc_output $INPUTS -field_strategy Union -single -nln rates_of_change -overwrite_ds -t_srs EPSG:$gpkg_epsg


#azcopy copy $output https://deppcpublicstorage.blob.core.windows.net/output/coastlines/$version/?$AZURE_STORAGE_SAS_TOKEN --from-to=LocalBlob --blob-type BlockBlob 
#
#azcopy copy $output_geojson https://deppcpublicstorage.blob.core.windows.net/output/coastlines/$version/?$AZURE_STORAGE_SAS_TOKEN --from-to=LocalBlob --blob-type BlockBlob 
##
#mkdir -p $tile_folder
#tippecanoe -e $tile_folder/coastlines -z 12 -l coastlines $output_geojson --force
#
#ogr2ogr -nlt MULTILINESTRING $output_exploded $tile_folder/coastlines/12
#
#azcopy copy $tile_folder/coastlines https://deppcpublicstorage.blob.core.windows.net/output/tiles/$prefix/?$AZURE_STORAGE_SAS_TOKEN --from-to=LocalBlob --blob-type BlockBlob --content-type application/vnd.mapbox-vector-tile --content-encoding gzip --recursive
