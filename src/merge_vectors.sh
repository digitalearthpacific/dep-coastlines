#!/usr/bin/env bash

version=3Aug2023
# not tested yet 
az_src="coastlines/$version/lines"
dst=data/$version
output=data/$version/coastlines.gpkg
output_geojson=data/$version/coastlines.geojson
tile_folder=data/tiles/$version

mkdir -p $dst
az storage fs directory download -f output -s $az_src -d $dst --recursive 
ogrmerge.py -o $output $dst/lines/*.gpkg -single -nln coastlines -overwrite_ds

# python src/fix_lines_across_antimeridian.py $output $output_geojson

#mkdir -p $tile_folder
#tippecanoe -e $tile_folder/coastlines -z 12 -l coastlines data/coastlines.geojson --force
#
#az storage blob upload-batch -d output/tiles/$version/coastlines -s $tile_folder/coastlines --max-connections 10 --content-type application/vnd.mapbox-vector-tile --content-encoding gzip --overwrite
#

#export AZCOPY_CRED_TYPE=Anonymous;
#export AZCOPY_CONCURRENCY_VALUE=AUTO;
#azcopy copy data/tiles/coastlines https://deppcpublicstorage.blob.core.windows.net/output/tiles/?$AZURE_STORAGE_SAS_TOKEN --from-to=LocalBlob --blob-type BlockBlob --content-type application/vnd.mapbox-vector-tile --content-encoding gzip --recursive
#unset AZCOPY_CRED_TYPE;
#unset AZCOPY_CONCURRENCY_VALUE;

