#!/usr/bin/env bash

# not tested yet 
#az storage fs directory download -f output -s "coastlines-vector" -d data --recursive
#ogrmerge.py -o data/coastlines.gpkg data/coastlines-vector/*.gpkg -single -nln coastlines

#python src/fix_lines_across_antimeridian.py data/coastlines.gpkg data/coastlines.geojson

mkdir -p data/tiles
tippecanoe -e data/tiles/coastlines -z 12 -l coastlines data/coastlines.geojson --force

#az storage blob upload-batch -d output/tiles/coastlines -s data/tiles/coastlines --max-connections 10 --content-type application/vnd.mapbox-vector-tile --content-encoding gzip --overwrite


export AZCOPY_CRED_TYPE=Anonymous;
export AZCOPY_CONCURRENCY_VALUE=AUTO;
azcopy copy data/tiles/coastlines https://deppcpublicstorage.blob.core.windows.net/output/tiles/?$AZURE_STORAGE_SAS_TOKEN --from-to=LocalBlob --blob-type BlockBlob --content-type application/vnd.mapbox-vector-tile --content-encoding gzip --recursive
unset AZCOPY_CRED_TYPE;
unset AZCOPY_CONCURRENCY_VALUE;

