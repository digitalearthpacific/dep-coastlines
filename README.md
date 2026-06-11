# Digital Earth Pacific Coastlines

This is a work-in-progress adaptation of [Digital Earth Australia (DEA)](https://github.com/GeoscienceAustralia/dea-coastlines) and [Digital Earth
Africa
Coastlines](https://github.com/digitalearthafrica/deafrica-coastlines) products
for the Pacific nations. Please refer to the following for a description of the
algorithms on which we based this work:

> Bishop-Taylor, R., Nanson, R., Sagar, S., Lymburner, L. (2021). Mapping
> Australia's dynamic coastline at mean sea level using three decades of Landsat
> imagery. _Remote Sensing of Environment_, 267, 112734. Available:
> https://doi.org/10.1016/j.rse.2021.112734

> Bishop-Taylor, R., Sagar, S., Lymburner, L., Alam, I., Sixsmith, J. (2019). Sub-pixel waterline extraction: characterising accuracy and sensitivity to indices and spectra. _Remote Sensing_, 11 (24):2984. Available: https://doi.org/10.3390/rs11242984

We have adapted the algorithms where necessary to account for differences in the
study area and the available input datasets, as well as for performance. A full
writeup is in process.

The product is visible on the [Digital Earth Pacific](https://digitalearthpacific.org) [map interface](https://maps.digitalearthpacific.org).

## Notes for running for recent years

Configure the raster analysis to overwrite, and run for three years (like, for 2025, use 2024/2026).

## Command Line Tools

The coastlines process runs as a three-stage pipeline (raster → vector →
continental), with a shared task ID generator and several supporting utilities.
Argo Workflow definitions for running at scale are in the `.argo/` directory.

### `dep_coastlines/task_utils.py` — Generate Task IDs

Prints a JSON list of grid cell IDs to process, filtering by log status and
existing STAC items. Used by the Argo Workflows to fan out raster and vector
processing.

```bash
python dep_coastlines/task_utils.py \
  --dataset-id coastlines/interim/mosaic \
  --version 0.8.1 \
  --datetime 1984/2024 \
  --limit None \
  --retry-errors True \
  --overwrite-logs False \
  --filter-using-log True \
  --filter-existing-stac-items False
```

### `dep_coastlines/raster.py` — Raster Processing (Stage 1)

Calculates water indices (e.g., TWNDWI) from Landsat imagery for a given grid
cell and saves the output rasters to S3.

```bash
python dep_coastlines/raster.py \
  --column 57 \
  --row 21 \
  --version 0.8.1 \
  --load-before-write True \
  --fail-on-read-error True
```

### `dep_coastlines/vector.py` — Vector Processing (Stage 2)

Post-processes water index rasters into annual shoreline vectors and rates of
change statistics, then writes the output GeoPackages to S3.

```bash
python dep_coastlines/vector.py \
  --column 57 \
  --row 21 \
  --version 0.8.1 \
  --start-year 1984 \
  --end-year 2024 \
  --water-index twndwi
```

### `dep_coastlines/continental.py` — Continental Merge (Stage 3)

Combines individual tiled shorelines and rates of change layers into
continental-scale datasets, generates coastal change hotspot summaries, and
produces GeoPackage, shapefile, and PMTiles outputs.

```bash
python dep_coastlines/continental.py \
  --vector-version 0.8.1 \
  --continental-version 0.8.1 \
  --shorelines True \
  --ratesofchange True \
  --hotspots True \
  --hotspots-radius 15000 \
  --hotspots-radius 5000 \
  --hotspots-radius 1000 \
  --baseline-year 2024 \
  --include-styles \
  --include-tiles
```

### `dep_coastlines/clip_tide_data.py` — Clip Tide Model Data

Clips global FES2022b tidal model files to the Pacific region, optionally
uploads them to S3, and writes a URL listing file.

```bash
python dep_coastlines/clip_tide_data.py \
  /path/to/fes2022b \
  --output-dir data/raw/tidal_models/fes2022b \
  --copy-to-s3 \
  --write-urls-to-file
```

### `dep_coastlines/fix_lines_across_antimeridian.py` — Fix Antimeridian Crossings

Reprojects vector geometries to EPSG:4326 and shifts negative longitudes so
that lines crossing the antimeridian render correctly.

```bash
python dep_coastlines/fix_lines_across_antimeridian.py \
  input.gpkg output.gpkg
```
