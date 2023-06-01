# dep-coastlines

## Installation

### Development

1. Pull the submodules with `git submodule update --init`
2. Install local packages from requirements.txt

### On Microsoft Planetary Computer Hub Instance

** to be written **

## Status and Work to Date

This is a work-in-progress attempt to duplicate the Digital Earth Australia
(DEA) [coastline product](https://github.com/GeoscienceAustralia/dea-coastlines)
for the Pacific nations. The major modifications / adapations that are required to do so include:

1. Running the analysis as a [kbatch](https://github.com/kbatch-dev/kbatch)
   process.
2. Pulling data from the Planetary Computer via `pystac_client` rather than from
   an Open Data Cube instance.
3. Study area-specific modifications to the workflow to account for innate
   differences in the land itself as well as differential data availability of
   (primarily) ancillary datasets.
4. Modifying the workflow to utilize a Dask Gateway whenever reasonable.

We have processed and are currently evaluating draft vector output across the
study area for Landsat 8 data (~2013-2023).
