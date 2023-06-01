# Digital Earth Pacific Coastlines

This is a work-in-progress adaptation of [Digital Earth Australia (DEA)]
(https://github.com/GeoscienceAustralia/dea-coastlines) and [Digital Earth
Africa
Coastlines](https://github.com/digitalearthafrica/deafrica-coastlines) products
for the Pacific nations. Please refer to the following for
description of the algorithms:

> Bishop-Taylor, R., Nanson, R., Sagar, S., Lymburner, L. (2021). Mapping
> Australia's dynamic coastline at mean sea level using three decades of Landsat
> imagery. _Remote Sensing of Environment_, 267, 112734. Available:
> https://doi.org/10.1016/j.rse.2021.112734

> Bishop-Taylor, R., Sagar, S., Lymburner, L., Alam, I., Sixsmith, J. (2019). Sub-pixel waterline extraction: characterising accuracy and sensitivity to indices and spectra. _Remote Sensing_, 11 (24):2984. Available: https://doi.org/10.3390/rs11242984

In addition to the new study area, this project is being computed on the
Microsoft Planetary Computer. Technical modifications to do so include

1. Running the analysis as a [kbatch](https://github.com/kbatch-dev/kbatch)
   process.
2. Pulling data from the Planetary Computer via `pystac_client` rather than from
   an Open Data Cube instance.
3. Study area-specific modifications to the workflow to account for innate
   differences in the land itself as well as differential data availability of
   (primarily) ancillary datasets.
4. Modifying the workflow to utilize a Dask Gateway whenever reasonable.

We have processed and are currently evaluating draft vector output across the
study area for Landsat 8 data (2013 - 2023).

## Development

1. Pull the submodules with `git submodule update --init`
2. Install local packages from requirements.txt
