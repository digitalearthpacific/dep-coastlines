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

## Processing

Coastline creation consists of five primary steps:

1. Calculating tides for each location and available landsat image. The code for
   this is in `dep_coastlines/calculate_tides.py`. (See the notes in that file
   on tide model data location.)
2. Filtering landsat images based on the tide values and creating annual
   mosaics. This is done in `dep_coastlines/tide_corrected_mosaics.py`.
3. Post-processing the annual mosaics to remove clouds, and mask the analysis
   zone to the coastal zone. In `dep_coastlines/clean_rasters.py`. Note that
   this step requires a model file.
4. Delineating the coastline using the cleaned annual mosaic (a second step in
   the script in #3).
5. Calculating rates of change over time and merging data into final products
   (see `dep_coastlines/continental.py`).

Processing is done over an grid of arbitrary size, as defined in
`dep_coastlines/grid.py` using Landsat Collection-2 Level-2 (tiers 1 and 2) data
loaded from the Microsoft "Planetary Computer". The tides dataset is TPXO9.

Steps 2, 3, and 4 were accomplished using a deployment of
[argo](https://argoproj.github.io/). The workflows for these are saved in
`.argo/`. Steps 1 and 5 were completed locally using a modest
workstation-level computer. The dockerfile for step 1 is `Dockerfile.tides`. The
remaining step use `Dockerfile`.
