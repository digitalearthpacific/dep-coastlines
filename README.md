# dep-coastlines

## Status and Work to Date

This is a work-in-progress attempt to duplicate the Digital Earth Australia
(DEA) [coastline product](https://github.com/GeoscienceAustralia/dea-coastlines)
for the Pacific nations. The major modifications / adapations that are required to do so include:

1. Running the analysis as a [kbatch](https://github.com/kbatch-dev/kbatch)
   process.
2. Pulling data from the Planetary Computer via `pystac_client` rather than from
   an Open Data Cube instance.
3. Processing in an iterative fashion across Landsat tiles rather than on the
   whole AOI at once.
4. Study area-specific modifications to the workflow to account for innate
   differences in the land itself as well as differential data availability of
   (primarily) ancillary datasets.

### Migration to kbatch

Migrating to kbatch included several primary steps

#### 1. Dockerfile

I modified the existing Dockerfile from DEA work to include several additional
packages and to add support for the Dask Gateway, Planetary Computer, and
reading and writing to blob storage. These are listed in ./requirements.in with
full versions in ./requirements.txt.

I also added compressed tidal model data received from SPC staff to the image
itself.

#### 2. Outputs written to blob storage

Output raster and/or vector products are now written directly to blob storage,
as the kbatch endpoint does not support writing locally to persistent storage.

#### 3. Dask gateway support

I've endeavored to support a dask gateway-based workflow for processing to
hopefully improve processing time, but it is a work in progress.

### Using the Planetary Computer

Accessing Landsat data from the Planetary Computer using the pystac_client
package has been a pretty seamless drop in for existing DEA ODC data pulls.

### Iterative processing

As for prior work, I've elected to process by Landsat scenes, collecting all the
data for a specific year at a time. In this case, tri-annual mosaics are
required to gapfill years and pixels of over-cloudiness, so three years of data
are pulled at once. (I am not settled on this workflow - the DEA processing
pulls data for all available years at once, for instance).

### DEP-specific Modifications

The creation of intermediate raster products is relatively straightforward. It's
essentially the calculation of the modified normalized differential water index
(MNDWI) for an annual cloud-free mosaic. The vectorization process is much more
customized to the desired output as well as the area of interest (in prior
cases, the Australian coastline), and requires ancillary information such as
geomorphological data to correct errors in rocky places (note: please see links
to this data in the DEAustralia and DEAfrica coastline repositories). I've begun
working through the processing steps from the DEA work but haven't yet produced
a satisfactory vectorization workflow for the DEP study area.

In addition, consultation with SPC staff has suggested that additional
processing steps may be required to produce an accurate and reliable product for
Pacific nations. For example, lagoons formed by atholls may have consistently
higher water levels than their external boundaries, which may be at odds with
coarse scale (5km) tidal model information.

Finally, most of the documentation for prior coastline work treats the raster
and vector steps as separate products. This may ultimately prove to be the best
approach since there seems to be a large amount of trial-and-error type
processing to create the vector coastlines, while the raster creation is
relatively straightforward. However, to date I have attempted to combine these
steps into one.

## Code and Processing

The docker image can be created and uploaded to dockerhub using the usual
process. The current working version I have created is available at
`docker.io/numbersandstrings/dep-coastlines`; it was created in the root
directory of this repository. Tidal data are stored in `data/fes2014.tar.gz` and
this file must be present in this location in the docker image for the current
workflow to function correctly (if it is located elsewhere or named differently,
please modify src/process.sh).

`src/kbatch.yml` holds the processing information for kbatch (as well as a
typical shell invocation of processing). In short, this file calls
src/process.sh which in turns uncompresses the tidal models and runs the python
script run_coastlines.py.

## Next Steps

First, note that the official DEA coastline repository now recommends using the
Digital Earth Africa [coastline
repository](https://github.com/digitalearthafrica/deafrica-coastlines/) as a
starting point for a non-Australian analysis. I would first adapt existing code
to use this repository rather than the current dea repository.

I began to clip the tidal model data (see src/clip_tidal_models.py) but never
verified it was completed correctly nor incorporated it in the docker image. The
goal of this was to reduce the size of the docker image, since it needs to be
re-pulled and the data re-extracted for any kbatch run. This may also speed up
the computations somewhat.

I was similarly working to clip data to buffered land earlier in the process as
I suspected the tidal model calculations (which were taking the bulk of
processing time) were being completed for areas outside of possible coastline
zones.

Otherwise, I was working through the processing steps in
`coastlines.vector.contours_preprocess`, which were essentially the steps to
filter and clean up the raster data prior to vectorization. For a quick win, I
would probably skip some of these steps and pass the raster mosaics directly to
`subpixel_contours` without much, if any, preprocessing to see how good they
look.

Finally and perhaps most important to other DEP work, you will note the majority
of the content in `src/run_coastlines` is adapted from
[src/Processor.py](https://github.com/jessjaco/DigitalEarthPacific/blob/dev/jesse/src/Processor.py)
in my fork of the [Digital Earth Pacific
repository](https://github.com/PacificCommunity/DigitalEarthPacific). This
workflow has been used to produce existing estimates of evi and wofs. Changes
here should be reincorporated in that code and it should be in only one place.
