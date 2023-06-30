# This dockerfile is needed for anything that calls dea_tools.xxx. If using
# for the tidal calculations, would need to add the tidal models (or copy on
# startup from somewhere). Currently it's just used for vectorization.

# We need to base this on dea-coastlines instead of the microsoft pc image
# because a req. for dea-notebooks (hdstats) needs gcc to install. 
# Technically I don't think that piece is used at all for the pieces of dea_tools
# we need (at this point just subpixel_contours), so hopefully they modularize
# the dea-notebooks stuff at some point soon.
# Alternatively we could just pull the subpixel_contours stuff out of the repo.
 

FROM mcr.microsoft.com/planetary-computer/python:latest AS base

ADD . /tmp/src
ADD requirements.txt /tmp/requirements.txt
RUN conda install gcc -y \
  && pip install --no-cache-dir --upgrade pip \
  && pip install -r /tmp/requirements.txt
