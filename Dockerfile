FROM geoscienceaustralia/dea-coastlines:latest AS base

ADD . /tmp/src
RUN pip install --no-cache-dir --upgrade pip &&  \
  pip install /tmp/src/dep-tools && \
  pip install /tmp/src/deafrica-coastlines && \
  pip install /tmp/src/dea-notebooks/Tools && \
  pip install odc-geo
