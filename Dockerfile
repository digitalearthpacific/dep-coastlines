FROM geoscienceaustralia/dea-coastlines:latest AS base

ADD . /tmp/src
RUN pip install --no-cache-dir --upgrade pip \
  && /tmp/src/install.sh
