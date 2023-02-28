FROM geoscienceaustralia/dea-coastlines:latest AS base

ADD . /tmp/src
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r /tmp/src/requirements.txt \
  && pip install --no-cache-dir /tmp/src/dea-coastlines
