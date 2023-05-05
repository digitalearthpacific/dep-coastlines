FROM geoscienceaustralia/dea-coastlines:latest AS base

ADD . /tmp/src
RUN pip install --no-cache-dir --upgrade pip \
  && /tmp/src/install.sh

ENTRYPOINT mkdir -p /var/share/tide_models \
  && tar zxvf /src/data/fes2014.tar.gz -C /var/share/tide_models \
  && rm /src/data/fes2014.tar.gz \
  && /bin/bash
