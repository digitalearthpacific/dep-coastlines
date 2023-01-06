FROM geoscienceaustralia/dea-coastlines:latest AS base

ADD . /src
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r /src/requirements.txt \
  && pip install --no-cache-dir /src/dea-coastlines

WORKDIR /src

ENTRYPOINT mkdir -p /var/share/tide_models \
  && tar zxvf /src/data/fes2014.tar.gz -C /var/share/tide_models \
  && rm /src/data/fes2014.tar.gz \
  && /bin/bash 
