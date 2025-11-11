FROM ghcr.io/osgeo/gdal:ubuntu-full-3.10.3

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libpq-dev \
    ca-certificates \
    build-essential \
    # These 2 needed for tippecanoe
    libsqlite3-dev \
    zlib1g-dev \
    && apt-get autoclean \
    && apt-get autoremove \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# Install tippecanoe, needed for creating pmtiles file for tileserver
RUN git clone https://github.com/mapbox/tippecanoe.git \
  && cd tippecanoe \ 
  && make -j \
  && make install

ADD . /code
WORKDIR /code

RUN python3 -m pip install --upgrade uv --break-system-packages && uv sync --no-install-package pygeos

# Download tide data
RUN wget -nH -r --cut-dirs=1 -P data -i data/tide_data_urls.txt

ENV PATH="/code/.venv/bin:$PATH"

