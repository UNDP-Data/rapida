# Stage 1: Build Tippecanoe
FROM ubuntu:22.04 AS tippecanoe-builder

RUN apt-get update && \
    apt-get install -y build-essential libsqlite3-dev zlib1g-dev git && \
    git clone https://github.com/felt/tippecanoe && \
    cd tippecanoe && \
    make -j$(nproc)

# Stage 2: Final image based on GDAL
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.10.0 AS base


# Set environment vars
ENV PIPENV_VENV_IN_PROJECT=1
ENV PLAYWRIGHT_BROWSERS_PATH=0
ENV DEBIAN_FRONTEND=noninteractive

# Install app dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip \
      python3-gdal \
      python3-dev \
      pipenv \
      gcc \
      g++ \
      git \
      cmake \
      libgeos-dev && \
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install playwright --break-system-packages && \
    playwright install chromium --with-deps

# Copy Tippecanoe binaries
COPY --from=tippecanoe-builder /tippecanoe/tippecanoe* /usr/local/bin/
COPY --from=tippecanoe-builder /tippecanoe/tile-join /usr/local/bin/


# Setup application working directory
WORKDIR /rapida
RUN mkdir -p /rapida && chown -R 1000:1000 /rapida
RUN ln -s /rapida/.venv/bin/rapida /usr/local/bin/rapida
USER 1000


# Install Python dependencies
RUN pipenv --python 3 --site-packages


COPY . .

# Docker image for development
FROM base AS dev

RUN pipenv run pip install -e .

# Docker image for production
FROM base AS prod

RUN pipenv run pip install .




