# Build felt/tippecanoe
# Dockerfile from https://github.com/felt/tippecanoe/blob/main/Dockerfile
FROM ubuntu:22.04 AS tippecanoe-builder

RUN apt-get update \
  && apt-get -y install build-essential libsqlite3-dev zlib1g-dev git

RUN git clone https://github.com/felt/tippecanoe
WORKDIR tippecanoe
RUN make


# Use the GDAL image as the base
FROM ghcr.io/osgeo/gdal:ubuntu-full-3.10.0

ARG GROUP_NAME="cbsurge"
ARG DATA_DIR='/data'
ARG PRODUCTION

# Install necessary tools and Python packages
RUN apt-get update && \
    apt-get install -y python3-pip pipenv \
        gcc cmake libgeos-dev git vim sudo \
        curl gnupg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy pyproject.toml to install dependencies
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# install dev and jupyter dependencies
ENV PIPENV_VENV_IN_PROJECT=1
ENV PLAYWRIGHT_BROWSERS_PATH=0
RUN pipenv install --python 3 && \
    pipenv run pip install .[dev] && \
    pipenv run pip install playwright && \
    pipenv run playwright install chromium --with-deps
ENV VIRTUAL_ENV=/app/.venv

# copy tippecanoe to production docker image
COPY --from=tippecanoe-builder /tippecanoe/tippecanoe* /usr/local/bin/
COPY --from=tippecanoe-builder /tippecanoe/tile-join /usr/local/bin/

# copy rest of files to the image.
COPY . .

# Conditional installation based on PRODUCTION variable
RUN if [ -z "$PRODUCTION" ]; then \
        pipenv run pip install -e . ; \
    else \
        pipenv run pip install . ; \
    fi
RUN pipenv --clear

CMD ["sh"]