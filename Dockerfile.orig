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

ENV GROUP_NAME $GROUP_NAME
ENV DATA_DIR $DATA_DIR

# Install necessary tools and Python packages
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get update && \
    apt-get install -y python3-pip pipenv \
        gcc cmake libgeos-dev git vim sudo \
        ca-certificates curl gnupg nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    npm install -g configurable-http-proxy

WORKDIR /app

# copy pyproject.toml to install dependencies
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# install dev and jupyter dependencies
ENV PIPENV_VENV_IN_PROJECT=1
ENV PLAYWRIGHT_BROWSERS_PATH=0
RUN pipenv install --python 3 && \
    pipenv run pip install .[dev,jupyter] && \
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
# Create a group and set permissions for /app
RUN groupadd ${GROUP_NAME} && \
    usermod -aG ${GROUP_NAME} root && \
    mkdir -p /app && \
    chown -R :${GROUP_NAME} /app && \
    chmod -R g+rwx /app && \
    mkdir -p $DATA_DIR && \
    chown -R :${GROUP_NAME} $DATA_DIR

RUN chmod +x /app/create_user.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]