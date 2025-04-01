# Stage 1: Build Tippecanoe
FROM debian:bookworm-slim AS tippecanoe-builder

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential libsqlite3-dev zlib1g-dev git \
  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/felt/tippecanoe
WORKDIR /tippecanoe
RUN make \
  && strip tippecanoe tile-join \
  && rm -rf .git *.o

# Stage 2: Final Image
FROM ghcr.io/osgeo/gdal:ubuntu-full-3.10.0

ARG GROUP_NAME="cbsurge"
ARG DATA_DIR='/data'
ARG PRODUCTION

ENV GROUP_NAME $GROUP_NAME
ENV DATA_DIR $DATA_DIR

# Install necessary tools and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip pipenv gcc cmake libgeos-dev curl gnupg nodejs \
  && rm -rf /var/lib/apt/lists/*

# Install Node.js and NPM packages efficiently
RUN npm install -g --omit=dev configurable-http-proxy

WORKDIR /app

# Copy only necessary dependency files first to leverage Docker caching
COPY pyproject.toml README.md ./

# Set up virtual environment and install dependencies
ENV PIPENV_VENV_IN_PROJECT=1
ENV PLAYWRIGHT_BROWSERS_PATH=0
RUN pipenv install --python 3 --deploy --ignore-pipfile && \
    pipenv run pip install .[dev,jupyter] && \
    pipenv run pip install playwright && \
    pipenv run playwright install chromium --with-deps
ENV VIRTUAL_ENV=/app/.venv

# Copy Tippecanoe binaries from the build stage
COPY --from=tippecanoe-builder /tippecanoe/tippecanoe* /usr/local/bin/
COPY --from=tippecanoe-builder /tippecanoe/tile-join /usr/local/bin/

# Copy remaining application files
COPY . .

# Conditional installation based on PRODUCTION variable
RUN if [ -z "$PRODUCTION" ]; then \
        pipenv run pip install -e . ; \
    else \
        pipenv run pip install . ; \
    fi

# Set up user group and permissions
RUN groupadd ${GROUP_NAME} && \
    usermod -aG ${GROUP_NAME} root && \
    mkdir -p /app $DATA_DIR && \
    chown -R :${GROUP_NAME} /app $DATA_DIR && \
    chmod -R g+rwx /app $DATA_DIR

RUN chmod +x /app/create_user.sh /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
