# Stage 1: Build Tippecanoe with Nix
FROM nixos/nix as tippecanoe-builder

# Install dependencies using Nix
RUN nix-env -iA nixpkgs.gcc \
    nixpkgs.sqlite \
    nixpkgs.zlib \
    nixpkgs.git \
    nixpkgs.make \
    nixpkgs.curl \
    nixpkgs.unzip

# Download Tippecanoe and build it
RUN curl -L https://github.com/felt/tippecanoe/archive/refs/heads/main.zip -o tippecanoe.zip && \
    unzip tippecanoe.zip && \
    mv tippecanoe-main tippecanoe && \
    rm tippecanoe.zip && \
    cd tippecanoe && \
    make && \
    strip tippecanoe tile-join && \
    rm -rf .git *.o

# Stage 2: Final image using Nix for environment management
FROM nixos/nix as final-image

# Set environment variables
ARG GROUP_NAME="cbsurge"
ARG DATA_DIR='/data'
ARG PRODUCTION

ENV GROUP_NAME $GROUP_NAME
ENV DATA_DIR $DATA_DIR

# Install specific versions of GDAL (3.10) and Python (3.12) via Nix
RUN nix-env -iA nixpkgs.python3_12 \
    nixpkgs.python3Packages.pip \
    nixpkgs.gdal_3_10 \
    nixpkgs.nodejs \
    nixpkgs.cmake \
    nixpkgs.geos \
    nixpkgs.git \
    nixpkgs.vim \
    nixpkgs.sudo \
    nixpkgs.curl \
    nixpkgs.npm

# Set up the environment for Python dependencies
WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# Install Python dependencies using Pipenv and Nix
RUN pip install pipenv && \
    pipenv install --python 3.12 && \
    pipenv run pip install .[dev,jupyter] && \
    pipenv run pip install playwright && \
    pipenv run playwright install chromium --with-deps

# Copy Tippecanoe binaries from the build stage
COPY --from=tippecanoe-builder /tippecanoe/tippecanoe* /usr/local/bin/
COPY --from=tippecanoe-builder /tippecanoe/tile-join /usr/local/bin/

# Copy application files
COPY . .

# Conditional installation based on PRODUCTION variable
RUN if [ -z "$PRODUCTION" ]; then \
        pipenv run pip install -e . ; \
    else \
        pipenv run pip install . ; \
    fi

# Clear unnecessary pipenv cache
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
