# Use a minimal Debian slim image
FROM debian:11-slim

# Install dependencies for GDAL and cleaning up apt cache in one step
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    python3-pip \
    pipenv \
    nodejs \
    npm \
    curl \
    sudo \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && npm install -g configurable-http-proxy

# Optional: Set GDAL and PROJ_LIB environment variables
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

WORKDIR /app











# Verify the GDAL installation
CMD gdalinfo --version
