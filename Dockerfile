# Use a minimal Debian slim image
FROM debian:11-slim

# Install dependencies for GDAL and cleaning up apt cache in one step
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    python3-pip \
    pipenv \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Optional: Set GDAL and PROJ_LIB environment variables
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

# Verify the GDAL installation
CMD gdalinfo --version
