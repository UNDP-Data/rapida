# Use the GDAL image as the base
FROM ghcr.io/osgeo/gdal:ubuntu-full-3.10.0

# Install necessary tools and Python packages
RUN apt-get update && \
    apt-get install -y python3-pip pipenv gcc cmake libgeos-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
