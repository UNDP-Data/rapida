FROM ghcr.io/osgeo/gdal:ubuntu-full-3.10.0 as gdal


# Adjust ownership of the workspace (optional)
WORKDIR /workspace
RUN chown -R $USERNAME:$USERNAME /workspace

# Switch to the new user (optional, for debugging or initial setup)
USER $USERNAME