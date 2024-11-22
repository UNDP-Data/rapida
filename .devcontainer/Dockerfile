FROM ghcr.io/osgeo/gdal:ubuntu-full-3.10.0 as GDAL
ARG USERNAME=janf
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Install sudo package and create user
RUN apt-get update && apt-get install -y sudo && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Adjust ownership of the workspace (optional)
WORKDIR /workspace
RUN chown -R $USERNAME:$USERNAME /workspace

# Switch to the new user (optional, for debugging or initial setup)
USER $USERNAME