# Use the GDAL image as the base
FROM ghcr.io/osgeo/gdal:ubuntu-full-3.10.0

ARG GROUP_NAME="cbsurge"
ARG DATA_DIR='/data'

ENV GROUP_NAME $GROUP_NAME
ENV DATA_DIR $DATA_DIR

# Install necessary tools and Python packages
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get update && \
    apt-get install -y python3-pip pipenv \
        gcc cmake libgeos-dev git \
        ca-certificates curl gnupg nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
     npm install -g configurable-http-proxy

# install azure-cli
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

WORKDIR /app

COPY . .

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

# install package
ENV PIPENV_VENV_IN_PROJECT=1
RUN pipenv install --dev --python 3 && \
    pipenv run pip install -e .
ENV VIRTUAL_ENV=/app/.venv

ENTRYPOINT ["/app/entrypoint.sh"]