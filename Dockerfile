# Use the GDAL image as the base
FROM ghcr.io/osgeo/gdal:ubuntu-full-3.10.0

ARG GROUPNAME="cbsurge"

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
RUN groupadd ${GROUPNAME} && \
    usermod -aG ${GROUPNAME} root && \
    mkdir -p /app && \
    chown -R :${GROUPNAME} /app && \
    chmod -R g+rwx /app

# create home directory under /data folder
ARG HOME_DIR='/data/home'
RUN mkdir -p $HOME_DIR
RUN chown -R :${GROUPNAME} $HOME_DIR
ENV HOME $HOME_DIR

RUN chmod +x /app/create_user.sh
RUN chmod +x /app/entrypoint.sh

# install package
ENV PIPENV_VENV_IN_PROJECT=1
RUN pipenv install --dev --python 3 && \
    pipenv run pip install -e .
ENV VIRTUAL_ENV=/app/.venv

ENTRYPOINT ["/app/entrypoint.sh"]