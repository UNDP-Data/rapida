# Use the GDAL image as the base
FROM ghcr.io/osgeo/gdal:ubuntu-full-3.10.0

ARG GROUPNAME="cbsurge"

# Install necessary tools and Python packages
RUN apt-get update && \
    apt-get install -y python3-pip pipenv gcc cmake libgeos-dev openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install azure-cli
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

RUN mkdir /var/run/sshd && \
    echo 'PermitRootLogin no' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config

WORKDIR /app

COPY . .

# Create a group and set permissions for /app
RUN groupadd ${GROUPNAME} && \
    usermod -aG ${GROUPNAME} root && \
    mkdir -p /app && \
    chown -R :${GROUPNAME} /app && \
    chmod -R g+rwx /app

RUN chmod +x /app/create_user.sh
RUN chmod +x /app/entrypoint.sh

# install package
ENV PIPENV_VENV_IN_PROJECT=1
RUN pipenv install --dev --python 3 && \
    pipenv run pip install -e .
ENV VIRTUAL_ENV=/app/.venv

EXPOSE 22

ENTRYPOINT ["/app/entrypoint.sh"]