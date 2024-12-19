# geo-cb-surge
A repo to hold python tools that facilitate the assessment of natural hazards over various domains like population, landuse, infrastructure, etc  

## Installation

Install the project with dependencies to virtual environment as below.

```shell
pipenv run pip install -e .
```

To uninstall the project from Python environment, execute the following command.

```shell
pipenv run pip uninstall geo-cb-surge
```

## Usage

Then, run the below command to show help menu.

```shell
pipenv run rapida --help
```

## Setup

To access blob storage in Azure, each user must have a role of `Storage Blob Data Contributor`.

- inside Docker container

Since it has an issue of opening browser by azure.identity package inside docker container, use `az login` to authenticate prior to use API.

```shell
az login # authenticate with az login
pipenv run rapida init
```

- without Docker

`init` command will open browser to authenticate to Azure

```shell
pipenv run rapida init
```

## Admin

`admin` command provides functionality to retrieve admin data for passed bounding bbox from either OpenStreetMap or OCHA.

- OSM

```shell
pipenv run rapida admin osm --help
```

- ocha

```shell
pipenv run rapida admin ocha --help
```

## Run test

Each `cbsurge`'s modules has its own test suite which can be ran independently

```shell
make test
```

before running the above command, please use `devcontainer` or `make shell` to enter to docker container first.

## Using docker

### build docker-image

```shell
make build
```

### Launch SSH server

- set users

```
cp .env.example .env
vi .env
```

SSH_USERS can have multiple users (username:password) for SSH login

```shell
SSH_USERS=docker:docker user:user
```

- launch docker container

```shell
make up
```

The below command is connecting to `localhost` with user `docker` through port `2222`.

```shell
ssh docker@localhost -p 2222

# make sure installing the package first
cd /app
pipenv run pip install -e .
```

### destroy docker container

```shell
make down
```

### enter to Docker container

```shell
make shell
pipenv run rapida --help # run CLI in shell on docker container
```