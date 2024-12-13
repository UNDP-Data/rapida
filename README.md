# geo-cb-surge
A repo to hold python tools that facilitate the assessment of natural hazards over various domains like population, landuse, infrastructure, etc  

## Usage

Install dependencies to virtual environment as below.

```shell
pipenv install -r requirements.txt
```

Then, run the below command to show help menu.

```shell
pipenv run python -m cbsurge.cli --help
```

## Admin

`admin` command provides functionality to retrieve admin data for passed bounding bbox from either OpenStreetMap or OCHA.

- OSM

```shell
pipenv run python -m cbsurge.cli admin osm --help
```

- ocha

```shell
pipenv run python -m cbsurge.cli admin ocha --help
```

## Run test

Each `cbsurge`'s modules has its own test suite which can be ran independently

```shell
# cbsurge.stats
python -m pytest cbsurge/stats
```

## Using docker

- build docker-image

```shell
docker compose build
```

- run CLI

```shell
docker compose run cbsurge python -m cbsurge.cli --help
```

- enter to Docker container

```shell
docker-compose run cbsurge /bin/bash
```