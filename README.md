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
make test
```

before running the above command, please use `devcontainer` or `make shell` to enter to docker container first.

## Using docker

- build docker-image

```shell
make build
```

- destroy docker container

```shell
make down
```

- enter to Docker container

```shell
make shell
./cbsurge.sh --help # run CLI in shell on docker container
```