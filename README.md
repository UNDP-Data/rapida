# geo-cb-surge
A repo to hold python tools that facilitate the assessment of natural hazards over various domains like population, landuse, infrastructure, etc  

## Usage

Install the project with dependencies to virtual environment as below.

```shell
pipenv run pip install -e .
```

Then, run the below command to show help menu.

```shell
pipenv run rapida --help
```

To uninstall the project from Python environment, execute the following command.

```shell
pipenv run pip uninstall geo-cb-surge
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
pipenv run rapida --help # run CLI in shell on docker container
```