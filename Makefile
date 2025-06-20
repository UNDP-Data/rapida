.PHONY: help test build build-prod down shell

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  shell            to shell in dev mode"
	@echo "  test             to execute test cases"
	@echo "  build            to build docker image"
	@echo "  build-prod       to build production docker image"
	@echo "  down             to destroy docker containers"


shell:
	@echo
	@echo "------------------------------------------------------------------"
	@echo "Shelling in dev mode"
	@echo "------------------------------------------------------------------"
	docker compose -f docker-compose.dev.yaml run --remove-orphans --entrypoint /bin/bash rapida-dev


test:
	@echo
	@echo "------------------------------------------------------------------"
	@echo "Execute test cases"
	@echo "------------------------------------------------------------------"
	pipenv run python -m pytest tests

build:
	@echo
	@echo "------------------------------------------------------------------"
	@echo "Building Docker image"
	@echo "------------------------------------------------------------------"
	docker compose -f docker-compose.dev.yaml build

build-prod:
	@echo
	@echo "------------------------------------------------------------------"
	@echo "Building Production Docker image"
	@echo "------------------------------------------------------------------"
	docker compose -f docker-compose.yaml build

down:
	@echo
	@echo "------------------------------------------------------------------"
	@echo "Destroy docker containers"
	@echo "------------------------------------------------------------------"
	docker compose -f docker-compose.dev.yaml down


