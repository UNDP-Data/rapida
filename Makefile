.PHONY: help test build down shell

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  shell            to shell in dev mode"
	@echo "  test             to execute test cases"
	@echo "  build            to build docker image"
	@echo "  down             to destroy docker containers"


shell:
	@echo
	@echo "------------------------------------------------------------------"
	@echo "Shelling in dev mode"
	@echo "------------------------------------------------------------------"
	docker compose -f docker-compose.yaml run cbsurge /bin/bash


test:
	@echo
	@echo "------------------------------------------------------------------"
	@echo "Execute test cases"
	@echo "------------------------------------------------------------------"
	pipenv run python -m pytest cbsurge/stats

build:
	@echo
	@echo "------------------------------------------------------------------"
	@echo "Building Docker image"
	@echo "------------------------------------------------------------------"
	docker compose -f docker-compose.yaml build

down:
	@echo
	@echo "------------------------------------------------------------------"
	@echo "Destroy docker containers"
	@echo "------------------------------------------------------------------"
	docker compose -f docker-compose.yaml down


