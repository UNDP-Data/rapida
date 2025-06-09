.PHONY: help test build build-prod down shell

# Detect host machine architecture
ARCH := $(shell uname -m)
TARGET ?= dev

# Set platform for ARM64 environment
ifeq ($(ARCH),arm64)
    PLATFORM_ARG = --platform linux/amd64
    COMPOSE_PLATFORM = DOCKER_DEFAULT_PLATFORM=linux/amd64
else
    PLATFORM_ARG =
    COMPOSE_PLATFORM =
endif

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  shell            to shell in dev mode"
	@echo "  test             to execute test cases"
	@echo "  build            to build docker image"
	@echo "  build-prod       to build production docker image"
	@echo "  down             to destroy docker containers"
	@echo ""
	@echo "Detected architecture: $(ARCH)"
ifneq ($(PLATFORM_ARG),)
	@echo "Platform override: linux/amd64"
endif

shell:
	@echo
	@echo "------------------------------------------------------------------"
	@echo "Shelling in dev mode"
	@echo "------------------------------------------------------------------"
	$(COMPOSE_PLATFORM) docker compose -f docker-compose.dev.yaml run --remove-orphans --entrypoint /bin/bash rapida-dev


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
	$(COMPOSE_PLATFORM) docker compose -f docker-compose.dev.yaml build

build-prod:
	@echo
	@echo "------------------------------------------------------------------"
	@echo "Building Production Docker image"
	@echo "------------------------------------------------------------------"
	$(COMPOSE_PLATFORM) docker compose -f docker-compose.yaml build

down:
	@echo
	@echo "------------------------------------------------------------------"
	@echo "Destroy docker containers"
	@echo "------------------------------------------------------------------"
	$(COMPOSE_PLATFORM) docker compose -f docker-compose.dev.yaml down


