# Contributing Guidelines

This document contains a set of guidelines to help developers during the contribution process.

## Local development

We recommend you to use Docker to develop rapida tool in your local computer.

[docker-compose.dev.yaml](./docker-compose.dev.yaml) can be used for local development while [docker-compose.yaml](./docker-compose.yaml) is for production Docker image generation.

If you are in either Linux or Mac environment, you can use Makefile commands.

- Build dev Docker image

This command will build Docker image until `dev` state.

```shell
make build
```

- Enter to shell in Docker container

After building dev version image, you can enter to the shell through the below command.

```shell
make shell
```

- Destroy dev Docker image

```shell
make down
```