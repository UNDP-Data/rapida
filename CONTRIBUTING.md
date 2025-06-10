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

# after entering to the shell
pipenv shell # enter pipenv environment
rapida init # initialize rapida command
```

Note. You may need to edit [docker-compose.dev.yaml](./docker-compose.dev.yaml) to mount volumes from your local storage.

```yaml
    volumes:
      - ./Makefile:/rapida/Makefile
      - ./rapida:/rapida/rapida # mount rapida folder to container
      - ./tests:/rapida/tests
      # uncomment to mount token info from local
      - ~/.rapida:/root/.rapida # <- uncomment this if you want to keep authenticated credentials
      # uncomment to mount data folder from local
      - ./data:/data # <- uncomment and modify this if you want to mount /data folder from local storage
```

- Destroy dev Docker image

```shell
make down
```

## Building for production Docker image

- Using Makefile command

```shell
make build-prod
```

- Using `docker build`

```shell
docker build . --target=prod -t rapida 
```

Note. `--target=prod` option is required to build Docker image for production.

Real production image is built automatically by GitHub Actions, and will be pushed into both GitHub Container Registry and Azure Container Registry.
