#!/bin/bash

YAML_FILE="./.azure/aci-deploy.yaml"

if [ -z "$AZURE_STORAGE_ACCOUNT_NAME" ]; then
    echo "Error: AZURE_STORAGE_ACCOUNT_NAME is not set. Please provide a valid value."
    exit 1
fi

if [ -z "$AZURE_STORAGE_ACCOUNT_KEY" ]; then
    echo "Error: AZURE_STORAGE_ACCOUNT_KEY is not set. Please provide a valid value."
    exit 1
fi

if [ -z "$ACR_PASSWORD" ]; then
    echo "Error: ACR_PASSWORD is not set. Please provide a valid value."
    exit 1
fi

if [ -z "$SSH_USERS" ]; then
    echo "Error: SSH_USERS is not set. Please provide a valid value."
    exit 1
fi

if [ -z "$1" ]; then
    echo "Error: Output file name is not specified. Please provide an output file name as the first argument."
    exit 1
fi

OUTPUT_FILE="$1"

sed -e "s/{ACR_PASSWORD}/$ACR_PASSWORD/" \
    -e "s/{AZURE_STORAGE_ACCOUNT_NAME}/$AZURE_STORAGE_ACCOUNT_NAME/" \
    -e "s/{AZURE_STORAGE_ACCOUNT_KEY}/$AZURE_STORAGE_ACCOUNT_KEY/" \
    -e "s/{SSH_USERS}/$SSH_USERS/" \
    "$YAML_FILE" > $OUTPUT_FILE

echo "$YAML_FILE was exported"