# Deploy docker iamge to ACI

## Create ACI

The below commands are to create new ACI by mounting fileshare storage.

This creation can only be done one time if no need to change settings.

- first replace variables in YAML

```shell
source ./.azure/.env

ACR_PASSWORD=${ACR_PASSWORD} \
AZURE_STORAGE_ACCOUNT_NAME=${AZURE_STORAGE_ACCOUNT_NAME} \
AZURE_STORAGE_ACCOUNT_KEY=${AZURE_STORAGE_ACCOUNT_KEY} \
SSH_USERS=$SSH_USERS \
./.azure/aci-deploy.sh temp.yaml
```

- deploy

```shell
az login
az container create --resource-group ${AZURE_RESOUCE_GROUP} --file temp.yaml

# delete temp.yaml
rm temp.yaml
```

## Restart

when new image is deployed, ACI can be restarted.

```shell
# restart
az container restart --resource-group ${AZURE_RESOUCE_GROUP} --name cbsurge-rapida
```