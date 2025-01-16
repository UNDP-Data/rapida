#!/bin/bash

# Create multiple users from environment variable SSH_USERS
# Format: JUPYTER_USERS="user1:password1 user2:password2 user3:password3"
if [ ! -z "$JUPYTER_USERS" ]; then
    for user_info in $JUPYTER_USERS; do
        IFS=':' read -r username password <<< "$user_info"
        if [ ! -z "$username" ] && [ ! -z "$password" ]; then
            /app/create_user.sh "$username" "$password"
        else
            echo "Invalid user format: $user_info"
        fi
    done
fi

# Determine the port based on the PRODUCTION environment variable
if [ "$PRODUCTION" = "true" ]; then
    JUPYTER_PORT=80
else
    JUPYTER_PORT=8000
fi

# Start JupyterLab in the foreground (so the container keeps running)
pipenv run jupyterhub -f jupyterhub_config.py --port=$JUPYTER_PORT