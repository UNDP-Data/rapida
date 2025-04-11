#!/bin/bash

# Create multiple users from environment variable SSH_USERS
# Format: JUPYTER_USERS="user1:password1 user2:password2 user3:password3"
if [ ! -z "$JUPYTER_USERS" ]; then
    for user_info in $JUPYTER_USERS; do
        IFS=':' read -r username password <<< "$user_info"
        if [ ! -z "$username" ] && [ ! -z "$password" ]; then
            /app/create_user.sh "$username" "$password"

            echo "Creating Jupyter user $username profile directories..."

            # Ensure the home directory exists first
            if [ ! -d "/home/$username" ]; then
                echo "Home directory for $username does not exist. Creating..."
                mkdir -p /home/$username
                chown $username:$username /home/$username
            fi

            # Create the full path for .ipython/profile_default/startup
            mkdir -p /home/$username/.ipython/profile_default/startup
            chown -R $username:$username /home/$username/.ipython

            # Now copy the cell hook
            cp /app/cbsurge/az/cell_hook.py /home/$username/.ipython/profile_default/startup/cell_hook.py
            chown $username:$username /home/$username/.ipython/profile_default/startup/cell_hook.py

            #/usr/local/share/jupyter/notebook/static/custom/
            mkdir -p /usr/local/share/jupyter/notebook/static/custom/
            cp /app/custom.js /usr/local/share/jupyter/notebook/static/custom/custom.js
            chmod 644 /usr/local/share/jupyter/notebook/static/custom/custom.js


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