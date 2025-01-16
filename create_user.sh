#!/bin/bash

USERNAME=$1
PASSWORD=$2
GROUPNAME=cbsurge

# skip if user already exists
if id "$USERNAME" &>/dev/null; then
    echo "User $USERNAME already exists."
else
    # create new user
    useradd -m -s /bin/bash "$USERNAME"
    echo "$USERNAME:$PASSWORD" | chpasswd
    echo "User $USERNAME created."

    # Add the user to the group
    usermod -aG $GROUPNAME "$USERNAME"
    echo "User $USERNAME added to $GROUPNAME group."

    # Grant sudo access (optional)
    usermod -aG sudo "$USERNAME"
    echo "User $USERNAME granted sudo privileges."

    # change user home directory
    USER_HOME_DIR=$HOME/$USERNAME
    mkdir -p "$USER_HOME_DIR"
    chown "$USERNAME:$GROUPNAME" "$USER_HOME_DIR"  # Ensure ownership is set
    usermod -d "$USER_HOME_DIR" "$USERNAME"
    echo "User $USERNAME home directory changed to $USER_HOME_DIR."
    rm -rf /home/$USERNAME

    echo "cd /app; pipenv shell;" >> $USER_HOME_DIR/.bashrc
    echo "User $USERNAME profile was modified to launch venv in starting."
fi
