#!/bin/bash

USERNAME=$1
PASSWORD=$2

# skip if user already exists
if id "$USERNAME" &>/dev/null; then
    echo "User $USERNAME already exists."
else
    # create new user
    useradd -m -s /bin/bash "$USERNAME"
    echo "$USERNAME:$PASSWORD" | chpasswd
    echo "User $USERNAME created."

    # Add the user to the group
    usermod -aG $GROUP_NAME "$USERNAME"
    echo "User $USERNAME added to $GROUPNAME group."

    # Grant sudo access (optional)
    usermod -aG sudo "$USERNAME"
    echo "User $USERNAME granted sudo privileges."

    # change user home directory
    USER_DATA_DIR=$DATA_DIR/$USERNAME
    mkdir -p "$USER_DATA_DIR"
    chown "$USERNAME:$GROUP_NAME" "$USER_DATA_DIR"
    echo "User $USERNAME data directory was created at $USER_DATA_DIR."
    exit

fi
