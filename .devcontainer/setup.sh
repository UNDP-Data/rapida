#!/bin/bash

# Variables
USERNAME="janf"     # Change as needed
USER_UID=1000          # Change to match desired UID
USER_GID=1000          # Change to match desired GID
WORKSPACE="/workspace" # Adjust to your working directory

# 1. Create the user if it doesn't already exist
if ! id -u "$USERNAME" &>/dev/null; then
    echo "Creating user: $USERNAME with UID: $USER_UID and GID: $USER_GID"
    groupadd --gid "$USER_GID" "$USERNAME"
    useradd --uid "$USER_UID" --gid "$USER_GID" -m "$USERNAME"
else
    echo "User $USERNAME already exists."
fi

# 2. Adjust ownership of the workspace
if [ -d "$WORKSPACE" ]; then
    echo "Adjusting ownership of $WORKSPACE"
    chown -R "$USERNAME":"$USERNAME" "$WORKSPACE"
else
    echo "Workspace directory $WORKSPACE does not exist. Skipping ownership adjustment."
fi

# 3. Ensure the user has sudo permissions (optional)
if ! grep -q "^$USERNAME" /etc/sudoers; then
    echo "Granting sudo permissions to $USERNAME"
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
else
    echo "User $USERNAME already has sudo permissions."
fi

# 4. Optional: Run additional setup commands as the new user
# Uncomment the following lines to execute commands as the new user
# echo "Running post-create setup as $USERNAME"
# sudo -u "$USERNAME" bash -c "your-commands-here"

echo "Setup complete for user $USERNAME."
