#!/bin/bash

# Create multiple users from environment variable SSH_USERS
# Format: SSH_USERS="user1:password1 user2:password2 user3:password3"
if [ ! -z "$SSH_USERS" ]; then
    for user_info in $SSH_USERS; do
        IFS=':' read -r username password <<< "$user_info"
        if [ ! -z "$username" ] && [ ! -z "$password" ]; then
            /app/create_user.sh "$username" "$password"
        else
            echo "Invalid user format: $user_info"
        fi
    done
fi

# launch ssh server
/usr/sbin/sshd -D