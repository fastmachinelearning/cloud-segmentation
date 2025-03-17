#!/bin/bash

USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}
USER_NAME=${LOCAL_USER:user}

echo "Starting with USER_NAME: $USER_NAME UID: $USER_ID, GID: $GROUP_ID"
useradd -u $USER_ID -o -m $USER_NAME
# setting passwd on empty string, U6aMy0wojraho is Hash of empty string
echo ${USER_NAME}:U6aMy0wojraho | chpasswd -e
groupmod -g $GROUP_ID ${USER_NAME}
usermod -aG sudo $USER_NAME
export HOME=/home/${USER_NAME}
export PYTHONPATH=$PYTHONPATH:"/workspace/SCRIPTS/"


exec /usr/sbin/gosu ${USER_NAME} "$@"