#!/bin/bash

# Set build arguments
BASE_IMAGE="owa/train:latest"
USERNAME="vscode"
USER_UID="1000"
USER_GID="1000"

# Build the Docker image
docker build \
  --build-arg BASE_IMAGE=$BASE_IMAGE \
  --build-arg USERNAME=$USERNAME \
  --build-arg USER_UID=$USER_UID \
  --build-arg USER_GID=$USER_GID \
  -f Dockerfile \
  -t owa/dev \
  ..