#!/bin/bash
IMAGE_NAME=dixuson/facereg_triton
docker build -t $IMAGE_NAME -f Dockerfile_controller .
docker image push $IMAGE_NAME