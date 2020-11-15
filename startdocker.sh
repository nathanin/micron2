#!/usr/bin/env bash

docker_image=rapidsai/rapidsai:0.16-cuda10.1-runtime-ubuntu16.04-py3.8
docker run --rm -it --gpus 1 -v $(pwd):/workspace $docker_image
