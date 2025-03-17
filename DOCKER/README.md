# DOCKER

Put here every docker-related files.
Docker is a platform designed to help developers build, share, and run modern applications. We used it to containerized our environment (dev, exec) to port them accross different computers. For more information, see the [docker web site](https://www.docker.com/get-started/).

## Basis

By default the nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 image is used but a simple ubuntu20.04 or another image can be used.

The basis docker installs a python (3.9) environment in conda installing the packages listed in [requirements.txt](requirements.txt). With some basics library as GDAL.

The [entrypoint.sh](entrypoint.sh) file is call at every new instance of an image to setup the user in the docker (same as th caling user).

## Run Docker

To build the docker just use:

```sh
make build
```

To start the docker you can use:

```sh
make bash
```
