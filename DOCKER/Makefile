help:
	@cat Makefile

# to see the logs of a container just do $>docker logs -t wtf_docker_name
TIME=$(shell date +'%Y-%m-%d')
HERE?=$(shell dirname `pwd`)
GPU?=all
PYTHON_VERSION?=3.9
DOCKER_FILE=Dockerfile
NAS=/NAS_SHARE/

UID=$(shell id -u)
GID=$(shell id -g)
USER=$(shell whoami)

TAG=he_edge_spaice_sample_train_alcd
VERSION=latest

build:
	docker build -t $(TAG):$(VERSION) -f $(DOCKER_FILE) .

bash: build
	docker run  -it --network=host \
				--shm-size=40gb \
				-e LOCAL_USER=$(USER) -e LOCAL_GID=$(GID) -e LOCAL_UID=$(UID) \
				-v ${NAS}:/nas \
			   	-v "$(HERE):/workspace" \
				-w /workspace \
				$(TAG):$(VERSION) \
				bash 

