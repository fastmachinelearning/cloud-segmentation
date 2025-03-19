# Package description
This folder contains example training scripts of AGENIUM SPACE tiny unet model on ALCD Cloud DB. It contains training script, the ALCD DB reprocessed in RGB a docker recipe.

# Docker
The folder contains a docker recipe and a make file to build an run it, got check the Readme in DOCKER folder. It is assumed you have a NVIDIA GPU on your computer.
To build the docker:
```sh
cd DOCKER
make build
```

To run the docker : 
```sh
cd DOCKER
make bash
```

# Database 
You can find the DB in DATA. It is a tiled and rgb format of the ALCD Cloud DB available online [here](https://zenodo.org/records/1460961).

# Trained Models
This section assume you are using the provided docker.

For training the tiny unet 100k model use :
```sh
    python ./SCRIPTS/train.py run --model ags_tiny_unet_100k --data_path ./DATA/
```

For training the tiny unet 50k model use :
```sh
    python ./SCRIPTS/train.py run --model ags_tiny_unet_50k --data_path ./DATA/
```

You can access the help info using
```sh
    python ./SCRIPTS/train.py run -- --help
```

To use 2 gpus:
```bash
# using torchrun
torchrun --nproc_per_node=2 ./SCRIPTS/train.py run --model ags_tiny_unet_100k --data_path ./DATA/ --backend="nccl"
```

python ./SCRIPTS/train.py run --model ags_tiny_unet_50k --data_path ./DATA/


# Models Score
The table below contains the score results for the the models

| Model | Train Score (mean F1-score) | Valid Score (F1-score) |
|-------|-------------------|----------------|
| tiny_unet_50k | 0.94 - 0.78 | 0.94 - 0.77 |
| tiny_unet_100k | 0.94 - 0.79 | 0.84 - 0.78 |
