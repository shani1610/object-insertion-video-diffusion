#!/bin/bash

# this file should be inside Tune-A-Video directory

# Run training for original pairs
accelerate launch train_tuneavideo.py --config="configs/original/suitcase.yml"
accelerate launch train_tuneavideo.py --config="configs/original/toolbox.yml"
accelerate launch train_tuneavideo.py --config="configs/original/racket.yml"
accelerate launch train_tuneavideo.py --config="configs/original/stool.yml"

# Run training for pretending pairs
accelerate launch train_tuneavideo.py --config="configs/pretending/suitcase.yml"
accelerate launch train_tuneavideo.py --config="configs/pretending/toolbox.yml"
accelerate launch train_tuneavideo.py --config="configs/pretending/racket.yml"
accelerate launch train_tuneavideo.py --config="configs/pretending/stool.yml"
