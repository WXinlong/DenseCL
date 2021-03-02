#!/bin/bash
DET_CFG=$1
WEIGHTS=$2

python $(dirname "$0")/train_net.py --config-file $DET_CFG \
    --dist-url tcp://127.0.0.1:50010 \
    --num-gpus 8 --eval-only MODEL.WEIGHTS $WEIGHTS
