#!/bin/bash

ATTACK_CONFIG=$1
SELECTED_ATTACKS=$2
EOT=$3
TARGET_MODEL=$4

EXPERIMENT_ROOT="../../../"
POOL="../configs/experiment/cifar100/full-pool.json"
#t="revisionES1-ens1"
MODEL_INFO="../configs/experiment/cifar100/model-info.json"
DATA_INFO="../configs/experiment/cifar100/data-info.json"
BENIGN_SAMPLES="bs_ratio0002"
#ATTACK="../configs/experiment/cifar100/revision-attack-pgd-eot.json"
#SELECTED_ATTACKS="translation"
TARGET="ensemble"
#EOT="True"
OUTPUT="../../experiment/cifar100/results"

echo "[TARGET MODEL]: $TARGET_MODEL; [ATTACK CONFIG]: $SELECTED_ATTACKS; [EOT]: $EOT;..."
python craft_ae_cifar100.py --experiment-root $EXPERIMENT_ROOT --pool-configs $POOL --selected-pool $TARGET_MODEL --model-configs $MODEL_INFO --data-configs $DATA_INFO --benign-samples $BENIGN_SAMPLES  --attack-configs $ATTACK_CONFIG --selected-attacks $SELECTED_ATTACKS --targeted-model $TARGET --eot $EOT --output-root $OUTPUT

