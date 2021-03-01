#!/bin/bash

EXPERIMENT_ROOT="../../../"
POOL="../configs/experiment/cifar100/vanilla-athena.json"
SELECTED_POOL="ensemble"
MODEL_INFO="../configs/experiment/cifar100/model-info.json"
DATA_INFO="../configs/experiment/cifar100/data-info.json"
ATTACK="../configs/experiment/cifar100/attack-zk-eot.json"
SELECTED_ATTACKS="pgd_eps0.005_rot"
#BENIGN_SAMPLES="bs_ratio03"
BENIGN_SAMPLES="bs_full"
TARGET="single"
EOT="False"
OUTPUT="../../experiment/cifar100/results"

echo "Generating STA($SELECTED_ATTACKS) in the context of the zero-knowledge model..."
echo "::: Targeting $TARGET"
python craft_ae_cifar100.py --experiment-root $EXPERIMENT_ROOT --pool-configs $POOL --selected-pool $SELECTED_POOL --model-configs $MODEL_INFO --data-configs $DATA_INFO --benign-samples $BENIGN_SAMPLES --attack-configs $ATTACK --selected-attacks $SELECTED_ATTACKS --targeted-model $TARGET --eot $EOT --output-root $OUTPUT

echo "<<< DONE!"
