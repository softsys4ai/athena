#!/bin/bash

EXPERIMENT_ROOT="../../../"
POOL="../configs/experiment/mnist/vanilla-athena.json"
KEY="ensemble"
MODEL_INFO="../configs/experiment/mnist/model-info.json"
DATA_INFO="../configs/experiment/mnist/data-info.json"
ATTACK="../configs/experiment/mnist/attack-zk-sta.json"
SELECTED_ATTACKS="translations"
BENIGN_SAMPLES="bs_full"
TARGET="single"
EOT="False"
OUTPUT="../../experiment/mnist/results"

echo "Generating STA($SELECTED_ATTACKS) AEs in the context of the zero-knowledge model..."
echo "::: Targeting $TARGET"
python craft_ae_mnist_cnn.py --experiment-root $EXPERIMENT_ROOT --pool-configs $POOL --model-configs $MODEL_INFO --data-configs $DATA_INFO --benign-samples $BENIGN_SAMPLES  --attack-configs $ATTACK --selected-attacks $SELECTED_ATTACKS --targeted-model $TARGET --eot $EOT --output-root $OUTPUT

echo "<<< DONE!"
