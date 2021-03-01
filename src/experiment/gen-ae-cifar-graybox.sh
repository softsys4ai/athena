#!/bin/bash

EXPERIMENT_ROOT="../../../"
POOL="../configs/experiment/cifar100/vanilla-athena.json"
KEY="ensemble"
#SELECTED_POOL="ensemble2"
MODEL_INFO="../configs/experiment/cifar100/model-info.json"
DATA_INFO="../configs/experiment/cifar100/data-info.json"
ATTACK="../configs/experiment/cifar100/attack-wb-cnn.json"
SELECTED_ATTACKS="graybox"
BENIGN_SAMPLES="bs_ratio01"
TARGET="ensemble"
EOT="True"
OUTPUT="../../experiment/cifar100/results"

echo "Generating adversarial examples in the context of the gray-box model..."

for i in {0..9}
do
  SELECTED_POOL="$KEY$i"
  echo "::: Targeting $SELECTED_POOL"
  python craft_ae_cifar100.py --experiment-root $EXPERIMENT_ROOT --pool-configs $POOL --selected-pool $SELECTED_POOL --model-configs $MODEL_INFO --data-configs $DATA_INFO --benign-samples $BENIGN_SAMPLES --attack-configs $ATTACK --selected-attacks $SELECTED_ATTACKS --targeted-model $TARGET --eot $EOT --output-root $OUTPUT

  echo ""
done
