#!/bin/bash

ensembleTag="prob1"
queriedDataDir=$1

mkdir -p data
mkdir -p ./data/models
mkdir -p ./data/adversarial_examples
mkdir -p ./data/analyse
mkdir -p ./data/figures
mkdir -p ./data/results

python generate_bb_AE.py ${queriedDataDir} fgsm ${ensembleTag}
python generate_bb_AE.py ${queriedDataDir} bim ${ensembleTag} # bim inf
python generate_bb_AE.py ${queriedDataDir} cw_l2 ${ensembleTag}

