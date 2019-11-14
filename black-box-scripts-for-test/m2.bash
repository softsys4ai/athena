#!/bin/bash

ensembleTag="prob1"
queriedDataDir=$1

mkdir -p data
mkdir -p ./data/models
mkdir -p ./data/adversarial_examples
mkdir -p ./data/analyse
mkdir -p ./data/figures
mkdir -p ./data/results

python generate_bb_AE.py ${queriedDataDir} pgd ${ensembleTag}
python generate_bb_AE.py ${queriedDataDir} bim ${ensembleTag} # bim l2
python generate_bb_AE.py ${queriedDataDir} mim ${ensembleTag}

