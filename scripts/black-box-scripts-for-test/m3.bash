#!/bin/bash

ensembleTag="prob1"
queriedDataDir=$1

mkdir -p data
mkdir -p ./data/models
mkdir -p ./data/adversarial_examples
mkdir -p ./data/analyse
mkdir -p ./data/figures
mkdir -p ./data/results

python generate_bb_AE.py ${queriedDataDir} jsma ${ensembleTag}
