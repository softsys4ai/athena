#!/bin/bash

echo "Generating adversarial examples in the context of the zero-knowledge model..."
python craft_ae_mnist_cnn.py --experiment-root "../../../" --pool-configs "../configs/experiment/mnist/vanilla-athena.json" --model-configs "../configs/experiment/mnist/model-info.json" --data-configs "../configs/experiment/mnist/data-info.json" --attack-configs "../configs/experiment/mnist/attack-demo.json" --targeted-model "single" --eot "False" --output-root "../../experiment/mnist/results"