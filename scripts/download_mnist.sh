#!/usr/bin/env bash
set -euo pipefail

# Downloads the MNIST CSV mirror used by examples/mnist.cpp.
# Source verified live 2026-08-24: https://data.pjreddie.com/files/
mkdir -p data
echo "Downloading mnist_train.csv (~109MB)..."
curl -L -o data/mnist_train.csv https://data.pjreddie.com/files/mnist_train.csv
echo "Downloading mnist_test.csv (~18MB)..."
curl -L -o data/mnist_test.csv https://data.pjreddie.com/files/mnist_test.csv
echo "Done. Files are in data/."
