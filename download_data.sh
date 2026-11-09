#!/bin/bash
set -euo pipefail

mkdir -p data

python -m pip install gdown

python -m gdown \
  1a0SF6xbMDwxmAde2VgBoP4q_qYX2hslm \
  -O data/plantnet-trunc.zip

unzip -o data/plantnet-trunc.zip -d data/plantnet-trunc

echo "Dataset downloaded and extracted."
