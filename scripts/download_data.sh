#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p data

uv run --locked gdown --no-cookies \
  1a0SF6xbMDwxmAde2VgBoP4q_qYX2hslm \
  -O data/plantnet-trunc.zip

unzip -o data/plantnet-trunc.zip -d data/plantnet-trunc

echo "Dataset downloaded and extracted."
