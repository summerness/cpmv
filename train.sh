#!/bin/bash
set -euo pipefail
CONFIG=${1:-config/m1_convnext_unetpp_512.yaml}
python src/train.py --config "$CONFIG"
