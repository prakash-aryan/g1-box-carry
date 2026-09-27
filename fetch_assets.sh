#!/usr/bin/env bash
# Download NVIDIA PhysicalAI-SimReady-Warehouse-01 (CC-BY-4.0) into ./assets.
#   ./fetch_assets.sh          full warehouse (~14 GB), required for --scene warehouse
#   ./fetch_assets.sh box      box asset only (40 MB), sufficient for --scene lab
# Requires the Hugging Face CLI (hf, from huggingface_hub).
set -euo pipefail
DEST="$(cd "$(dirname "$0")" && pwd)/assets/simready_warehouse"
if [ "${1:-all}" = "box" ]; then
    hf download nvidia/PhysicalAI-SimReady-Warehouse-01 --repo-type dataset \
        --include "Props/general/sm_cubebox_b04_brown_01/*" --local-dir "$DEST"
else
    hf download nvidia/PhysicalAI-SimReady-Warehouse-01 --repo-type dataset \
        --exclude "computex_handmanip_renders/*" --local-dir "$DEST"
fi
