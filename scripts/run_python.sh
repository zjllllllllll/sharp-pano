#!/usr/bin/env bash
set -euo pipefail
PANO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PANO_ROOT/pano-web:$PANO_ROOT/repair:$PANO_ROOT/providers:$PANO_ROOT/third_party/ml-sharp/src:$PANO_ROOT/third_party/DepthMaster:$PANO_ROOT/third_party/DepthMaster/third_party/utils3d:$PANO_ROOT/third_party/lingbot-depth:$PANO_ROOT/sharp-sixface:$PANO_ROOT/room-outline-experiment"
exec "$PANO_ROOT/.envs/runtime/bin/python" "$@"
