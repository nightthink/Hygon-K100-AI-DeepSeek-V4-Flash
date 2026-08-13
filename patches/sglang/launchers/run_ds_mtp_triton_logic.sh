#!/bin/bash

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_HACK_FLASHMLA_BACKEND=triton_logic

exec bash "$SCRIPT_DIR/run_ds_mtp.sh" "$@"
