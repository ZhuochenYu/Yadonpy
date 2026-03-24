#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export YADONPY_GMX_CMD="${YADONPY_GMX_CMD:-gmx}"
export YADONPY_RESTART="${YADONPY_RESTART:-1}"
export YADONPY_MPI="${YADONPY_MPI:-1}"
if [[ -z "${YADONPY_OMP:-}" ]]; then
    export YADONPY_OMP="$(getconf _NPROCESSORS_ONLN)"
fi
export YADONPY_GPU="${YADONPY_GPU:-1}"
export YADONPY_GPU_ID="${YADONPY_GPU_ID:-0}"
export YADONPY_EG12_TERM_QM="${YADONPY_EG12_TERM_QM:-0}"

if [[ $# -eq 0 ]]; then
    python "${SCRIPT_DIR}/run_cmcna_interface.py" --profile full
else
    python "${SCRIPT_DIR}/run_cmcna_interface.py" "$@"
fi
