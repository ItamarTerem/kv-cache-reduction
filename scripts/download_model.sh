#!/usr/bin/env bash
# scripts/download_model.sh
#
# Download DeepSeek-V2-Lite weights from HuggingFace.
#
# Model : deepseek-ai/DeepSeek-V2-Lite
# Source: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
# Size  : ~31 GB  (15.7B parameters × 2 bytes BF16)
# Notes : NOT gated — no HF token required.
#
# Usage:
#   bash scripts/download_model.sh
#
# Overrides:
#   MODEL_ID   — HuggingFace repo ID   (default: deepseek-ai/DeepSeek-V2-Lite)
#   MODEL_DIR  — local save path       (default: <repo_root>/models/DeepSeek-V2-Lite)
#   REVISION   — git revision / branch (default: main)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V2-Lite}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/models/DeepSeek-V2-Lite}"
REVISION="${REVISION:-main}"

# Safety margin over the ~31 GB actual size
APPROX_SIZE_GIB=35

echo "======================================"
echo " DeepSeek-V2-Lite BF16 downloader"
echo " MODEL_ID  : $MODEL_ID"
echo " MODEL_DIR : $MODEL_DIR"
echo " REVISION  : $REVISION"
echo " Est. size : ~${APPROX_SIZE_GIB} GiB (~31 GB)"
echo "======================================"

# ── Check hf CLI ─────────────────────────────────────────────────────────────
if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: 'hf' CLI not found."
    echo "Activate your venv and ensure huggingface_hub is installed:"
    echo "  source .venv/bin/activate"
    echo "  pip install -U 'huggingface_hub[cli]'"
    exit 1
fi

# ── Disk space check ─────────────────────────────────────────────────────────
PARENT_DIR="$(dirname "$MODEL_DIR")"
mkdir -p "$PARENT_DIR"

AVAILABLE_GIB=$(df -BG "$PARENT_DIR" | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "Available disk space at $PARENT_DIR: ${AVAILABLE_GIB} GiB"

if (( AVAILABLE_GIB < APPROX_SIZE_GIB )); then
    echo ""
    echo "ERROR: Insufficient disk space."
    echo "  Required : ~${APPROX_SIZE_GIB} GiB"
    echo "  Available: ${AVAILABLE_GIB} GiB"
    echo ""
    echo "Free up space or set MODEL_DIR to a path with enough capacity:"
    echo "  MODEL_DIR=/path/to/large/disk bash scripts/download_model.sh"
    exit 1
fi
echo "✔ Disk space OK"

# ── HF token warning (not required for this model) ───────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo ""
    echo "NOTE: HF_TOKEN is not set."
    echo "deepseek-ai/DeepSeek-V2-Lite is public — no token is needed."
    echo "If you see 401 errors, run: hf auth login"
fi

# ── Enable fast transfer if hf_transfer is installed ─────────────────────────
export HF_HUB_ENABLE_HF_TRANSFER=1

# ── Prepare destination ───────────────────────────────────────────────────────
mkdir -p "$MODEL_DIR"

# ── Download ──────────────────────────────────────────────────────────────────
echo ""
echo "Starting download — ~31 GB, should complete in a few minutes on a fast connection..."
echo ""

CMD=(
    hf download "$MODEL_ID"
    --local-dir "$MODEL_DIR"
    --revision "$REVISION"
)

echo "Command: ${CMD[*]}"
echo ""

"${CMD[@]}"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo " Download complete ✔"
echo " Weights saved to: $MODEL_DIR"
echo ""
echo " Next step — run the verification suite:"
echo "   source .venv/bin/activate"
echo "   python tests/verify_kv_relation.py \\"
echo "     --model_path $MODEL_DIR"
echo "======================================"