#!/usr/bin/env bash
# scripts/download_model.sh
#
# Download model weights from HuggingFace.
#
# Supported models:
#
#   deepseek  — deepseek-ai/DeepSeek-V2-Lite
#               https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
#               ~31 GB  (15.7B params, BF16). Public — no token required.
#               Use for local debugging (same MLA architecture as Kimi).
#
#   kimi      — nvidia/Kimi-K2.6-NVFP4
#               https://huggingface.co/nvidia/Kimi-K2.6-NVFP4
#               ~500 GB (1T params, NVFP4). Public — no token required.
#
# Usage:
#   bash scripts/download_model.sh                   # downloads deepseek (default)
#   MODEL=kimi bash scripts/download_model.sh        # downloads Kimi-K2.6-NVFP4
#
# Overrides (apply to either model):
#   MODEL      — which model to download: deepseek | kimi  (default: deepseek)
#   MODEL_ID   — override the HuggingFace repo ID
#   MODEL_DIR  — override the local save path
#   REVISION   — git revision / branch                     (default: main)
#   HF_TOKEN   — HuggingFace token (required for kimi, optional for deepseek)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-deepseek}"
REVISION="${REVISION:-main}"

# ── Per-model defaults ────────────────────────────────────────────────────────
case "$MODEL" in
    deepseek)
        DEFAULT_MODEL_ID="deepseek-ai/DeepSeek-V2-Lite"
        DEFAULT_MODEL_DIR="${ROOT}/models/DeepSeek-V2-Lite"
        APPROX_SIZE_GIB=35
        MODEL_LABEL="DeepSeek-V2-Lite BF16 (proxy)"
        REQUIRES_TOKEN=false
        ;;
    kimi)
        DEFAULT_MODEL_ID="nvidia/Kimi-K2.6-NVFP4"
        DEFAULT_MODEL_DIR="${ROOT}/models/Kimi-K2.6-NVFP4"
        APPROX_SIZE_GIB=520
        MODEL_LABEL="Kimi-K2.6 NVFP4"
        REQUIRES_TOKEN=false
        ;;
    *)
        echo "ERROR: Unknown MODEL='$MODEL'. Valid options: deepseek | kimi"
        exit 1
        ;;
esac

MODEL_ID="${MODEL_ID:-$DEFAULT_MODEL_ID}"
MODEL_DIR="${MODEL_DIR:-$DEFAULT_MODEL_DIR}"

echo "======================================"
echo " $MODEL_LABEL downloader"
echo " MODEL_ID  : $MODEL_ID"
echo " MODEL_DIR : $MODEL_DIR"
echo " REVISION  : $REVISION"
echo " Est. size : ~${APPROX_SIZE_GIB} GiB"
echo "======================================"

# ── Check hf CLI ─────────────────────────────────────────────────────────────
if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: 'hf' CLI not found."
    echo "Activate your venv and ensure huggingface_hub is installed:"
    echo "  source .venv/bin/activate"
    echo "  pip install -U 'huggingface_hub[cli]'"
    exit 1
fi

# ── HF token check ───────────────────────────────────────────────────────────
if [[ "$REQUIRES_TOKEN" == true ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo ""
        echo "ERROR: HF_TOKEN is required for $MODEL_LABEL (private repo)."
        echo "Export your token before running:"
        echo "  export HF_TOKEN=hf_..."
        echo "  MODEL=kimi bash scripts/download_model.sh"
        exit 1
    fi
    echo "✔ HF_TOKEN set"
else
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo ""
        echo "NOTE: HF_TOKEN is not set."
        echo "$DEFAULT_MODEL_ID is public — no token needed."
        echo "If you see 401 errors, run: hf auth login"
    fi
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
    echo "  MODEL_DIR=/path/to/large/disk MODEL=$MODEL bash scripts/download_model.sh"
    exit 1
fi
echo "✔ Disk space OK"

# ── Enable fast transfer if hf_transfer is installed ─────────────────────────
export HF_HUB_ENABLE_HF_TRANSFER=1

# ── Prepare destination ───────────────────────────────────────────────────────
mkdir -p "$MODEL_DIR"

# ── Download ──────────────────────────────────────────────────────────────────
echo ""
echo "Starting download — ~${APPROX_SIZE_GIB} GiB..."
echo ""

CMD=(
    hf download "$MODEL_ID"
    --local-dir "$MODEL_DIR"
    --revision "$REVISION"
)
if [[ -n "${HF_TOKEN:-}" ]]; then
    CMD+=(--token "$HF_TOKEN")
fi

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
