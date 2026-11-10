#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON="${PYTHON:-python3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating virtual environment at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
if ! "$VENV_PYTHON" -c "import datasets, tokenizers" >/dev/null 2>&1; then
    "$VENV_PYTHON" -m pip install --upgrade pip
    "$VENV_PYTHON" -m pip install datasets tokenizers
fi

if ! "$VENV_PYTHON" -c "import torch" >/dev/null 2>&1; then
    if [[ -z "$PYTORCH_INDEX_URL" ]]; then
        echo "PyTorch is missing. Set PYTORCH_INDEX_URL to the ROCm wheel index selected for this server." >&2
        echo "Choose the matching build at https://pytorch.org/get-started/locally/ and rerun." >&2
        exit 1
    fi
    "$VENV_PYTHON" -m pip install torch --index-url "$PYTORCH_INDEX_URL"
fi

check_gpu() {
    "$VENV_PYTHON" -c '
import sys
import torch

if not torch.cuda.is_available():
    sys.exit(
        "PyTorch cannot access an AMD GPU through ROCm. Check the server ROCm "
        "driver/runtime and install the matching ROCm-enabled PyTorch build. "
        "See https://pytorch.org/get-started/locally/."
    )
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
'
}

if ! check_gpu; then
    if [[ -z "$PYTORCH_INDEX_URL" ]]; then
        echo "Set PYTORCH_INDEX_URL to the ROCm wheel index selected for this server, then rerun." >&2
        exit 1
    fi
    echo "Installing/updating PyTorch from the selected ROCm wheel index."
    "$VENV_PYTHON" -m pip install --upgrade torch --index-url "$PYTORCH_INDEX_URL"
    check_gpu
fi

echo
echo "Environment is ready. Activate it with:"
echo "  source \"$VENV_DIR/bin/activate\""
echo
echo "Then, from $PROJECT_DIR, train the tokenizer if needed and start training:"
echo "  python tokenizer.py"
echo "  python train.py --subject math --max-steps 10000"
