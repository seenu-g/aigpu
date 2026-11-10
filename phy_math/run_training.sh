#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
VENV_DIR=${VENV_DIR:-"$PROJECT_DIR/.venv"}
PYTHON=${PYTHON:-python3}
PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-}

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating virtual environment at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
if ! "$VENV_PYTHON" -c "import datasets, tokenizers" >/dev/null 2>&1; then
    "$VENV_PYTHON" -m pip install --upgrade pip
    "$VENV_PYTHON" -m pip install datasets tokenizers
fi

if ! "$VENV_PYTHON" -c "import torch" >/dev/null 2>&1; then
    if [ -z "$PYTORCH_INDEX_URL" ]; then
        echo "PyTorch is missing. Set PYTORCH_INDEX_URL to the ROCm wheel index selected for this server." >&2
        echo "Choose the matching build at https://pytorch.org/get-started/locally/ and rerun." >&2
        exit 1
    fi
    "$VENV_PYTHON" -m pip install torch --index-url "$PYTORCH_INDEX_URL"
fi

if ! "$VENV_PYTHON" -c '
import torch

if not torch.cuda.is_available():
    raise SystemExit(1)
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
'; then
    if [ -z "$PYTORCH_INDEX_URL" ]; then
        echo "PyTorch cannot access the AMD GPU, and PYTORCH_INDEX_URL is empty." >&2
        echo "Check the server ROCm driver/runtime, select the matching PyTorch ROCm wheel at https://pytorch.org/get-started/locally/, and rerun with PYTORCH_INDEX_URL set." >&2
        exit 1
    fi
    echo "Installing/updating PyTorch from the selected ROCm wheel index."
    "$VENV_PYTHON" -m pip install --upgrade torch --index-url "$PYTORCH_INDEX_URL"
    "$VENV_PYTHON" -c '
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "PyTorch still cannot access the AMD GPU. Verify the ROCm driver/runtime "
        "and that the selected PyTorch wheel matches this server."
    )
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
'
fi

echo
echo "Environment is ready. Activate it with:"
echo "  . \"$VENV_DIR/bin/activate\""
echo
echo "Then, from $PROJECT_DIR, train the tokenizer if needed and start training:"
echo "  python tokenizer.py"
echo "  python train.py --subject math --max-steps 10000"
