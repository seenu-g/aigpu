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

(
    . "$VENV_DIR/bin/activate"
    python -m pip install --upgrade pip

    if ! python -c "import torch" >/dev/null 2>&1; then
        if [ -n "$PYTORCH_INDEX_URL" ]; then
            python -m pip install torch --index-url "$PYTORCH_INDEX_URL"
        else
            python -m pip install torch
        fi
    fi

    python -m pip install datasets tokenizers matplotlib
)

echo
echo "Virtual environment is ready. Activate it in your shell with:"
echo "  . \"$VENV_DIR/bin/activate\""
echo
echo "Then, from $PROJECT_DIR, run:"
echo "  python tokenizer.py"
echo "  python train.py --subject math --max-steps 10000"
