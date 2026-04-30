#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# YOLOv7 Gym Equipment Detection — Environment Setup
# Run once from the project root:  bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="gym-detection"
PYTHON_MIN="3.9"

echo "========================================="
echo "  Gym Equipment Detection — Setup"
echo "========================================="

# ── Option A: conda (recommended) ────────────────────────────────────────────
if command -v conda &>/dev/null; then
    echo "[conda] Creating environment '$ENV_NAME' from environment.yml ..."
    conda env create -f "$PROJECT_DIR/environment.yml" --force
    echo ""
    echo "Done. Activate with:"
    echo "  conda activate $ENV_NAME"
    echo "  jupyter notebook"
    exit 0
fi

# ── Option B: pip + venv (fallback) ──────────────────────────────────────────
echo "[venv] conda not found — falling back to Python venv"

# Check Python version
PYTHON_BIN="python3"
PY_VERSION=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Using Python $PY_VERSION ($($PYTHON_BIN --version))"

# Warn if Python >= 3.12 (YOLOv7 internals best on 3.10)
PY_MAJOR=$($PYTHON_BIN -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON_BIN -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 12 ]; then
    echo ""
    echo "  NOTE: Python $PY_VERSION detected. YOLOv7 internals are best tested"
    echo "  on Python 3.10. If you hit import errors after cloning yolov7/,"
    echo "  install Miniforge and use environment.yml instead:"
    echo "  https://github.com/conda-forge/miniforge"
    echo ""
fi

VENV_DIR="$PROJECT_DIR/gym-detection"
echo "[venv] Creating virtual environment at gym-detection ..."
$PYTHON_BIN -m venv "$VENV_DIR"

echo "[pip]  Installing packages from requirements.txt ..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

echo "[jupyter] Registering kernel '$ENV_NAME' ..."
"$VENV_DIR/bin/python" -m ipykernel install --user --name "$ENV_NAME" \
    --display-name "Python ($ENV_NAME)"

echo ""
echo "========================================="
echo "  Setup complete!"
echo "========================================="
echo ""
echo "  Activate:        source gym-detection/bin/activate"
echo "  Launch notebook: jupyter notebook"
echo "  Kernel name:     $ENV_NAME"
echo ""
echo "  First run will auto-clone yolov7/ and download"
echo "  pretrained weights (~72 MB) — internet required."
echo "========================================="
