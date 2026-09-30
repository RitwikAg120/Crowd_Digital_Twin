#!/usr/bin/env bash
# One-time setup on the GB10 (ARM64 + NVIDIA GPU): a venv with CUDA PyTorch
# and the project's requirements, then a check that PyTorch sees the GPU.
#
#   bash scripts/gb10_setup.sh
#
# PyTorch must be the CUDA build for ARM64 and must be installed *before*
# requirements.txt (Ultralytics would otherwise pull a CPU-only wheel).
# The GB10 (Grace Blackwell) needs a CUDA 12.8+ build; override TORCH_INDEX
# if NVIDIA's or PyTorch's current instructions name a different one, e.g.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/gb10_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}

if [ ! -d venv ]; then
  "$PY" -m venv venv
fi
. venv/bin/activate
pip install --upgrade pip wheel

pip install torch torchvision --index-url "$TORCH_INDEX"
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch does not see the GPU: check the driver and TORCH_INDEX"
print("PyTorch", torch.__version__, "CUDA", torch.version.cuda, "on", torch.cuda.get_device_name(0))
EOF

pip install -r requirements.txt
# Optional: MQTT gate counters (iot.py) and TensorRT/ONNX export (export.py)
pip install "paho-mqtt>=2.0" onnx onnxslim || echo "optional packages not installed"

python -c "import main; print('main.py imports fine')"
echo "Setup done. Next: python scripts/fetch_datasets.py all   then   bash scripts/run_gb10.sh"
