# FlyRot-v0 usage guide

This guide covers a clean local setup. It deliberately uses placeholder paths
so it can be copied to another machine without exposing local infrastructure.

## 1. Install

```bash
git clone https://github.com/OWNER/REPOSITORY.git
cd REPOSITORY
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e '.[dev]'
```

On Windows PowerShell, activate with:

```powershell
.venv\\Scripts\\Activate.ps1
```

For GPU use, install a compatible PyTorch build for the local driver before
running the project commands. No specific GPU, CUDA installation, hostname,
or device identifier is required by the public code.

## 2. Prepare a dataset index

The project reads TartanAir-style RGB images and pose text files through a
generated JSON index. The raw dataset remains outside the repository.

```bash
python scripts/build_index.py \
  --root /path/to/tartanair-v2 \
  --output artifacts/tartanair2_index.json
```

The index stores paths local to the current machine and is ignored by Git.
Never commit it if it contains personal filesystem paths. The index must have
trajectory-level records with RGB and pose paths, counts, camera name,
environment, and trajectory metadata.

## 3. Evaluate the release checkpoint

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint models/flyrot_v0_best.pt \
  --index artifacts/tartanair2_index.json \
  --split validation \
  --image-size 128 \
  --window-length 3 \
  --frame-gap 4 \
  --target-direction optical_image_motion
```

The output reports model error and the zero-rotation baseline. Use the same
trajectory split when comparing experiments.

## 4. Train a new experiment

```bash
python scripts/train.py \
  --index artifacts/tartanair2_index.json \
  --run-dir runs/example \
  --model flyrot_motion_direct \
  --steps 1200 \
  --image-size 128 \
  --window-length 3 \
  --frame-gap 4 \
  --batch-size 32 \
  --target-direction optical_image_motion
```

Useful options include `--device cpu`, `--device cuda:0`, `--num-workers`,
`--preload-images`, `--resume`, and `--seed`. Run directories contain
generated metrics and checkpoints and are ignored by Git.

## 5. Run the metric evaluator

```bash
python scripts/evaluate_metrics.py \
  --checkpoint models/flyrot_v0_best.pt \
  --index artifacts/tartanair2_index.json \
  --split validation \
  --output reports/local_metrics.json
```

The evaluator includes MSE against the model and zero baseline, SO(3)
geodesic errors, component errors, direction accuracy, confidence bins, and
selective risk.

## 6. Use the model from Python

```python
import torch

from flyrot.models.flyrot_v0 import FlyRotV0

model = FlyRotV0(
    scales=(4, 8, 16),
    magnitude_confidence=True,
    magnitude_confidence_center=0.0103562189,
    magnitude_confidence_scale=0.0052184332,
)
state = torch.load("models/flyrot_v0_best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(state["model"] if "model" in state else state)
model.eval()

frames = torch.rand(1, 3, 3, 128, 128)
with torch.no_grad():
    output = model(frames)

rotation = output["rotation_vector"][:, -1]
confidence = output["confidence"][:, -1]
```

The rotation vector is a relative camera rotation in the convention selected
by the training target. The confidence output is a useful ranking/abstention
signal, not a calibrated probability.

## 7. Verification before publishing

```bash
pytest -q
python -m compileall -q src scripts
git diff --check
```

Before a public push, scan staged files for absolute paths, usernames,
hostnames, IP addresses, UUIDs, local configs, caches, and generated data.
