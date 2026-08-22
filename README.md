# FlyRot-v0

FlyRot-v0 is a small research prototype for estimating relative camera
rotation from RGB video. Its architecture is inspired by motion vision in
fruit flies: ON/OFF temporal channels, local direction-selective cells,
rotational-flow geometry, and a causal evidence accumulator.

Topics: `drosophila melanogaster`, `drosophila`, `melanogaster`, computer
vision, visual odometry.

The project does not use depth, IMU, optical flow, or semantic labels as model
inputs. Pose is used only as a training/evaluation target. The current model
uses 128x128 RGB crops, three frames per window, and a four-frame sampling gap.

The accepted checkpoint is included at `models/flyrot_v0_best.pt`. It is a
small state-dict-only PyTorch checkpoint. Detailed evaluation is in
[`reports/evaluation_metrics.md`](reports/evaluation_metrics.md).

Current research status: the checkpoint beats the zero-rotation baseline on
trajectory-disjoint validation and aggregate held-out test evaluation. It is
not yet a general-purpose visual odometry system: translation-induced
parallax and calibrated uncertainty remain open limitations.

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install -U pip
pip install -e '.[dev]'
```

For CUDA, install the PyTorch wheel appropriate for your driver and GPU, then
install the remaining project dependencies. The model also runs on CPU for
tests and small experiments.

## Data preparation

The loader expects a TartanAir-style RGB+pose index. Raw data is never copied
or modified by the project. Create an index on your machine:

```bash
python scripts/build_index.py \
  --root /path/to/tartanair-v2 \
  --output artifacts/tartanair2_index.json
```

The generated index contains local paths and is intentionally ignored by Git.
Keep dataset licenses and download instructions with the dataset source.

Copy the example local configuration if desired:

```bash
cp configs/local.example.yaml configs/local.yaml
```

Edit `configs/local.yaml` with paths that exist on your machine. Do not commit
that file.

## Run tests

```bash
pytest -q
python -m compileall -q src scripts
```

Tests that require a real dataset are skipped unless the corresponding local
dataset/index is available; unit and geometry tests run from a clean checkout.

## Evaluate the included checkpoint

After generating a local index:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint models/flyrot_v0_best.pt \
  --index artifacts/tartanair2_index.json \
  --split validation \
  --image-size 128 \
  --frame-gap 4
```

For the complete metric report:

```bash
python scripts/evaluate_metrics.py \
  --checkpoint models/flyrot_v0_best.pt \
  --index artifacts/tartanair2_index.json \
  --split validation
```

## Train

```bash
python scripts/train.py \
  --index artifacts/tartanair2_index.json \
  --run-dir runs/my-experiment \
  --model flyrot_motion_direct \
  --steps 1200 \
  --image-size 128 \
  --window-length 3 \
  --frame-gap 4 \
  --batch-size 32 \
  --target-direction optical_image_motion
```

Training outputs belong under `runs/`, which is ignored by Git. Use a
trajectory-disjoint split and report the zero-rotation baseline alongside
model metrics.

## Architecture at a glance

```text
RGB frames -> adaptive luminance -> ON/OFF changes
           -> 8 local motion directions at 3 scales
           -> camera-intrinsics-aware rotational evidence
           -> causal temporal accumulator
           -> 3D relative rotation vector + confidence proxy
```

See [`docs/USAGE.md`](docs/USAGE.md) for the longer usage guide and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the model map. The
rotation-residual diagnostic and its current evidence are described in
[`docs/rotation_residual_report.md`](docs/rotation_residual_report.md).

## License and data

The code and included model artifact are released under the MIT License. The
training dataset is not redistributed here and remains subject to its own
license and terms.
