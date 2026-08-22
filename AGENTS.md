# FlyRot contribution notes

- Keep the project reproducible from a clean checkout.
- Keep raw datasets, caches, virtual environments, generated runs, and local
  configuration out of version control.
- Read dataset roots and output directories from command-line arguments or
  `configs/local.yaml`; never hard-code a personal filesystem path.
- Use trajectory-disjoint evaluation and compare against the zero-rotation
  baseline before describing a model as improved.
- Run `pytest -q` and `python -m compileall -q src scripts` before publishing
  a change.
