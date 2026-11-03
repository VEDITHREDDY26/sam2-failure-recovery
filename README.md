# SAM 2 Failure Recovery

A training-free wrapper around SAM 2.1 for semi-supervised video object segmentation. It detects
frames where SAM 2 has lost the object, using only SAM 2's own outputs, and recovers by
re-prompting SAM 2 at that frame and re-propagating from there. It is evaluated on DAVIS 2017 val
and on SA-V.

## Setup

Mac (CPU, tests only):

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Colab (GPU runs):

```
pip install -r requirements-colab.txt
```

Datasets, checkpoints and cached masks live in Google Drive. The environment variable that gives
their location is documented in `configs/`.
