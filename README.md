# CuRe

Image authenticity inference with PE Core L/14, Q/V LoRA, a 128-dimensional
feature projection, and a binary classifier.

## Setup

Use Python 3.10 or newer. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The complete model is stored in `weights/cure.pt`. If it is not included in
your local copy, download the release attachment from inside the cloned
repository using an authenticated GitHub CLI:

```bash
gh release download v1.0.0 --pattern cure.pt --dir weights
shasum -a 256 -c weights/cure.pt.sha256
```

The checkpoint includes all required visual backbone, LoRA, projection and
classifier parameters. Inference works offline after installing dependencies
and obtaining this file. No separate PE installation is required.

## Inference

One image:

```bash
python infer.py --input images/example.jpg
```

A directory, including subdirectories:

```bash
python infer.py --input images --recursive --batch-size 8 --output predictions.jsonl
```

`--device auto` uses CUDA when available and CPU otherwise. Use `--device cpu`
to select CPU explicitly, or `--device cuda:0` to select a GPU. CUDA uses
automatic mixed precision. Start with a small batch if GPU memory is limited.
Use `--checkpoint weights/cure.pt` to select a different checkpoint location.

Each output line contains the image name relative to the input directory,
the probability of being generated, and the decision at threshold 0.5:

```json
{"image": "example.jpg", "probability_fake": 0.02, "prediction": "real"}
```

Supported image formats are JPEG, PNG, WebP, BMP and TIFF. Invalid images stop
the command with an error. Images are converted to RGB; PNG inputs are
re-encoded as JPEG Q96. All images then use bicubic short-side resize to 390,
390 x 390 center crop, bicubic resize to 336 x 336, JPEG Q70, and channel
normalization with mean and standard deviation both equal to 0.5.

## Python API

```python
import torch
from cure import load_model, preprocess_image

model = load_model(device="cpu")
image = preprocess_image("images/example.jpg").unsqueeze(0)
with torch.inference_mode():
    probability_fake = model(image).softmax(dim=1)[0, 1].item()
print(probability_fake)
```

The model returns two logits in the order `[real, fake]`. Checkpoint loading
uses `strict=True`. See `NOTICE` and `LICENSE` for the PE component attribution.
