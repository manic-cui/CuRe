# CuRe

Image authenticity inference. Requires Python 3.10 or newer.

## Setup

```bash
pip install -r requirements.txt
```

Place the complete checkpoint at `weights/cure.pt`, or download it from
the release using an authenticated GitHub CLI inside the cloned repository:

```bash
gh release download v1.0.0 --pattern cure.pt --dir weights
shasum -a 256 -c weights/cure.pt.sha256
```

## Inference

```bash
python infer.py --input image.jpg
python infer.py --input images --recursive --batch-size 8
```

Predictions are written to stdout. Use `--output output.jsonl` to save them,
`--device cpu` or `--device cuda:0` to select a device, and `--checkpoint`
to select another checkpoint. Run `python infer.py --help` for all options.

Checkpoint loading uses `strict=True`. See `LICENSE` and `NOTICE` for
component attribution.
