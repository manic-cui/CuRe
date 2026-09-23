# CuRe

Image authenticity inference. Requires Python 3.10 or newer.

## Setup

```bash
pip install -r requirements.txt
shasum -a 256 -c weights/cure_adapter.pt.sha256
```

`weights/cure_adapter.pt` (about 25 MB) ships with this repository and holds
every trained parameter of CuRe: the Stage-1 LoRA updates, the Stage-2
source-response subspace (`mu`, `W`), and the linear classifier head.

The frozen backbone, Perception Encoder Core L/14 at 336 px, is not stored
here. On first use it is downloaded automatically from the public Hugging Face
repository `facebook/PE-Core-L14-336` (file `PE-Core-L14-336.pt`, 2.7 GB,
Apache-2.0) and cached by `huggingface_hub`. If the machine cannot reach
Hugging Face, either set `HF_ENDPOINT` to a mirror or download the file once
and pass `--base-checkpoint /path/to/PE-Core-L14-336.pt`.

## Inference

```bash
python infer.py --input image.jpg
python infer.py --input images --recursive --batch-size 8
python infer.py --input image.jpg --base-checkpoint /path/to/PE-Core-L14-336.pt
```

Predictions are written to stdout. Use `--output output.jsonl` to save them
and `--checkpoint` to select another adapter. Run `python infer.py --help`
for all options.

Inference runs on the GPU by default (`--device cuda`, or `--device cuda:N`
to pick a card) with fp16 automatic mixed precision, which is the setting
used for all reported results. `--device cpu` is supported for machines
without a GPU; it computes in fp32, so scores can differ from the reported
ones in the third decimal place.

The backbone weights and the adapter are merged into one state dict and loaded
with `strict=True`, so any missing or unexpected tensor is an error. See
`LICENSE` and `NOTICE` for component attribution.
