#!/usr/bin/env python3
"""Predict image authenticity for one image or a directory of images."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import torch

from cure import load_model, preprocess_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Image file or directory")
    parser.add_argument("--output", type=Path, help="JSONL destination; defaults to stdout")
    parser.add_argument("--checkpoint", type=Path, help="CuRe adapter weights; defaults to weights/cure_adapter.pt")
    parser.add_argument("--base-checkpoint", type=Path,
                        help="Local PE-Core-L14-336.pt; defaults to downloading it from Hugging Face")
    parser.add_argument("--device", default="cuda",
                        help="cuda (default, fp16 autocast as in the paper), cuda:N, or cpu (fp32; "
                             "scores may differ from the paper in the third decimal)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--recursive", action="store_true", help="Include images in subdirectories")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    source = args.input.expanduser().resolve()
    if source.is_file():
        images, relative_root = [source], source.parent
    elif source.is_dir():
        paths = source.rglob("*") if args.recursive else source.iterdir()
        images = sorted(p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        relative_root = source
    else:
        parser.error("--input does not exist")
    if not images:
        parser.error("No supported images found")
    if args.output and args.output.resolve() in images:
        parser.error("--output must not overwrite an input image")

    torch.set_num_threads(4)
    try:
        model = load_model(args.checkpoint, args.device, args.base_checkpoint)
        device = next(model.parameters()).device
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        destination = args.output.open("w", encoding="utf-8") if args.output else nullcontext(sys.stdout)
        with destination as handle, torch.inference_mode():
            for start in range(0, len(images), args.batch_size):
                batch_paths = images[start:start + args.batch_size]
                tensors = []
                for path in batch_paths:
                    try:
                        tensors.append(preprocess_image(path))
                    except Exception as exc:
                        raise RuntimeError(f"Could not read image: {path.relative_to(relative_root)}") from exc
                batch = torch.stack(tensors).to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    scores = model(batch).softmax(dim=1)[:, 1]
                if not torch.isfinite(scores).all():
                    raise RuntimeError("Model returned non-finite probabilities")
                for path, probability in zip(batch_paths, scores.float().cpu().tolist()):
                    result = {
                        "image": path.relative_to(relative_root).as_posix(),
                        "probability_fake": probability,
                        "prediction": "fake" if probability >= 0.5 else "real",
                    }
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Processed {len(images)} image(s).", file=sys.stderr)


if __name__ == "__main__":
    main()
