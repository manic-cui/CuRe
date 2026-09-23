"""CuRe image classifier: public PE Core backbone plus the released CuRe adapter."""

from pathlib import Path

import torch
from torch import nn

from .backbone import PECore


WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"
DEFAULT_CHECKPOINT = WEIGHTS_DIR / "cure_adapter.pt"

# Frozen backbone: Perception Encoder Core L/14 (336 px), released by Meta under
# Apache-2.0. Only its visual tower is used; it is downloaded once and cached.
BASE_REPO_ID = "facebook/PE-Core-L14-336"
BASE_FILENAME = "PE-Core-L14-336.pt"


class CuRe(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = PECore()
        self.register_buffer("mu", torch.empty(1024))
        # Preserve the checkpoint's column-major projection layout.
        self.register_buffer("W", torch.empty(128, 1024).t())
        self.head = nn.Linear(128, 2)

    @torch.inference_mode()
    def forward(self, images):
        with torch.amp.autocast("cuda", enabled=images.is_cuda):
            features = self.backbone(images)
        return self.head((features - self.mu) @ self.W)


def _resolve_base_checkpoint(base_checkpoint=None):
    """Return a local path to the PE Core checkpoint, downloading it if needed."""
    if base_checkpoint is not None:
        path = Path(base_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Base checkpoint not found: {path}")
        return path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the PE Core backbone; "
            "install it or pass --base-checkpoint."
        ) from exc
    return Path(hf_hub_download(BASE_REPO_ID, BASE_FILENAME))


def _load_base_state(base_checkpoint=None):
    """Load the frozen visual-tower weights and rename them into this module."""
    path = _resolve_base_checkpoint(base_checkpoint)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    prefix = "visual."
    return {"backbone." + k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def load_model(checkpoint=None, device="auto", base_checkpoint=None):
    """Load all model parameters and return a frozen model producing two logits.

    ``checkpoint`` is the CuRe adapter (LoRA, subspace and classifier head). The
    frozen backbone is taken from ``base_checkpoint`` or downloaded from
    Hugging Face; the two are merged and loaded strictly.
    """
    path = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
    if not path.is_file():
        raise FileNotFoundError("CuRe weights are missing. Place cure_adapter.pt in weights/ or pass --checkpoint.")
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu.")
    model = CuRe().float()
    expected = set(model.state_dict())
    adapter = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    base = _load_base_state(base_checkpoint)
    state = {k: v for k, v in base.items() if k in expected}
    overlap = sorted(set(state) & set(adapter))
    if overlap:
        raise RuntimeError(f"Adapter overrides frozen backbone weights: {overlap[:3]}")
    state.update(adapter)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.to(device).eval()
