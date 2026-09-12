"""CuRe image classifier and strict checkpoint loading."""

from pathlib import Path

import torch
from torch import nn

from .backbone import PECore


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "weights" / "cure.pt"


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


def load_model(checkpoint=None, device="auto"):
    """Load all model parameters and return a frozen model producing two logits."""
    path = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
    if not path.is_file():
        raise FileNotFoundError("CuRe weights are missing. Place cure.pt in weights/ or pass --checkpoint.")
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu.")
    model = CuRe().float()
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.to(device).eval()
