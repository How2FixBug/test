#!/usr/bin/env python3
"""Export the trained ASTRA 0.1 checkpoint to a dynamic-sequence ONNX model.

Run from this directory:
    python3 export_onnx.py

Requires the existing PyTorch environment plus the 'onnx' package.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import MethodType

import torch

from model import AstraGPT

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / "checkpoints" / "astra-0.1-best.pt"
OUTPUT = HERE / "astra-0.1.onnx"


def export_attention(self, x: torch.Tensor) -> torch.Tensor:
    """Equivalent causal attention written with ONNX-export-friendly ops."""
    batch, length, width = x.shape
    qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_width)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_width)
    causal_mask = torch.triu(
        torch.ones((length, length), dtype=torch.bool, device=x.device),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores, dim=-1)
    attended = torch.matmul(probabilities, v)
    attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
    return self.out(attended)


def main() -> None:
    if not CHECKPOINT.exists():
        raise SystemExit(f"Checkpoint not found: {CHECKPOINT}")

    try:
        import onnx  # noqa: F401
    except ImportError:
        raise SystemExit(
            "The ONNX exporter package is missing. Install it with:\n"
            "python3 -m pip install onnx"
        )

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture_version") != 3:
        raise SystemExit("This is not an ASTRA 0.1 architecture-v3 checkpoint.")

    config = checkpoint["model_config"]
    model = AstraGPT(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Replace SDPA only for export. The mathematical operation and weights
    # remain unchanged; this expresses causal attention as standard ONNX ops.
    for block in model.blocks:
        block.attention.forward = MethodType(export_attention, block.attention)

    example = torch.tensor([[256, 10, 258, 10, 300, 301, 302, 303]], dtype=torch.long)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (example,),
            str(OUTPUT),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {1: "sequence_length"},
                "logits": {1: "sequence_length"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    exported = onnx.load(str(OUTPUT))
    onnx.checker.check_model(exported)
    print("ASTRA 0.1 ONNX export successful.")
    print(f"Model: {OUTPUT}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Size: {OUTPUT.stat().st_size / (1024 * 1024):.1f} MB")
    print("Input: int64 [1, sequence_length]")
    print("Output: float32 [1, sequence_length, vocab_size]")


if __name__ == "__main__":
    main()
