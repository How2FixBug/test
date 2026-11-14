#!/usr/bin/env python3
"""Export and publish the current ASTRA 0.1 browser model in one command.

Run from any directory:
    python3 "/path/to/Main/Models/Astra 0.1/publish.py"

The script exports ONNX, verifies that the checkpoint and tokenizer match,
stages only the two browser model artifacts, commits them if changed, and
pushes the current branch to its configured upstream.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CHECKPOINT = HERE / "checkpoints" / "astra-0.1-best.pt"
TOKENIZER_CONFIG = HERE / "tokenizer_config.json"
ONNX_MODEL = HERE / "astra-0.1.onnx"

RELEASE_FILES = [
    ONNX_MODEL.relative_to(REPO_ROOT),
    TOKENIZER_CONFIG.relative_to(REPO_ROOT),
]


def run(command: list[str], *, cwd: Path = REPO_ROOT) -> str:
    """Run a command and return stdout, preserving useful error output."""
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode:
        raise SystemExit(result.returncode)
    return result.stdout.strip()


def validate_source_artifacts() -> dict:
    if not CHECKPOINT.is_file():
        raise SystemExit(f"Checkpoint not found: {CHECKPOINT}")
    if not TOKENIZER_CONFIG.is_file():
        raise SystemExit(f"Tokenizer config not found: {TOKENIZER_CONFIG}")

    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch is missing from this Python environment.")

    config = json.loads(TOKENIZER_CONFIG.read_text(encoding="utf-8"))
    required = ("fingerprint", "vocab_size", "vocab_hex", "merges")
    if any(key not in config for key in required):
        raise SystemExit("Tokenizer config is incomplete. Run tokenizer.py first.")
    if config["vocab_size"] != len(config["vocab_hex"]):
        raise SystemExit("Tokenizer vocab_size does not match vocab_hex length.")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture_version") != 3:
        raise SystemExit("Checkpoint is not an ASTRA 0.1 architecture-v3 checkpoint.")
    if checkpoint.get("tokenizer_fingerprint") != config["fingerprint"]:
        raise SystemExit(
            "Checkpoint and tokenizer fingerprints do not match. "
            "Do not publish until the model has been trained with this tokenizer."
        )
    model_vocab = checkpoint.get("model_config", {}).get("vocab_size")
    if model_vocab != config["vocab_size"]:
        raise SystemExit(
            f"Checkpoint vocabulary ({model_vocab}) does not match "
            f"tokenizer vocabulary ({config['vocab_size']})."
        )
    return config


def main() -> None:
    if not (REPO_ROOT / ".git").exists():
        raise SystemExit(f"Git repository not found at {REPO_ROOT}")

    print("\n[1/5] Checking checkpoint and tokenizer...")
    config = validate_source_artifacts()
    print(
        f"Matched tokenizer: {config['vocab_size']:,} tokens "
        f"| fingerprint {config['fingerprint'][:12]}..."
    )

    print("\n[2/5] Exporting ONNX model...")
    run([sys.executable, str(HERE / "export_onnx.py")], cwd=HERE)
    if not ONNX_MODEL.is_file() or ONNX_MODEL.stat().st_size == 0:
        raise SystemExit("ONNX export did not produce a model file.")

    print("\n[3/5] Checking Git publishing destination...")
    branch = run(["git", "branch", "--show-current"])
    if not branch:
        raise SystemExit("Could not determine the current Git branch.")
    upstream = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if upstream.returncode:
        raise SystemExit(
            "This branch has no configured Git upstream. Set its upstream before publishing."
        )
    print(f"Branch: {branch} -> {upstream.stdout.strip()}")

    print("\n[4/5] Staging browser model files...")
    run(["git", "add", "--", *[str(path) for path in RELEASE_FILES]])
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=REPO_ROOT,
    ).returncode != 0

    if changed:
        run([
            "git", "commit",
            "-m", "Publish ASTRA 0.1 browser model",
        ])
    else:
        print("Browser model files are already up to date; no commit needed.")

    print("\n[5/5] Pushing to GitHub...")
    run(["git", "push"])
    print("\nASTRA 0.1 is published. GitHub Pages can now deploy the updated model.")


if __name__ == "__main__":
    main()
