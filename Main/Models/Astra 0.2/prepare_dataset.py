#!/usr/bin/env python3
"""Prepare and audit ASTRA 0.2's conversational corpus; never trains the model."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset_conversation_candidate.txt"


def run(label: str, *args: str) -> None:
    print(f"\\n{'=' * 10} {label} {'=' * 10}", flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def main() -> None:
    print("ASTRA 0.2: CONVERSATIONAL CORPUS", flush=True)
    run("1/3 Build tokenizer and encoded conversations", str(HERE / "tokenizer.py"), "--dataset", str(DATASET))
    run("2/3 Audit the exact candidate corpus", str(HERE / "audit_dataset.py"), "--dataset", str(DATASET))
    run("3/3 Run tokenizer and causality checks", str(HERE / "test_causality.py"))
    print("\\nPreparation complete. No model training was started.", flush=True)


if __name__ == "__main__":
    main()
