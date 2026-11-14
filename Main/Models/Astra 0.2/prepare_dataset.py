#!/usr/bin/env python3
"""Prepare the ASTRA 0.2 dataset in one repeatable command.

Imports up to 4,000 examples from each configured source, regenerates the BPE
tokenizer and tokenized data, audits the result, and runs correctness checks.
Existing prompt/answer pairs are deduplicated, so rerunning is safe.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

def run(label: str, *args: str) -> None:
    print(f"\n{'=' * 12} {label} {'=' * 12}", flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)

def main() -> None:
    run("1/4 Importing conversational examples", str(HERE / "import_datasets.py"),
        "--limit-per-source", "4000")
    run("2/4 Rebuilding tokenizer and encoded dataset", str(HERE / "tokenizer.py"))
    run("3/4 Auditing dataset", str(HERE / "audit_dataset.py"))
    run("4/4 Running correctness checks", str(HERE / "test_causality.py"))
    print("\nASTRA 0.2 dataset preparation finished.", flush=True)
    print("Review the audit and inspect samples before training.", flush=True)
    print("Note: this corpus is currently single-turn; multi-turn training support is still pending.", flush=True)

if __name__ == "__main__":
    main()
