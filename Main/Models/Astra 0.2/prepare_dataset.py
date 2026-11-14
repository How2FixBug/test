#!/usr/bin/env python3
"""Prepare ASTRA 0.2's conversation-only corpus in one command."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

def run(label: str, *args: str) -> None:
    print(f"\n{'=' * 10} {label} {'=' * 10}", flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)

def main() -> None:
    print("ASTRA 0.2: BASIC CONVERSATION ONLY", flush=True)
    print("No external datasets are downloaded or imported.", flush=True)
    run("1/3 Build tokenizer and encoded dataset", str(HERE / "tokenizer.py"))
    run("2/3 Audit conversation corpus", str(HERE / "audit_dataset.py"))
    run("3/3 Run correctness checks", str(HERE / "test_causality.py"))
    print("\nPreparation complete. Review the audit before training.", flush=True)

if __name__ == "__main__":
    main()
