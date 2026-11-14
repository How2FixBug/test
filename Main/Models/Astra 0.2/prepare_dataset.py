#!/usr/bin/env python3
"""Import approved conversation data, then prepare and audit ASTRA 0.2."""
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
    run("1/4 Import and filter BYU Chit-Chat conversations", str(HERE / "import_chitchat.py"))
    run("2/4 Build tokenizer and encoded dataset", str(HERE / "tokenizer.py"))
    run("3/4 Audit conversation corpus", str(HERE / "audit_dataset.py"))
    run("4/4 Run correctness checks", str(HERE / "test_causality.py"))
    print("\nPreparation complete. Review the audit before training.", flush=True)
    print("To refresh the source data later: python3 import_chitchat.py --refresh", flush=True)


if __name__ == "__main__":
    main()
