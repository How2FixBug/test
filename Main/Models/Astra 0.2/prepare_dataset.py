#!/usr/bin/env python3
"""Clean legacy candidates, then prepare and audit ASTRA 0.2."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(label: str, *args: str) -> None:
    print(f"\n{'=' * 10} {label} {'=' * 10}", flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def main() -> None:
    print("ASTRA 0.2: GENERAL-PURPOSE ASSISTANT", flush=True)
    run("1/4 Remove legacy unreviewed candidates", str(HERE / "clean_dataset.py"))
    run("2/4 Build tokenizer and encoded dataset", str(HERE / "tokenizer.py"))
    run("3/4 Audit general-purpose corpus", str(HERE / "audit_dataset.py"))
    run("4/4 Run correctness checks", str(HERE / "test_causality.py"))
    print("\nPreparation complete. Review the audit before training.", flush=True)
    print("No model training was started.", flush=True)


if __name__ == "__main__":
    main()
