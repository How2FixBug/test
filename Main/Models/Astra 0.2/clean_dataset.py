#!/usr/bin/env python3
"""Remove legacy, unreviewed BYU candidate blocks while preserving verified imports."""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "dataset.txt"
# Older candidate exports used this marker without a license suffix. The new
# verified importer uses "(MIT)" and must remain in the training corpus.
LEGACY_BLOCK_RE = re.compile(
    r"(?ms)^<!-- ASTRA_SOURCE: BYU-PCCL/chitchat-dataset -->\n"
    r"<user>\n.*?\n</astra>(?:\n\s*---)?\n*"
)


def main() -> None:
    if not DATASET_PATH.exists():
        print("dataset.txt not found; nothing to clean.")
        return
    raw = DATASET_PATH.read_text(encoding="utf-8")
    cleaned, removed = LEGACY_BLOCK_RE.subn("", raw)
    if removed:
        DATASET_PATH.write_text(cleaned.rstrip() + "\n", encoding="utf-8")
    print(f"Legacy unreviewed BYU candidate blocks removed: {removed:,}")
    print("Curated examples and verified MIT imports preserved.")


if __name__ == "__main__":
    main()
