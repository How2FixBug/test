#!/usr/bin/env python3
"""Remove unreviewed BYU Chit-Chat imports from ASTRA's training dataset."""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "dataset.txt"
SOURCE_BLOCK_RE = re.compile(
    r"(?ms)^<!-- ASTRA_SOURCE: BYU-PCCL/chitchat-dataset(?: \\(MIT\\))? -->\\n"
    r"<user>\\n.*?\\n</astra>(?:\\n\\s*---)?\\n*"
)


def main() -> None:
    raw = DATASET_PATH.read_text(encoding="utf-8")
    cleaned, removed = SOURCE_BLOCK_RE.subn("", raw)
    if removed:
        DATASET_PATH.write_text(cleaned.rstrip() + "\\n", encoding="utf-8")
    print(f"Unreviewed BYU examples removed: {removed:,}")
    print("Curated examples preserved.")


if __name__ == "__main__":
    main()
