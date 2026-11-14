#!/usr/bin/env python3
"""External fact/instruction datasets are intentionally disabled for ASTRA 0.2.

ASTRA 0.2 is trained only on a curated basic-conversation corpus. Do not import
general-purpose instruction, trivia, science, or knowledge datasets here.
"""
raise SystemExit(
    "Dataset importing is disabled for ASTRA 0.2. "
    "Its scope is basic conversation only; edit dataset.txt with reviewed "
    "everyday conversational examples instead."
)
