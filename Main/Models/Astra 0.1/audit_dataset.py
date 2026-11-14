#!/usr/bin/env python3
"""Audit ASTRA's text corpus and tokenized training data without changing them."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIR_RE = re.compile(
    r"(?m)^<user>\\s*\\n([\\s\\S]*?)\\n</user>\\s*\\n"
    r"<astra>\\s*\\n([\\s\\S]*?)\\n</astra>"
)
