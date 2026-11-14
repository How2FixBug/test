#!/usr/bin/env python3
"""Compatibility entry point for the ASTRA general-purpose dataset importer."""
from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("fetch_and_filter.py")), run_name="__main__")
