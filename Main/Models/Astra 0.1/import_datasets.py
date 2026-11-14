#!/usr/bin/env python3
"""Import attribution-compatible public conversation datasets into dataset.txt.

Uses only the Python standard library. It downloads OASST1 and queries the
Hugging Face dataset viewer for Aya. Imported records receive an EMPTY think
section: the sources do not provide ASTRA reasoning summaries, and we must not
invent or imitate hidden chain-of-thought.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset.txt"
OASST_URL = (
    "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/"
    "694bedfd102490866caa5507d1c93b6c66484367/"
    "2023-04-12_oasst_ready.messages.jsonl.gz"
)
AYA_ROWS_URL = "https://datasets-server.huggingface.co/rows"
OASST_SOURCE = "OpenAssistant/oasst1"
AYA_SOURCE = "CohereLabs/aya_dataset"


def clean(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", "").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def safe_pair(user: str, answer: str) -> bool:
    if not user or not answer or len(user) < 3 or len(answer) < 8:
        return False
    if len(user) > 3000 or len(answer) > 5000:
        return False
    if re.search(r"<\s*/?\s*(user|astra|think)\s*>", user + answer, re.I):
        return False
    return True


def existing_pairs(raw: str) -> set[tuple[str, str]]:
    pattern = re.compile(
        r"(?m)^<user>\s*\n([\s\S]*?)\n</user>\s*\n"
        r"<astra>\s*\n<think>[\s\S]*?</think>\s*\n([\s\S]*?)\n</astra>"
    )
    return {
        (normalize(u), normalize(a))
        for u, a in pattern.findall(raw)
    }


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def quality_ok(message: dict) -> bool:
    if message.get("deleted") is True or message.get("review_result") is False:
        return False
    labels = message.get("labels") or {}
    for name in ("pii", "sexual_content", "hate_speech", "not_appropriate", "spam"):
        item = labels.get(name) or {}
        value = item.get("value")
        if isinstance(value, (int, float)) and value >= 0.5:
            return False
    return True


def import_oasst(limit: int) -> list[tuple[str, str]]:
    print("Downloading OASST1 ready-for-export messages (about 34 MB)...")
    request = urllib.request.Request(OASST_URL, headers={"User-Agent": "ASTRA-dataset-import/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with gzip.GzipFile(fileobj=response) as stream:
            messages = {}
            for raw_line in stream:
                try:
                    item = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if item.get("lang") != "en" or not quality_ok(item):
                    continue
                if item.get("synthetic") is True:
                    continue
                text = clean(item.get("text"))
                if not text:
                    continue
                messages[item.get("message_id")] = item

    pairs = []
    for item in messages.values():
        if item.get("role") != "assistant":
            continue
        parent = messages.get(item.get("parent_id"))
        if not parent or parent.get("role") != "prompter":
            continue
        if item.get("rank") not in (None, 0):
            continue
        user, answer = clean(parent.get("text")), clean(item.get("text"))
        if safe_pair(user, answer):
            pairs.append((user, answer))
    # Stable ordering and exact deduplication.
    unique = {}
    for pair in pairs:
        unique[(normalize(pair[0]), normalize(pair[1]))] = pair
    result = list(unique.values())
    result.sort(key=lambda pair: (len(pair[0]) + len(pair[1]), normalize(pair[0])))
    print(f"OASST1: {len(result):,} clean English prompt/answer pairs available.")
    return result[:limit]


def import_aya(limit: int) -> list[tuple[str, str]]:
    print("Reading Aya rows through the Hugging Face dataset viewer (no extra packages)...")
    pairs, offset, page_size = [], 0, 100
    while len(pairs) < limit:
        params = urllib.parse.urlencode({
            "dataset": AYA_SOURCE,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": page_size,
        })
        request = urllib.request.Request(
            AYA_ROWS_URL + "?" + params,
            headers={"User-Agent": "ASTRA-dataset-import/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
        except Exception as exc:
            if offset == 0:
                raise RuntimeError(
                    "Aya viewer endpoint was unavailable. OASST1 can still be imported; "
                    "rerun with --sources oasst to skip Aya."
                ) from exc
            break
        rows = payload.get("rows", [])
        if not rows:
            break
        for entry in rows:
            row = entry.get("row", {})
            language = str(row.get("language", row.get("lang", ""))).strip().lower()
            if language not in {"en", "eng", "english"}:
                continue
            user = clean(row.get("inputs", row.get("input", row.get("prompt", row.get("instruction", "")))))
            answer = clean(row.get("targets", row.get("target", row.get("response", row.get("completion", "")))))
            if safe_pair(user, answer):
                pairs.append((user, answer))
                if len(pairs) >= limit:
                    break
        offset += len(rows)
        if offset % 1000 == 0:
            print(f"  checked {offset:,} Aya rows; kept {len(pairs):,} English pairs")
        if len(rows) < page_size:
            break
        time.sleep(0.05)
    unique = {}
    for pair in pairs:
        unique[(normalize(pair[0]), normalize(pair[1]))] = pair
    result = list(unique.values())
    print(f"Aya: {len(result):,} English prompt/answer pairs available.")
    return result[:limit]


def append_pairs(pairs: list[tuple[str, str]], source: str) -> int:
    raw = DATASET.read_text(encoding="utf-8")
    known = existing_pairs(raw)
    added = []
    for user, answer in pairs:
        key = (normalize(user), normalize(answer))
        if key in known:
            continue
        known.add(key)
        added.append(
            "\n\n---\n\n<user>\n" + user + "\n</user>\n<astra>\n<think>\n</think>\n"
            + answer + "\n</astra>"
        )
    if added:
        with DATASET.open("a", encoding="utf-8") as output:
            output.write("\n" + "".join(added) + "\n")
    print(f"{source}: added {len(added):,} new examples.")
    return len(added)


def main():
    parser = argparse.ArgumentParser(description="Add clean, credited public datasets to ASTRA.")
    parser.add_argument("--limit-per-source", type=int, default=5000)
    parser.add_argument("--sources", nargs="+", choices=("oasst", "aya"), default=("oasst", "aya"))
    args = parser.parse_args()
    if args.limit_per_source < 1:
        parser.error("--limit-per-source must be positive")
    total = 0
    if "oasst" in args.sources:
        total += append_pairs(import_oasst(args.limit_per_source), OASST_SOURCE)
    if "aya" in args.sources:
        total += append_pairs(import_aya(args.limit_per_source), AYA_SOURCE)
    print(f"Done. Added {total:,} examples. Now run python3 tokenizer.py.")
    print("Review the resulting dataset and credits before training.")


if __name__ == "__main__":
    main()
