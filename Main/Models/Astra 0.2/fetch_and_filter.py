#!/usr/bin/env python3
"""Fetch and filter Hugging Face everyday conversations for ASTRA 0.2."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "dataset.txt"
MANIFEST_PATH = HERE / "import_manifest.json"

SOURCE_ID = "HuggingFaceTB/everyday-conversations-llama3.1-2k"
SOURCE_LICENSE = "Apache-2.0"
SOURCE_REPO_URL = "https://huggingface.co/datasets/" + SOURCE_ID
SOURCE_LICENSE_URL = SOURCE_REPO_URL
SOURCE_API = "https://datasets-server.huggingface.co/rows"
EXPECTED_ROWS = 2260
PAGE_SIZE = 100
SOURCE_MARKER = f"<!-- ASTRA_SOURCE: {SOURCE_ID} ({SOURCE_LICENSE}) -->"

PAIR_RE = re.compile(
    r"(?m)^<user>\s*\n([\s\S]*?)\n</user>\s*\n"
    r"<astra>\s*\n(?:<think>[\s\S]*?</think>\s*)?([\s\S]*?)\n</astra>"
)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[ .()/-]?){8,15}(?!\w)")
UNSAFE_RE = re.compile(
    r"\b(?:porn|nudes?|explicit sex|suicide|self[- ]harm|"
    r"cocaine|heroin|methamphetamine|"
    r"nigger|faggot|cunt|whore|slut|"
    r"fuck(?:ing|ed|er)?|shit(?:ty)?|bitch(?:es)?|asshole)\b", re.I
)
PERSONA_RE = re.compile(
    r"\b(?:"
    r"i\s+(?:live|lived|work|worked|study|studied|own|owned|visited|went|"
    r"grew up|used to|remember|recall|love|loved|like|liked|hate|hated|"
    r"prefer|preferred|enjoy|enjoyed|adore|miss|made|cooked|bought|played|"
    r"travelled|traveled|am from|come from|have a (?:dog|cat|pet|wife|husband|"
    r"girlfriend|boyfriend|brother|sister|mom|dad))|"
    r"i(?:'ve| have)\s+(?:been|visited|lived|worked|studied|owned|travelled|traveled)|"
    r"i(?:'m| am)\s+(?:from|at|in)\b|"
    r"i(?:'m| am)\s+(?:a|an)\s+(?!ai\b|chatbot\b|assistant\b|language model\b)[a-z]+\b|"
    r"my\s+(?:mom|mother|dad|father|family|parents|brother|sister|friend|"
    r"girlfriend|boyfriend|wife|husband|kids|children|pet|dog|cat)\b"
    r")",
    re.I,
)
MAX_USER_CHARS = 350
MAX_ASSISTANT_CHARS = 300


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def acceptable_turn(value: object, *, assistant: bool) -> bool:
    if not isinstance(value, str):
        return False
    text = re.sub(r"\s+", " ", value).strip()
    words = text.split()
    limit = MAX_ASSISTANT_CHARS if assistant else MAX_USER_CHARS
    if not (1 <= len(text) <= limit and 1 <= len(words) <= 65):
        return False
    if URL_RE.search(text) or EMAIL_RE.search(text) or PHONE_RE.search(text):
        return False
    if UNSAFE_RE.search(text):
        return False
    if assistant and PERSONA_RE.search(text):
        return False
    return True


def fetch_rows() -> tuple[list[dict], str]:
    rows: list[dict] = []
    total = None
    for offset in range(0, EXPECTED_ROWS, PAGE_SIZE):
        params = urllib.parse.urlencode({
            "dataset": SOURCE_ID,
            "config": "default",
            "split": "train_sft",
            "offset": offset,
            "length": min(PAGE_SIZE, EXPECTED_ROWS - offset),
        })
        request = urllib.request.Request(
            f"{SOURCE_API}?{params}",
            headers={"User-Agent": "ASTRA-0.2-dataset-importer"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            page = json.loads(response.read().decode("utf-8"))
        if total is None:
            total = int(page.get("num_rows_total", -1))
            if total != EXPECTED_ROWS:
                raise ValueError(
                    f"Expected {EXPECTED_ROWS} training rows, but the Hub reports {total}."
                )
        page_rows = page.get("rows")
        if not isinstance(page_rows, list) or not page_rows:
            raise ValueError(f"Dataset API returned no rows at offset {offset}.")
        rows.extend(item["row"] for item in page_rows if isinstance(item, dict) and isinstance(item.get("row"), dict))
        print(f"Fetched {len(rows):,}/{EXPECTED_ROWS:,} rows...", flush=True)

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, received {len(rows)}.")
    digest = hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return rows, digest


def existing_pairs() -> set[tuple[str, str]]:
    if not DATASET_PATH.exists():
        return set()
    raw = DATASET_PATH.read_text(encoding="utf-8")
    return {(normalize(user), normalize(answer)) for user, answer in PAIR_RE.findall(raw)}


def collect_candidates(rows: list[dict]) -> tuple[list[tuple[str, str]], Counter]:
    stats = Counter()
    candidates: list[tuple[str, str]] = []
    for row in rows:
        stats["conversations_seen"] += 1
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
            stats["invalid_conversations"] += 1
            continue
        turns = []
        valid = True
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                valid = False
                break
            expected_role = "user" if index % 2 == 0 else "assistant"
            role = message.get("role")
            content = message.get("content")
            if role != expected_role or not isinstance(content, str) or not content.strip():
                valid = False
                break
            turns.append(content.strip())
        if not valid:
            stats["invalid_conversations"] += 1
            continue

        for index in range(0, len(turns), 2):
            user, assistant = turns[index], turns[index + 1]
            if not acceptable_turn(user, assistant=False):
                stats["filtered_user_turn"] += 1
                continue
            if not acceptable_turn(assistant, assistant=True):
                stats["filtered_assistant_turn"] += 1
                continue
            candidates.append((user, assistant))
            stats["passed_filters"] += 1

    random.Random(20261114).shuffle(candidates)
    return candidates, stats


def write_manifest(added: int, stats: Counter, source_sha256: str) -> None:
    manifest = {}
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {}
    sources = manifest.setdefault("sources", {})
    previous = sources.get(SOURCE_ID, {})
    sources[SOURCE_ID] = {
        "license": SOURCE_LICENSE,
        "license_url": SOURCE_LICENSE_URL,
        "repository_url": SOURCE_REPO_URL,
        "source_api": SOURCE_API,
        "expected_training_rows": EXPECTED_ROWS,
        "snapshot_sha256": source_sha256,
        "last_imported_utc": datetime.now(timezone.utc).isoformat(),
        "total_added": int(previous.get("total_added", 0)) + added,
        "last_run_added": added,
        "last_run_filter_counts": dict(stats),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter everyday assistant conversations for ASTRA 0.2."
    )
    parser.add_argument("--max-add", type=int, default=2500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()
    if args.max_add < 1 or args.sample_count < 0:
        parser.error("--max-add must be positive and --sample-count cannot be negative.")

    print(f"Fetching {SOURCE_ID} training split...")
    rows, source_sha256 = fetch_rows()
    candidates, stats = collect_candidates(rows)

    known = existing_pairs()
    additions = []
    seen = set(known)
    for user, assistant in candidates:
        key = (normalize(user), normalize(assistant))
        if key in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(key)
        additions.append((user, assistant))
        if len(additions) >= args.max_add:
            break

    print("\nASTRA PUBLIC DATASET IMPORT")
    print(f"Source: {SOURCE_ID}")
    print(f"License: {SOURCE_LICENSE}")
    print(f"Training conversations scanned: {stats['conversations_seen']:,}")
    print(f"Pairs passing filters: {stats['passed_filters']:,}")
    print(f"New unique pairs: {len(additions):,}")
    print(f"Existing dataset pairs: {len(known):,}")
    print(f"Snapshot SHA-256: {source_sha256}")
    print("Filter counts:")
    for name, count in sorted(stats.items()):
        if name not in {"conversations_seen", "passed_filters"}:
            print(f"  {name.replace('_', ' ')}: {count:,}")

    if args.sample_count and additions:
        sample = random.Random(42).sample(additions, min(args.sample_count, len(additions)))
        print(f"\nQUALITY SAMPLE ({len(sample)} examples)")
        for number, (user, assistant) in enumerate(sample, 1):
            print(f"\n[{number}/{len(sample)}]")
            print(f"USER: {user}")
            print(f"ASTRA: {assistant}")

    if args.dry_run:
        print("\nDry run only. dataset.txt and import_manifest.json were not changed.")
        return

    if additions:
        blocks = [
            f"{SOURCE_MARKER}\n<user>\n{user}\n</user>\n"
            f"<astra>\n{assistant}\n</astra>"
            for user, assistant in additions
        ]
        with DATASET_PATH.open("a", encoding="utf-8") as output:
            if DATASET_PATH.stat().st_size:
                output.write("\n\n")
            output.write("\n\n---\n\n".join(blocks) + "\n")
    write_manifest(len(additions), stats, source_sha256)
    print(f"\nAdded {len(additions):,} filtered examples to dataset.txt.")
    print("Apache-2.0 source attribution saved in import_manifest.json and dataset markers.")
    print("No training was started.")


if __name__ == "__main__":
    main()
