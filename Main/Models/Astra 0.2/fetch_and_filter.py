#!/usr/bin/env python3
"""Import general-purpose assistant conversations from UltraChat 200k."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "dataset.txt"
MANIFEST_PATH = HERE / "import_manifest.json"

SOURCE_ID = "HuggingFaceH4/ultrachat_200k"
SOURCE_LICENSE = "MIT"
SOURCE_URL = f"https://huggingface.co/datasets/{SOURCE_ID}"
SOURCE_API = "https://datasets-server.huggingface.co/rows"
EXPECTED_ROWS = 207865
SCAN_ROWS = 12000
PAGE_SIZE = 100
SOURCE_MARKER = f"<!-- ASTRA_SOURCE: {SOURCE_ID} ({SOURCE_LICENSE}) -->"

PAIR_RE = re.compile(
    r"(?m)^<user>\s*\n([\s\S]*?)\n</user>\s*\n"
    r"<astra>\s*\n(?:<think>[\s\S]*?</think>\s*)?([\s\S]*?)\n</astra>"
)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[ .()/-]?){8,15}(?!\w)")
LONG_NUMBER_RE = re.compile(r"\b\d{7,}\b")

UNSAFE_RE = re.compile(
    r"\b(?:porn(?:ography)?|nudes?|explicit sexual content|"
    r"suicide|self[- ]harm|"
    r"cocaine|heroin|methamphetamine|fentanyl|"
    r"how to make (?:a bomb|an explosive)|"
    r"how to build (?:a bomb|a weapon)|"
    r"sexual intercourse|graphic sexual)\b",
    re.I,
)
HATE_RE = re.compile(
    r"\b(?:nigger|faggot|kike|spic|chink|tranny)\b",
    re.I,
)
PERSONA_RE = re.compile(
    r"\b(?:"
    r"i (?:live|lived|work|worked|study|studied|visited|went to|"
    r"grew up|used to|remember when|recall when|was born|"
    r"have a (?:wife|husband|girlfriend|boyfriend|son|daughter|"
    r"brother|sister|mom|dad|dog|cat))|"
    r"my (?:wife|husband|girlfriend|boyfriend|son|daughter|"
    r"brother|sister|mom|mother|dad|father|family|childhood|job)\b"
    r")",
    re.I,
)

MAX_PROMPT_CHARS = 1800
MAX_ANSWER_CHARS = 1400
MAX_CONTEXT_MESSAGES = 8


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def acceptable_text(value: object, *, assistant: bool) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    limit = MAX_ANSWER_CHARS if assistant else MAX_PROMPT_CHARS
    if not text or len(text) > limit or len(text.split()) > 280:
        return False
    if URL_RE.search(text) or EMAIL_RE.search(text) or PHONE_RE.search(text):
        return False
    if LONG_NUMBER_RE.search(text) or UNSAFE_RE.search(text) or HATE_RE.search(text):
        return False
    if assistant and PERSONA_RE.search(text):
        return False
    return True


def fetch_rows(scan_rows: int) -> tuple[list[dict], str]:
    rows: list[dict] = []
    total = None
    for offset in range(0, scan_rows, PAGE_SIZE):
        length = min(PAGE_SIZE, scan_rows - offset)
        params = urllib.parse.urlencode({
            "dataset": SOURCE_ID,
            "config": "default",
            "split": "train_sft",
            "offset": offset,
            "length": length,
        })
        request = urllib.request.Request(
            f"{SOURCE_API}?{params}",
            headers={"User-Agent": "ASTRA-0.2-general-dataset-importer"},
        )
        page = None
        for attempt in range(7):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    page = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 6:
                    raise
                retry_after = error.headers.get("Retry-After")
                try:
                    delay = max(1, min(120, int(retry_after))) if retry_after else min(60, 5 * (2 ** attempt))
                except ValueError:
                    delay = min(60, 5 * (2 ** attempt))
                print(f"Hub is busy. Retrying in {delay}s ({attempt + 1}/7)...", flush=True)
                time.sleep(delay)
            except (TimeoutError, urllib.error.URLError):
                if attempt == 6:
                    raise
                delay = min(60, 5 * (2 ** attempt))
                print(f"Network hiccup. Retrying in {delay}s ({attempt + 1}/7)...", flush=True)
                time.sleep(delay)
        if page is None:
            raise RuntimeError(f"Could not fetch dataset page at offset {offset}.")
        if total is None:
            total = int(page.get("num_rows_total", -1))
            if total != EXPECTED_ROWS:
                raise ValueError(f"Expected {EXPECTED_ROWS} rows, Hub reports {total}.")
        page_rows = page.get("rows")
        if not isinstance(page_rows, list):
            raise ValueError(f"Invalid dataset response at offset {offset}.")
        rows.extend(
            item["row"] for item in page_rows
            if isinstance(item, dict) and isinstance(item.get("row"), dict)
        )
        print(f"Fetched {len(rows):,}/{scan_rows:,} conversations...", flush=True)

    if len(rows) != scan_rows:
        raise ValueError(f"Expected to fetch {scan_rows} rows, received {len(rows)}.")
    digest = hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return rows, digest


def extract_pairs(rows: list[dict]) -> tuple[list[tuple[str, str]], Counter]:
    stats = Counter()
    candidates: list[tuple[str, str]] = []

    for row in rows:
        stats["conversations_scanned"] += 1
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            stats["invalid_conversations"] += 1
            continue

        history: list[tuple[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                stats["invalid_messages"] += 1
                history = []
                break
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
                stats["invalid_messages"] += 1
                history = []
                break
            history.append((role, content.strip()))

        if len(history) < 2 or len(history) % 2:
            stats["invalid_conversations"] += 1
            continue
        if any(history[i][0] != ("user" if i % 2 == 0 else "assistant") for i in range(len(history))):
            stats["invalid_turn_order"] += 1
            continue

        for assistant_index in range(1, len(history), 2):
            prompt_start = max(0, assistant_index - MAX_CONTEXT_MESSAGES)
            context = history[prompt_start:assistant_index]
            if not context or context[-1][0] != "user":
                continue
            prompt = "\n".join(
                ("User: " if role == "user" else "Assistant: ") + content
                for role, content in context
            )
            answer = history[assistant_index][1]
            if not acceptable_text(prompt, assistant=False):
                stats["filtered_prompts"] += 1
                continue
            if not acceptable_text(answer, assistant=True):
                stats["filtered_answers"] += 1
                continue
            candidates.append((prompt, answer))
            stats["pairs_extracted"] += 1

    return candidates, stats


def existing_pairs() -> set[tuple[str, str]]:
    if not DATASET_PATH.exists():
        return set()
    raw = DATASET_PATH.read_text(encoding="utf-8")
    return {(normalize(user), normalize(answer)) for user, answer in PAIR_RE.findall(raw)}


def write_manifest(added: int, stats: Counter, snapshot_hash: str, scanned: int) -> None:
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
        "repository_url": SOURCE_URL,
        "source_api": SOURCE_API,
        "training_split": "train_sft",
        "source_training_rows": EXPECTED_ROWS,
        "rows_scanned_this_run": scanned,
        "snapshot_sha256": snapshot_hash,
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
        description="Import broad, general-purpose assistant conversations into ASTRA 0.2."
    )
    parser.add_argument("--max-add", type=int, default=5000)
    parser.add_argument("--scan-rows", type=int, default=SCAN_ROWS)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_add < 1 or args.scan_rows < 100 or args.scan_rows > EXPECTED_ROWS:
        parser.error("--max-add must be positive and --scan-rows must be 100..207865.")
    if args.sample_count < 0:
        parser.error("--sample-count cannot be negative.")

    print(f"Fetching {SOURCE_ID} ({SOURCE_LICENSE})...")
    rows, snapshot_hash = fetch_rows(args.scan_rows)
    candidates, stats = extract_pairs(rows)

    known = existing_pairs()
    seen = set(known)
    unique = []
    for user, answer in candidates:
        key = (normalize(user), normalize(answer))
        if key in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(key)
        unique.append((user, answer))

    random.Random(20261114).shuffle(unique)
    additions = unique[:args.max_add]

    print("\nASTRA GENERAL-PURPOSE DATASET IMPORT")
    print(f"Source: {SOURCE_ID}")
    print(f"License: {SOURCE_LICENSE}")
    print(f"Training rows scanned: {len(rows):,}/{EXPECTED_ROWS:,}")
    print(f"Conversation pairs extracted: {stats['pairs_extracted']:,}")
    print(f"New unique pairs available: {len(unique):,}")
    print(f"Selected for import: {len(additions):,}")
    print(f"Existing pairs preserved: {len(known):,}")
    print(f"Snapshot SHA-256: {snapshot_hash}")
    print("Filter counts:")
    for name, count in sorted(stats.items()):
        if name not in {"conversations_scanned", "pairs_extracted"}:
            print(f"  {name.replace('_', ' ')}: {count:,}")

    if args.sample_count and additions:
        sample = random.Random(42).sample(additions, min(args.sample_count, len(additions)))
        print(f"\nQUALITY SAMPLE ({len(sample)} examples)")
        for number, (user, answer) in enumerate(sample, 1):
            print(f"\n[{number}/{len(sample)}]")
            print(f"USER:\n{user}")
            print(f"ASTRA:\n{answer}")

    if args.dry_run:
        print("\nDry run only. No files were changed.")
        return
    if not additions:
        raise SystemExit("No new examples passed the filters. Nothing was changed.")

    blocks = [
        f"{SOURCE_MARKER}\n<user>\n{user}\n</user>\n<astra>\n{answer}\n</astra>"
        for user, answer in additions
    ]
    with DATASET_PATH.open("a", encoding="utf-8") as output:
        if DATASET_PATH.stat().st_size:
            output.write("\n\n")
        output.write("\n\n---\n\n".join(blocks) + "\n")

    write_manifest(len(additions), stats, snapshot_hash, len(rows))
    print(f"\nImported {len(additions):,} general-purpose conversation examples.")
    print("The existing dataset, including the original curated examples, was preserved.")
    print("No tokenizer rebuild or model training was started.")


if __name__ == "__main__":
    main()
