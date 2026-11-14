#!/usr/bin/env python3
"""Fetch and conservatively filter public, licensed dialogue for ASTRA 0.2.

This importer currently uses only Blended Skill Talk, whose dataset is released
under CC BY 4.0. It never imports the Wizard of Wikipedia subset. Imported
examples are marked in dataset.txt so their source and license remain traceable.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import tarfile
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "dataset.txt"
CACHE_DIR = HERE / ".cache"
ARCHIVE_PATH = CACHE_DIR / "blended_skill_talk.tar.gz"
MANIFEST_PATH = HERE / "import_manifest.json"

SOURCE_ID = "facebookresearch/ParlAI blended_skill_talk"
SOURCE_URL = "https://parl.ai/downloads/blended_skill_talk/blended_skill_talk.tar.gz"
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_LICENSE_URL = "https://github.com/facebookresearch/ParlAI/blob/main/parlai/tasks/blended_skill_talk/README.md"
EXPECTED_SHA256 = "5fbed0068ee89e2d43b93c3ecb341e784617033efa5e8e911a219d4eda6134a6"
SOURCE_MARKER = f"<!-- ASTRA_SOURCE: {SOURCE_ID} ({SOURCE_LICENSE}) -->"

PAIR_RE = re.compile(
    r"(?m)^<user>\s*\n([\s\S]*?)\n</user>\s*\n"
    r"<astra>\s*\n(?:<think>[\s\S]*?</think>\s*)?([\s\S]*?)\n</astra>"
)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[ .()/-]?){8,15}(?!\w)")
SENSITIVE_RE = re.compile(
    r"\b(?:"
    r"porn|nudes?|explicit sex|suicide|self[- ]harm|"
    r"cocaine|heroin|methamphetamine|"
    r"nigger|faggot|cunt|whore|slut|"
    r"fuck(?:ing|ed|er)?|shit(?:ty)?|bitch(?:es)?|asshole"
    r")\b",
    re.I,
)
KNOWLEDGE_RE = re.compile(
    r"\b(?:"
    r"capital of|population of|president of|prime minister|"
    r"who invented|who discovered|when was .{0,40} invented|"
    r"what year did|how many|how much does|define |definition of|"
    r"explain (?:how|why)|according to|in conclusion|"
    r"photosynthesis|quantum mechanics|chemical reaction|"
    r"solar system|historical event|"
    r"wizard of wikipedia|wikipedia article"
    r")\b",
    re.I,
)
QUESTION_RE = re.compile(r"\b(?:what|who|when|where|which|how)\b.{0,90}\?", re.I)
SOCIAL_QUESTION_RE = re.compile(
    r"\b(?:how are you|how's it going|what are you up to|"
    r"what are you doing|what do you think|what do you like|"
    r"what's your favorite|what are your plans|where are you going|"
    r"what do you want to do|who are you with)\b",
    re.I,
)
ASSISTANT_BIO_RE = re.compile(
    r"\b(?:"
    r"i (?:am|was|have|had|live|lived|work|worked|study|studied|"
    r"own|owned|visited|went|grew up|used to|remember when)|"
    r"my (?:name|job|wife|husband|girlfriend|boyfriend|children|kids|"
    r"brother|sister|parents|family|house|car|school)"
    r")\b",
    re.I,
)
SCRIPTED_RE = re.compile(
    r"\b(?:as an ai language model|i cannot assist with that request|"
    r"i'll take that as a no|are you still there|"
    r"let me know if you need anything else)\b",
    re.I,
)
EXPLANATION_RE = re.compile(
    r"\b(?:therefore|for example|in conclusion|scientifically|"
    r"historically|the process of|is defined as|firstly|secondly)\b",
    re.I,
)
MAX_USER_CHARS = 240
MAX_ASSISTANT_CHARS = 220
MIN_WORDS = 2


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def looks_corrupted(value: str) -> bool:
    words = re.findall(r"[a-z']+", value.casefold())
    if len(words) >= 8:
        for i in range(len(words) - 3):
            if words[i] == words[i + 1] == words[i + 2] == words[i + 3]:
                return True
    if re.search(r"(.)\1{5,}", value):
        return True
    if len(value) >= 18:
        useful = sum(ch.isalnum() or ch.isspace() for ch in value)
        if useful / len(value) < 0.58:
            return True
    return False


def acceptable_turn(value: object, *, assistant: bool) -> bool:
    if not isinstance(value, str):
        return False
    text = re.sub(r"\s+", " ", value).strip()
    if not (2 <= len(text) <= (MAX_ASSISTANT_CHARS if assistant else MAX_USER_CHARS)):
        return False
    if len(text.split()) < MIN_WORDS:
        return False
    if URL_RE.search(text) or EMAIL_RE.search(text) or PHONE_RE.search(text):
        return False
    if SENSITIVE_RE.search(text) or looks_corrupted(text):
        return False
    if KNOWLEDGE_RE.search(text):
        return False
    if QUESTION_RE.search(text) and not SOCIAL_QUESTION_RE.search(text):
        return False
    if assistant and (ASSISTANT_BIO_RE.search(text) or SCRIPTED_RE.search(text)):
        return False
    if assistant and len(text) > 150 and EXPLANATION_RE.search(text):
        return False
    return True


def download_archive(force: bool = False) -> bytes:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_PATH.exists() and not force:
        payload = ARCHIVE_PATH.read_bytes()
    else:
        request = urllib.request.Request(
            SOURCE_URL, headers={"User-Agent": "ASTRA-0.2-public-dataset-importer"}
        )
        print("Downloading Blended Skill Talk (CC BY 4.0)...", flush=True)
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read(100 * 1024 * 1024 + 1)
        if len(payload) > 100 * 1024 * 1024:
            raise ValueError("Dataset archive exceeded the 100 MB safety limit.")
        ARCHIVE_PATH.write_bytes(payload)

    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        ARCHIVE_PATH.unlink(missing_ok=True)
        raise ValueError(
            "Dataset archive SHA-256 did not match the official ParlAI checksum. "
            "The download was discarded."
        )
    print(f"Verified source archive: {len(payload):,} bytes, SHA-256 OK.")
    return payload


def load_training_conversations(payload: bytes) -> list[dict]:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        member = next(
            (item for item in archive.getmembers()
             if item.isfile() and Path(item.name).name == "train.json"),
            None,
        )
        if member is None or member.size > 80 * 1024 * 1024:
            raise ValueError("Archive did not contain a safe, expected train.json.")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("Could not read train.json from the verified archive.")
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError("Unexpected Blended Skill Talk training format.")
    return data


def existing_pairs() -> set[tuple[str, str]]:
    if not DATASET_PATH.exists():
        return set()
    raw = DATASET_PATH.read_text(encoding="utf-8")
    return {
        (normalize(user), normalize(answer))
        for user, answer in PAIR_RE.findall(raw)
    }


def collect_candidates(conversations: list[dict]) -> tuple[list[tuple[str, str]], Counter]:
    stats = Counter()
    candidates: list[tuple[str, str]] = []

    for episode in conversations:
        stats["conversations_seen"] += 1
        if not isinstance(episode, dict):
            stats["invalid_conversations"] += 1
            continue

        # The Wizard of Wikipedia portion is deliberately excluded.
        context_source = str(episode.get("context_dataset", "")).casefold()
        if context_source not in {"convai2", "empathetic_dialogues"}:
            stats["excluded_context_source"] += 1
            continue

        dialog = episode.get("dialog")
        chosen = episode.get("chosen_suggestions", [])
        if not isinstance(dialog, list) or len(dialog) < 2:
            stats["invalid_conversations"] += 1
            continue

        # The source defines speaker 0 as the free participant and speaker 1
        # as the guided participant. We use only that direction consistently.
        for index in range(0, len(dialog) - 1, 2):
            if index + 1 >= len(chosen):
                stats["missing_suggestion_label"] += 1
                continue
            if str(chosen[index + 1]).casefold() == "wizard_of_wikipedia":
                stats["knowledge_suggestion"] += 1
                continue

            first, second = dialog[index], dialog[index + 1]
            if (
                not isinstance(first, (list, tuple)) or len(first) < 2
                or not isinstance(second, (list, tuple)) or len(second) < 2
                or str(first[0]) != "0" or str(second[0]) != "1"
            ):
                stats["invalid_turn_order"] += 1
                continue

            user, assistant = first[1], second[1]
            if not acceptable_turn(user, assistant=False):
                stats["filtered_user_turn"] += 1
                continue
            if not acceptable_turn(assistant, assistant=True):
                stats["filtered_assistant_turn"] += 1
                continue

            key = (normalize(user), normalize(assistant))
            candidates.append((user.strip(), assistant.strip()))
            stats["passed_filters"] += 1

    random.Random(20261114).shuffle(candidates)
    return candidates, stats


def write_manifest(added: int, stats: Counter, archive_sha: str) -> None:
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
        "source_url": SOURCE_URL,
        "archive_sha256": archive_sha,
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
        description="Fetch, filter, deduplicate, and import licensed ASTRA conversation data."
    )
    parser.add_argument("--max-add", type=int, default=2500,
                        help="Maximum new examples to append (default: 2500).")
    parser.add_argument("--refresh", action="store_true",
                        help="Redownload and verify the source archive.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run filtering and deduplication without changing training files.")
    parser.add_argument("--sample-count", type=int, default=15,
                        help="Print a random quality sample after filtering (default: 15).")
    args = parser.parse_args()
    if args.max_add < 1 or args.sample_count < 0:
        parser.error("--max-add must be positive and --sample-count cannot be negative.")

    payload = download_archive(force=args.refresh)
    conversations = load_training_conversations(payload)
    candidates, stats = collect_candidates(conversations)

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
    print(f"Conversations scanned: {stats['conversations_seen']:,}")
    print(f"Pairs passing filters: {stats['passed_filters']:,}")
    print(f"New unique pairs: {len(additions):,}")
    print(f"Existing dataset pairs: {len(known):,}")
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
        print("\nDry run only. No files were changed.")
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
            output.write("\n\n---\n\n".join(blocks))
            output.write("\n")
    write_manifest(len(additions), stats, hashlib.sha256(payload).hexdigest())

    print(f"\nAdded {len(additions):,} filtered examples to dataset.txt.")
    print("Source and license attribution saved in import_manifest.json and dataset markers.")
    print("Next: run python3 prepare_dataset.py to rebuild and audit. No training was started.")


if __name__ == "__main__":
    main()
