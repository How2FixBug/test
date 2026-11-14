#!/usr/bin/env python3
"""Fetch and heavily filter BYU Chit-Chat for ASTRA 0.2.

Only short, everyday conversational exchanges are retained. The source file is
pinned by its Git blob SHA-1, and imports remain removable via clean_dataset.py.
This is heuristic filtering, not a guarantee of perfect data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "dataset.txt"
CACHE_DIR = HERE / ".cache"
RAW_PATH = CACHE_DIR / "byu_chitchat_dataset.json"
MANIFEST_PATH = HERE / "import_manifest.json"

SOURCE_ID = "BYU-PCCL/chitchat-dataset"
SOURCE_URL = (
    "https://raw.githubusercontent.com/BYU-PCCL/chitchat-dataset/"
    "master/chitchat_dataset/dataset.json"
)
SOURCE_LICENSE = "MIT"
SOURCE_LICENSE_URL = "https://github.com/BYU-PCCL/chitchat-dataset/blob/master/LICENSE"
SOURCE_REPO_URL = "https://github.com/BYU-PCCL/chitchat-dataset"
# GitHub's contents API reports this exact file's Git blob SHA-1. Verify the
# Git object format, not merely the downloaded JSON syntax.
EXPECTED_GIT_BLOB_SHA1 = "9c01c26ec1af30cb2a868708e7be59cb4ee2e441"
SOURCE_MARKER = f"<!-- ASTRA_SOURCE: {SOURCE_ID} ({SOURCE_LICENSE}) -->"

PAIR_RE = re.compile(
    r"(?m)^<user>\\s*\\n([\\s\\S]*?)\\n</user>\\s*\\n"
    r"<astra>\\s*\\n(?:<think>[\\s\\S]*?</think>\\s*)?([\\s\\S]*?)\\n</astra>"
)
URL_RE = re.compile(r"(?:https?://|www\\.)\\S+", re.I)
EMAIL_RE = re.compile(r"\\b[\\w.+-]+@[\\w.-]+\\.[A-Za-z]{2,}\\b")
PHONE_RE = re.compile(r"(?<!\\w)(?:\\+?\\d[ .()/-]?){8,15}(?!\\w)")
LONG_NUMBER_RE = re.compile(r"\\b\\d{4,}\\b")
UNSAFE_RE = re.compile(
    r"\\b(?:"
    r"porn|nudes?|explicit sex|suicide|self[- ]harm|"
    r"cocaine|heroin|methamphetamine|"
    r"nigger|faggot|cunt|whore|slut|"
    r"fuck(?:ing|ed|er)?|shit(?:ty)?|bitch(?:es)?|asshole"
    r")\\b", re.I,
)
FACTUAL_RE = re.compile(
    r"\\b(?:"
    r"did you know|i learned that|what have you learned|"
    r"capital of|population of|president of|prime minister|"
    r"who (?:was|is|invented|discovered)|when (?:was|did|were)|"
    r"where (?:is|was|were) .{0,55}\\b(?:located|born|founded)|"
    r"how many|how much|what year|which country|"
    r"explain (?:why|how)|define |definition of|according to|"
    r"photosynthesis|quantum mechanics|chemical reaction|"
    r"solar system|historical event|machine learning|"
    r"computer science|programming language|neural network|"
    r"artificial intelligence|stock market|cryptocurrency|"
    r"physics|chemistry|biology|geography|"
    r"the process of|is defined as|in conclusion"
    r")\\b", re.I,
)
DIRECT_QUESTION_RE = re.compile(r"\\b(?:what|who|when|where|which|how)\\b.{0,90}\\?", re.I)
SOCIAL_QUESTION_RE = re.compile(
    r"\\b(?:how are you|how's it going|what are you up to|"
    r"what are you doing|what do you think|what do you like|"
    r"what's your favorite|what are your plans|where are you going|"
    r"what do you want to do|who are you with|when are you free|"
    r"how was your day|how have you been)\\b", re.I,
)
# Assistant-side human biography, relationships, possessions, identity,
# locations, memories, tastes, and offline activities are all excluded.
PERSONA_RE = re.compile(
    r"\\b(?:"
    r"i\\s+(?:am|was|have|had|live|lived|work|worked|study|studied|"
    r"own|owned|visited|went|grew up|used to|remember|recall|"
    r"love|loved|like|liked|hate|hated|prefer|preferred|enjoy|enjoyed|"
    r"adore|miss|made|cooked|bought|played|travelled|traveled|"
    r"am from|come from|have a|have an|have two|have three|"
    r"am dating|am married|am single|can relate|"
    r"my mom|my mother|my dad|my father|my family|my parents|"
    r"my brother|my sister|my friend|my best friend|my girlfriend|"
    r"my boyfriend|my wife|my husband|my kids|my children|my pet)|"
    r"i(?:'m| am)\\s+(?:a |an |from |at |in |on |with |tall|short|"
    r"single|married|dating|studying|working|living|getting in touch)|"
    r"my\\s+(?:name|job|wife|husband|girlfriend|boyfriend|children|kids|"
    r"brother|sister|parents|family|house|car|school|cat|dog|pet|mom|dad)\\b"
    r")", re.I,
)
# Keep only short conversational stance phrases when first person is used.
SAFE_FIRST_PERSON_RE = re.compile(
    r"^(?:i think so|i don't think so|i agree|i disagree|i guess so|"
    r"i'm not sure|i do(?:n't)? know|i understand|i see|i hope so|"
    r"i'm sorry|i apologize|i can imagine|i'd say so|i suppose so)"
    r"(?:[.!?, ]|$)", re.I,
)
SCRIPTED_RE = re.compile(
    r"\\b(?:as an ai language model|i cannot assist with that request|"
    r"i'll take that as a no|are you still there|"
    r"let me know if you need anything else|"
    r"when my page first loaded|"
    r"well, have a nice day|thank you for participating)\\b", re.I,
)
LOW_QUALITY_RE = re.compile(
    r"\\b(?:wre|teh|thier|becuase|definately|defiantly|recieve|"
    r"seperate|occured|belive|skiddish|phonenix)\\b", re.I,
)
MARKETING_RE = re.compile(
    r"\\b(?:subscribe|follow me|check out my channel|"
    r"visit my website|click the link|promo code|giveaway|"
    r"download my app|sign up now)\\b", re.I,
)
MAX_USER_CHARS = 180
MAX_ASSISTANT_CHARS = 170
MIN_WORDS = 2
MAX_WORDS = 30


def normalize(value: str) -> str:
    return re.sub(r"\\s+", " ", value).strip().casefold()


def looks_corrupted(value: str) -> bool:
    words = re.findall(r"[a-z']+", value.casefold())
    if len(words) >= 7:
        for i in range(len(words) - 3):
            if words[i] == words[i + 1] == words[i + 2] == words[i + 3]:
                return True
    if re.search(r"(.)\\1{5,}", value):
        return True
    if len(value) >= 14:
        useful = sum(ch.isalnum() or ch.isspace() for ch in value)
        if useful / len(value) < 0.65:
            return True
    if re.search(r"\\b(?:lol|haha|hehe)(?:\\s+\\1){3,}", value, re.I):
        return True
    return False


def acceptable_turn(value: object, *, assistant: bool) -> bool:
    if not isinstance(value, str):
        return False
    text = re.sub(r"\\s+", " ", value).strip()
    words = text.split()
    limit = MAX_ASSISTANT_CHARS if assistant else MAX_USER_CHARS
    if not (3 <= len(text) <= limit and MIN_WORDS <= len(words) <= MAX_WORDS):
        return False
    if URL_RE.search(text) or EMAIL_RE.search(text) or PHONE_RE.search(text):
        return False
    if LONG_NUMBER_RE.search(text) or UNSAFE_RE.search(text) or looks_corrupted(text):
        return False
    if FACTUAL_RE.search(text) or MARKETING_RE.search(text):
        return False
    if DIRECT_QUESTION_RE.search(text) and not SOCIAL_QUESTION_RE.search(text):
        return False
    if assistant:
        if PERSONA_RE.search(text) and not SAFE_FIRST_PERSON_RE.search(text):
            return False
        if SCRIPTED_RE.search(text) or LOW_QUALITY_RE.search(text):
            return False
        # Reject explanatory or multi-sentence answers; ASTRA is not a tutor.
        if len(re.findall(r"[.!?]", text)) > 2:
            return False
        if re.search(r"\\b(?:therefore|for example|in conclusion|"
                     r"scientifically|historically|firstly|secondly|"
                     r"the reason is|this means that)\\b", text, re.I):
            return False
    return True


def clean_turn(messages) -> tuple[str, str] | None:
    if not isinstance(messages, list) or not messages:
        return None
    senders = {
        str(item.get("sender", "")) for item in messages
        if isinstance(item, dict)
    }
    if len(senders) != 1 or not next(iter(senders), ""):
        return None
    pieces = []
    for item in messages:
        if not isinstance(item, dict):
            return None
        value = re.sub(r"\\s+", " ", str(item.get("text", ""))).strip()
        if value:
            pieces.append(value)
    text = " ".join(pieces).strip()
    if not text:
        return None
    return next(iter(senders)), text


def is_knowledge_question(text: str) -> bool:
    if FACTUAL_RE.search(text):
        return True
    if DIRECT_QUESTION_RE.search(text) and not SOCIAL_QUESTION_RE.search(text):
        return True
    return False


def download_dataset(force: bool = False) -> bytes:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists() and not force:
        payload = RAW_PATH.read_bytes()
    else:
        request = urllib.request.Request(
            SOURCE_URL,
            headers={"User-Agent": "ASTRA-0.2-chitchat-importer"},
        )
        print("Downloading BYU Chit-Chat dataset...", flush=True)
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read(80 * 1024 * 1024 + 1)
        if len(payload) > 80 * 1024 * 1024:
            raise ValueError("Dataset exceeded the 80 MB safety limit.")
        RAW_PATH.write_bytes(payload)

    git_blob = hashlib.sha1(
        b"blob " + str(len(payload)).encode("ascii") + b"\\0" + payload
    ).hexdigest()
    if git_blob != EXPECTED_GIT_BLOB_SHA1:
        RAW_PATH.unlink(missing_ok=True)
        raise ValueError(
            "Dataset Git blob SHA-1 did not match the pinned GitHub file. "
            "The download was discarded; review the source before updating the pin."
        )
    print(f"Verified source file: {len(payload):,} bytes, Git blob SHA-1 OK.")
    return payload


def load_conversations(payload: bytes) -> dict:
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict) or not (100 <= len(data) <= 20000):
        raise ValueError("Unexpected Chit-Chat conversation mapping.")
    return data


def existing_pairs() -> set[tuple[str, str]]:
    if not DATASET_PATH.exists():
        return set()
    raw = DATASET_PATH.read_text(encoding="utf-8")
    return {
        (normalize(user), normalize(answer))
        for user, answer in PAIR_RE.findall(raw)
    }


def collect_candidates(conversations: dict) -> tuple[list[tuple[str, str]], Counter]:
    stats = Counter()
    candidates: list[tuple[str, str]] = []

    for convo in conversations.values():
        stats["conversations_seen"] += 1
        if not isinstance(convo, dict):
            stats["invalid_conversations"] += 1
            continue
        prompt = str(convo.get("prompt", "")).strip()
        if prompt and (
            is_knowledge_question(prompt)
            or UNSAFE_RE.search(prompt)
            or URL_RE.search(prompt)
            or MARKETING_RE.search(prompt)
        ):
            stats["rejected_conversation_prompt"] += 1
            continue

        raw_turns = convo.get("messages")
        if not isinstance(raw_turns, list) or len(raw_turns) < 2:
            stats["invalid_conversations"] += 1
            continue
        turns = []
        invalid = False
        for raw_turn in raw_turns:
            turn = clean_turn(raw_turn)
            if turn is None:
                stats["invalid_turns"] += 1
                invalid = True
                break
            turns.append(turn)
        if invalid:
            stats["rejected_conversations"] += 1
            continue

        speakers = {speaker for speaker, _ in turns}
        if len(speakers) != 2:
            stats["rejected_non_two_speaker"] += 1
            continue
        first_speaker = turns[0][0]
        roles = ["user" if speaker == first_speaker else "assistant"
                 for speaker, _ in turns]
        if any(roles[i] == roles[i - 1] for i in range(1, len(roles))):
            stats["rejected_non_alternating"] += 1
            continue

        # A contaminated or knowledge-heavy turn rejects the whole conversation.
        if any(
            UNSAFE_RE.search(text) or URL_RE.search(text) or EMAIL_RE.search(text)
            or PHONE_RE.search(text) or MARKETING_RE.search(text)
            or FACTUAL_RE.search(text)
            for _, text in turns
        ):
            stats["rejected_contaminated_conversation"] += 1
            continue

        for index in range(1, len(turns), 2):
            user = turns[index - 1][1]
            assistant = turns[index][1]
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


def write_manifest(added: int, stats: Counter, source_sha1: str) -> None:
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
        "source_url": SOURCE_URL,
        "git_blob_sha1": source_sha1,
        "last_imported_utc": datetime.now(timezone.utc).isoformat(),
        "total_added": int(previous.get("total_added", 0)) + added,
        "last_run_added": added,
        "last_run_filter_counts": dict(stats),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heavily filter BYU Chit-Chat into basic ASTRA conversation data."
    )
    parser.add_argument("--max-add", type=int, default=2500)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()
    if args.max_add < 1 or args.sample_count < 0:
        parser.error("--max-add must be positive and --sample-count cannot be negative.")

    payload = download_dataset(force=args.refresh)
    conversations = load_conversations(payload)
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

    print("\\nASTRA PUBLIC DATASET IMPORT")
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
        print(f"\\nQUALITY SAMPLE ({len(sample)} examples)")
        for number, (user, assistant) in enumerate(sample, 1):
            print(f"\\n[{number}/{len(sample)}]")
            print(f"USER: {user}")
            print(f"ASTRA: {assistant}")

    if args.dry_run:
        print("\\nDry run only. No files were changed.")
        return

    if additions:
        blocks = [
            f"{SOURCE_MARKER}\\n<user>\\n{user}\\n</user>\\n"
            f"<astra>\\n{assistant}\\n</astra>"
            for user, assistant in additions
        ]
        with DATASET_PATH.open("a", encoding="utf-8") as output:
            if DATASET_PATH.stat().st_size:
                output.write("\\n\\n")
            output.write("\\n\\n---\\n\\n".join(blocks))
            output.write("\\n")
    write_manifest(len(additions), stats, EXPECTED_GIT_BLOB_SHA1)

    print(f"\\nAdded {len(additions):,} filtered examples to dataset.txt.")
    print("MIT source attribution saved in import_manifest.json and dataset markers.")
    print("Next: run python3 prepare_dataset.py to rebuild and audit. No training was started.")


if __name__ == "__main__":
    main()
