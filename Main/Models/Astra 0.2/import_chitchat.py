#!/usr/bin/env python3
"""Import filtered, everyday dialogue from BYU's public Chit-Chat dataset.

The raw corpus is downloaded locally and cached. Only two-speaker, alternating,
short conversational turns are converted into ASTRA's user/assistant format.
This is a heuristic filter, so review the audit before training.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "dataset.txt"
CACHE_DIR = HERE / ".cache"
RAW_PATH = CACHE_DIR / "byu_chitchat_dataset.json"
SOURCE_URL = (
    "https://raw.githubusercontent.com/BYU-PCCL/chitchat-dataset/"
    "master/chitchat_dataset/dataset.json"
)
SOURCE_MARKER = "<!-- ASTRA_SOURCE: BYU-PCCL/chitchat-dataset -->"
PAIR_RE = re.compile(
    r"(?m)^<user>\s*\n([\s\S]*?)\n</user>\s*\n"
    r"<astra>\s*\n([\s\S]*?)\n</astra>"
)
FACTUAL_RE = re.compile(
    r"\b(?:"
    r"did you know|i learned that|what have you learned|"
    r"what is the (?:capital|population|definition)|"
    r"who (?:was|is|invented|discovered)|when (?:was|did|were)|"
    r"where (?:is|was|were) .{0,50}\b(?:located|born|founded)|"
    r"how many|how much|what year|which country|"
    r"explain (?:why|how)|define |capital of|population of|"
    r"was invented by|was discovered by|was founded in|"
    r"president of|prime minister of|"
    r"physics|chemistry|biology|geography|historical event|"
    r"solar system|chemical reaction|mathematical equation"
    r")\b",
    re.IGNORECASE,
)
FACTUAL_RESPONSE_RE = re.compile(
    r"\b(?:according to|the capital of|was born in|"
    r"was invented by|was discovered by|was founded in|"
    r"is located in|the population is|in the year \d{3,4}|"
    r"known as the .{0,30} of)\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
DIGIT_RE = re.compile(r"\d{4,}")
# Conservative content exclusions for a youth-oriented conversational corpus.
UNSUITABLE_RE = re.compile(
    r"\b(?:"
    r"fuck(?:ing|ed|er)?|shit(?:ty)?|bitch(?:es)?|asshole|"
    r"nigger|faggot|cunt|whore|slut|"
    r"porn|nude(?:s)?|sexually explicit|"
    r"suicide|self[- ]harm|"
    r"cocaine|heroin|methamphetamine"
    r")\b",
    re.IGNORECASE,
)
CONVERSATIONAL_WHAT_RE = re.compile(
    r"\bwhat (?:do you|did you|are you|have you|would you|"
    r"kind of|sort of|about you|do we|should we)\b",
    re.IGNORECASE,
)
MAX_TURN_CHARS = 360
MAX_CONTEXT_TURNS = 6
MAX_EXAMPLES = 10000


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def clean_turn(messages) -> tuple[str, str] | None:
    if not isinstance(messages, list) or not messages:
        return None
    senders = {str(item.get("sender", "")) for item in messages if isinstance(item, dict)}
    if len(senders) != 1 or not next(iter(senders), ""):
        return None
    pieces = []
    for item in messages:
        if not isinstance(item, dict):
            return None
        value = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
        if value:
            pieces.append(value)
    text = " ".join(pieces).strip()
    if len(text) < 2 or len(text) > MAX_TURN_CHARS:
        return None
    if URL_RE.search(text) or EMAIL_RE.search(text) or DIGIT_RE.search(text):
        return None
    if UNSUITABLE_RE.search(text):
        return None
    return next(iter(senders)), text


def is_knowledge_question(text: str) -> bool:
    lowered = text.casefold()
    if CONVERSATIONAL_WHAT_RE.search(lowered):
        return False
    if FACTUAL_RE.search(lowered):
        return True
    # Reject direct factual question structures while retaining ordinary
    # social questions such as "how are you?" and "what are you doing?"
    if re.search(
        r"\b(?:what|who|when|where|which|how)\b.{0,70}\?",
        text,
        re.IGNORECASE,
    ):
        if re.search(
            r"\b(?:how are you|how's it going|what are you doing|"
            r"what do you think|what do you like|what's your favorite|"
            r"who are you with|where are you going|when are you free|"
            r"what are your plans|what are you up to|what do you want to do)\b",
            lowered,
        ):
            return False
        return True
    return False


def looks_factual_response(text: str) -> bool:
    if FACTUAL_RESPONSE_RE.search(text):
        return True
    # Long explanatory turns are poor fits for this small conversational model.
    if len(text) > 220 and re.search(
        r"\b(?:therefore|for example|in conclusion|scientifically|"
        r"historically|the process of|is defined as)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    return False


def download_dataset(force: bool = False) -> Path:
    if RAW_PATH.exists() and RAW_PATH.stat().st_size > 1000 and not force:
        print(f"Using cached source: {RAW_PATH}")
        return RAW_PATH
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "ASTRA-0.2-conversation-dataset-importer"},
    )
    print("Downloading BYU Chit-Chat dataset...")
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(80 * 1024 * 1024 + 1)
    if len(payload) > 80 * 1024 * 1024:
        raise ValueError("Source file exceeded the 80 MB safety limit.")
    parsed = json.loads(payload.decode("utf-8"))
    if not isinstance(parsed, dict) or len(parsed) < 100:
        raise ValueError("Downloaded source did not match the expected conversation mapping.")
    RAW_PATH.write_bytes(payload)
    print(f"Cached {len(payload):,} bytes at {RAW_PATH}")
    return RAW_PATH


def existing_pairs(raw: str) -> set[tuple[str, str]]:
    return {
        (normalize(user), normalize(answer))
        for user, answer in PAIR_RE.findall(raw)
    }


def make_examples(conversations: dict) -> tuple[list[tuple[str, str]], dict[str, int]]:
    candidates: list[tuple[str, str]] = []
    stats = {
        "conversations_seen": 0,
        "conversations_rejected": 0,
        "turns_rejected": 0,
        "pairs_rejected": 0,
        "pairs_kept_before_dedup": 0,
    }
    for convo in conversations.values():
        stats["conversations_seen"] += 1
        if not isinstance(convo, dict):
            stats["conversations_rejected"] += 1
            continue
        prompt = str(convo.get("prompt", "")).strip()
        if prompt and (is_knowledge_question(prompt) or UNSUITABLE_RE.search(prompt)):
            stats["conversations_rejected"] += 1
            continue
        raw_turns = convo.get("messages")
        if not isinstance(raw_turns, list) or len(raw_turns) < 2:
            stats["conversations_rejected"] += 1
            continue

        turns = []
        invalid = False
        for raw_turn in raw_turns:
            turn = clean_turn(raw_turn)
            if turn is None:
                stats["turns_rejected"] += 1
                invalid = True
                break
            turns.append(turn)
        if invalid:
            stats["conversations_rejected"] += 1
            continue

        speaker_ids = {speaker for speaker, _ in turns}
        if len(speaker_ids) != 2:
            stats["conversations_rejected"] += 1
            continue
        first_speaker = turns[0][0]
        roles = ["User" if speaker == first_speaker else "ASTRA" for speaker, _ in turns]
        if any(roles[i] == roles[i - 1] for i in range(1, len(roles))):
            stats["conversations_rejected"] += 1
            continue

        # Every odd turn is the second participant responding to the first.
        # Keep up to six prior turns so later samples teach conversational flow.
        for response_index in range(1, len(turns), 2):
            history_start = max(0, response_index - MAX_CONTEXT_TURNS)
            # Keep context aligned to a User turn. If the six-turn window begins
            # on the other speaker, drop that leading turn rather than teaching
            # ASTRA prompts that begin with an assistant message.
            if history_start % 2 == 1:
                history_start += 1
            history = turns[history_start:response_index]
            target = turns[response_index][1]
            latest_user = history[-1][1]
            if is_knowledge_question(latest_user) or looks_factual_response(target):
                stats["pairs_rejected"] += 1
                continue
            if any(is_knowledge_question(text) for _, text in history):
                stats["pairs_rejected"] += 1
                continue

            if response_index == 1:
                user_input = latest_user
            else:
                transcript = "\n".join(
                    f"{role}: {text}" for role, (_, text) in zip(roles[response_index - len(history):response_index], history)
                )
                user_input = (
                    "Conversation so far:\n"
                    + transcript
                    + "\n\nReply naturally to the latest message."
                )
            if len(user_input) > 1800:
                stats["pairs_rejected"] += 1
                continue
            candidates.append((user_input, target))
            stats["pairs_kept_before_dedup"] += 1

    # Stable shuffle prevents early conversations from dominating the cap.
    random.Random(20261114).shuffle(candidates)
    return candidates, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Import conversation-only examples from BYU Chit-Chat.")
    parser.add_argument("--refresh", action="store_true", help="Download a fresh copy of the source dataset.")
    parser.add_argument("--max-examples", type=int, default=MAX_EXAMPLES)
    args = parser.parse_args()
    if args.max_examples < 1:
        parser.error("--max-examples must be at least 1")

    raw_path = download_dataset(force=args.refresh)
    conversations = json.loads(raw_path.read_text(encoding="utf-8"))
    candidates, stats = make_examples(conversations)

    raw_dataset = DATASET_PATH.read_text(encoding="utf-8")
    # Rebuild this source section on every import so improved filters replace
    # old imported examples instead of leaving stale low-quality records behind.
    source_block_re = re.compile(
        r"(?ms)^<!-- ASTRA_SOURCE: BYU-PCCL/chitchat-dataset(?: \(MIT\))? -->\n"
        r"<user>\n.*?\n</astra>(?:\n\s*---)?\n*"
    )
    raw_dataset = source_block_re.sub("", raw_dataset)
    raw_dataset = re.sub(r"(?m)^TOTAL:.*\n?", "", raw_dataset, count=1)
    known = existing_pairs(raw_dataset)
    additions = []
    for user, answer in candidates:
        key = (normalize(user), normalize(answer))
        if key in known:
            continue
        known.add(key)
        additions.append((user, answer))
        if len(additions) >= args.max_examples:
            break

    if additions:
        block = "\n\n---\n\n".join(
            f"{SOURCE_MARKER}\n<user>\n{user}\n</user>\n<astra>\n{answer}\n</astra>"
            for user, answer in additions
        )
        updated = raw_dataset.rstrip() + "\n\n---\n\n" + block + "\n"
        total = len(existing_pairs(updated))
        original = len(existing_pairs(raw_dataset))
        updated = re.sub(
            r"(?m)^TOTAL:.*$",
            f"TOTAL: {total:,} unique examples ({original:,} existing + "
            f"{total - original:,} BYU Chit-Chat additions).",
            updated,
            count=1,
        )
        DATASET_PATH.write_text(updated, encoding="utf-8")
    else:
        total = len(known)

    imported_total = len(re.findall(re.escape(SOURCE_MARKER), DATASET_PATH.read_text(encoding="utf-8")))
    print("\nBYU CHIT-CHAT IMPORT")
    print(f"Source conversations: {len(conversations):,}")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value:,}")
    print(f"New unique examples added: {len(additions):,}")
    print(f"BYU examples already in dataset: {imported_total:,}")
    print(f"Total unique ASTRA examples: {total:,}")
    if total < 5000:
        print(f"Still need {5000 - total:,} examples to reach the 5,000-example milestone.")
    else:
        print("5,000-example milestone reached.")
    print("\nReview dataset.txt and audit_dataset.py output before training.")


if __name__ == "__main__":
    main()
