#!/usr/bin/env python3
"""Create a human-review queue for ASTRA examples without editing the dataset."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIR_RE = re.compile(
    r"(?m)^<user>\s*\n([\s\S]*?)\n</user>\s*\n"
    r"<astra>\s*\n(?:<think>[\s\S]*?</think>\s*)?([\s\S]*?)\n</astra>"
)
SOURCE_RE = re.compile(r"<!-- ASTRA_SOURCE: ([^>]+) -->")
# Multi-turn context is an intentional ASTRA input format, not a defect.
# Only flag claims that explicitly present time-sensitive information.
TIME_RE = re.compile(
    r"\b(?:as of (?:19|20)\d{2}|(?:current|latest|recent) "
    r"(?:population|price|rate|rankings|statistics|figures|data|version|"
    r"president|prime minister|CEO|weather|forecast|schedule|standings)|"
    r"today's (?:weather|news|forecast|schedule)|"
    r"right now,? (?:the|a) (?:price|rate|population|weather))\b",
    re.I,
)
SPECIFIC_RE = re.compile(
    r"(?:\b\d{4}\b|\b\d+(?:\.\d+)?\s*(?:%|percent|million|billion|"
    r"thousand|dollars?|miles?|kilometers?|years?|days?)\b|"
    r"[$€£]\s?\d)",
    re.I,
)
SOURCE_CLAIM_RE = re.compile(
    r"\b(?:according to|studies show|research (?:shows|indicates)|"
    r"experts say|it is widely believed|statistics show|data indicates)\b",
    re.I,
)
PLACEHOLDER_RE = re.compile(
    r"\[(?:insert|your|website|name|link|organization)[^\]]*\]|"
    r"@(?:BrandName|YourName|username)|\b(?:placeholder|lorem ipsum)\b",
    re.I,
)
PERSONA_RE = re.compile(
    r"\b(?:i remember when|when i was|i grew up|i was born|"
    r"i have (?:visited|lived|worked|studied)|i've (?:visited|lived|worked)|"
    r"my (?:wife|husband|girlfriend|boyfriend|son|daughter|childhood|family))\b",
    re.I,
)
CAPABILITY_RE = re.compile(
    r"\b(?:i don't have the capability|i cannot create|i can't create|"
    r"i don't have access to|outside my scope|i am unable to)\b",
    re.I,
)
GENERIC_RE = re.compile(r"^(?:enjoy!?|thanks!?|you're welcome!?|okay!?|sure!?)$", re.I)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def source_for(raw: str, position: int) -> str:
    marker = None
    for match in SOURCE_RE.finditer(raw, 0, position):
        marker = match.group(1).strip()
    return marker or "original/unspecified"


def inspect(prompt: str, answer: str, prompt_counts: Counter, answer_counts: Counter) -> list[str]:
    flags: list[str] = []
    if TIME_RE.search(prompt) or TIME_RE.search(answer):
        flags.append("time_sensitive_claim")
    if SPECIFIC_RE.search(answer):
        flags.append("specific_numbers_or_dates")
    if SOURCE_CLAIM_RE.search(answer):
        flags.append("unsupported_source_or_research_claim")
    if PLACEHOLDER_RE.search(prompt) or PLACEHOLDER_RE.search(answer):
        flags.append("template_placeholder")
    if PERSONA_RE.search(answer):
        flags.append("possible_false_personal_experience")
    if CAPABILITY_RE.search(answer):
        flags.append("capability_claim_check")
    if len(answer) > 1000 or len(answer.split()) > 170:
        flags.append("possibly_overlong_answer")
    if GENERIC_RE.fullmatch(norm(answer)):
        flags.append("very_generic_answer")
    if prompt_counts[norm(prompt)] > 1:
        flags.append("repeated_prompt")
    if answer_counts[norm(answer)] > 1:
        flags.append("repeated_answer")
    if "\\n" in prompt or "\\n" in answer:
        flags.append("literal_backslash_newline")
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flag ASTRA dataset examples for human review; never edits dataset.txt."
    )
    parser.add_argument("--dataset", type=Path, default=HERE / "dataset.txt")
    parser.add_argument("--output", type=Path, default=HERE / "quality_review.jsonl")
    parser.add_argument("--priority-output", type=Path, default=HERE / "quality_review_priority.jsonl",
                        help="Separate queue for placeholders, unsupported source claims, and capability claims.")
    parser.add_argument("--show", type=int, default=12,
                        help="Print up to this many flagged examples (default: 12; 0 disables).")
    args = parser.parse_args()
    if args.show < 0:
        parser.error("--show cannot be negative.")

    raw = args.dataset.read_text(encoding="utf-8")
    first = re.search(r"(?m)^<user>\s*$", raw)
    if not first:
        raise SystemExit("No ASTRA examples found.")
    matches = list(PAIR_RE.finditer(raw[first.start():]))
    prompts = [m.group(1).strip() for m in matches]
    answers = [m.group(2).strip() for m in matches]
    prompt_counts = Counter(map(norm, prompts))
    answer_counts = Counter(map(norm, answers))

    reviewed = []
    totals: Counter[str] = Counter()
    for index, (match, prompt, answer) in enumerate(zip(matches, prompts, answers), 1):
        flags = inspect(prompt, answer, prompt_counts, answer_counts)
        if not flags:
            continue
        absolute_position = first.start() + match.start()
        record = {
            "example": index,
            "source": source_for(raw, absolute_position),
            "flags": flags,
            "prompt": prompt,
            "answer": answer,
        }
        reviewed.append(record)
        totals.update(flags)

    priority_flags = {
        "template_placeholder",
        "unsupported_source_or_research_claim",
        "capability_claim_check",
    }
    priority = [record for record in reviewed if priority_flags.intersection(record["flags"])]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in reviewed:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    args.priority_output.parent.mkdir(parents=True, exist_ok=True)
    with args.priority_output.open("w", encoding="utf-8") as stream:
        for record in priority:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("ASTRA HUMAN-REVIEW QUEUE")
    print(f"Dataset examples scanned: {len(matches):,}")
    print(f"Examples flagged: {len(reviewed):,}")
    print(f"Examples not flagged by these heuristics: {len(matches) - len(reviewed):,}")
    print(f"Review file: {args.output}")
    print("\nFlag counts (one example can have multiple flags):")
    for flag, count in totals.most_common():
        print(f"  {flag}: {count:,}")

    print(f"\nPriority review examples: {len(priority):,}")
    print(f"Priority review file: {args.priority_output}")
    if args.show and priority:
        print(f"\nFIRST {min(args.show, len(priority))} PRIORITY EXAMPLES")
        for record in priority[:args.show]:
            print(f"\n[{record['example']}] {', '.join(record['flags'])}")
            print(f"SOURCE: {record['source']}")
            print(f"USER: {record['prompt'][:500].replace(chr(10), ' ')}")
            print(f"ASTRA: {record['answer'][:700].replace(chr(10), ' ')}")

    print("\nReview queues only. dataset.txt, tokenizer files, and model weights were not changed.")
    print("Flags are heuristics, not proof of an error. Unflagged examples are not certified as correct.")


if __name__ == "__main__":
    main()
