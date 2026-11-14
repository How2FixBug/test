#!/usr/bin/env python3
"""Generate and manually review basic-conversation examples using local Ollama."""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "dataset.txt"
OLLAMA_URL = "http://localhost:11434/api/chat"
PAIR_RE = re.compile(
    r"(?m)^<user>\s*\n([\s\S]*?)\n</user>\s*\n"
    r"<astra>\s*\n([\s\S]*?)\n</astra>"
)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
UNSUITABLE_RE = re.compile(
    r"\b(?:porn|nude|suicide|self[- ]harm|cocaine|heroin|methamphetamine|"
    r"fuck(?:ing|ed|er)?|shit(?:ty)?|bitch(?:es)?|asshole)\b",
    re.IGNORECASE,
)
FACTUAL_RE = re.compile(
    r"\b(?:did you know|what is the (?:capital|population|definition)|"
    r"who (?:was|is|invented|discovered)|when (?:was|did|were)|"
    r"how many|what year|which country|explain (?:why|how)|"
    r"capital of|population of|physics|chemistry|biology|geography|"
    r"solar system|chemical reaction|mathematical equation)\b",
    re.IGNORECASE,
)
PERSONA_RE = re.compile(
    r"\b(?:i work as|i work at|my job is|i went to .{0,30} (?:last year|yesterday)|"
    r"i visited|my childhood|when i was a kid|my wife|my husband|"
    r"my son|my daughter|my brother|my sister|i served in the military)\b",
    re.IGNORECASE,
)
DIGITS_RE = re.compile(r"\d{4,}")


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def load_existing() -> tuple[str, set[tuple[str, str]]]:
    raw = DATASET_PATH.read_text(encoding="utf-8")
    pairs = {(normalize(user), normalize(answer)) for user, answer in PAIR_RE.findall(raw)}
    return raw, pairs


def request_examples(model: str, count: int, examples: list[tuple[str, str]]) -> list[dict]:
    examples_text = "\n".join(
        f"- User: {user[:180]}\n  ASTRA: {answer[:180]}"
        for user, answer in examples
    )
    prompt = f"""Create exactly {count} NEW training examples for ASTRA 0.2.

ASTRA's only job is basic, natural conversation. It is NOT a factual assistant.
Allowed: greetings, small talk, casual reactions, preferences, hobbies, everyday
plans, encouragement, listening, apologies, thanks, topic changes, and natural
follow-up replies that use short conversation history.

Hard rules:
- Do not include trivia, factual questions, explanations, school lessons, science,
  history, geography, politics, religion, finance, medical advice, or instructions.
- Do not make ASTRA claim a human biography, job, family, possessions, memories,
  offline activities, or real-world personal experiences.
- No roleplay involving weapons, crime, or dangerous activities.
- No song lyrics, quotes from media, URLs, email addresses, or personal data.
- Replies must directly fit the latest user message. Avoid generic filler.
- Vary the situations and wording. Do not create near-duplicates.
- Around one quarter may include 2-4 turns of short conversation history. For those,
  put the complete transcript in the user field and end it with
  "Reply naturally to the latest message."
- Keep ordinary single-turn user messages under 180 characters and replies under
  220 characters. Keep multi-turn prompts under 900 characters.
- Make each pair self-contained and believable. The assistant may be warm but must
  not pretend to be human.

Existing examples for style and duplicate avoidance:
{examples_text}

Return ONLY valid JSON in this exact shape:
{{"examples":[{{"user":"...","astra":"..."}}]}}
No markdown fences or commentary."""
    body = {
        "model": model,
        "stream": False,
        "format": "json",
        "think": False,
        "messages": [
            {"role": "system", "content": "You create tightly scoped conversational training data. Follow the JSON format exactly."},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.85, "top_p": 0.9, "num_predict": 4096},
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama at localhost:11434. Start Ollama and make sure "
            f"the '{model}' model is available. Details: {exc}"
        ) from exc
    message = result.get("message", {}).get("content", "")
    try:
        parsed = json.loads(message)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama did not return valid JSON: {message[:500]}") from exc
    items = parsed.get("examples")
    if not isinstance(items, list):
        raise RuntimeError("The model response did not contain an examples list.")
    return items


def flags_for(user: str, answer: str) -> list[str]:
    combined = user + "\n" + answer
    flags = []
    if not user.strip() or not answer.strip():
        flags.append("empty prompt or response")
    if len(user) > 900 or len(answer) > 360:
        flags.append("unusually long")
    if URL_RE.search(combined) or DIGITS_RE.search(combined):
        flags.append("URL or long number")
    if UNSUITABLE_RE.search(combined):
        flags.append("potentially unsuitable language/topic")
    if FACTUAL_RE.search(combined):
        flags.append("possible factual/knowledge content")
    if PERSONA_RE.search(answer):
        flags.append("possible invented human biography")
    if re.search(r"\b(?:according to|the capital of|is defined as|for example)\b", answer, re.I):
        flags.append("explanatory/factual-style response")
    if len(answer.split()) < 2:
        flags.append("very short response")
    return flags


def append_pair(user: str, answer: str) -> None:
    with DATASET_PATH.open("a", encoding="utf-8") as output:
        output.write("\n\n---\n\n")
        output.write(f"<user>\n{user.strip()}\n</user>\n<astra>\n{answer.strip()}\n</astra>\n")


def review_one(index: int, total: int, item: dict, known: set[tuple[str, str]]) -> bool:
    user = str(item.get("user", "")).strip()
    answer = str(item.get("astra", "")).strip()
    key = (normalize(user), normalize(answer))
    print("\n" + "=" * 68)
    print(f"CANDIDATE {index}/{total}")
    print("USER:\n" + (user or "[EMPTY]"))
    print("\nASTRA:\n" + (answer or "[EMPTY]"))
    flags = flags_for(user, answer)
    if key in known:
        flags.append("duplicate of an existing or already-reviewed example")
    if flags:
        print("\nAUTO-FLAGS: " + "; ".join(flags))
    else:
        print("\nAUTO-FLAGS: none detected (manual review is still required)")
    print("\n[y] approve  [n] reject  [e] edit, then review  [q] save and quit")
    while True:
        choice = input("> ").strip().lower()
        if choice == "y":
            if not user or not answer or key in known:
                print("Cannot approve an empty or duplicate pair.")
                continue
            append_pair(user, answer)
            known.add(key)
            return True
        if choice == "n":
            return False
        if choice == "e":
            user = input("Edited USER: ").strip()
            answer = input("Edited ASTRA: ").strip()
            item["user"], item["astra"] = user, answer
            return review_one(index, total, item, known)
        if choice == "q":
            raise KeyboardInterrupt
        print("Enter y, n, e, or q.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ASTRA conversation candidates with local Ollama and manually approve them."
    )
    parser.add_argument("--model", default="qwen2.5:7b", help="Local Ollama model (default: qwen2.5:7b).")
    parser.add_argument("--count", type=int, default=20, help="Number of candidates to generate (default: 20; max: 80).")
    parser.add_argument("--batch-size", type=int, default=2, help="Candidates per Ollama request (default: 2; max: 20).")
    args = parser.parse_args()
    if not 1 <= args.count <= 80:
        parser.error("--count must be between 1 and 80")
    if not 1 <= args.batch_size <= 20:
        parser.error("--batch-size must be between 1 and 20")

    raw, known = load_existing()
    before = len(known)
    print("ASTRA 0.2 CONVERSATION WORKBENCH")
    print(f"Training examples currently in dataset: {before:,}")
    print(f"Target milestone: 2,000 | Remaining: {max(0, 2000 - before):,}")
    print(f"Local generator: {args.model}")
    print("Generated candidates are NEVER added automatically. You approve each pair.")
    if before >= 2000:
        print("The 2,000-example milestone is already reached. You can still review more.")
    rng = random.Random()
    existing_examples = PAIR_RE.findall(raw)
    style_examples = rng.sample(existing_examples, min(24, len(existing_examples)))

    approved = 0
    generated = 0
    try:
        while generated < args.count:
            batch_count = min(args.batch_size, args.count - generated)
            print(f"\nGenerating {batch_count} candidates with local Ollama...", flush=True)
            batch = request_examples(args.model, batch_count, style_examples)
            if not batch:
                print("The model returned no candidates; stopping.")
                break
            for item in batch:
                if not isinstance(item, dict):
                    generated += 1
                    continue
                generated += 1
                if review_one(generated, args.count, item, known):
                    approved += 1
                if generated >= args.count:
                    break
    except KeyboardInterrupt:
        print("\nReview stopped. All previously approved examples were saved.")

    _, final_known = load_existing()
    print("\nWORKBENCH SUMMARY")
    print(f"Candidates reviewed: {generated:,}")
    print(f"New examples approved: {approved:,}")
    print(f"Training examples now: {len(final_known):,}")
    print(f"Remaining to 2,000: {max(0, 2000 - len(final_known)):,}")
    print("Next: run python3 prepare_dataset.py to rebuild the tokenizer and audit.")


if __name__ == "__main__":
    main()
