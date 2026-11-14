# ASTRA 0.2 — Basic Conversation

## Goal

ASTRA 0.2 is a small chatbot focused exclusively on basic, natural conversation.
It should handle greetings, small talk, casual reactions, preferences, simple
back-and-forth exchanges, and conversational follow-ups.

**It is not a factual assistant.** Do not add trivia, science, history, geography,
encyclopedic answers, general knowledge, or instruction-following datasets.
The goal is conversational behavior, not answering factual questions.

## Dataset

The target is **5,000+ reviewed conversation examples**.

- `dataset.txt` is the training source and should contain only reviewed examples.
- The original curated seed examples are retained.
- BYU Chit-Chat is no longer imported into the training dataset automatically.
- `import_chitchat.py` creates a separate `byu_review_candidates.txt` file.
  Those candidates are not training data and must be reviewed before being copied
  into `dataset.txt`.
- `prepare_dataset.py` automatically removes any BYU source-marked blocks left
  in `dataset.txt`, then rebuilds the tokenizer, audits the corpus, and runs checks.
- The importer uses heuristic filters; a candidate file is not a guarantee of
  conversational quality. Review for factual discussion, invented assistant
  personal experiences, irrelevant replies, scripted dialogue, and off-topic content.

Prefer:
- Greetings and goodbyes.
- Check-ins and small talk.
- Casual reactions and acknowledgements.
- Preferences, hobbies, and everyday plans.
- Listening, empathy, apologies, thanks, and topic changes.
- Follow-up replies that depend on earlier conversational context.

Avoid repetitive paraphrase spam. Prefer varied, believable exchanges. ASTRA's
assistant responses should not invent a human biography, personal history, or
real-world experiences.

## Prepare

Run the full preparation pipeline with one command:

```bash
python3 prepare_dataset.py
```

This removes unreviewed BYU imports from the training file, rebuilds the tokenizer,
audits the curated corpus, and runs correctness checks. It does not train ASTRA.

To generate a separate BYU review-candidate file:

```bash
python3 import_chitchat.py
```

The raw BYU source is cached in `.cache/`; refresh it with:

```bash
python3 import_chitchat.py --refresh
```

The source is [BYU-PCCL/chitchat-dataset](https://github.com/BYU-PCCL/chitchat-dataset).
It lists an MIT license and should be credited if any reviewed examples are retained.
The source is mixed open-domain conversation, so filtering is heuristic.

## Train

```bash
python3 train.py
```

Training starts from random initialization and uses ASTRA 0.2's own checkpoint.
Do not train until the reviewed dataset is large and its audit samples are acceptable.

## Generate

```bash
python3 generate.py "Hey! How's it going?"
```

## Status

- [x] Separate ASTRA 0.2 copy created from 0.1.
- [x] Replaced inherited knowledge-heavy seed data with conversation-only examples.
- [x] Separated external BYU candidates from the actual training dataset.
- [x] Added automatic removal of unreviewed source-marked imports.
- [ ] Reach and review at least 5,000 unique examples.
- [ ] Improve training to represent longer conversational histories directly.
- [ ] Train and evaluate ASTRA 0.2.
