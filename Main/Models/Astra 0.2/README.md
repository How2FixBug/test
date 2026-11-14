# ASTRA 0.2: Basic Conversation

## Goal

ASTRA 0.2 is a small chatbot focused exclusively on basic, natural conversation.
It should handle greetings, small talk, casual reactions, preferences, simple
back-and-forth exchanges, and conversational follow-ups.

**It is not a factual assistant.** Do not add trivia, science, history, geography,
encyclopedic answers, general knowledge, or instruction-following datasets.
The goal is conversational behavior, not answering factual questions.

## Dataset

The target is **5,000+ quality-filtered conversation examples**.

- `dataset.txt` is the training source.
- The original 531 curated seed examples remain untouched by the importer.
- `fetch_and_filter.py` uses BYU's Chit-Chat dataset, distributed in the
  original repository under the **MIT License**.
- The raw JSON is pinned to a specific Git blob SHA-1. If the upstream file
  changes, the importer refuses it until the source is reviewed and the pin is
  deliberately updated.
- Heavy filters reject factual/knowledge topics, unsafe language, personal
  contact details, URLs, marketing, corrupted text, poor-quality spelling,
  long answers, and assistant-side human biography or personal experiences.
- Only two-speaker, strictly alternating conversations are considered. A
  knowledge-heavy or contaminated turn rejects the whole conversation.
- Imports are deduplicated against the existing dataset and tagged with source
  attribution. `import_manifest.json` records the source, license, checksum,
  and filter counts.
- These are heuristic filters, not a guarantee that every retained example is
  perfect. Inspect the dry-run samples and run the full audit before training.

Prefer:
- Greetings and goodbyes.
- Check-ins and small talk.
- Casual reactions and acknowledgements.
- Preferences, hobbies, and everyday plans expressed by the user.
- Listening, empathy, apologies, thanks, and topic changes.
- Short, natural follow-up replies.

Avoid repetitive paraphrase spam. ASTRA's assistant responses must not invent
a human biography, personal history, or real-world experiences.

## Fetch and filter public data

Start with a dry run:

```bash
python3 fetch_and_filter.py --dry-run
```

If the samples and counts look good, import with:

```bash
python3 fetch_and_filter.py
```

The importer downloads and verifies the pinned source file, filters it, removes
duplicates against the existing dataset, and appends up to 2,500 new examples.
The source file is cached in `.cache/`. Use `--refresh` to download and verify
it again.

Useful options:

```bash
python3 fetch_and_filter.py --dry-run --sample-count 30
python3 fetch_and_filter.py --max-add 1000
python3 fetch_and_filter.py --refresh --dry-run
```

The importer uses only Python's standard library. It does not install packages,
call Ollama, or train the model.

Source:
- [BYU Chit-Chat dataset](https://github.com/BYU-PCCL/chitchat-dataset)
- [Original repository license](https://github.com/BYU-PCCL/chitchat-dataset/blob/master/LICENSE)
- [Chit-Chat Challenge research paper](https://www.scitepress.org/PublishedPapers/2020/89399/pdf/index.html)

Other public datasets must be checked individually for conversational fit and
reuse permissions before being added. DailyDialog and EmpatheticDialogues are
not included because their noncommercial license terms are not a good default
for a project whose future distribution may change.

## Prepare

Run the full preparation pipeline with one command:

```bash
python3 prepare_dataset.py
```

This removes only legacy unreviewed candidate blocks, rebuilds the tokenizer,
audits the corpus, and runs correctness checks. Verified MIT imports are
preserved. It does not train ASTRA.

## Generate and review new examples

Use the local Ollama model to draft a small batch when additional hand-written
examples are useful:

```bash
python3 conversation_workbench.py
```

The workbench generates 20 candidates per run. It shows each prompt/response
pair with automatic warning flags. Manually review only when you choose to use
this optional path. The public dataset importer does not require approving each
pair one by one.

## Train

```bash
python3 train.py
```

Training starts from random initialization and uses ASTRA 0.2's own checkpoint.
Do not train until the dataset is large and its audit samples are acceptable.

## Generate

```bash
python3 generate.py "Hey! How's it going?"
```
