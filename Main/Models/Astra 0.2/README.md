# ASTRA 0.2 — Basic Conversation

ASTRA 0.2 is a separate development copy of ASTRA 0.1. The first milestone is
basic, coherent conversation: natural greetings, small talk, relevant answers,
simple instructions, explanations, and follow-up context.

## Starting point

This directory was cloned from the ASTRA 0.1 code and its 280-example starter
corpus. It has its own tokenizer artifacts, tokenized dataset, checkpoints, and
training outputs. It does not reuse the ASTRA 0.1 checkpoint.

The initial model architecture and training pipeline are intentionally kept
unchanged while the 0.2 data pipeline is prepared. The next dataset milestone
is a reviewed corpus of roughly 8,000 high-quality examples, with a stronger
focus on conversational variety and multi-turn context.

## Dataset

The starter `dataset.txt` is the original 280-example seed corpus copied from
0.1. `import_datasets.py` can add filtered English examples from OpenAssistant
(OASST1) and Aya, with source credits and exact-pair deduplication. Inspect the
corpus after importing; the importer is a filter, not a guarantee of quality.

Run:

```bash
python3 import_datasets.py --limit-per-source 4000
python3 audit_dataset.py
```

Then regenerate the tokenizer and encoded dataset:

```bash
python3 tokenizer.py
python3 test_causality.py
```

**Important:** the current cloned trainer still treats each record as one
user/assistant exchange. Multi-turn conversation records and assistant-turn
masking are a separate required data/training upgrade before ASTRA 0.2 can
properly learn multi-turn dialogue. Do not train on concatenated turns until
that support is implemented.

## Train

```bash
python3 train.py
```

Optional quick smoke run:

```bash
python3 train.py --steps 100 --batch-size 4 --block-size 128
```

Training starts from random initialization. It does not load the ASTRA 0.1
checkpoint.

## Generate

```bash
python3 generate.py "Hello! What can you do?"
```

## Current status

- [x] Separate ASTRA 0.2 directory created from the 0.1 baseline.
- [x] Independent checkpoint naming.
- [x] Starter dataset copied.
- [ ] Expand and review the 0.2 corpus.
- [ ] Add multi-turn conversation encoding and assistant-only loss masking.
- [ ] Train and evaluate ASTRA 0.2.
