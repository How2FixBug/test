# ASTRA 0.1 — LoopCore

ASTRA is a small autoregressive language model trained from random initialization.
It is not a wrapper around another model and does not use pretrained weights.

## Architecture

- **LoopCore:** one shared neural block is reused for four recurrent passes.
- **Local causal attention:** each token can attend only to earlier/current tokens
  within a 128-token sliding window.
- **Causal recurrent memory:** each token has its own gated memory, updated from a
  prefix summary containing only that token and earlier tokens. No future-token pooling.
- **Causal loop controller:** computes the update strength separately at each position
  from that position's causal prefix.
- **Tied embeddings:** input embeddings are reused as the output vocabulary projection.
- **Next-token objective:** the model predicts the next token, including conversation
  structure markers.

The initial configuration uses width 512, 8 attention heads, an MLP width of 1792,
four shared passes, and a maximum context of 512 tokens. The shared block is reused;
there are not four separate transformer blocks.

## Dataset

The starter corpus includes original ASTRA examples. The importer can add up to 15,000 examples per source (subject to availability and filtering). Run:

```bash
python3 import_datasets.py
python3 tokenizer.py
python3 test_causality.py
```

The importer adds English examples from OASST1 and Aya, deduplicates prompt/answer pairs, and uses empty `<think>` sections for imported records. It uses only the Python standard library. Source credits and license notes are in the repository-root [CREDITS.md](../../../CREDITS.md). The default import limit is 5,000 examples per source; inspect the imported data before training.

The starter set also contains original ASTRA examples across science, computing, writing, creative design, and project skills. During tokenization, all think sections are normalized to empty so the model is trained to answer directly rather than imitate inconsistent reasoning notes. More data is not automatically better: quality, licensing, deduplication, and held-out evaluation still matter.

## Requirements

- Python 3.10+
- PyTorch 2.x
- No additional Python packages

For Apple Silicon, the trainer automatically selects Apple's MPS backend when available.
Otherwise it falls back to CPU.

## Audit the dataset

Before spending time on another training run, inspect the actual local corpus and token distribution:

```bash
python3 audit_dataset.py
```

This reports unique prompts and answers, repeated-answer clusters, empty think sections, sequence lengths, and the most frequent token pieces. It does not modify the dataset.

## Prepare

After pulling the latest code, regenerate the tokenizer artifacts:

```bash
python3 tokenizer.py
```

This produces a BPE `tokenizer_config.json` containing `vocab_hex` and `merges`,
plus a matching `tokenized_dataset.jsonl`. Both files carry a shared fingerprint;
the trainer rejects mismatched or outdated artifacts. The currently committed
generated artifacts may be from the older 262-token byte tokenizer, so regenerate
them before training. Keep the two generated files together; they must come from
the same run.

Run the correctness checks:

```bash
python3 test_causality.py
```

This checks tokenizer round-tripping (including Unicode) and verifies that changing
future input tokens cannot change earlier-position model outputs.

## Train

From this directory:

```bash
python3 train.py
```

Optional smaller/faster first run:

```bash
python3 train.py --steps 100 --batch-size 4 --block-size 128
```

The trainer:
- Splits exact duplicate ASTRA responses as groups, preventing those answers from
  appearing in both training and validation.
- Keeps each user prompt at the start of its training sequence instead of randomly
  cropping away the question.
- Uses the user prompt as context but calculates loss only on assistant reply tokens.
- Uses padding masks for shorter examples.
- Saves a versioned best checkpoint to `checkpoints/astra-0.1-best.pt`.

Training starts from random weights every run; this initial version does not resume
optimizer state from an interrupted run. The revised causal-memory architecture
requires a fresh training run; old checkpoints are rejected by the generator.

## Generate

After training:

```bash
python3 generate.py "Why does ice float?"
```

Generation streams answer text to the terminal as tokens arrive. Use `--no-stream` to
print the completed answer instead. Generation uses a fixed seed by default for
reproducible comparisons. Change it with `--seed`. The think section is normalized to
empty during tokenization, so ASTRA learns to produce direct answers.

This remains an experimental first model, not a capable assistant yet. With a small or repetitive training corpus, it will likely overfit and produce weak or repetitive text. Increase data variety and inspect validation behavior before interpreting training loss. A low
training loss alone does not mean it has learned general language ability. Evaluate
held-out examples and output quality before making claims about capability.
