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

## Requirements

- Python 3.10+
- PyTorch 2.x
- No additional Python packages

For Apple Silicon, the trainer automatically selects Apple's MPS backend when available.
Otherwise it falls back to CPU.

## Prepare

After pulling the latest code, regenerate the tokenizer artifacts:

```bash
python3 tokenizer.py
```

This produces a BPE `tokenizer_config.json` containing `vocab_hex` and `merges`,
plus a matching `tokenized_dataset.jsonl`. The currently committed generated
artifacts may be from the older 262-token byte tokenizer, so regenerate them before
training. Keep the two generated files together; they must come from the same run.

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
- Samples each batch row from one conversation, never joining unrelated conversations.
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

Generation uses a fixed seed by default for reproducible comparisons. Change it with
`--seed`. It also constrains the structural markers so the reasoning-summary section
closes before the answer and the assistant message closes at the end.

This remains an experimental first model, not a capable assistant yet. With only 280
training examples, it will likely overfit and produce weak or repetitive text. A low
training loss alone does not mean it has learned general language ability. Evaluate
held-out examples and output quality before making claims about capability.
