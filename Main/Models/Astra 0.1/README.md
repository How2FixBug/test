# ASTRA 0.1 — LoopCore

ASTRA is a small autoregressive language model trained from random initialization.
It is not a wrapper around another model and does not use pretrained weights.

## Architecture

- **LoopCore:** one shared neural block is reused for four recurrent passes.
- **Local causal attention:** each token can attend only to earlier/current tokens
  within a 128-token sliding window.
- **Hybrid recurrent memory:** a gated sequence-level memory is updated on each pass.
- **Loop controller:** conditions each pass and controls how strongly its update is applied.
- **Tied embeddings:** input embeddings are reused as the output vocabulary projection.
- **Next-token objective:** the model learns to predict the next token across the full
  conversation stream, including the user, ASTRA, and think markers.

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

Make sure you have already run:

```bash
python3 tokenizer.py
```

This must produce a BPE `tokenizer_config.json` containing `vocab_hex` and `merges`,
plus `tokenized_dataset.jsonl`. The trainer checks that token IDs fit the vocabulary.

## Train

From this directory:

```bash
python3 train.py
```

Optional smaller/faster first run:

```bash
python3 train.py --steps 100 --batch-size 4 --block-size 128
```

The best validation checkpoint is saved to `checkpoints/astra-0.1-best.pt`.
Training starts from random weights every run; this initial version does not resume
optimizer state from an interrupted run.

## Generate

After training:

```bash
python3 generate.py "Why does ice float?"
```

This is an experimental first model, not a capable assistant yet. With 280 training
examples, it will likely overfit and produce weak or repetitive text. A low training
loss alone does not mean it has learned general language ability. Evaluate outputs,
validation loss, and held-out examples before making claims about quality.
