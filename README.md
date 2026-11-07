# littleGPT

A tiny decoder-only Transformer trained **from random initialization** on a GitHub Actions CPU runner, then executed entirely in the browser.

## Run it

1. Open the **Actions** tab in this repository.
2. Select **Train littleGPT** → **Run workflow**.
3. Choose 200, 500, or 1,000 training steps.
4. When the workflow finishes, it commits the learned `model.json` weights to `main`.
5. Open the GitHub Pages site and press **Check for updated weights**.

The site uses those exported weights directly. Inference is JavaScript running in your browser; no API, server, Ollama, or pretrained model is involved.

## Architecture

- Character-level tokenizer built from the training corpus.
- 2 decoder blocks, 4 attention heads, 48-dimensional embeddings.
- 96-character context window.
- AdamW optimization and next-character cross-entropy.
- Tiny Shakespeare training text.
- Plain JSON weight export consumed by a small JavaScript inference engine.

This is intentionally a small educational model, not a competitive general-purpose assistant. It learns character patterns and Shakespeare-like text structure. Training more steps can improve its output, but the small model and dataset impose real limits.

## GitHub Pages

Publish from the repository's `main` branch, root folder. The workflow only updates `model.json`; the static site remains on Pages.

## Files

- `index.html`, `styles.css`, `app.js`: browser interface and inference runtime.
- `train.py`: PyTorch model and CPU training/export.
- `.github/workflows/train.yml`: manual GitHub Actions training job.
