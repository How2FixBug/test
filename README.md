# littleGPT v2

A small decoder-only Transformer trained **from random initialization** on a GitHub Actions CPU runner, then executed in the browser.

## Train it

1. Open the **Actions** tab.
2. Select **Train littleGPT** → **Run workflow**.
3. Choose 750, 1,500, or 2,500 steps.
4. Wait for the workflow to finish. It commits the new `model.json` weights.
5. Refresh the GitHub Pages site.

The workflow downloads a small, fixed-size sample of TinyStories from the Hugging Face dataset viewer API. It does not download the multi-gigabyte full dataset or use a pretrained model.

## What changed in v2

- TinyStories instead of Tiny Shakespeare: short, simple modern-English stories.
- Word-and-punctuation tokens instead of individual characters.
- 2,048-token vocabulary.
- 2 decoder blocks, 4 attention heads, 96-dimensional embeddings.
- 96-token context window.
- PyTorch scaled-dot-product attention for a faster CPU training path.
- 1,500 default training steps, with 750/2,500 options.
- Browser-side JavaScript inference; no model-serving backend.

## Important limitations

This is still a small next-token language model, not an instruction-tuned assistant. It is better suited to continuing simple story-like text than answering arbitrary questions. More steps can help, but a small model and a small training sample cannot provide ChatGPT-level reasoning or factual reliability.

## Files

- `index.html`, `styles.css`, `app.js`: browser chat UI and inference.
- `train.py`: data download, model, CPU training, and JSON export.
- `.github/workflows/train.yml`: manual training workflow.
