# littleGPT v3

A compact decoder-only Transformer trained **from random initialization** on a GitHub Actions CPU runner, then executed locally in the browser.

## Train it

1. Open the **Actions** tab.
2. Select **Train littleGPT** → **Run workflow**.
3. Choose 3,000, 5,000, or 7,000 steps.
4. Wait for the workflow to finish. It commits the new `model.json` weights.
5. Refresh the GitHub Pages site.

The workflow downloads 6,000 TinyStories examples through the Hugging Face dataset viewer API. It does not download the full dataset or use a pretrained model.

## What changed in v3

- 6,000 training stories, shuffled before the train/validation split.
- 4,096-token word-and-punctuation vocabulary.
- 3 decoder blocks, 4 attention heads, 112-dimensional embeddings.
- 128-token context window.
- Warmup plus cosine learning-rate decay.
- Optimized PyTorch scaled-dot-product attention for CPU training.
- Browser-side JavaScript inference; no model-serving backend.

## Limitations

This is a small next-token model trained on children's stories. It can learn story patterns, but it is not instruction-tuned and should not be expected to answer arbitrary questions like ChatGPT. A successful training run and a lower validation loss are useful signals, not proof that every generated passage will be coherent.

## Files

- `index.html`, `styles.css`, `app.js`: browser chat UI and inference.
- `train.py`: data download, model, CPU training, and JSON export.
- `.github/workflows/train.yml`: manual training workflow.
