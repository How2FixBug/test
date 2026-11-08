# littleGPT v6 training

## Fast local training on Apple Silicon

From the repository folder on an Apple Silicon Mac with PyTorch installed:

```bash
python3 train_v6.py
```

The script automatically selects PyTorch's MPS GPU backend when available and falls back to CPU otherwise. It saves resumable checkpoints every 500 steps and writes the finished `model.json` into the repository folder.

After training, commit and push `model.json` to publish the new model on GitHub Pages.

## GitHub Actions

The `Train littleGPT v6` workflow runs on GitHub's CPU runner. It uses 3,000-step chunks rather than v5's 500-step chunks, vectorized packed-story batches, one-time CPU profiling, and throughput logging. GitHub's runner is still CPU-only; use the local command above to use MPS acceleration.
