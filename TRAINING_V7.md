# littleGPT v7

v7 is a new model format and training pipeline; it does not reuse v6 checkpoints.

## What changed

- **9,796,448 parameters:** 5 transformer blocks, width 288, 6 attention heads, 256-token context.
- **8,192-token byte-subword vocabulary:** frequent byte sequences are learned from a representative sample of training stories. Every UTF-8 byte has a fallback, so unfamiliar words are composed from smaller pieces rather than becoming `<unk>`.
- **250,000 TinyStories rows** by default, with resumable download caching.
- **Story-boundary-safe training:** each training window is sampled wholly inside one story. Context from one story cannot bleed into another.
- **Vectorized batches, fixed tensor shapes, gradient clipping, cosine learning-rate decay, held-out validation, and best-validation checkpoint export.**
- **Apple Silicon MPS:** automatically selected when available. CPU is the fallback.
- **Browser KV cache:** v7 inference caches each layer's keys and values instead of recalculating attention for the entire prompt on every generated token.
- **Compact model export:** learned weights are stored as float16 base64 blobs (about 20 MB of raw weight data) and unpacked into typed arrays in the browser, avoiding a massive nested-number JSON file and staying below GitHub's per-file size limit.

## Train locally on an Apple Silicon Mac

Use the existing Python/PyTorch environment. From the repository folder:

```bash
python3 train_v7.py
```

The defaults are 250,000 stories and 30,000 optimizer steps. The first run downloads and caches the dataset and builds the tokenizer; later runs reuse the story cache. Checkpoints are written every 500 steps, so rerunning resumes from the latest saved checkpoint.

For a smaller smoke run (still a separate v7 dataset/tokenizer configuration):

```bash
DATASET_ROWS=30000 TRAIN_STEPS=500 python3 train_v7.py
```

Do not switch `DATASET_ROWS` after creating a checkpoint unless you remove `checkpoint-v7.pt`; the script intentionally rejects mismatched checkpoints.

When training finishes, `model.json` is exported in the repository folder. Commit and push it to publish the model through GitHub Pages.

## Performance notes

- The MPS backend uses the Mac's Apple GPU. It shares unified memory with the CPU, so close memory-heavy apps if the machine starts swapping.
- v7 deliberately uses fixed batch and context shapes. Mixed precision is **not enabled by default**: compare it against FP32 on the actual Mac before using it, because unsupported or numerically unstable kernels can make training slower or less reliable.
- KV caching removes repeated key/value projections and attention over old tokens. The remaining dense matrix operations are still JavaScript CPU work in the current browser runtime; a future WebGPU backend would be a separate optimization. Float16 packing is only an export/storage format; training remains FP32.
- Training speed and story quality are not guaranteed by parameter count alone. Compare validation loss and inspect generated samples after the first completed run.
