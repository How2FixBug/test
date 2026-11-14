# ASTRA 0.2: General-Purpose Assistant

## Goal

ASTRA 0.2 is a stepping stone toward a general-purpose language model. Its
training data should cover a broad range of everyday and knowledge-based
conversations, explanations, writing, reasoning, and useful instructions.

## Dataset

- `dataset.txt` is the training source.
- Existing curated examples are preserved.
- `fetch_and_filter.py` imports general-purpose conversations from
  **UltraChat 200k**.
- UltraChat 200k provides 207,865 English training conversations in its
  `train_sft` split and is distributed under the **MIT License**.
- The dataset is synthetic assistant dialogue data, not a raw-text
  pretraining corpus. It is suitable for conversational supervised fine-tuning.
- The importer preserves prior turns in each prompt, filters obvious unsafe
  material and personal contact details, removes duplicates, and records
  source attribution in `import_manifest.json`.
- Automated import filters are not a guarantee that every retained answer is
  factually correct.

Source:
- [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- [MIT License](https://opensource.org/license/mit)

## Import the dataset

From this folder, run:

```bash
python3 fetch_and_filter.py
```

This scans 12,000 source conversations and imports up to 5,000 unique
assistant-response examples. It uses Python's standard library, downloads
through the Hugging Face dataset viewer API, and does not install packages,
use Ollama, or train ASTRA.

Options:

```bash
python3 fetch_and_filter.py --dry-run
python3 fetch_and_filter.py --max-add 2500
python3 fetch_and_filter.py --scan-rows 20000
```

The importer appends to `dataset.txt`; it does not replace the original
curated examples. Re-running it will deduplicate against the existing corpus.

## Review dataset quality

Run the non-destructive review tool:

```bash
python3 review_dataset.py
```

It scans every example and writes flagged records to
`quality_review.jsonl`. Flags include nested conversation transcripts,
time-sensitive claims, specific numbers and dates, unsupported source claims,
template placeholders, possible false personal-experience claims, capability
claims, unusually long answers, generic answers, and exact repeated prompts or
answers.

The script does not decide that a flagged example is wrong, and it does not
certify unflagged examples as correct. It creates a review queue for a human
to inspect. It never edits `dataset.txt`, tokenizer files, or model weights.
The generated review file is local and ignored by Git.

You can print fewer examples with:

```bash
python3 review_dataset.py --show 5
python3 review_dataset.py --show 0
```

## Prepare

After reviewing the dataset and deciding what should remain, rebuild the
tokenizer and encoded dataset, audit the corpus, and run correctness checks:

```bash
python3 prepare_dataset.py
```

This does not train ASTRA.

## Train

```bash
python3 train.py
```

Do not train until the dataset audit and quality samples look acceptable.

## Generate

```bash
python3 generate.py "Explain how a solar eclipse works."
```
