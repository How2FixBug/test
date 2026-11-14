# ASTRA 0.2 — Conversational Model

## Current objective

ASTRA 0.2 is a small stepping stone toward a general-purpose model. This stage
focuses on **natural everyday conversation**: listening, reacting, remembering
details within a conversation, asking useful follow-ups, and knowing when a
request needs an information lookup.

It is not intended to memorize a large encyclopedia. A future information
layer will handle changing facts and external lookups.

## Training source

- `dataset_conversation_candidate.txt` is the current curated conversation corpus.
- Current corpus target: **2,500 serialized conversations**.
- Latest structural audit: 2,500 conversations, 2,630 user turns, and 2,638
  ASTRA turns. All user and ASTRA message bodies are unique.
- **100 conversations are multi-turn**, including 65 newly added contextual
  follow-up exchanges. The audit now enforces a minimum of 100 multi-turn
  conversations. There are 8 information-tool request/result examples and 7
  concise thinking notes.
- The audit reports recurring 4–6 word phrases for manual review. Exact
  uniqueness does not prove every example is semantically distinct or equally
  useful; review the sampled conversations before training.
- It contains serialized conversations, including real multi-turn examples.
- `<think>...</think>` notes appear only in selected cases where a short
  decision note is useful. They are inside the corresponding `<astra>` message.
- `<getinfo>...</getinfo>` marks a request for external information.
- `<getinforesult>...</getinforesult>` represents information returned by an
  external service; it is context, not an ASTRA response.
- `dataset.txt` is the older imported corpus and is intentionally preserved.
  It is not the default source for this conversational stage.

The tokenizer preserves complete conversation sequences. The training window
builder calculates loss only on ASTRA message bodies and closing markers; user
messages and external tool results remain context and are not targets.

## Prepare and audit

From this folder, run:

```bash
python3 prepare_dataset.py
```

This rebuilds the tokenizer and encoded candidate corpus, audits the exact
conversation file, and runs tokenizer/causality checks. It does **not** train
the language model.

## Train

Do not run training until the corpus audit and sampled conversations have been
reviewed and training is explicitly approved.

```bash
python3 train.py
```

## Generate

Generation requires a checkpoint trained with the matching tokenizer
configuration.

```bash
python3 generate.py "Hello! What can you do?"
```

The generator hides `<think>` notes from the displayed reply. If ASTRA emits a
complete `<getinfo>...</getinfo>` request, it prints an
`ASTRA_TOOL_REQUEST:` handoff instead of pretending it fetched the information.
The external information service and automatic tool-result round trip are a
separate future component; this repository currently defines the request format
but does not execute live lookups.
