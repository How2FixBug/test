# ASTRA 0.2 — Conversational Model

## Current objective

ASTRA 0.2 is a stepping stone toward a general-purpose model. This stage
focuses on **natural everyday conversation**: listening, reacting, following
context across turns, asking useful follow-ups, and recognizing when a request
needs external information.

It is not intended to memorize a large encyclopedia. A future information
layer will handle changing facts and external lookups.

## Training source

- `dataset_conversation_candidate.txt` is the current curated conversation corpus.
- Current corpus: **512 serialized conversations**.
- Every conversation now contains **exactly 3 user messages**.
- Current totals: **1,536 user messages** and **1,544 ASTRA messages**. The eight
  additional ASTRA messages are part of the information-lookup handoff examples.
- All 512 conversations are multi-turn.
- The two earlier four-user-message threads were consolidated into three
  messages while preserving their key context.
- Added 230 new hand-authored conversations across recent expansions; each has
  three user messages and context-specific replies.
- Exact duplicate user-message bodies: **0**.
- Exact duplicate ASTRA-message bodies: **0**.
- These uniqueness checks compare normalized text exactly; they do not prove
  that every message is semantically unique or that every example is equally
  useful. A human quality review is still required.
- Eight information-tool request/result examples and seven concise thinking
  notes remain in the corpus.
- The 2,228 formulaic entries from the earlier 2,500-example candidate were
  removed rather than retained to preserve an arbitrary count. The current
  corpus is intentionally smaller and more conversationally varied.
- `dataset.txt` is the older imported corpus and is intentionally preserved.
  It is not the default source for this conversational stage.

The tokenizer preserves complete conversation sequences. The training window
builder is intended to calculate loss only on ASTRA message bodies and closing
markers; user messages and external tool results remain context and are not
targets. The training implementation still needs its own execution tests.

## Prepare and audit

From this folder, run:

```bash
python3 prepare_dataset.py
```

This rebuilds the tokenizer and encoded candidate corpus, audits the exact
conversation file, and runs tokenizer/causality checks. It does **not** train
the language model.

The dataset auditor now requires exactly three user messages per conversation
by default, checks exact normalized uniqueness of user and ASTRA message
bodies, and requires at least 512 multi-turn conversations.

## Train

Do not run training until the corpus audit and sampled conversations have been
reviewed and training is explicitly approved.

```bash
python3 train.py
```

## Interactive chat

Start a persistent terminal conversation with the trained model:

```bash
python3 chat.py
```

The model is loaded once and reused for every reply. ASTRA keeps the current
session's conversation history as context. Use `/reset` to clear that context
or `/quit` to exit. Older turns are dropped automatically when the context
window fills. This does not save chat history to disk or modify the dataset.

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
