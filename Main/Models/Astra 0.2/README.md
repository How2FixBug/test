# ASTRA 0.2 — Basic Conversation

## Goal

ASTRA 0.2 is a small chatbot focused exclusively on basic, natural conversation.
It should handle greetings, small talk, casual reactions, preferences, simple
back-and-forth exchanges, and conversational follow-ups.

**It is not a factual assistant.** Do not add trivia, science, history, geography,
encyclopedic answers, general knowledge, or instruction-following datasets.
The goal is conversational behavior, not answering factual questions.

## Dataset

`dataset.txt` contains a small, curated seed corpus of original everyday
conversation examples. Some prompts include short prior conversation transcripts
to teach context-aware replies. There are no external knowledge datasets.

`import_datasets.py` is deliberately disabled so broad datasets cannot be
accidentally imported into this version.

When expanding the corpus, add only natural everyday conversation:
- Greetings and goodbyes.
- Check-ins and small talk.
- Casual reactions and acknowledgements.
- Preferences, hobbies, and everyday plans.
- Listening, empathy, apologies, thanks, and topic changes.
- Follow-up replies that depend on earlier conversational context.

Avoid repetitive paraphrase spam. Prefer varied, believable exchanges. If an
example mainly tests factual knowledge or teaches information about the world,
it does not belong in ASTRA 0.2.

## Prepare

Run the whole local preparation pipeline with one command:

```bash
python3 prepare_dataset.py
```

This builds the tokenizer, encodes the dataset, audits it, and runs correctness
checks. It does not download or import external data.

## Train

```bash
python3 train.py
```

Training starts from random initialization and uses ASTRA 0.2's own checkpoint.

## Generate

```bash
python3 generate.py "Hey! How's it going?"
```

## Status

- [x] Separate ASTRA 0.2 copy created from 0.1.
- [x] Replaced the inherited knowledge-heavy seed data with conversation-only examples.
- [x] Disabled external general-purpose dataset imports.
- [x] Preparation pipeline no longer downloads outside datasets.
- [ ] Expand the curated conversation corpus with more varied multi-turn examples.
- [ ] Improve training to represent longer conversational histories directly.
- [ ] Train and evaluate ASTRA 0.2.
