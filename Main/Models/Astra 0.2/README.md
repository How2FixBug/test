# ASTRA 0.2 — Basic Conversation

## Goal

ASTRA 0.2 is a small chatbot focused exclusively on basic, natural conversation.
It should handle greetings, small talk, casual reactions, preferences, simple
back-and-forth exchanges, and conversational follow-ups.

**It is not a factual assistant.** Do not add trivia, science, history, geography,
encyclopedic answers, general knowledge, or instruction-following datasets.
The goal is conversational behavior, not answering factual questions.

## Dataset

The target is **5,000+ reviewed conversation examples**.

- \`dataset.txt\` contains the original seed examples plus imported conversational pairs.
- \`import_chitchat.py\` downloads BYU's public Chit-Chat corpus and converts eligible
  two-person exchanges into ASTRA user/assistant examples.
- The importer filters obvious factual questions, knowledge-heavy replies, malformed
  turns, links, overly long messages, duplicates, and clearly unsuitable language.
- Conversation history is preserved for follow-up examples.
- The raw source file is cached in \`.cache/\` and is not intended to be committed.

The source is [BYU-PCCL/chitchat-dataset](https://github.com/BYU-PCCL/chitchat-dataset),
which lists an MIT license. ASTRA's imported examples retain source markers, and
the original dataset should be credited using the paper citation in that repository.
The source contains mixed open-domain conversation, so filtering is heuristic:
**review the audit and sample the imported examples before training.** The importer
does not guarantee that every factual or off-topic sentence is removed.

Do not import general-purpose instruction, trivia, science, or knowledge datasets.
The generic \`import_datasets.py\` remains disabled; the BYU Chit-Chat importer is the
single explicitly approved source for this version.

When expanding the corpus, prefer:
- Greetings and goodbyes.
- Check-ins and small talk.
- Casual reactions and acknowledgements.
- Preferences, hobbies, and everyday plans.
- Listening, empathy, apologies, thanks, and topic changes.
- Follow-up replies that depend on earlier conversational context.

Avoid repetitive paraphrase spam. Prefer varied, believable exchanges.

## Prepare

Run the full pipeline with one command:

\`\`\`bash
python3 prepare_dataset.py
\`\`\`

This downloads/caches the BYU source if needed, filters and imports examples,
rebuilds the tokenizer, audits the corpus, and runs correctness checks. It does
not train the neural network.

To refresh the cached source later:

\`\`\`bash
python3 import_chitchat.py --refresh
\`\`\`

## Train

\`\`\`bash
python3 train.py
\`\`\`

Training starts from random initialization and uses ASTRA 0.2's own checkpoint.

## Generate

\`\`\`bash
python3 generate.py "Hey! How's it going?"
\`\`\`

## Status

- [x] Separate ASTRA 0.2 copy created from 0.1.
- [x] Replaced inherited knowledge-heavy seed data with conversation-only examples.
- [x] Added an allowlisted importer for BYU's open-domain Chit-Chat dataset.
- [x] Added heuristic filtering, multi-turn context conversion, deduplication, and source attribution.
- [ ] Reach and review at least 5,000 unique examples.
- [ ] Improve training to represent longer conversational histories directly.
- [ ] Train and evaluate ASTRA 0.2.
