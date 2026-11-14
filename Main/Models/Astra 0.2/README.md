# ASTRA 0.2: Basic Conversation

## Goal

ASTRA 0.2 is a small chatbot focused exclusively on basic, natural conversation.
It should handle greetings, small talk, casual reactions, preferences, simple
back-and-forth exchanges, and conversational follow-ups.

**It is not a factual assistant.** Do not add trivia, science, history, geography,
encyclopedic answers, general knowledge, or instruction-following datasets.
The goal is conversational behavior, not answering factual questions.

## Dataset

The target is **5,000+ quality-filtered conversation examples**.

- `dataset.txt` is the training source.
- The original curated seed examples are retained.
- `fetch_and_filter.py` downloads Blended Skill Talk, verifies its official
  SHA-256 checksum, filters out knowledge-focused and unsuitable turns, removes
  duplicates, and automatically appends high-confidence examples.
- Blended Skill Talk is released under **CC BY 4.0**. Source and license markers
  are retained beside imported examples, and `import_manifest.json` records
  source details and import counts.
- The importer deliberately excludes the Wizard of Wikipedia portion and
  filters assistant responses that appear to invent personal biography.
- The importer is heuristic. Inspect the printed quality sample and the full
  dataset audit before training. Automatic filtering reduces manual work but
  does not guarantee every retained pair is perfect.
- `import_chitchat.py` remains a separate BYU candidate exporter. It does not
  add anything to training data.

Prefer:
- Greetings and goodbyes.
- Check-ins and small talk.
- Casual reactions and acknowledgements.
- Preferences, hobbies, and everyday plans.
- Listening, empathy, apologies, thanks, and topic changes.
- Follow-up replies that depend on earlier conversational context.

Avoid repetitive paraphrase spam. Prefer varied, believable exchanges. ASTRA's
assistant responses should not invent a human biography, personal history, or
real-world experiences.

## Fetch and filter public data

Run one command from this directory:

```bash
python3 fetch_and_filter.py
```

This downloads and verifies the Blended Skill Talk archive, filters it, removes
duplicates against the existing dataset, and appends up to 2,500 new examples.
The source archive is cached in `.cache/`. Use `--refresh` to download it again.

Useful options:

```bash
python3 fetch_and_filter.py --dry-run
python3 fetch_and_filter.py --max-add 1000
python3 fetch_and_filter.py --refresh
```

The importer uses only Python's standard library. It does not install packages,
call Ollama, or train the model.

Source:
- [Blended Skill Talk dataset documentation](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/blended_skill_talk)
- [ParlAI dataset build script and official archive checksum](https://github.com/facebookresearch/ParlAI/blob/main/parlai/tasks/blended_skill_talk/build.py)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Other public datasets must be checked individually for conversational fit and
reuse permissions before being added to the importer. DailyDialog and
EmpatheticDialogues are not included because their noncommercial license terms
are not a good default for a project whose future distribution may change.

## Prepare

Run the full preparation pipeline with one command:

```bash
python3 prepare_dataset.py
```

This removes unreviewed BYU imports from the training file, rebuilds the tokenizer,
audits the corpus, and runs correctness checks. It does not train ASTRA.

## Generate and review new examples

Use the local Ollama model to draft a small batch when additional hand-written
examples are useful:

```bash
python3 conversation_workbench.py
```

The workbench generates 20 candidates per run. It shows each prompt/response
pair with automatic warning flags. Manually review only when you choose to use
this optional path. The public dataset importer does not require approving each
pair one by one.

## Train

```bash
python3 train.py
```

Training starts from random initialization and uses ASTRA 0.2's own checkpoint.
Do not train until the dataset is large and its audit samples are acceptable.

## Generate

```bash
python3 generate.py "Hey! How's it going?"
```
