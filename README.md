# Roger AI

A fully custom, small GPT-style chatbot project. The neural network, tokenizer, training loop, inference code, chat server, and interface are project code; **no pretrained model weights or hosted LLM API are used**.

## Reality check

This is not equivalent to a 10B-parameter model. A 10B model cannot fit in 2 GB RAM: 4-bit weights alone require about 5 GB. This project instead uses a compact byte-level GPT (default: 384 width, 8 layers, 6 heads; roughly tens of millions of parameters) and leaves room for the runtime and conversation memory. Smart retrieval and good training data can improve usefulness, but cannot magically give a tiny model the knowledge or reasoning capacity of a 10B model.

## Requirements

- Python 3.10+
- PyTorch 2.2+
- About 2 GB RAM budget for the process; reduce model size/context if needed.

On Apple Silicon, PyTorch can use MPS automatically.

## 1. Set up

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Add training data

Create `data/train.txt` with plain-text material you have permission to use. For a chatbot, use lots of examples in this format:

```text
User: What is a variable in programming?
Assistant: A variable is a named place to store a value...

User: Explain photosynthesis simply.
Assistant: Plants use light energy to turn water and carbon dioxide into sugars...
```

The included byte tokenizer has no vocabulary-training step. It can represent any UTF-8 text, but byte-level tokens make sequences longer, so use a reasonably large and varied dataset.

## 3. Train from random initialization

```bash
python train.py --steps 3000
```

The default is intentionally small. Training quality depends heavily on the amount, cleanliness, and diversity of your text. A tiny dataset will overfit and will not produce a capable general assistant. Save the trained checkpoint at `checkpoints/roger-small.pt`.

For a smaller/faster experiment:

```bash
python train.py --width 256 --layers 4 --heads 4 --context 256 --batch-size 4 --steps 1000
```

## 4. Chat

```bash
python app.py
```

Open http://127.0.0.1:8000. Conversations are stored locally in `data/memory.jsonl`; relevant snippets are retrieved by keyword overlap. No conversation is sent to a remote service.

## Current limitations

- This is an educational first version, not a production-grade assistant.
- Training uses next-byte prediction and does not include preference tuning, tool use, or a large curated instruction corpus yet.
- The memory layer is lightweight keyword retrieval, not a learned long-term memory.
- The model only knows what it learns from its training text and current prompt.
- Training on a Mac may take much longer than five minutes; measure before scaling up.
