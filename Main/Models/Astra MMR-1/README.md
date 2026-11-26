# ASTRA MMR-1

Independent experimental recurrent language-model architecture. This folder
does not replace or modify ASTRA 0.2.

## Initial configuration

- Vocabulary: 8,192
- Hidden width: 256
- Recurrent blocks: 4
- Memories per block: 3 (fast, working, slow)
- SwiGLU width: 512
- Dropout: 0.05
- Tied input/output embeddings
- No self-attention or positional embeddings

## Memory behavior

Each block maintains three vectors of shape `[batch, 3, width]`.
Input-dependent write gates update the vectors. Each memory has a learned
scalar update rate constrained to its own interval:

| Memory | Initial rate | Allowed interval |
| --- | ---: | ---: |
| Fast | 0.50 | 0.25–1.00 |
| Working | 0.10 | 0.03–0.25 |
| Slow | 0.01 | 0.001–0.03 |

A learned softmax controller reads a weighted mixture of the three memories.
The fused representation passes through a residual SwiGLU processor.

The write gates and candidate values depend only on the current block input.
This makes each memory update an affine recurrence for a fixed input sequence,
which is a prerequisite for investigating associative parallel scans.

## Conversation state

`model(tokens, state=None)` starts a new conversation and returns
`(logits, state)`. Pass the returned state to the next call to continue that
conversation. Discard it when the conversation ends. Never share state between
independent conversations.

For truncated backpropagation through time, call
`model.detach_state(state)` between training chunks. This detaches gradients
without resetting the numerical memory values.

## Run tests

From this directory:

```bash
python3 -m unittest -v
```

The current implementation is a correctness-first sequential reference.
Parallel-scan optimization, tokenizer/runtime integration, training scripts,
and sampling are intentionally not included yet.

## Important status

This is an untrained architecture prototype, not a working chatbot. The
configuration and memory mechanisms are hypotheses to evaluate. No claim of
better quality, speed, or thermal performance has been established.
