"""ASTRA MMR-1: compact multi-memory recurrent language model.

This is an independent experimental architecture. It does not load or modify
ASTRA 0.2 checkpoints. The reference implementation prioritizes correctness;
its time loop is intentionally sequential. Parallel-scan optimization comes
only after equivalence tests exist.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MemoryBlock(nn.Module):
    """One processing block with fast, working, and slow recurrent memories."""

    # Initial alpha values and learnable bounds for the three memory systems.
    ALPHA_INITIAL = (0.50, 0.10, 0.01)
    ALPHA_BOUNDS = ((0.25, 1.00), (0.03, 0.25), (0.001, 0.03))

    def __init__(self, width: int, ffn_width: int, dropout: float = 0.05):
        super().__init__()
        self.width = width

        # All write gates and candidates depend on the current input only.
        self.write_gate = nn.Linear(width, 3 * width)
        self.write_candidate = nn.Linear(width, 3 * width)

        self.read_controller = nn.Linear(width, 3)
        self.memory_projection = nn.Linear(width, width)

        self.norm = nn.LayerNorm(width)
        self.swiglu_gate = nn.Linear(width, ffn_width)
        self.swiglu_value = nn.Linear(width, ffn_width)
        self.swiglu_output = nn.Linear(ffn_width, width)
        self.dropout = nn.Dropout(dropout)

        # Each block learns three scalar timescales, constrained to distinct
        # ranges so the memories cannot trivially collapse to the same rate.
        initial_logits = []
        for initial, (low, high) in zip(
            self.ALPHA_INITIAL, self.ALPHA_BOUNDS
        ):
            p = (initial - low) / (high - low)
            p = min(max(p, 1e-5), 1.0 - 1e-5)
            initial_logits.append(math.log(p / (1.0 - p)))
        self.alpha_logits = nn.Parameter(torch.tensor(initial_logits))

    def timescales(self) -> Tensor:
        """Return the three learned update rates, each inside its bounds."""
        bounds = self.alpha_logits.new_tensor(self.ALPHA_BOUNDS)
        low, high = bounds[:, 0], bounds[:, 1]
        return low + (high - low) * torch.sigmoid(self.alpha_logits)

    def forward(
        self, x: Tensor, state: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Process one timestep.

        Args:
            x: [batch, width] current representation.
            state: [batch, 3, width] fast/working/slow memory state.
        """
        batch = x.shape[0]
        if x.ndim != 2 or x.shape[-1] != self.width:
            raise ValueError(f"x must have shape [batch, {self.width}]")
        if state.shape != (batch, 3, self.width):
            raise ValueError(
                f"state must have shape [batch, 3, {self.width}], "
                f"got {tuple(state.shape)}"
            )

        gates = torch.sigmoid(self.write_gate(x)).view(batch, 3, self.width)
        candidates = torch.tanh(
            self.write_candidate(x)
        ).view(batch, 3, self.width)
        alpha = self.timescales().view(1, 3, 1)

        write = alpha * gates
        new_state = (1.0 - write) * state + write * candidates

        read_weights = torch.softmax(self.read_controller(x), dim=-1)
        memory = torch.sum(
            read_weights.unsqueeze(-1) * new_state, dim=1
        )
        fused = x + self.memory_projection(memory)

        normalized = self.norm(fused)
        gated = F.silu(self.swiglu_gate(normalized))
        value = self.swiglu_value(normalized)
        processed = self.swiglu_output(gated * value)
        output = fused + self.dropout(processed)
        return output, new_state


class MMR1Config:
    """Small explicit config object without an external dependency."""

    def __init__(
        self,
        vocab_size: int = 8192,
        width: int = 256,
        layers: int = 4,
        ffn_width: int = 512,
        dropout: float = 0.05,
    ):
        if vocab_size <= 0 or width <= 0 or layers <= 0 or ffn_width <= 0:
            raise ValueError("vocab_size, width, layers, and ffn_width must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.vocab_size = vocab_size
        self.width = width
        self.layers = layers
        self.ffn_width = ffn_width
        self.dropout = dropout


class ASTRA_MMR1(nn.Module):
    """Positionless recurrent language model with persistent conversation state.

    Pass the returned state back into the next forward call to continue the
    same conversation. Pass state=None to start a fresh conversation.
    """

    def __init__(self, config: Optional[MMR1Config] = None, **kwargs):
        super().__init__()
        if config is not None and kwargs:
            raise ValueError("Pass either config or keyword configuration, not both")
        self.config = config or MMR1Config(**kwargs)
        c = self.config

        self.token_embedding = nn.Embedding(c.vocab_size, c.width)
        self.embedding_dropout = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(
            MemoryBlock(c.width, c.ffn_width, c.dropout)
            for _ in range(c.layers)
        )
        self.final_norm = nn.LayerNorm(c.width)

        # The output projection is tied to the input token embeddings.
        self.apply(self._initialize)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, ...]:
        """Create zeroed memory for a new conversation."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        parameter = self.token_embedding.weight
        device = device or parameter.device
        dtype = dtype or parameter.dtype
        return tuple(
            torch.zeros(batch_size, 3, self.config.width, device=device, dtype=dtype)
            for _ in self.blocks
        )

    @staticmethod
    def detach_state(state: Sequence[Tensor]) -> tuple[Tensor, ...]:
        """Detach recurrent state between truncated-BPTT training chunks."""
        return tuple(memory.detach() for memory in state)

    def forward(
        self,
        token_ids: Tensor,
        state: Optional[Sequence[Tensor]] = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Return next-token logits and updated conversation state.

        token_ids: [batch, time], int64 token IDs.
        logits:    [batch, time, vocab_size].
        """
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, time]")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must have dtype torch.long")
        if token_ids.shape[1] == 0:
            raise ValueError("token sequence must contain at least one token")

        batch_size = token_ids.shape[0]
        if state is None:
            current_state = self.initial_state(
                batch_size,
                device=token_ids.device,
                dtype=self.token_embedding.weight.dtype,
            )
        else:
            if len(state) != len(self.blocks):
                raise ValueError(
                    f"Expected {len(self.blocks)} block states, got {len(state)}"
                )
            current_state = tuple(state)
            for memory in current_state:
                if memory.shape != (batch_size, 3, self.config.width):
                    raise ValueError("Recurrent state shape does not match input batch")

        x = self.embedding_dropout(self.token_embedding(token_ids))
        outputs = []

        # Correctness-first reference recurrence. A later optimized scan must
        # match this implementation numerically before replacing it.
        for timestep in range(token_ids.shape[1]):
            hidden = x[:, timestep, :]
            next_states = []
            for block, memory_state in zip(self.blocks, current_state):
                hidden, updated_memory = block(hidden, memory_state)
                next_states.append(updated_memory)
            current_state = tuple(next_states)
            outputs.append(self.final_norm(hidden))

        hidden_sequence = torch.stack(outputs, dim=1)
        logits = F.linear(hidden_sequence, self.token_embedding.weight)
        return logits, current_state
