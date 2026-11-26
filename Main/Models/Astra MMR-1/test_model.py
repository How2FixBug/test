"""CPU-friendly correctness tests for ASTRA MMR-1."""
import unittest

import torch

from model import ASTRA_MMR1, MMR1Config


class MMR1Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = ASTRA_MMR1(
            MMR1Config(
                vocab_size=128,
                width=32,
                layers=2,
                ffn_width=64,
                dropout=0.0,
            )
        )
        self.model.eval()
        self.tokens = torch.randint(0, 128, (2, 9), dtype=torch.long)

    def test_output_and_state_shapes(self):
        logits, state = self.model(self.tokens)
        self.assertEqual(tuple(logits.shape), (2, 9, 128))
        self.assertEqual(len(state), 2)
        for memory in state:
            self.assertEqual(tuple(memory.shape), (2, 3, 32))

    def test_state_carries_between_chunks(self):
        first_logits, state = self.model(self.tokens[:, :4])
        second_logits, continued_state = self.model(self.tokens[:, 4:], state)
        whole_logits, _ = self.model(self.tokens)
        joined = torch.cat((first_logits, second_logits), dim=1)
        self.assertTrue(torch.allclose(joined, whole_logits, atol=1e-6, rtol=1e-5))
        self.assertEqual(len(continued_state), 2)

    def test_fresh_state_resets_conversation(self):
        _, state = self.model(self.tokens)
        with_state, _ = self.model(self.tokens[:, :3], state)
        fresh, _ = self.model(self.tokens[:, :3], None)
        self.assertFalse(torch.allclose(with_state, fresh))

    def test_timescales_stay_inside_bounds(self):
        for block in self.model.blocks:
            alpha = block.timescales().detach()
            bounds = alpha.new_tensor(block.ALPHA_BOUNDS)
            self.assertTrue(torch.all(alpha >= bounds[:, 0]))
            self.assertTrue(torch.all(alpha <= bounds[:, 1]))

    def test_gradients_are_finite(self):
        self.model.train()
        logits, _ = self.model(self.tokens)
        loss = logits.square().mean()
        loss.backward()
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(
                    torch.isfinite(parameter.grad).all(),
                    msg=f"Non-finite gradient in {name}",
                )
        self.assertIsNotNone(self.model.blocks[0].alpha_logits.grad)
        self.assertGreater(
            self.model.blocks[0].alpha_logits.grad.abs().sum().item(), 0.0
        )

    def test_parameter_count_is_compact(self):
        count = sum(p.numel() for p in self.model.parameters())
        self.assertLess(count, 100_000)


if __name__ == "__main__":
    unittest.main()
