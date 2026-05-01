"""Regression tests for STANDALONE spec-decode idle EagleDraftInput hidden-size handling."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import inspect
import re
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.test.test_utils import CustomTestCase

TOPK = 10
DEVICE = torch.device("cpu")
DTYPE = torch.float32

# (target_hidden, draft_hidden, label)
HIDDEN_SIZE_CASES = [
    (4096, 2048, "llama8b_+_llama1b"),
    (2048, 8192, "llama70b_+_llama1b"),
    (4096, 4096, "equal_sizes"),
]


def _make_real_target_input(batch_size: int, target_hidden: int) -> EagleDraftInput:
    return EagleDraftInput(
        hidden_states=torch.randn(batch_size, target_hidden, dtype=DTYPE),
        verified_id=torch.zeros(batch_size, dtype=torch.int32),
        topk_p=torch.zeros((batch_size, TOPK), dtype=torch.float32),
        topk_index=torch.zeros((batch_size, TOPK), dtype=torch.int64),
    )


def _make_mock_eagle_worker(target_hidden: int, draft_hidden: int) -> SimpleNamespace:
    return SimpleNamespace(
        device=DEVICE,
        model_config=SimpleNamespace(
            hidden_size=draft_hidden, spec_hidden_size=draft_hidden, dtype=DTYPE
        ),
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(
                    hidden_size=target_hidden,
                    spec_hidden_size=target_hidden,
                    dtype=DTYPE,
                )
            )
        ),
        topk=TOPK,
    )


class TestEagleIdleHiddenSize(CustomTestCase):
    def test_draft_preprocess_idle_uses_target_hidden_size(self):
        for target_hidden, draft_hidden, label in HIDDEN_SIZE_CASES:
            with self.subTest(case=label):
                mock_worker = _make_mock_eagle_worker(target_hidden, draft_hidden)
                batch = SimpleNamespace()
                EAGLEWorker._draft_preprocess_idle(mock_worker, batch)

                self.assertEqual(
                    batch.spec_info.hidden_states.shape, (0, target_hidden)
                )
                self.assertEqual(batch.spec_info.hidden_states.dtype, DTYPE)

                real = _make_real_target_input(
                    batch_size=3, target_hidden=target_hidden
                )
                real.merge_batch(batch.spec_info)
                self.assertEqual(real.hidden_states.shape, (3, target_hidden))

    def test_draft_extend_idle_fallback_uses_target_hidden_size(self):
        # forward_draft_extend_after_decode is too heavy to invoke directly,
        # so source-inspect the verified_id.numel() == 0 idle-fallback block.
        src = inspect.getsource(EAGLEWorker.forward_draft_extend_after_decode)

        guard_match = re.search(r"verified_id\.numel\(\)\s*==\s*0", src)
        self.assertIsNotNone(
            guard_match, "idle-fallback guard not found; test is stale"
        )
        fallback_block = src[guard_match.start() :]

        call_match = re.search(
            r"EagleDraftInput\.create_idle_input\([^)]*\)", fallback_block, re.DOTALL
        )
        self.assertIsNotNone(call_match, "create_idle_input call not found in fallback")
        relevant = fallback_block[: call_match.end()]

        uses_target_cfg = (
            "target_worker.model_runner.model_config" in relevant
            or "self.target_worker.model_config" in relevant
        )
        self.assertTrue(uses_target_cfg, f"must read target config, got:\n{relevant}")

        for forbidden in (
            "self.model_config.hidden_size",
            "self.model_config.spec_hidden_size",
            "self.model_config.dtype",
        ):
            self.assertNotIn(forbidden, relevant, f"still references {forbidden}")


if __name__ == "__main__":
    unittest.main()
