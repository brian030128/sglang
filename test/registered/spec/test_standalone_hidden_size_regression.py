import inspect
import re
import unittest

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_STANDALONE,
    DEFAULT_TARGET_MODEL_STANDALONE,
    CustomTestCase,
)

register_cuda_ci(est_time=30, suite="stage-b-test-1-gpu-small")

TOPK = 10


class _FakeWorker:
    def __init__(self, target_config: ModelConfig, draft_config: ModelConfig):
        self.device = get_device()
        self.model_config = draft_config

        target_runner = type("_R", (), {})()
        target_runner.model_config = target_config
        target_worker = type("_W", (), {})()
        target_worker.model_runner = target_runner
        self.target_worker = target_worker

        self.topk = TOPK


class _FakeBatch:
    spec_info = None


class TestStandaloneHiddenSizeRegression(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(get_device())
        cls.target_config = ModelConfig(
            model_path=DEFAULT_TARGET_MODEL_STANDALONE, trust_remote_code=True
        )
        cls.draft_config = ModelConfig(
            model_path=DEFAULT_DRAFT_MODEL_STANDALONE,
            trust_remote_code=True,
            is_draft_model=True,
        )
        assert (
            cls.target_config.hidden_size != cls.draft_config.hidden_size
        ), "test pair must have mismatched hidden sizes to exercise the bug"

    def _make_target_running_batch(self, batch_size: int) -> EagleDraftInput:
        target_hidden = self.target_config.hidden_size
        return EagleDraftInput(
            hidden_states=torch.randn(
                batch_size,
                target_hidden,
                device=self.device,
                dtype=self.target_config.dtype,
            ),
            verified_id=torch.zeros(batch_size, device=self.device, dtype=torch.int32),
            topk_p=torch.zeros(
                (batch_size, TOPK), device=self.device, dtype=torch.float32
            ),
            topk_index=torch.zeros(
                (batch_size, TOPK), device=self.device, dtype=torch.int64
            ),
        )

    def test_draft_preprocess_idle_merges_into_target_batch(self):
        worker = _FakeWorker(self.target_config, self.draft_config)
        batch = _FakeBatch()

        EAGLEWorker._draft_preprocess_idle(worker, batch)

        idle = batch.spec_info
        self.assertIsInstance(idle, EagleDraftInput)

        running = self._make_target_running_batch(batch_size=3)
        running.merge_batch(idle)

        self.assertEqual(
            running.hidden_states.shape, (3, self.target_config.hidden_size)
        )
        self.assertEqual(idle.hidden_states.dtype, self.target_config.dtype)
        self.assertEqual(idle.hidden_states.device.type, self.device.type)

    def test_forward_draft_extend_idle_fallback_uses_target_hidden_size(self):
        src = inspect.getsource(EAGLEWorker.forward_draft_extend_after_decode)
        guard = re.search(r"verified_id\.numel\(\)\s*==\s*0", src)
        self.assertIsNotNone(guard, "fallback guard not found; test is stale")
        fallback = src[guard.start() :]

        call = re.search(
            r"EagleDraftInput\.create_idle_input\([^)]*\)", fallback, re.DOTALL
        )
        self.assertIsNotNone(call, "create_idle_input call not found in fallback")
        relevant = fallback[: call.end()]

        self.assertTrue(
            "target_worker.model_runner.model_config" in relevant
            or "self.target_worker.model_config" in relevant,
            f"fallback must read target config, got:\n{relevant}",
        )
        for forbidden in (
            "self.model_config.hidden_size",
            "self.model_config.spec_hidden_size",
            "self.model_config.dtype",
        ):
            self.assertNotIn(
                forbidden, relevant, f"fallback still references {forbidden}"
            )


if __name__ == "__main__":
    unittest.main()
