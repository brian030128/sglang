"""Unmodified FlashInfer cascade attention as an EAGLE draft decode backend.

Baseline for Fast Draft: uses the stock ``flashinfer.MultiLevelCascadeAttentionWrapper``
(one ``BatchPrefillWithPagedKVCacheWrapper`` per level, combined with
``merge_state_in_place``) with the same 2-level decomposition as Fast Draft:

  Level 0: shared prefix, one KV segment per request, topk queries attend to it
  Level 1: unique suffix, one KV segment per branch (its draft path so far)

Index generation reuses the Triton kernels from ``cascade_index_gen`` so every
backend reads exactly the same KV entries. Unlike Fast Draft, the stock wrapper
has no plan-once/patch path, so it runs the full ``plan()`` for every draft step
of every iteration, and the levels execute as separate kernels plus a merge.

CUDA graphs: supported. The stock prefill wrapper pins its batch size at
construction, so one wrapper set is created per captured batch size. Their int
workspaces, pinned host buffers and index buffers are shared per (step, level)
across batch sizes (each captured graph is replayed right after its own plan),
but never across steps (all steps are planned before the graph replays them).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List

import torch
from flashinfer.cascade import MultiLevelCascadeAttentionWrapper

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.cascade_index_gen import (
    build_unique_indices,
    generate_cascade_shared_kv_indices,
    next_power_of_2,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

NUM_LEVELS = 2
FLOAT_WORKSPACE_BYTES = 256 * 1024 * 1024
INT_WORKSPACE_BYTES = 8 * 1024 * 1024


class FIMultiLevelCascadeDraftAttnBackend(AttentionBackend):
    """Per-step backend; runs whichever wrapper the parent planned for this step."""

    def __init__(self, model_runner: ModelRunner, parent, step_index: int):
        super().__init__()
        self._parent = parent
        self._step_index = step_index

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Planning is done by the parent multi-step backend
        pass

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if k is not None and v is not None and save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        wrapper = self._parent._active_wrappers[self._step_index]
        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        kv_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        o = wrapper.run(q_reshaped, kv_cache)
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)


class FIMultiLevelCascadeMultiStepDraftBackend:
    """Drop-in replacement for FlashInferMultiStepDraftBackend (SGLANG_FI_CASCADE_DRAFT=1)."""

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.num_draft_steps = speculative_num_steps - 1
        self.page_size = model_runner.page_size
        assert self.page_size == 1, "FI cascade draft backend assumes page_size=1"
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.device = model_runner.device
        self.max_context_len = model_runner.model_config.context_len

        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.q_data_type = model_runner.dtype
        self.kv_data_type = model_runner.kv_cache_dtype

        self.float_workspace = torch.empty(
            FLOAT_WORKSPACE_BYTES, dtype=torch.uint8, device=self.device
        )

        # Eager path: one stock wrapper per step.
        self._eager_wrappers = [
            MultiLevelCascadeAttentionWrapper(NUM_LEVELS, self.float_workspace, "NHD")
            for _ in range(self.num_draft_steps)
        ]
        self._active_wrappers: List[MultiLevelCascadeAttentionWrapper] = (
            self._eager_wrappers
        )
        self._cg_wrappers: Dict[int, List[MultiLevelCascadeAttentionWrapper]] = {}

        self.attn_backends: List[FIMultiLevelCascadeDraftAttnBackend] = [
            FIMultiLevelCascadeDraftAttnBackend(model_runner, self, i)
            for i in range(self.num_draft_steps)
        ]
        self._debug_logged = False

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan_all_steps(self, forward_batch: ForwardBatch, wrappers):
        """Build 2-level indices and run the stock plan() for every draft step."""
        num_seqs = forward_batch.batch_size
        total_branches = num_seqs * self.topk
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens

        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.to("cpu")

        # Host-side indptrs, so the stock plan() needs no device->host copy.
        kv_indptr_shared_cpu = torch.zeros(num_seqs + 1, dtype=torch.int32)
        torch.cumsum(seq_lens_cpu.to(torch.int32), dim=0, out=kv_indptr_shared_cpu[1:])
        qo_indptr_shared_cpu = torch.arange(
            0, (num_seqs + 1) * self.topk, self.topk, dtype=torch.int32
        )
        qo_indptr_unique_cpu = torch.arange(0, total_branches + 1, dtype=torch.int32)
        last_page_len_shared_cpu = torch.ones(num_seqs, dtype=torch.int32)
        last_page_len_unique_cpu = torch.ones(total_branches, dtype=torch.int32)

        total_prefix = int(kv_indptr_shared_cpu[-1])
        kv_indices_shared = torch.empty(
            max(total_prefix, 1), dtype=torch.int32, device=self.device
        )
        kv_indptr_shared_gpu = torch.empty(
            num_seqs + 1, dtype=torch.int32, device=self.device
        )
        generate_cascade_shared_kv_indices[(num_seqs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            kv_indices_shared,
            kv_indptr_shared_gpu,
            self.pool_len,
            next_power_of_2(num_seqs),
            128,
        )

        for i in range(self.num_draft_steps):
            step_offset = i + 1
            kv_indptr_unique_cpu = torch.arange(
                0, (total_branches + 1) * step_offset, step_offset, dtype=torch.int32
            )[: total_branches + 1]
            _, kv_indices_unique, _ = build_unique_indices(
                req_pool_indices,
                req_to_token,
                seq_lens,
                self.topk,
                step_offset,
                self.speculative_num_steps,
                self.page_size,
                self.device,
                self.pool_len,
            )
            wrappers[i].plan(
                qo_indptr_arr=[qo_indptr_shared_cpu, qo_indptr_unique_cpu],
                paged_kv_indptr_arr=[kv_indptr_shared_cpu, kv_indptr_unique_cpu],
                paged_kv_indices_arr=[kv_indices_shared, kv_indices_unique],
                paged_kv_last_page_len=[
                    last_page_len_shared_cpu,
                    last_page_len_unique_cpu,
                ],
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=1,
                causal=False,
                q_data_type=self.q_data_type,
                kv_data_type=self.kv_data_type,
            )

        if not self._debug_logged and os.environ.get("SGLANG_DEBUG_DRAFT_PARAMS") == "1":
            self._debug_logged = True
            print(
                f"[FI CASCADE DRAFT] num_seqs={num_seqs} topk={self.topk} "
                f"steps={self.num_draft_steps} prefix={seq_lens_cpu.tolist()} "
                f"(pid={os.getpid()})",
                flush=True,
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._active_wrappers = self._eager_wrappers
        self._plan_all_steps(forward_batch, self._eager_wrappers)

    # ------------------------------------------------------------------
    # CUDA graph
    # ------------------------------------------------------------------

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        max_branches = max_bs * self.topk
        n = self.num_draft_steps
        # Index/indptr storage shared across batch sizes, separate per step.
        self._cg_qo_indptr = [
            [torch.zeros(max_bs + 1, dtype=torch.int32, device=self.device),
             torch.zeros(max_branches + 1, dtype=torch.int32, device=self.device)]
            for _ in range(n)
        ]
        self._cg_kv_indptr = [
            [torch.zeros(max_bs + 1, dtype=torch.int32, device=self.device),
             torch.zeros(max_branches + 1, dtype=torch.int32, device=self.device)]
            for _ in range(n)
        ]
        self._cg_last_page_len = [
            [torch.ones(max_bs, dtype=torch.int32, device=self.device),
             torch.ones(max_branches, dtype=torch.int32, device=self.device)]
            for _ in range(n)
        ]
        # Level-0 indices are identical for every step of an iteration, so all
        # steps can share one buffer; level-1 indices differ per step.
        shared_indices = torch.zeros(
            max_bs * self.max_context_len, dtype=torch.int32, device=self.device
        )
        self._cg_kv_indices = [
            [shared_indices,
             torch.zeros(
                 max_branches * self.speculative_num_steps,
                 dtype=torch.int32,
                 device=self.device,
             )]
            for _ in range(n)
        ]
        self._cg_int_workspace = [
            [torch.empty(INT_WORKSPACE_BYTES, dtype=torch.uint8, device=self.device)
             for _ in range(NUM_LEVELS)]
            for _ in range(n)
        ]
        self._cg_pinned_workspace = [
            [torch.empty(INT_WORKSPACE_BYTES, dtype=torch.uint8, pin_memory=True)
             for _ in range(NUM_LEVELS)]
            for _ in range(n)
        ]
        self._cg_wrappers = {}

    def _get_cg_wrappers(self, bs: int):
        wrappers = self._cg_wrappers.get(bs)
        if wrappers is not None:
            return wrappers
        branches = bs * self.topk
        sizes = [bs, branches]
        wrappers = []
        for i in range(self.num_draft_steps):
            w = MultiLevelCascadeAttentionWrapper(
                NUM_LEVELS,
                self.float_workspace,
                "NHD",
                use_cuda_graph=True,
                qo_indptr_buf_arr=[
                    self._cg_qo_indptr[i][l][: sizes[l] + 1] for l in range(NUM_LEVELS)
                ],
                paged_kv_indptr_buf_arr=[
                    self._cg_kv_indptr[i][l][: sizes[l] + 1] for l in range(NUM_LEVELS)
                ],
                paged_kv_indices_buf_arr=[
                    self._cg_kv_indices[i][l] for l in range(NUM_LEVELS)
                ],
                paged_kv_last_page_len_buf_arr=[
                    self._cg_last_page_len[i][l][: sizes[l]] for l in range(NUM_LEVELS)
                ],
            )
            # Share int workspaces across batch sizes instead of allocating a
            # fresh 8 MB device + 8 MB pinned buffer per (bs, step, level).
            for l, pw in enumerate(w._batch_prefill_wrappers):
                pw._int_workspace_buffer = self._cg_int_workspace[i][l]
                pw._pin_memory_int_workspace_buffer = self._cg_pinned_workspace[i][l]
            wrappers.append(w)
        self._cg_wrappers[bs] = wrappers
        return wrappers

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        wrappers = self._get_cg_wrappers(forward_batch.batch_size)
        self._active_wrappers = wrappers
        self._plan_all_steps(forward_batch, wrappers)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        assert forward_batch.batch_size == bs
        wrappers = self._get_cg_wrappers(bs)
        self._active_wrappers = wrappers
        self._plan_all_steps(forward_batch, wrappers)
