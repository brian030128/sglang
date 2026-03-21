"""Cascade-aware draft decode attention backend for EAGLE speculative decoding.

Replaces FlashInferMultiStepDraftBackend. Uses CascadeBatchAttention to read
the shared prefix KV cache once and combine with each branch's unique suffix,
eliminating redundant prefix reads across topk branches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
from flashinfer.attention import CascadeBatchAttentionWrapper as CascadeBatchAttention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.cascade_index_gen import build_shared_indices, build_unique_indices

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class CascadeDraftAttnBackend(AttentionBackend):
    """Per-step attention backend using CascadeBatchAttention.

    Each instance handles one draft decode step. All instances share a single
    CascadeBatchAttentionWrapper from the parent CascadeMultiStepDraftBackend.
    Before each step's forward, update_draft_step() patches the shared
    workspace buffer with this step's kv_len/kv_end.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        cascade_attn: CascadeBatchAttention,
        step_index: int,
    ):
        super().__init__()
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.max_context_len = model_runner.model_config.context_len

        self.cascade_attn = cascade_attn  # shared reference
        self._step_index = step_index

        # Set by parent before forward
        self._step_kv_len = None
        self._step_kv_end = None
        self._plan_ready = False

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Planning is done by the parent CascadeMultiStepDraftBackend
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
        cache_loc = forward_batch.out_cache_loc

        if k is not None and v is not None and save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        # Patch the shared workspace for this step's kv_len/kv_end
        self.cascade_attn.update_draft_step(self._step_kv_len, self._step_kv_end)

        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        kv_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        o, _ = self.cascade_attn.run(
            q_reshaped,
            kv_cache,
            logits_soft_cap=layer.logit_cap if layer.logit_cap is not None else 0.0,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)


class CascadeMultiStepDraftBackend:
    """Drop-in replacement for FlashInferMultiStepDraftBackend.

    Uses 2-level cascade attention:
      Level 1: shared prefix (read once across all topk branches)
      Level 2: unique suffix (draft tokens per branch)

    Plans once per speculative round for the max draft step, then patches
    kv_len/kv_end per step via update_draft_step(). With page_size=1, the
    kernel reads up to kv_end tokens from pre-allocated kv_indices — extra
    entries for later steps are simply not accessed.

    NOTE: page_size=1 is always used for FlashInfer cascade plan calls because
    SGLang's req_to_token stores per-token physical slot IDs (not page IDs).
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.page_size = model_runner.page_size
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.device = model_runner.device

        # Single shared wrapper for all steps (non-CUDA-graph mode)
        self.shared_cascade_attn = CascadeBatchAttention(
            num_levels=2,
            kv_layout="NHD",
            device="cuda",
        )

        # N backends all referencing the shared wrapper
        self.attn_backends: List[CascadeDraftAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                CascadeDraftAttnBackend(model_runner, self.shared_cascade_attn, i)
            )

        self.max_context_len = model_runner.model_config.context_len
        self._modules_initialized = False

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Plan cascade attention once for the max draft step.

        Builds kv_indices for the max step (all draft tokens pre-loaded in
        kv_indices). Plans once via plan_for_draft(), which extracts workspace
        offsets so update_draft_step() can patch kv_len/kv_end per step without
        re-running the C++ scheduler.

        CPU tensors (kv_indptr, kv_len) are computed directly from
        forward_batch.seq_lens_cpu to avoid GPU->CPU sync.
        """
        num_seqs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens

        # Use CPU seq_lens to build CPU tensors without GPU->CPU sync
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.to("cpu")
            torch.cuda.synchronize()

        # Level 1 CPU tensors: computed directly on CPU (no GPU transfer)
        kv_len_shared_cpu = seq_lens_cpu.to(torch.int32)
        kv_indptr_shared_cpu = torch.zeros(num_seqs + 1, dtype=torch.int32, device="cpu")
        torch.cumsum(kv_len_shared_cpu, dim=0, out=kv_indptr_shared_cpu[1:])
        actual_total_prefix = int(kv_indptr_shared_cpu[-1].item())

        # Level 1 GPU tensors: kv_indices from Triton kernel (stays on GPU)
        kv_indices_shared = torch.empty(actual_total_prefix, dtype=torch.int32, device=self.device)
        kv_indptr_shared_gpu = torch.empty(num_seqs + 1, dtype=torch.int32, device=self.device)

        BLOCK_SIZE = 128
        from sglang.srt.speculative.cascade_index_gen import (
            generate_cascade_shared_kv_indices,
            next_power_of_2,
        )
        generate_cascade_shared_kv_indices[(num_seqs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            kv_indices_shared,
            kv_indptr_shared_gpu,
            self.pool_len,
            next_power_of_2(num_seqs),
            BLOCK_SIZE,
        )

        # Level 2 CPU tensors: deterministic, computed on CPU
        max_step_offset = self.speculative_num_steps - 1
        total_branches = num_seqs * self.topk
        kv_len_unique_cpu = torch.full(
            (total_branches,), max_step_offset, dtype=torch.int32, device="cpu"
        )
        kv_indptr_unique_cpu = torch.arange(
            0, (total_branches + 1) * max_step_offset, max_step_offset,
            dtype=torch.int32, device="cpu",
        )[:total_branches + 1]

        # Level 2 GPU tensors: kv_indices from Triton kernel
        kv_indptr_unique, kv_indices_unique, _ = build_unique_indices(
            req_pool_indices,
            req_to_token,
            seq_lens,
            self.topk,
            max_step_offset,
            self.speculative_num_steps,
            self.page_size,
            self.device,
            self.pool_len,
        )

        # qo_indptr: deterministic, built on CPU
        qo_indptr_shared_cpu = torch.arange(
            0, (num_seqs + 1) * self.topk, self.topk,
            dtype=torch.int32, device="cpu",
        )
        qo_indptr_unique_cpu = torch.arange(
            0, total_branches + 1, dtype=torch.int32, device="cpu",
        )

        # No torch.cuda.synchronize() needed — all CPU tensors computed on CPU.
        # GPU tensors (kv_indices) are passed directly to the kernel.

        # Plan once for the max step
        backend = self.attn_backends[0]  # all share same config
        first_call = not self._modules_initialized
        self.shared_cascade_attn.plan_for_draft(
            max_draft_depth=max_step_offset,
            first_call=first_call,
            qo_indptr_host_arr=[qo_indptr_shared_cpu, qo_indptr_unique_cpu],
            kv_indptr_host_arr=[kv_indptr_shared_cpu, kv_indptr_unique_cpu],
            kv_indices_arr=[kv_indices_shared, kv_indices_unique],
            kv_len_host_arr=[kv_len_shared_cpu, kv_len_unique_cpu],
            num_qo_heads=backend.num_qo_heads,
            num_kv_heads=backend.num_kv_heads,
            head_dim_qk=backend.head_dim,
            head_dim_vo=backend.head_dim,
            page_size=1,  # SGLang uses per-token slot IDs, not page IDs
            causal=False,
            sm_scale=None,
            q_data_type=backend.q_data_type,
            kv_data_type=backend.data_type,
        )
        self._modules_initialized = True

        # Set per-step kv_len/kv_end for update_draft_step().
        # Non-causal level 2: kv_len_for_work = kv_len_level + qo_len = step_offset + 1
        # kv_end = effective_kv_len = step_offset
        for i in range(self.speculative_num_steps - 1):
            step_offset = i + 1
            self.attn_backends[i]._step_kv_len = step_offset + 1
            self.attn_backends[i]._step_kv_end = step_offset
            self.attn_backends[i]._plan_ready = True

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Pre-allocate buffers for CUDA graph capture."""
        max_branches = max_bs * self.topk
        max_shared_pages = max_bs * self.max_context_len
        max_unique_pages = max_branches * self.speculative_num_steps
        max_total_pages = max_shared_pages + max_unique_pages

        self.cuda_graph_kv_indices_buf = torch.zeros(
            max_total_pages, dtype=torch.int32, device="cuda"
        )

        # For CUDA graph mode, create separate wrappers per step
        # (workspace buffers are captured in the graph)
        for i in range(self.speculative_num_steps - 1):
            old = self.attn_backends[i]
            self.attn_backends[i] = CascadeDraftAttnBackend.__new__(CascadeDraftAttnBackend)
            new = self.attn_backends[i]
            new.num_qo_heads = old.num_qo_heads
            new.num_kv_heads = old.num_kv_heads
            new.head_dim = old.head_dim
            new.data_type = old.data_type
            new.q_data_type = old.q_data_type
            new.max_context_len = old.max_context_len
            new._step_index = i
            new._step_kv_len = None
            new._step_kv_end = None
            new._plan_ready = False
            new.cascade_attn = CascadeBatchAttention(
                num_levels=2,
                kv_layout="NHD",
                device="cuda",
                use_cuda_graph=True,
                kv_indices_buffer=self.cuda_graph_kv_indices_buf,
            )
        self._modules_initialized = False

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        """Called during CUDA graph capture."""
        self._init_forward_metadata_per_step(forward_batch, use_fast_plan=False)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        """Called during CUDA graph replay."""
        self._init_forward_metadata_per_step(forward_batch, use_fast_plan=True)

    def _init_forward_metadata_per_step(self, forward_batch: ForwardBatch, use_fast_plan: bool):
        """CUDA graph path: plan each step separately (each has its own wrapper)."""
        num_seqs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens

        max_total_prefix = num_seqs * self.max_context_len
        kv_indptr_shared, kv_indices_shared, kv_len_shared = build_shared_indices(
            req_pool_indices, req_to_token, seq_lens, self.device,
            self.pool_len, max_total_prefix=max_total_prefix,
        )

        qo_indptr_shared_cpu = torch.arange(
            0, (num_seqs + 1) * self.topk, self.topk,
            dtype=torch.int32, device="cpu",
        )
        total_branches = num_seqs * self.topk
        qo_indptr_unique_cpu = torch.arange(
            0, total_branches + 1, dtype=torch.int32, device="cpu",
        )

        all_kv_indices_unique = []
        all_kv_indptr_unique = []
        all_kv_len_unique = []
        for i in range(self.speculative_num_steps - 1):
            step_offset = i + 1
            kv_indptr_unique, kv_indices_unique, kv_len_unique = build_unique_indices(
                req_pool_indices, req_to_token, seq_lens, self.topk,
                step_offset, self.speculative_num_steps, self.page_size,
                self.device, self.pool_len,
            )
            all_kv_indices_unique.append(kv_indices_unique)
            all_kv_indptr_unique.append(kv_indptr_unique)
            all_kv_len_unique.append(kv_len_unique)

        kv_indptr_shared_cpu = kv_indptr_shared.to("cpu", non_blocking=True)
        kv_len_shared_cpu = kv_len_shared.to("cpu", non_blocking=True)
        all_kv_indptr_unique_cpu = [t.to("cpu", non_blocking=True) for t in all_kv_indptr_unique]
        all_kv_len_unique_cpu = [t.to("cpu", non_blocking=True) for t in all_kv_len_unique]
        torch.cuda.synchronize()

        actual_total_prefix = int(kv_indptr_shared_cpu[-1].item())
        kv_indices_shared = kv_indices_shared[:actual_total_prefix]

        for i in range(self.speculative_num_steps - 1):
            backend = self.attn_backends[i]
            plan_kwargs = dict(
                num_qo_heads=backend.num_qo_heads,
                num_kv_heads=backend.num_kv_heads,
                head_dim_qk=backend.head_dim,
                head_dim_vo=backend.head_dim,
                page_size=1,  # SGLang uses per-token slot IDs
                causal=False,
                sm_scale=None,
                q_data_type=backend.q_data_type,
                kv_data_type=backend.data_type,
            )
            if use_fast_plan:
                backend.cascade_attn.fast_cascade_plan(
                    qo_indptr_host_arr=[qo_indptr_shared_cpu, qo_indptr_unique_cpu],
                    kv_indptr_host_arr=[kv_indptr_shared_cpu, all_kv_indptr_unique_cpu[i]],
                    kv_indices_arr=[kv_indices_shared, all_kv_indices_unique[i]],
                    kv_len_host_arr=[kv_len_shared_cpu, all_kv_len_unique_cpu[i]],
                    **plan_kwargs,
                )
            else:
                backend.cascade_attn.plan(
                    qo_indptr_arr=[qo_indptr_shared_cpu, qo_indptr_unique_cpu],
                    kv_indptr_arr=[kv_indptr_shared_cpu, all_kv_indptr_unique_cpu[i]],
                    kv_indices_arr=[kv_indices_shared, all_kv_indices_unique[i]],
                    kv_len_arr=[kv_len_shared_cpu, all_kv_len_unique_cpu[i]],
                    **plan_kwargs,
                )
            backend._step_kv_len = (i + 1) + 1
            backend._step_kv_end = i + 1
            backend._plan_ready = True
