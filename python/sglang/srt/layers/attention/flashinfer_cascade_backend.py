"""Cascade-aware draft decode attention backend for speculative decoding.

Mirrors FlashInferMultiStepDraftBackend but uses CascadeBatchAttentionWrapper
to read the shared prefix KV cache once and combine with each branch's unique
suffix, eliminating redundant prefix reads across topk branches.

Non-CUDA-graph path: one shared wrapper planned once via plan_for_draft() at
max draft depth. Each step patches kv_len/kv_end via update_draft_step()
(2 trivial GPU buffer writes) instead of re-running the full C++ scheduler.

CUDA-graph path: three-phase planning:
  - Capture: full cascade_plan on last wrapper, copy workspace to others,
    patch per-step Level 2 kv_len/kv_end, save replay layout.
  - Replay (fast): skip cascade_plan entirely. Regenerate kv_indices via
    Triton, patch Level 0 kv_len/kv_end and Level 1 kv_indptr directly
    in each wrapper's workspace buffer (~0.1ms vs ~3.5ms per iteration).
  - Replay (fallback): re-run full plan if chunk count changes (~every
    kv_limit/accept_rate iterations; rare).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, List

import torch
from flashinfer.attention import CascadeBatchAttentionWrapper as CascadeBatchAttention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.cascade_index_gen import (
    build_shared_indices,
    build_unique_indices,
    generate_cascade_shared_kv_indices,
    next_power_of_2,
)
from sglang.srt.speculative.draft_utils import nvtx_pop, nvtx_push

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class CascadeDraftAttnBackend(AttentionBackend):
    """Per-step attention backend using CascadeBatchAttention.

    In non-CUDA-graph mode, all steps share one wrapper. Each backend has a
    step_index and calls update_draft_step before its first layer.
    In CUDA-graph mode, each backend owns its own wrapper.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        cascade_attn: CascadeBatchAttention,
        step_index: int = None,
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

        self.cascade_attn = cascade_attn
        self._step_index = step_index
        self._step_updated = False

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Planning is done by the parent CascadeMultiStepDraftBackend
        pass

    _debug_decode_logged = False

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

        # Patch shared wrapper for this step (once, before first layer)
        if self._step_index is not None and not self._step_updated:
            step_offset = self._step_index + 1  # suffix length
            self.cascade_attn.update_draft_step(
                step_kv_len=step_offset + 1,  # kv_len + qo_len (non-causal)
                step_kv_end=step_offset,       # effective kv_len
            )
            self._step_updated = True

        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        kv_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        if not CascadeDraftAttnBackend._debug_decode_logged and os.environ.get("SGLANG_DEBUG_DRAFT_PARAMS") == "1":
            CascadeDraftAttnBackend._debug_decode_logged = True
            if isinstance(kv_cache, tuple):
                kv_shapes = [t.shape for t in kv_cache]
            else:
                kv_shapes = kv_cache.shape
            print(f"\n[CASCADE DRAFT] forward_decode (pid={os.getpid()}, layer={layer.layer_id})", flush=True)
            print(f"  q.shape={q_reshaped.shape} (num_qo_heads={layer.tp_q_head_num}, head_dim={layer.head_dim})", flush=True)
            print(f"  kv_data shapes={kv_shapes}", flush=True)
            print(f"  num_kv_heads={layer.tp_k_head_num}, scaling={layer.scaling}", flush=True)

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

    Non-CUDA-graph path: one shared wrapper planned once via plan_for_draft()
    at max draft depth (1 cascade_plan call). Each step patches kv_len/kv_end
    via update_draft_step() (2 GPU buffer writes). Total per iteration:
    2 Triton kernels + 1 cascade_plan (vs N-1 Triton kernels + N-1 cascade_plan).

    CUDA-graph path: per-step wrappers with plan-once-copy. Plans cascade_plan
    once on the last wrapper (max depth), GPU memcpy workspace to others, then
    patches Level 2 kv_len/kv_end per step. Same 1 cascade_plan call.
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
        self.max_context_len = model_runner.model_config.context_len

        # One shared wrapper for all steps (non-CUDA-graph mode).
        # plan_for_draft() plans once at max depth; update_draft_step() patches per step.
        self._shared_wrapper = CascadeBatchAttention(
            num_levels=2,
            kv_layout="NHD",
            device="cuda",
        )
        self.attn_backends: List[CascadeDraftAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                CascadeDraftAttnBackend(model_runner, self._shared_wrapper, step_index=i)
            )

        self._modules_initialized = False
        self._debug_logged = False

    def _build_cascade_indices_and_plan(
        self,
        forward_batch: ForwardBatch,
        call_fn: Callable,
    ):
        """Build cascade indices and call plan per step.

        Generates Level 1 (shared prefix) and Level 2 (unique suffix) indices
        using Triton kernels, then calls call_fn(i, plan_args) for each step.
        CPU tensors are computed directly on CPU — no GPU->CPU sync.
        """
        num_seqs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens
        total_branches = num_seqs * self.topk

        # --- CPU tensors (no GPU->CPU sync) ---

        # Use CPU seq_lens to avoid sync
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.to("cpu")
            torch.cuda.synchronize()

        # Level 1 CPU: deterministic from seq_lens
        kv_len_shared_cpu = seq_lens_cpu.to(torch.int32)
        kv_indptr_shared_cpu = torch.zeros(
            num_seqs + 1, dtype=torch.int32, device="cpu"
        )
        torch.cumsum(kv_len_shared_cpu, dim=0, out=kv_indptr_shared_cpu[1:])

        # qo_indptr: deterministic
        qo_indptr_shared_cpu = torch.arange(
            0, (num_seqs + 1) * self.topk, self.topk,
            dtype=torch.int32, device="cpu",
        )
        qo_indptr_unique_cpu = torch.arange(
            0, total_branches + 1, dtype=torch.int32, device="cpu",
        )

        # --- GPU tensors: shared prefix indices (same for all steps) ---
        actual_total_prefix = int(kv_indptr_shared_cpu[-1].item())
        kv_indices_shared = torch.empty(
            actual_total_prefix, dtype=torch.int32, device=self.device
        )
        kv_indptr_shared_gpu = torch.empty(
            num_seqs + 1, dtype=torch.int32, device=self.device
        )

        from sglang.srt.speculative.cascade_index_gen import (
            generate_cascade_shared_kv_indices,
            next_power_of_2,
        )

        nvtx_push("cascade/shared_indices")
        generate_cascade_shared_kv_indices[(num_seqs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            kv_indices_shared,
            kv_indptr_shared_gpu,
            self.pool_len,
            next_power_of_2(num_seqs),
            128,  # BLOCK_SIZE
        )
        nvtx_pop()

        _do_debug = not self._debug_logged and os.environ.get("SGLANG_DEBUG_DRAFT_PARAMS") == "1" and int(kv_indptr_shared_cpu[-1].item()) > total_branches
        if _do_debug:
            self._debug_logged = True
            print(f"\n[CASCADE DRAFT] _build_cascade_indices_and_plan (pid={os.getpid()})", flush=True)
            print(f"  num_seqs={num_seqs}, topk={self.topk}, total_branches={total_branches}", flush=True)
            print(f"  speculative_num_steps={self.speculative_num_steps}, page_size={self.page_size}", flush=True)
            print(f"  seq_lens={seq_lens_cpu.tolist()}", flush=True)
            print(f"  SHARED: kv_indices_shared.shape={kv_indices_shared.shape}, "
                  f"first20={kv_indices_shared[:20].cpu().tolist()}, "
                  f"last10={kv_indices_shared[-10:].cpu().tolist()}", flush=True)
            print(f"  SHARED: kv_indptr={kv_indptr_shared_cpu.tolist()}", flush=True)
            print(f"  SHARED: kv_len={kv_len_shared_cpu.tolist()}", flush=True)
            print(f"  SHARED: qo_indptr={qo_indptr_shared_cpu.tolist()}", flush=True)

        # --- Per-step: build unique indices and plan ---
        for i in range(self.speculative_num_steps - 1):
            step_offset = i + 1

            # Level 2 CPU: deterministic from geometry
            kv_len_unique_cpu = torch.full(
                (total_branches,), step_offset, dtype=torch.int32, device="cpu"
            )
            kv_indptr_unique_cpu = torch.arange(
                0, (total_branches + 1) * step_offset, step_offset,
                dtype=torch.int32, device="cpu",
            )[: total_branches + 1]

            # Level 2 GPU: unique suffix indices from Triton
            nvtx_push(f"cascade/step_{i}_unique_indices")
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
            nvtx_pop()

            if _do_debug:
                print(f"  UNIQUE step {i}: kv_indices_unique.shape={kv_indices_unique.shape}, "
                      f"values={kv_indices_unique.cpu().tolist()}", flush=True)
                print(f"  UNIQUE step {i}: kv_indptr={kv_indptr_unique_cpu.tolist()}", flush=True)
                print(f"  UNIQUE step {i}: kv_len={kv_len_unique_cpu.tolist()}", flush=True)
                print(f"  UNIQUE step {i}: qo_indptr={qo_indptr_unique_cpu.tolist()}", flush=True)

            call_fn(
                i,
                qo_indptr_shared_cpu=qo_indptr_shared_cpu,
                qo_indptr_unique_cpu=qo_indptr_unique_cpu,
                kv_indptr_shared_cpu=kv_indptr_shared_cpu,
                kv_indptr_unique_cpu=kv_indptr_unique_cpu,
                kv_indices_shared=kv_indices_shared,
                kv_indices_unique=kv_indices_unique,
                kv_len_shared_cpu=kv_len_shared_cpu,
                kv_len_unique_cpu=kv_len_unique_cpu,
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        first_call = not self._modules_initialized

        # Reset step_updated flags so each step calls update_draft_step once
        for backend in self.attn_backends:
            backend._step_updated = False

        nvtx_push("cascade/plan_for_draft")
        self._plan_once_for_all_steps(forward_batch, first_call)
        nvtx_pop()
        self._modules_initialized = True

    def _plan_once_for_all_steps(self, forward_batch: ForwardBatch, first_call: bool):
        """Plan once at max draft depth. Steps patch via update_draft_step.

        Generates shared prefix indices (1 Triton kernel) + unique suffix
        indices at max depth (1 Triton kernel) + 1 cascade_plan call.
        Replaces N-1 Triton kernels + N-1 cascade_plan calls.
        """
        num_seqs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens
        total_branches = num_seqs * self.topk
        max_depth = self.speculative_num_steps - 1  # max suffix length

        # --- CPU tensors (no GPU->CPU sync) ---
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.to("cpu")
            torch.cuda.synchronize()

        # Level 1 CPU
        kv_len_shared_cpu = seq_lens_cpu.to(torch.int32)
        kv_indptr_shared_cpu = torch.zeros(
            num_seqs + 1, dtype=torch.int32, device="cpu"
        )
        torch.cumsum(kv_len_shared_cpu, dim=0, out=kv_indptr_shared_cpu[1:])

        # qo_indptr
        qo_indptr_shared_cpu = torch.arange(
            0, (num_seqs + 1) * self.topk, self.topk,
            dtype=torch.int32, device="cpu",
        )
        qo_indptr_unique_cpu = torch.arange(
            0, total_branches + 1, dtype=torch.int32, device="cpu",
        )

        # Level 2 CPU: max depth
        kv_len_unique_cpu = torch.full(
            (total_branches,), max_depth, dtype=torch.int32, device="cpu"
        )
        kv_indptr_unique_cpu = torch.arange(
            0, (total_branches + 1) * max_depth, max_depth,
            dtype=torch.int32, device="cpu",
        )[: total_branches + 1]

        # --- GPU: shared prefix indices (1 Triton kernel) ---
        actual_total_prefix = int(kv_indptr_shared_cpu[-1].item())
        kv_indices_shared = torch.empty(
            actual_total_prefix, dtype=torch.int32, device=self.device
        )
        kv_indptr_shared_gpu = torch.empty(
            num_seqs + 1, dtype=torch.int32, device=self.device
        )

        from sglang.srt.speculative.cascade_index_gen import (
            generate_cascade_shared_kv_indices,
            next_power_of_2,
        )

        nvtx_push("cascade/shared_indices")
        generate_cascade_shared_kv_indices[(num_seqs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            kv_indices_shared,
            kv_indptr_shared_gpu,
            self.pool_len,
            next_power_of_2(num_seqs),
            128,  # BLOCK_SIZE
        )
        nvtx_pop()

        # --- GPU: unique suffix indices at max depth (1 Triton kernel) ---
        nvtx_push("cascade/unique_indices_max")
        _, kv_indices_unique, _ = build_unique_indices(
            req_pool_indices,
            req_to_token,
            seq_lens,
            self.topk,
            max_depth,
            self.speculative_num_steps,
            self.page_size,
            self.device,
            self.pool_len,
        )
        nvtx_pop()

        # --- plan_for_draft: 1 cascade_plan call ---
        backend = self.attn_backends[0]
        self._shared_wrapper.plan_for_draft(
            max_draft_depth=max_depth,
            first_call=first_call,
            qo_indptr_host_arr=[qo_indptr_shared_cpu, qo_indptr_unique_cpu],
            kv_indptr_host_arr=[kv_indptr_shared_cpu, kv_indptr_unique_cpu],
            kv_indices_arr=[kv_indices_shared, kv_indices_unique],
            kv_len_host_arr=[kv_len_shared_cpu, kv_len_unique_cpu],
            num_qo_heads=backend.num_qo_heads,
            num_kv_heads=backend.num_kv_heads,
            head_dim_qk=backend.head_dim,
            head_dim_vo=backend.head_dim,
            page_size=1,
            causal=False,
            sm_scale=None,
            q_data_type=backend.q_data_type,
            kv_data_type=backend.data_type,
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        max_branches = max_bs * self.topk
        max_shared_pages = max_bs * self.max_context_len
        max_unique_pages = max_branches * self.speculative_num_steps
        max_total_pages = max_shared_pages + max_unique_pages

        self.cuda_graph_kv_indices_buf = torch.zeros(
            max_total_pages, dtype=torch.int32, device="cuda"
        )

        # CUDA graph mode: per-step wrappers (can't use plan_for_draft because
        # single-layer EAGLE captures the entire multi-step loop in ONE graph,
        # making it impossible to insert update_draft_step between steps).
        for i in range(self.speculative_num_steps - 1):
            old = self.attn_backends[i]
            new = CascadeDraftAttnBackend.__new__(CascadeDraftAttnBackend)
            new.num_qo_heads = old.num_qo_heads
            new.num_kv_heads = old.num_kv_heads
            new.head_dim = old.head_dim
            new.data_type = old.data_type
            new.q_data_type = old.q_data_type
            new.max_context_len = old.max_context_len
            new._step_index = None  # disable update_draft_step in CUDA graph mode
            new._step_updated = False
            new.cascade_attn = CascadeBatchAttention(
                num_levels=2,
                kv_layout="NHD",
                device="cuda",
                use_cuda_graph=True,
                kv_indices_buffer=self.cuda_graph_kv_indices_buf,
            )
            self.attn_backends[i] = new
        self._modules_initialized = False

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        self._cuda_graph_plan(forward_batch, use_fast_plan=False)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        self._cuda_graph_plan(forward_batch, use_fast_plan=True)

    def _cuda_graph_plan(self, forward_batch: ForwardBatch, use_fast_plan: bool):
        """Router: try fast replay on replay path, fall back to full plan."""
        if use_fast_plan and hasattr(self, "_replay_kv_limit"):
            if self._cuda_graph_fast_replay(forward_batch):
                return
        self._cuda_graph_full_plan(forward_batch, use_fast_plan)

    def _cuda_graph_full_plan(self, forward_batch: ForwardBatch, use_fast_plan: bool):
        """Full plan: plan-once-copy + save replay layout for fast_replay.

        Called during CUDA graph capture and as fallback when chunk count changes.
        Plans cascade_plan on last wrapper, copies workspace to others, patches
        per-step Level 2 kv_len/kv_end, then extracts replay layout so subsequent
        iterations can skip the C++ scheduler entirely.
        """
        num_seqs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens
        total_branches = num_seqs * self.topk
        max_depth = self.speculative_num_steps - 1

        # --- CPU tensors (no GPU->CPU sync) ---
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.to("cpu")
            torch.cuda.synchronize()

        kv_len_shared_cpu = seq_lens_cpu.to(torch.int32)
        kv_indptr_shared_cpu = torch.zeros(
            num_seqs + 1, dtype=torch.int32, device="cpu"
        )
        torch.cumsum(kv_len_shared_cpu, dim=0, out=kv_indptr_shared_cpu[1:])

        qo_indptr_shared_cpu = torch.arange(
            0, (num_seqs + 1) * self.topk, self.topk,
            dtype=torch.int32, device="cpu",
        )
        qo_indptr_unique_cpu = torch.arange(
            0, total_branches + 1, dtype=torch.int32, device="cpu",
        )

        # Level 2 CPU at max depth
        kv_len_unique_cpu = torch.full(
            (total_branches,), max_depth, dtype=torch.int32, device="cpu"
        )
        kv_indptr_unique_cpu = torch.arange(
            0, (total_branches + 1) * max_depth, max_depth,
            dtype=torch.int32, device="cpu",
        )[: total_branches + 1]

        # --- GPU: shared prefix indices (1 Triton kernel) ---
        actual_total_prefix = int(kv_indptr_shared_cpu[-1].item())
        kv_indices_shared = torch.empty(
            actual_total_prefix, dtype=torch.int32, device=self.device
        )

        nvtx_push("cascade/shared_indices")
        generate_cascade_shared_kv_indices[(num_seqs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            kv_indices_shared,
            torch.empty(num_seqs + 1, dtype=torch.int32, device=self.device),
            self.pool_len,
            next_power_of_2(num_seqs),
            128,
        )
        nvtx_pop()

        # --- GPU: unique suffix indices at max depth (1 Triton kernel) ---
        nvtx_push("cascade/unique_indices_max")
        _, kv_indices_unique, _ = build_unique_indices(
            req_pool_indices,
            req_to_token,
            seq_lens,
            self.topk,
            max_depth,
            self.speculative_num_steps,
            self.page_size,
            self.device,
            self.pool_len,
        )
        nvtx_pop()

        # --- Plan ONCE on last wrapper ---
        last_idx = len(self.attn_backends) - 1
        last_backend = self.attn_backends[last_idx]
        last_wrapper = last_backend.cascade_attn

        plan_kwargs = dict(
            num_qo_heads=last_backend.num_qo_heads,
            num_kv_heads=last_backend.num_kv_heads,
            head_dim_qk=last_backend.head_dim,
            head_dim_vo=last_backend.head_dim,
            page_size=1,
            causal=False,
            sm_scale=None,
            q_data_type=last_backend.q_data_type,
            kv_data_type=last_backend.data_type,
        )

        nvtx_push("cascade/cuda_graph_plan")
        if use_fast_plan:
            last_wrapper.fast_cascade_plan(
                qo_indptr_host_arr=[qo_indptr_shared_cpu, qo_indptr_unique_cpu],
                kv_indptr_host_arr=[kv_indptr_shared_cpu, kv_indptr_unique_cpu],
                kv_indices_arr=[kv_indices_shared, kv_indices_unique],
                kv_len_host_arr=[kv_len_shared_cpu, kv_len_unique_cpu],
                **plan_kwargs,
            )
        else:
            last_wrapper.plan(
                qo_indptr_arr=[qo_indptr_shared_cpu, qo_indptr_unique_cpu],
                kv_indptr_arr=[kv_indptr_shared_cpu, kv_indptr_unique_cpu],
                kv_indices_arr=[kv_indices_shared, kv_indices_unique],
                kv_len_arr=[kv_len_shared_cpu, kv_len_unique_cpu],
                **plan_kwargs,
            )

        # --- Extract field offsets from plan_info ---
        # plan_info layout: [num_blks_x(0), num_blks_y(1),
        #   task0(2-13, 12 fields), task1(14-25, 12 fields), shared(26+)]
        # Per-task fields: q_indptr(0), kv_indptr(1), partial_indptr(2),
        #   q_len(3), kv_len(4), q_start(5), kv_start(6), kv_end(7),
        #   kv_head_idx(8), work_indptr(9), cascade_num_kv_chunks(10),
        #   cascade_kv_chunk_idx(11)
        TASK0_BASE = 2
        TASK1_BASE = 2 + 12  # = 14
        SHARED_BASE = 2 + 12 + 12  # = 26

        page_locked_buf = last_wrapper.page_locked_int_workspace_buffer.view(
            torch.int32
        )
        num_clusters = last_wrapper._plan_info[1]

        # Task 1 (Level 1 / unique suffix) offsets
        l1_kv_len_start = last_wrapper._plan_info[TASK1_BASE + 4] // 4
        l1_kv_end_start = last_wrapper._plan_info[TASK1_BASE + 7] // 4
        l1_kv_indptr_start = last_wrapper._plan_info[TASK1_BASE + 1] // 4
        l1_work_indptr_start = last_wrapper._plan_info[TASK1_BASE + 9] // 4
        total_works_l1 = int(page_locked_buf[l1_work_indptr_start + num_clusters])

        # Level 2 indices (Task 1 items with kv_indptr >= shared offset)
        l1_kv_indptrs = page_locked_buf[
            l1_kv_indptr_start : l1_kv_indptr_start + total_works_l1
        ]
        level2_offset = kv_indices_shared.shape[0]
        level2_mask = l1_kv_indptrs >= level2_offset
        level2_indices = torch.where(level2_mask)[0].to(device=self.device)

        # --- Copy workspace + patch each step ---
        for i in range(len(self.attn_backends)):
            wrapper = self.attn_backends[i].cascade_attn
            if i < last_idx:
                # Copy workspace buffer and plan metadata from last wrapper
                wrapper.int_workspace_buffer.copy_(last_wrapper.int_workspace_buffer)
                wrapper._plan_info = list(last_wrapper._plan_info)
                wrapper._page_size = last_wrapper._page_size
                wrapper._sm_scale = last_wrapper._sm_scale
                wrapper._mask_mode = last_wrapper._mask_mode
                wrapper._num_qo_heads = last_wrapper._num_qo_heads
                wrapper._num_kv_heads = last_wrapper._num_kv_heads
                # _kv_indices_buf is the shared buffer (written by last wrapper's plan)
                wrapper._kv_indices = wrapper._kv_indices_buf
                if not use_fast_plan:
                    wrapper.module = last_wrapper.module

            # Patch Level 2 kv_len/kv_end for this step's suffix length
            step_offset = i + 1
            buf = wrapper.int_workspace_buffer.view(torch.int32)
            buf[l1_kv_len_start + level2_indices] = step_offset + 1
            buf[l1_kv_end_start + level2_indices] = step_offset

        nvtx_pop()
        self._modules_initialized = True

        # --- Save replay layout for fast_replay ---
        # The C++ scheduler assigns (level, request) pairs to Task 0 or Task 1
        # based on packed_qo_len. Level 0 (shared prefix) and Level 1 (unique
        # suffix) items can end up in EITHER task. We scan both tasks to find
        # Level 0 items (kv_indptr < shared_offset) and Level 1 items
        # (kv_indptr >= shared_offset) regardless of task assignment.

        # kv_limit from the shared section of plan_info
        # plan_info[SHARED_BASE+0] = len_kv_chunk byte offset
        # page_locked at that offset stores [kv_limit_task0, kv_limit_task1]
        len_kv_chunk_byte_offset = last_wrapper._plan_info[SHARED_BASE]
        # Use the max across both tasks (the one that actually chunks the prefix)
        kv_limit_t0 = int(page_locked_buf[len_kv_chunk_byte_offset // 4])
        kv_limit_t1 = int(page_locked_buf[len_kv_chunk_byte_offset // 4 + 1])
        kv_limit = max(kv_limit_t0, kv_limit_t1)

        # Scan both tasks and build a combined index tensor for single-scatter
        # per wrapper. This batches all Level 0 and Level 1 patches into one
        # scatter operation to minimize CUDA kernel launch overhead.
        idx_parts = []  # absolute positions in workspace buffer
        n_l0_kv_len = 0  # count of L0 kv_len entries
        n_l0_kv_end = 0  # count of L0 kv_end (last-chunk) entries
        l1_bases = []  # base kv_indptr values for L1 items

        for task in range(2):
            task_base = 2 + task * 12
            t_work_indptr_s = last_wrapper._plan_info[task_base + 9] // 4
            t_total = int(page_locked_buf[t_work_indptr_s + num_clusters])
            if t_total == 0:
                continue

            t_kv_indptr_s = last_wrapper._plan_info[task_base + 1] // 4
            t_kv_indptrs = page_locked_buf[t_kv_indptr_s : t_kv_indptr_s + t_total]

            # Level 0 items: kv_indptr < shared_offset (level_offsets[0] = 0)
            l0_mask = t_kv_indptrs < level2_offset
            if l0_mask.any():
                t_kv_len_s = last_wrapper._plan_info[task_base + 4] // 4
                t_kv_end_s = last_wrapper._plan_info[task_base + 7] // 4
                t_kv_start_s = last_wrapper._plan_info[task_base + 6] // 4
                l0_idx = torch.where(l0_mask)[0]

                # L0 kv_len: all Level 0 items
                idx_parts.append(torch.tensor(t_kv_len_s, device="cpu") + l0_idx)
                n_l0_kv_len += l0_idx.numel()

                # L0 kv_end: only last-chunk items (partial chunks)
                l0_kv_end = page_locked_buf[t_kv_end_s + l0_idx]
                l0_kv_start = page_locked_buf[t_kv_start_s + l0_idx]
                chunk_sizes = l0_kv_end - l0_kv_start
                last_mask = chunk_sizes < kv_limit
                if last_mask.any():
                    last_chunk = l0_idx[last_mask]
                    idx_parts.append(
                        torch.tensor(t_kv_end_s, device="cpu") + last_chunk
                    )
                    n_l0_kv_end += last_chunk.numel()

            # Level 1 items: kv_indptr >= shared_offset
            l1_mask = t_kv_indptrs >= level2_offset
            if l1_mask.any():
                l1_idx = torch.where(l1_mask)[0]
                idx_parts.append(
                    torch.tensor(t_kv_indptr_s, device="cpu") + l1_idx
                )
                l1_base = (t_kv_indptrs[l1_idx] - actual_total_prefix).clone()
                l1_bases.append(l1_base)

        # Combined index tensor (GPU, used for single scatter per wrapper)
        if idx_parts:
            all_indices_cpu = torch.cat(idx_parts)
            self._replay_all_indices = all_indices_cpu.to(
                device=self.device, dtype=torch.long
            )
        else:
            self._replay_all_indices = torch.empty(
                0, dtype=torch.long, device=self.device
            )

        # Layout: [l0_kv_len (n_l0_kv_len) | l0_kv_end (n_l0_kv_end) | l1_kv_indptr]
        self._replay_n_l0_kv_len = n_l0_kv_len
        self._replay_n_l0_kv_end = n_l0_kv_end
        self._replay_l1_kv_indptr_base = (
            torch.cat(l1_bases).to(device=self.device, dtype=torch.int32)
            if l1_bases
            else torch.empty(0, dtype=torch.int32, device=self.device)
        )
        # Pre-allocate values buffer
        self._replay_all_values = torch.empty(
            self._replay_all_indices.numel(),
            dtype=torch.int32,
            device=self.device,
        )

        max_prefix = int(seq_lens_cpu.max().item())
        self._replay_kv_limit = kv_limit
        self._replay_num_chunks = -(-max_prefix // kv_limit)  # ceil_div
        self._replay_unique_len = kv_indices_unique.shape[0]

    def _cuda_graph_fast_replay(self, forward_batch: ForwardBatch) -> bool:
        """Fast replay: skip cascade_plan, patch workspace directly.

        Regenerates kv_indices via Triton kernels, then patches only the
        workspace fields that change between iterations:
          - Level 0 kv_len (all items): prefix_len + topk
          - Level 0 kv_end (last-chunk items only): prefix_len
          - Level 1 kv_indptr (all items): base + new_shared_len

        Returns True on success, False to fall back to full plan.
        Falls back when chunk count changes (rare: ~every kv_limit tokens).
        """
        num_seqs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens

        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.to("cpu")
            torch.cuda.synchronize()

        # Check chunk count stability
        max_prefix = int(seq_lens_cpu.max().item())
        kv_limit = self._replay_kv_limit
        new_num_chunks = -(-max_prefix // kv_limit)  # ceil_div
        if new_num_chunks != self._replay_num_chunks:
            return False

        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        max_depth = self.speculative_num_steps - 1

        # --- Regenerate kv_indices via Triton ---
        actual_total_prefix = int(seq_lens_cpu.sum().item())
        kv_indices_shared = torch.empty(
            actual_total_prefix, dtype=torch.int32, device=self.device
        )

        nvtx_push("cascade/fast_replay")
        generate_cascade_shared_kv_indices[(num_seqs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            kv_indices_shared,
            torch.empty(num_seqs + 1, dtype=torch.int32, device=self.device),
            self.pool_len,
            next_power_of_2(num_seqs),
            128,
        )

        _, kv_indices_unique, _ = build_unique_indices(
            req_pool_indices,
            req_to_token,
            seq_lens,
            self.topk,
            max_depth,
            self.speculative_num_steps,
            self.page_size,
            self.device,
            self.pool_len,
        )

        # --- Copy indices to shared kv_indices buffer ---
        new_shared_len = kv_indices_shared.shape[0]
        unique_len = self._replay_unique_len
        kv_buf = self.cuda_graph_kv_indices_buf
        kv_buf[:new_shared_len].copy_(kv_indices_shared, non_blocking=True)
        kv_buf[new_shared_len : new_shared_len + unique_len].copy_(
            kv_indices_unique, non_blocking=True
        )

        # --- Patch workspace in each wrapper (single scatter per wrapper) ---
        # Build values tensor: [l0_kv_len... | l0_kv_end... | l1_kv_indptr...]
        new_prefix_i32 = seq_lens[0].to(torch.int32)
        vals = self._replay_all_values
        s0 = self._replay_n_l0_kv_len
        s1 = s0 + self._replay_n_l0_kv_end
        if s0 > 0:
            vals[:s0] = new_prefix_i32 + self.topk
        if s1 > s0:
            vals[s0:s1] = new_prefix_i32
        if s1 < vals.numel():
            vals[s1:] = self._replay_l1_kv_indptr_base + new_shared_len

        all_idx = self._replay_all_indices
        for backend in self.attn_backends:
            buf = backend.cascade_attn.int_workspace_buffer.view(torch.int32)
            buf[all_idx] = vals

        nvtx_pop()
        return True
