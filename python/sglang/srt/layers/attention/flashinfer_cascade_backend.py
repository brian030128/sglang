"""Cascade-aware draft decode attention backend for speculative decoding.

Mirrors FlashInferMultiStepDraftBackend but uses CascadeBatchAttentionWrapper
to read the shared prefix KV cache once and combine with each branch's unique
suffix, eliminating redundant prefix reads across topk branches.

Each draft step gets its own CascadeBatchAttentionWrapper with its own plan()
call — no update_draft_step patching, no plan_for_draft overhead.
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
)
from sglang.srt.speculative.draft_utils import nvtx_pop, nvtx_push

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class CascadeDraftAttnBackend(AttentionBackend):
    """Per-step attention backend using CascadeBatchAttention.

    Each instance owns its own CascadeBatchAttentionWrapper that is planned
    independently per step. No workspace patching needed.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        cascade_attn: CascadeBatchAttention,
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

    Mirrors the flat backend's structure: one wrapper per step, each with
    its own plan() call. CPU indptr/kv_len tensors are computed directly
    on CPU (deterministic from seq_lens and step geometry) — no GPU->CPU
    sync needed.
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

        # One wrapper per step (non-CUDA-graph mode)
        self.attn_backends: List[CascadeDraftAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            wrapper = CascadeBatchAttention(
                num_levels=2,
                kv_layout="NHD",
                device="cuda",
            )
            self.attn_backends.append(
                CascadeDraftAttnBackend(model_runner, wrapper)
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

        def call_fn(i, **tensors):
            backend = self.attn_backends[i]
            plan_kwargs = dict(
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
            arrays = dict(
                qo_indptr_arr=[tensors["qo_indptr_shared_cpu"], tensors["qo_indptr_unique_cpu"]],
                kv_indptr_arr=[tensors["kv_indptr_shared_cpu"], tensors["kv_indptr_unique_cpu"]],
                kv_indices_arr=[tensors["kv_indices_shared"], tensors["kv_indices_unique"]],
                kv_len_arr=[tensors["kv_len_shared_cpu"], tensors["kv_len_unique_cpu"]],
            )
            if first_call:
                backend.cascade_attn.plan(**arrays, **plan_kwargs)
            else:
                # fast_cascade_plan uses different parameter names
                fast_arrays = dict(
                    qo_indptr_host_arr=arrays["qo_indptr_arr"],
                    kv_indptr_host_arr=arrays["kv_indptr_arr"],
                    kv_indices_arr=arrays["kv_indices_arr"],
                    kv_len_host_arr=arrays["kv_len_arr"],
                )
                backend.cascade_attn.fast_cascade_plan(**fast_arrays, **plan_kwargs)

        nvtx_push("cascade/plan")
        self._build_cascade_indices_and_plan(forward_batch, call_fn)
        nvtx_pop()
        self._modules_initialized = True

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        max_branches = max_bs * self.topk
        max_shared_pages = max_bs * self.max_context_len
        max_unique_pages = max_branches * self.speculative_num_steps
        max_total_pages = max_shared_pages + max_unique_pages

        self.cuda_graph_kv_indices_buf = torch.zeros(
            max_total_pages, dtype=torch.int32, device="cuda"
        )

        # Replace each backend's wrapper with a CUDA-graph-enabled one
        for i in range(self.speculative_num_steps - 1):
            old = self.attn_backends[i]
            new = CascadeDraftAttnBackend.__new__(CascadeDraftAttnBackend)
            new.num_qo_heads = old.num_qo_heads
            new.num_kv_heads = old.num_kv_heads
            new.head_dim = old.head_dim
            new.data_type = old.data_type
            new.q_data_type = old.q_data_type
            new.max_context_len = old.max_context_len
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
        def call_fn(i, **tensors):
            backend = self.attn_backends[i]
            plan_kwargs = dict(
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
            if use_fast_plan:
                backend.cascade_attn.fast_cascade_plan(
                    qo_indptr_host_arr=[tensors["qo_indptr_shared_cpu"], tensors["qo_indptr_unique_cpu"]],
                    kv_indptr_host_arr=[tensors["kv_indptr_shared_cpu"], tensors["kv_indptr_unique_cpu"]],
                    kv_indices_arr=[tensors["kv_indices_shared"], tensors["kv_indices_unique"]],
                    kv_len_host_arr=[tensors["kv_len_shared_cpu"], tensors["kv_len_unique_cpu"]],
                    **plan_kwargs,
                )
            else:
                backend.cascade_attn.plan(
                    qo_indptr_arr=[tensors["qo_indptr_shared_cpu"], tensors["qo_indptr_unique_cpu"]],
                    kv_indptr_arr=[tensors["kv_indptr_shared_cpu"], tensors["kv_indptr_unique_cpu"]],
                    kv_indices_arr=[tensors["kv_indices_shared"], tensors["kv_indices_unique"]],
                    kv_len_arr=[tensors["kv_len_shared_cpu"], tensors["kv_len_unique_cpu"]],
                    **plan_kwargs,
                )

        nvtx_push("cascade/cuda_graph_plan")
        self._build_cascade_indices_and_plan(forward_batch, call_fn)
        nvtx_pop()
        self._modules_initialized = True
