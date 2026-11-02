"""Hydragen (Juravsky et al., 2024) as an EAGLE draft decode backend.

Baseline for Fast Draft. Runs Hydragen's own ``hydragen_attention`` from the
upstream repo (``3rdparty/hydragen``, unmodified): flash-attention over a
contiguous shared-prefix cache with all topk branch queries batched together,
flash-attention over each branch's unique suffix, then Hydragen's
``combine_lse`` merge.

Hydragen requires dense, contiguous K/V tensors rather than SGLang's paged pool:
  * shared prefix: kept in a per-layer contiguous cache. For a single request
    the cache is appended incrementally (only newly accepted tokens are copied
    each iteration); for batches of requests it is re-gathered every iteration.
  * unique suffix: gathered per draft step (topk * depth tokens, small).

Hydragen pins flash-attn 2.3.6, whose private ``_flash_attn_forward`` /
``_flash_attn_varlen_forward`` signatures predate torch 2.9 support. A shim
exposes that 2.3.6 calling convention on top of the flash-attn 2.8.3 kernels
(same FA2 forward kernels), installed via FLASH_ATTN2_PATH.

No CUDA graph support: Hydragen's shared-cache shape changes whenever the
prefix grows, which is every draft iteration in speculative decoding.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from typing import TYPE_CHECKING, List

import torch

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

_hydragen_attention = None


def _project_root():
    # .../fast-draft/3rdparty/sglang/python/sglang/srt/layers/attention/hydragen_backend.py
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), *[".."] * 7)
    )


def _install_flash_attn_236_shim():
    """Register ``flash_attn.flash_attn_interface`` with the 2.3.6 private API.

    Loads the flash-attn 2.8.3 interface from FLASH_ATTN2_PATH under a private
    module name (so the installed ``flash_attn`` package is not shadowed) and
    adapts the two functions Hydragen imports.
    """
    if "flash_attn.flash_attn_interface" in sys.modules and getattr(
        sys.modules["flash_attn.flash_attn_interface"], "_hydragen_shim", False
    ):
        return

    fa2_path = os.environ.get(
        "FLASH_ATTN2_PATH", os.path.join(_project_root(), ".venv", "fa2")
    )
    if fa2_path not in sys.path:
        sys.path.append(fa2_path)  # for the flash_attn_2_cuda extension
    spec = importlib.util.spec_from_file_location(
        "_fa2_interface", os.path.join(fa2_path, "flash_attn", "flash_attn_interface.py")
    )
    fa2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fa2)

    def _flash_attn_forward(
        q, k, v, dropout_p, softmax_scale, causal, window_size=(-1, -1),
        return_softmax=False, alibi_slopes=None,
    ):
        out, softmax_lse, S_dmask, rng_state = fa2._flash_attn_forward(
            q, k, v, dropout_p, softmax_scale, causal,
            window_size[0], window_size[1], 0.0, alibi_slopes, return_softmax,
        )
        return out, q, k, v, out, softmax_lse, S_dmask, rng_state

    def _flash_attn_varlen_forward(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        dropout_p, softmax_scale, causal, window_size=(-1, -1),
        return_softmax=False, alibi_slopes=None,
    ):
        out, softmax_lse, S_dmask, rng_state = fa2._flash_attn_varlen_forward(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            dropout_p, softmax_scale, causal,
            window_size_left=window_size[0], window_size_right=window_size[1],
            softcap=0.0, alibi_slopes=alibi_slopes, return_softmax=return_softmax,
        )
        # 2.3.6 returned the varlen LSE padded as [batch, heads, max_seqlen_q];
        # 2.8.3 returns it packed as [heads, total_q].
        batch = cu_seqlens_q.shape[0] - 1
        heads, total_q = softmax_lse.shape
        if total_q == batch * max_seqlen_q:
            softmax_lse = softmax_lse.view(heads, batch, max_seqlen_q).transpose(0, 1)
        else:
            padded = softmax_lse.new_full((batch, heads, max_seqlen_q), float("-inf"))
            for i, (s, e) in enumerate(zip(cu_seqlens_q[:-1].tolist(), cu_seqlens_q[1:].tolist())):
                padded[i, :, : e - s] = softmax_lse[:, s:e]
            softmax_lse = padded
        return out, q, k, v, out, softmax_lse.contiguous(), S_dmask, rng_state

    shim = types.ModuleType("flash_attn.flash_attn_interface")
    shim._flash_attn_forward = _flash_attn_forward
    shim._flash_attn_varlen_forward = _flash_attn_varlen_forward
    shim._hydragen_shim = True
    sys.modules["flash_attn.flash_attn_interface"] = shim


def _import_hydragen():
    global _hydragen_attention
    if _hydragen_attention is not None:
        return
    _install_flash_attn_236_shim()
    hydragen_root = os.path.join(_project_root(), "3rdparty", "hydragen")
    if hydragen_root not in sys.path:
        sys.path.insert(0, hydragen_root)
    from hydragen.attention import hydragen_attention

    _hydragen_attention = hydragen_attention


class HydragenDraftAttnBackend(AttentionBackend):
    """Per-step backend; step i attends to prefix + first i+1 suffix tokens."""

    def __init__(self, model_runner: ModelRunner, parent, step_index: int):
        super().__init__()
        self._parent = parent
        self._step_index = step_index

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Metadata is built by the parent multi-step backend
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
        p = self._parent
        assert not layer.logit_cap or layer.logit_cap <= 0, "logit cap unsupported"
        assert abs(layer.scaling - layer.head_dim**-0.5) < 1e-6, (
            "Hydragen uses the default 1/sqrt(d) softmax scale"
        )

        k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        shared_k, shared_v = p.shared_kv(layer.layer_id, k_buffer, v_buffer)

        b = p.num_branches
        idx = p.unique_indices[:, : self._step_index + 1].reshape(-1)
        uk = k_buffer[idx].view(b, self._step_index + 1, *k_buffer.shape[1:])
        uv = v_buffer[idx].view(b, self._step_index + 1, *v_buffer.shape[1:])
        qh = q.contiguous().view(b, 1, layer.tp_q_head_num, layer.head_dim)

        if p.num_seqs == 1:
            out = _hydragen_attention(
                qh, uk, uv,
                shared_ks=[shared_k.unsqueeze(0)],
                shared_vs=[shared_v.unsqueeze(0)],
                shared_cu_seq_lens=[None],
                shared_max_seq_lens=[None],
                use_varlens=[False],
            )
        else:
            out = _hydragen_attention(
                qh, uk, uv,
                shared_ks=[shared_k],
                shared_vs=[shared_v],
                shared_cu_seq_lens=[p.shared_cu_seq_lens],
                shared_max_seq_lens=[p.shared_max_seq_len],
                use_varlens=[True],
            )
        return out.reshape(-1, layer.tp_q_head_num * layer.head_dim)


class HydragenMultiStepDraftBackend:
    """Drop-in replacement for FlashInferMultiStepDraftBackend (SGLANG_HYDRAGEN_DRAFT=1)."""

    supports_cuda_graph = False

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        _import_hydragen()
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.num_draft_steps = speculative_num_steps - 1
        self.page_size = model_runner.page_size
        assert self.page_size == 1, "Hydragen draft backend assumes page_size=1"
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.device = model_runner.device

        self.attn_backends: List[HydragenDraftAttnBackend] = [
            HydragenDraftAttnBackend(model_runner, self, i)
            for i in range(self.num_draft_steps)
        ]

        # Per-layer contiguous shared-prefix caches.
        self._shared_k = {}
        self._shared_v = {}
        self._layer_len = {}
        # Identity of the single request the incremental cache belongs to:
        # (req_pool_index, pool index of token 0).
        self._cache_owner = None
        self._target_len = 0
        self._last_loc = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        num_seqs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.to("cpu")

        self.num_seqs = num_seqs
        self.num_branches = num_seqs * self.topk

        cu = torch.zeros(num_seqs + 1, dtype=torch.int32)
        torch.cumsum(seq_lens_cpu.to(torch.int32), dim=0, out=cu[1:])
        total_prefix = int(cu[-1])
        self.shared_max_seq_len = int(seq_lens_cpu.max())

        self.shared_indices = torch.empty(
            max(total_prefix, 1), dtype=torch.int32, device=self.device
        )
        self.shared_cu_seq_lens = torch.empty(
            num_seqs + 1, dtype=torch.int32, device=self.device
        )
        generate_cascade_shared_kv_indices[(num_seqs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            self.shared_indices,
            self.shared_cu_seq_lens,
            self.pool_len,
            next_power_of_2(num_seqs),
            128,
        )

        # Unique suffix indices at max depth; step i uses the first i+1 columns.
        _, unique, _ = build_unique_indices(
            req_pool_indices,
            req_to_token,
            seq_lens,
            self.topk,
            self.num_draft_steps,
            self.speculative_num_steps,
            self.page_size,
            self.device,
            self.pool_len,
        )
        self.unique_indices = unique.view(self.num_branches, self.num_draft_steps).long()

        # Decide whether the incremental prefix cache is still valid.
        if num_seqs == 1:
            # One small D2H read: slot, location of token 0, and location of
            # the last token already cached. Any mismatch means a different
            # request now owns the slot, so the cache is rebuilt.
            rpi = req_pool_indices[0]
            last = max(self._target_len - 1, 0)
            cols = torch.tensor(
                [0, last, max(total_prefix - 1, 0)], dtype=torch.long, device=self.device
            )
            probe = torch.cat(
                [rpi.view(1).long(), req_to_token[rpi, cols].long()]
            ).tolist()
            owner = (probe[0], probe[1])
            if (
                owner != self._cache_owner
                or total_prefix < self._target_len
                or probe[2] != self._last_loc
            ):
                self._layer_len.clear()
                self._cache_owner = owner
            self._last_loc = probe[3]
        else:
            self._layer_len.clear()
            self._cache_owner = None
        self._target_len = total_prefix

    def shared_kv(self, layer_id: int, k_buffer: torch.Tensor, v_buffer: torch.Tensor):
        """Return the contiguous shared prefix K/V for this layer, appending new tokens."""
        target = self._target_len
        have = self._layer_len.get(layer_id, 0)
        if have > target:  # prefix shrank (e.g. retraction): rebuild
            have = 0
        if have < target:
            kbuf = self._shared_k.get(layer_id)
            if kbuf is None or kbuf.shape[0] < target:
                cap = max(target, int((kbuf.shape[0] if kbuf is not None else 0) * 1.5), 4096)
                new_k = torch.empty((cap, *k_buffer.shape[1:]), dtype=k_buffer.dtype, device=self.device)
                new_v = torch.empty((cap, *v_buffer.shape[1:]), dtype=v_buffer.dtype, device=self.device)
                if kbuf is not None and have > 0:
                    new_k[:have].copy_(kbuf[:have])
                    new_v[:have].copy_(self._shared_v[layer_id][:have])
                self._shared_k[layer_id] = new_k
                self._shared_v[layer_id] = new_v
            idx = self.shared_indices[have:target].long()
            torch.index_select(k_buffer, 0, idx, out=self._shared_k[layer_id][have:target])
            torch.index_select(v_buffer, 0, idx, out=self._shared_v[layer_id][have:target])
            self._layer_len[layer_id] = target
        return self._shared_k[layer_id][:target], self._shared_v[layer_id][:target]
