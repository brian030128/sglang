"""DeFT (Yao et al., ICLR'25) as an EAGLE draft decode backend.

Baseline for Fast Draft. Runs DeFT's tree-attention kernels (``DeFT_decode``:
KV-guided grouping into fixed-size subtrees with per-token causal bitmasks,
then a cross-subtree merge) from the DeFT kernel used in the FastTree artifact
(``3rdparty/FastTree-Artifact/kernel_bench/DeFT.py``, unmodified), the same
kernel the paper's kernel benchmark uses.

DeFT's planning (``DeFT_preparation``) is a per-token Python loop, which at a
60K-token prefix would dominate the draft step and make the comparison a
strawman. ``deft_plan`` below is a vectorized reimplementation that produces
identical metadata (verified against ``DeFT_preparation`` in
``tests/test_deft_plan.py``), generalized to a forest so batches of requests
are handled without DeFT's single-root restriction. For one request the
subtrees that lie entirely inside the shared prefix are planned once per draft
iteration; only the tail (last partial prefix subtree + branch suffixes) is
re-planned per draft step.

The draft tree is the same 2-level tree every other backend sees: prefix of
each request, then topk branch paths whose step-i length is i+1.

No CUDA graph support: DeFT's grid and metadata sizes depend on the tree.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, List, Sequence, Tuple

import numpy as np
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

# DeFT settings used by the paper's kernel benchmark (tests/bench_tree_attn.py).
SUBTREE_LEN = 128
MASK_LEN = 64
Q_TILE_SIZE = 16
KV_TILE_SIZE = 32

_DeFT_decode = None

_U64_ONES = np.uint64(0xFFFFFFFFFFFFFFFF)


# DeFT's kernels hard-code fp16 in three places, so they cannot read a bf16 KV
# cache. These substitutions use the input dtype instead (a no-op for fp16
# inputs), applied to a generated copy so the artifact itself stays unmodified.
_DTYPE_GENERIC_SUBS = [
    ("tl.dot(exp_logics.to(tl.float16), v)", "tl.dot(exp_logics.to(v.dtype), v)"),
    ("tl.cast(mid_o, dtype=tl.float16)", "tl.cast(mid_o, dtype=Q.dtype.element_ty)"),
    ("tl.cast(tl.ravel(acc / sum_exp), dtype=tl.float16)",
     "tl.cast(tl.ravel(acc / sum_exp), dtype=O.dtype.element_ty)"),
]


def _import_deft():
    global _DeFT_decode
    if _DeFT_decode is not None:
        return
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 7))
    path = os.path.join(root, "3rdparty", "FastTree-Artifact", "kernel_bench")
    if path not in sys.path:
        sys.path.insert(0, path)  # for kv_tree_simple

    with open(os.path.join(path, "DeFT.py")) as f:
        src = f.read()
    for old, new in _DTYPE_GENERIC_SUBS:
        assert src.count(old) == 1, f"DeFT.py changed upstream: {old!r}"
        src = src.replace(old, new)
    gen_dir = os.path.join(root, ".venv", "deft_generated")
    os.makedirs(gen_dir, exist_ok=True)
    gen_file = os.path.join(gen_dir, "DeFT_dtype_generic.py")
    # Triton reads kernel source from the file, so write a real module
    # (atomically: every TP rank imports this).
    tmp = f"{gen_file}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(src)
    os.replace(tmp, gen_file)
    if gen_dir not in sys.path:
        sys.path.insert(0, gen_dir)
    from DeFT_dtype_generic import DeFT_decode

    _DeFT_decode = DeFT_decode


# ----------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------

Segment = Tuple[int, int, int, int]  # (num_tokens, req_lo, req_hi, src_start)


def _low_bits(n: np.ndarray) -> np.ndarray:
    """uint64 with the low n bits set, for n in [0, 64]."""
    n = n.astype(np.int64)
    out = _U64_ONES >> np.minimum(64 - n, 63).astype(np.uint64)
    return np.where(n == 0, np.uint64(0), out)


def deft_plan(
    segs: Sequence[Segment],
    subtree_len: int = SUBTREE_LEN,
    mask_len: int = MASK_LEN,
) -> dict:
    """Plan DeFT metadata for a token stream given in DFS order.

    Each segment is a run of tokens belonging to one tree node: every token is
    visible to requests [req_lo, req_hi] and reads source slot src_start + j.
    The stream is cut into subtrees of ``subtree_len`` tokens, exactly as
    DeFT's ``_group_subtree`` does. Returns host (numpy) arrays; ``src`` holds
    source slots instead of KV entries.
    """
    assert mask_len == 64
    segs = [s for s in segs if s[0] > 0]
    n = np.array([s[0] for s in segs], dtype=np.int64)
    lo_s = np.array([s[1] for s in segs], dtype=np.int64)
    hi_s = np.array([s[2] for s in segs], dtype=np.int64)
    src_s = np.array([s[3] for s in segs], dtype=np.int64)

    total = int(n.sum())
    seg_of_tok = np.repeat(np.arange(len(segs)), n)
    seg_start = np.concatenate(([0], np.cumsum(n)[:-1]))
    tok = np.arange(total, dtype=np.int64)
    src = src_s[seg_of_tok] + (tok - seg_start[seg_of_tok])
    lo = lo_s[seg_of_tok]
    hi = hi_s[seg_of_tok]

    num_chunks = (total - 1) // subtree_len + 1
    starts = np.arange(num_chunks, dtype=np.int64) * subtree_len
    kv_lens = np.diff(np.append(starts, total))
    clo = np.minimum.reduceat(lo, starts)
    chi = np.maximum.reduceat(hi, starts)
    q_lens = chi - clo + 1
    nblk = (q_lens - 1) // mask_len + 1

    chunk_of_tok = tok // subtree_len
    cm_sizes = nblk * kv_lens
    cm_off = np.concatenate(([0], np.cumsum(cm_sizes)[:-1]))
    causalmask = np.zeros(int(cm_sizes.sum()), dtype=np.uint64)
    rel_lo = lo - clo[chunk_of_tok]
    rel_hi = hi - clo[chunk_of_tok] + 1
    pos_in_chunk = tok - starts[chunk_of_tok]
    for m in range(int(nblk.max())):
        sel = nblk[chunk_of_tok] > m
        a = np.clip(rel_lo[sel] - m * mask_len, 0, mask_len)
        b = np.clip(rel_hi[sel] - m * mask_len, 0, mask_len)
        bits = _low_bits(b) ^ _low_bits(a)
        c = chunk_of_tok[sel]
        causalmask[cm_off[c] + m * kv_lens[c] + pos_in_chunk[sel]] = bits

    q_entries = np.concatenate(
        [np.arange(l, h + 1, dtype=np.int64) for l, h in zip(clo, chi)]
    )
    q_offs = np.concatenate(([0], np.cumsum(q_lens)[:-1]))
    return dict(
        src=src,
        kv_offs=starts,
        kv_lens=kv_lens,
        q_entries=q_entries,
        q_offs=q_offs,
        q_lens=q_lens,
        causalmask=causalmask.view(np.int64),
        causalmask_offset=cm_off,
    )


def deft_concat_plans(a: dict, b: dict) -> dict:
    """Concatenate two plans for consecutive (subtree-aligned) token streams."""
    kv_total = int(a["kv_lens"].sum())
    q_total = int(a["q_lens"].sum())
    cm_total = len(a["causalmask"])
    return dict(
        src=np.concatenate([a["src"], b["src"]]),
        kv_offs=np.concatenate([a["kv_offs"], b["kv_offs"] + kv_total]),
        kv_lens=np.concatenate([a["kv_lens"], b["kv_lens"]]),
        q_entries=np.concatenate([a["q_entries"], b["q_entries"]]),
        q_offs=np.concatenate([a["q_offs"], b["q_offs"] + q_total]),
        q_lens=np.concatenate([a["q_lens"], b["q_lens"]]),
        causalmask=np.concatenate([a["causalmask"], b["causalmask"]]),
        causalmask_offset=np.concatenate(
            [a["causalmask_offset"], b["causalmask_offset"] + cm_total]
        ),
    )


def deft_req_to_vnode(q_entries: np.ndarray, num_requests: int):
    """For each request, the positions in q_entries that hold it (ascending)."""
    order = np.argsort(q_entries, kind="stable")
    lens = np.bincount(q_entries, minlength=num_requests)[:num_requests]
    offs = np.concatenate(([0], np.cumsum(lens)[:-1]))
    return order, offs, lens


def two_level_segments(
    prefix_lens: Sequence[int], topk: int, depth: int, max_depth: int
) -> List[Segment]:
    """DFS token stream of the draft forest.

    Source slots: prefixes are concatenated first (prefix p starts at its
    cumulative offset), then branch r's suffix starts at
    total_prefix + r * max_depth.
    """
    total_prefix = int(sum(prefix_lens))
    segs: List[Segment] = []
    off = 0
    for p, L in enumerate(prefix_lens):
        segs.append((int(L), p * topk, (p + 1) * topk - 1, off))
        off += int(L)
        for k in range(topk):
            r = p * topk + k
            segs.append((depth, r, r, total_prefix + r * max_depth))
    return segs


# ----------------------------------------------------------------------
# Backend
# ----------------------------------------------------------------------


class DeFTDraftAttnBackend(AttentionBackend):
    """Per-step backend running DeFT_decode with this step's plan."""

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
        aux = self._parent.step_aux[self._step_index]
        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        o = torch.empty_like(q_reshaped)
        _DeFT_decode(
            q_reshaped,
            k_buffer,
            v_buffer,
            o,
            *aux,
            Q_TILE_SIZE,
            KV_TILE_SIZE,
            layer.scaling,
            MASK_LEN,
            logit_cap=layer.logit_cap if layer.logit_cap else -1,
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)


class DeFTMultiStepDraftBackend:
    """Drop-in replacement for FlashInferMultiStepDraftBackend (SGLANG_DEFT_DRAFT=1)."""

    supports_cuda_graph = False

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        _import_deft()
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.num_draft_steps = speculative_num_steps - 1
        self.page_size = model_runner.page_size
        assert self.page_size == 1, "DeFT draft backend assumes page_size=1"
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.device = model_runner.device
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim

        self.attn_backends: List[DeFTDraftAttnBackend] = [
            DeFTDraftAttnBackend(model_runner, self, i)
            for i in range(self.num_draft_steps)
        ]
        self.step_aux = [None] * self.num_draft_steps

    def _to_aux(self, plan: dict, src_gpu: torch.Tensor, prefix_entries=None,
                prefix_mask=None, num_requests: int = 0):
        """Materialize a host plan as the 13 DeFT_decode metadata tensors."""
        dev = self.device
        i32 = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.int32)).to(
            dev, non_blocking=True
        )
        entries = src_gpu[torch.from_numpy(plan["src"]).to(dev, non_blocking=True)]
        mask = torch.from_numpy(plan["causalmask"]).to(dev, non_blocking=True)
        if prefix_entries is not None:
            entries = torch.cat([prefix_entries, entries])
            mask = torch.cat([prefix_mask, mask])
        pad = torch.full((32,), -1, dtype=torch.int32, device=dev)
        entries = torch.cat([entries.to(torch.int32), pad])

        order, r_offs, r_lens = deft_req_to_vnode(plan["q_entries"], num_requests)
        n_q = len(plan["q_entries"])
        return (
            entries,
            i32(plan["kv_offs"]),
            i32(plan["kv_lens"]),
            i32(plan["q_entries"]),
            i32(plan["q_offs"]),
            i32(plan["q_lens"]),
            mask,
            i32(plan["causalmask_offset"]),
            torch.cat([i32(order), pad]),
            i32(r_offs),
            i32(r_lens),
            torch.empty((n_q, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=dev),
            torch.empty((n_q, self.num_qo_heads), dtype=torch.float32, device=dev),
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        num_seqs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.to("cpu")
        prefix_lens = [int(x) for x in seq_lens_cpu.tolist()]
        total_prefix = sum(prefix_lens)
        num_requests = num_seqs * self.topk
        D = self.num_draft_steps

        shared = torch.empty(max(total_prefix, 1), dtype=torch.int32, device=self.device)
        shared_indptr = torch.empty(num_seqs + 1, dtype=torch.int32, device=self.device)
        generate_cascade_shared_kv_indices[(num_seqs,)](
            req_pool_indices, req_to_token, seq_lens, shared, shared_indptr,
            self.pool_len, next_power_of_2(num_seqs), 128,
        )
        _, unique, _ = build_unique_indices(
            req_pool_indices, req_to_token, seq_lens, self.topk, D,
            self.speculative_num_steps, self.page_size, self.device, self.pool_len,
        )
        src_gpu = torch.cat([shared[:total_prefix], unique[: num_requests * D]]).long()
        self.step_aux = self.build_step_aux(prefix_lens, src_gpu)

    def build_step_aux(self, prefix_lens: List[int], src_gpu: torch.Tensor):
        """DeFT_decode metadata for every draft step.

        src_gpu holds KV pool locations: the concatenated prefixes, then each
        branch's max-depth suffix (branch r at total_prefix + r * max_depth).
        """
        num_seqs = len(prefix_lens)
        total_prefix = sum(prefix_lens)
        num_requests = num_seqs * self.topk
        D = self.num_draft_steps
        step_aux = [None] * D

        if num_seqs == 1:
            # Subtrees fully inside the prefix are the same for every step.
            L = prefix_lens[0]
            n_full = (L // SUBTREE_LEN) * SUBTREE_LEN
            head = (
                deft_plan([(n_full, 0, self.topk - 1, 0)]) if n_full > 0 else None
            )
            head_entries = head_mask = None
            if head is not None:
                head_entries = src_gpu[:n_full]  # src is the identity on the prefix
                head_mask = torch.from_numpy(head["causalmask"]).to(
                    self.device, non_blocking=True
                )
            for i in range(D):
                segs = [(L - n_full, 0, self.topk - 1, n_full)] + [
                    (i + 1, r, r, total_prefix + r * D) for r in range(self.topk)
                ]
                tail = deft_plan(segs)
                if head is not None:
                    # Per-token arrays of the head are already on the GPU;
                    # _to_aux prepends them to the tail's.
                    plan = deft_concat_plans(head, tail)
                    plan["src"] = tail["src"]
                    plan["causalmask"] = tail["causalmask"]
                else:
                    plan = tail
                step_aux[i] = self._to_aux(
                    plan, src_gpu, head_entries, head_mask, num_requests
                )
        else:
            for i in range(D):
                plan = deft_plan(two_level_segments(prefix_lens, self.topk, i + 1, D))
                step_aux[i] = self._to_aux(plan, src_gpu, num_requests=num_requests)
        return step_aux
