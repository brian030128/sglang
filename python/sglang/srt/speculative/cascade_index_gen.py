"""Triton kernels for generating 2-level cascade KV indices.

Level 1 (shared prefix): one entry per request, grouping topk Q tokens.
Level 2 (unique suffix): one entry per branch (request * topk), with draft tokens only.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def generate_cascade_shared_kv_indices(
    req_pool_indices,
    req_to_token,
    seq_lens,
    kv_indices_shared,
    kv_indptr_shared,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Generate KV indices for the shared prefix level.

    Grid: (num_seqs,)
    Each program handles one request's prefix KV indices.

    kv_indices_shared: flat buffer, size = sum(prefix_lens)
    kv_indptr_shared: [num_seqs + 1], cumsum of prefix lens
    """
    bid = tl.program_id(axis=0)
    num_seqs = tl.num_programs(axis=0)

    # Compute prefix length for this request
    prefix_len = tl.load(seq_lens + bid)

    # Compute cumulative offset for this request
    load_offset = tl.arange(0, bs_upper)
    prev_lens = tl.load(seq_lens + load_offset, mask=load_offset < bid, other=0)
    cum_prefix = tl.sum(prev_lens)

    # Write kv_indptr entry
    if bid == 0:
        tl.store(kv_indptr_shared, 0)
    tl.store(kv_indptr_shared + bid + 1, cum_prefix + prefix_len)

    # Copy page indices from req_to_token pool
    token_pool_ptr = req_to_token + tl.load(req_pool_indices + bid) * pool_len
    kv_ptr = kv_indices_shared + cum_prefix

    kv_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(prefix_len, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = kv_offset < prefix_len
        data = tl.load(token_pool_ptr + kv_offset, mask=mask)
        tl.store(kv_ptr + kv_offset, data, mask=mask)
        kv_offset += BLOCK_SIZE


@triton.jit
def generate_cascade_unique_kv_indices(
    req_pool_indices,
    req_to_token,
    seq_lens,
    kv_indices_unique,
    kv_indptr_unique,
    step_offset,
    pool_len: tl.constexpr,
    num_steps: tl.constexpr,
    page_size: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
):
    """Generate KV indices for unique draft suffix level.

    Grid: (num_seqs, topk)
    Each program handles one branch's suffix KV indices.

    The suffix consists of `step_offset` draft tokens for this branch.
    kv_indices_unique: flat buffer for all branches' suffix indices
    kv_indptr_unique: [num_seqs * topk + 1], cumsum of suffix lens
    """
    bid = tl.program_id(axis=0)
    topk_id = tl.program_id(axis=1)
    num_seqs = tl.num_programs(axis=0)
    topk = tl.num_programs(axis=1)

    zid = bid * topk + topk_id  # branch index

    # Each branch has step_offset draft tokens
    suffix_len = step_offset

    # Compute kv_indptr for this branch
    if zid == 0:
        tl.store(kv_indptr_unique, 0)
    tl.store(kv_indptr_unique + zid + 1, (zid + 1) * suffix_len)

    # Compute where this branch's draft tokens are in the req_to_token pool
    prefix_len = tl.load(seq_lens + bid)
    token_pool_ptr = req_to_token + tl.load(req_pool_indices + bid) * pool_len

    extend_offset = tl.arange(0, iter_upper)
    if page_size == 1 or topk == 1:
        # Simple case: draft tokens are at prefix_len + topk_id * num_steps + offset
        extend_data = tl.load(
            token_pool_ptr + prefix_len + topk_id * num_steps + extend_offset,
            mask=extend_offset < suffix_len,
        )
    else:
        # page_size > 1: need to account for page alignment
        last_page_len = prefix_len % page_size
        num_new_pages_per_topk = (
            last_page_len + num_steps + page_size - 1
        ) // page_size
        prefix_base = prefix_len // page_size * page_size
        start = (
            prefix_base + topk_id * num_new_pages_per_topk * page_size + last_page_len
        )
        extend_data = tl.load(
            token_pool_ptr + start + extend_offset,
            mask=extend_offset < suffix_len,
        )

    kv_ptr = kv_indices_unique + zid * suffix_len
    tl.store(kv_ptr + extend_offset, extend_data, mask=extend_offset < suffix_len)


def next_power_of_2(n: int) -> int:
    n = max(n, 1)
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def build_shared_indices(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    device: torch.device,
    pool_len: int,
    max_total_prefix: int = 0,
) -> tuple:
    """Build Level 1 (shared prefix) indices.

    Args:
        max_total_prefix: Upper bound on total prefix pages. If >0, used to
            allocate kv_indices_shared without a GPU->CPU sync. The caller
            should slice kv_indices_shared to the actual size after a later
            sync (kv_indptr_shared[-1] gives the true total).

    Returns:
        kv_indptr_shared: [num_seqs + 1]
        kv_indices_shared: flat page indices for all prefixes
        kv_len_shared: [num_seqs] -- prefix lengths
    """
    num_seqs = req_pool_indices.shape[0]

    if max_total_prefix > 0:
        total_prefix = max_total_prefix
    else:
        # Fallback: sync to get exact size (used in tests)
        total_prefix = int(seq_lens.sum().item())

    kv_indices_shared = torch.empty(total_prefix, dtype=torch.int32, device=device)
    kv_indptr_shared = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)

    BLOCK_SIZE = 128
    generate_cascade_shared_kv_indices[(num_seqs,)](
        req_pool_indices,
        req_to_token,
        seq_lens,
        kv_indices_shared,
        kv_indptr_shared,
        pool_len,
        next_power_of_2(num_seqs),
        BLOCK_SIZE,
    )

    kv_len_shared = seq_lens.to(torch.int32)
    return kv_indptr_shared, kv_indices_shared, kv_len_shared


def build_unique_indices(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int,
    step_offset: int,
    num_steps: int,
    page_size: int,
    device: torch.device,
    pool_len: int,
) -> tuple:
    """Build Level 2 (unique suffix) indices for a given step.

    Returns:
        kv_indptr_unique: [num_seqs * topk + 1]
        kv_indices_unique: flat page indices for all suffixes
        kv_len_unique: [num_seqs * topk] -- all equal to step_offset
    """
    num_seqs = req_pool_indices.shape[0]
    total_branches = num_seqs * topk
    total_suffix = total_branches * step_offset

    kv_indices_unique = torch.empty(max(total_suffix, 1), dtype=torch.int32, device=device)
    kv_indptr_unique = torch.empty(total_branches + 1, dtype=torch.int32, device=device)

    if step_offset == 0:
        kv_indptr_unique.zero_()
        kv_len_unique = torch.zeros(total_branches, dtype=torch.int32, device=device)
        return kv_indptr_unique, kv_indices_unique, kv_len_unique

    generate_cascade_unique_kv_indices[(num_seqs, topk)](
        req_pool_indices,
        req_to_token,
        seq_lens,
        kv_indices_unique,
        kv_indptr_unique,
        step_offset,
        pool_len,
        num_steps,
        page_size,
        next_power_of_2(num_seqs),
        next_power_of_2(step_offset),
    )

    kv_len_unique = torch.full(
        (total_branches,), step_offset, dtype=torch.int32, device=device
    )
    return kv_indptr_unique, kv_indices_unique, kv_len_unique
