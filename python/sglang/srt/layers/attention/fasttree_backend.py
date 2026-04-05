"""FastTree-based draft decode attention backend for speculative decoding.

Mirrors FlashInferMultiStepDraftBackend but uses the FastTree kernel
(MLSys'25) to compute tree-structured attention. FastTree decomposes the
EAGLE draft tree (shared prefix + topk branch suffixes) into virtual nodes
and schedules them with a cost-model-driven heuristic.

Non-CUDA-graph path only: FastTree's Triton kernels cannot be captured in
CUDA graphs. The draft decode loop runs in eager mode while target model
verification and draft-extend remain on CUDA graphs.

Per-iteration: builds a KVTreeNode tree from EAGLE spec info, calls
fasttree_preparation() once at max draft depth, remaps KV indices through
SGLang's paged KV pool.

Per-step: patches vnode_to_kv_lens for leaf vnodes to the current suffix
length, then calls fasttree_decode().
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

# Lazy imports for FastTree (resolved on first use)
_fasttree_imported = False
_fasttree_decode = None
_fasttree_preparation = None
_FastTreeParams = None
_KVTreeNode = None


def _import_fasttree():
    global _fasttree_imported, _fasttree_decode, _fasttree_preparation
    global _FastTreeParams, _KVTreeNode
    if _fasttree_imported:
        return

    # Add FastTree-Artifact to path
    # __file__ is .../fast-draft/3rdparty/sglang/python/sglang/srt/layers/attention/fasttree_backend.py
    # Project root (fast-draft/) is 7 levels up from dirname(__file__)
    root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", ".."
        )
    )
    ft_path = os.path.join(root, "3rdparty", "FastTree-Artifact", "kernel_bench")
    if ft_path not in sys.path:
        sys.path.insert(0, ft_path)

    from fasttree import (
        FastTreeParams,
        fasttree_decode,
        fasttree_preparation,
    )
    from kv_tree_simple import KVTreeNode

    _fasttree_decode = fasttree_decode
    _fasttree_preparation = fasttree_preparation
    _FastTreeParams = FastTreeParams
    _KVTreeNode = KVTreeNode
    _fasttree_imported = True


class FastTreeDraftAttnBackend(AttentionBackend):
    """Per-step attention backend using the FastTree kernel.

    All steps share metadata from the parent FastTreeMultiStepDraftBackend.
    Each backend has a step_index and patches vnode_to_kv_lens before its
    first layer call to set the correct suffix length.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        parent: "FastTreeMultiStepDraftBackend",
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

        self._parent = parent
        self._step_index = step_index
        self._step_updated = False

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Planning is done by the parent FastTreeMultiStepDraftBackend
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

        # Save KV to pool first (so attention can read current step's KV)
        if k is not None and v is not None and save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        # Patch suffix length for this step (once, before first layer)
        if not self._step_updated:
            suffix_len = self._step_index + 1
            parent = self._parent
            if parent._leaf_vnode_indices is not None:
                parent._ft_vnode_to_kv_lens[parent._leaf_vnode_indices] = suffix_len
            self._step_updated = True

        parent = self._parent
        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

        # Get KV buffers for this layer
        k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )

        # Allocate output
        o = torch.empty_like(q_reshaped)

        _fasttree_decode(
            q_reshaped,
            k_buffer,
            v_buffer,
            o,
            parent._ft_vnode_to_kv_entries,
            parent._ft_vnode_to_kv_offs,
            parent._ft_vnode_to_kv_lens,
            parent._ft_vnode_to_q_entries,
            parent._ft_vnode_to_q_offs,
            parent._ft_vnode_to_q_lens,
            parent._ft_req_to_vnode_entries,
            parent._ft_req_to_vnode_offs,
            parent._ft_req_to_vnode_lens,
            parent._ft_mid_o,
            parent._ft_mid_lse,
            parent._ft_phase_node_nums,
            parent._ft_phase_node_offsets,
            parent._ft_phase_q_tile_sizes,
            parent._ft_phase_kv_tile_sizes,
            layer.scaling,
            logit_cap=layer.logit_cap if layer.logit_cap is not None else -1,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)


class FastTreeMultiStepDraftBackend:
    """Drop-in replacement for FlashInferMultiStepDraftBackend using FastTree.

    Builds a KVTreeNode tree from EAGLE's draft structure (shared prefix +
    topk branch suffixes), runs fasttree_preparation() once at max draft
    depth, then patches vnode_to_kv_lens per step.

    Does NOT support CUDA graph capture.
    """

    supports_cuda_graph = False

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        _import_fasttree()

        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.device = model_runner.device
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]

        num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        head_dim = model_runner.model_config.head_dim
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

        # Configure FastTree parameters
        kv_group_num = num_qo_heads // num_kv_heads
        self._ft_params = _FastTreeParams()
        self._ft_params.set_values(0.66, 0.33, 0.1)
        # Use small tile sizes appropriate for EAGLE (small batch, small suffix)
        self._ft_params.set_q_tile_sizes([16, 4])
        self._ft_params.set_kv_tile_sizes([32, 32])
        self._ft_params.set_kv_group_num(kv_group_num)

        # Per-step attention backends
        self.attn_backends: List[FastTreeDraftAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FastTreeDraftAttnBackend(model_runner, self, step_index=i)
            )

        # FastTree metadata (populated in init_forward_metadata)
        self._ft_vnode_to_kv_entries = None
        self._ft_vnode_to_kv_offs = None
        self._ft_vnode_to_kv_lens = None
        self._ft_vnode_to_q_entries = None
        self._ft_vnode_to_q_offs = None
        self._ft_vnode_to_q_lens = None
        self._ft_req_to_vnode_entries = None
        self._ft_req_to_vnode_offs = None
        self._ft_req_to_vnode_lens = None
        self._ft_mid_o = None
        self._ft_mid_lse = None
        self._ft_phase_node_nums = None
        self._ft_phase_node_offsets = None
        self._ft_phase_q_tile_sizes = None
        self._ft_phase_kv_tile_sizes = None
        self._leaf_vnode_indices = None

        self._debug_logged = False

    def _build_eagle_tree(self, num_seqs, seq_lens_cpu, max_depth):
        """Build KVTreeNode list + KV_ptrs for EAGLE draft decode.

        Tree structure:
          num_seqs == 1: root(prefix_len) -> topk leaves(max_depth)
          num_seqs > 1:  dummy_root(1) -> num_seqs prefix(prefix_len) -> topk leaves(max_depth)

        Returns (tree_info, kv_ptrs, leaf_node_ids).
        """
        nodes = []
        leaf_node_ids = []

        if num_seqs == 1:
            prefix_len = int(seq_lens_cpu[0].item())
            root = _KVTreeNode()
            root.parent = -1
            root.id = 0
            root.seqlen = prefix_len
            root.num_children = self.topk
            root.requests = []
            nodes.append(root)

            for k in range(self.topk):
                leaf = _KVTreeNode()
                leaf.parent = 0
                leaf.id = 1 + k
                leaf.seqlen = max_depth
                leaf.num_children = 0
                leaf.requests = []
                nodes.append(leaf)
                leaf_node_ids.append(leaf.id)
        else:
            # Dummy root with seqlen=1 (will map to first prefix token)
            root = _KVTreeNode()
            root.parent = -1
            root.id = 0
            root.seqlen = 1
            root.num_children = num_seqs
            root.requests = []
            nodes.append(root)

            for p in range(num_seqs):
                prefix_len = int(seq_lens_cpu[p].item())
                mid = _KVTreeNode()
                mid.parent = 0
                mid.id = 1 + p
                mid.seqlen = prefix_len
                mid.num_children = self.topk
                mid.requests = []
                nodes.append(mid)

            nid = 1 + num_seqs
            for p in range(num_seqs):
                for k in range(self.topk):
                    leaf = _KVTreeNode()
                    leaf.parent = 1 + p
                    leaf.id = nid
                    leaf.seqlen = max_depth
                    leaf.num_children = 0
                    leaf.requests = []
                    nodes.append(leaf)
                    leaf_node_ids.append(nid)
                    nid += 1

        # Propagate request IDs (leaf-to-root)
        req_id = 0
        for n in range(len(nodes)):
            if nodes[n].num_children == 0:
                node_idx = n
                while node_idx != -1:
                    nodes[node_idx].requests.append(req_id)
                    node_idx = nodes[node_idx].parent
                req_id += 1

        # Build KV_ptrs (cumulative seqlens)
        kv_ptrs = [0]
        for node in nodes:
            kv_ptrs.append(kv_ptrs[-1] + node.seqlen)

        return nodes, kv_ptrs, leaf_node_ids

    def _build_flat_page_table(
        self,
        nodes,
        kv_ptrs,
        num_seqs,
        seq_lens,
        req_pool_indices,
        req_to_token,
        out_cache_loc_all,
        max_depth,
    ):
        """Build flat_page_table mapping virtual contiguous offsets to paged KV indices.

        out_cache_loc_all: (speculative_num_steps, num_seqs * topk) — cache
            locations for all draft steps. Reshaped from forward_batch.out_cache_loc.
        """
        total_tokens = kv_ptrs[-1]
        flat_page_table = torch.empty(total_tokens, dtype=torch.int32, device=self.device)

        for node in nodes:
            nid = node.id
            start = kv_ptrs[nid]
            length = node.seqlen

            if node.num_children > 0 or (num_seqs > 1 and nid == 0):
                # Prefix node (or dummy root): map from req_to_token
                if num_seqs == 1 and nid == 0:
                    # Single prefix root
                    rpi = req_pool_indices[0]
                    indices = req_to_token[rpi, :length].to(torch.int32)
                    flat_page_table[start : start + length] = indices
                elif num_seqs > 1 and nid == 0:
                    # Dummy root: map to first token of first prefix
                    rpi = req_pool_indices[0]
                    flat_page_table[start] = req_to_token[rpi, 0].to(torch.int32)
                else:
                    # Multi-seq prefix node (nid = 1..num_seqs)
                    p = nid - 1
                    rpi = req_pool_indices[p]
                    indices = req_to_token[rpi, :length].to(torch.int32)
                    flat_page_table[start : start + length] = indices
            else:
                # Leaf node: map from out_cache_loc
                # Determine which branch this leaf corresponds to
                if num_seqs == 1:
                    branch_idx = nid - 1  # leaves are nodes 1..topk
                else:
                    # leaves start at nid = 1 + num_seqs
                    leaf_offset = nid - (1 + num_seqs)
                    branch_idx = leaf_offset  # 0..num_seqs*topk-1

                # out_cache_loc_all[step, branch_idx] for steps 0..max_depth-1
                for s in range(max_depth):
                    flat_page_table[start + s] = out_cache_loc_all[s, branch_idx].to(
                        torch.int32
                    )

        return flat_page_table

    def _identify_leaf_vnodes(self, nodes, kv_ptrs, leaf_node_ids, vnode_to_kv_offs):
        """Find which vnode indices correspond to leaf tree nodes.

        Leaf vnodes are those whose kv_offs falls within a leaf node's KV range.
        Returns a tensor of vnode indices for all leaf vnodes.
        """
        leaf_ranges = set()
        for nid in leaf_node_ids:
            start = kv_ptrs[nid]
            end = kv_ptrs[nid + 1]
            leaf_ranges.add((start, end))

        vnode_kv_offs_cpu = vnode_to_kv_offs.cpu().tolist()
        leaf_vnode_indices = []
        for vi, off in enumerate(vnode_kv_offs_cpu):
            for start, end in leaf_ranges:
                if start <= off < end:
                    leaf_vnode_indices.append(vi)
                    break

        if leaf_vnode_indices:
            return torch.tensor(leaf_vnode_indices, dtype=torch.long, device=self.device)
        return None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Build FastTree metadata for all draft steps.

        Called once per EAGLE iteration (before the multi-step loop).
        """
        # Reset step_updated flags
        for backend in self.attn_backends:
            backend._step_updated = False

        num_seqs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens
        max_depth = self.speculative_num_steps - 1

        # CPU seq_lens (avoid GPU->CPU sync if available)
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.to("cpu")
            torch.cuda.synchronize()

        # Get out_cache_loc reshaped to (speculative_num_steps, num_seqs * topk)
        out_cache_loc = forward_batch.out_cache_loc
        out_cache_loc_all = out_cache_loc.reshape(
            num_seqs, self.topk, self.speculative_num_steps
        ).permute(2, 0, 1).reshape(self.speculative_num_steps, -1)

        # 1. Build tree
        tree_info, kv_ptrs, leaf_node_ids = self._build_eagle_tree(
            num_seqs, seq_lens_cpu, max_depth
        )
        batch_size = num_seqs * self.topk  # FastTree "batch size" = total requests

        # 2. Build flat page table
        flat_page_table = self._build_flat_page_table(
            tree_info, kv_ptrs, num_seqs, seq_lens, req_pool_indices,
            req_to_token, out_cache_loc_all, max_depth,
        )

        # 3. Run fasttree_preparation (CPU, once at max depth)
        # Suppress fasttree_preparation's debug prints (fires every iteration)
        import io as _io
        _old_stdout = sys.stdout
        sys.stdout = _io.StringIO()
        try:
            ft_aux, _ = _fasttree_preparation(
                tree_info,
                kv_ptrs,
                batch_size,
                self._num_qo_heads,
                self._num_kv_heads,
                self._head_dim,
                [1024, 128],   # KV_SPLIT_SIZES
                [132, 528],    # para_threshs1
                [132, 132],    # para_threshs2
                self._ft_params,
            )
        finally:
            sys.stdout = _old_stdout

        # Unpack the 13 metadata tensors
        (
            vnode_to_kv_entries,
            vnode_to_kv_offs,
            vnode_to_kv_lens,
            vnode_to_q_entries,
            vnode_to_q_offs,
            vnode_to_q_lens,
            req_to_vnode_entries,
            req_to_vnode_offs,
            req_to_vnode_lens,
            mid_o,
            mid_lse,
            phase_node_nums,
            phase_node_offsets,
        ) = ft_aux

        # 4. Remap vnode_to_kv_entries through flat_page_table
        # Entries contain indices into virtual contiguous layout; remap to paged pool
        valid_mask = vnode_to_kv_entries >= 0
        # Clamp to avoid OOB on padding entries (-1 → 0, will be masked by vnode_to_kv_lens)
        safe_indices = vnode_to_kv_entries.clamp(min=0).long()
        vnode_to_kv_entries[valid_mask] = flat_page_table[safe_indices[valid_mask]]

        # 5. Identify leaf vnodes for per-step patching
        leaf_vnode_indices = self._identify_leaf_vnodes(
            tree_info, kv_ptrs, leaf_node_ids, vnode_to_kv_offs
        )

        # Store all metadata
        self._ft_vnode_to_kv_entries = vnode_to_kv_entries
        self._ft_vnode_to_kv_offs = vnode_to_kv_offs
        self._ft_vnode_to_kv_lens = vnode_to_kv_lens
        self._ft_vnode_to_q_entries = vnode_to_q_entries
        self._ft_vnode_to_q_offs = vnode_to_q_offs
        self._ft_vnode_to_q_lens = vnode_to_q_lens
        self._ft_req_to_vnode_entries = req_to_vnode_entries
        self._ft_req_to_vnode_offs = req_to_vnode_offs
        self._ft_req_to_vnode_lens = req_to_vnode_lens
        self._ft_mid_o = mid_o
        self._ft_mid_lse = mid_lse
        self._ft_phase_node_nums = phase_node_nums
        self._ft_phase_node_offsets = phase_node_offsets
        self._ft_phase_q_tile_sizes = self._ft_params.TSQs
        self._ft_phase_kv_tile_sizes = self._ft_params.TSKs
        self._leaf_vnode_indices = leaf_vnode_indices

        if not self._debug_logged and os.environ.get("SGLANG_DEBUG_DRAFT_PARAMS") == "1":
            self._debug_logged = True
            print(
                f"\n[FASTTREE DRAFT] init_forward_metadata (pid={os.getpid()})",
                flush=True,
            )
            print(f"  num_seqs={num_seqs}, topk={self.topk}, max_depth={max_depth}", flush=True)
            print(f"  tree nodes={len(tree_info)}, batch_size(requests)={batch_size}", flush=True)
            print(f"  total vnodes={len(vnode_to_kv_offs)}", flush=True)
            print(f"  leaf vnodes={len(leaf_vnode_indices) if leaf_vnode_indices is not None else 0}", flush=True)
            print(f"  phase_node_nums={phase_node_nums}", flush=True)
            print(f"  TSQs={self._ft_params.TSQs}, TSKs={self._ft_params.TSKs}", flush=True)
