from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache_vanilla import RadixKey


logger = logging.getLogger(__name__)


@dataclass
class _CustomRadixNode:
    """Radix-tree node used by the custom prefix cache.

    Each node stores a compressed token span and the KV indices for that same
    local span. Full-prefix KV is reconstructed by concatenating node-local KV
    along a matched path from the root.

    The fields here are the pieces that matter for runtime behavior:
    - each node represents a compressed token span (`token_segment`)
    - children continue the prefix from that span
    - the node carries KV indices for that same local compressed span
    - `lock_ref`: active users of this cached prefix, so in-use entries are not
      evicted
    - `last_access_ts`: recency signal used by the leaf-only LRU policy
    """

    token_segment: tuple[int, ...] = ()
    parent: Optional["_CustomRadixNode"] = None
    children: dict[int, "_CustomRadixNode"] = field(default_factory=dict)
    kv_indices: Optional[torch.Tensor] = None
    lock_ref: int = 0
    last_access_ts: int = 0


@dataclass
class _CustomLruEntry:
    """Node in a doubly linked list tracking evictable radix-tree leaves.

    The radix tree answers prefix-search questions.
    The LRU list answers eviction-order questions.

    Each entry points back to the corresponding radix node. The list is meant
    to contain only leaves that are currently evictable under your policy.
    """

    radix_node: _CustomRadixNode
    prev: Optional["_CustomLruEntry"] = None
    next: Optional["_CustomLruEntry"] = None


class _CustomLeafLru:
    """Leaf-only doubly linked list for leaf-first LRU eviction.

    This structure tracks only nodes that are both:
    - leaves in the radix tree
    - currently evictable (`lock_ref == 0`)

    The tree answers prefix-search questions. The LRU list answers eviction
    ordering questions. Keeping those concerns separate keeps eviction policy
    local to the small set of current candidates.

    This implementation uses leaf-only tracking plus timestamp-ordered
    insertion:
    - it tracks only nodes that are both leaves and currently evictable
    - it does not decide whether a node should be tracked; the radix-cache
      logic decides that and calls into this helper
    - recency is carried by `last_access_ts` on the radix nodes
    - reinsertion is O(n) over the leaf set, which is acceptable in this
      implementation because the candidate set is much smaller than the full
      tree and eviction is not on the critical GPU path
    """

    def __init__(self):
        self.head: Optional[_CustomLruEntry] = None
        self.tail: Optional[_CustomLruEntry] = None

    def clear(self):
        """Drop all tracked entries."""
        self.head = None
        self.tail = None

    def _find_entry(self, radix_node: _CustomRadixNode) -> Optional[_CustomLruEntry]:
        cur = self.head
        while cur is not None:
            if cur.radix_node is radix_node:
                return cur
            cur = cur.next
        return None

    def _unlink(self, entry: _CustomLruEntry):
        if entry.prev is not None:
            entry.prev.next = entry.next
        else:
            self.head = entry.next

        if entry.next is not None:
            entry.next.prev = entry.prev
        else:
            self.tail = entry.prev

        entry.prev = None
        entry.next = None

    def _insert_before(
        self,
        existing: Optional[_CustomLruEntry],
        entry: _CustomLruEntry,
    ):
        if existing is None:
            if self.tail is None:
                self.head = entry
                self.tail = entry
                return
            entry.prev = self.tail
            self.tail.next = entry
            self.tail = entry
            return

        entry.next = existing
        entry.prev = existing.prev
        if existing.prev is not None:
            existing.prev.next = entry
        else:
            self.head = entry
        existing.prev = entry

    def touch(self, radix_node: _CustomRadixNode):
        """Mark an evictable leaf as recently used.

        The caller is responsible for ensuring the node is eligible to be in
        the leaf LRU. Reordering is done by `last_access_ts`.
        """
        existing = self._find_entry(radix_node)
        if existing is not None:
            self._unlink(existing)

        entry = _CustomLruEntry(radix_node=radix_node)
        cur = self.head
        while cur is not None and cur.radix_node.last_access_ts <= radix_node.last_access_ts:
            cur = cur.next
        self._insert_before(cur, entry)

    def remove(self, radix_node: _CustomRadixNode):
        """Remove a node from LRU tracking.

        Typical reasons:
        - the node is no longer a leaf
        - it became protected
        - it was evicted
        """
        entry = self._find_entry(radix_node)
        if entry is not None:
            self._unlink(entry)

    def pop_oldest(self) -> Optional[_CustomRadixNode]:
        """Return the least-recently-used evictable leaf."""
        if self.head is None:
            return None
        entry = self.head
        self._unlink(entry)
        return entry.radix_node


class CustomRadixCacheImpl(BasePrefixCache):
    """Custom radix-tree prefix cache for the page-size-1 execution path.

    The outer contract follows `BasePrefixCache`, but the internal structure is
    intentionally narrower than the production implementation. The cache is
    built around the core RadixAttention ideas:
    - nodes represent shared prefixes
    - edges are compressed token spans rather than single-token trie edges
    - insertion may split an existing edge/node when prefixes diverge
    - each node stores only its own local token span and local KV span
    - eviction should prefer leaves so shared ancestors survive longer
    """

    def __init__(self, params: CacheInitParams):
        logger.warning(
            "[RADIX_CACHE_IMPL] CUSTOM RADIX_CACHE.PY source=%s",
            getattr(params, "radix_cache_source", "main"),
        )

        # These public fields are part of the outer contract expected by the
        # scheduler/runtime code. Keep them available even if your inner design
        # changes completely.
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events

        if self.page_size != 1:
            raise ValueError(
                f"CustomRadixCacheImpl currently only supports page_size=1, got {self.page_size}"
            )

        if self.token_to_kv_pool_allocator is not None:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        # Outer runtime code expects the cache to be able to report roughly how
        # much of the cache is currently protected versus evictable. These are
        # intentionally simple placeholders, not a recommendation for how your
        # final bookkeeping should look.
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.kv_event_queue = []
        self._access_clock = 0

        self.reset()

    def _empty_match_result(self) -> MatchResult:
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
        )

    def _next_access_ts(self) -> int:
        self._access_clock += 1
        return self._access_clock

    @staticmethod
    def _lcp_len(segment: tuple[int, ...], token_ids: list[int]) -> int:
        i = 0
        upper = min(len(segment), len(token_ids))
        while i < upper and segment[i] == token_ids[i]:
            i += 1
        return i

    @staticmethod
    def _is_leaf(node: _CustomRadixNode) -> bool:
        return len(node.children) == 0

    @staticmethod
    def _segment_len(node: _CustomRadixNode) -> int:
        return len(node.token_segment)

    def _is_evictable_leaf(self, node: _CustomRadixNode) -> bool:
        return (
            node is not self.root_node
            and self._is_leaf(node)
            and node.lock_ref == 0
            and node.kv_indices is not None
            and len(node.kv_indices) > 0
        )

    def _update_leaf_lru_membership(self, node: Optional[_CustomRadixNode]):
        if node is None or node is self.root_node:
            return
        if self._is_evictable_leaf(node):
            self.leaf_lru.touch(node)
        else:
            self.leaf_lru.remove(node)

    def _refresh_path_access(self, path: list[_CustomRadixNode]):
        if not path:
            return
        access_ts = self._next_access_ts()
        for node in path:
            node.last_access_ts = access_ts
        self._update_leaf_lru_membership(path[-1])

    def _path_to_node(self, node: _CustomRadixNode) -> list[_CustomRadixNode]:
        path: list[_CustomRadixNode] = []
        while node is not None and node is not self.root_node:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def _collect_path_kv(self, path: list[_CustomRadixNode]) -> torch.Tensor:
        chunks = [
            node.kv_indices
            for node in path
            if node.kv_indices is not None and len(node.kv_indices) > 0
        ]
        if not chunks:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        return torch.cat(chunks)

    def _create_child(
        self,
        parent: _CustomRadixNode,
        token_segment: list[int] | tuple[int, ...],
        kv_indices: torch.Tensor,
        access_ts: int,
    ) -> _CustomRadixNode:
        node = _CustomRadixNode(
            token_segment=tuple(token_segment),
            parent=parent,
            kv_indices=kv_indices.to(dtype=torch.int64, copy=True),
            lock_ref=0,
            last_access_ts=access_ts,
        )
        parent.children[node.token_segment[0]] = node
        self.evictable_size_ += len(node.token_segment)
        self._update_leaf_lru_membership(node)
        self._update_leaf_lru_membership(parent)
        return node

    def _split_child(
        self,
        parent: _CustomRadixNode,
        child: _CustomRadixNode,
        split_len: int,
    ) -> _CustomRadixNode:
        prefix_segment = child.token_segment[:split_len]
        suffix_segment = child.token_segment[split_len:]
        prefix_kv = child.kv_indices[:split_len].clone()
        suffix_kv = child.kv_indices[split_len:].clone()

        new_parent = _CustomRadixNode(
            token_segment=prefix_segment,
            parent=parent,
            kv_indices=prefix_kv,
            lock_ref=child.lock_ref,
            last_access_ts=child.last_access_ts,
        )

        parent.children[prefix_segment[0]] = new_parent
        child.parent = new_parent
        child.token_segment = suffix_segment
        child.kv_indices = suffix_kv
        new_parent.children[suffix_segment[0]] = child

        self._update_leaf_lru_membership(child)
        self._update_leaf_lru_membership(new_parent)
        return new_parent

    def _remove_leaf(self, node: _CustomRadixNode):
        assert node.parent is not None, "Root cannot be removed"
        assert self._is_leaf(node), "Only leaves can be removed"
        self.leaf_lru.remove(node)
        del node.parent.children[node.token_segment[0]]
        if node.lock_ref == 0:
            self.evictable_size_ -= len(node.token_segment)
        else:
            self.protected_size_ -= len(node.token_segment)
        self._update_leaf_lru_membership(node.parent)

    def _lookup_longest_prefix(
        self, key: RadixKey
    ) -> tuple[torch.Tensor, Any]:
        """Traverse the radix tree and return the longest reusable prefix.

        This is the read-side operation behind RadixAttention.
        Given a token sequence, answer:
        - "How much of this prefix has already been seen before?"
        - "Which cached KV indices correspond to that reusable prefix?"
        - "Which radix node is the deepest matched cached node?"

        In a radix tree, matching is edge-based rather than token-by-token
        trie traversal. That means the method should compare the remaining
        request suffix against each child node's compressed `token_segment`,
        then:
        - consume a full segment when it matches fully
        - stop when the next segment only partially matches
        - return the deepest fully matched cached node

        Because nodes store only local KV spans, the reusable KV result is
        normally assembled by concatenating the `kv_indices` from each matched
        node along the path from the root to the deepest matched node.

        This implementation supports only `page_size=1`, so matching is done
        directly at token granularity rather than page granularity.

        If a matched node remains an evictable leaf after access, it is a good
        candidate to `touch` in the leaf-LRU structure.
        """
        node = self.root_node
        path: list[_CustomRadixNode] = []
        remaining = key.token_ids

        while remaining:
            child = node.children.get(remaining[0])
            if child is None:
                break

            prefix_len = self._lcp_len(child.token_segment, remaining)
            if prefix_len < len(child.token_segment):
                break

            path.append(child)
            node = child
            remaining = remaining[prefix_len:]

        self._refresh_path_access(path)
        return self._collect_path_kv(path), node

    def _insert_prefix(
        self, key: RadixKey, value: torch.Tensor
    ) -> tuple[int, Any]:
        """Insert a prefix into the radix tree, splitting nodes if needed.

        This is the write-side counterpart to lookup.
        A request has produced KV for some prefix, and this method decides how
        that prefix is represented in the radix tree.

        The key radix-tree behavior to preserve is:
        - follow the existing shared prefix as far as possible
        - if an existing edge only partially matches, split that node
        - attach new suffix material as a new child
        - keep token spans and KV spans aligned locally at each node
        - after a split, slice both `token_segment` and `kv_indices` in parallel

        This is the method where the compressed-edge nature of a radix tree
        matters most.

        This is also where leaf-LRU membership changes usually happen:
        - a node that gains a child stops being a leaf and should leave LRU
        - a newly created cacheable leaf may need to enter LRU if unlocked
        """
        if len(key) != len(value):
            raise ValueError(
                f"Custom radix insertion requires aligned token/KV spans, got {len(key)=} and {len(value)=}"
            )

        node = self.root_node
        remaining_tokens = key.token_ids
        remaining_values = value.to(dtype=torch.int64, copy=False)
        matched_len = 0
        path: list[_CustomRadixNode] = []
        access_ts = self._next_access_ts()

        while remaining_tokens:
            child = node.children.get(remaining_tokens[0])
            if child is None:
                new_leaf = self._create_child(
                    node,
                    remaining_tokens,
                    remaining_values,
                    access_ts=access_ts,
                )
                path.append(new_leaf)
                self._refresh_path_access(path)
                return matched_len, new_leaf

            prefix_len = self._lcp_len(child.token_segment, remaining_tokens)
            if prefix_len == len(child.token_segment):
                matched_len += prefix_len
                child.last_access_ts = access_ts
                path.append(child)
                node = child
                remaining_tokens = remaining_tokens[prefix_len:]
                remaining_values = remaining_values[prefix_len:]
                continue

            split_node = self._split_child(node, child, prefix_len)
            split_node.last_access_ts = access_ts
            matched_len += prefix_len
            path.append(split_node)
            node = split_node
            remaining_tokens = remaining_tokens[prefix_len:]
            remaining_values = remaining_values[prefix_len:]

            if remaining_tokens:
                new_leaf = self._create_child(
                    node,
                    remaining_tokens,
                    remaining_values,
                    access_ts=access_ts,
                )
                path.append(new_leaf)
                self._refresh_path_access(path)
                return matched_len, new_leaf

            self._refresh_path_access(path)
            return matched_len, split_node

        self._refresh_path_access(path)
        return matched_len, node

    def insert(self, key: RadixKey, value=None, **kwargs) -> int:
        if self.disable:
            return 0

        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64, device=self.device)

        matched_len, _ = self._insert_prefix(key, value)
        return matched_len

    def reset(self):
        """Initialize an empty cache state.

        For a radix tree, the simplest empty state is:
        - a root sentinel with an empty token segment
        - no children
        - no cached KV stored at the root
        - an empty leaf-LRU tracker
        - zeroed size / protection / event bookkeeping
        """
        self.root_node = _CustomRadixNode()
        self.leaf_lru = _CustomLeafLru()
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.kv_event_queue = []
        self._access_clock = 0

    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
        """Public longest-prefix lookup used by the outer runtime.

        Think of this as the adapter between your custom cache design and the
        rest of the runtime.

        Upstream code does not care how you store the cache. It only expects:
        - a tensor of reusable KV indices
        - a "last matched node" object

        In other words, this method should translate your internal lookup result
        into the `MatchResult` shape the runtime already knows how to consume.
        """
        if self.disable or len(key) == 0:
            return self._empty_match_result()

        matched_indices, last_node = self._lookup_longest_prefix(key)
        return MatchResult(
            device_indices=matched_indices,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def cache_finished_req(self, req, is_insert: bool = True, **kwargs):
        """Finalize a request and commit its KV to cache.

        This method is about the end of a request lifecycle.
        By the time it runs, the request has finished generating, and you need
        to decide what part of its KV becomes durable radix-tree state.

        Good questions to answer when implementing it:
        - Which prefix should remain cacheable for future requests?
        - Which KV indices are now owned by the cache vs still owned by the req?
        - Which temporary protection on the matched radix node should now be released?

        Since the tree stores node-local KV spans rather than full-prefix KV at
        every node, this method should think in terms of attaching only the new
        suffix material that was produced for the request.

        Upstream mostly cares that memory and request state remain coherent. It
        does not require you to mirror SGLang's internal sequence of steps.
        """
        kv_committed_len = req.pop_committed_kv_cache()

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        radix_key = RadixKey(token_ids, req.extra_key)
        values = kv_indices[: len(token_ids)].to(dtype=torch.int64, copy=True)

        if is_insert:
            matched_prefix_len, new_last_node = self._insert_prefix(radix_key, values)
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : matched_prefix_len]
            )
        else:
            new_last_node = req.last_node
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : len(token_ids)]
            )

        self.req_to_token_pool.free(req.req_pool_idx)
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)
        req.last_node = new_last_node

    def cache_unfinished_req(self, req, **kwargs):
        """Make the current request prefix visible in the radix tree early.

        This matters because multi-turn and iterative generation often need the
        cache to become visible before the request is fully complete.

        Conceptually:
        - the request has produced new KV
        - generation is not fully over
        - but the currently known prefix should already be reusable

        The most important part here is not the exact storage strategy. It is
        making the request object and cache agree about what prefix is now
        protected and reusable.

        In practice, the outer runtime will care that these fields stay
        coherent:
        - `req.prefix_indices`
        - `req.last_node`
        - `req.cache_protected_len`

        At the radix-tree level, this usually means:
        - insert the currently known prefix
        - update the request to point at the resulting radix node
        - hand protection from the old matched node to the new one if the
          request extended the cached prefix

        That protection handoff may also imply leaf-LRU updates, because a node
        becoming protected should generally stop being evictable.
        """
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        radix_key = RadixKey(token_ids, req.extra_key)
        values = kv_indices[: len(token_ids)].to(dtype=torch.int64, copy=True)

        old_last_node = req.last_node
        matched_prefix_len, _ = self._insert_prefix(radix_key, values)
        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : matched_prefix_len]
        )

        match_result = self.match_prefix(radix_key)
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        assert len(new_indices) == len(token_ids), (
            f"Custom radix cache expects a full prefix hit after insertion, "
            f"got {len(new_indices)=} and {len(token_ids)=}"
        )

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        req.cache_protected_len = len(new_indices)
        req.prefix_indices = new_indices

        if old_last_node is not None and old_last_node is not new_last_node:
            self.dec_lock_ref(old_last_node)
            self.inc_lock_ref(new_last_node)
        elif old_last_node is None and new_last_node is not None:
            self.inc_lock_ref(new_last_node)

        req.last_node = new_last_node

    def evict(self, num_tokens: int):
        """Evict cached radix-tree entries when memory pressure requires it.

        This is the policy side of the cache, separate from lookup/insertion.
        It is reasonable to keep this crude at first and improve it only after
        the core reuse path works.

        The paper's intended story is leaf-first eviction under an LRU policy:
        - prefer evicting leaves that are not actively protected
        - this preserves shared ancestors for as long as possible
        - freeing a leaf may turn its parent into a leaf, making that parent
          evictable later if it is also unused

        The natural driver for this method is `_CustomLeafLru.pop_oldest()`.
        """
        if self.disable:
            return

        start_time = time.perf_counter()
        num_evicted = 0
        while num_evicted < num_tokens:
            leaf = self.leaf_lru.pop_oldest()
            if leaf is None:
                break
            if leaf.kv_indices is not None and len(leaf.kv_indices) > 0:
                self.token_to_kv_pool_allocator.free(leaf.kv_indices)
                num_evicted += len(leaf.kv_indices)
            parent = leaf.parent
            self._remove_leaf(leaf)
            if parent is not None:
                self._update_leaf_lru_membership(parent)

        self.update_eviction_metrics(num_evicted, start_time)

    def inc_lock_ref(self, node: Any):
        """Mark a matched prefix as protected.

        Conceptually, the runtime is telling the cache:
        - "this prefix is being actively used right now"
        - "do not let eviction reclaim it"

        The implementation uses a simple ancestor refcount scheme. A node is
        evictable only when its refcount returns to zero.

        If a node becomes protected and it is currently tracked in the leaf-LRU,
        it should usually be removed from that LRU structure.
        """
        if self.disable or node is None or node is self.root_node:
            return 0

        delta = 0
        cur = node
        while cur is not None and cur is not self.root_node:
            if cur.lock_ref == 0:
                self.evictable_size_ -= len(cur.token_segment)
                self.protected_size_ += len(cur.token_segment)
                delta -= len(cur.token_segment)
                self.leaf_lru.remove(cur)
            cur.lock_ref += 1
            cur = cur.parent
        return delta

    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: Optional[str] = None):
        """Release protection previously added by `inc_lock_ref`.

        This is the inverse of `inc_lock_ref`.
        Once a request stops actively relying on a prefix, that prefix should be
        allowed to become evictable again according to your policy.

        If the node becomes an unlocked leaf again, this is where it may rejoin
        the leaf-LRU structure.
        """
        if self.disable or node is None or node is self.root_node:
            return 0

        delta = 0
        cur = node
        while cur is not None and cur is not self.root_node:
            if cur.lock_ref <= 0:
                raise RuntimeError("Custom radix lock_ref underflow")
            if cur.lock_ref == 1:
                self.evictable_size_ += len(cur.token_segment)
                self.protected_size_ -= len(cur.token_segment)
                delta += len(cur.token_segment)
            cur.lock_ref -= 1
            self._update_leaf_lru_membership(cur)
            cur = cur.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        return self.protected_size_

    def total_size(self):
        """Return total cached KV represented by your structure.

        This is mostly for diagnostics and runtime sanity checks. It does not
        need to expose every nuance of your implementation, but it should track
        the amount of node-local cached KV currently represented by the tree.
        """
        total = 0
        stack = list(self.root_node.children.values())
        while stack:
            node = stack.pop()
            if node.kv_indices is not None:
                total += len(node.kv_indices)
            stack.extend(node.children.values())
        return total

    def pretty_print(self):
        """Optional debugging helper.

        This is purely for your own debugging. It can stay minimal until the
        real cache logic exists.
        """
        lines = []
        stack = [(self.root_node, 0)]
        while stack:
            node, depth = stack.pop()
            if node is self.root_node:
                lines.append("root")
            else:
                lines.append(
                    f"{' ' * depth}{list(node.token_segment[:8])}{'...' if len(node.token_segment) > 8 else ''} "
                    f"len={len(node.token_segment)} lock_ref={node.lock_ref} ts={node.last_access_ts}"
                )
            children = list(node.children.values())
            for child in reversed(children):
                stack.append((child, depth + 2))
        lines.append(f"evictable={self.evictable_size_} protected={self.protected_size_}")
        return "\n".join(lines)

    def take_events(self):
        """Keep the same outer method even if custom cache ignores KV events."""
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
