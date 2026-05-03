from __future__ import annotations

import heapq
import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.radix_cache_custom import _CustomRadixNode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class CustomHiRadixCache(RadixCache):
    """CPU+GPU-only HiCache implementation for experiments on custom radix cache."""

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self._validate_custom_config(params, server_args)
        logger.warning("Using custom HiCache implementation.")

        if server_args.hicache_io_backend == "direct":
            if server_args.hicache_mem_layout == "page_first":
                server_args.hicache_mem_layout = "page_first_direct"
                logger.warning(
                    "Page first layout is not supported with direct IO backend, switching to page first direct layout"
                )

        self.page_size = params.page_size
        self.disable_finished_insert = params.disable_finished_insert
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        else:
            raise ValueError("CustomHiRadixCache only supports MHA and MLA pools")

        self.tp_group = params.tp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.enable_storage = False
        self.enable_storage_metrics = False
        self.hicache_storage_pass_prefix_keys = False

        self.load_cache_event = threading.Event()
        self.cache_controller = HiCacheController(
            params.token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            self.page_size,
            self.tp_group,
            load_cache_event=self.load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=None,
            prefetch_threshold=0,
            model_name=server_args.served_model_name,
            storage_backend_extra_config={},
        )

        self.ongoing_write_through: dict[int, _CustomRadixNode] = {}
        self.ongoing_load_back: dict[int, _CustomRadixNode] = {}
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.custom_backup_policy = getattr(
            server_args, "hicache_custom_backup_policy", "length_gated"
        )
        self.min_backup_len = max(
            1, getattr(server_args, "hicache_min_backup_len", 128)
        )
        self.load_back_threshold = 10
        self.host_backup_tokens = 0
        self.host_backup_count = 0
        self.host_backup_total_span_len = 0
        self.host_backup_skipped_by_length_count = 0
        self.host_backup_skipped_by_length_tokens = 0

        super().__init__(params=params)
        self.evictable_size_ = self.impl.evictable_size_
        self.protected_size_ = self.impl.protected_size_
        self._ensure_node_state(self.root_node, extra_key=None)

        if self.custom_backup_policy == "baseline":
            logger.info("Using custom HiCache with baseline host backup policy")
        else:
            logger.info(
                "Using custom HiCache with length-gated host backup: min_backup_len=%d",
                self.min_backup_len,
            )

    def _validate_custom_config(
        self, params: CacheInitParams, server_args: ServerArgs
    ) -> None:
        if params.is_eagle:
            raise ValueError(
                "CustomHiRadixCache does not support EAGLE / bigram radix keys."
            )
        if server_args.hicache_storage_backend is not None:
            raise ValueError(
                "CustomHiRadixCache is CPU+GPU-only and does not support "
                "--hicache-storage-backend."
            )
        if server_args.hicache_storage_backend_extra_config is not None:
            raise ValueError(
                "CustomHiRadixCache does not support "
                "--hicache-storage-backend-extra-config."
            )

    def reset(self):
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        super().reset()
        self.evictable_size_ = self.impl.evictable_size_
        self.protected_size_ = self.impl.protected_size_
        self._ensure_node_state(self.root_node, extra_key=None)

    def clear_storage_backend(self) -> bool:
        logger.warning("CustomHiRadixCache does not enable a storage backend.")
        return False

    @staticmethod
    def _child_key(key: RadixKey) -> int:
        return key.token_ids[0]

    @staticmethod
    def _prefix_match_len(node: _CustomRadixNode, key: RadixKey) -> int:
        upper = min(len(node.token_segment), len(key.token_ids))
        idx = 0
        while idx < upper and node.token_segment[idx] == key.token_ids[idx]:
            idx += 1
        return idx

    def _set_device_indices(
        self, node: _CustomRadixNode, kv_indices: Optional[torch.Tensor]
    ) -> None:
        node.kv_indices = kv_indices
        node.value = kv_indices

    def _set_host_indices(
        self, node: _CustomRadixNode, host_indices: Optional[torch.Tensor]
    ) -> None:
        node.host_value = host_indices

    def _node_is_evicted(self, node: _CustomRadixNode) -> bool:
        return node is not self.root_node and node.kv_indices is None

    def _node_is_backuped(self, node: _CustomRadixNode) -> bool:
        return getattr(node, "host_value", None) is not None

    def _node_key(self, node: _CustomRadixNode) -> RadixKey:
        return RadixKey(list(node.token_segment), getattr(node, "extra_key", None))

    def _node_token_len(self, node: _CustomRadixNode) -> int:
        return len(node.token_segment)

    def _ensure_node_state(
        self, node: _CustomRadixNode, extra_key: Optional[str]
    ) -> _CustomRadixNode:
        if not hasattr(node, "extra_key"):
            node.extra_key = extra_key
        elif extra_key is not None:
            node.extra_key = extra_key

        if not hasattr(node, "value"):
            node.value = node.kv_indices
        else:
            node.value = node.kv_indices

        if not hasattr(node, "host_value"):
            node.host_value = None
        if not hasattr(node, "hit_count"):
            node.hit_count = 0
        if not hasattr(node, "host_ref_counter"):
            node.host_ref_counter = 0
        if not hasattr(node, "priority"):
            node.priority = 0
        if not hasattr(node, "hash_value"):
            node.hash_value = []
        if not hasattr(node, "last_access_time"):
            node.last_access_time = time.monotonic()
        return node

    @staticmethod
    def _node_priority(node: _CustomRadixNode) -> tuple[int, float]:
        return (-getattr(node, "priority", 0), getattr(node, "last_access_ts", 0))

    def inc_lock_ref(self, node):
        delta = self.impl.inc_lock_ref(node)
        self.evictable_size_ += delta
        self.protected_size_ -= delta
        return delta

    def dec_lock_ref(self, node, swa_uuid_for_lock=None):
        delta = self.impl.dec_lock_ref(node, swa_uuid_for_lock=swa_uuid_for_lock)
        self.evictable_size_ += delta
        self.protected_size_ -= delta
        return delta

    def _page_align_len(self, token_count: int) -> int:
        if self.page_size == 1:
            return token_count
        return token_count // self.page_size * self.page_size

    def _aligned_radix_inputs(
        self,
        token_ids: list[int],
        kv_indices: torch.Tensor,
        extra_key,
    ) -> tuple[RadixKey, torch.Tensor]:
        radix_key = RadixKey(token_ids, extra_key, is_bigram=False)
        values = kv_indices
        aligned_len = self._page_align_len(len(radix_key))
        if aligned_len != len(radix_key):
            radix_key = RadixKey(
                radix_key.token_ids[:aligned_len],
                radix_key.extra_key,
                is_bigram=radix_key.is_bigram,
            )
            values = values[:aligned_len]
        return radix_key, values

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        if self.disable_finished_insert:
            is_insert = False

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
        radix_key, values = self._aligned_radix_inputs(
            req.fill_ids,
            kv_indices[: len(req.fill_ids)].to(dtype=torch.int64, copy=True),
            req.extra_key,
        )

        if is_insert:
            new_prefix_len = self.insert(
                radix_key,
                values,
                priority=getattr(req, "priority", 0) or 0,
            )
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : new_prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : len(radix_key)]
            )

        self.token_to_kv_pool_allocator.free(kv_indices[len(radix_key) :])
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        if self.disable:
            return

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]
        radix_key, values = self._aligned_radix_inputs(
            req.fill_ids,
            kv_indices[: len(req.fill_ids)].to(dtype=torch.int64, copy=True),
            req.extra_key,
        )

        new_prefix_len = self.insert(
            radix_key,
            values,
            chunked=chunked,
            priority=getattr(req, "priority", 0) or 0,
        )
        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )

        match_result = self.match_prefix(radix_key)
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        assert len(new_indices) == len(radix_key), (
            f"{len(new_indices)=}, {len(radix_key)=}"
        )

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )
        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node

    def write_backup(self, node: _CustomRadixNode, write_back=False):
        admit_len = len(node.kv_indices)
        backup_indices = self.cache_controller.write(
            device_indices=node.kv_indices,
            node_id=id(node),
        )
        if backup_indices is None:
            self.evict_host(admit_len)
            backup_indices = self.cache_controller.write(
                device_indices=node.kv_indices,
                node_id=id(node),
            )
        if backup_indices is None:
            return 0

        self._set_host_indices(node, backup_indices)
        self.host_backup_tokens += admit_len
        self.host_backup_count += 1
        self.host_backup_total_span_len += admit_len
        self.ongoing_write_through[id(node)] = node
        if not write_back:
            self.inc_lock_ref(node)
        return len(backup_indices)

    def get_runtime_stats(self) -> dict[str, float | int]:
        avg_host_backup_len = (
            self.host_backup_total_span_len / self.host_backup_count
            if self.host_backup_count > 0
            else 0.0
        )
        return {
            "host_backup_count": int(self.host_backup_count),
            "host_backup_tokens": int(self.host_backup_tokens),
            "host_backup_avg_len": float(avg_host_backup_len),
            "host_backup_skipped_by_length_count": int(
                self.host_backup_skipped_by_length_count
            ),
            "host_backup_skipped_by_length_tokens": int(
                self.host_backup_skipped_by_length_tokens
            ),
        }

    def _record_length_gate_skip(self, node: _CustomRadixNode) -> None:
        if self.custom_backup_policy != "length_gated":
            return
        skip_len = self._node_token_len(node)
        self.host_backup_skipped_by_length_count += 1
        self.host_backup_skipped_by_length_tokens += skip_len

    def _should_backup_to_host(self, node: _CustomRadixNode) -> bool:
        if self.custom_backup_policy == "baseline":
            return True
        return self._node_token_len(node) >= self.min_backup_len

    def _inc_hit_count(self, node: _CustomRadixNode, chunked=False):
        if self.cache_controller.write_policy == "write_back" or chunked:
            return

        node.hit_count += 1
        if not self._node_is_backuped(node) and node.hit_count >= self.write_through_threshold:
            if self._should_backup_to_host(node):
                self.write_backup(node)
            else:
                self._record_length_gate_skip(node)

    def writing_check(self, write_back=False):
        if write_back:
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        self.ongoing_write_through.pop(ack_id)
                self.cache_controller.ack_write_queue.clear()
            return

        if len(self.ongoing_write_through) == 0:
            return

        finish_count = 0
        for _, finish_event, _ in self.cache_controller.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )

        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                backuped_node = self.ongoing_write_through.pop(ack_id)
                self.dec_lock_ref(backuped_node)
            finish_count -= 1

    def loading_check(self):
        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
            if not finish_event.query():
                break
            finish_count += 1
            for ack_id in ack_list:
                end_node = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(end_node)

        del self.cache_controller.ack_load_queue[:finish_count]

    def evictable_size(self):
        return self.evictable_size_

    def _delete_leaf(self, node: _CustomRadixNode):
        for key, child in node.parent.children.items():
            if child == node:
                break
        del node.parent.children[key]
        self.evictable_size_ -= len(node.token_segment)

    def _collect_device_leaves(self):
        leaves = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node == self.root_node:
                stack.extend(node.children.values())
                continue
            if self._node_is_evicted(node):
                continue
            live_children = [
                child
                for child in node.children.values()
                if not self._node_is_evicted(child)
            ]
            if not live_children:
                leaves.append(node)
            else:
                stack.extend(live_children)
        return leaves

    def _collect_host_leaves(self):
        leaves = []
        stack = list(self.root_node.children.values())
        while stack:
            node = stack.pop()
            if len(node.children) == 0:
                if node.lock_ref == 0 and self._node_is_backuped(node):
                    leaves.append(node)
            else:
                stack.extend(node.children.values())
        return leaves

    def _evict_backuped(self, node: _CustomRadixNode):
        num_evicted = self.cache_controller.evict_device(node.kv_indices)
        self.evictable_size_ -= num_evicted
        self._set_device_indices(node, None)
        return num_evicted

    def _evict_regular(self, node: _CustomRadixNode):
        self.cache_controller.mem_pool_device_allocator.free(node.kv_indices)
        num_evicted = len(node.kv_indices)
        self._delete_leaf(node)
        return num_evicted

    def evict(self, num_tokens: int):
        start_time = time.perf_counter()
        eviction_heap = [
            (self._node_priority(node), id(node), node)
            for node in self._collect_device_leaves()
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        write_back_nodes = []
        while num_evicted < num_tokens and eviction_heap:
            _priority, _tie_breaker, node = heapq.heappop(eviction_heap)
            if node.lock_ref > 0:
                continue

            if not self._node_is_backuped(node):
                if self.cache_controller.write_policy == "write_back":
                    if self._should_backup_to_host(node):
                        num_evicted += self.write_backup(node, write_back=True)
                        write_back_nodes.append(node)
                    else:
                        self._record_length_gate_skip(node)
                        num_evicted += self._evict_regular(node)
                else:
                    num_evicted += self._evict_regular(node)
            else:
                num_evicted += self._evict_backuped(node)

            for child in node.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not self._node_is_evicted(child):
                    break
            else:
                heapq.heappush(
                    eviction_heap,
                    (self._node_priority(node.parent), id(node.parent), node.parent),
                )

        if self.cache_controller.write_policy == "write_back":
            self.writing_check(write_back=True)
            for node in write_back_nodes:
                self._evict_backuped(node)

        self.update_eviction_metrics(num_evicted, start_time)

    def evict_host(self, num_tokens: int):
        eviction_heap = [
            (self._node_priority(node), id(node), node)
            for node in self._collect_host_leaves()
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and eviction_heap:
            _priority, _tie_breaker, node = heapq.heappop(eviction_heap)
            if node == self.root_node or not self._node_is_evicted(node):
                continue
            if node.host_ref_counter > 0:
                continue

            num_evicted += self.cache_controller.evict_host(node.host_value)
            self._set_host_indices(node, None)
            for key, child in node.parent.children.items():
                if child == node:
                    break
            del node.parent.children[key]

            if len(node.parent.children) == 0 and self._node_is_evicted(node.parent):
                heapq.heappush(
                    eviction_heap,
                    (self._node_priority(node.parent), id(node.parent), node.parent),
                )

    def load_back(
        self, node: _CustomRadixNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        start_time = time.perf_counter()
        last_hit_node = node
        nodes_to_load = []
        while self._node_is_evicted(node):
            assert self._node_is_backuped(node), (
                "Evicted custom HiCache node must have a host backup"
            )
            nodes_to_load.insert(0, node)
            node = node.parent
        ancestor_node = node

        delta = self.inc_lock_ref(ancestor_node)
        host_indices = torch.cat([cur.host_value for cur in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            mem_quota is not None and len(host_indices) > mem_quota + delta
        ):
            self.dec_lock_ref(ancestor_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=id(last_hit_node)
        )
        if device_indices is None:
            self.evict(len(host_indices))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=id(last_hit_node)
            )
        self.dec_lock_ref(ancestor_node)
        if device_indices is None:
            return None

        self.ongoing_load_back[id(last_hit_node)] = last_hit_node
        offset = 0
        for cur in nodes_to_load:
            self._set_device_indices(
                cur, device_indices[offset : offset + len(cur.host_value)]
            )
            offset += len(cur.host_value)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))
        return device_indices

    def init_load_back(
        self,
        last_node: _CustomRadixNode,
        host_hit_length: int,
        mem_quota: Optional[int] = None,
    ):
        _ = host_hit_length
        if self._node_is_evicted(last_node):
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                return loading_values, last_node
            while self._node_is_evicted(last_node):
                last_node = last_node.parent

        return torch.empty((0,), dtype=torch.int64, device=self.device), last_node

    def ready_to_load_host_cache(self) -> int:
        return self.cache_controller.start_loading()

    def check_hicache_events(self):
        self.writing_check()
        self.loading_check()

    def prefetch_from_storage(self, *args, **kwargs):
        raise RuntimeError("CustomHiRadixCache does not support storage prefetch.")

    def match_prefix(self, key: RadixKey, **kwargs):
        empty_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_indices,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        if self.page_size != 1:
            key = key[: self._page_align_len(len(key))]

        matched_values, device_tail_node = self._match_prefix_helper(self.root_node, key)
        device_indices = torch.cat(matched_values) if matched_values else empty_indices

        host_hit_length = 0
        host_tail_node = device_tail_node
        while self._node_is_evicted(device_tail_node):
            if device_tail_node.host_value is not None:
                host_hit_length += len(device_tail_node.host_value)
            device_tail_node = device_tail_node.parent
        while host_tail_node is not self.root_node and not self._node_is_backuped(
            host_tail_node
        ):
            host_tail_node = host_tail_node.parent

        return MatchResult(
            device_indices=device_indices,
            last_device_node=device_tail_node,
            last_host_node=host_tail_node,
            host_hit_length=host_hit_length,
        )

    def _match_prefix_helper(self, node: _CustomRadixNode, key: RadixKey):
        values = []
        cursor = node
        cursor.last_access_time = time.monotonic()
        branch_key = self._child_key(key)

        while len(key) > 0 and branch_key in cursor.children:
            child = cursor.children[branch_key]
            self._ensure_node_state(child, extra_key=key.extra_key)
            child.last_access_time = time.monotonic()
            prefix_len = self._prefix_match_len(child, key)
            if prefix_len < len(child.token_segment):
                cursor = self._split_node(child, prefix_len, extra_key=key.extra_key)
                if not self._node_is_evicted(cursor):
                    values.append(cursor.kv_indices)
                break

            if not self._node_is_evicted(child):
                values.append(child.kv_indices)
            cursor = child
            key = key[prefix_len:]
            if len(key):
                branch_key = self._child_key(key)

        return values, cursor

    def _split_node(
        self, child: _CustomRadixNode, split_len: int, extra_key: Optional[str]
    ):
        key = self._node_key(child)
        new_node = _CustomRadixNode(
            token_segment=tuple(child.token_segment[:split_len]),
            parent=child.parent,
            kv_indices=None,
            lock_ref=child.lock_ref,
            last_access_ts=child.last_access_ts,
        )
        self._ensure_node_state(new_node, extra_key=extra_key)
        new_node.priority = child.priority
        new_node.children = {self._child_key(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.hit_count = child.hit_count

        if self._node_is_evicted(child):
            self._set_device_indices(new_node, None)
        else:
            self._set_device_indices(new_node, child.kv_indices[:split_len])
            self._set_device_indices(child, child.kv_indices[split_len:])

        if self._node_is_backuped(child):
            self._set_host_indices(new_node, child.host_value[:split_len])
            self._set_host_indices(child, child.host_value[split_len:])
        else:
            self._set_host_indices(new_node, None)

        new_node.hash_value = list(getattr(child, "hash_value", []))
        child.hash_value = []
        child.parent = new_node
        child.token_segment = tuple(child.token_segment[split_len:])
        child.extra_key = extra_key
        new_node.parent.children[self._child_key(key)] = new_node
        return new_node

    def insert(
        self,
        key: RadixKey,
        value=None,
        chunked: bool = False,
        priority: int | None = None,
    ):
        priority = 0 if priority is None else priority
        if len(key) == 0:
            return 0

        cursor = self.root_node
        branch_key = self._child_key(key)
        matched_prefix_len = 0

        while len(key) > 0 and branch_key in cursor.children:
            cursor = cursor.children[branch_key]
            self._ensure_node_state(cursor, extra_key=key.extra_key)
            cursor.last_access_time = time.monotonic()
            cursor.priority = max(cursor.priority, priority)
            prefix_len = self._prefix_match_len(cursor, key)

            if prefix_len == len(cursor.token_segment):
                if self._node_is_evicted(cursor):
                    self._set_device_indices(cursor, value[:prefix_len])
                    self.evictable_size_ += len(cursor.value)
                else:
                    self._inc_hit_count(cursor, chunked)
                    matched_prefix_len += prefix_len
            else:
                cursor = self._split_node(cursor, prefix_len, extra_key=key.extra_key)
                cursor.priority = max(cursor.priority, priority)
                if self._node_is_evicted(cursor):
                    self._set_device_indices(cursor, value[:prefix_len])
                    self.evictable_size_ += len(cursor.value)
                else:
                    self._inc_hit_count(cursor, chunked)
                    matched_prefix_len += prefix_len

            key = key[prefix_len:]
            value = value[prefix_len:]
            if len(key):
                branch_key = self._child_key(key)

        if len(key):
            appended_node = _CustomRadixNode(
                token_segment=tuple(key.token_ids),
                parent=cursor,
                kv_indices=value,
                lock_ref=0,
                last_access_ts=getattr(cursor, "last_access_ts", 0),
            )
            self._ensure_node_state(appended_node, extra_key=key.extra_key)
            appended_node.priority = priority
            cursor.children[branch_key] = appended_node
            self.evictable_size_ += len(value)
            if self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(appended_node, chunked)

        return matched_prefix_len
