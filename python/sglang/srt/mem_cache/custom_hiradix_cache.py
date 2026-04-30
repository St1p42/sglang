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
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class CustomHiRadixCache(RadixCache):
    """A narrow CPU+GPU-only HiCache implementation for experiments.

    This class intentionally supports a much smaller surface than the vanilla
    implementation:
    - no storage backends
    - no storage prefetch
    - no storage metrics/control queues

    The only experiment knob inside custom HiCache is the backup policy:
    fixed vs adaptive.
    """

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self._validate_custom_config(server_args)
        print(
            ("*" * 2000)
            + " ENTERED CUSTOM HICACHE IMPLEMENTATION "
            + ("&" * 2000)
        )
        logger.warning(
            "Confirmed active HiCache implementation: custom (%s) backup_policy=%s.",
            self.__class__.__name__,
            server_args.hicache_backup_policy,
        )

        if server_args.hicache_io_backend == "direct":
            if server_args.hicache_mem_layout == "page_first":
                server_args.hicache_mem_layout = "page_first_direct"
                logger.warning(
                    "Page first layout is not supported with direct IO backend, switching to page first direct layout"
                )

        self.page_size = params.page_size
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
        self.hicache_backup_policy = server_args.hicache_backup_policy

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

        self.ongoing_write_through: dict[int, TreeNode] = {}
        self.ongoing_load_back: dict[int, TreeNode] = {}
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10

        super().__init__(params=params)

    def _validate_custom_config(self, server_args: ServerArgs) -> None:
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
        if server_args.hicache_backup_policy not in ("fixed", "adaptive"):
            raise ValueError(
                "Unsupported hicache backup policy: "
                f"{server_args.hicache_backup_policy}"
            )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        super().reset()

    def clear_storage_backend(self) -> bool:
        logger.warning("CustomHiRadixCache does not enable a storage backend.")
        return False

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
        radix_key = RadixKey(token_ids, extra_key, is_bigram=self.is_eagle)
        radix_key, values = self.maybe_bigram_convert(radix_key, kv_indices)
        assert values is not None
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

    def write_backup(self, node: TreeNode, write_back=False):
        backup_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
        )
        if backup_indices is None:
            self.evict_host(len(node.value))
            backup_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
            )
        if backup_indices is None:
            return 0

        node.host_value = backup_indices
        self.ongoing_write_through[node.id] = node
        if not write_back:
            self.inc_lock_ref(node)
        return len(backup_indices)

    def _get_host_memory_pressure(self) -> float:
        host_pool = self.cache_controller.mem_pool_host
        if host_pool.size <= 0:
            return 1.0
        available_capacity = host_pool.available_size()
        used_ratio = 1.0 - (available_capacity / host_pool.size)
        return max(0.0, min(1.0, used_ratio))

    def _get_backup_threshold(self, node: TreeNode) -> int:
        if self.hicache_backup_policy == "fixed":
            return self.write_through_threshold

        pressure_ratio = self._get_host_memory_pressure()
        threshold = self.write_through_threshold
        if pressure_ratio >= 0.90:
            threshold += 2
        elif pressure_ratio >= 0.75:
            threshold += 1
        if len(node.key) >= max(self.page_size * 16, 64):
            threshold = max(1, threshold - 1)
        return threshold

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        if self.cache_controller.write_policy == "write_back" or chunked:
            return

        node.hit_count += 1
        if not node.backuped and node.hit_count >= self._get_backup_threshold(node):
            self.write_backup(node)

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

    def _delete_leaf(self, node: TreeNode):
        for key, child in node.parent.children.items():
            if child == node:
                break
        del node.parent.children[key]
        self.evictable_size_ -= len(node.key)

    def _collect_device_leaves(self):
        leaves = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node == self.root_node:
                stack.extend(node.children.values())
                continue
            if node.evicted:
                continue
            live_children = [child for child in node.children.values() if not child.evicted]
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
                if node.lock_ref == 0:
                    leaves.append(node)
            else:
                stack.extend(node.children.values())
        return leaves

    def _evict_backuped(self, node: TreeNode):
        num_evicted = self.cache_controller.evict_device(node.value)
        self.evictable_size_ -= num_evicted
        node.value = None
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def evict(self, num_tokens: int):
        start_time = time.perf_counter()
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node)
            for node in self._collect_device_leaves()
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        write_back_nodes = []
        while num_evicted < num_tokens and eviction_heap:
            _priority, node = heapq.heappop(eviction_heap)
            if node.lock_ref > 0:
                continue

            if not node.backuped:
                if self.cache_controller.write_policy == "write_back":
                    num_evicted += self.write_backup(node, write_back=True)
                    write_back_nodes.append(node)
                else:
                    num_evicted += self._evict_regular(node)
            else:
                num_evicted += self._evict_backuped(node)

            for child in node.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                heapq.heappush(
                    eviction_heap,
                    (self.eviction_strategy.get_priority(node.parent), node.parent),
                )

        if self.cache_controller.write_policy == "write_back":
            self.writing_check(write_back=True)
            for node in write_back_nodes:
                self._evict_backuped(node)

        self.update_eviction_metrics(num_evicted, start_time)

    def evict_host(self, num_tokens: int):
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node)
            for node in self._collect_host_leaves()
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and eviction_heap:
            _priority, node = heapq.heappop(eviction_heap)
            if node == self.root_node or not node.evicted:
                continue
            if node.host_ref_counter > 0:
                continue

            num_evicted += self.cache_controller.evict_host(node.host_value)
            for key, child in node.parent.children.items():
                if child == node:
                    break
            del node.parent.children[key]

            if len(node.parent.children) == 0 and node.parent.evicted:
                heapq.heappush(
                    eviction_heap,
                    (self.eviction_strategy.get_priority(node.parent), node.parent),
                )

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        start_time = time.perf_counter()
        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert node.backuped, "Evicted node must have a host backup"
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
            host_indices=host_indices, node_id=last_hit_node.id
        )
        if device_indices is None:
            self.evict(len(host_indices))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id
            )
        self.dec_lock_ref(ancestor_node)
        if device_indices is None:
            return None

        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        for cur in nodes_to_load:
            cur.value = device_indices[offset : offset + len(cur.host_value)]
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
        last_node: TreeNode,
        host_hit_length: int,
        mem_quota: Optional[int] = None,
    ):
        _ = host_hit_length
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                return loading_values, last_node
            while last_node.evicted:
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
        key, _ = self.maybe_bigram_convert(key)
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
        while device_tail_node.evicted:
            host_hit_length += len(device_tail_node.host_value)
            device_tail_node = device_tail_node.parent
        while not host_tail_node.backuped:
            host_tail_node = host_tail_node.parent

        return MatchResult(
            device_indices=device_indices,
            last_device_node=device_tail_node,
            last_host_node=host_tail_node,
            host_hit_length=host_hit_length,
        )

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        values = []
        cursor = node
        cursor.last_access_time = time.monotonic()
        branch_key = self.get_child_key_fn(key)

        while len(key) > 0 and branch_key in cursor.children:
            child = cursor.children[branch_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                cursor = self._split_node(child.key, child, prefix_len)
                if not cursor.evicted:
                    values.append(cursor.value)
                break

            if not child.evicted:
                values.append(child.value)
            cursor = child
            key = key[prefix_len:]
            if len(key):
                branch_key = self.get_child_key_fn(key)

        return values, cursor

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        new_node = TreeNode(priority=child.priority)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]

        if child.backuped:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]
        else:
            new_node.host_value = None

        new_node.hash_value = []
        child.hash_value = []
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    def insert(
        self,
        key: RadixKey,
        value=None,
        chunked: bool = False,
        priority: int | None = None,
    ):
        priority = 0 if priority is None else priority
        key, value = self.maybe_bigram_convert(key, value)
        if len(key) == 0:
            return 0

        if self.is_eagle and value is not None:
            value = value[: len(key)]

        cursor = self.root_node
        branch_key = self.get_child_key_fn(key)
        matched_prefix_len = 0

        while len(key) > 0 and branch_key in cursor.children:
            cursor = cursor.children[branch_key]
            cursor.last_access_time = time.monotonic()
            cursor.priority = max(cursor.priority, priority)
            prefix_len = self.key_match_fn(cursor.key, key)

            if prefix_len == len(cursor.key):
                if cursor.evicted:
                    cursor.value = value[:prefix_len]
                    self.evictable_size_ += len(cursor.value)
                else:
                    self._inc_hit_count(cursor, chunked)
                    matched_prefix_len += prefix_len
            else:
                cursor = self._split_node(cursor.key, cursor, prefix_len)
                cursor.priority = max(cursor.priority, priority)
                if cursor.evicted:
                    cursor.value = value[:prefix_len]
                    self.evictable_size_ += len(cursor.value)
                else:
                    self._inc_hit_count(cursor, chunked)
                    matched_prefix_len += prefix_len

            key = key[prefix_len:]
            value = value[prefix_len:]
            if len(key):
                branch_key = self.get_child_key_fn(key)

        if len(key):
            appended_node = TreeNode(priority=priority)
            appended_node.parent = cursor
            appended_node.key = key
            appended_node.value = value
            cursor.children[branch_key] = appended_node
            self.evictable_size_ += len(value)
            if self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(appended_node, chunked)

        return matched_prefix_len
