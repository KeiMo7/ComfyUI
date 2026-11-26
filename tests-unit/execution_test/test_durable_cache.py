import asyncio
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy_execution.durable_cache as durable_cache
from comfy_api.latest._caching import CacheContext, CacheValue
from comfy_execution.cache_provider import _clear_cache_providers, _get_cache_providers
from comfy_execution.graph import DynamicPrompt
from execution import CacheEntry, CacheSet, CacheType

pytestmark = pytest.mark.asyncio

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_context(digest=DIGEST_A, scope="root", parent=None, class_type="TestNode", node_id="1"):
    return CacheContext(
        node_id=node_id,
        class_type=class_type,
        cache_key_hash=digest,
        scope=scope,
        parent_node_id=parent,
    )


def open_store(directory, **kwargs):
    kwargs.setdefault("max_bytes", 8 * 1024 ** 3)
    return durable_cache.open_durable_cache(str(directory), **kwargs)


def object_files(directory):
    objects_dir = os.path.join(str(directory), "objects")
    found = []
    if not os.path.isdir(objects_dir):
        return found
    for subdir in os.listdir(objects_dir):
        subdir_path = os.path.join(objects_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        found.extend(os.path.join(subdir_path, name) for name in os.listdir(subdir_path))
    return found


def read_index(directory):
    with open(os.path.join(str(directory), "index.json"), "r") as f:
        return json.load(f)


def payload_bytes(directory, digest):
    with open(os.path.join(str(directory), "objects", digest[:2], digest + ".bin"), "rb") as f:
        head = f.read(8)
        assert head[:4] == b"CDC1"
        meta_len = int.from_bytes(head[4:], "big")
        f.read(meta_len)
        return len(f.read())


@pytest.fixture(autouse=True)
def cleanup_registry():
    yield
    with durable_cache._registry_lock:
        stores = list(durable_cache._registry.values())
        durable_cache._registry.clear()
    for store, _refs in stores:
        durable_cache.unregister_cache_provider(store.provider)
        store._close_sync()
    _clear_cache_providers()


async def test_roundtrip_across_store_instances(tmp_path):
    value = CacheValue(
        outputs=[[{"samples": torch.zeros(2, 3, dtype=torch.float32),
                   "meta": {"nested": [1, 2.5, "x", None, True]}}],
                 [torch.ones(4, dtype=torch.bfloat16)]],
        ui={"images": [{"filename": "a.png", "type": "temp"}]},
    )
    store = open_store(tmp_path)
    await store.store(make_context(DIGEST_A), value)
    durable_cache.close_durable_cache(store)

    reopened = open_store(tmp_path)
    loaded = await reopened.lookup(make_context(DIGEST_A))
    assert loaded is not None
    assert loaded.ui == value.ui
    assert len(loaded.outputs) == 2
    samples = loaded.outputs[0][0]["samples"]
    assert samples.dtype == torch.float32
    assert samples.shape == (2, 3)
    assert torch.equal(samples, torch.zeros(2, 3, dtype=torch.float32))
    assert loaded.outputs[0][0]["meta"] == {"nested": [1, 2.5, "x", None, True]}
    bf16 = loaded.outputs[1][0]
    assert bf16.dtype == torch.bfloat16
    assert torch.equal(bf16, torch.ones(4, dtype=torch.bfloat16))


async def test_missing_ui_comes_back_as_empty_dict(tmp_path):
    store = open_store(tmp_path)
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[1]], ui=None))
    loaded = await store.lookup(make_context(DIGEST_A))
    assert loaded is not None
    assert loaded.ui == {}
    assert loaded.outputs == [[1]]


async def test_custom_object_is_skipped_and_writes_no_object(tmp_path):
    store = open_store(tmp_path)
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[object()]]))
    stats = store.snapshot()
    assert stats.skipped == 1
    assert stats.stores == 0
    assert object_files(tmp_path) == []
    assert await store.lookup(make_context(DIGEST_A)) is None


async def test_generator_output_is_skipped(tmp_path):
    store = open_store(tmp_path)
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[(x for x in range(3))]]))
    assert store.snapshot().skipped == 1
    assert object_files(tmp_path) == []


async def test_recursive_reference_is_skipped(tmp_path):
    looped = []
    looped.append(looped)
    store = open_store(tmp_path)
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[looped]))
    assert store.snapshot().skipped == 1
    assert object_files(tmp_path) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
async def test_cuda_tensor_is_skipped(tmp_path):
    store = open_store(tmp_path)
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[torch.zeros(4, device="cuda")]]))
    assert store.snapshot().skipped == 1
    assert object_files(tmp_path) == []


async def test_second_lookup_hits_and_counts_payload_bytes(tmp_path):
    store = open_store(tmp_path)
    assert await store.lookup(make_context(DIGEST_A)) is None

    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[torch.zeros(8)]]))
    loaded = await store.lookup(make_context(DIGEST_A))
    assert loaded is not None
    stats = store.snapshot()
    assert stats.misses == 1
    assert stats.hits == 1
    assert stats.bytes_read == payload_bytes(tmp_path, DIGEST_A)


async def test_invalid_hash_lookup_returns_none_and_writes_nothing(tmp_path):
    store = open_store(tmp_path)
    for bad in ("zz", "a" * 63, "A" * 64, "", None):
        assert await store.lookup(make_context(bad)) is None
        assert store.should_cache(make_context(bad)) is False
    stats = store.snapshot()
    assert stats.misses == 5
    assert stats.hits == 0
    assert object_files(tmp_path) == []
    assert os.listdir(os.path.join(str(tmp_path), "tmp")) == []


async def test_subgraph_scope_is_skipped(tmp_path):
    store = open_store(tmp_path)
    context = make_context(DIGEST_A, scope="subgraph", parent="parent")
    assert store.should_cache(context) is False
    await store.store(context, CacheValue(outputs=[[1]]))
    assert await store.lookup(context) is None
    stats = store.snapshot()
    assert stats.skipped == 3
    assert stats.misses == 0
    assert stats.stores == 0
    assert object_files(tmp_path) == []


async def test_concurrent_lookups_read_object_once(tmp_path):
    store = open_store(tmp_path)
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[torch.zeros(16)]]))

    results = await asyncio.gather(*(store.lookup(make_context(DIGEST_A)) for _ in range(8)))
    assert all(result is not None for result in results)
    assert all(torch.equal(r.outputs[0][0], torch.zeros(16)) for r in results)
    stats = store.snapshot()
    assert stats.hits == 8
    assert stats.dedup_waits == 7
    assert stats.bytes_read == payload_bytes(tmp_path, DIGEST_A)


async def test_two_stores_share_directory_without_losing_entries(tmp_path):
    # Bursts of fsync on the same disk can exceed the default lock timeout, which
    # would abandon a store by design and hide a real merge bug.
    first = durable_cache.DurableCacheStore(str(tmp_path), max_bytes=8 * 1024 ** 3, lock_timeout=30.0)
    second = durable_cache.DurableCacheStore(str(tmp_path), max_bytes=8 * 1024 ** 3, lock_timeout=30.0)
    contexts = [make_context(f"{i:064x}") for i in range(20)]
    start = asyncio.Event()

    async def write(store, context):
        await start.wait()
        await store.store(context, CacheValue(outputs=[[torch.zeros(3)]]))

    writers = []
    for index, context in enumerate(contexts):
        store = first if index % 2 == 0 else second
        writers.append(write(store, context))
    tasks = [asyncio.create_task(writer) for writer in writers]
    start.set()
    await asyncio.gather(*tasks)

    index = read_index(tmp_path)
    assert first.snapshot().failures == 0
    assert second.snapshot().failures == 0
    assert len(index["entries"]) == 20
    for context in contexts:
        meta = index["entries"][context.cache_key_hash]
        with open(os.path.join(str(tmp_path), "objects", context.cache_key_hash[:2],
                               context.cache_key_hash + ".bin"), "rb") as f:
            head = f.read(8)
            meta_len = int.from_bytes(head[4:], "big")
            assert len(f.read(meta_len)) == meta_len
            payload = f.read()
        assert hashlib.sha256(payload).hexdigest() == meta["payload_sha256"]


async def test_capacity_evicts_oldest_entries(tmp_path):
    clock = FakeClock()
    store = open_store(tmp_path, max_bytes=10 ** 9, clock=clock)
    probe = CacheValue(outputs=[[torch.zeros(4)]])
    await store.store(make_context(DIGEST_A), probe)
    entry_size = store.snapshot().bytes_on_disk
    store.max_bytes = entry_size * 3

    for index in range(1, 5):
        clock.advance(10)
        await store.store(make_context(f"{index:064x}"), probe)

    stats = store.snapshot()
    assert stats.entries == 3
    assert stats.evictions == 2
    entries = read_index(tmp_path)["entries"]
    assert set(entries) == {f"{index:064x}" for index in (2, 3, 4)}
    assert sorted(meta["last_used_at"] for meta in entries.values()) == [1020.0, 1030.0, 1040.0]


async def test_entry_larger_than_capacity_is_skipped(tmp_path):
    store = open_store(tmp_path, max_bytes=16)
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[torch.zeros(64)]]))
    stats = store.snapshot()
    assert stats.skipped == 1
    assert stats.stores == 0
    assert object_files(tmp_path) == []


async def test_ttl_expires_entry_on_lookup(tmp_path):
    clock = FakeClock()
    store = open_store(tmp_path, ttl_seconds=60, clock=clock)
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[torch.zeros(4)]]))
    assert object_files(tmp_path)

    clock.advance(61)
    assert await store.lookup(make_context(DIGEST_A)) is None
    stats = store.snapshot()
    assert stats.expired == 1
    assert stats.misses == 1
    assert object_files(tmp_path) == []
    assert read_index(tmp_path)["entries"] == {}


async def test_tampered_object_is_corrupt_and_purged(tmp_path):
    store = open_store(tmp_path)
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[torch.zeros(8)]]))
    path = os.path.join(str(tmp_path), "objects", DIGEST_A[:2], DIGEST_A + ".bin")
    with open(path, "r+b") as f:
        f.seek(-1, os.SEEK_END)
        tail = f.read(1)
        f.seek(-1, os.SEEK_END)
        f.write(bytes([tail[0] ^ 0xFF]))

    assert await store.lookup(make_context(DIGEST_A)) is None
    stats = store.snapshot()
    assert stats.corrupt == 1
    assert stats.hits == 0
    assert stats.entries == 0
    assert not os.path.exists(path)
    assert read_index(tmp_path)["entries"] == {}


async def test_missing_object_is_miss_and_cleans_index(tmp_path):
    store = open_store(tmp_path)
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[torch.zeros(8)]]))
    os.unlink(os.path.join(str(tmp_path), "objects", DIGEST_A[:2], DIGEST_A + ".bin"))
    assert await store.lookup(make_context(DIGEST_A)) is None
    assert store.snapshot().misses == 1
    assert read_index(tmp_path)["entries"] == {}


async def test_index_rebuilt_after_deletion(tmp_path):
    store = open_store(tmp_path)
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[torch.zeros(4)]]))
    durable_cache.close_durable_cache(store)
    os.unlink(os.path.join(str(tmp_path), "index.json"))

    reopened = open_store(tmp_path)
    assert reopened.snapshot().entries == 1
    assert await reopened.lookup(make_context(DIGEST_A)) is not None


async def test_corrupt_index_is_renamed_and_rebuilt(tmp_path):
    store = open_store(tmp_path)
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[torch.zeros(4)]]))
    durable_cache.close_durable_cache(store)
    with open(os.path.join(str(tmp_path), "index.json"), "w") as f:
        f.write("{not json")

    reopened = open_store(tmp_path)
    assert reopened.snapshot().entries == 1
    names = os.listdir(str(tmp_path))
    assert any(name.startswith("index.corrupt-") and name.endswith(".json") for name in names)
    assert await reopened.lookup(make_context(DIGEST_A)) is not None


async def test_unreadable_object_is_not_rebuilt_into_index(tmp_path):
    store = open_store(tmp_path)
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[torch.zeros(4)]]))
    durable_cache.close_durable_cache(store)
    with open(os.path.join(str(tmp_path), "objects", DIGEST_A[:2], DIGEST_A + ".bin"), "wb") as f:
        f.write(b"garbage")
    os.unlink(os.path.join(str(tmp_path), "index.json"))

    reopened = open_store(tmp_path)
    assert reopened.snapshot().entries == 0
    assert reopened.snapshot().bytes_on_disk == 0


async def test_unwritable_directory_disables_store(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("disk is full")

    monkeypatch.setattr(durable_cache.os, "makedirs", explode)
    store = durable_cache.DurableCacheStore(str(tmp_path), max_bytes=1024)
    assert store.disabled is True
    assert store.last_error
    assert store.snapshot().failures == 1
    assert await store.lookup(make_context(DIGEST_A)) is None
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[1]]))
    assert store.snapshot().misses == 1
    await store.flush()
    await store.close()


async def test_path_that_is_a_file_disables_store(tmp_path):
    target = tmp_path / "durable"
    target.write_text("not a directory")
    store = durable_cache.DurableCacheStore(str(target), max_bytes=1024)
    assert store.disabled is True
    assert await store.lookup(make_context(DIGEST_A)) is None


async def test_cancelled_store_leaves_no_entry(tmp_path, monkeypatch):
    store = open_store(tmp_path)
    release = threading.Event()

    def blocking_serialize(value):
        release.wait(5)
        buffer = io.BytesIO()
        torch.save({"outputs": list(value.outputs), "ui": value.ui or {}}, buffer)
        return buffer.getvalue()

    monkeypatch.setattr(store, "_serialize_value", blocking_serialize)
    task = asyncio.create_task(store.store(make_context(DIGEST_A), CacheValue(outputs=[[1]])))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.sleep(0.05)

    assert read_index(tmp_path)["entries"] == {}
    assert object_files(tmp_path) == []

    stale = os.path.join(str(tmp_path), "tmp", "stale.tmp")
    with open(stale, "w") as f:
        f.write("x")
    old = time.time() - 7200
    os.utime(stale, (old, old))
    await store.gc()
    assert os.listdir(os.path.join(str(tmp_path), "tmp")) == []


async def test_index_write_failure_keeps_memory_and_retries_on_flush(tmp_path, monkeypatch):
    store = open_store(tmp_path)
    original = store._write_index

    def failing(entries):
        raise OSError("no space left")

    monkeypatch.setattr(store, "_write_index", failing)
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[torch.zeros(4)]]))
    stats = store.snapshot()
    assert stats.failures >= 1
    assert stats.stores == 0

    monkeypatch.setattr(store, "_write_index", original)
    await store.flush()
    assert read_index(tmp_path)["entries"].keys() == {DIGEST_A}


async def test_gc_removes_orphans_and_stale_temp_files(tmp_path):
    store = open_store(tmp_path)
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[torch.zeros(4)]]))
    orphan_dir = os.path.join(str(tmp_path), "objects", DIGEST_B[:2])
    os.makedirs(orphan_dir, exist_ok=True)
    orphan = os.path.join(orphan_dir, DIGEST_B + ".bin")
    with open(orphan, "wb") as f:
        f.write(b"orphan")
    stale_tmp = os.path.join(str(tmp_path), "tmp", "old.tmp")
    with open(stale_tmp, "w") as f:
        f.write("x")
    stale_lock = os.path.join(str(tmp_path), "locks", "old.lock")
    with open(stale_lock, "w") as f:
        f.write("x")
    old = time.time() - 7200
    for path in (orphan, stale_tmp, stale_lock):
        os.utime(path, (old, old))

    removed = await store.gc()
    assert removed == 1
    assert not os.path.exists(orphan)
    assert not os.path.exists(stale_tmp)
    assert not os.path.exists(stale_lock)
    assert os.path.exists(os.path.join(str(tmp_path), "objects", DIGEST_A[:2], DIGEST_A + ".bin"))


async def test_open_is_idempotent_and_close_releases(tmp_path):
    first = open_store(tmp_path)
    second = open_store(tmp_path)
    assert first is second
    durable_cache.close_durable_cache(first)
    assert first is not None
    assert second.provider in _get_cache_providers()
    durable_cache.close_durable_cache(second)
    assert second.provider not in _get_cache_providers()
    reopened = open_store(tmp_path)
    assert reopened is not second


async def test_stats_exposed_for_active_store(tmp_path):
    store = open_store(tmp_path)
    await store.store(make_context(DIGEST_A), CacheValue(outputs=[[torch.zeros(4)]]))
    assert await store.lookup(make_context(DIGEST_A)) is not None
    stats = durable_cache.get_durable_cache_stats()
    assert stats is not None
    assert stats.hits == 1
    assert stats.stores == 1
    assert stats.entries == 1
    assert stats == store.snapshot()

    durable_cache.reset_durable_cache_stats()
    reset = durable_cache.get_durable_cache_stats()
    assert reset.hits == 0
    assert reset.stores == 0
    assert reset.entries == 1

    durable_cache.close_durable_cache(store)
    if not durable_cache._registry:
        assert durable_cache.get_durable_cache_stats() is None


async def test_stats_snapshot_is_frozen_and_non_negative(tmp_path):
    store = open_store(tmp_path)
    stats = store.snapshot()
    with pytest.raises(Exception):
        stats.hits = 5
    values = vars(stats)
    assert all(isinstance(value, int) and value >= 0 for value in values.values())


async def test_context_scope_marks_subgraph_nodes():
    from comfy_execution.caching import CacheKeySetID, HierarchicalCache

    prompt = {"1": {"class_type": "Root", "inputs": {}}}
    dynprompt = DynamicPrompt(prompt)
    dynprompt.add_ephemeral_node("2", {"class_type": "Child", "inputs": {}}, "1", "1")
    cache = HierarchicalCache(CacheKeySetID)
    await cache.set_prompt(dynprompt, ["1", "2"], _IsChanged())
    await cache._ensure_subcache("1", ["2"])

    root_cache = cache._get_cache_for("1")
    root_context = root_cache._build_context("1", ("1", "Root"))
    assert root_context.scope == "root"
    assert root_context.parent_node_id is None

    sub_cache = cache._get_cache_for("2")
    sub_context = sub_cache._build_context("2", ("2", "Child"))
    assert sub_context.scope == "subgraph"
    assert sub_context.parent_node_id == "1"


async def test_subgraph_entries_are_skipped_end_to_end(tmp_path, monkeypatch):
    import nodes

    class _SubgraphNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"value": ("INT",)}}

        RETURN_TYPES = ("INT",)
        FUNCTION = "execute"

        def execute(self, value):
            return (value,)

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "_DurableSub", _SubgraphNode)
    prompt = {"child": {"class_type": "_DurableSub", "inputs": {"value": 1}}}
    dynprompt = DynamicPrompt(prompt)
    dynprompt.add_ephemeral_node("child", prompt["child"], "parent", "parent")
    caches = CacheSet(cache_type=CacheType.CLASSIC,
                      cache_args={"durable": {"directory": str(tmp_path), "max_bytes": 10 ** 6}})
    try:
        await caches.outputs.set_prompt(dynprompt, ["child"], _IsChanged())
        subcache = await caches.outputs.ensure_subcache_for("parent", ["child"])
        await subcache._set_immediate("child", CacheEntry(ui={"value": 1}, outputs=[1]))
        await asyncio.gather(*subcache._pending_store_tasks)
        assert await subcache._get_immediate("child") == CacheEntry(ui={"value": 1}, outputs=[1])
        stats = caches.durable_store.snapshot()
        assert stats.stores == 0
        assert stats.skipped >= 1
        assert object_files(tmp_path) == []
    finally:
        caches.close()


async def test_consumer_node_is_its_own_file_set(tmp_path):
    pass


class _EchoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value):
        return (value,)


class _IsChanged:
    async def get(self, node_id):
        return False


async def test_local_hit_does_not_touch_durable_cache(tmp_path, monkeypatch):
    import nodes

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "_DurableEcho", _EchoNode)
    prompt = {"1": {"class_type": "_DurableEcho", "inputs": {"value": 3}}}
    dynprompt = DynamicPrompt(prompt)
    caches = CacheSet(cache_type=CacheType.CLASSIC,
                      cache_args={"durable": {"directory": str(tmp_path), "max_bytes": 10 ** 6}})
    try:
        await caches.outputs.set_prompt(dynprompt, prompt.keys(), _IsChanged())
        await caches.outputs.set("1", CacheEntry(ui={"value": 3}, outputs=[3]))
        await asyncio.gather(*caches.outputs._pending_store_tasks)
        store = caches.durable_store

        await caches.outputs.get("1")
        await caches.outputs.get("1")
        stats = store.snapshot()
        assert stats.stores == 1
        assert stats.hits == 0
        assert stats.bytes_read == 0
        assert stats.misses == 0

        caches.outputs.cache.clear()
        cached = await caches.outputs.get("1")
        assert cached.outputs == [3]
        stats = store.snapshot()
        assert stats.hits == 1
        assert stats.bytes_read > 0
    finally:
        caches.close()


async def test_none_cache_type_ignores_durable_config(tmp_path):
    directory = tmp_path / "durable"
    caches = CacheSet(cache_type=CacheType.NONE,
                      cache_args={"durable": {"directory": str(directory), "max_bytes": 10 ** 6}})
    try:
        assert not directory.exists()
        assert caches.durable_store is None
        assert all(provider.__class__.__name__ != "DurableCacheProvider" for provider in _get_cache_providers())
        assert await caches.outputs.get("1") is None
    finally:
        caches.close()


async def test_provider_drain_waits_for_pending_writes(tmp_path):
    store = open_store(tmp_path)
    provider = store.provider
    finished = []

    async def slow_store(context, value):
        await asyncio.sleep(0.05)
        finished.append(context.node_id)

    original = store.store

    async def wrapped(context, value):
        provider._pending_tasks.add(asyncio.current_task())
        try:
            await slow_store(context, value)
        finally:
            provider._pending_tasks.discard(asyncio.current_task())

    store.store = wrapped
    task = asyncio.create_task(provider.on_store(make_context(DIGEST_A), CacheValue(outputs=[[1]])))
    await asyncio.sleep(0)
    await provider.on_drain()
    assert finished == ["1"]
    assert task.done()
    store.store = original


async def test_drain_skips_providers_without_hook():
    class Plain:
        pass

    from execution import PromptExecutor

    executor = PromptExecutor.__new__(PromptExecutor)
    await executor._drain_cache_providers()


async def test_second_process_reuses_entries(tmp_path):
    script = (
        "import asyncio, sys\n"
        "from comfy_api.latest._caching import CacheContext, CacheValue\n"
        "import comfy_execution.durable_cache as dc\n"
        "async def main():\n"
        "    store = dc.open_durable_cache(sys.argv[1], max_bytes=10 ** 7)\n"
        "    for index in range(20):\n"
        "        digest = format(index, '064x')\n"
        "        ctx = CacheContext(node_id=str(index), class_type='TestNode', cache_key_hash=digest)\n"
        "        await store.store(ctx, CacheValue(outputs=[[index]], ui={'id': index}))\n"
        "    dc.close_durable_cache(store)\n"
        "asyncio.run(main())\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, cwd=str(tmp_path), env=env)

    store = open_store(tmp_path)
    store.reset_stats()
    for index in range(20):
        loaded = await store.lookup(make_context(format(index, "064x")))
        assert loaded is not None
        assert loaded.outputs == [[index]]
        assert loaded.ui == {"id": index}
    stats = store.snapshot()
    assert stats.hits == 20
    assert stats.misses == 0
