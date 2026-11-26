import asyncio
import hashlib
import io
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import torch

from comfy_api.latest._caching import CacheContext, CacheProvider, CacheValue
from comfy_execution.cache_provider import register_cache_provider, unregister_cache_provider

_logger = logging.getLogger(__name__)

_MAGIC = b"CDC1"
_SCHEMA_VERSION = 1
_MAX_METADATA_BYTES = 1024 * 1024
_OBJECT_SUFFIX = ".bin"
_INDEX_FILENAME = "index.json"
_OBJECTS_DIRNAME = "objects"
_LOCKS_DIRNAME = "locks"
_TMP_DIRNAME = "tmp"
_INDEX_LOCK_FILENAME = "index.lock"
_HEX_DIGITS = frozenset("0123456789abcdef")
_STALE_LOCK_SECONDS = 30.0
_LOCK_RETRY_INITIAL = 0.02
_LOCK_RETRY_MAX = 0.2
_GC_STALE_SECONDS = 3600.0
_DRAIN_TIMEOUT_SECONDS = 5.0
_CANCELLED = object()

_COUNTERS = (
    "hits", "misses", "stores", "skipped", "bytes_read", "bytes_written",
    "evictions", "expired", "corrupt", "dedup_waits", "failures",
)


@dataclass(frozen=True)
class DurableCacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    skipped: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    evictions: int = 0
    expired: int = 0
    corrupt: int = 0
    dedup_waits: int = 0
    failures: int = 0
    entries: int = 0
    bytes_on_disk: int = 0


class _NotCacheable(Exception):
    pass


class _CorruptObject(Exception):
    pass


class _IndexLockTimeout(Exception):
    pass


class _Aborted(Exception):
    pass


class _LockHandle:
    def __init__(self, path, token):
        self.path = path
        self.token = token

    def release(self):
        try:
            with open(self.path, "r") as f:
                holder = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        if holder != self.token:
            return
        try:
            os.unlink(self.path)
        except OSError:
            pass


def _is_valid_digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX_DIGITS for c in value)


def _is_serializable(value, seen) -> bool:
    if isinstance(value, torch.Tensor):
        return value.device.type == "cpu"
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            return False
        seen.add(marker)
        try:
            for key, item in value.items():
                if not _is_serializable(key, seen) or not _is_serializable(item, seen):
                    return False
        finally:
            seen.discard(marker)
        return True
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in seen:
            return False
        seen.add(marker)
        try:
            return all(_is_serializable(item, seen) for item in value)
        finally:
            seen.discard(marker)
    return value is None or isinstance(value, (str, int, float, bool))


def _fsync_directory(path):
    if os.name != "posix":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_atomic(tmp_path, final_path, data):
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _fsync_parent(final_path):
    parent = os.path.dirname(final_path)
    if parent:
        _fsync_directory(parent)


def _encode_object(meta, payload) -> bytes:
    meta_bytes = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _MAGIC + len(meta_bytes).to_bytes(4, "big") + meta_bytes + payload


def _read_object_header(f):
    head = f.read(8)
    if len(head) != 8 or head[:4] != _MAGIC:
        raise _CorruptObject
    meta_len = int.from_bytes(head[4:], "big")
    if meta_len > _MAX_METADATA_BYTES:
        raise _CorruptObject
    meta_bytes = f.read(meta_len)
    if len(meta_bytes) != meta_len:
        raise _CorruptObject
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _CorruptObject
    if not isinstance(meta, dict):
        raise _CorruptObject
    return meta, meta_len


def _read_object_file(path):
    with open(path, "rb") as f:
        meta, _ = _read_object_header(f)
        payload = f.read()
    return meta, payload


def _read_object_metadata(path):
    try:
        file_size = os.stat(path).st_size
    except OSError:
        raise _CorruptObject
    with open(path, "rb") as f:
        meta, meta_len = _read_object_header(f)
    size = meta.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise _CorruptObject
    if file_size != 8 + meta_len + size:
        raise _CorruptObject
    return meta


def _valid_entry(digest, meta) -> bool:
    if not _is_valid_digest(digest) or not isinstance(meta, dict):
        return False
    if meta.get("digest") != digest:
        return False
    if meta.get("schema_version") != _SCHEMA_VERSION:
        return False
    if not _is_valid_digest(meta.get("payload_sha256")):
        return False
    size = meta.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        return False
    for key in ("created_at", "last_used_at"):
        value = meta.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
    if not isinstance(meta.get("class_type"), str) or not isinstance(meta.get("node_id"), str):
        return False
    return True


def _acquire_index_lock(locks_dir, timeout):
    path = os.path.join(locks_dir, _INDEX_LOCK_FILENAME)
    deadline = time.monotonic() + timeout
    delay = _LOCK_RETRY_INITIAL
    removed_stale = False
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if not removed_stale and _lock_is_stale(path):
                removed_stale = True
                try:
                    os.unlink(path)
                except OSError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise _IndexLockTimeout(f"Timed out waiting for {path}")
            time.sleep(delay)
            delay = min(delay * 2, _LOCK_RETRY_MAX)
            continue
        try:
            token = {"pid": os.getpid(), "acquired_at": time.time()}
            with os.fdopen(fd, "w") as f:
                json.dump(token, f)
            return _LockHandle(path, token)
        except OSError:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise


def _lock_is_stale(path) -> bool:
    try:
        with open(path, "r") as f:
            content = json.load(f)
        acquired_at = float(content.get("acquired_at"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        try:
            acquired_at = os.stat(path).st_mtime
        except OSError:
            return False
    return time.time() - acquired_at > _STALE_LOCK_SECONDS


class DurableCacheStore:
    def __init__(self, directory, *, max_bytes, ttl_seconds=None, clock=time.time, lock_timeout=2.0):
        self.directory = os.path.realpath(directory)
        self.max_bytes = int(max_bytes)
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.lock_timeout = lock_timeout
        self._disabled = False
        self._last_error = None
        self._closed = False
        self._index_path = os.path.join(self.directory, _INDEX_FILENAME)
        self._objects_dir = os.path.join(self.directory, _OBJECTS_DIRNAME)
        self._locks_dir = os.path.join(self.directory, _LOCKS_DIRNAME)
        self._tmp_dir = os.path.join(self.directory, _TMP_DIRNAME)
        self._entries = {}
        self._index_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._counters = dict.fromkeys(_COUNTERS, 0)
        self._digest_locks_lock = threading.Lock()
        self._digest_locks = {}
        self._flight_lock = threading.Lock()
        self._flights = {}
        self._bg_lock = threading.Lock()
        self._bg_tasks = set()
        self._reload_task = None
        self._loop = None
        self.provider = DurableCacheProvider(self)

        try:
            self._prepare_storage()
            self._entries = self._load_index()
        except OSError as e:
            self._disable(e, "Failed to open durable cache")

    @property
    def disabled(self):
        return self._disabled

    @property
    def last_error(self):
        return self._last_error

    def _disable(self, error, message):
        self._disabled = True
        self._last_error = str(error)
        self._bump("failures")
        _logger.warning("%s at %s: %s", message, self.directory, error)

    def _prepare_storage(self):
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(self._objects_dir, exist_ok=True)
        os.makedirs(self._locks_dir, exist_ok=True)
        os.makedirs(self._tmp_dir, exist_ok=True)
        probe = os.path.join(self._tmp_dir, f".probe.{os.getpid()}.{uuid.uuid4().hex}")
        with open(probe, "w") as f:
            f.write("ok")
        os.unlink(probe)

    def _bump(self, name, amount=1):
        if amount <= 0:
            return
        with self._stats_lock:
            self._counters[name] += amount

    def _record_failure(self, error):
        self._last_error = str(error)
        self._bump("failures")

    def _object_path(self, digest):
        return os.path.join(self._objects_dir, digest[:2], digest + _OBJECT_SUFFIX)

    def snapshot(self):
        with self._stats_lock:
            counters = dict(self._counters)
        with self._index_lock:
            entries = len(self._entries)
            bytes_on_disk = sum(int(meta.get("size", 0)) for meta in self._entries.values())
        return DurableCacheStats(entries=entries, bytes_on_disk=bytes_on_disk, **counters)

    def reset_stats(self):
        with self._stats_lock:
            for name in _COUNTERS:
                self._counters[name] = 0

    # Index persistence

    def _read_index_file(self):
        try:
            with open(self._index_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            self._rename_corrupt_index()
            return None
        if not isinstance(data, dict) or data.get("schema_version") != _SCHEMA_VERSION:
            self._rename_corrupt_index()
            return None
        raw_entries = data.get("entries")
        if not isinstance(raw_entries, dict):
            self._rename_corrupt_index()
            return None
        entries = {}
        for digest, meta in raw_entries.items():
            if _valid_entry(digest, meta):
                entries[digest] = meta
        return entries

    def _rename_corrupt_index(self):
        target = os.path.join(self.directory, f"index.corrupt-{int(time.time())}.json")
        try:
            os.replace(self._index_path, target)
        except OSError:
            pass

    def _rebuild_from_objects(self):
        entries = {}
        try:
            subdirs = sorted(os.listdir(self._objects_dir))
        except OSError:
            return entries
        for subdir in subdirs:
            subdir_path = os.path.join(self._objects_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            try:
                names = sorted(os.listdir(subdir_path))
            except OSError:
                continue
            for name in names:
                if not name.endswith(_OBJECT_SUFFIX):
                    continue
                digest = name[:-len(_OBJECT_SUFFIX)]
                if not _is_valid_digest(digest) or digest[:2] != subdir:
                    continue
                try:
                    meta = _read_object_metadata(os.path.join(subdir_path, name))
                except (_CorruptObject, OSError):
                    continue
                if _valid_entry(digest, meta):
                    entries[digest] = meta
        return entries

    def _load_index(self):
        entries = self._read_index_file()
        if entries is None:
            entries = self._prune_missing_objects(self._rebuild_from_objects())
            self._write_index(entries)
        return entries

    def _prune_missing_objects(self, entries):
        kept = {}
        for digest, meta in entries.items():
            try:
                if os.stat(self._object_path(digest)).st_size >= 8:
                    kept[digest] = meta
            except OSError:
                continue
        return kept

    def _write_index(self, entries):
        body = json.dumps({"schema_version": _SCHEMA_VERSION, "entries": entries},
                          sort_keys=True, separators=(",", ":")).encode("utf-8")
        tmp_path = os.path.join(self._tmp_dir, f"index.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        _write_atomic(tmp_path, self._index_path, body)
        _fsync_parent(self._index_path)

    def _merge_entries(self, disk_entries):
        with self._index_lock:
            memory = {digest: dict(meta) for digest, meta in self._entries.items()}
        merged = dict(disk_entries or {})
        merged.update(memory)
        return merged

    def _apply_capacity(self, entries):
        total = sum(int(meta.get("size", 0)) for meta in entries.values())
        if total <= self.max_bytes:
            return []
        order = sorted(entries, key=lambda digest: (entries[digest].get("last_used_at", 0), digest))
        evicted = []
        while total > self.max_bytes and order:
            digest = order.pop(0)
            total -= int(entries[digest].get("size", 0))
            del entries[digest]
            evicted.append(digest)
        return evicted

    def _delete_objects(self, digests):
        for digest in digests:
            try:
                os.unlink(self._object_path(digest))
            except OSError:
                pass

    def _purge_entry(self, digest, delete_object):
        with self._index_lock:
            self._entries.pop(digest, None)
        if delete_object:
            try:
                os.unlink(self._object_path(digest))
            except OSError:
                pass
        try:
            lock = _acquire_index_lock(self._locks_dir, self.lock_timeout)
        except _IndexLockTimeout:
            return
        try:
            disk_entries = self._read_index_file()
            if disk_entries is None:
                disk_entries = self._rebuild_from_objects()
            disk_entries.pop(digest, None)
            self._write_index(disk_entries)
        except OSError as e:
            self._record_failure(e)
        finally:
            lock.release()

    def _reload_sync(self):
        if self.disabled or self._closed:
            return
        try:
            entries = self._read_index_file()
            if entries is None:
                entries = self._rebuild_from_objects()
                self._write_index(entries)
            with self._index_lock:
                self._entries = entries
        except OSError as e:
            self._record_failure(e)

    def _flush_sync(self):
        if self.disabled or self._closed:
            return
        try:
            lock = _acquire_index_lock(self._locks_dir, self.lock_timeout)
        except _IndexLockTimeout as e:
            self._record_failure(e)
            return
        try:
            disk_entries = self._read_index_file()
            if disk_entries is None:
                disk_entries = self._rebuild_from_objects()
            merged = self._merge_entries(disk_entries)
            for digest in list(merged):
                try:
                    os.stat(self._object_path(digest))
                except FileNotFoundError:
                    del merged[digest]
                except OSError:
                    pass
            evicted = self._apply_capacity(merged)
            self._write_index(merged)
            with self._index_lock:
                self._entries = merged
            if evicted:
                self._bump("evictions", len(evicted))
                self._delete_objects(evicted)
        except OSError as e:
            self._record_failure(e)
        finally:
            lock.release()

    # Serialization

    def _serialize_value(self, value):
        ui = value.ui
        if ui is None:
            ui = {}
        if not isinstance(ui, dict):
            raise _NotCacheable("UI payload is not a dict")
        outputs = value.outputs
        if not isinstance(outputs, (list, tuple)):
            raise _NotCacheable("Outputs payload is not a list")
        if not _is_serializable(outputs, set()) or not _is_serializable(ui, set()):
            raise _NotCacheable("Value contains unsupported types")
        buffer = io.BytesIO()
        torch.save({"outputs": list(outputs), "ui": ui}, buffer)
        return buffer.getvalue()

    def _commit_object(self, context, payload, cancel_event):
        digest = context.cache_key_hash
        if cancel_event.is_set():
            raise _Aborted("Store cancelled")
        now = self.clock()
        meta = {
            "digest": digest,
            "class_type": context.class_type,
            "node_id": context.node_id,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "created_at": now,
            "last_used_at": now,
            "schema_version": _SCHEMA_VERSION,
        }
        object_path = self._object_path(digest)
        os.makedirs(os.path.dirname(object_path), exist_ok=True)
        reused_object = os.path.exists(object_path)
        tmp_path = os.path.join(self._tmp_dir, f"{digest}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        _write_atomic(tmp_path, object_path, _encode_object(meta, payload))
        _fsync_parent(object_path)
        self._bump("bytes_written", len(payload))
        if cancel_event.is_set():
            if not reused_object:
                self._delete_objects([digest])
            raise _Aborted("Store cancelled")

        try:
            lock = _acquire_index_lock(self._locks_dir, self.lock_timeout)
        except _IndexLockTimeout as e:
            self._record_failure(e)
            return
        try:
            disk_entries = self._read_index_file()
            if disk_entries is None:
                disk_entries = self._rebuild_from_objects()
            merged = self._merge_entries(disk_entries)
            merged[digest] = meta
            for candidate in list(merged):
                if candidate == digest:
                    continue
                try:
                    os.stat(self._object_path(candidate))
                except FileNotFoundError:
                    del merged[candidate]
                except OSError:
                    pass
            if cancel_event.is_set():
                merged.pop(digest, None)
                raise _Aborted("Store cancelled")
            evicted = self._apply_capacity(merged)
            try:
                self._write_index(merged)
            except OSError as e:
                # Keep the entry in memory so the next flush retries the write.
                with self._index_lock:
                    self._entries = merged
                self._record_failure(e)
                return
            with self._index_lock:
                self._entries = merged
            self._bump("stores")
            if evicted:
                self._bump("evictions", len(evicted))
                self._delete_objects(evicted)
        except OSError as e:
            self._record_failure(e)
        finally:
            lock.release()

    # Lookup

    def _lookup_sync(self, context):
        digest = context.cache_key_hash
        with self._index_lock:
            entry = self._entries.get(digest)
        if entry is None:
            return None
        if self.ttl_seconds is not None and self.clock() - entry.get("created_at", 0) > self.ttl_seconds:
            self._purge_entry(digest, delete_object=True)
            self._bump("expired")
            return None
        try:
            meta, payload = _read_object_file(self._object_path(digest))
        except FileNotFoundError:
            self._purge_entry(digest, delete_object=False)
            return None
        except _CorruptObject:
            self._purge_entry(digest, delete_object=True)
            self._bump("corrupt")
            return None
        except OSError as e:
            self._record_failure(e)
            return None

        self._bump("bytes_read", len(payload))
        expected_size = entry.get("size")
        if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != entry.get("payload_sha256"):
            self._purge_entry(digest, delete_object=True)
            self._bump("corrupt")
            return None
        try:
            loaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        except Exception as e:
            self._record_failure(e)
            self._purge_entry(digest, delete_object=True)
            self._bump("corrupt")
            return None
        if not isinstance(loaded, dict) or not isinstance(loaded.get("outputs"), list):
            self._purge_entry(digest, delete_object=True)
            self._bump("corrupt")
            return None
        ui = loaded.get("ui")
        if not isinstance(ui, dict):
            ui = {}
        self._touch(digest)
        return CacheValue(outputs=list(loaded["outputs"]), ui=ui)

    def _touch(self, digest):
        with self._index_lock:
            entry = self._entries.get(digest)
            if entry is not None:
                entry["last_used_at"] = self.clock()

    # Event loop helpers

    def _ensure_loop(self):
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            with self._digest_locks_lock:
                self._digest_locks.clear()
            with self._flight_lock:
                self._flights.clear()
            self._loop = loop
        return loop

    def _digest_lock(self, digest):
        with self._digest_locks_lock:
            lock = self._digest_locks.get(digest)
            if lock is None:
                lock = asyncio.Lock()
                self._digest_locks[digest] = lock
            return lock

    def _usable(self) -> bool:
        return not self.disabled and not self._closed

    def should_cache(self, context, value=None) -> bool:
        if not self._usable() or context.scope != "root" or not _is_valid_digest(context.cache_key_hash):
            self._bump("skipped")
            return False
        return True

    async def lookup(self, context):
        if context.scope != "root":
            self._bump("skipped")
            return None
        if not _is_valid_digest(context.cache_key_hash) or not self._usable():
            self._bump("misses")
            return None

        self._ensure_loop()
        digest = context.cache_key_hash
        loop = asyncio.get_running_loop()
        waited = False
        while True:
            with self._flight_lock:
                flight = self._flights.get(digest)
                leader = flight is None
                if leader:
                    flight = loop.create_future()
                    self._flights[digest] = flight

            if not leader:
                if not waited:
                    waited = True
                    self._bump("dedup_waits")
                result = await asyncio.shield(flight)
                if result is _CANCELLED:
                    continue
                self._bump("hits" if result is not None else "misses")
                return result

            try:
                reload_task = self._reload_task
                if reload_task is not None and not reload_task.done():
                    try:
                        await asyncio.shield(reload_task)
                    except Exception:
                        pass
                async with self._digest_lock(digest):
                    result = await asyncio.to_thread(self._lookup_sync, context)
            except asyncio.CancelledError:
                with self._flight_lock:
                    if self._flights.get(digest) is flight:
                        del self._flights[digest]
                if not flight.done():
                    flight.set_result(_CANCELLED)
                raise
            except Exception as e:
                self._record_failure(e)
                result = None
            with self._flight_lock:
                if self._flights.get(digest) is flight:
                    del self._flights[digest]
            if not flight.done():
                flight.set_result(result)
            self._bump("hits" if result is not None else "misses")
            return result

    async def store(self, context, value):
        if context.scope != "root" or not _is_valid_digest(context.cache_key_hash) or not self._usable():
            self._bump("skipped")
            return

        self._ensure_loop()
        digest = context.cache_key_hash
        cancel_event = threading.Event()
        try:
            async with self._digest_lock(digest):
                try:
                    payload = await asyncio.to_thread(self._serialize_value, value)
                except _NotCacheable:
                    self._bump("skipped")
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self._record_failure(e)
                    self._bump("skipped")
                    return
                if len(payload) > self.max_bytes:
                    self._bump("skipped")
                    return
                try:
                    await asyncio.to_thread(self._commit_object, context, payload, cancel_event)
                except _Aborted:
                    return
                except asyncio.CancelledError:
                    raise
                except OSError as e:
                    self._record_failure(e)
        except asyncio.CancelledError:
            cancel_event.set()
            raise

    async def flush(self):
        if not self._usable():
            return
        await asyncio.to_thread(self._flush_sync)

    async def close(self):
        if self._closed:
            return
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self):
        if self._closed:
            return
        self._flush_sync()
        self._closed = True
        with self._digest_locks_lock:
            self._digest_locks.clear()
        with self._flight_lock:
            self._flights.clear()

    async def gc(self):
        if not self._usable():
            return 0
        return await asyncio.to_thread(self._gc_sync)

    def _gc_sync(self):
        try:
            lock = _acquire_index_lock(self._locks_dir, self.lock_timeout)
        except _IndexLockTimeout as e:
            self._record_failure(e)
            return 0
        try:
            disk_entries = self._read_index_file()
            if disk_entries is None:
                disk_entries = self._rebuild_from_objects()
            with self._index_lock:
                self._entries = disk_entries
            referenced = set(disk_entries)
            removed_objects = 0
            now = time.time()
            for subdir in os.listdir(self._objects_dir):
                subdir_path = os.path.join(self._objects_dir, subdir)
                if not os.path.isdir(subdir_path):
                    continue
                for name in os.listdir(subdir_path):
                    if not name.endswith(_OBJECT_SUFFIX):
                        continue
                    digest = name[:-len(_OBJECT_SUFFIX)]
                    if digest in referenced:
                        continue
                    try:
                        os.unlink(os.path.join(subdir_path, name))
                        removed_objects += 1
                    except OSError:
                        pass
            self._remove_stale_files(self._tmp_dir, now, skip=None)
            self._remove_stale_files(self._locks_dir, now,
                                     skip=os.path.join(self._locks_dir, _INDEX_LOCK_FILENAME))
            return removed_objects
        except OSError as e:
            self._record_failure(e)
            return 0
        finally:
            lock.release()

    @staticmethod
    def _remove_stale_files(directory, now, skip):
        try:
            names = os.listdir(directory)
        except OSError:
            return
        for name in names:
            path = os.path.join(directory, name)
            if skip is not None and os.path.abspath(path) == os.path.abspath(skip):
                continue
            try:
                if now - os.stat(path).st_mtime <= _GC_STALE_SECONDS:
                    continue
                os.unlink(path)
            except OSError:
                pass

    def _register_background(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        task = loop.create_task(coro)
        with self._bg_lock:
            self._bg_tasks.add(task)
        task.add_done_callback(self._forget_background)
        return task

    def _forget_background(self, task):
        with self._bg_lock:
            self._bg_tasks.discard(task)
        if task.cancelled():
            return
        task.exception()

    def on_prompt_start(self):
        if not self._usable():
            return
        self._reload_task = self._register_background(asyncio.to_thread(self._reload_sync))

    def on_prompt_end(self):
        if not self._usable():
            return
        self._register_background(asyncio.to_thread(self._flush_sync))

    def pending_tasks(self):
        with self._bg_lock:
            return [task for task in self._bg_tasks if not task.done()]


class DurableCacheProvider(CacheProvider):
    def __init__(self, store):
        self.store = store
        self._pending_tasks = set()

    async def on_lookup(self, context: CacheContext) -> Optional[CacheValue]:
        return await self.store.lookup(context)

    async def on_store(self, context: CacheContext, value: CacheValue) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._pending_tasks.add(task)
        try:
            await self.store.store(context, value)
        finally:
            self._pending_tasks.discard(task)

    def should_cache(self, context: CacheContext, value: Optional[CacheValue] = None) -> bool:
        return self.store.should_cache(context, value)

    def on_prompt_start(self, prompt_id: str) -> None:
        self.store.on_prompt_start()

    def on_prompt_end(self, prompt_id: str) -> None:
        self.store.on_prompt_end()

    async def on_drain(self) -> None:
        loop = asyncio.get_running_loop()
        pending = [task for task in self._pending_tasks if not task.done() and task.get_loop() is loop]
        pending.extend(task for task in self.store.pending_tasks() if task.get_loop() is loop)
        if not pending:
            return
        done, still_pending = await asyncio.wait(pending, timeout=_DRAIN_TIMEOUT_SECONDS)
        if still_pending:
            _logger.warning("Durable cache still has %d pending write(s) after %.0fs",
                            len(still_pending), _DRAIN_TIMEOUT_SECONDS)


_registry_lock = threading.Lock()
_registry = {}
_last_store = None


def open_durable_cache(directory, *, max_bytes, ttl_seconds=None, clock=time.time, lock_timeout=2.0):
    global _last_store
    real_directory = os.path.realpath(directory)
    with _registry_lock:
        record = _registry.get(real_directory)
        if record is not None:
            store, refs = record
            _registry[real_directory] = (store, refs + 1)
            _last_store = store
            return store
        store = DurableCacheStore(real_directory, max_bytes=max_bytes, ttl_seconds=ttl_seconds,
                                  clock=clock, lock_timeout=lock_timeout)
        _registry[real_directory] = (store, 1)
        _last_store = store
    register_cache_provider(store.provider)
    return store


def close_durable_cache(store):
    global _last_store
    released = None
    with _registry_lock:
        record = _registry.get(store.directory)
        if record is None or record[0] is not store:
            return
        store_record, refs = record
        if refs > 1:
            _registry[store.directory] = (store_record, refs - 1)
            return
        del _registry[store.directory]
        released = store_record
        if _last_store is released:
            _last_store = None
    unregister_cache_provider(released.provider)
    released._close_sync()


def get_durable_cache_stats():
    store = _last_store
    if store is None or store._closed:
        return None
    return store.snapshot()


def reset_durable_cache_stats():
    with _registry_lock:
        stores = [record[0] for record in _registry.values()]
    for store in stores:
        store.reset_stats()
