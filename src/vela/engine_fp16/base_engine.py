"""Execute v13 storage sideband as bounded memory transactions and compute jobs.

DDR traffic is charged per 64-byte transaction including scale and alignment.
This is an architectural simulation, not a numerical or RTL execution.
"""
import json, math, csv, argparse, sys, collections, importlib.util
import numpy as np
import gzip
import os as _os

MW = int(_os.environ.get("MERGE_W", "24"))
DMA_OWN = _os.environ.get("DMA_DEDICATED", "0") == "1"
READ_W = _os.environ.get("READ_WIDTH", "0") == "128"
FP16 = _os.environ.get("FP16_DIRECT", "0") == "1"
from pathlib import Path

P = Path(__file__).resolve().parent
from .. import common as old
from ..common import C, refs

DATA = P.parent / "data"


class LeanScheduler(old.Scheduler):
    def add(self, *args, **kwargs):
        i = super().add(*args, **kwargs)
        e = self.events[i]
        e["reason"] = sys.intern(e["reason"])
        return i

    def summary(self, terminal):
        end = self.end(terminal)
        busy = collections.Counter()
        wait = collections.Counter()
        traffic = collections.Counter()
        stages = collections.defaultdict(collections.Counter)
        for e in self.events:
            d = e["end"] - e["start"]
            busy[e["category"]] += d
            wait[e["category"]] += e["resource_wait"]
            traffic[e["traffic"]] += e["read_bytes"] + e["write_bytes"]
            stages[e["stage"]][e["category"]] += d
        partition = collections.Counter()
        path = []
        node = terminal
        ties = 0
        while node >= 0:
            e = self.events[node]
            partition[e["category"]] += e["end"] - e["start"]
            path.append([e["start"], e["end"], e["category"]])
            pred = e["critical_predecessors"]
            ties += len(pred) > 1
            node = min(pred) if pred else -1
        assert sum(partition.values()) == end
        return {
            "cycles": end,
            "ms": end / 300000,
            "calls_per_s": 300e6 / end,
            "busy_cycles": dict(busy),
            "resource_wait_cycles": dict(wait),
            "traffic_bytes": dict(traffic),
            "stage_busy": dict(stages),
            "critical_cycles": dict(partition),
            "critical_segments": list(reversed(path)),
            "critical_path_ties": ties,
            "critical_note": "one valid longest dependency path; equal blockers are not unique causes",
            "event_count": len(self.events),
        }


class Engine:
    def __init__(self, plan, bandwidth=64):
        self.plan = plan
        self.cfg = plan["config"]
        self.level = self.cfg["level"]
        self.bandwidth = bandwidth
        self.s = LeanScheduler(old.Config(ddr_bytes=bandwidth))
        self.last = -1
        self.op = -1
        self.ledger = []
        self.cache = {}
        self.backing = set(k for k, v in plan["storage"].items() if v["external"])
        self.objects = {k: dict(v) for k, v in plan["storage"].items()}
        self.arena = self.cfg["arena_bytes"]
        self.pinned = set()
        self.allocations = []
        self.peak = 0
        self.ops = []
        self.extra_work = collections.Counter()
        self.ports = plan["resources"]
        self.temp_index = 0
        self.formats = {}
        self.cache_hits = collections.Counter()
        self.mac = 0
        self.last_access = {}
        self.checks = collections.Counter()
        if FP16:
            plan["group_stream"] = {}
            for _k in plan["storage"]:
                if _k.startswith("weight:"):
                    plan["storage"][_k]["bytes"] = int(plan["storage"][_k]["bytes"] * 8 / 3) + 64
        _f = plan.get("formats") or {}
        self.fmt_wbits = _f.get("weight_bits", 8)
        self.fmt_ga = _f.get("activation_group_k", 64)
        self.fmt_gw = _f.get("weight_group_k", self.fmt_ga)
        self.fmt_gb = self.fmt_ga + 8
        self.embedding_indices = json.loads((DATA / "embedding_indices.json").read_text())

    def stage(self, module):
        return (
            "vision"
            if module.split(".")[0] in ("vision_encoder", "vision_projection")
            else "language_interaction"
            if module.split(".")[0] in ("text_encoder", "vision_language_interaction")
            else "action"
        )

    def emit(self, cat, cycles, base, resource=None, reason="", onchip=0):
        return self.s.add(
            cat, max(1, int(cycles)), [base], resource=resource, reason=reason, onchip=onchip
        )

    def end(self, *ids):
        return max(ids, key=self.s.end, default=-1)

    def dma(self, key, n, base, write=False, kind="activation", reason="", port=64, offset=0):
        if n <= 0:
            return base
        # Two 4 KiB adapter buffers bound in-flight transfers; direction shares DDR.
        slots = [base, base]
        ends = []
        for pos in range(0, int(n), 4096):
            logical = min(4096, n - pos)
            wire = C(((offset + pos) % 64) + logical, 64) * 64
            slot = (pos // 4096) % 2
            start = self.end(base, slots[slot])
            if write:
                a = self.emit(
                    "local_move",
                    C(logical, port),
                    start,
                    resource=["adapter_source", "dma_source_read" if DMA_OWN else "arena_read"],
                    reason="source port -> bounded DMA buffer",
                    onchip=logical,
                )
                b = self.s.add(
                    "ddr_write",
                    C(wire, self.bandwidth),
                    [a],
                    resource="ddr",
                    reason=reason,
                    write=wire,
                    traffic=kind,
                )
            else:
                a = self.s.add(
                    "ddr_read",
                    C(wire, self.bandwidth),
                    [start],
                    resource="ddr",
                    reason=reason,
                    read=wire,
                    traffic=kind,
                )
                _dst = (
                    "weight_cache_write"
                    if kind == "weight"
                    else ("dma_dest_write" if DMA_OWN else "arena_write")
                )
                b = self.emit(
                    "local_move",
                    C(logical, port),
                    a,
                    resource=["adapter_destination", _dst],
                    reason="bounded DMA buffer -> destination port",
                    onchip=logical,
                )
            slots[slot] = b
            ends.append(b)
            self.ledger.append(
                {
                    "op": self.op,
                    "storage": str(key),
                    "category": kind,
                    "direction": "write" if write else "read",
                    "offset": offset + pos,
                    "payload_bytes": logical,
                    "wire_bytes": wire,
                    "padding_bytes": wire - logical,
                    "reason": reason,
                    "from": "onchip" if write else "DDR",
                    "to": "DDR" if write else "onchip",
                    "port_bytes": port,
                    "event": b,
                    "ddr_event": b if write else a,
                }
            )
        return self.end(*ends)

    def local(self, key, n, base, write=False, port=64, reason="", offset=0):
        end = self.emit(
            "local_move",
            C(offset % min(port, 64) + n, port),
            self.end(base, self.last_access.get(key, -1)),
            resource="arena_write" if write else "arena_read",
            reason=reason or ("arena " + str(key)),
            onchip=n,
        )
        self.last_access[key] = end
        self.s.events[end]["storage_access"] = {
            "storage": key,
            "offset": offset,
            "bytes": n,
            "write": write,
        }
        return end

    def gap(self, n):
        n = C(n, 64) * 64
        cursor = 0
        for x in sorted(self.cache.values(), key=lambda v: v["offset"]):
            if x["offset"] - cursor >= n:
                return cursor
            cursor = x["offset"] + x["allocated"]
        return cursor if cursor + n <= self.arena else None

    def release(self, key, base):
        if key in self.cache:
            x = self.cache.pop(key)
            self.allocations.append(
                {
                    "op": self.op,
                    "action": "free",
                    "key": key,
                    "offset": x["offset"],
                    "bytes": x["allocated"],
                    "after": base,
                }
            )

    def allocate(self, key, base):
        if self.level < 2 and not key.startswith("lut:"):
            return False, base
        if key in self.cache:
            return True, base
        n = self.objects[key]["bytes"]
        space = C(n, 64) * 64
        if space > self.arena or not self.arena:
            return False, base
        at = self.gap(n)
        while at is None:
            options = [k for k in self.cache if k not in self.pinned]
            if not options:
                return False, base

            def rank(k):
                ob = self.objects[k]
                future = [x for x in ob.get("next_uses", []) if x > self.op]
                return (
                    ob.get("last_use", -1) <= self.op,
                    min(future) if future else 10**9,
                    self.cache[k]["allocated"],
                )

            victim = max(options, key=rank)
            v = self.cache[victim]
            obj = self.objects[victim]
            base = self.end(base, self.last_access.get(victim, -1))
            if v["dirty"] and obj.get("last_use", 10**9) > self.op:
                base = self.dma(
                    victim, obj["bytes"], base, True, "spill", "live value evicted for finite arena"
                )
                self.backing.add(victim)
            self.release(victim, base)
            at = self.gap(n)
        self.cache[key] = {"offset": at, "allocated": space, "dirty": False}
        self.peak = max(self.peak, sum(x["allocated"] for x in self.cache.values()))
        self.allocations.append(
            {
                "op": self.op,
                "action": "alloc",
                "key": key,
                "offset": at,
                "bytes": space,
                "after": base,
            }
        )
        return True, base

    def temp(self, size, base, label):
        key = f"t{self.op}_{self.temp_index}_{label}"
        self.temp_index += 1
        self.objects[key] = {
            "bytes": int(size),
            "last_use": 10**9,
            "next_uses": [],
            "external": False,
        }
        ok, base = self.allocate(key, base)
        self.pinned.add(key)
        return key, base

    def store(self, key, n, base, kind="activation", offset=0):
        assert offset >= 0 and n >= 0 and offset + n <= self.objects[key]["bytes"], (
            "store bounds",
            self.op,
            key,
            offset,
            n,
            self.objects[key]["bytes"],
        )
        self.checks["write_ranges"] += 1
        if key in self.cache:
            self.cache[key]["dirty"] = True
            return self.local(key, n, base, True, reason="store " + key, offset=offset)
        self.backing.add(key)
        return self.dma(
            key, n, base, True, kind, "output or temporary does not fit live arena", offset=offset
        )

    def read(self, key, n, base, kind="activation", offset=0, port=64):
        if READ_W and kind in ("activation", "parameter") and port < self.port:
            port = self.port
        assert offset >= 0 and n >= 0 and offset + n <= self.objects[key]["bytes"], (
            "read bounds",
            self.op,
            key,
            offset,
            n,
            self.objects[key]["bytes"],
        )
        self.checks["read_ranges"] += 1
        if key in self.cache:
            self.cache_hits[kind] += n
            return self.local(key, n, base, port=port, reason="resident " + key, offset=offset)
        assert key in self.backing, ("read before write", self.op, key)
        ob = self.objects[key]
        if self.level >= 2 and n >= ob["bytes"] // 4:
            ok, base = self.allocate(key, base)
            if ok:
                base = self.dma(
                    key,
                    ob["bytes"],
                    base,
                    False,
                    kind,
                    "first use / reload after eviction",
                    port=64,
                )
                return self.local(key, n, base, port=port, reason="read after load " + key)
        return self.dma(
            key,
            n,
            base,
            False,
            kind,
            "stream: no fitting arena allocation",
            port=port,
            offset=offset,
        )

    def handle(self, tid):
        t = self.plan["tensor_storage"][str(tid)]
        key = t["storage"]
        return {
            "key": key,
            "bytes": math.prod(t["shape"]) * t["width"],
            "elements": math.prod(t["shape"]),
            "width": t["width"],
            "shape": t["shape"],
            "stride": t["stride"],
            "offset": t["offset"] * t["width"],
            "compact": key in self.formats,
        }

    def data_read(self, h, n, base, offset=0):
        if not n:
            return base
        if h["key"] not in self.metadata_seen and not h["key"].startswith("lut:"):
            self.metadata_seen.add(h["key"])
            meta = 16 if h.get("compact") else 8
            at = 0 if h.get("compact") else self.objects[h["key"]]["bytes"] - 8
            base = self.read(h["key"], meta, base, "tensor_scale", offset=at)
        if "gather_rows" in h:
            rowbytes = h["rowbytes"]
            end = base
            for rr in range(offset // rowbytes, C(offset + n, rowbytes)):
                idx = h["gather_rows"][rr]
                lo = max(offset, rr * rowbytes) - rr * rowbytes
                hi = min(offset + n, (rr + 1) * rowbytes) - rr * rowbytes
                end = self.read(h["key"], hi - lo, end, "parameter", offset=idx * rowbytes + lo)
            return end
        key = h["key"]
        width = h.get("width", 4)
        shape = h.get("shape")
        stride = h.get("stride")
        off = h.get("offset", 0)
        contiguous = True
        expected = 1
        if stride:
            for size, st in zip(reversed(shape), reversed(stride)):
                if size > 1 and st != expected:
                    contiguous = False
                expected *= size
        if not contiguous:
            # Gather in original storage coordinates, including broadcast and views.
            flat = np.arange(offset // width, C(offset + n, width), dtype=np.int64)
            indices = np.full(flat.shape, off // width, dtype=np.int64)
            for size, st in zip(reversed(shape), reversed(stride)):
                indices += (flat % size) * st
                flat //= size
            indices = np.unique(indices)
        else:
            indices = None
        if h.get("compact"):
            fmt = self.formats[key]
            bb, mm, nnn = fmt["shape"]
            if fmt["action"]:
                ranges = [(0, self.objects[key]["bytes"])]
            elif indices is not None:
                groups = np.unique((indices // nnn) * C(nnn, 32) + (indices % nnn) // 32)
                ranges = self.runs(groups, 40, 16)
            else:
                first = (off + offset) // 4
                last = C(off + offset + n, 4)
                grp = lambda i: (i // nnn) * C(nnn, 32) + (i % nnn) // 32
                ranges = [(16 + grp(first) * 40, (grp(max(first, last - 1)) - grp(first) + 1) * 40)]
            end = base
            for p, nn in ranges:
                end = self.read(key, nn, end, "compact_output", offset=p)
            return self.emit(
                "restore",
                C(C(n, 4), 32) + 64,
                end,
                resource="vector",
                reason="exact deferred restore + saved common shift",
            )
        kind = "parameter" if self.objects[key].get("constant") else "activation"
        if indices is not None:
            assert len(indices) == 0 or (
                indices.min() >= 0 and (indices.max() + 1) * width <= self.objects[key]["bytes"]
            ), ("gather bounds", self.op, key)
            if key not in self.cache:
                ok, base = self.allocate(key, base)
                if ok:
                    base = self.dma(
                        key,
                        self.objects[key]["bytes"],
                        base,
                        False,
                        kind,
                        "load strided source before gather",
                    )
            if key in self.cache:
                self.cache_hits[kind] += len(indices) * width
                words = (indices * width + self.cache[key]["offset"]) // 8
                if self.level >= 3:
                    words = np.unique(words)  # one read returns both INT32 halves
                    banks = self.bank_ids(words)
                    cycles = max(1, int(np.bincount(banks, minlength=8).max()))
                    reason = "XOR bank mapping; <=one requested value per bank per cycle"
                else:
                    cycles = max(1, len(indices))
                    reason = "serial gather baseline"
                end = self.emit(
                    "local_move",
                    cycles,
                    base,
                    resource="arena_read",
                    reason=reason,
                    onchip=len(indices) * width,
                )
                self.last_access[key] = end
                return end
            lines = np.unique(indices * width // 64)
            end = base
            for p, nn in self.runs(lines, 64, 0):
                end = self.read(key, min(nn, self.objects[key]["bytes"] - p), end, kind, offset=p)
            return end
        return self.read(key, n, base, kind, offset=off + offset, port=64)

    @staticmethod
    def runs(indices, unit, offset):
        if not len(indices):
            return []
        cuts = np.flatnonzero(np.diff(indices) > 1)
        starts = np.r_[0, cuts + 1]
        ends = np.r_[cuts, len(indices) - 1]
        return [
            (offset + int(indices[a]) * unit, (int(indices[b]) - int(indices[a]) + 1) * unit)
            for a, b in zip(starts, ends)
        ]

    def grouped_read(self, key, rows, count, base, offset, stride, port):
        assert offset + (rows - 1) * stride + count <= self.objects[key]["bytes"], (
            key,
            offset,
            rows,
            count,
            stride,
            self.objects[key]["bytes"],
        )
        if key in self.cache:
            self.cache_hits["quantized_operand"] += rows * count
            end = self.emit(
                "local_move",
                rows * C(count, port),
                self.end(base, self.last_access.get(key, -1)),
                resource="arena_read",
                reason="banked row/group reads; partial beats charged",
                onchip=rows * count,
            )
            self.last_access[key] = end
            return end
        assert key in self.backing
        # Describe repeated small DMA transactions as one bounded batch event.
        # Every row keeps its own 64-byte padding; only the Python event count shrinks.
        limit = max(1, 4096 // count)
        for row in range(0, rows, limit):
            nr = min(limit, rows - row)
            off = offset + row * stride
            wire = sum(C((off + j * stride) % 64 + count, 64) * 64 for j in range(nr))
            logical = nr * count
            a = self.s.add(
                "ddr_read",
                sum(
                    C(C((off + j * stride) % 64 + count, 64) * 64, self.bandwidth)
                    for j in range(nr)
                ),
                [base],
                resource="ddr",
                reason="strided group DMA: separate aligned rows",
                read=wire,
                traffic="quantized_operand",
            )
            base = self.emit(
                "local_move",
                nr * C(count, port),
                a,
                resource=["adapter_destination", "arena_write"],
                reason="bounded <=4 KiB group transfer -> input port",
                onchip=logical,
            )
            self.ledger.append(
                {
                    "op": self.op,
                    "storage": key,
                    "category": "quantized_operand",
                    "direction": "read",
                    "offset": off,
                    "payload_bytes": logical,
                    "wire_bytes": wire,
                    "padding_bytes": wire - logical,
                    "reason": "strided rows; each row separately aligned",
                    "from": "DDR",
                    "to": "onchip",
                    "port_bytes": port,
                    "event": base,
                    "ddr_event": a,
                    "repeat": nr,
                    "stride": stride,
                    "row_bytes": count,
                }
            )
        return base

    def cleanup(self, base):
        for k in list(self.cache):
            if self.objects[k].get("last_use", 10**9) <= self.op and k not in self.pinned:
                self.release(k, base)

    def finish_temp(self, h, base):
        key = h["key"] if isinstance(h, dict) else h
        self.pinned.discard(key)
        self.release(key, base)
        self.backing.discard(key)

    def range_dma(self, key, ranges, base, kind, write=False):
        # Physical address spans, streamed through <=4 KiB adapters in sequence.
        batch = []
        size = 0

        def flush(batch, base):
            if not batch:
                return base
            payload = sum(n for o, n in batch)
            wire = sum(C(o % 64 + n, 64) * 64 for o, n in batch)
            if write:
                aa = self.emit(
                    "local_move",
                    C(payload, 64),
                    base,
                    resource=["adapter_source", "arena_read"],
                    reason="bounded scatter source",
                    onchip=payload,
                )
                end = self.s.add(
                    "ddr_write",
                    sum(C(C(o % 64 + n, 64) * 64, self.bandwidth) for o, n in batch),
                    [aa],
                    resource="ddr",
                    write=wire,
                    traffic=kind,
                    reason="bounded scatter transactions",
                )
                dd = end
            else:
                dd = self.s.add(
                    "ddr_read",
                    sum(C(C(o % 64 + n, 64) * 64, self.bandwidth) for o, n in batch),
                    [base],
                    resource="ddr",
                    read=wire,
                    traffic=kind,
                    reason="bounded gather transactions",
                )
                end = self.emit(
                    "local_move",
                    C(payload, 64),
                    dd,
                    resource=["adapter_destination", "arena_write"],
                    reason="bounded gather -> tile",
                    onchip=payload,
                )
            self.ledger.append(
                {
                    "op": self.op,
                    "storage": key,
                    "category": kind,
                    "direction": "write" if write else "read",
                    "offset": batch[0][0],
                    "payload_bytes": payload,
                    "wire_bytes": wire,
                    "padding_bytes": wire - payload,
                    "reason": "explicit tile spans; each transaction aligned",
                    "from": "onchip" if write else "DDR",
                    "to": "DDR" if write else "onchip",
                    "port_bytes": 64,
                    "event": end,
                    "ddr_event": dd,
                    "spans": batch,
                }
            )
            return end

        for off, n in ranges:
            assert off >= 0 and off + n <= self.objects[key]["bytes"], (
                "tile bounds",
                key,
                off,
                n,
                self.objects[key]["bytes"],
            )
            for pos in range(0, n, 4096):
                nn = min(4096, n - pos)
                if size + nn > 4096:
                    base = flush(batch, base)
                    batch = []
                    size = 0
                batch.append((off + pos, nn))
                size += nn
        return flush(batch, base)

    def quantize(self, h, rows, k, base, side):
        if FP16:
            return h, base
        groups = rows * C(k, self.fmt_ga)
        size = groups * self.fmt_gb
        key, base = self.temp(size, base, "quant_" + side)
        work = base
        shape = h.get("shape")
        stride = h.get("stride")
        contiguous = True
        expected = 1
        if stride:
            for dim, st in zip(reversed(shape), reversed(stride)):
                if dim > 1 and st != expected:
                    contiguous = False
                expected *= dim
        if not contiguous:
            for row in range(0, rows, 64):
                nr = min(64, rows - row)
                for gi in range(C(k, self.fmt_ga)):
                    kk = min(self.fmt_ga, k - gi * self.fmt_ga)
                    flat = (
                        (np.arange(row, row + nr)[:, None] * k) + np.arange(gi * 64, gi * 64 + kk)
                    ).reshape(-1)
                    ix = np.full(flat.shape, h.get("offset", 0) // h["width"], dtype=np.int64)
                    for dim, st in zip(reversed(shape), reversed(stride)):
                        ix += (flat % dim) * st
                        flat //= dim
                    if h.get("compact"):
                        fm = self.formats[h["key"]]
                        nn = fm["shape"][-1]
                        idx = np.unique((ix // nn) * C(nn, 32) + (ix % nn) // 32)
                        ranges = self.runs(idx, 40, 16)
                    else:
                        ranges = self.runs(np.unique(ix), h["width"], 0)
                    if h["key"] in self.cache:
                        work = self.gather_ranges(h["key"], ranges, work)
                    else:
                        work = self.range_dma(h["key"], ranges, work, "activation")
                    if h.get("compact"):
                        work = self.emit(
                            "restore",
                            C(nr * kk, 32) + 64,
                            work,
                            resource="vector",
                            reason="tile decode with saved common shift",
                        )
                    work = self.emit(
                        "layout",
                        C(nr * kk, 4),
                        work,
                        resource="transpose_tile",
                        reason="64x64 bounded transpose tile",
                    )
                    work = self.emit(
                        "quantize",
                        2 * C(nr * kk, 32) + nr + 2,
                        work,
                        resource="quant",
                        reason="tile has complete K64 groups; RNE quantization",
                    )
                    if key in self.cache:
                        self.cache[key]["dirty"] = True
                        work = self.emit(
                            "local_move",
                            nr * 2,
                            work,
                            resource="arena_write",
                            reason="strided quantized group stores",
                            onchip=nr * self.fmt_gb,
                        )
                        self.last_access[key] = work
                        self.s.events[work]["storage_access"] = {
                            "storage": key,
                            "offset": (row * C(k, self.fmt_ga) + gi) * self.fmt_gb,
                            "repeat": nr,
                            "stride": C(k, self.fmt_ga) * self.fmt_gb,
                            "bytes": self.fmt_gb,
                            "write": True,
                        }
                    else:
                        self.backing.add(key)
                        work = self.range_dma(
                            key,
                            [
                                ((rr * C(k, self.fmt_ga) + gi) * self.fmt_gb, self.fmt_gb)
                                for rr in range(row, row + nr)
                            ],
                            work,
                            "quantized_operand",
                            True,
                        )
        else:
            # Two rows share the existing 16 KiB input scratch. Larger rows keep one slot.
            nslots = 2 if k * h.get("width", 4) <= 8192 else 1
            assert k * h.get("width", 4) <= 16384
            slots = [base] * nslots
            ends = []
            for row in range(rows):
                slot = row % nslots
                start = self.end(base, slots[slot])
                birth = self.s.end(start)
                rd = self.data_read(h, k * h.get("width", 4), start, row * k * h.get("width", 4))
                work = self.emit(
                    "quantize",
                    2 * C(k, 32) + C(k, self.fmt_ga) + 2,
                    rd,
                    resource="quant",
                    reason="row uses <=16 KiB scratch; complete group scales",
                )
                work = self.store(
                    key,
                    C(k, self.fmt_ga) * self.fmt_gb,
                    work,
                    "quantized_operand",
                    row * C(k, self.fmt_ga) * self.fmt_gb,
                )
                self.quant_leases.append(
                    {
                        "op": self.op,
                        "slot": slot,
                        "start": birth,
                        "end": self.s.end(work),
                        "bytes": k * h.get("width", 4),
                        "slots": nslots,
                    }
                )
                slots[slot] = work
                ends.append(work)
            work = self.end(*ends)

        return {"key": key, "bytes": size, "elements": rows * k, "compact": False, "width": 1}, work

    def matrix_fp16(self, r, ha, hb, base):
        # Same-hardware FP16-direct GEMM: one full-K pass per tile, no quantize /
        # coefficient / merge / encode stages, FP16 weight panels, plain FP16 output.
        m, n, k, b = [r[x] for x in ("m", "n", "k", "batch")]
        self.mac += m * n * k * b
        cols = min(r["cols"], 48)
        outbytes = b * m * n * 2
        packed, base = self.temp(outbytes, base, "fp16_result")
        ends = []
        prefetched = {}
        wpc = k * 2
        bh = dict(hb) if hb is not None else None
        if (
            bh
            and bh.get("stride")
            and not (self.current_name == "scaled_dot_product_attention" and r["index"] == 0)
        ):
            bh["shape"] = bh["shape"][:-2] + list(reversed(bh["shape"][-2:]))
            bh["stride"] = bh["stride"][:-2] + list(reversed(bh["stride"][-2:]))
        for col in range(0, n, cols):
            nc = min(cols, n - col)
            panelbytes = nc * wpc
            load = base
            if self.cfg["weight_mode"] == "double" and r["constant_b"]:
                if col not in prefetched:
                    prefetched[col] = self.dma(
                        "weight:" + r["weight_storage"],
                        panelbytes,
                        base,
                        False,
                        "weight",
                        "double bank initial FP16 panel",
                        96,
                        col * wpc,
                    )
                nxt = col + cols
                if nxt < n:
                    prefetched[nxt] = self.dma(
                        "weight:" + r["weight_storage"],
                        min(cols, n - nxt) * wpc,
                        base,
                        False,
                        "weight",
                        "prefetch next FP16 panel after its bank is free",
                        96,
                        nxt * wpc,
                    )
                load = self.end(base, prefetched[col])
            elif r["constant_b"]:
                load = self.dma(
                    "weight:" + r["weight_storage"],
                    panelbytes,
                    base,
                    False,
                    "weight",
                    "full-K FP16 panel retained across all input-row tiles",
                    96,
                    col * wpc,
                )
            colends = []
            for bi in range(b):
                for row in range(0, m, 16):
                    mr = min(16, m - row)
                    a = self.data_read(ha, mr * k * 2, base, ((bi * m + row) * k) * 2)
                    if r["constant_b"]:
                        wfeed = self.emit(
                            "weight_read",
                            C(panelbytes, 96),
                            load,
                            resource="wram_read",
                            reason="resident FP16 panel -> array; 2 bytes per weight",
                            onchip=panelbytes,
                        )
                    else:
                        wfeed = self.data_read(bh, nc * k * 2, load, ((bi * n + col) * k) * 2)
                    core = self.emit(
                        "gemm",
                        k + 16,
                        self.end(a, wfeed),
                        resource="array",
                        reason="FP16 full-K single pass; 1 MAC/DSP/cycle; valid columns %d/48" % nc,
                    )
                    wr = self.store(
                        packed, mr * nc * 2, core, "activation", ((bi * m + row) * n + col) * 2
                    )
                    ends.append(wr)
                    colends.append(wr)
            base = self.end(*colends)
        base = self.end(*ends)
        for h in (ha, hb):
            if h and h["key"] not in self.later_inputs:
                self.pinned.discard(h["key"])
        return {
            "key": packed,
            "bytes": b * m * n * 2,
            "elements": b * m * n,
            "width": 2,
            "shape": [b, m, n],
        }, base

    def matrix(self, r, ha, hb, base):
        if FP16:
            return self.matrix_fp16(r, ha, hb, base)
        m, n, k, b = [r[x] for x in ("m", "n", "k", "batch")]
        self.mac += m * n * k * b
        cols = r["cols"]
        groups = C(k, self.fmt_ga)
        fused = self.plan.get("group_stream", {}).get(str(self.op))
        if fused:
            base = self.start_stream(fused, b, m, n, base)
        qa, base = self.quantize(ha, b * m, k, base, "a")
        if r["constant_b"]:
            qb = None
        else:
            bh = dict(hb)
            if bh.get("stride") and not (
                self.current_name == "scaled_dot_product_attention" and r["index"] == 0
            ):
                bh["shape"] = bh["shape"][:-2] + list(reversed(bh["shape"][-2:]))
                bh["stride"] = bh["stride"][:-2] + list(reversed(bh["stride"][-2:]))
            qb, base = self.quantize(bh, b * n, k, base, "b")
        # Input handles are already read; their graph objects can be evicted but are
        # never silently discarded if a later consumer still needs the value.
        for h in (ha, hb):
            if h and h["key"] not in self.later_inputs:
                self.pinned.discard(h["key"])
        if qa.get("borrowed"):
            base = self.prepare_g64_scales(qa, r, base)
        else:
            base = self.emit(
                "scale",
                b * groups + 2,
                base,
                resource="scale",
                reason="whole-operation common exponent barrier",
            )
        out_groups = b * (n * C(m, 32) if r["action"] else m * C(n, 32))
        outbytes = out_groups * 40 + 16
        if fused:
            outbytes = b * m * C(n, self.fmt_ga) * self.fmt_gb
        packed, base = self.temp(outbytes, base, "packed_result")
        lastenc = base
        buffers = [base, base]
        snapshots = [base, base]
        feed_slots = [base, base]
        group_index = 0
        tile = 0
        ends = []
        prefetched = {}
        wdata = k if self.fmt_wbits == 8 else C(k, 2)
        wrec = C(k, self.fmt_gw) * 8
        wpc = wdata + wrec
        for col in range(0, n, cols):
            panel_ready = base
            if self.cfg["weight_mode"] == "double" and r["constant_b"]:
                if col not in prefetched:
                    sz = min(cols, n - col) * wpc
                    prefetched[col] = self.dma(
                        "weight:" + r["weight_storage"],
                        sz,
                        base,
                        False,
                        "weight",
                        "double bank initial panel",
                        96,
                        col * wpc,
                    )
                nxt = col + cols
                if nxt < n:
                    sz = min(cols, n - nxt) * wpc
                    prefetched[nxt] = self.dma(
                        "weight:" + r["weight_storage"],
                        sz,
                        base,
                        False,
                        "weight",
                        "prefetch next panel after its bank is free",
                        96,
                        nxt * wpc,
                    )
            for bi in range(b):
                nc = min(cols, n - col)
                panelbytes = nc * wpc
                if r["constant_b"] and bi == 0:
                    # Stream an entire full-K column panel once, reuse across every M tile.
                    load = (
                        self.end(base, prefetched[col])
                        if col in prefetched
                        else self.dma(
                            "weight:" + r["weight_storage"],
                            panelbytes,
                            base,
                            False,
                            "weight",
                            "full-K panel retained across all input-row tiles",
                            port=96,
                            offset=col * wpc,
                        )
                    )
                    panel_ready = load
                else:
                    load = self.end(base, panel_ready)
                panel_end = load
                for row in range(0, m, 16):
                    mr = min(16, m - row)
                    e = mr * nc
                    bank = tile % 2
                    prev = base
                    for gi in range(groups):
                        kk = min(self.fmt_ga, k - gi * self.fmt_ga)
                        slot = group_index % 2
                        group_index += 1
                        feedbase = self.end(load, feed_slots[slot])
                        a = self.grouped_read(
                            qa["key"],
                            mr,
                            self.fmt_gb,
                            feedbase,
                            ((bi * m + row) * groups + gi) * self.fmt_gb,
                            groups * self.fmt_gb,
                            64,
                        )
                        if qb is not None:
                            a = self.grouped_read(
                                qb["key"],
                                nc,
                                self.fmt_gb,
                                a,
                                ((bi * n + col) * groups + gi) * self.fmt_gb,
                                groups * self.fmt_gb,
                                64,
                            )
                        else:
                            a = self.end(
                                a,
                                self.emit(
                                    "weight_read",
                                    C(nc * (kk if self.fmt_wbits == 8 else C(kk, 2)), 96),
                                    feedbase,
                                    resource="wram_read",
                                    reason="resident panel -> array; nibble panel half bytes",
                                    onchip=nc * C(kk, 2),
                                ),
                            )
                        coef = self.emit(
                            "scale",
                            C(e, MW) + 2,
                            a,
                            resource="coefficient",
                            reason="Q24 coefficient per output/group",
                        )
                        core = self.emit(
                            "gemm",
                            kk + 16,
                            self.end(a, snapshots[slot]),
                            resource="array",
                            reason=f"valid columns {nc}/96; tail capacity not counted as useful MAC",
                        )
                        feed_slots[slot] = core
                        merge = self.emit(
                            "merge",
                            C(e, MW) + 2,
                            self.end(core, coef, buffers[bank], prev),
                            resource="merge",
                            reason="INT64 banked sum; final group waits for encoder",
                        )
                        prev = merge
                        snapshots[slot] = merge
                    og = n * C(mr, 32) if r["action"] else mr * C(col % 32 + nc, 32)
                    enc = self.emit(
                        "encode",
                        2 * C(e, 8) + og + 14,
                        self.end(prev, lastenc),
                        resource="encoder",
                        reason="P1 peaks / P2 RNE output; finite two result banks",
                    )
                    lastenc = enc
                    wr = enc
                    if fused:
                        wr = self.stream_tile(packed, bi, row, mr, col, nc, m, n, enc)
                    else:
                        if r["action"]:
                            wr = self.store(
                                packed,
                                og * 40,
                                wr,
                                "compact_output",
                                offset=16 + bi * n * C(m, 32) * 40,
                            )
                        elif packed in self.cache:
                            self.cache[packed]["dirty"] = True
                            wr = self.emit(
                                "local_move",
                                sum(
                                    C(
                                        (16 + ((bi * m + row + ri) * C(n, 32) + col // 32) * 40)
                                        % 64
                                        + C(col % 32 + nc, 32) * 40,
                                        64,
                                    )
                                    for ri in range(mr)
                                ),
                                wr,
                                resource="arena_write",
                                reason="packed output row stores; straddled start group fully rewritten",
                                onchip=og * 40,
                            )
                            self.last_access[packed] = wr
                        else:
                            for ri in range(mr):
                                off = 16 + ((bi * m + row + ri) * C(n, 32) + col // 32) * 40
                                wr = self.store(
                                    packed,
                                    C(col % 32 + nc, 32) * 40,
                                    wr,
                                    "compact_output",
                                    offset=off,
                                )
                    buffers[bank] = wr
                    ends.append(wr)
                    panel_end = self.end(panel_end, wr)
                    tile += 1
                # Single panel ownership: no replacement until all consumers have finished.
                # Double mode prefetches into the other bank; bank reuse waits for this panel.
                base = panel_end
        if fused:
            base = self.end(*ends)
            self.finish_stream(base)
            self.finish_temp(qa, base)
            if qb:
                self.finish_temp(qb, base)
            self.formats[packed] = {
                "kind": "g64_after_gelu",
                "shape": [b, m, n],
                "action": False,
                "groups": b * m * C(n, 64),
                "logical_bytes": b * m * n * 4,
            }
            return {
                "key": packed,
                "bytes": b * m * n * 4,
                "elements": b * m * n,
                "width": 4,
                "shape": [b, m, n],
                "compact": True,
            }, base
        base = self.end(*ends)
        base = self.emit(
            "scale",
            out_groups + 2,
            base,
            resource="scale",
            reason="output common exponent; wait for every output group",
        )
        base = self.read(packed, outbytes - 16, base, "compact_output", offset=16)
        base = self.emit(
            "restore",
            C(b * m * n, 32) + 64,
            base,
            resource="vector",
            reason="scan restored values for exact norm shift; retain packed representation",
        )
        base = self.store(
            packed, 16, base, "compact_output", 0
        )  # exponent and norm shift are now known
        self.finish_temp(qa, base)
        if qb:
            self.finish_temp(qb, base)
        self.formats[packed] = {
            "kind": "compact_c2",
            "shape": [b, m, n],
            "action": r["action"],
            "groups": out_groups,
        }
        return {
            "key": packed,
            "bytes": b * m * n * 4,
            "elements": b * m * n,
            "width": 4,
            "shape": [b, m, n],
            "compact": True,
        }, base

    def vector(self, name, inputs, n, outkey, outbytes, base, axis=1, final=False):
        norm = name in (
            "add",
            "add_",
            "mul",
            "sub",
            "rsub",
            "div",
            "gelu",
            "tanh",
            "sin",
            "cos",
            "layer_norm",
        )
        passes = (4 if name == "layer_norm" else 2 if norm else 1) if self.level >= 3 else 1
        # Retain the exact global scale barriers by recomputation instead of DDR
        # wide-temporary materialization. Passes before the last produce only peaks.
        if outkey is None:
            outkey, base = self.temp(outbytes, base, name)
        else:
            if outkey in self.formats:
                hh = {
                    "key": outkey,
                    "bytes": self.formats[outkey]["logical_bytes"],
                    "width": 4,
                    "compact": True,
                }
                base = self.data_read(hh, hh["bytes"], base)
                self.release(outkey, base)
                self.formats.pop(outkey)
                self.objects[outkey]["bytes"] = hh["bytes"] + 8
            self.pinned.discard(outkey)
            _, base = self.allocate(outkey, base)
            self.pinned.add(outkey)
        lutkey = None
        if name in ("gelu", "tanh", "sin", "cos", "softmax"):
            lutkey = "lut:" + ("exp" if name == "softmax" else name)
            if lutkey not in self.objects:
                self.objects[lutkey] = {
                    "bytes": 65537 * 4,
                    "constant": True,
                    "external": True,
                    "last_use": 10**9,
                    "next_uses": [],
                }
                self.backing.add(lutkey)
            self.pinned.add(lutkey)
            ok, base = self.allocate(lutkey, base)
            if not ok and outkey in self.cache and outkey not in {h["key"] for h in inputs}:
                # Output is still uninitialized: direct its future writes to DDR.
                self.release(outkey, base)
                self.checks["output_spill_for_LUT"] += 1
                ok, base = self.allocate(lutkey, base)
            while not ok:
                victims = [
                    h["key"] for h in inputs if h["key"] in self.cache and h["key"] != lutkey
                ]
                assert victims, ("LUT reservation impossible", self.op, name)
                victim = max(victims, key=lambda k: self.objects[k]["bytes"])
                if self.cache[victim]["dirty"]:
                    base = self.dma(
                        victim,
                        self.objects[victim]["bytes"],
                        base,
                        True,
                        "spill",
                        "reserve LUT; preserve live input",
                    )
                    self.backing.add(victim)
                self.release(victim, base)
                self.checks["input_spill_for_LUT"] += 1
                ok, base = self.allocate(lutkey, base)
            if not self.cache[lutkey].get("loaded"):
                base = self.dma(
                    lutkey,
                    self.objects[lutkey]["bytes"],
                    base,
                    False,
                    "nonlinear_lut",
                    "load exact integer nonlinear table",
                )
                self.cache[lutkey]["loaded"] = True
        chunk = (
            max(axis, 1024 // max(1, axis) * axis) if name in ("layer_norm", "softmax") else 1024
        )
        for passno in range(passes):
            ends = []
            scratch_slots = [base, base]
            for pos in range(0, n, chunk):
                count = min(chunk, n - pos)
                scratch = (
                    sum(min(h["bytes"], count * h.get("width", 4)) for h in inputs) + count * 8
                )
                slots = 2 if scratch <= 32768 else 1
                assert scratch <= 65536, ("vector scratch overflow", self.op, name, scratch)
                slot = (pos // chunk) % slots
                cur = self.end(base, scratch_slots[slot])
                for h in inputs:
                    # Broadcast constants are read once per chunk, with their actual size.
                    p = h["bytes"] * pos // n
                    nb = h["bytes"] * (pos + count) // n - p
                    if h.get("elements", n) < n and name not in (
                        "reshape",
                        "repeat",
                        "embedding",
                        "cat",
                        "stack",
                        "tile",
                    ):
                        p = 0
                        nb = h["bytes"]
                    if nb:
                        cur = self.data_read(h, nb, cur, p)
                if name == "layer_norm":
                    cy = 6 * C(count, 32) + C(count, axis) * (32 + 128) + 64
                    cat = "nonlinear"
                elif name == "softmax":
                    cy = 4 * C(count, 32) + C(count, 2) + 64 + 2 * C(count, axis)
                    cat = "nonlinear"
                elif name in ("gelu", "tanh", "sin", "cos"):
                    cy = C(count, 2) + 2 * C(count, 32) + 64
                    cat = "nonlinear"
                elif name in ("mean", "max", "all", "nonzero", "is_nonzero", "item"):
                    cy = C(count, 32) + 64
                    cat = "reduce"
                elif name in (
                    "clone",
                    "copy_",
                    "index_put_",
                    "repeat",
                    "reshape",
                    "contiguous",
                    "cat",
                    "stack",
                    "tile",
                    "embedding",
                    "to",
                    "zeros",
                    "eye",
                    "arange",
                    "new_zeros",
                    "zeros_like",
                ):
                    cy = C(count, 32) + 2
                    cat = "layout"
                else:
                    cy = 2 * C(count, 32) + 2 + (64 if name == "div" else 0)
                    cat = "elementwise"
                cur = self.emit(
                    cat,
                    cy,
                    cur,
                    resource=["vector", "arena_read", "arena_write"] if lutkey else "vector",
                    reason=name + f" pass {passno+1}/{passes}",
                )
                if passno < passes - 1:
                    self.extra_work[cat] += cy
                if passno == passes - 1 and not (norm and self.level < 3):
                    if name in old.Model.REDUCE:
                        if pos + count == n:
                            cur = self.store(outkey, outbytes, cur, "activation", offset=0)
                    else:
                        nb = outbytes * (pos + count) // n - outbytes * pos // n
                        cur = self.store(outkey, nb, cur, "activation", offset=outbytes * pos // n)
                ends.append(cur)
                scratch_slots[slot] = cur
            base = self.end(*ends)
            if passno < passes - 1:
                base = self.emit(
                    "reduce",
                    2,
                    base,
                    resource="vector",
                    reason="full tensor peak ready before next pass",
                )
        if norm and self.level < 3:
            # Historical wide temporary path: explicitly write/read full 64-bit result.
            base = self.dma(
                "wide:" + outkey,
                n * 8,
                base,
                True,
                "wide_temporary",
                "unfused full tensor norm temporary",
            )
            base = self.dma(
                "wide:" + outkey,
                n * 8,
                base,
                False,
                "wide_temporary",
                "read after whole tensor peak is ready",
            )
            base = self.emit(
                "elementwise", C(n, 32) + 2, base, resource="vector", reason="global RNE shift"
            )
            base = self.store(outkey, outbytes, base, "activation")
        if lutkey:
            self.pinned.discard(lutkey)
        return {
            "key": outkey,
            "bytes": outbytes,
            "elements": n,
            "width": 4,
            "compact": False,
            "shape": [n],
        }, base

    def bind(self, h, outtid, base, consume=True):
        dest = self.handle(outtid)
        key = dest["key"]
        oldkey = h["key"]
        if h.get("compact") and self.level >= 2 and self.objects[key]["bytes"] == dest["bytes"] + 8:
            # Preserve q/scale plus the common norm shift; views decode on read.
            self.objects[key]["bytes"] = self.objects[oldkey]["bytes"]
            self.formats[key] = {**self.formats[oldkey], "logical_bytes": dest["bytes"]}
            if oldkey in self.cache:
                self.cache[key] = self.cache.pop(oldkey)
                self.allocations.append(
                    {
                        "op": self.op,
                        "action": "rename",
                        "key": oldkey,
                        "new_key": key,
                        "after": base,
                    }
                )
            elif oldkey in self.backing:
                self.backing.add(key)
            self.pinned.discard(oldkey)
            return base
        _, base = self.allocate(key, base)
        self.pinned.add(key)
        for pos in range(0, dest["bytes"], 4096):
            if pos >= h["bytes"]:
                break
            nb = min(4096, dest["bytes"] - pos, h["bytes"] - pos)
            rd = self.data_read(h, nb, base, pos)
            base = self.store(key, nb, rd, "activation", pos)
        if consume and oldkey.startswith("t"):
            self.finish_temp(h, base)
        return base

    def run(self):
        graph = json.loads((DATA / "capture.json").read_text())
        evs = {e["id"]: e for e in graph["events"]}
        for cmd in self.plan["commands"]:
            self.op = cmd["id"]
            self.s.now_op = self.op
            self.s.module = cmd["module"]
            self.s.stage = self.stage(cmd["module"])
            before = len(self.s.events)
            ev = evs[self.op]
            name = cmd["op"].split(".")[1]
            ins = [self.handle(i) for i in cmd["inputs"]]
            outs = [self.handle(i) for i in cmd["outputs"]]
            self.pinned = {h["key"] for h in ins}
            base = self.begin_command(cmd)
            start = self.s.end(base)
            self.metadata_seen = set()
            self.current_name = name
            self.later_inputs = {
                h["key"]
                for h in (
                    ins[2:]
                    if name in ("scaled_dot_product_attention", "linear", "conv2d")
                    else ins[:1]
                    if name == "baddbmm"
                    else []
                )
            }
            assert (
                name
                in old.Model.VIEW
                | old.Model.LAYOUT
                | old.Model.ELEMS
                | old.Model.REDUCE
                | old.Model.FILL
                | old.Model.NONLIN
                | old.Model.MATRIX
                | {"dropout"}
            ), ("unsupported operator", name)
            if not cmd["metadata_only"]:
                base = self.emit("control", 2, base, resource="control", reason="descriptor")
                if cmd["matrices"]:
                    if name == "scaled_dot_product_attention":
                        h, base = self.matrix(cmd["matrices"][0], ins[0], ins[1], base)
                        h2, base = self.vector("mul", [h], h["elements"], None, h["bytes"], base)
                        self.finish_temp(h, base)
                        if len(ins) > 3:
                            masked, base = self.vector(
                                "masked_fill_",
                                [h2, ins[3]],
                                h2["elements"],
                                None,
                                h2["elements"] * 8,
                                base,
                            )
                            masked["width"] = 8
                            self.finish_temp(h2, base)
                            h2 = masked
                        soft, base = self.vector(
                            "softmax",
                            [h2],
                            h2["elements"],
                            None,
                            h2["elements"] * 4,
                            base,
                            axis=cmd["matrices"][0]["n"],
                        )
                        self.finish_temp(h2, base)
                        self.later_inputs = set()
                        h, base = self.matrix(cmd["matrices"][1], soft, ins[2], base)
                        self.finish_temp(soft, base)
                        base = self.bind(h, cmd["outputs"][0], base)
                    else:
                        aa, bb = (ins[1], ins[2]) if name == "baddbmm" else (ins[0], ins[1])
                        if name == "conv2d":
                            r = cmd["matrices"][0]
                            ne = r["batch"] * r["m"] * r["k"]
                            aa, base = self.vector("reshape", [aa], ne, None, ne * 4, base)
                        h, base = self.matrix(cmd["matrices"][0], aa, bb, base)
                        if name == "conv2d":
                            self.finish_temp(aa, base)
                        if str(self.op) in self.plan.get("group_stream", {}):
                            base = self.bind(h, cmd["outputs"][0], base)
                        elif (name in ("linear", "conv2d") and len(ins) > 2) or name == "baddbmm":
                            bias = ins[0] if name == "baddbmm" else ins[2]
                            # Alpha/beta are 1 in the captured workload; preserve their numeric path.
                            if name == "baddbmm":
                                tmp, base = self.vector(
                                    "mul", [h], h["elements"], None, h["bytes"], base
                                )
                                self.finish_temp(h, base)
                                h = tmp
                                bias, base = self.vector(
                                    "mul", [bias], bias["elements"], None, bias["bytes"], base
                                )
                            dest = outs[0]
                            h2, base = self.vector(
                                "add", [h, bias], dest["elements"], dest["key"], dest["bytes"], base
                            )
                            self.finish_temp(h, base)
                            h = h2
                            if name == "baddbmm":
                                self.finish_temp(bias, base)
                        else:
                            base = self.bind(h, cmd["outputs"][0], base)
                else:
                    n = max([o["elements"] for o in outs] or [1])
                    outbytes = sum(o["bytes"] for o in outs) or 8
                    axis = 1
                    if name in ("max", "mean", "all", "nonzero", "item", "is_nonzero"):
                        n = max([h["elements"] for h in ins] or [1])
                    args = ev["args"]["tuple"]
                    if name == "layer_norm":
                        axis = math.prod(args[1])
                    if name == "softmax":
                        axis = ins[0]["shape"][args[1]]
                    if name in ("zeros", "eye", "arange", "new_zeros", "zeros_like"):
                        ins = []
                    if name == "embedding":
                        ins = [
                            {
                                **ins[0],
                                "bytes": n * 4,
                                "elements": n,
                                "gather_rows": self.embedding_indices[str(self.op)],
                                "rowbytes": ins[0]["shape"][-1] * 4,
                            },
                            *ins[1:],
                        ]
                    key = outs[0]["key"] if len(outs) == 1 else None
                    h, base = self.vector(name, ins, n, key, outbytes, base, axis)
                    if len(outs) > 1:
                        # Tuple results (e.g. max values and indices) are split by byte range.
                        off = 0
                        for tid in cmd["outputs"]:
                            base = self.bind({**h, "offset": off}, tid, base, consume=False)
                            off += self.handle(tid)["bytes"]
                        self.finish_temp(h, base)
            if not cmd["metadata_only"]:
                for oh in outs:
                    if oh["key"] not in self.formats:
                        base = self.store(
                            oh["key"], 8, base, "tensor_scale", self.objects[oh["key"]]["bytes"] - 8
                        )
            self.done[self.op] = base
            self.last = self.end(self.last, base)
            self.pinned.clear()
            self.cleanup(base)
            if self.op % 200 == 0:
                print(
                    "PROFILE",
                    self.op,
                    "events",
                    len(self.s.events),
                    "wire",
                    sum(x["wire_bytes"] for x in self.ledger),
                    flush=True,
                )
            self.ops.append(
                {
                    "op": self.op,
                    "operator": cmd["op"],
                    "module": cmd["module"],
                    "stage": self.s.stage,
                    "start": start,
                    "end": self.s.end(base),
                    "cycles": self.s.end(base) - start,
                    "events": len(self.s.events) - before,
                }
            )
        # Final action must be visible in DDR even if still resident.
        for tid in self.plan["output_tensors"]:
            h = self.handle(tid)
            if h["key"] in self.cache:
                self.last = self.dma(
                    h["key"],
                    h["bytes"],
                    self.last,
                    True,
                    "final_action",
                    "final 12x7 action external visibility",
                )
        result = self.s.summary(self.last)
        result.pop("critical_segments", None)
        payload = sum(x["payload_bytes"] for x in self.ledger)
        wire = sum(x["wire_bytes"] for x in self.ledger)
        result.update(
            {
                "model_revision": "v7-r5-tail-records",
                "payload_bytes": payload,
                "wire_bytes": wire,
                "padding_bytes": wire - payload,
                "effective_mac": self.mac,
                "operator_count": len(self.ops),
                "checked_ranges": dict(self.checks),
                "peak_arena_bytes": self.peak,
                "arena_capacity": self.arena,
                "extra_compute_cycles": dict(self.extra_work),
                "cache_served_bytes": dict(self.cache_hits),
                "bandwidth_bytes": self.bandwidth,
                "resources": self.plan["resources"],
                "evidence": "architecture prediction; buffer/port implementation not synthesized",
            }
        )
        assert self.mac == self.plan["effective_mac"]
        return result


def execute(plan, out, bandwidth=64, detail=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    e = Engine(plan, bandwidth)
    r = e.run()
    (out / "summary.json").write_text(json.dumps(r, indent=2))
    (out / "operators.json").write_text(json.dumps(e.ops, indent=2))
    (out / "allocations.json").write_text(json.dumps(e.allocations, indent=2))
    (out / "representations.json").write_text(json.dumps(e.formats, indent=2))
    if detail:
        for name, rows in [("events", e.s.events), ("transfers", e.ledger)]:
            with gzip.open(out / (name + ".jsonl.gz"), "wt", compresslevel=1) as f:
                for x in rows:
                    f.write(json.dumps(x, separators=(",", ":")) + "\n")
        with gzip.open(out / "transfers.csv.gz", "wt", newline="", compresslevel=1) as f:
            w = csv.DictWriter(
                f, fieldnames=list(dict.fromkeys(k for row in e.ledger for k in row))
            )
            w.writeheader()
            w.writerows(e.ledger)
    (out / "resolved_plan.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_revision": r["model_revision"],
                "config": plan["config"],
                "resources": plan["resources"],
                "storage_formats": plan.get("storage_formats", {}),
                "allocation_schedule": "allocations.json",
                "transfer_schedule": "transfers.jsonl.gz" if detail else None,
                "event_schedule": "events.jsonl.gz" if detail else None,
                "numeric_reference": "compiler v12 + v13 paired replay",
                "hardware_ready": False,
            },
            indent=2,
        )
    )
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bandwidth", type=int, default=64)
    ap.add_argument("--compact", action="store_true")
    a = ap.parse_args()
    r = execute(json.loads(Path(a.plan).read_text()), a.out, a.bandwidth, not a.compact)
    print(json.dumps(r, indent=2))
