"""Finite-buffer architecture experiment. FP16 numerical quality is NOT tested."""
import json, sys, math, collections, argparse, gzip, time, heapq
from pathlib import Path
import numpy as np
import gc, functools, os

gc.disable()  # event graph contains no reference cycles; avoid repeated full-heap scans
from . import base_engine as b

P = Path(__file__).resolve().parent
from ..compiler.compile_schedule import build

C = b.C


class Scheduler(b.LeanScheduler):
    def __init__(self, cfg, banks):
        super().__init__(cfg)
        self.banks = banks
        self.bank_resources = {
            r: tuple(r + ":" + str(i) for i in range(banks)) for r in ("arena_read", "arena_write")
        }
        self.resource_sets = {}

    def add(self, cat, duration, deps=(), resource=None, **kw):
        rs = (
            [cat]
            if resource is None
            else ([resource] if isinstance(resource, str) else list(resource))
        )
        expanded = []
        for r in rs:
            if r in ("arena_read", "arena_write"):
                expanded += self.bank_resources[r]
            else:
                expanded.append(r)
        signature = tuple(expanded)
        resources = self.resource_sets.setdefault(signature, signature)
        return super().add(cat, duration, deps, resources, **kw)


from .stream_mixin import StreamMixin


class Engine(StreamMixin, b.Engine):
    def __init__(self, p):
        super().__init__(p, 64)
        self.banks = p["resources"]["arena_banks"]
        self.port = p["resources"]["arena_port_bytes"]
        self.s = Scheduler(b.old.Config(ddr_bytes=64), self.banks)
        self.done = {}
        self.hazards = {}
        self.retired = -1
        self.matrix_done = -1
        self.quant_done = -1
        self.vector_done = -1
        self.sfu_done = -1
        self.stats = collections.Counter()
        self.sfu_work = collections.Counter()
        self.streams = []
        self.micro_cache = {}
        self.micro_examples = {}
        self.quant_leases = []
        self.raw_leases = []

    def bank_ids(self, words):
        bits = self.cfg.get("bank_xor_bits", 3 if self.banks == 8 else 4)
        z = np.array(words, dtype=np.int64)
        out = z.copy()
        for sh in range(bits, 48, bits):
            out ^= z >> sh
        return out & (self.banks - 1)

    def dma(self, key, n, base, write=False, kind="activation", reason="", port=64, offset=0):
        # External bus stays 64 B/cycle. Only the on-chip side of the two
        # finite 4 KiB width-adapter buffers is widened and separately budgeted.
        actual = self.port if port == 64 else port
        return super().dma(key, n, base, write, kind, reason, actual, offset)

    def begin_command(self, c):
        deps = [self.done[i] for i in c["schedule_dependencies"]]
        if self.cfg.get("serial"):
            deps.append(self.last)
        return self.end(*deps)

    def release(self, key, base):
        if key in self.cache:
            # Physical storage may be reused only after all prior readers/writers retire.
            self.retired = self.end(self.retired, base, self.last_access.get(key, -1))
            for x in self.hazards.pop(key, {}).values():
                self.retired = self.end(self.retired, *x)
        super().release(key, self.end(base, self.retired))

    def allocate(self, key, base):
        was = key in self.cache
        ok, end = super().allocate(key, base)
        if ok and not was:
            end = self.end(end, self.retired)
            self.allocations[-1]["after"] = end
        return ok, end

    def hazard(self, key, offset, n, write, base):
        records = self.hazards.setdefault(key, {})
        pages = range(offset // 1024, C(offset + n, 1024))
        deps = [base]
        for p in pages:
            wr, rd = records.get(p, (-1, -1))
            deps.append(wr)
            if write:
                deps.append(rd)
        return self.end(*deps), pages

    def local(self, key, n, base, write=False, port=64, reason="", offset=0):
        base, pages = self.hazard(key, offset, n, write, base)
        width = min(port, self.port) if port < 64 else self.port
        end = self.emit(
            "local_move",
            C(offset % width + n, width),
            base,
            resource="arena_write" if write else "arena_read",
            reason=reason or "banked resident access",
            onchip=n,
        )
        for p in pages:
            wr, rd = self.hazards[key].get(p, (-1, -1))
            self.hazards[key][p] = (end, -1) if write else (wr, self.end(rd, end))
        self.last_access[key] = self.end(self.last_access.get(key, -1), end)
        self.s.events[end]["storage_access"] = {
            "storage": key,
            "offset": offset,
            "bytes": n,
            "write": write,
            "hazard_block_bytes": 1024,
        }
        return end

    @functools.lru_cache(maxsize=8192)
    def gather_pattern(self, ranges, physical_offset):
        words = np.unique(
            np.concatenate([np.arange(o // 8, C(o + n, 8), dtype=np.int64) for o, n in ranges])
        )
        counts = np.bincount(self.bank_ids(words + physical_offset // 8), minlength=self.banks)
        return (
            max(1, int(counts.max()), C(len(words) * 8, 64)),
            tuple("arena_read:" + str(i) for i, c in enumerate(counts) if c),
            len(words),
        )

    def gather_ranges(self, key, ranges, base):
        dur, rs, nwords = self.gather_pattern(tuple(ranges), self.cache[key]["offset"])
        for o, n in ranges:
            base, _ = self.hazard(key, o, n, False, base)
        end = self.emit(
            "local_move",
            dur,
            base,
            resource=rs,
            reason="coalesced 64-bit word gather with bank conflicts",
            onchip=sum(n for o, n in ranges),
        )
        for o, n in ranges:
            for page in range(o // 1024, C(o + n, 1024)):
                wr, rd = self.hazards[key].get(page, (-1, -1))
                self.hazards[key][page] = (wr, self.end(rd, end))
        self.last_access[key] = self.end(end, self.last_access.get(key, -1))
        self.stats["gather_words"] += nwords
        return end

    def grouped_read(self, key, rows, count, base, offset, stride, port):
        if key not in self.cache:
            return super().grouped_read(key, rows, count, base, offset, stride, port)
        assert offset + (rows - 1) * stride + count <= self.objects[key]["bytes"]
        self.cache_hits["quantized_operand"] += rows * count
        ranges = [(offset + i * stride, count) for i in range(rows)]
        return self.gather_ranges(key, ranges, base)

    def quantize(self, h, rows, k, base, side):
        # Shared 16 KiB quantization scratch cannot hold two independent contexts.
        out, end = super().quantize(h, rows, k, self.end(base, self.quant_done), side)
        self.quant_done = end
        return out, end

    def matrix(self, r, ha, hb, base):
        out, end = super().matrix(r, ha, hb, self.end(base, self.matrix_done))
        self.matrix_done = end
        return out, end

    def bind(self, h, outtid, base, consume=True):
        dest = self.handle(outtid)
        key = dest["key"]
        oldkey = h["key"]
        # Rename a private temporary rather than copying identical bytes.
        if (
            consume
            and oldkey.startswith("t")
            and not h.get("compact")
            and h["bytes"] == dest["bytes"]
            and h.get("offset", 0) == 0
            and oldkey in self.cache
            and key not in self.cache
        ):
            self.cache[key] = self.cache.pop(oldkey)
            self.hazards[key] = self.hazards.pop(oldkey, {})
            self.last_access[key] = self.last_access.get(oldkey, base)
            self.allocations.append(
                {"op": self.op, "action": "rename", "key": oldkey, "new_key": key, "after": base}
            )
            self.pinned.discard(oldkey)
            self.stats["alias_saved_bytes"] += 2 * dest["bytes"]
            return base
        return super().bind(h, outtid, base, consume)

    def pipe(self, unit, n, base, precision="FP16"):
        lanes = self.cfg["sfu_lanes"]
        lat = self.plan["sfu"]["latencies"][unit] * self.cfg["sfu_latency_factor"]
        count = C(n, lanes)
        issue = self.emit(
            "nonlinear",
            count,
            base,
            resource="sfu_" + unit,
            reason=precision + " " + unit + " issue; II=1 per lane",
        )
        done = self.emit(
            "nonlinear",
            max(0, lat - 1),
            issue,
            resource=[],
            reason=precision + " " + unit + " pipeline drain",
        )
        self.sfu_work[precision + "_" + unit] += n
        return done

    def function(self, name, n, axis, base):
        if name in ("gelu", "tanh", "sin", "cos"):
            return self.elementwise_sfu(name, n, base)
        cv = self.pipe("convert", n, base)
        rows = C(n, axis)
        if name == "gelu":
            # tanh approximation: x*x*x; .044715*x^3; +x; *sqrt(2/pi);
            # tanh(z)=1-2/(exp(2z)+1); *.5*x. Explicit shared mul/add use.
            x2 = self.pipe("mul", n, cv)
            x3 = self.pipe("mul", n, x2)
            a = self.pipe("mul", n, x3)
            a = self.pipe("add", n, a)
            a = self.pipe("mul", n, a)
            a = self.pipe("mul", n, a)
            a = self.pipe("exp", n, a)
            a = self.pipe("add", n, a)
            a = self.pipe("reciprocal", n, a)
            a = self.pipe("mul", n, a)
            a = self.pipe("add", n, a)
            a = self.pipe("add", n, a)
            a = self.pipe("mul", n, a)
            a = self.pipe("mul", n, a)
        elif name == "tanh":
            a = self.pipe("mul", n, cv)
            a = self.pipe("exp", n, a)
            a = self.pipe("add", n, a)
            a = self.pipe("reciprocal", n, a)
            a = self.pipe("mul", n, a)
            a = self.pipe("add", n, a)
        elif name in ("sin", "cos"):
            a = self.pipe("sincos", n, cv)
        elif name == "softmax":
            a = self.pipe("reduce", n, cv, "FP32 max")
            a = self.pipe("add", n, a, "FP32 subtract")
            a = self.pipe("convert", n, a)
            a = self.pipe("exp", n, a)
            a = self.pipe("convert", n, a)
            a = self.pipe("reduce", n, a, "FP32 sum")
            a = self.pipe("reciprocal", rows, a, "FP32 reciprocal")
            a = self.pipe("mul", n, a, "FP32 normalize")
        elif name == "layer_norm":
            a = self.pipe("reduce", n, cv, "FP32 sum")
            a = self.pipe("mul", rows, a, "FP32 mean")
            a = self.pipe("add", n, a, "FP32 centered")
            a = self.pipe("mul", n, a, "FP32 square")
            a = self.pipe("reduce", n, a, "FP32 variance sum")
            a = self.pipe("mul", rows, a, "FP32 variance mean")
            a = self.pipe("add", rows, a, "FP32 epsilon")
            a = self.pipe("rsqrt", rows, a, "FP32 inverse std")
            a = self.pipe("mul", n, a, "FP32 normalize")
            a = self.pipe("mul", n, a, "FP32 gamma")
            a = self.pipe("add", n, a, "FP32 beta")
        else:
            raise ValueError(name)
        return self.pipe("convert", n, a, "FP16 output / INT32 scale encoding")

    def elementwise_sfu(self, name, n, base):
        # Fine-grain ready scheduling INSIDE the SFU, compressed into one reservation.
        # Every 32-value microblock traverses the chain, sharing one unit of each type
        # per lane. Unit issue interval is 1; pipeline drain does not monopolize it.
        chains = {
            "gelu": [
                "convert",
                "mul",
                "mul",
                "mul",
                "add",
                "mul",
                "mul",
                "exp",
                "add",
                "reciprocal",
                "mul",
                "add",
                "add",
                "mul",
                "mul",
                "convert",
            ],
            "tanh": ["convert", "mul", "exp", "add", "reciprocal", "mul", "add", "convert"],
            "sin": ["convert", "sincos", "convert"],
            "cos": ["convert", "sincos", "convert"],
        }
        chain = chains[name]
        key = (name, n, self.cfg["sfu_lanes"], self.cfg["sfu_latency_factor"])
        if key not in self.micro_cache:
            queue = []
            avail = collections.Counter()
            issued = collections.Counter()
            trace = []
            finish = 0
            for block, pos in enumerate(range(0, n, 32)):
                heapq.heappush(queue, (0, block, 0, min(32, n - pos)))
            while queue:
                ready, block, step, count = heapq.heappop(queue)
                unit = chain[step]
                start = max(ready, avail[unit])
                cycles = C(count, self.cfg["sfu_lanes"])
                avail[unit] = start + cycles
                end = (
                    avail[unit]
                    + self.plan["sfu"]["latencies"][unit] * self.cfg["sfu_latency_factor"]
                    - 1
                )
                issued[unit] += cycles
                trace.append(
                    {
                        "block": block,
                        "step": step,
                        "unit": unit,
                        "start": start,
                        "issue_end": avail[unit],
                        "end": end,
                        "values": count,
                    }
                )
                finish = max(finish, end)
                if step + 1 < len(chain):
                    heapq.heappush(queue, (end, block, step + 1, count))
            self.micro_cache[key] = (finish, dict(issued))
            self.micro_examples["/".join(map(str, key))] = trace
        cycles, issued = self.micro_cache[key]
        for unit in chain:
            self.sfu_work["FP16_" + unit] += n
        for unit, work in issued.items():
            self.stats["sfu_issue_cycles_" + unit] += work
        return self.emit(
            "nonlinear",
            cycles,
            base,
            resource="sfu_microprogram",
            reason=name + " microblock-ready pipeline; shared arithmetic units; II=1",
        )

    def vector(self, name, inputs, n, outkey, outbytes, base, axis=1, final=False):
        if (
            name == "gelu"
            and inputs
            and self.formats.get(inputs[0]["key"], {}).get("kind") == "g64_after_gelu"
        ):
            return self.forward_gelu(inputs[0], outkey, base)
        if not self.cfg["sfu_lanes"] or name not in b.old.Model.NONLIN:
            h, end = super().vector(
                name, inputs, n, outkey, outbytes, self.end(base, self.vector_done), axis, final
            )
            self.vector_done = end
            return h, end
        base = self.end(base, self.sfu_done)
        norm = name != "softmax"
        passes = 2 if norm else 1
        if outkey is None:
            outkey, base = self.temp(outbytes, base, name)
        else:
            self.pinned.discard(outkey)
            _, base = self.allocate(outkey, base)
            self.pinned.add(outkey)
        chunk = max(axis, (512 // axis) * axis) if name in ("softmax", "layer_norm") else 512
        retain = passes == 2 and n * 8 <= self.cfg["raw_cache_bytes"]
        raw_birth = self.s.end(base)
        for passno in range(passes):
            slots = [base, base]
            ends = []
            for pos in range(0, n, chunk):
                count = min(chunk, n - pos)
                scratch = count * 32
                assert scratch <= 65536, ("SFU scratch", name, axis, scratch)
                nslots = 2 if scratch <= 32768 else 1
                slot = (pos // chunk) % nslots
                cur = self.end(base, slots[slot])
                acquire = self.s.end(cur)
                for h in [] if retain and passno == 1 else inputs:
                    p = h["bytes"] * pos // n
                    nb = h["bytes"] * (pos + count) // n - p
                    if h.get("elements", n) < n:
                        p = 0
                        nb = h["bytes"]
                    if nb:
                        cur = self.data_read(h, nb, cur, p)
                first = self.s.end(cur)
                if retain and passno == 1:
                    cur = self.emit(
                        "local_move",
                        C(count * 8, self.cfg["raw_cache_port"]),
                        cur,
                        resource="raw_cache_read",
                        reason="retained identical SFU result; finite URAM buffer",
                        onchip=count * 8,
                    )
                    cur = self.pipe("convert", count, cur, "cached result / output scale encoding")
                    self.stats["sfu_recompute_values_avoided"] += count
                else:
                    cur = self.function(name, count, axis, cur)
                    if retain:
                        cur = self.emit(
                            "local_move",
                            C(count * 8, self.cfg["raw_cache_port"]),
                            cur,
                            resource="raw_cache_write",
                            reason="retain first-pass SFU result before global scale barrier",
                            onchip=count * 8,
                        )

                if passno == passes - 1:
                    nb = outbytes * (pos + count) // n - outbytes * pos // n
                    cur = self.store(outkey, nb, cur, "activation", outbytes * pos // n)
                else:
                    self.extra_work["sfu_recompute_elements"] += count
                self.streams.append(
                    {
                        "op": self.op,
                        "function": name,
                        "pass": passno,
                        "slot": slot,
                        "bytes": scratch,
                        "start": acquire,
                        "compute_ready": first,
                        "end": self.s.end(cur),
                        "values": count,
                    }
                )
                slots[slot] = cur
                ends.append(cur)
            base = self.end(*ends)
            if passno < passes - 1:
                base = self.emit(
                    "reduce",
                    C(n, 32) + 2,
                    base,
                    resource="sfu_peak",
                    reason="whole-tensor output exponent barrier; no early consumer",
                )
        if retain:
            self.raw_leases.append(
                {"op": self.op, "bytes": n * 8, "start": raw_birth, "end": self.s.end(base)}
            )
        self.sfu_done = base
        return {
            "key": outkey,
            "bytes": outbytes,
            "elements": n,
            "width": 4,
            "compact": False,
            "shape": [n],
        }, base


def execute(lanes, port, out, latency=1, serial=False, detail=False, limit=None):
    p = build(
        lanes,
        port,
        latency,
        serial,
        raw_banks=int(os.environ.get("RAW_BANKS", "0")),
        config=os.environ.get("W4A8_CONFIG", "A"),
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if limit:
        p["commands"] = p["commands"][:limit]
        p["output_tensors"] = []
        p["effective_mac"] = sum(
            r["batch"] * r["m"] * r["n"] * r["k"] for c in p["commands"] for r in c["matrices"]
        )
    (out / "plan.json").write_text(json.dumps(p, indent=2))
    e = Engine(p)
    t = time.time()
    r = e.run()
    r.update(
        model_revision="v11-group-stream-r1",
        sfu_work=dict(e.sfu_work),
        improvements=dict(e.stats),
        elapsed_seconds=time.time() - t,
        sfu_lanes=lanes,
        port_bytes=port,
        latency_factor=latency,
        serial=serial,
    )
    # Count union busy per five classes and simultaneous activity without summing
    # overlapping SFU micro-operations as if they were sequential work.
    groups = {
        "local_move": "Memory",
        "weight_read": "Memory",
        "ddr_read": "Memory",
        "ddr_write": "Memory",
        "gemm": "Matrix",
        "quantize": "Quantization",
        "scale": "Quantization",
        "merge": "Quantization",
        "encode": "Quantization",
        "restore": "Quantization",
        "nonlinear": "SFU",
        "elementwise": "Vector",
        "reduce": "Vector",
        "layout": "Vector",
        "control": "Vector",
    }
    spans = collections.defaultdict(list)
    for ev in e.s.events:
        spans[groups[ev["category"]]].append((ev["start"], ev["end"]))
    points = collections.defaultdict(collections.Counter)
    busy = {}
    for g, ss in spans.items():
        ss.sort()
        merged = []
        for a, z in ss:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(z, merged[-1][1])
            else:
                merged.append([a, z])
        busy[g] = sum(z - a for a, z in merged)
        for a, z in merged:
            points[a][g] += 1
            points[z][g] -= 1
        spans[g] = merged
    active = collections.Counter()
    last = 0
    dist = collections.Counter()
    for t, delta in sorted(points.items()):
        dist[sum(v > 0 for v in active.values())] += t - last
        active.update(delta)
        last = t
    resource_work = collections.Counter()
    for ev in e.s.events:
        for res in ev["resources"]:
            resource_work[res] += ev["end"] - ev["start"]
    r.update(
        group_busy_union_cycles=busy,
        concurrency_cycles=dict(dist),
        resource_work_cycles=dict(resource_work),
        ideal_resource_lower_bound_cycles=max(resource_work.values()),
    )
    r["gather_cache"] = str(e.gather_pattern.cache_info())
    r["goal_access_below_matrix"] = r["critical_cycles"].get("local_move", 0) < r[
        "critical_cycles"
    ].get("gemm", 0)
    reasons = collections.Counter()
    critical_reasons = collections.Counter()

    def reason_group(s):
        if s.startswith(("resident ", "read after load ")):
            return "Resident contiguous read"
        if s.startswith("store "):
            return "Resident write"
        if "gather" in s or "bank" in s:
            return "Gather and bank conflicts"
        if "port" in s or "DMA" in s:
            return "DMA adapters"
        if "packed" in s or "group" in s:
            return "Packed groups"
        return s

    for ev in e.s.events:
        if ev["category"] == "local_move":
            reasons[reason_group(ev["reason"])] += ev["end"] - ev["start"]
    node = e.last
    while node >= 0:
        ev = e.s.events[node]
        if ev["category"] == "local_move":
            critical_reasons[reason_group(ev["reason"])] += ev["end"] - ev["start"]
        node = min(ev["critical_predecessors"]) if ev["critical_predecessors"] else -1
    r.update(
        local_access_busy_by_reason=dict(reasons),
        local_access_critical_by_reason=dict(critical_reasons),
    )
    (out / "summary.json").write_text(json.dumps(r, indent=2))
    (out / "operators.json").write_text(json.dumps(e.ops, indent=2))
    (out / "allocations.json").write_text(json.dumps(e.allocations, indent=2))
    (out / "streams.json").write_text(json.dumps(e.streams, indent=2))
    (out / "union_timeline.json").write_text(json.dumps(spans))
    (out / "quant_leases.json").write_text(json.dumps(e.quant_leases))
    (out / "raw_leases.json").write_text(json.dumps(e.raw_leases))
    (out / "validation.json").write_text(json.dumps(validate(e), indent=2))
    (out / "sfu_microtraces.json").write_text(json.dumps(e.micro_examples, indent=2))
    if detail:
        for name, rows in [("events", e.s.events), ("transfers", e.ledger)]:
            with gzip.open(out / (name + ".jsonl.gz"), "wt", compresslevel=1) as f:
                for row in rows:
                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
    return r


def validate(e):
    last = {}
    total = 0
    slots = [0, 0]
    for q in e.quant_leases:
        used = [q["slot"]] if q["slots"] == 2 else [0, 1]
        assert all(slots[i] <= q["start"] for i in used)
        assert q["bytes"] * q["slots"] <= 16384
        for i in used:
            slots[i] = q["end"]
    prev = 0
    for q in e.raw_leases:
        assert q["start"] >= prev and q["bytes"] <= e.cfg["raw_cache_bytes"]
        prev = q["end"]
    for x in e.s.events:
        assert all(e.s.end(d) <= x["start"] for d in x["predecessors"])
        for res in x["resources"]:
            assert last.get(res, 0) <= x["start"], res
            last[res] = x["end"]
        total += x["read_bytes"] + x["write_bytes"]
    assert total == sum(t["wire_bytes"] for t in e.ledger)
    assert e.peak <= e.arena
    for x in e.ops:
        assert x["end"] >= x["start"]
    # Reconstruct physical allocation lifetimes, including renames.
    live = {}
    intervals = []
    for a in e.allocations:
        k = a["key"]
        t = e.s.end(a["after"])
        if a["action"] == "alloc":
            assert k not in live, (k, "duplicate allocation")
            live[k] = (a["offset"], a["bytes"], t)
        elif a["action"] == "rename":
            if k in live:
                live[a["new_key"]] = live.pop(k)
        elif k in live:
            off, size, birth = live.pop(k)
            assert t >= birth
            intervals.append((birth, t, off, off + size))
    terminal = max(x["end"] for x in e.s.events)
    intervals.extend((birth, terminal, off, off + size) for off, size, birth in live.values())
    ends = []
    seq = 0
    for birth, death, lo, hi in sorted(intervals):
        ends = [x for x in ends if x[0] > birth]
        assert all(hi <= a or lo >= z for d, a, z in ends), (
            "physical reuse overlaps live allocation",
            birth,
            lo,
            hi,
        )
        if death > birth:
            ends.append((death, lo, hi))
    for name, trace in e.micro_examples.items():
        last_unit = {}
        previous = {}
        for a in trace:
            assert a["start"] >= last_unit.get(a["unit"], 0)
            last_unit[a["unit"]] = a["issue_end"]
            assert a["start"] >= previous.get(a["block"], 0)
            previous[a["block"]] = a["end"]
    slots = [0, 0]
    for x in e.streams:
        uses = [0, 1] if x["bytes"] > 32768 else [x["slot"]]
        assert x["bytes"] <= 65536 and all(slots[i] <= x["start"] for i in uses)
        for i in uses:
            slots[i] = x["end"]
    return {
        "quant_row_slots": True,
        "raw_cache_capacity_and_lifetime": True,
        "dependencies": True,
        "bank_ports": True,
        "DDR_ledger": True,
        "arena_capacity": True,
        "physical_allocation_lifetimes": True,
        "SFU_microblock_dependencies_and_II": True,
        "SFU_finite_slots": True,
        "operators": len(e.ops),
        "MAC": e.mac,
        "numeric_quality_tested": False,
        "events": len(e.s.events),
    }


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--lanes", type=int, default=32, choices=[32])
    a.add_argument("--port", type=int, default=128)
    a.add_argument("--out", required=True)
    a.add_argument("--latency", type=int, default=1)
    a.add_argument("--serial", action="store_true")
    a.add_argument("--detail", action="store_true")
    a.add_argument("--limit", type=int)
    x = a.parse_args()
    print(
        json.dumps(
            execute(x.lanes, x.port, x.out, x.latency, x.serial, x.detail, x.limit), indent=2
        )
    )
