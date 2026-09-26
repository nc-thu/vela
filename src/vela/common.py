"""Shared configuration, causal event scheduling, and operator categories.

An event reserves its resources for an integer number of cycles. The engines
extend these rules with bank hazards, finite storage, and operator scheduling.
"""
import json, math, csv, argparse, collections
from pathlib import Path
from dataclasses import dataclass, asdict, replace

C = lambda n, d: (n + d - 1) // d


def refs(x):
    if isinstance(x, dict):
        if "tensor" in x:
            yield x["tensor"]
        else:
            for v in x.values():
                yield from refs(v)
    elif isinstance(x, list):
        for v in x:
            yield from refs(v)


@dataclass
class Config:
    frequency_mhz: int = 300
    ddr_bytes: int = 16
    burst_latency: int = 0
    max_burst: int = 4096
    rows: int = 16
    cols: int = 96
    group: int = 64
    ctx_bytes: int = 16384
    wram_bytes: int = 98304
    result_elements: int = 1536
    result_banks: int = 2
    skid_beats: int = 4
    snapshot_banks: int = 2
    quant_lanes: int = 32
    merge_lanes: int = 24
    encode_lanes: int = 8
    scale_lanes: int = 1
    vector_lanes: int = 32
    lut_ports: int = 2
    vector_scratch_bytes: int = 16384
    divider_latency: int = 64
    divider_ii: int = 1
    sqrt_latency: int = 32
    array_fill: int = 16
    c2_speed: int = 1
    matrix_speed: int = 1
    nonlinear_speed: int = 1
    overlap: bool = True
    interop_overlap: bool = False


class Scheduler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.events = []
        self.last = {}
        self.now_op = 0
        self.stage = ""
        self.module = ""
        self.serial = -1

    def end(self, i):
        return self.events[i]["end"] if i >= 0 else 0

    def add(
        self,
        cat,
        duration,
        deps=(),
        resource=None,
        reason="",
        read=0,
        write=0,
        onchip=0,
        traffic="",
    ):
        if duration < 0:
            raise ValueError("negative duration")
        resources = [cat] if resource is None else resource
        resources = [resources] if isinstance(resources, str) else resources
        dep = set(i for i in deps if i is not None and i >= 0)
        if not self.cfg.overlap and self.serial >= 0:
            dep.add(self.serial)
        ready = max((self.end(i) for i in dep), default=0)
        dep.update(self.last[r] for r in resources if r in self.last)
        start = max((self.end(i) for i in dep), default=0)
        blockers = [i for i in dep if self.end(i) == start] if start else []
        ev = {
            "id": len(self.events),
            "op": self.now_op,
            "stage": self.stage,
            "module": self.module,
            "category": cat,
            "start": start,
            "end": start + int(duration),
            "ready": ready,
            "resource_wait": start - ready,
            "predecessors": sorted(dep),
            "critical_predecessors": blockers,
            "resources": resources,
            "reason": reason,
            "read_bytes": int(read),
            "write_bytes": int(write),
            "onchip_bytes": int(onchip),
            "traffic": traffic,
        }
        self.events.append(ev)
        for r in resources:
            self.last[r] = ev["id"]
        self.serial = ev["id"]
        return ev["id"]

    def dma(self, n, deps, write=False, traffic="activation"):
        if not n:
            return max(deps, key=self.end, default=-1)
        return self.add(
            "ddr_write" if write else "ddr_read",
            C(n, self.cfg.ddr_bytes) + C(n, self.cfg.max_burst) * self.cfg.burst_latency,
            deps,
            "ddr",
            reason=traffic,
            read=0 if write else n,
            write=n if write else 0,
            traffic=traffic,
        )

    def summary(self, terminal):
        end = self.end(terminal)
        critical = set()
        stack = [terminal]
        while stack:
            i = stack.pop()
            if i < 0 or i in critical:
                continue
            critical.add(i)
            stack.extend(self.events[i]["critical_predecessors"])
        # Union of all equal-length critical paths. Concurrent critical categories
        # are counted ONCE as 'joint', instead of arbitrarily selecting one cause.
        points = collections.defaultdict(collections.Counter)
        for i in critical:
            e = self.events[i]
            if e["end"] > e["start"]:
                points[e["start"]][e["category"]] += 1
                points[e["end"]][e["category"]] -= 1
        active = collections.Counter()
        previous = 0
        partition = collections.Counter()
        critical_segments = []
        for t, changes in sorted(points.items()):
            if t > previous:
                cats = sorted(k for k, v in active.items() if v > 0)
                cat = cats[0] if len(cats) == 1 else "joint" if cats else "unexplained"
                partition[cat] += t - previous
                if (
                    critical_segments
                    and critical_segments[-1][2] == cat
                    and critical_segments[-1][1] == previous
                ):
                    critical_segments[-1][1] = t
                else:
                    critical_segments.append([previous, t, cat])
            active.update(changes)
            previous = t
        if previous < end:
            partition["unexplained"] += end - previous
        assert sum(partition.values()) == end
        assert not partition.get("unexplained"), partition
        busy = collections.Counter()
        wait = collections.Counter()
        traffic = collections.Counter()
        stage_busy = collections.defaultdict(collections.Counter)
        for e in self.events:
            busy[e["category"]] += e["end"] - e["start"]
            wait[e["category"]] += e["resource_wait"]
            traffic[e["traffic"]] += e["read_bytes"] + e["write_bytes"]
            stage_busy[e["stage"]][e["category"]] += e["end"] - e["start"]
        return {
            "cycles": end,
            "ms": end / (self.cfg.frequency_mhz * 1000),
            "calls_per_s": self.cfg.frequency_mhz * 1e6 / end,
            "critical_cycles": dict(partition),
            "busy_cycles": dict(busy),
            "resource_wait_cycles": dict(wait),
            "traffic_bytes": dict(traffic),
            "stage_busy": dict(stage_busy),
            "critical_segments": critical_segments,
            "critical_event_count": len(critical),
        }


class Model:
    VIEW = {
        "view",
        "transpose",
        "permute",
        "unsqueeze",
        "squeeze",
        "flatten",
        "unflatten",
        "expand",
        "slice",
        "select",
        "unbind",
        "split_with_sizes",
        "detach_",
        "lift_fresh",
        "meshgrid",
    }
    LAYOUT = {
        "clone",
        "repeat",
        "copy_",
        "index_put_",
        "reshape",
        "contiguous",
        "cat",
        "stack",
        "tile",
        "embedding",
        "to",
    }
    ELEMS = {
        "add",
        "add_",
        "sub",
        "rsub",
        "mul",
        "div",
        "neg",
        "clamp",
        "relu",
        "eq",
        "__ior__",
        "bitwise_not",
        "masked_fill_",
        "isfinite",
    }
    REDUCE = {"mean", "max", "all", "nonzero", "is_nonzero", "item"}
    FILL = {"zeros", "eye", "arange", "new_zeros", "zeros_like"}
    NONLIN = {"layer_norm", "softmax", "gelu", "tanh", "cos", "sin"}
    MATRIX = {"linear", "conv2d", "bmm", "baddbmm", "scaled_dot_product_attention"}
