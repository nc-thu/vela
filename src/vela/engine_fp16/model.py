"""v12 vector-core integration; timing model, not numeric execution or RTL."""
import argparse
import os
import collections
import heapq
import json
from pathlib import Path
from . import inherited_model as m

P = Path(__file__).resolve().parent
C = m.C


class Engine(m.Engine):
    def __init__(self, plan):
        super().__init__(plan)
        self.unit_issues = collections.Counter()
        self.function_work = collections.defaultdict(collections.Counter)

    def pipe(self, unit, n, base, precision="FP16"):
        lanes = self.plan["sfu"]["values_per_cycle"][unit]
        count = C(n, lanes)
        lat = self.plan["sfu"]["latencies"][unit]
        issue = self.emit(
            "nonlinear",
            count,
            base,
            resource="sfu_" + unit,
            reason=precision + " " + unit + " mixed-unit issue",
        )
        done = self.emit(
            "nonlinear",
            lat - 1,
            issue,
            resource=[],
            reason=precision + " " + unit + " pipeline drain",
        )
        self.unit_issues[unit] += count
        self.sfu_work[precision + "_" + unit] += n
        return done

    def elementwise_sfu(self, name, n, base):
        assert name in self.plan["sfu"]["elementwise_functions"]
        chain = self.plan["sfu"]["elementwise_chain"]
        key = (name, n)
        if key not in self.micro_cache:
            queue = [(0, b, 0, min(32, n - pos)) for b, pos in enumerate(range(0, n, 32))]
            heapq.heapify(queue)
            avail = collections.Counter()
            issued = collections.Counter()
            trace = []
            finish = 0
            while queue:
                ready, block, step, count = heapq.heappop(queue)
                unit = chain[step]
                start = max(ready, avail[unit])
                cycles = C(count, self.plan["sfu"]["values_per_cycle"][unit])
                avail[unit] = start + cycles
                end = avail[unit] + self.plan["sfu"]["latencies"][unit] - 1
                issued[unit] += cycles
                trace.append(
                    dict(
                        block=block,
                        step=step,
                        unit=unit,
                        start=start,
                        issue_end=avail[unit],
                        end=end,
                        values=count,
                    )
                )
                finish = max(finish, end)
                if step + 1 < len(chain):
                    heapq.heappush(queue, (end, block, step + 1, count))
            self.micro_cache[key] = (finish, dict(issued))
            self.micro_examples[f"{name}/{n}/PWL64"] = trace
        cycles, issued = self.micro_cache[key]
        for unit in chain:
            self.sfu_work["FP16_" + unit] += n
        for unit, work in issued.items():
            self.stats["sfu_issue_cycles_" + unit] += work
            self.unit_issues[unit] += work
        self.function_work[name]["values"] += n
        self.function_work[name]["compute_cycles"] += cycles
        # A compressed block owns the actual shared arithmetic resources.
        # This prevents an independent softmax/LN issue from using the same units.
        return self.emit(
            "nonlinear",
            cycles,
            base,
            resource=["sfu_microprogram", "sfu_convert", "sfu_mul", "sfu_add"],
            reason=name + " PWL64 32-value microblocks; shared convert/mul/add",
        )

    def function(self, name, n, axis, base):
        before = len(self.s.events)
        end = super().function(name, n, axis, base)
        if name not in self.plan["sfu"]["elementwise_functions"]:
            self.function_work[name]["values"] += n
            self.function_work[name]["compute_cycles"] += sum(
                e["end"] - e["start"]
                for e in self.s.events[before:]
                if e["category"] == "nonlinear"
            )
        return end

    def run(self):
        r = super().run()
        # Compute module unions, not sums of overlapping stage durations.
        module_cats = {
            "matrix": {"gemm"},
            "quantization": {"quantize", "scale", "merge", "encode", "restore"},
            "sfu": {"nonlinear"},
            "vector": {"elementwise", "reduce", "control"},
            "layout": {"layout"},
            "onchip": {"local_move", "weight_read"},
            "ddr": {"ddr_read", "ddr_write"},
        }
        cat_to_module = {c: k for k, cs in module_cats.items() for c in cs}
        spans = collections.defaultdict(list)
        for ev in self.s.events:
            spans[cat_to_module[ev["category"]]].append((ev["start"], ev["end"]))
        unions = {}
        for k, ss in spans.items():
            ss.sort()
            total = 0
            lo = hi = 0
            for a, z in ss:
                if a > hi:
                    total += hi - lo
                    lo, hi = a, z
                else:
                    hi = max(hi, z)
            unions[k] = total + hi - lo
        stage_path = collections.defaultdict(collections.Counter)
        cause_path = collections.Counter()
        node = self.last
        while node >= 0:
            ev = self.s.events[node]
            d = ev["end"] - ev["start"]
            stage_path[ev["stage"]][cat_to_module[ev["category"]]] += d
            cause_path[ev["reason"]] += d
            node = min(ev["critical_predecessors"]) if ev["critical_predecessors"] else -1
        r.update(
            module_busy_union_cycles=unions,
            unit_issue_cycles=dict(self.unit_issues),
            sfu_function_work={k: dict(v) for k, v in self.function_work.items()},
            stage_critical_cycles={k: dict(v) for k, v in stage_path.items()},
            critical_reason_cycles=dict(cause_path),
            vector_core_spec=self.plan["sfu"],
        )
        return r


m.Engine = Engine


def execute(out, limit=None):
    # Capture the engine instance m.execute builds so the critical path can be
    # inspected afterwards without changing any scheduling semantics.
    real = m.Engine
    captured = []

    class Capturing(real):
        def __init__(self, plan):
            captured.append(self)
            super().__init__(plan)

    m.Engine = Capturing
    try:
        r = m.execute(32, 128, out, 1, False, False, limit)
    finally:
        m.Engine = real
    if os.environ.get("DUMP_CP") == "1" and captured:
        e = captured[0]
        chain = []
        node = e.last
        while node >= 0:
            ev = e.s.events[node]
            chain.append(ev)
            node = min(ev["critical_predecessors"]) if ev["critical_predecessors"] else -1
        lm = [ev for ev in chain if ev["category"] == "local_move"]
        agg = {}
        for ev in lm:
            agg[ev["reason"]] = agg.get(ev["reason"], 0) + ev["end"] - ev["start"]
        dump = [
            {
                "start": ev["start"],
                "end": ev["end"],
                "dur_ms": (ev["end"] - ev["start"]) / 3e5,
                "reason": ev["reason"],
                "resource_wait": ev["resource_wait"],
                "resources": list(ev["resources"]),
            }
            for ev in sorted(lm, key=lambda x: -(x["end"] - x["start"]))[:80]
        ]
        (Path(out) / "critical_localmove.json").write_text(
            json.dumps(
                {
                    "total_lm_events": len(lm),
                    "total_lm_cycles": sum(ev["end"] - ev["start"] for ev in lm),
                    "by_reason_ms": {
                        k: v / 3e5 for k, v in sorted(agg.items(), key=lambda x: -x[1])[:25]
                    },
                    "top_events": dump,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    r["model_revision"] = "vela-paper-1.0"
    (Path(out) / "summary.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    r = execute(a.out, a.limit)
    print(
        json.dumps(
            {
                k: r[k]
                for k in (
                    "model_revision",
                    "ms",
                    "operator_count",
                    "effective_mac",
                    "wire_bytes",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )
