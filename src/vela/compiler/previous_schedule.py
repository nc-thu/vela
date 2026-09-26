"""Timing-only block scheduling sideband; arithmetic quality not evaluated."""
import json, argparse, heapq, copy
from pathlib import Path

P = Path(__file__).resolve().parent
SOURCE = P.parent / "data/workload.json"


def build(lanes=32, port=128, latency=1, serial=False, raw_banks=0):
    assert lanes == 32, "SFU throughput fixed by user: 32 equivalent lanes"
    p = json.loads(SOURCE.read_text())
    p["version"] = 5
    bits = 3 if port == 64 else 4
    p["config"].update(
        sfu_lanes=lanes,
        arena_port_bytes=port,
        sfu_latency_factor=latency,
        serial=serial,
        block_bytes=1024,
        bank_xor_bits=bits,
        arena_banking=f"XOR of {bits}-bit word-address fields",
    )
    # 72-bit simple-dual-port BRAM36: ceil(port*8/72), rounded to
    # 8/16 banks for byte lanes. One write port and one read port per bank.
    feed_bram = 8
    dma_extra = 8 if port == 128 else 0
    p["resources"]["new_control_staging_BRAM36"] += dma_extra
    p["resources"]["staging_banks"]["two_4KiB_DMA_buffers_64B_port"] = 8 + dma_extra
    p["resources"].update(
        arena_port_bytes=port,
        arena_banks=port // 8,
        ctx_fetch_bytes=64,
        dma_cache_port_bytes=port,
        new_feed_BRAM36=feed_bram,
        sfu_scratch_BRAM36=16 if lanes else 0,
        total_BRAM36=p["resources"]["total_BRAM36"] + feed_bram + dma_extra + (16 if lanes else 0),
    )
    p["resources"]["total_URAM"] = (port // 8) * (
        (2 * 1024 * 1024 + port * 4096 - 1) // (port * 4096)
    )
    assert p["resources"]["total_BRAM36"] <= 249 and p["resources"]["total_URAM"] <= 76
    p["execution_contract"] = {
        "ready_queue": "oldest ready command; block operations overlap under resource reservations",
        "memory_dependency": "1024-byte conservative region RAW/WAR/WAW; whole-output scale barriers retained",
        "sfu_scratch_bytes": 65536 if lanes else 0,
        "queue_slots": 2,
        "queue_slot_bytes": 32768,
        "feed_buffer_bytes": 4096,
        "feed_to_array_bytes_per_cycle": 16,
        "forwarding": "temporary bind becomes storage alias when same shape and no live alias; arithmetic stages keep registers/scratch",
        "matrix_contexts": 1,
        "quantization_contexts": 1,
        "quality": "FP16 not numerically validated"
        if lanes
        else "unchanged arithmetic schedule; no new numerical replay",
    }
    p["sfu"] = {
        "lanes": lanes,
        "II": 1,
        "latency_factor": latency,
        "latencies": {
            "convert": 4,
            "add": 8,
            "mul": 6,
            "exp": 20,
            "reciprocal": 16,
            "rsqrt": 20,
            "sincos": 24,
            "reduce": 16,
        },
        "precision": "FP16 elementary functions; FP32 reduction/normalization; existing INT32+scale storage",
        "evidence": "pipeline depths are explicit modeling assumptions, not generated IP configuration",
        "sources": [
            "https://docs.amd.com/v/u/en-US/pg060-floating-point",
            "https://download.amd.com/docnav/documents/ip_attachments/floating-point.html",
        ],
    }
    # Storage hazards augment captured graph dependencies, including aliases.
    writer = {}
    readers = {}
    seen = set()
    for c in p["commands"]:
        d = set(c["dependencies"])
        rd = {p["tensor_storage"][str(t)]["storage"] for t in c["inputs"]}
        wr = (
            {p["tensor_storage"][str(t)]["storage"] for t in c["outputs"]}
            if not c["metadata_only"]
            else set()
        )
        for k in rd:
            d.update([writer[k]] if k in writer else [])
        for k in wr:
            d.update(readers.get(k, set()))
            d.update([writer[k]] if k in writer else [])
        d.discard(c["id"])
        c["schedule_dependencies"] = sorted(d)
        assert d <= seen, (c["id"], d - seen)
        for k in rd:
            readers.setdefault(k, set()).add(c["id"])
        for k in wr:
            writer[k] = c["id"]
            readers[k] = set()
        c["block_plan"] = {
            "bytes": 1024,
            "input_storages": sorted(rd),
            "output_storages": sorted(wr),
            "barrier": c["scale_barrier"],
            "location": "finite arena or charged DDR spill",
            "ready": "inputs and prior alias mutations complete",
        }
        seen.add(c["id"])
    # Explicit ready-queue ordering, stable oldest-ready selection.
    pending = {c["id"]: c for c in p["commands"]}
    done = set()
    ordered = []
    while pending:
        ready = [i for i, c in pending.items() if set(c["schedule_dependencies"]) <= done]
        assert ready, "cyclic graph"
        i = min(ready)
        ordered.append(pending.pop(i))
        done.add(i)
    p["commands"] = ordered
    if lanes:
        p["storage_formats"][
            "nonlinear_tables"
        ] = "removed; finite FP16 SFU scratch, no large function tables"
    assert raw_banks in (0, 8, 12)
    p["config"].update(raw_cache_bytes=raw_banks * 32768, raw_cache_port=max(1, raw_banks * 8))
    p["resources"]["total_URAM"] += raw_banks
    p["resources"]["retained_SFU_result_URAM"] = raw_banks
    p["execution_contract"][
        "input_row_slots"
    ] = "two if <=8 KiB per row, otherwise one; existing 16 KiB scratch"
    p["execution_contract"][
        "raw_result_cache"
    ] = "64 bits per value; entire tensor must fit; first-pass reuse, final scale barrier retained"
    p["execution_contract"][
        "merge_encoder_overlap"
    ] = "merge waits for its own result bank only; encoder is a shared reserved resource"
    assert p["resources"]["total_URAM"] <= 76
    return p


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--lanes", type=int, default=32)
    a.add_argument("--port", type=int, default=128)
    a.add_argument("--out", required=True)
    a.add_argument("--raw-banks", type=int, default=0)
    x = a.parse_args()
    Path(x.out).write_text(json.dumps(build(x.lanes, x.port, raw_banks=x.raw_banks), indent=2))
