"""Command-line entry point; each simulation runs in a clean child process."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PRESETS = {
    "full": dict(SNAPSHOT="1", FP16DOMAIN="1", MERGE_W="96"),
    "h1": dict(SNAPSHOT="0", FP16DOMAIN="1", MERGE_W="24"),
    "h2": dict(SNAPSHOT="1", FP16DOMAIN="1", MERGE_W="96", PACK1="1"),
    "h3": dict(SNAPSHOT="1", FP16DOMAIN="1", MERGE_W="96", DEDICATED_SFU="1"),
    "fp16": dict(FP16_DIRECT="1", MERGE_W="24"),
}
EXPECTED = {"full": 128304013, "h1": 135663376, "h2": 167795501, "h3": 163204617, "fp16": 176801157}
FLAGS = {
    "SNAPSHOT",
    "FP16DOMAIN",
    "MERGE_W",
    "PACK1",
    "FP16_DIRECT",
    "DEDICATED_SFU",
    "HNU_MUTEX",
    "FP16_SFU",
    "PWL_CHAIN",
    "DUMP_CP",
    "W4A8_CONFIG",
    "RAW_BANKS",
    "SIM_FAST",
    "READ_WIDTH",
    "DMA_DEDICATED",
    "READ_PORTS",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=PRESETS, default="full")
    p.add_argument("--out", type=Path, required=True, help="new output directory")
    p.add_argument("--limit", type=int, help="run only the first N operators")
    p.add_argument("--scheduler", choices=("fast", "reference"), default="fast")
    p.add_argument("--timeout", type=int, default=540, help="child deadline, at most 540 seconds")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.limit is not None and not 1 <= a.limit <= 2277:
        p.error("--limit must be between 1 and 2277")
    if not 1 <= a.timeout <= 540:
        p.error("--timeout must be between 1 and 540")
    out = a.out.resolve()
    if not a.worker:
        out.mkdir(parents=True, exist_ok=False)
        env = {k: v for k, v in os.environ.items() if k not in FLAGS}
        env.update(
            W4A8_CONFIG="A", RAW_BANKS="0", READ_WIDTH="64", DMA_DEDICATED="1", READ_PORTS="4"
        )
        env.update(PRESETS[a.preset])
        cmd = [
            sys.executable,
            "-m",
            "vela",
            "--preset",
            a.preset,
            "--out",
            str(out),
            "--scheduler",
            a.scheduler,
            "--worker",
        ]
        if a.limit is not None:
            cmd += ["--limit", str(a.limit)]
        started = time.monotonic()
        with (out / "run.log").open("w", encoding="utf-8") as log:
            try:
                code = subprocess.run(
                    cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout
                ).returncode
            except subprocess.TimeoutExpired:
                code = 124
        record = dict(
            preset=a.preset,
            limit=a.limit,
            scheduler=a.scheduler,
            wall_seconds=time.monotonic() - started,
            returncode=code,
            timeout_seconds=a.timeout,
            python=sys.version,
            flags={k: env[k] for k in sorted(FLAGS) if k in env},
        )
        (out / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        if code:
            print(f"Simulation failed (exit {code}); see {out / 'run.log'}", file=sys.stderr)
        else:
            s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            print(
                json.dumps(
                    {k: s[k] for k in ("cycles", "ms", "operator_count", "effective_mac")}, indent=2
                )
            )
        return code
    package = "vela.engine_fp16" if a.preset == "fp16" else "vela.engine"
    model = importlib.import_module(package + ".model")
    if a.scheduler == "fast":
        importlib.import_module(package + ".fast_scheduler").install_engine(model.m)
    importlib.import_module(package + ".overlap_options").install(model.m)
    result = model.execute(out, a.limit)
    result["preset"] = a.preset
    if a.limit is None or a.limit == 2277:
        assert result["cycles"] == EXPECTED[a.preset], (
            a.preset,
            result["cycles"],
            EXPECTED[a.preset],
        )
        assert result["operator_count"] == 2277
        assert result["effective_mac"] == 52673989632
    assert sum(result["critical_cycles"].values()) == result["cycles"]
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0
