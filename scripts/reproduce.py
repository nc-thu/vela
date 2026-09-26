"""Run the paper presets sequentially, retaining each run's provenance."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from vela.cli import EXPECTED, PRESETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    results = {}
    for preset in PRESETS:
        target = args.out / preset
        command = [sys.executable, "-m", "vela", "--preset", preset, "--out", str(target)]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        print(f"Running {preset}", flush=True)
        subprocess.run(command, check=True)
        summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
        results[preset] = {
            key: summary[key] for key in ("cycles", "ms", "operator_count", "effective_mac")
        }
        if args.limit is None or args.limit == 2277:
            if summary["cycles"] != EXPECTED[preset]:
                raise RuntimeError(f"{preset}: reference cycle mismatch")
    (args.out / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
