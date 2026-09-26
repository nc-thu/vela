"""Event-level equivalence and workload integrity checks."""
import importlib
import json
from pathlib import Path
import random
import unittest

from vela.common import Config
from vela.cli import EXPECTED, PRESETS
from vela.compiler.compile_schedule import build


class SchedulerTests(unittest.TestCase):
    def test_fast_matches_reference(self):
        for package in ("vela.engine", "vela.engine_fp16"):
            module = importlib.import_module(package + ".inherited_model")
            fast = importlib.import_module(package + ".fast_scheduler")
            for overlap in (True, False):
                with self.subTest(package=package, overlap=overlap):
                    reference = module.Scheduler(Config(overlap=overlap), 16)
                    accelerated = fast.make_scheduler(module.Scheduler)(Config(overlap=overlap), 16)
                    rng = random.Random(17)
                    for i in range(400):
                        deps = rng.sample(range(i), min(i, rng.randrange(4)))
                        resource = rng.choice(
                            [
                                "array",
                                "arena_read",
                                "arena_write",
                                "merge",
                                ["arena_read", "vector"],
                            ]
                        )
                        duration = rng.randrange(1, 65)
                        for scheduler in (reference, accelerated):
                            scheduler.add("test", duration, deps, resource, reason="regression")
                    self.assertEqual(reference.events, accelerated.events)
                    self.assertEqual(reference.summary(399), accelerated.summary(399))
                    with self.assertRaises(ValueError):
                        accelerated.add("invalid", -1)

    def test_parallel_and_shared_resources(self):
        from vela.engine.inherited_model import Scheduler

        scheduler = Scheduler(Config(), 16)
        a = scheduler.add("gemm", 20, resource="array")
        b = scheduler.add("merge", 7, resource="merge")
        c = scheduler.add("gemm", 5, resource="array")
        d = scheduler.add("restore", 3, [b, c], resource="vector")
        self.assertEqual([scheduler.end(i) for i in (a, b, c, d)], [20, 7, 25, 28])
        self.assertEqual(sum(scheduler.summary(d)["critical_cycles"].values()), 28)


class WorkloadTests(unittest.TestCase):
    def test_plan_and_capture_ids(self):
        import vela

        data = Path(vela.__file__).parent / "data"
        graph = json.loads((data / "capture.json").read_text(encoding="utf-8"))
        plan = build(32, 128, config="A")
        self.assertEqual(len(plan["commands"]), 2277)
        self.assertEqual(plan["effective_mac"], 52673989632)
        ids = {event["id"] for event in graph["events"]}
        self.assertEqual(len(ids), 2277)
        self.assertEqual({command["id"] for command in plan["commands"]}, ids)
        seen = set()
        for command in plan["commands"]:
            self.assertTrue(set(command["schedule_dependencies"]).issubset(seen))
            seen.add(command["id"])

    def test_preset_registry(self):
        self.assertEqual(set(PRESETS), set(EXPECTED))
        self.assertEqual(PRESETS["full"]["MERGE_W"], "96")
        self.assertEqual(PRESETS["h1"]["MERGE_W"], "24")
        self.assertEqual(PRESETS["h2"]["PACK1"], "1")
        self.assertEqual(PRESETS["h3"]["DEDICATED_SFU"], "1")

    def test_archived_results(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/paper.json").read_text())
        self.assertEqual(config["presets"], PRESETS)
        for preset, cycles in EXPECTED.items():
            result = json.loads((root / "results/reference" / (preset + ".json")).read_text())
            self.assertEqual(result["cycles"], cycles)
            self.assertEqual(sum(result["critical_cycles"].values()), cycles)
            self.assertEqual(result["operator_count"], 2277)
            self.assertEqual(result["effective_mac"], 52673989632)


if __name__ == "__main__":
    unittest.main()
