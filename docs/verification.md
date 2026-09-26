# Release verification

Checked on 2026-09-26 17:00:27 (Asia/Shanghai).

All five complete-workload runs matched the archived cycle counts and every
critical-path category. Each run covered 2,277 operators and 52,673,989,632
MACs. The machine used Windows and Python 3.11.5.

| Preset | Cycles | Host runtime |
| :-- | --: | --: |
| `full` | 128,304,013 | 172.1 s |
| `h1` | 135,663,376 | 180.3 s |
| `h2` | 167,795,501 | 203.1 s |
| `h3` | 163,204,617 | 248.0 s |
| `fp16` | 176,801,157 | 157.9 s |

These host runtimes describe the release check, not the simulated accelerator.
Each experiment completed within its 540-second deadline.
Exact results and flags are in [validation.json](../results/validation.json).

Additional checks:

- Five unit tests cover scheduler equivalence, resource serialization, workload dependencies, preset definitions, and archived results.
- All five presets pass a 110-operator smoke run.
- The reference and fast schedulers return 478,414 cycles for the W8A8 smoke workload.
- The built wheel installs into a separate environment and passes the same smoke check.
- GitHub Actions passed on Windows and Linux, using Python 3.10 and 3.12.
- The 12-page paper compiles without undefined references or overfull boxes. Its first page was rendered at 300 dpi and checked for author, affiliation, and link placement.
