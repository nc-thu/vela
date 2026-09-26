# Reproducing the paper results

## Inputs

The supplied workload describes one TurboVLA policy invocation:

- 2,277 captured operators;
- 52,673,989,632 effective MACs;
- a 12-by-7 output action chunk;
- the spatial task observation identified as task 0, state 23, with capture seed 1000.

`src/vela/data/capture.json` holds operator arguments, tensor metadata, and dependencies. `workload.json` is the initial selected schedule. `embedding_indices.json` supplies the small index lists needed to model embedding gathers. These files contain no model weights or observation images.

The compiler rebuilds the finite-buffer execution schedule from these inputs. The paper presets set all architecture switches explicitly in a fresh process, so shell variables left over from another experiment do not change a run.

## Run one experiment

```sh
python -m vela --preset full --out runs/full
python -m vela --preset h1 --out runs/h1
python -m vela --preset h2 --out runs/h2
python -m vela --preset h3 --out runs/h3
python -m vela --preset fp16 --out runs/fp16
```

Run them sequentially: full simulations retain millions of event records in memory. The default deadline is 540 seconds per experiment. A timeout returns exit code 124 and is not a valid result. For a quick installation check, add `--limit 110`. Partial runs are not comparable with the complete-invocation numbers.

`scripts/reproduce.py` runs the five presets sequentially and produces a compact `comparison.json`:

```sh
python scripts/reproduce.py --out runs/paper
```

For a short check of all five paths:

```sh
python scripts/reproduce.py --limit 110 --out runs/quick
```

Each output directory must be new. Existing results are never overwritten.

## Read the outputs

| File | Contents |
| :-- | :-- |
| `summary.json` | Invocation cycles, operator count, MACs, traffic, and critical-path categories |
| `run.json` | Selected preset, switches, Python version, elapsed host time, and return code |
| `run.log` | Simulator output and diagnostics |
| Other emitted files | Compiled schedule, operator records, allocation and validation details |

Latency in milliseconds is `cycles / 300000`. The full workload checks its exact cycle count, operator count, and MAC count before returning success. Critical-path category cycles must sum to the invocation total. Busy cycles describe resource occupancy and can overlap.

The fast scheduler preserves the event rules of the reference scheduler. Use `--scheduler reference` on a small workload to inspect that path. Unit tests compare events, dependencies, and resource waits between the two implementations.

## Preset definitions

- **full:** 16-by-48 physical matrix array; Pack2 exposes 96 logical columns. G64 quantization, two snapshot slots, 96 merge lanes, shared HNU datapaths, four local read ports, and a dedicated DMA port.
- **h1:** the complete snapshot mechanism is rolled back: group-boundary stalls and 24 merge lanes. This changes both latency and merge resources.
- **h2:** one product per DSP replaces Pack2. Other full-design choices remain.
- **h3:** resource-matched dedicated nonlinear datapaths replace the shared HNU datapath.
- **fp16:** a separate direct-FP16 engine models the same-resource array reference. This is not an INT8 run with quantization costs hidden.

The FP16 comparison measures the grouped quantization framework as a whole. H1, H2, and H3 are separate comparisons against the complete design.

## Evidence included

The reference summaries preserve the paper's W8A8 baseline and controlled ablations. They originate from the archived v21 ablation runs (`full`, `h1b`, `h2`, `h3ded`) and the v17 direct-FP16 run (`fp16_A_ports`). The W8 baseline matches the v18 paper schedule.

The repository reproduces architecture-model timing. Post-route module results and closed-loop task quality are reported in the paper; reproducing those requires their respective RTL and policy-evaluation environments, which are not part of this simulator package.
