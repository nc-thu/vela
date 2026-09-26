# Vela

**An Efficient Open-Source FPGA Hardware Accelerator for End to End Vision-Language-Action Inference**

Vela combines group-wise INT8 quantization, two-product DSP packing, and a heterogeneous nonlinear unit for TurboVLA inference. This repository contains the paper, its figure assets, and the architecture simulator used for the W8A8 G64 performance evaluation.

[Paper](paper/main.pdf) · [Architecture](docs/architecture.md) · [Reproduction guide](docs/reproduction.md)

## Getting started

Requires Python 3.10 or newer and NumPy. No GPU, Vivado installation, model checkpoint, or LIBERO installation is needed to replay the supplied operator trace.

```sh
git clone https://github.com/nc-thu/vela.git
cd vela
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m vela --preset full --limit 110 --out runs/smoke
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.

For the complete policy invocation:

```sh
python -m vela --preset full --out runs/full
```

The expected result is **128,304,013 cycles**, or **427.680 ms at 300 MHz**, for 2,277 operators and 52,673,989,632 MACs. Full runs construct millions of scheduling events; use a machine with ample free RAM. Each invocation has a 540-second deadline, and output directories must be new.

## Paper experiments

| Preset | Configuration | Cycles | Latency at 300 MHz |
| :-- | :-- | --: | --: |
| `full` | W8A8 G64, complete design | 128,304,013 | 427.680 ms |
| `fp16` | Same-resource FP16 array reference | 176,801,157 | 589.337 ms |
| `h1` | Group-boundary stall and 24-lane merge | 135,663,376 | 452.211 ms |
| `h2` | One product per DSP | 167,795,501 | 559.318 ms |
| `h3` | Dedicated nonlinear datapaths, matched resources | 163,204,617 | 544.015 ms |

The complete design reduces latency by 27.4% relative to the FP16 reference. The other presets reproduce the mechanism comparisons in Fig. 11. Reference results are stored in [results/reference](results/reference); each full run checks its cycle count against them.

```sh
python scripts/reproduce.py --out runs/paper
python -m unittest discover -s tests -v
```

The simulator schedules data transfers and arithmetic events from a captured operator graph. Task success rates in the paper come from separate closed-loop policy evaluations; replaying this graph does not rerun robot episodes. See the [reproduction guide](docs/reproduction.md) for the supplied inputs and experiment boundaries.

## Repository layout

```text
src/vela/          Simulator, schedule construction, and workload trace
configs/           Paper preset definitions
results/reference/ Archived results for exact cycle comparisons
tests/             Scheduler and workload regression tests
scripts/           Experiment runner
docs/              Model details and reproduction instructions
paper/             IEEEtran source, bibliography, figures, and compiled paper
```

## Paper and citation

Cheng Nian, Jiaqi Zhang, Fasih Ud Din Farrukh, Jiaying Peng, Xiaorui Mo, Fei Chen, and Chun Zhang.
*Vela: An Efficient Open-Source FPGA Hardware Accelerator for End to End Vision-Language-Action Inference.*

Citation metadata is provided in [CITATION.cff](CITATION.cff). To build the paper:

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## License

The software is released under the [MIT license](LICENSE). Dependency and workload credits are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
