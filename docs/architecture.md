# Architecture model

Vela evaluates a grouped W8A8 datapath with a heterogeneous nonlinear unit. The simulator converts an operator graph into integer-duration events and schedules them subject to dependencies, finite storage, and resource availability.

## Data path

```text
Activations -> group quantizer --+
                                v
Weights -----------------> Pack2 array -> group snapshots -> scale-aligned merge
                                                                |
                                                           FP16 restore
                                                                |
                                                     HNU / output encoding
                                                                |
                                                            next operator
```

H1 covers group quantization, scale handling, snapshots, and merging. H2 maps two signed W8A8 products with a shared activation to each matrix DSP. H3 organizes elementwise functions, reductions, and special functions into a shared nonlinear datapath.

The full paper preset uses G64, 768 matrix DSPs, and 96 logical output columns. Dynamic attention uses eight-bit operands on both sides. The published CLI exposes the W8A8 and FP16 experiments; W4A8 remains a numerical-quality comparison in the paper.

## Schedule construction

`compiler/previous_schedule.py` builds panels, operator dependencies, and finite-buffer descriptors. `group_schedule.py` adds group-stream handling. `compile_schedule.py` supplies nonlinear-unit parameters and applies the paper's data format.

The captured workload stores tensor shapes and operator arguments rather than tensor values. The simulator accounts for arithmetic latency and movement; it does not compute a new action prediction.

## Event execution

`engine/base_engine.py` implements transfers, arithmetic categories, storage, and summary construction. `inherited_model.py` adds finite-bank hazards and streaming. `model.py` implements the paper configuration, including FP16 intermediate domains and snapshot overlap. `overlap_options.py` applies the ablation switches. The FP16 reference uses its own engine under `engine_fp16/`.

Each event records its start and end cycles, data predecessors, occupied resources, resource wait, and traffic. A resource cannot accept another reservation until its previous event finishes. The model also tracks read/write hazards before reusing storage. Unknown operators raise an error.

Two schedulers are available. The reference implementation stores event completion times in the event dictionaries. The fast implementation keeps a compact completion-time array and caches repeated resource expansion and gather patterns. Both implement the same scheduling decisions.

## Time and resource accounting

All paper presets run at 300 MHz. External transfer service is modeled at 64 bytes per cycle. These are model inputs, not host-computer timings.

`critical_cycles` partitions the invocation's completion path into work categories. The partition sums to the total cycle count. `busy_cycles` records work performed by resources; simultaneous work can make its sum larger than invocation latency.

The paper reports resource and power estimates from separate post-route module runs. Simulator schedule parameters and implementation reports have distinct scopes; the event model does not synthesize an integrated FPGA design.
