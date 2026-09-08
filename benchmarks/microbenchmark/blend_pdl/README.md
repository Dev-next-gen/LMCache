# Blend retrieve: programmatic dependent launch (PDL)

The blend retrieve's per-wave kernel chain (`re-RoPE -> scatter`, one true
dependency edge per group) launches the scatter with
`cudaLaunchAttributeProgrammaticStreamSerialization` on compute capability
>= 9.0; the scatter kernel runs its rope-independent preamble (slot-mapping
and pointer-table loads) and `cudaGridDependencySynchronize()`s before
reading the staged K. `LMCACHE_CB_PDL=0` disables.

## Measurements (H200, CUDA 13.0)

Standalone kernel chain (`pdl_bench.cu`, same grids/traffic as the real op):

| config | classic | PDL attribute-only | PDL + early trigger |
|---|---|---|---|
| llama-8B 16k, 4 waves | 2135 us | -0.2% | -0.4% |
| llama-8B 4k, 1 wave | 141 us | -1.2% | -1.6% |
| hybrid 3-group | 397 us | -2.0% | **+28% (regression)** |
| tiny launch-bound | 6.7 us | -12.5% | -4% |

Real op (`real_op_bench.py`, `execute_cb_retrieve_plan_flat`, llama-8B
16k-shaped plan including H2D staging): 39.19 ms -> 39.18 ms (**+0.01%**) —
the retrieve is H2D-bandwidth-bound; the kernel chain hides under the
staging copies.

## Conclusions

- Attribute-only PDL is uniformly safe and kept (it is free and helps the
  launch-bound tail); the early-trigger form is intentionally NOT used:
  large dependent grids occupy SMs spinning at the sync and regress
  multi-group plans by ~28%.
- End-to-end retrieve gains are negligible; the dominant cost is the
  staged H2D transfer. Meaningful next steps would target memory traffic
  (rope folded into the scatter's registers, staging-copy overlap depth),
  not launch latency.

Build the standalone bench:
    nvcc -O3 -arch=sm_90 -o pdl_bench pdl_bench.cu && ./pdl_bench
