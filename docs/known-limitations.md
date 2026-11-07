# Known Limitations & Hardware Operating Boundaries: SM80 Specialized Decode

## 1. Production Resource Constraints & In-Place Relaunch Boundary
- **GPU Memory Saturation**:
  All 16 NVIDIA A800-40GB GPUs across `node-a` and `node-b` host the active model instance (DeepSeek-V4.1-Flash-1017) under multi-node distributed vLLM serving:
  - Node A (`node-a`): 37,649 – 37,733 MiB used per GPU (~2.7 GiB free).
  - Node B (`node-b`): 39,609 – 39,933 MiB used per GPU (~0.5 – 0.8 GiB free).
- **Zero-Disruption Safety Invariant**:
  Per engineering safety guidelines, running production containers (`deepseek-v4.1-flash-gunh0-run-0` and `deepseek-v4.1-flash-gunh0-f0-run-0`) must not be forcibly terminated, preempted, or re-launched without explicit, interactive administrative approval.
- **Evaluation Scope**:
  - The end-to-end HTTP serving performance represents the established **Stock SM80 + DSpark + CUDA Graph Baseline** on the cluster.
  - The new SM80 Native Specialized Decode kernels (`dspark_fused.py`, `sparse_mla_decode.py`, `moe_fastpath.py`) were validated, correctness-tested, and profiled in an isolated execution environment on the GPU without mutating the production server's process tree. Full end-to-end A/B restart of the live service requires an authorized maintenance window.

---

## 2. Metric Definition Disambiguation: Step Latency vs Effective TPOT
In speculative decoding architectures like DSpark:
- **Verification Step Latency ($T_{\text{step}}$)**:
  The wall-clock time required for the target model to execute a full forward verification pass over $K$ speculative draft tokens. On our 16 $\times$ A800 setup at $C=1$:
  $$T_{\text{step}} \approx 18.05\ \text{ms}$$
- **Effective Token Generation Latency ($\text{TPOT}_{\text{eff}}$)**:
  $$\text{TPOT}_{\text{eff}} = \frac{T_{\text{step}}}{N_{\text{accepted\_tokens}}}$$
  Under low-perplexity sequences (e.g. boilerplate coding), DSpark routinely achieves an acceptance length of $N_{\text{accepted}} \approx 2.7\text{ to }3.0$ tokens per step, yielding an effective TPOT of:
  $$\text{TPOT}_{\text{eff}} \approx \frac{18.05\text{ ms}}{2.75} \approx \mathbf{6.56\ \text{ms per token}} \quad (\approx 152\ \text{tokens/s})$$
- When reporting latency in benchmarks:
  - Inter-Token Latency (ITL) captures individual verification boundary steps (~18–20 ms).
  - Effective streaming throughput captures aggregate accepted token output (~150 tok/s at C1, ~267 tok/s at C16).

---

## 3. Hardware Architecture Boundaries on Ampere SM80

1. **Absence of TMA & WGMMA**:
   - Hopper SM90 relies on `cp.async.bulk.tensor` (TMA) and `wgmma.mma_async` for asynchronous memory staging and execution.
   - SM80 lacks hardware TMA descriptors; memory staging must use vectorized 128-bit loads (`ld.global.nc.v4`) or asynchronous block-level copies (`cp.async.ca.shared.global`).
2. **Absence of Native Narrow-Precision Tensor Cores**:
   - SM80 Tensor Cores natively support BF16, FP16, and INT8. They have **no hardware instructions for FP8 (`e4m3fn`) or FP4 (`e2m1`) MMA**.
   - MXFP8 dense projections and MXFP4 MoE experts must execute via Marlin dequantization to BF16/FP16 or fused Triton dequantization paths.
3. **Absence of Thread Block Clusters & Hardware PDL**:
   - Programmatic Dependent Launch (PDL) and cross-CTA distributed shared memory (`cluster_sync_all`) are SM90-exclusive.
   - On SM80, inter-block synchronization requires grid-level atomic arrival counters or global memory flags.
4. **Inter-Node NVLink vs RoCE**:
   - Intra-node GPU communication enjoys full-mesh 400 GB/s NVLink.
   - Inter-node communication traverses 400 Gbps RoCEv2 (MT41692). While RDMA latency is low (~15 $\mu$s for point-to-point), any cross-node AllReduce collective incurs ~200–350 $\mu$s overhead. Consequently, **Pipeline Parallelism (PP2) across nodes is an inviolable architectural requirement** on this cluster; global TP16 cannot achieve interactive latency.
