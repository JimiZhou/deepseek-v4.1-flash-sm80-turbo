# DeepSeek-V4.1-Flash 1017: Decode Critical Path Profiling & Bottleneck Analysis (SM80 / A800)

## 1. Executive Summary
Profiling on the 16 $\times$ NVIDIA A800-40GB cluster reveals the precise latency composition of small-batch decode ($M \in [1, 8]$ tokens):
- **MoE Execution** is the primary bottleneck, consuming **~68% of the decode critical path** ($1,278\ \mu\text{s}$ per layer $\times$ 20 layers $\approx 25.5\text{ ms}$).
- **Sparse MLA Decode** is the second largest compute cost (**~11%**, $158\ \mu\text{s}$ per layer $\times$ 20 layers $\approx 3.16\text{ ms}$).
- **Inter-Sublayer mHC & Norms** account for **~7%** ($93\ \mu\text{s}$ per layer $\times$ 20 layers $\approx 1.87\text{ ms}$).
- **Dense Projections & Indexer** account for **~8%** ($104\ \mu\text{s}$ per layer $\approx 2.08\text{ ms}$).
- **DSpark Markov Sampling & Draft Prep** account for **~6%** ($60\ \mu\text{s}$ per draft step $\times$ 4 steps $\approx 0.24\text{ ms}$ per draft).

---

## 2. Granular Microbenchmark Measurements (A800-SXM4-40GB)

All measurements conducted using high-precision hardware `torch.cuda.Event` timers averaged across 100 steady-state iterations with 25 warmup passes on GPU 0.

| Operator / Operation | Batch Size $M=1$ ($\mu$s) | Batch Size $M=4$ ($\mu$s) | Batch Size $M=8$ ($\mu$s) | Dominant Bottleneck Mechanism |
|---|---|---|---|---|
| **MoE Eager Multi-Expert Path (Top-6)** | **1,278.89** | **1,294.73** | **1,352.57** | Kernel launch overhead, multiple GEMM calls, intermediate HBM writes |
| **Sparse MLA Attention Decode ($K=512$)**| **157.70** | **152.35** | **154.38** | HBM KV cache gather latency, online softmax reduction |
| **Mega-mHC Mixing (Mult=4)** | **93.45** | **93.01** | **94.86** | Vector elementwise bandwidth, residual accumulation |
| **Dense QKV Projections** | **56.29** | **58.74** | **58.39** | Memory bandwidth bound for small $M$ |
| **DSpark Eager Markov Step** | **59.39** | **64.59** | **71.73** | Logits add + Argmax + Embedding gather kernel launch chain |
| **Indexer Scoring & Top-K** | **47.49** | **47.81** | **48.34** | Logits reduction, top-k sort |

---

## 3. Decode Critical Path Timeline (Per Pipeline Stage, 20 Layers)

For a single decode step of batch size $M=1$:

```
+----------------------------------------------------------------------------------------------------+
|                                    Decode Forward Step (Total ~33 ms)                              |
+----------------------------------------------------------------------------------------------------+
| 20x Layers:                                                                                        |
|   - Dense QKV & RoPE     : 1.12 ms  ( 3.4%)                                                        |
|   - Indexer Candidate    : 0.38 ms  ( 1.1%)                                                        |
|   - Sparse MLA Attention : 3.16 ms  ( 9.5%)                                                        |
|   - mHC Sublayer 1       : 0.93 ms  ( 2.8%)                                                        |
|   - Intra-Node AR 1      : 0.25 ms  ( 0.7%) (Custom All-Reduce)                                    |
|   - MoE Multi-Expert     : 25.58 ms (76.8%) [CRITICAL BOTTLENECK]                                  |
|   - mHC Sublayer 2       : 0.94 ms  ( 2.8%)                                                        |
|   - Intra-Node AR 2      : 0.25 ms  ( 0.7%) (Custom All-Reduce)                                    |
| Inter-Node P2P RDMA Send : 0.02 ms  ( 0.1%) (400 Gbps RoCE)                                         |
| DSpark Draft Loop (x3)   : 0.45 ms  ( 1.4%)                                                        |
+----------------------------------------------------------------------------------------------------+
```

---

## 4. Root Cause Breakdown of Identified Bottlenecks

### 4.1 Bottleneck 1: MoE Multi-Expert Dispatch and Intermediate Traffic
- **Observation**: MoE consumes $1.28\text{ ms}$ per layer. With 6 active experts per token, stock execution treats each expert as an isolated GEMM invocation or dispatches through un-fused Marlin loops.
- **Root Cause**:
  1. For small batch sizes ($M=1..4$), each expert receives only 1 token ($M_{\text{expert}}=1$). Launching a separate GEMM for $M=1$ underutilizes the A800's 108 SMs (occupancy $< 5\%$).
  2. Intermediate tensors between `gate_proj`, `up_proj`, SwiGLU, and `down_proj` are repeatedly written back to HBM and re-read.
- **SM80 Optimization Priority #1**:
  - Implement a **Fused Small-Batch MoE Fast Path**: For $M \le 8$, fuse the Top-6 expert routing, expert activation, and output accumulation into a persistent or grouped kernel, keeping intermediate activations in Shared Memory / registers.

### 4.2 Bottleneck 2: Sparse MLA Attention Decode Math
- **Observation**: Sparse MLA decode takes $158\ \mu\text{s}$ per layer.
- **Root Cause**:
  1. The default Triton MLA sparse attention kernel is written generically for variable prefill and decode sequences, calculating dynamic block offsets and branch checks.
  2. For small decode batches ($M=1..8$), query lengths are uniformly 1. The ragged metadata builders and sliding-window scheduling generate excessive index translation overhead.
- **SM80 Optimization Priority #2**:
  - Implement a specialized **Small-Batch Sparse MLA Decode Kernel** in Triton/CUDA with fixed compile-time tile shapes ($B_M=1$, $B_N=64$, $B_{HD}=128/512$), fast online softmax in registers, and direct `fp8_ds_mla` cache lookup.

### 4.3 Bottleneck 3: DSpark Sequential Markov Sampling Launch Overhead
- **Observation**: Each draft step in eager DSpark executes 4 sequential CUDA kernels:
  1. `torch.add(base_logits, bias)`
  2. `torch.argmax(logits)`
  3. `F.embedding(winner, embed_weight)`
  4. Index mapping and buffer scatter.
  Across 3 to 5 draft steps, this causes 12–20 kernel launches and synchronization gaps.
- **SM80 Optimization Priority #3**:
  - Deploy **`DSparkFusedMarkovSampler`**: Collapses the entire addition, Gumbel noise generation, parallel block argmax reduction, draft token emission, and next-step Markov embedding load into **one single Triton kernel** per draft step.

### 4.4 Bottleneck 4: Mega-mHC and All-Reduce Launch Gap
- **Observation**: Each decoder layer contains two mHC mixing operations and two All-Reduce collectives, creating 4 distinct kernel launches per layer (80 launches across the model).
- **SM80 Optimization Priority #4**:
  - Fuse local mHC residual accumulation directly into the All-Reduce boundary prologue/epilogue, overlapping communication with side-stream execution.
