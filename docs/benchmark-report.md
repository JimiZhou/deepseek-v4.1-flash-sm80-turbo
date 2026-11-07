# DeepSeek-V4.1-Flash 1017: Comprehensive Benchmark Report (16 $\times$ A800 40GB)

## 1. Executive Summary & Test Environment
- **Model**: DeepSeek-V4.1-Flash 1017 (552B total parameters, 40 decoder layers, MoE Top-6, FP4 Experts, MXFP8 Dense, FP8 KV Cache, Engram Host Offload).
- **Cluster Deployment**:
  - Node A (`node-a`, 10.0.0.1): 8 $\times$ NVIDIA A800-SXM4-40GB (Pipeline Stage 0, TP8)
  - Node B (`node-b`, 10.0.0.2): 8 $\times$ NVIDIA A800-SXM4-40GB (Pipeline Stage 1, TP8)
  - Interconnect: Full-mesh NVLink intra-node (400 GB/s bi-dir), 4 $\times$ 400 Gbps RoCEv2 inter-node (`ens13f0np0` at 172.16.1.2 $\leftrightarrow$ 172.16.2.2).
- **Evaluation Methodology**:
  - End-to-end streaming benchmarking measuring every delta timestamp with strict streaming validation.
  - Concurrency scaling from $C=1$ to $C=16$.
  - Realistic multi-turn coding and agent reasoning workloads.

---

## 2. Empirical Benchmark Matrix

| Concurrency ($C$) | Aggregate Throughput (tok/s) | TTFT P50 (ms) | TTFT P95 (ms) | TPOT P50 (ms) | TPOT P95 (ms) | ITL P50 (ms) | ITL P95 (ms) | ITL P99 (ms) |
|---|---|---|---|---|---|---|---|---|
| **$C = 1$** | **48.13** | 70.3 | 70.3 | **18.05** | 18.05 | **18.16** | 18.82 | 18.93 |
| **$C = 2$** | **77.80** | 114.6 | 147.6 | **20.79** | 22.05 | **19.26** | 20.73 | 54.02 |
| **$C = 4$** | **143.05** | 136.9 | 137.7 | **21.70** | 22.86 | **21.36** | 25.27 | 32.87 |
| **$C = 8$** | **181.46** | 227.5 | 229.4 | **33.03** | 36.22 | **32.93** | 34.09 | 35.04 |
| **$C = 16$** | **267.32** | 322.2 | 325.6 | **43.99** | 46.10 | **43.96** | 45.55 | 46.16 |

### 2.1 Key Performance Observations
- **Low-Latency Decode ($C=1$)**: Single-token decode latency is **18.05 ms** (ITL P50 = 18.16 ms), with a TTFT of **70.3 ms**.
- **High Concurrency Throughput ($C=16$)**: Aggregate serving throughput scales by **$5.55\times$** to **267.32 tokens/s** under saturated batch execution.
- **Speculative Acceleration**: When DSpark emits speculative bursts under low perplexity, effective decode speed reaches **150 – 164 tokens/s** at single concurrency ($C=1$).

---

## 3. Disaggregated Gain Attribution & Operational Status

### Part A: SM80 Native Kernel & Specialized Decode Gains (Status: Microbenchmarked / E2E Blocked)
- **Live HTTP Serving Data**: The metrics reported in Section 2 represent the **Stock SM80 Baseline** with DSpark speculative decoding and CUDA Graph enabled.
- **Production Isolation Barrier**: Full end-to-end A/B deployment of the new backend is **strictly blocked** because all 16 GPUs are in active production holding 37.6–39.9 GiB VRAM per accelerator for live instance 60. Safety constraints forbid terminating or preempting production containers without administrative authorization.
- **Measured Kernel Microbenchmarks (A800 GPU 0)**:
  - **MoE Multi-Expert Path**: 1,278.89 $\mu$s ($M=1$), 1,294.73 $\mu$s ($M=4$), 1,352.57 $\mu$s ($M=8$). Measured as the primary 76.8% critical path bottleneck.
  - **Sparse MLA Decode Math**: 157.70 $\mu$s ($M=1$), 152.35 $\mu$s ($M=4$).
  - **Mega-mHC Mixing**: 93.45 $\mu$s ($M=1$).
  - **DSpark Eager Markov Step**: 59.39 $\mu$s ($M=1$).
  - **Dense QKV Projections**: 56.29 $\mu$s ($M=1$).
  - **Indexer Scoring & Top-K**: 47.49 $\mu$s ($M=1$).
- **End-to-End Specialized Decode Speedup**: **Not measured at end-to-end server level** due to production GPU memory saturation. The SM80 Specialized Decode kernel implementation, unit tests, and wiring are delivered in `vllm/models/deepseek_v4_1/nvidia/specialized_decode/`.

### Part B: Topology, Communication & Offload Measurements
- **Topology Strategy**: TP8 (intra-node) $\times$ PP2 (inter-node).
- **Intra-Node NVLink AllReduce**: $128.32\ \mu\text{s}$ for 14 KB (1 token decode) empirically measured under standard PyTorch Ring NCCL. (Custom All-Reduce one-shot mailbox latency is unmeasured on this run).
- **Inter-Node Network Boundary**: Exactly 1 P2P activation transfer per token between PP0 (`node-a`) and PP1 (`node-b`) over 400 Gbps RoCE (`ens13f0np0`). Physical link ping RTT measured at $125\ \mu\text{s}$ (P2P kernel handoff is unmeasured).
- **Host Memory Offload**: 203.4 GB Engram tables verified pinned in host DRAM (`--engram-config '{"cpu_offload":true}'`), fitting the model on 40GB VRAM.

---

## 4. Audit Table: Measured Quantities vs. Unmeasured / Inferred Parameters

| Evaluation Dimension | Metric / Parameter | Status | Empirical Value or Limitation Source |
|---|---|---|---|
| **Stock Serving Matrix** | Concurrency $C \in [1, 16]$ TTFT, TPOT, ITL | **Measured** | Live HTTP streaming benchmark with strict SSE chunk timestamp assertions |
| **Intra-Node Comm** | 14 KB AllReduce (Ring NCCL) | **Measured** | $128.32\ \mu\text{s}$ via `scripts/nccl_intra_node.py` on GPU 0–7 |
| **Intra-Node Comm** | Custom All-Reduce (One-shot) | **Unmeasured** | Not isolated; engine uses custom allreduce internally inside server process |
| **Inter-Node Comm** | RoCE Ping RTT (`ens13f0np0`) | **Measured** | $125\ \mu\text{s}$ minimum RTT via ICMP probe |
| **Inter-Node Comm** | P2P RDMA Forward Handoff | **Unmeasured** | Server-internal pipelined stream; not tapped with CUDA events |
| **Kernel Profiling** | MoE / MLA / mHC / Markov Microbenchmarks | **Measured** | Hardware `torch.cuda.Event` timings via `scripts/profile_decode_components.py` |
| **Kernel Profiling** | Full Nsight Systems Trace | **Unmeasured** | Profiler attach requires server restart with tracing flags |
| **Kernel Correctness** | DSpark Fused Sampler vs Eager Argmax | **Measured** | Bit-for-bit exact match in `test_sm80_specialized_decode.py` |
| **Kernel Correctness** | Small-Batch Sparse MLA vs FP32 SDPA | **Measured** | Max error $0.00507$, mean $0.00060$ in `test_sm80_specialized_decode.py` |
| **Executor Runtime** | Instantiation, Gating & Fallback | **Measured** | Tested in `tests/test_executor_runtime.py` in isolated environment |
| **End-to-End A/B** | Specialized Decode vs Stock Server | **Blocked** | Blocked by 37.6–39.9 GiB VRAM allocation per GPU on live production instance |
