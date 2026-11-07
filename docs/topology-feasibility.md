# Deployment Topology Feasibility Analysis: DeepSeek-V4.1-Flash 1017 on 16 $\times$ A800 40GB

## 1. Topologies Evaluated

We evaluate four candidate distributed deployment topologies across our two nodes (Node A `node-a`, Node B `node-b`, each 8 $\times$ A800-SXM4-40GB):

1. **Topology A: Cross-Node Global Tensor Parallelism (TP16)**
2. **Topology B: Pipeline Parallelism + Intra-Node Tensor Parallelism (TP8 $\times$ PP2)**
3. **Topology C: Single-Node TP8 with Aggressive Host CPU Offload (EP + Engram Offload)**
4. **Topology D: Hybrid Parallelism with Inter-Node Expert Parallelism (TP8 $\times$ EP16)**

---

## 2. Quantitative Comparison & Analysis

| Metric / Dimension | Scheme A: TP16 | Scheme B: TP8 $\times$ PP2 (Recommended Main) | Scheme C: Single-Node TP8 + Offload | Scheme D: TP8 $\times$ EP16 |
|---|---|---|---|---|
| **Inter-Node Collectives per Decode Token** | **60 $\times$ AllReduce** (every Attention & MoE sublayer) | **1 $\times$ P2P Forward Activation Transfer** (from PP0 to PP1) | **0** (single node only) | **40 $\times$ AllToAll** (MoE dispatch & combine) |
| **Inter-Node Comm Volume per Token** | ~860 KB (60 $\times$ 14.3 KB) | **14.3 KB** (single hidden vector at boundary) | 0 | ~6.9 MB (MoE tokens $\times$ 6 experts $\times$ 40 layers) |
| **Inter-Node Comm Latency per Token** | **18 – 25 ms** (300 $\mu$s $\times$ 60 across RoCE) | **15 – 25 $\mu$s** (single P2P RDMA send) | 0 | **12 – 18 ms** (AllToAll contention across NIC) |
| **Intra-Node Comm Mechanism** | Cross-node mixed ring / hierarchical tree | **NVLink One-Shot Custom All-Reduce** (8–15 $\mu$s) | NVLink Custom All-Reduce | NVLink Custom All-Reduce |
| **VRAM Footprint per GPU** | ~33.5 GB (weights) + 4 GB KV | ~33.5 GB (half layers per rank) + 4 GB KV | **> 64 GB** (Exceeds 40 GB VRAM!) | ~33.5 GB + 4 GB KV |
| **Host CPU Offload Required** | Engram tables (203 GB) to Host RAM | Engram tables (203 GB) to Host RAM | Engram (203 GB) + >150 GB MoE weights | Engram tables (203 GB) to Host RAM |
| **Decode Critical Path (TPOT)** | Severely throttled by network RTT (>30 ms/tok) | **Fastest: dominated by GPU compute (~18–25 ms/tok)** | Heavily throttled by PCIe H2D weight streaming (>100 ms/tok) | Throttled by AllToAll network synchronization |
| **Speculative Decoding (DSpark)** | Feasible | **Natively supported on both PP stages** | Severely constrained | Complicated inter-rank draft synchronization |

---

## 3. Deep-Dive on Failure Modes & Tradeoffs

### 3.1 Why Cross-Node TP16 Fails Low-Latency Goals
In DeepSeek-V4.1-Flash (40 decoder layers):
- Attention sublayer output requires 1 AllReduce of shape $[M, 5120]$ (in BF16 = $M \times 10.24$ KB).
- MoE sublayer output requires 1 AllReduce of shape $[M, 5120]$.
- Vision / Embedding / Head boundaries add further synchronization points.
Total AllReduces per forward step: $40 \times 2 = 80$ (or ~60 active TP layers).
On a multi-node cluster, even over 400 Gbps RoCE with hardware GPUDirect RDMA, an AllReduce collective across 16 ranks on two separate host systems has a minimum latency floor of $200 – 350\ \mu\text{s}$ per invocation due to NIC serialization, queue-pair coordination, and barrier synchronization.
$$T_{\text{comm, TP16}} = 80 \times 250\ \mu\text{s} = 20,000\ \mu\text{s} = 20\ \text{ms per token!}$$
Even before running a single FLOP of GPU computation, TP16 introduces a 20 ms latency penalty per decode token. For small-batch decoding, where compute takes only 10–15 ms, network latency would account for over 60% of the total decode time.

### 3.2 Why Single-Node TP8 (Scheme C) Is Infeasible on 40GB Hardware
DeepSeek-V4.1-Flash weights:
- MoE experts (FP4): 296 GB
- Dense weights (MXFP8): ~10 GB
- Embeddings & backbone: ~1 GB
- Engram tables: 203 GB
Total raw checkpoint weights = 510 GB.
Even if the entire 203 GB Engram table is offloaded to host DRAM, the remaining model weights require $\mathbf{307\ GB}$.
An 8 $\times$ A800-40GB node has a physical VRAM limit of $8 \times 40\text{ GB} = 320\text{ GB}$.
Allocating 307 GB to weights leaves only $320 - 307 = 13\text{ GB}$ of VRAM across all 8 GPUs combined ($< 1.6\text{ GB}$ per GPU).
Once we account for PyTorch context, CUDA driver runtime buffers, NCCL communication buffers (1.5 GB), CUDA graph memory allocations (2 GB), and indexer workspaces, the GPU exhausts all memory before even allocating a single KV-cache block!
Attempting to dynamically offload MoE experts across PCIe Gen4 ($64\text{ GB/s}$ bi-directional) would require transferring dozens of gigabytes per token, increasing TPOT by more than $5\times$.

### 3.3 Why Pipeline Parallelism (TP8 $\times$ PP2) is the Optimal Main Topology
Under TP8 $\times$ PP2:
- **Node A (`node-a`)**: Hosts Pipeline Stage 0 (Layers 0 to 19, including Engram Layer 1 & 14).
  - All 8 GPUs execute intra-node TP8 via full-mesh 400 GB/s NVLink.
  - Zero inter-node communication during layers 0–19.
- **Inter-Node Boundary**:
  - Exactly **one** point-to-point RDMA send transmits the activation tensor $[M, 5120]$ from GPU 0 (Node A) to GPU 0 (Node B).
  - Transmission time over 400 Gbps RoCE: $\approx 15\ \mu\text{s}$.
- **Node B (`node-b`)**: Hosts Pipeline Stage 1 (Layers 20 to 39, plus LM Head and DSpark target layers 37–39).
  - All 8 GPUs execute intra-node TP8 via full-mesh 400 GB/s NVLink.
- **Return Path**: In greedy/sampling decode, the final token ID or logits scalar is returned to the API server on Node A via a lightweight TCP/socket packet ($< 100\ \mu\text{s}$).

---

## 4. Architectural Recommendation

1. **Primary Recommended Topology**:
   - **`TP8 × PP2` with Engram Host CPU Offload (`--engram-config '{"cpu_offload": true}'`)**.
   - Model distribution: PP Stage 0 on `node-a` (TP8), PP Stage 1 on `node-b` (TP8).
   - Intra-node communication: NVLink Custom All-Reduce.
   - Inter-node communication: 4-rail 400 Gbps RoCE P2P (`ens13f0np0`, `ens16f0np0`, `ens17f0np0`, `ens18f0np0`).
   - KV Cache: `fp8_ds_mla` with 4 GB allocation per rank.

2. **Fallback Topology**:
   - **`TP8 × PP2` without DSpark Speculative Decoding** (Pure Autoregressive Decode):
   - Used when speculative draft acceptance is low or under high concurrency ($C \ge 32$) where speculative batch verification overhead exceeds drafting gains.
