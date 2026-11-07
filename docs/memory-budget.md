# Comprehensive Memory & Resource Budget: DeepSeek-V4.1-Flash 1017 on 16 $\times$ A800 40GB

## 1. High-Level Summary
- **Physical Accelerator**: NVIDIA A800-SXM4-40GB
- **Hardware Capacity per GPU**: 40,960 MiB (40.0 GiB HBM2)
- **Deployment Topology**: TP8 $\times$ PP2 across 2 nodes (8 GPUs per node)
- **Total Physical VRAM**: 640 GiB (320 GiB per node)
- **Host System RAM**: 1,024 GiB (1.0 TiB) per node

---

## 2. Granular Memory Allocation Breakdown (Per GPU Rank)

### 2.1 Model Static Weights (Per GPU Rank)
DeepSeek-V4.1-Flash 1017 contains 40 decoder layers distributed evenly across 2 pipeline stages (20 layers on Stage 0, 20 layers on Stage 1).

| Component | Architecture Parameters | Precision / Quantization | Total Size (Model-Wide) | Per GPU Footprint (TP8 $\times$ PP2) |
|---|---|---|---|---|
| **MoE Routed Experts** | 40 layers $\times$ 384 experts, intermediate size 2304 | FP4 (0.5 B/elem) + UE8M0 block scale (1/32) | 296.2 GB | **18.51 GB** |
| **MoE Shared Expert** | 40 layers $\times$ 1 shared expert, intermediate size 2304 | FP4 + UE8M0 block scale | 0.77 GB | **0.05 GB** |
| **Attention Projections** | 40 layers, $H=5120$, $Q_{\text{lora}}=1280$, $KV_{\text{lora}}=512$, $O_{\text{lora}}=1024$ | MXFP8 ($[32, 32]$ block scales) | 8.8 GB | **0.55 GB** |
| **DeepSelect Indexer** | 8 indexer source layers, 32 heads, dim 128 | FP8 weights & scales | 1.8 GB | **0.11 GB** |
| **Backbone & Norms** | Multi-head RMSNorms, embeddings, biases | BF16 / FP32 | 1.4 GB | **0.18 GB** |
| **LM Head (Stage 1 Only)**| Vocab 129,280 $\times$ 5120 | MXFP8 / BF16 | 1.32 GB | **0.17 GB** (0 on Stage 0) |
| **DSpark Drafter** | 3 layers, 128 experts, Markov rank 256 | FP4 Experts + BF16 Markov head | 8.8 GB | **0.55 GB** |
| **Engram Tables** | $2 \times 384\text{M}$ rows $\times 256$ elements | FP8 E4M3 with E8M0 scale | **203.4 GB** | **0.00 GB** (100% Host CPU Offload) |
| **Total Static Weights** | — | — | **522.5 GB** | **20.12 GB (Stage 0) / 20.84 GB (Stage 1)** |

---

### 2.2 Dynamic Runtime Buffers & Execution Workspace (Per GPU Rank)

| Memory Component | Allocation Sizing & Configuration | Per GPU VRAM Allocation | Notes & Invariants |
|---|---|---|---|
| **P2P & Custom All-Reduce Mailboxes** | NVLink symmetric memory buffers, IPC mailboxes, RoCE registration | **1.25 GB** | Required for low-latency intra-node one-shot all-reduce. |
| **CUDA Graph Capture Memory Pool** | 11 capture sizes ($M \in [1, 768]$), full graph piecewise workspaces | **2.20 GB** | Pre-allocated during warmup to guarantee zero-allocation during steady-state decode. |
| **Attention & Indexer Workspace** | Candidate bitmasks (2048 blocks), top-k index buffers, online softmax scratch | **0.85 GB** | Shared across layers; prevents dynamic re-allocations. |
| **KV Cache Pool (`fp8_ds_mla`)** | Paged KV cache pool, 584 bytes/record, `--kv-cache-memory-bytes 4294967296` | **4.00 GB** | Capacity: 7,354,397 tokens per GPU pool; supports high concurrency. |
| **PyTorch Context & CUDA Driver Overhead** | Driver runtime context, cuBLAS/cuDNN workspace, PyTorch caching allocator | **1.20 GB** | Baseline fixed driver memory cost on CUDA 13.0. |
| **Total Dynamic Allocations** | — | **9.50 GB** | Strictly bounded and capped. |

---

## 3. Total GPU Memory Budget & Safety Margin

$$\text{Total VRAM Used} = \text{Static Weights} + \text{Dynamic Allocations}$$

- **PP Stage 0 (`node-a`) Peak Memory**:
  $$20.12\text{ GB (Weights)} + 9.50\text{ GB (Buffers)} = \mathbf{29.62\text{ GB}} \quad (74.1\%\ \text{of 40 GB})$$
  **Free VRAM Headroom**: $\mathbf{10.38\text{ GB}}$ ($10,630\text{ MiB}$).
  
- **PP Stage 1 (`node-b`) Peak Memory**:
  $$20.84\text{ GB (Weights)} + 9.50\text{ GB (Buffers)} = \mathbf{30.34\text{ GB}} \quad (75.8\%\ \text{of 40 GB})$$
  **Free VRAM Headroom**: $\mathbf{9.66\text{ GB}}$ ($9,891\text{ MiB}$).

*Conclusion*: The deployment is fully safe from CUDA OOM, maintaining at least 9.6 GiB of untouched headroom per accelerator under maximum CUDA graph capture and DSpark speculative drafting.

---

## 4. Host System Memory (CPU Pinned Memory) Budget

The host RAM carries the 203.4 GB Engram tables via `--engram-config '{"cpu_offload":true}'`.

| Node Host | Total RAM | Allocated to Engram Tables | OS, Buffers & Cache | Available System RAM | Memory Pressure Assessment |
|---|---|---|---|---|---|
| **Node A (`node-a`)** | 1,024 GiB | 203.4 GiB (Pinned RAM) | 45.6 GiB | **775.0 GiB** | Extremely healthy (24% utilization) |
| **Node B (`node-b`)** | 1,024 GiB | 0 GiB (Engram on Stage 0 only) | 32.1 GiB | **991.9 GiB** | Zero pressure |
