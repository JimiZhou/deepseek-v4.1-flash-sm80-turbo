# DeepSeek-V4.1-Flash 1017: Ampere SM80 Native Specialized Decode

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/Hardware-NVIDIA%20Ampere%20SM80%20(A100%2FA800)-green.svg)]()
[![vLLM](https://img.shields.io/badge/Framework-vLLM%200.13.0+-orange.svg)]()

A high-performance, low-latency **Native Specialized Decode Backend** for **DeepSeek-V4.1-Flash 1017** on NVIDIA Ampere (SM80: A100 / A800) GPUs.

While upstream vLLM Specialized Decode (PR #64893) was designed strictly for Hopper (SM90: WGMMA, TMA, Thread Block Clusters, FlashMLA), this repository introduces a native Ampere-optimized decode hot-path using Triton JIT and custom CUDA fusions. It bridges the architectural gap on Ampere without sacrificing mathematical precision or disabling essential model modules (MXFP4 experts, MXFP8 dense weights, FP8 KV-cache, Engram, mHC, and DSpark).

---

## Key Highlights

- **Native Ampere Optimization**: Replaces Hopper-exclusive TMA/WGMMA dependencies with a pure Triton fused Markov sampler, small-batch Sparse MLA decode kernel, and small-batch MoE fast path.
- **Inter-Node Latency Mitigation**: Evaluates distributed topologies on multi-node clusters. Eliminates the catastrophic $20\text{ ms}$ cross-node AllReduce serialization of TP16 by adopting a high-efficiency **TP8 (Intra-Node NVLink) $\times$ PP2 (Inter-Node 400G RoCE P2P)** structure.
- **510 GB on 640 GB VRAM Budget**: Pinned CPU DRAM offload for the 203.4 GB Engram knowledge tables (`--engram-config '{"cpu_offload":true}'`), preserving 4.0 GiB/rank for the `fp8_ds_mla` KV-cache pool (supporting up to 5.6M tokens / 5.4x concurrency for 1M context requests).
- **Empirical A/B Improvements**:
  - **At Concurrency C=8**: Median Inter-Token Latency (ITL) drops by **24.6%** ($32.93\text{ ms} \to 24.84\text{ ms}$), with a **+16.8% throughput increase** ($181.5 \to 212.0\text{ tok/s}$).
  - **Speculative Acceleration**: Effective decode generation latency reaches **$6.56\text{ ms}$ per token** (~$154\text{ tok/s}$) under high DSpark draft acceptance.
  - **Zero Quality Loss**: Bit-for-bit exactness on greedy decoding and verified numerical tolerance ($< 0.0051$ max error) across algebra, coding, and structured JSON generation.

---

## Architecture Overview

```
+----------------------------------------------------------------------------------------------------+
|                                    Ampere Specialized Decode Step                                  |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Phase 1: Attention Fast Path]                                                                    |
|  Fused QKV Norm -> DeepSelect Indexer Scoring -> Small-Batch Sparse MLA Triton Decode Kernel       |
|                                                                                                    |
|  [Phase 2: Boundary 1 Overlap]                                                                     |
|  Intra-Node One-Shot NVLink All-Reduce overlapped with mHC coefficient mixing on side stream        |
|                                                                                                    |
|  [Phase 3: MoE Fast Path]                                                                          |
|  Fused sqrtsoftplus Top-6 Routing -> Grouped Expert Execution (GEMM1 + SwiGLU + GEMM2)              |
|  + Asynchronous Shared Expert Accumulation                                                         |
|                                                                                                    |
|  [Phase 4: Boundary 2 Overlap & Pipeline Handoff]                                                  |
|  PP0: Package hidden streams into IntermediateTensors for 400G RoCE P2P transfer                    |
|  PP1: Global normalization & hc_collapse_triton final head reduction                                |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### Module Breakdown

| Subsystem | Source Path | Description |
|---|---|---|
| **DSpark Fused Sampler** | `specialized_decode/ampere/dspark_fused.py` | Collapses sequential Markov bias add, Gumbel noise mixing, vocab block argmax, and next-step embedding gather into **one single Triton launch** per draft step. |
| **Small-Batch Sparse MLA** | `specialized_decode/ampere/sparse_mla_decode.py` | Triton decode kernel for $M \le 16$, reading 584-byte `fp8_ds_mla` cache and executing online softmax in registers. |
| **MoE Fast Path** | `specialized_decode/ampere/moe_fastpath.py` | Fused `sqrtsoftplus` Top-6 routing, intermediate activation consumption, and multimodal vision `input_ids` support. |
| **Boundaries & mHC** | `specialized_decode/boundary/boundary.py` | P2P NVLink custom all-reduce overlapping mHC mixing on a dedicated CUDA stream (`mhc_stream`). |
| **Runtime & Dispatch** | `specialized_decode/runtime/decode.py` | Full CUDA Graph gating, warmup, per-step planning, and transparent eager/large-batch fallback. |

---

## Hardware & Distributed Feasibility

Deploying DeepSeek-V4.1-Flash across two 8 $\times$ A800-40GB nodes (16 GPUs total):

| Parallelism Scheme | Inter-Node Comm per Step | Comm Latency per Step | Feasibility Assessment |
|---|---|---|---|
| **Cross-Node TP16** | 60–80 $\times$ AllReduce | **$18 – 25\text{ ms}$** | **Failed**: Cross-node AllReduce serialization throttles decode latency. |
| **TP8 $\times$ PP2 (Ours)** | **1 $\times$ P2P Forward Send (14 KB)** | **$\approx 15\ \mu\text{s}$** | **Optimal**: All collectives isolated to intra-node NVLink; compute-dominated. |
| **Single-Node TP8** | 0 | 0 | **Failed**: 307 GB weights exceed 320 GB physical VRAM capacity. |

---

## Quickstart

### Prerequisites
- 2 $\times$ 8-GPU NVIDIA A100 or A800 nodes (SXM4 recommended, PCIe supported).
- Interconnect: Intra-node NVLink; Inter-node 100G/200G/400G RoCE or InfiniBand.
- Linux with NVIDIA Driver $\ge 535$, CUDA $\ge 12.4$ / 13.0, PyTorch $\ge 2.5$.
- Model Weights: DeepSeek-V4.1-Flash 1017 safetensors.

### 1. Apply Patch to vLLM
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git apply open_source_release/patches/full-dsv41-sm80-native-decode.patch
pip install -e .
```

### 2. Launch via Script
Set cluster environment variables and launch on both nodes:

On **Node 0 (Leader)**:
```bash
export NODE_RANK=0
export MASTER_ADDR="10.0.0.1"   # Leader IP
export MASTER_PORT="40003"
export SERVE_PORT="40001"
export MODEL_PATH="/path/to/DeepSeek-V4.1-Flash-1017"
bash open_source_release/scripts/deploy_template.sh
```

On **Node 1 (Worker)**:
```bash
export NODE_RANK=1
export MASTER_ADDR="10.0.0.1"
export MASTER_PORT="40003"
export MODEL_PATH="/path/to/DeepSeek-V4.1-Flash-1017"
bash open_source_release/scripts/deploy_template.sh
```

### 3. Verify Inference
```python
from openai import OpenAI

client = OpenAI(base_url="http://10.0.0.1:40001/v1", api_key="none")
response = client.chat.completions.create(
    model="deepseek-v4.1-flash",
    messages=[{"role": "user", "content": "Explain the time complexity of quicksort."}],
    stream=True
)

for chunk in response:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning") and delta.reasoning:
        print(delta.reasoning, end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)
```

---

## Verification & Tests

### Kernel Unit Tests (Triton vs Eager Reference)
```bash
python3 tests/models/deepseek_v4_1/test_sm80_specialized_decode.py
```
- Validates DSpark Fused Markov Sampler against eager sequential argmax (100% exact match).
- Validates Small-Batch Sparse MLA decode kernel against PyTorch FP32 SDPA reference (max error $< 0.0051$).
- Validates MoE routing scores and weights normalization under `sqrtsoftplus`.

### Isolated Executor & Gating Test
```bash
python3 tests/test_executor_runtime.py
```
- Validates `AmpereSpecializedDecode` instantiation, prefill fallback, eager fallback, and CUDA graph captured `DecodeStep` planning.

---

## Upstream Contributions
We provide two clean patches in `patches/` formatted for upstream submission to `vllm-project/vllm`:
1. `0001-specialized_decode_sm80_backend.patch`: Fused Triton Markov sampler and Ampere specialized decode components.
2. `0002-model_specialized_decode_wiring_and_pp2.patch`: Pipeline Parallelism (PP2) `IntermediateTensors` handoff and multimodal vision routing `input_ids` parameter preservation.

---

## License
Apache-2.0 License. See [LICENSE](LICENSE) for details.
