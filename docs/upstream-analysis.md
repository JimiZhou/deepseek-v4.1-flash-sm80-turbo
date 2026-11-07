# DeepSeek-V4.1-Flash 1017: Upstream & Hopper Specialized Decode Analysis for Ampere SM80

## 1. Upstream Context & Lineage

The specialized decode architecture for DeepSeek-V4.1 represents an evolution across three generations of hardware targets in vLLM:

1. **Specialized Decode Base (PR #61868)**:
   - Target: NVIDIA GB300 (SM100/SM103) with Blackwell `tcgen05` Tensor Core instructions and hardware memory descriptors.
   - Core Concept: Replace the standard layer-by-layer eager PyTorch loop with a monolithic `DecodeStep` executor under FULL CUDA graph capture for small decode batches ($M \le 384$ rows).
   - Boundaries: Fused Lamport All-Reduce + Mega-mHC kernels between sublayers, eliminating intermediate memory roundtrips and separate synchronization points.

2. **DSpark Low-Latency Speculative Decoding (PR #61869)**:
   - Integrated the DSpark speculative drafting model into the specialized decode loop.
   - Introduced vocabulary-sharded Markov sampling (`sharded_markov.py`) and fused Markov kernels (`fused_sample.py`) to collapse the multi-step speculative draft loop into single-launch kernels.

3. **Hopper Native Decode (PR #64893)**:
   - Ported the specialized decode concepts to NVIDIA Hopper (SM90: H100 / H200 / GH200).
   - Replaced Blackwell `tcgen05` with SM90-native mechanisms: WGMMA (Warpgroup MMA), TMA (`cp.async.bulk.tensor`), asynchronous transaction barriers (`mbarrier.arrive.expect_tx`), Thread Block Clusters (`cluster_sync_all`), and Programmatic Dependent Launch (PDL).
   - Introduced small-batch sparse MLA (`hopper/mla_cute.py`, `hopper/mla_small.py`), fused FP8 indexer (`hopper/fused_indexer.py`), small-batch CuTe GEMMs (`hopper/dense_gemm_cute.py`), and persistent low-batch MoE (`hopper/moe_cute.py`).

4. **Community SM80 Efforts (Issue #57144, zenglanmu/vllm, Tokha233/a100-turbo)**:
   - Established that sparse-MLA attention math and FP8 KV-cache layout (`fp8_ds_mla`, 584 bytes/record) are functionally portable to SM80 via Triton and TileLang.
   - Revealed the primary SM80 barriers: absence of FP8/FP4 MMA instructions, absence of TMA/WGMMA/Clusters/PDL, and lack of DeepGEMM support.
   - On SM80 stock vLLM, execution currently falls back to Triton MLA sparse attention, Marlin W8A16 / Marlin FP4 MoE, and standard eager/piecewise CUDA graph capture, leaving significant kernel launch gaps, memory traffic, and multi-stream synchronization overhead.

---

## 2. Hopper Executor Architecture & Execution Workflow

### 2.1 Lifetime and Initialization
- **Construction**: During model loading, `DeepseekV4SpecializedDecodeModelState` checks environment capability and builds `HopperSpecializedDecode` via `SpecializedDecode.for_target(...)`.
- **Pre-Capture Warmup (`warm_up`)**: Prior to CUDA graph memory profiling, the executor inspects all graph capture sizes specified in `--compilation-config` (e.g. 1, 2, 4, 6, 12, 24, 48, 96, 192 rows) and triggers JIT compilation and memory allocation for all dense linears, MoE workspaces, fused indexer tables, and attention chains.
- **DSpark Co-Registration**: If DSpark speculative decoding is active, `HopperDraftDecode` is instantiated alongside the target executor, sharing the target's fused boundary buffers, mailbox, and mHC side stream.

### 2.2 Per-Step Dispatch (`SpecializedDecode.plan` & `DecodeStep`)
For every forward invocation:
1. **Gate Inspection**: Evaluates purely host-side state shared by all TP ranks (token count $M$, positions shape, pipeline stage, and active CUDA graph mode).
2. **Path Selection**:
   - If $M \le \text{max\_rows}$ (192 rows for Hopper, up to 32 DSpark requests) and current stream is in FULL CUDA graph capture/replay: Generates a specialized `DecodeStep`.
   - If $M > \text{max\_rows}$, or in Prefill phase, or in piecewise graph / eager fallback: Returns `None`, causing `DeepseekV4Model.forward` to execute the stock layer loop untouched.
3. **Execution Pipeline inside `DecodeStep.run`**:
   - **Engram Prefetch**: Dispatches Engram WKV projection asynchronously on a dedicated CUDA stream (`side_stream`).
   - **Layer Loop**: Iterates through `LayerPlan` instances.
     - Sublayer 1 (Attention): Executes `AttentionStep` (Norm $\to$ QKV Projections $\to$ Indexer $\to$ Sparse MLA $\to$ Output Linear $\to$ Mailbox Publish).
     - Boundary 1: Fused All-Reduce + mHC kernel consumes published outputs, performs cross-rank reduction, and computes mHC mixing for the feed-forward sublayer.
     - Sublayer 2 (MoE / Dense FFN): Executes `MoEStep` (Routing $\to$ Grouped Expert GEMM1 $\to$ Activation $\to$ Expert GEMM2 $\to$ Shared Expert). Finalize reduction is deferred to Boundary 2.
     - Boundary 2: Next sublayer boundary all-reduce collects MoE expert outputs and prepares residual for the subsequent decoder layer.

---

## 3. Hopper Custom Kernels: Shapes, Roles, and SM90 Dependencies

The following table details every custom kernel introduced in PR #64893:

| Kernel / Module | Mathematical Shape & Operation | Inputs & Outputs | SM90 (Hopper) Hardware Dependencies | Portability Assessment to SM80 |
|---|---|---|---|---|
| `DenseFp8Gemm` (`hopper/dense_gemm_cute.py`) | $Y[M, N] = X[M, K] \times \text{dequant}(W)^T$<br>$M \le 48$, Swap-AB tile | In: $X$ [M, K] BF16, $W$ [K/16, N*4] int32, Scales [K/32, N] UE8M0.<br>Out: $Y$ [M, N] BF16 | - `cp.async.bulk.tensor` (TMA)<br>- `mbarrier.arrive.expect_tx`<br>- `mma.sync.aligned.m16n8k16` (f16/bf16)<br>- Mailbox direct publish | **Incompatible as-is**.<br>Must replace TMA and mbarrier with SM80 `cp.async.ca` or Marlin / vectorized BF16 GEMV. |
| `LowBatchMoEKernel` (`hopper/moe_cute.py`, `moe_ptx.py`) | Persistent MoE:<br>Routing $\to$ MXFP4 GEMM1 $\to$ SwiGLU $\to$ MXFP4 GEMM2 $\to$ Shared Add.<br>$M \le 144$, Top-6 | In: $X$ [M, hidden], Router Logits [M, E].<br>Out: Top-k partial accumulators in global workspace. | - WGMMA (`wgmma.mma_async`)<br>- TMA 3D/4D descriptors<br>- Asynchronous cluster barriers<br>- PDL (`griddep_launch`/`wait`) | **Incompatible as-is**.<br>Requires redesigning persistent/grouped MoE kernel using SM80 BF16 MMA or optimized Marlin multi-expert dispatch. |
| `SmallDecode` & `CuPrep3` (`hopper/mla_cute.py`, `mla_small.py`) | Small-batch Sparse MLA Decode:<br>$M \le 12$ verify rows.<br>SWA + Top-2048 Compressed KV | In: $Q$ [M, H, 512], KV Cache [pages, 584B], Indexer Top-k.<br>Out: Context $O$ [M, H, 512] BF16. | - WGMMA FP16/BF16 tiles<br>- Programmatic Dependent Launch (PDL)<br>- CuTe DSL Hopper tensor layouts | **Incompatible as-is**.<br>SM80 must utilize optimized Triton Sparse MLA decode kernel with low-batch tile sizes ($B=1, 2, 4, 8$). |
| `FusedDecodeIndexer` (`hopper/fused_indexer.py`) | Logits computation against Index-K tiles + Candidate filtering.<br>$M \le 48$, Top-k=2048 | In: $Q_{\text{idx}}$ [M, 1, 64], Index-K [pages, 64] FP8.<br>Out: Logits [M, N_tiles] $\to$ Bitmask. | - FlashMLA FP8 Indexer layout<br>- CuTe logit reduction | **Rewrite needed**.<br>Can be cleanly implemented in Triton for SM80 using FP8 $\to$ FP32 dequant vectorization. |
| `AttnChain` (`hopper/attn_chain.py`, `attn_cute.py`) | Q/KV RMSNorm + KV Insert + Query RoPE + Output Combine Inv-RoPE | In: QKV hidden states.<br>Out: Normalized & RoPE-embedded states. | - Vectorized LDG/STG (`ldg4_cg`, `stg4`)<br>- CuTe register packing | **Highly Portable**.<br>Uses standard CUDA warp shuffle and math primitives; directly adaptable to SM80. |
| `AllReduceMHC` (`boundary/boundary.py`, `all_reduce.py`) | Fused Lamport One-Shot All-Reduce + Mega-mHC projection | In: Sublayer outputs [M, hidden], mHC weights.<br>Out: Reduced mixed state [M, hidden]. | - NVLink Multicast / Symmetric Memory<br>- Hopper fine-grained atomic memory | **Requires Fallback / Adaptation**.<br>Across PCIe / multi-node network, must use Custom All-Reduce / NCCL side-stream overlap. |
| `DSparkFusedMarkovSampler` (`dspark/fused_sample.py`) | Fused Base Logits Add + Markov Bias + Gumbel Noise Argmax + Embed Prep | In: `base_logits` [R, S, V], `bias` [R, V], `embed_weight` [V, D].<br>Out: `draft_tokens` [R, S], `markov_embed` [R, D]. | - Standard Triton (`tl.atomic_add`, `tl.load`, `tl.store`)<br>- No SM90 exclusive PTX | **Directly Compatible & Reusable on SM80**! |
| `DSparkShardedMarkovSampler` (`dspark/sharded_markov.py`, `sharded_markov_cute.py`) | Cooperative Vocabulary-Sharded Markov Sampling across TP ranks | In: Sharded vocab logits on each TP rank.<br>Out: Globally reduced winner token. | - CuTe PTX / SM80 `mma_16816`<br>- Inter-rank P2P synchronization | **Portable with SM80 adaptations**. |

---

## 4. Reusability vs. Rewrite Taxonomy for Ampere SM80

### 4.1 Reusable Runtime and Orchestration Logic (No Modification Needed)
1. **`SpecializedDecode` Base Class & Lifecycle**:
   - `DecodeStep` planning, per-step gating based on token count and CUDA graph mode.
   - Dual-executor architecture (Target model executor + DSpark draft model executor).
   - Side-stream scheduling infrastructure for overlapping asynchronous tasks (e.g. Engram prefetch and auxiliary norm calculations).
2. **CUDA Graph Capture & Replay Mechanisms**:
   - `post_forward_cudagraph.py` and Model Runner V2 post-forward capture for DSpark draft preparation.
   - Attention metadata recording patterns (`metadata_graphs.py`).
3. **DSpark Fused Sampling Pipeline**:
   - `DSparkFusedMarkovSampler` in Triton is completely arch-neutral and can be adopted immediately for SM80.

### 4.2 Kernels and Layers Requiring SM80 Redesign / Rewrite
1. **Dense Linears (`fused_wqa_wkv`, `wq_b`, `wo_a`, `wo_b`)**:
   - **Hopper Approach**: CuTe DSL FP8 GEMM with TMA load and WGMMA compute, or Humming W8A16 JIT.
   - **SM80 Reality**: No hardware TMA or FP8 MMA. Dense linears in the checkpoint use MXFP8 (`[32, 32]` blocks, UE8M0 scales).
   - **SM80 Strategy**:
     - Fast-path for small batches ($M \in [1, 16]$): Fused BF16 GEMV / vectorized Marlin W8A16.
     - For `wo_a`: Keep the verified offline dequantization to BF16 (`wo_a.mxfp8_dequant_to_bf16 = True`) and run through SM80 BF16 Tensor Core GEMM.
2. **MoE Layer (Small-Batch Fast-Path)**:
   - **Hopper Approach**: Persistent CTA launch (`moe_cute.py`) using TMA 3D loads and FP4/FP8 WGMMA.
   - **SM80 Strategy**:
     - Replace WGMMA with Marlin MXFP4 W4A16 / fused Triton grouped GEMM.
     - Fuse Top-K routing score calculation (softmax/sigmoid + bias) with token dispatch to reduce DRAM intermediate traffic.
3. **Sparse MLA Attention**:
   - **Hopper Approach**: Small-batch WGMMA decode (`mla_cute.py`) reading 584-byte `fp8_ds_mla` cache with PDL.
   - **SM80 Strategy**:
     - Specialize the existing `TRITON_MLA_SPARSE_DSV41` backend for decode ($M=1..16$), tailoring block shapes ($B_M=1$, $B_N=64$, $B_{HD}=128/512$), optimizing online softmax, and eliminating unneeded prefill metadata branches.
4. **Inter-Node Boundary All-Reduce**:
   - **Hopper Approach**: Fused one-shot Lamport all-reduce assuming intra-node NVLink multicast.
   - **SM80 Reality (2-Node Cluster)**: Multi-node boundary crosses RoCE / NIC network. Hardware NVLink multicast is strictly intra-node.
   - **SM80 Strategy**: Maintain intra-node custom all-reduce for TP8 ranks, and use pipelined NCCL / Gloo ring with side-stream overlapping for inter-node communication.

---

## 5. Upstream Dependencies, Potential Conflicts & Licensing

1. **Licensing**:
   - vLLM core, PR #61868, PR #61869, PR #64893: **Apache-2.0**.
   - Tokha233/deepseek-v4.1-flash-a100-turbo: **Apache-2.0**.
   - DeepSeek-V4.1 official reference: **MIT License**.
   - All proposed SM80 specialized decode components are fully compliant with Apache-2.0.
2. **Upstream Conflicts to Guard Against**:
   - **Namespace Divergence**: Upstream PR #64893 resides in `vllm/models/deepseek_v41/`, whereas our baseline container uses `vllm/models/deepseek_v4_1/`. The SM80 backend must be registered cleanly in `vllm/models/deepseek_v4_1/` without colliding with ROCm or generic paths.
   - **CUDA Graph Stream Capture Restrictions**: CuTe / CUDA kernels that allocate dynamic shared memory or call host synchronization during capture will trigger `cudaErrorStreamCaptureUnjoined`. All buffers must be pre-allocated and pinned during `warm_up`.
   - **Triton FP8 Convert Restrictions on SM80**: Native `tl.float8e4nv` conversions fail on SM80 Triton. Bit-manipulation or TileLang emulation kernels (`vllm.v1.attention.ops.fp8_sm80`) must be used for any FP8 cache packing.
