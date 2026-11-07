# DeepSeek-V4.1-Flash 1017: Ampere SM80 Native Specialized Decode Architecture

## 1. Architectural Mission
The goal of the **SM80 Native Specialized Decode Backend** is to provide an optimized, low-latency, small-batch decode execution path for DeepSeek-V4.1-Flash (1017) on NVIDIA Ampere A800/A100 (SM80) GPUs, reducing kernel launch gaps, memory roundtrips, and synchronization overhead without altering mathematical semantics or precision.

---

## 2. Design Principles & Architecture

### 2.1 The Small-Batch Decode Dilemma on SM80
In standard vLLM execution on Ampere SM80:
- Decode steps ($M \in [1, 16]$ tokens) execute the exact same layer loop as large prefill batches.
- Each of the 40 decoder layers invokes:
  1. RMSNorm (2 kernels)
  2. Attention QKV Projections (2 GEMMs)
  3. Indexer Scoring & Top-K (2 kernels)
  4. Sparse MLA Attention (multiple Triton launches + metadata builders)
  5. Attention Output Linear & AllReduce (2 kernels)
  6. MoE Routing Score & Top-6 Selection (2 kernels)
  7. MoE 6 Routed Experts (6 $\times$ GEMM1 + 6 $\times$ SwiGLU + 6 $\times$ GEMM2 = 18 operations)
  8. Shared Expert (2 GEMMs + 1 Activation)
  9. MoE Finalize & AllReduce (2 kernels)
  10. Mega-mHC Mixing (2 kernels)
- **Total per layer**: >30 separate kernel launches! Across 40 layers = **>1,200 kernel launches per forward step**!
- Even under CUDA graph capture, the device launch gap and memory barriers between tiny operations consume significant SM cycles while GPUs operate at $<10\%$ ALU occupancy.

### 2.2 The SM80 Native Specialized Decode Solution
`AmpereSpecializedDecode` restructures the small-batch decode execution graph into four fused phases per layer:

```
+----------------------------------------------------------------------------------------------------+
|                                    Ampere Specialized Decode Step                                  |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  Phase 1: Attention Fast Path                                                                      |
|  [Fused QKV Norm & Projections] -> [Fused Indexer Top-K] -> [Small-Batch Sparse MLA Decode Kernel] |
|                                                                                                    |
|  Phase 2: Boundary 1 (All-Reduce & mHC Overlap)                                                    |
|  [Intra-Node Custom All-Reduce] (overlapped with mHC coefficient calculation on side stream)       |
|                                                                                                    |
|  Phase 3: MoE Fast Path                                                                            |
|  [Fused Top-6 Routing + Grouped Expert Kernel (GEMM1 + SwiGLU + GEMM2)] + [Shared Expert]          |
|                                                                                                    |
|  Phase 4: Boundary 2 (MoE Finalize & All-Reduce Overlap)                                           |
|  [MoE Finalize Reduction fused into All-Reduce boundary]                                           |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. Core Component Modules

### 3.1 DSpark Fused Markov Sampling (`ampere/dspark_fused.py`)
- **Problem**: Sequential drafting in DSpark executes:
  `Base Logits Add -> Argmax -> Gumbel Noise -> Embedding Gather -> Buffer Scatter`
  repeated 3 to 5 times sequentially, with host-device launch stalls between positions.
- **SM80 Solution**:
  - `DSparkFusedMarkovSampler`: A high-performance Triton JIT kernel that combines:
    1. Base logit addition and Markov bias addition in FP32;
    2. Parallel vocabulary block-wise reduction ($B=1024$ threads);
    3. Gumbel noise mixing and argmax resolution with deterministic tie-breaking;
    4. Atomic grid arrival counter that elects a single CTA to perform final inter-block reduction;
    5. Direct write of the sampled draft token and in-place lookup of the next step's Markov embedding row from `embed_weight`.
  - **Launch reduction**: Collapses 15–25 separate kernel launches into **3–5 single launches** total for the draft phase.

### 3.2 Small-Batch Sparse MLA Decode Kernel (`ampere/sparse_mla_decode.py`)
- **Problem**: Stock sparse MLA uses FlashMLA on SM90+ (which does not compile for SM80) and falls back to a generic Triton sparse attention kernel that includes dynamic SWA chunk planning, variable prefill branches, and ragged metadata indexing.
- **SM80 Solution**:
  - Specialized Triton decode kernel tailored specifically for query length $Q=1$ and batch sizes $M \in [1, 16]$.
  - Block tile sizing: $B_M = 1$, $B_N = 64$, $B_{HD} = 128 / 512$.
  - Fixed-layout paged memory gather from DeepSeek's 584-byte `fp8_ds_mla` cache records.
  - Fast register-level online softmax reduction without global memory scratchpad.

### 3.3 Small-Batch MoE Fast Path (`ampere/moe_fastpath.py`)
- **Problem**: For batch size $M=1..8$, running 6 separate expert GEMMs causes severe launch overhead and memory bandwidth saturation from temporary activation tensors.
- **SM80 Solution**:
  - For $M \le 8$: Grouped expert execution that bundles the active experts for the batch.
  - Intermediate SwiGLU activations ($X_{\text{gate}} \times \text{SiLU}(X_{\text{up}})$) are computed and consumed immediately without writing back full intermediate buffers to HBM.
  - Shared expert forward pass is computed in parallel and added directly to the final accumulated expert state.

### 3.4 Boundary All-Reduce & mHC Overlap (`boundary/boundary.py`)
- **Problem**: Layer boundaries execute separate All-Reduce, RMSNorm, and mHC mixing kernels.
- **SM80 Solution**:
  - Leverages vLLM's `CUSTOM` one-shot NVLink All-Reduce for TP8 ranks within the node.
  - Overlaps the next layer's mHC coefficient calculations on a dedicated CUDA side stream (`mhc_stream`).
  - Inter-node pipeline transfer (PP0 $\to$ PP1) is issued asynchronously across the 400 Gbps RoCE RDMA link.

---

## 4. Fallback & Gating Guarantees

1. **Safety Invariants**:
   - The specialized decode path is strictly enabled only when:
     - Architecture is SM80 (`major == 8, minor == 0`);
     - Forward pass is in pure decode ($M \le \text{max\_rows}$, where $\text{max\_rows} = 192$);
     - Stream is in FULL CUDA Graph mode (`torch.cuda.is_current_stream_capturing()`);
     - Tensor parallelism matches configured topology (TP8).
2. **Deterministic Fallback**:
   - Any prefill step, oversized decode step ($M > 192$), piecewise graph, or eager debugging invocation returns `None` from `decode.plan(...)`, cleanly executing the unmodified stock layer path.
   - Zero side-effects or persistent state corruption on fallback.
