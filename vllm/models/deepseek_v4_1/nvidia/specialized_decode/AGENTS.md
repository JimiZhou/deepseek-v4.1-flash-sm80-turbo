# DeepSeek-V4.1 Specialized Decode for NVIDIA Ampere (SM80)

A specialized decode executor for small decode batches of DeepSeek-V4.1 on Ampere SM80 (A100 / A800), enabled with:
```bash
--additional-config '{"decode_backend": "latency_specialized"}'
```
or environment variable:
```bash
export VLLM_SM80_SPECIALIZED_DECODE=1
```

## Scope & Architectural Strategy

On NVIDIA Ampere (SM80), hardware architectures lack Hopper (SM90) features such as TMA (`cp.async.bulk.tensor`), WGMMA (`wgmma.mma_async`), Thread Block Clusters, and native narrow-precision FP8/FP4 Tensor Core MMA.

The SM80 Specialized Decode package provides native Ampere-optimized implementations that replace standard multi-launch execution with low-latency fused paths:

| Subsystem | SM80 Implementation & File | Optimization Objective |
|---|---|---|
| **DSpark Speculative Sampling** | `ampere/dspark_fused.py` | Collapses sequential Markov bias add, Gumbel noise mixing, vocab block argmax, and next-step embedding gather into 1 Triton launch per draft step. |
| **Small-Batch Sparse MLA** | `ampere/sparse_mla_decode.py` | Dedicated Triton decode kernel for $M \le 16$, reading 584-byte `fp8_ds_mla` cache and executing online softmax in registers. |
| **Small-Batch MoE Fast Path** | `ampere/moe_fastpath.py` | Fused `sqrtsoftplus` Top-6 routing with grouped expert execution and integrated shared expert addition. |
| **Boundary Collectives & mHC** | `boundary/boundary.py` | Overlaps intra-node custom all-reduce with mHC mixing on a dedicated CUDA stream (`mhc_stream`). |
| **Executor & Dispatch Runtime** | `runtime/decode.py`, `ampere/decode.py` | Gating, CUDA graph warmup, per-step planning, and transparent eager/large-batch fallback. |

## Fallback & Safety Guarantees

- **Prefill & Large Batches ($M > 192$)**: Automatically bypasses the specialized executor and executes standard stock layers.
- **Piecewise Graph & Eager Debugging**: Falls back automatically to unmodified stock layers without side effects.
- **Deterministic Correctness**: All floating-point operations maintain exact mathematical equivalence to reference paths within standard BF16 precision tolerances ($< 0.05$ max error).

## Verification & Testing

Unit tests for SM80 specialized decode components can be executed via:
```bash
python3 tests/models/deepseek_v4_1/test_sm80_specialized_decode.py
```
And model-level serving correctness via:
```bash
python3 tests/test_model_correctness.py
```
