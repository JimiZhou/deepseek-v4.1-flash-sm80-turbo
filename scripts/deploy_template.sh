#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Generic Parameterized Deployment Script for DeepSeek-V4.1-Flash on 16x A100/A800 (2x8 Nodes)
set -euo pipefail

# ================= Configuration Parameters =================
MODEL_PATH="${MODEL_PATH:-/data/models/DeepSeek-V4.1-Flash-1017}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"                          # 0 for Leader, 1 for Worker
MASTER_ADDR="${MASTER_ADDR:-10.0.0.1}"               # Leader Node IP
MASTER_PORT="${MASTER_PORT:-40003}"
SERVE_PORT="${SERVE_PORT:-40001}"
RDMA_IFACE="${RDMA_IFACE:-ens13f0np0}"              # High-speed RoCE/IB interface
KV_CACHE_BYTES="${KV_CACHE_BYTES:-4294967296}"       # 4.0 GiB per GPU
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"           # 1M Context Window
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"                     # DSpark draft tokens

# ================= RDMA & NCCL Environment =================
export NCCL_ALGO="Ring"
export NCCL_PROTO="Simple"
export NCCL_NET="IB"
export NCCL_IB_DISABLE="0"
export NCCL_CROSS_NIC="0"
export NCCL_IB_GID_INDEX="3"
export NCCL_DEBUG="INFO"
export NCCL_SOCKET_IFNAME="=${RDMA_IFACE}"
export GLOO_SOCKET_IFNAME="${RDMA_IFACE}"

# CPU & Threading Optimizations
export OMP_NUM_THREADS="8"
export MKL_NUM_THREADS="8"
export OPENBLAS_NUM_THREADS="1"
export OMP_WAIT_POLICY="PASSIVE"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

# Enable SM80 Native Specialized Decode
export VLLM_SM80_SPECIALIZED_DECODE="1"

echo "=========================================================="
echo "Launching DeepSeek-V4.1-Flash 1017 (SM80 Specialized Decode)"
echo "Node Rank     : ${NODE_RANK} / ${NNODES}"
echo "Master Address: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Model Path    : ${MODEL_PATH}"
echo "=========================================================="

vllm serve "${MODEL_PATH}" \
    --enable-prompt-tokens-details \
    --nnodes "${NNODES}" \
    --node-rank "${NODE_RANK}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --kv-cache-memory-bytes "${KV_CACHE_BYTES}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
    --long-prefill-token-threshold "${MAX_BATCHED_TOKENS}" \
    --kv-cache-dtype fp8_ds_mla \
    --attention-config '{"indexer_kv_dtype":"fp8"}' \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4, 6, 12, 24, 48, 96, 192, 384, 768], "max_cudagraph_capture_size": 768}' \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --hf-overrides '{"head_dtype":"float32","engram_layer_ids":[1,14]}' \
    --engram-config '{"cpu_offload":true}' \
    --distributed-executor-backend mp \
    --trust-remote-code \
    --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\"}" \
    --enable-expert-parallel \
    --additional-config '{"decode_backend":"latency_specialized"}' \
    --host "${MASTER_ADDR}" \
    --port "${SERVE_PORT}" \
    --served-model-name "deepseek-v4.1-flash"
