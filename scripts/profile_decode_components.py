import os
import time
import torch
import torch.nn.functional as F

def benchmark_op(name, fn, warmup=25, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event) / iters
    print(f"{name:<45} : {elapsed_ms * 1000.0:>8.2f} us ({elapsed_ms:>6.3f} ms)")
    return elapsed_ms * 1000.0

def main():
    device = torch.device("cuda:0")
    print(f"Running on device: {torch.cuda.get_device_name(0)}")
    
    # Batch sizes to test: M = 1, 2, 4, 8, 16
    for M in [1, 4, 8]:
        print(f"\n{'='*25} BATCH SIZE M = {M} {'='*25}")
        
        # 1. DSpark Markov Sampling
        # Vocabulary size = 129280, Rank = 256
        vocab_size = 129280
        rank = 256
        base_logits = torch.randn(M, 4, vocab_size, dtype=torch.bfloat16, device=device)
        bias = torch.randn(M, vocab_size, dtype=torch.bfloat16, device=device)
        embed_weight = torch.randn(vocab_size, rank, dtype=torch.bfloat16, device=device)
        
        def eager_markov_step():
            # Standard sequential Markov sampling
            logits = base_logits[:, 0] + bias
            sampled = torch.argmax(logits, dim=-1)
            embed = F.embedding(sampled, embed_weight)
            return sampled, embed
            
        benchmark_op(f"DSpark Eager Markov Step (M={M})", eager_markov_step)
        
        # 2. Dense Projections (Attention QKV)
        # Hidden = 5120, q_lora = 1280, kv_lora = 512
        x = torch.randn(M, 5120, dtype=torch.bfloat16, device=device)
        w_q = torch.randn(5120, 1280, dtype=torch.bfloat16, device=device)
        w_kv = torch.randn(5120, 512, dtype=torch.bfloat16, device=device)
        w_o = torch.randn(1024, 5120, dtype=torch.bfloat16, device=device)
        
        def dense_qkv():
            q = torch.matmul(x, w_q)
            kv = torch.matmul(x, w_kv)
            o = torch.matmul(q[:, :1024], w_o)
            return o
            
        benchmark_op(f"Dense QKV Projections (M={M})", dense_qkv)
        
        # 3. Indexer Candidate Filtering & Top-k
        # Query: [M, 32, 128], Index-K: [1024 tiles, 128]
        # In decode, query matches against paged key cache
        num_tiles = 1024 # corresponding to candidate search space
        q_idx = torch.randn(M, 1, 128, dtype=torch.bfloat16, device=device)
        k_idx = torch.randn(num_tiles, 128, dtype=torch.bfloat16, device=device)
        
        def indexer_topk():
            # Logits calculation + top-256 blocks
            logits = torch.matmul(q_idx, k_idx.t()).squeeze(1) # [M, num_tiles]
            topk_vals, topk_indices = torch.topk(logits, k=256, dim=-1)
            return topk_indices
            
        benchmark_op(f"Indexer Candidate Scoring & Top-K (M={M})", indexer_topk)
        
        # 4. Sparse MLA Attention (Small Batch Decode)
        # Q: [M, 8 heads, 512], KV: 2048 selected tokens [2048, 512]
        H = 8 # per TP8 rank
        D = 512
        K_tokens = 512 # top-k tokens gathered
        q_mla = torch.randn(M, H, D, dtype=torch.bfloat16, device=device)
        k_mla = torch.randn(K_tokens, D, dtype=torch.bfloat16, device=device)
        v_mla = torch.randn(K_tokens, D, dtype=torch.bfloat16, device=device)
        
        def sparse_mla_decode():
            # Flash / SDPA / Triton gather
            # QK^T: [M, H, K_tokens]
            scores = torch.einsum("mhd,kd->mhk", q_mla, k_mla) * (1.0 / (D ** 0.5))
            attn = F.softmax(scores, dim=-1)
            out = torch.einsum("mhk,kd->mhd", attn, v_mla)
            return out
            
        benchmark_op(f"Sparse MLA Decode Math (M={M}, K={K_tokens})", sparse_mla_decode)
        
        # 5. MoE Routing & Expert GEMMs
        # 48 local experts per rank, Intermediate = 288 per rank
        # Top-6 experts per token
        router_logits = torch.randn(M, 384, dtype=torch.float32, device=device)
        gate_weights = torch.randn(48, 5120, 288, dtype=torch.bfloat16, device=device)
        up_weights = torch.randn(48, 5120, 288, dtype=torch.bfloat16, device=device)
        down_weights = torch.randn(48, 288, 5120, dtype=torch.bfloat16, device=device)
        
        def moe_eager():
            # Routing
            scores, topk_idx = torch.topk(router_logits, k=6, dim=-1)
            weights = F.softmax(scores, dim=-1)
            
            # Experts (simplified dispatch)
            out = torch.zeros(M, 5120, dtype=torch.bfloat16, device=device)
            for i in range(6):
                # Pick an active expert
                e_id = (topk_idx[:, i] % 48)[0].item()
                h1 = F.silu(torch.matmul(x, gate_weights[e_id])) * torch.matmul(x, up_weights[e_id])
                h2 = torch.matmul(h1, down_weights[e_id])
                out += h2 * weights[:, i : i + 1].to(torch.bfloat16)
            return out
            
        benchmark_op(f"MoE Eager Multi-Expert Path (M={M}, Top-6)", moe_eager)
        
        # 6. Mega-mHC Mixing
        # Hidden = 5120, mult = 4 (mixes current and 3 prior layers)
        mhc_weights = torch.randn(4, 5120, dtype=torch.bfloat16, device=device)
        history = [torch.randn(M, 5120, dtype=torch.bfloat16, device=device) for _ in range(4)]
        
        def mhc_mix():
            res = torch.zeros_like(x)
            for i in range(4):
                res += history[i] * mhc_weights[i]
            return res
            
        benchmark_op(f"Mega-mHC Mixing (M={M})", mhc_mix)

if __name__ == "__main__":
    main()
