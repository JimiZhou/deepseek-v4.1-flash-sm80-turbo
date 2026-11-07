import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def run_worker(rank, world_size, sizes, iters=100):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank,
    )
    
    device = torch.device(f"cuda:{rank}")
    results = []
    
    for size in sizes:
        # size in bytes, convert to float32 or bfloat16 count
        num_elements = size // 2 # bf16 is 2 bytes
        t = torch.ones(num_elements, dtype=torch.bfloat16, device=device)
        
        # Warmup
        for _ in range(10):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        
        # Measure
        start = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iters
        
        # Calculate algbw and busbw for all_reduce
        # For ring/tree AllReduce: busbw = algbw * (2 * (n - 1) / n)
        algbw = (size / 1e9) / elapsed # GB/s
        busbw = algbw * (2 * (world_size - 1) / world_size)
        lat_us = elapsed * 1e6
        
        if rank == 0:
            results.append((size, lat_us, algbw, busbw))
            
    if rank == 0:
        print(f"{'Size (B)':>12} | {'Latency (us)':>14} | {'AlgBW (GB/s)':>14} | {'BusBW (GB/s)':>14}")
        print("-" * 62)
        for s, lat, alg, bus in results:
            print(f"{s:>12} | {lat:>14.2f} | {alg:>14.2f} | {bus:>14.2f}")
            
    dist.destroy_process_group()

def main():
    world_size = 8
    # Test typical decode sizes: 14KB (1 token), 28KB (2 tokens), 57KB (4 tokens), 114KB (8 tokens), 229KB (16 tokens), 1MB, 16MB
    sizes = [14336, 28672, 57344, 114688, 229376, 1048576, 16777216]
    mp.spawn(run_worker, args=(world_size, sizes), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
