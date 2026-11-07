import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
import numpy as np

URL = "http://10.0.0.1:40051/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"

def send_one_request(prompt, max_tokens, temperature):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.perf_counter()
    first_token_time = None
    token_times = []
    chunk_count = 0
    generated_text = ""
    
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                t_now = time.perf_counter()
                try:
                    chunk = json.loads(line[6:])
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        tok = (delta.get("content", "") or "") + (delta.get("reasoning", "") or "")
                        if tok:
                            if first_token_time is None:
                                first_token_time = t_now
                            token_times.append(t_now)
                            generated_text += tok
                            chunk_count += 1
                except Exception:
                    pass
                    
    t_end = time.perf_counter()
    total_time = t_end - t0
    ttft = (first_token_time - t0) if first_token_time else total_time
    
    itls = []
    for i in range(1, len(token_times)):
        itls.append((token_times[i] - token_times[i-1]) * 1000.0)
        
    num_out = max(len(token_times), 1)
    tpot = ((t_end - first_token_time) / max(num_out - 1, 1) * 1000.0) if (first_token_time and num_out > 1) else 0.0
    speed = (num_out / (t_end - first_token_time)) if (first_token_time and t_end > first_token_time) else 0.0
    
    return {
        "total_time": total_time,
        "ttft_ms": ttft * 1000.0,
        "tpot_ms": tpot,
        "itls_ms": itls,
        "num_out": num_out,
        "speed_tok_s": speed,
    }

def benchmark_concurrency(concurrency, prompt, max_tokens, temperature=0.0):
    print(f"\nEvaluating Concurrency C={concurrency} (max_tokens={max_tokens}, T={temperature})...")
    t_start = time.perf_counter()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_one_request, prompt, max_tokens, temperature) for _ in range(concurrency)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
    t_total = time.perf_counter() - t_start
    total_tokens = sum(r["num_out"] for r in results)
    agg_throughput = total_tokens / t_total
    
    ttfts = [r["ttft_ms"] for r in results]
    tpots = [r["tpot_ms"] for r in results]
    all_itls = []
    for r in results:
        all_itls.extend(r["itls_ms"])
        
    itl_p50 = statistics.median(all_itls) if all_itls else 0.0
    itl_p95 = np.percentile(all_itls, 95) if all_itls else 0.0
    itl_p99 = np.percentile(all_itls, 99) if all_itls else 0.0
    
    print(f"  Agg Throughput : {agg_throughput:.2f} tokens/s")
    print(f"  TTFT (ms)      : P50={statistics.median(ttfts):.1f}, P95={np.percentile(ttfts, 95):.1f}")
    print(f"  TPOT (ms)      : P50={statistics.median(tpots):.2f}, P95={np.percentile(tpots, 95):.2f}")
    print(f"  ITL (ms)       : P50={itl_p50:.2f}, P95={itl_p95:.2f}, P99={itl_p99:.2f}")
    
    return {
        "concurrency": concurrency,
        "total_tokens": total_tokens,
        "agg_throughput": agg_throughput,
        "ttft_p50": statistics.median(ttfts),
        "ttft_p95": float(np.percentile(ttfts, 95)),
        "tpot_p50": statistics.median(tpots),
        "tpot_p95": float(np.percentile(tpots, 95)),
        "itl_p50": itl_p50,
        "itl_p95": itl_p95,
        "itl_p99": itl_p99,
    }

def main():
    print("=" * 70)
    print("RUNNING BENCHMARK MATRIX ACROSS CONCURRENCIES ON 16x A800 CLUSTER")
    print("=" * 70)
    
    # Warmup
    print("Warming up engine...")
    send_one_request("Hello", 32, 0.0)
    
    prompt = "Write a comprehensive Python class for a multi-threaded LRU Cache with TTL support and O(1) eviction."
    concurrencies = [1, 2, 4, 8, 16]
    max_tokens = 64
    
    summary = []
    for c in concurrencies:
        res = benchmark_concurrency(c, prompt, max_tokens, temperature=0.0)
        summary.append(res)
        time.sleep(1)
        
    print("\n" + "=" * 80)
    print(f"{'Concurrency':>12} | {'Agg Throughput (tok/s)':>24} | {'TTFT P50 (ms)':>14} | {'TPOT P50 (ms)':>14} | {'ITL P50 (ms)':>12}")
    print("-" * 80)
    for s in summary:
        print(f"{s['concurrency']:>12} | {s['agg_throughput']:>24.2f} | {s['ttft_p50']:>14.1f} | {s['tpot_p50']:>14.2f} | {s['itl_p50']:>12.2f}")
    print("=" * 80)

    with open("/workspace/vllm/benchmarks/benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
