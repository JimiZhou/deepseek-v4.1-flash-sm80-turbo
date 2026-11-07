import json
import os

def render_ascii_charts(json_path):
    if not os.path.exists(json_path):
        print(f"File {json_path} not found.")
        return
        
    with open(json_path) as f:
        data = json.load(f)
        
    print("\n" + "=" * 65)
    print("CONCURRENCY vs AGGREGATE THROUGHPUT (Tokens/s):")
    print("=" * 65)
    max_throughput = max(d["agg_throughput"] for d in data)
    for d in data:
        bar_len = int((d["agg_throughput"] / max_throughput) * 40)
        bar = "#" * bar_len
        print(f"C={d['concurrency']:<2} | {bar:<40} {d['agg_throughput']:>6.1f} tok/s")
        
    print("\n" + "=" * 65)
    print("CONCURRENCY vs INTER-TOKEN LATENCY P50 (ms):")
    print("=" * 65)
    max_itl = max(d["itl_p50"] for d in data)
    for d in data:
        bar_len = int((d["itl_p50"] / max_itl) * 40)
        bar = "=" * bar_len
        print(f"C={d['concurrency']:<2} | {bar:<40} {d['itl_p50']:>6.2f} ms")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    render_ascii_charts("/workspace/vllm/benchmarks/benchmark_results.json")
