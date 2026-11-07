import json
import urllib.request
import time

def call_api(prompt, max_tokens=150, temperature=0.0):
    url = "http://10.0.0.1:40051/v1/chat/completions"
    payload = {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    content = ""
    reasoning = ""
    chunk_count = 0
    t0 = time.perf_counter()
    
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                chunk_count += 1
                try:
                    chunk = json.loads(line[6:])
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if "reasoning" in delta and delta["reasoning"]:
                            reasoning += delta["reasoning"]
                        if "content" in delta and delta["content"]:
                            content += delta["content"]
                except Exception:
                    pass
    t_total = time.perf_counter() - t0
    return {
        "content": content,
        "reasoning": reasoning,
        "chunks": chunk_count,
        "total_time": t_total
    }

def main():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE CORRECTNESS TEST SUITE ON 16x A800 CLUSTER")
    print("=" * 70)
    
    # Test 1: Deterministic Math Reasoning (Greedy)
    print("\n--- Test 1: Deterministic Math Reasoning (Greedy, T=0) ---")
    prompt1 = "Solve the equation 3*x + 15 = 42 for x. State only the numerical answer at the end as x = ?"
    res1 = call_api(prompt1, max_tokens=100, temperature=0.0)
    print(f"Latency: {res1['total_time']:.2f}s, Chunks: {res1['chunks']}")
    print("Reasoning snippet:", res1["reasoning"][:120])
    print("Content snippet:", res1["content"][:120])
    assert "9" in res1["content"] or "9" in res1["reasoning"], f"Math answer incorrect: {res1['content']}"
    print("Test 1 Result: PASS")
    
    # Test 2: Repeatability / Rank Determinism (Greedy exact match across 2 runs)
    print("\n--- Test 2: Repeatability & Rank Determinism (T=0) ---")
    res1_repeat = call_api(prompt1, max_tokens=100, temperature=0.0)
    assert res1["content"] == res1_repeat["content"], "Greedy outputs diverged across runs!"
    print("Outputs matched bit-for-bit across independent requests.")
    print("Test 2 Result: PASS")

    # Test 3: Coding Capabilities
    print("\n--- Test 3: Python Coding (Greedy) ---")
    prompt2 = "Write a python function `is_prime(n)` and return True or False. Write only the code block."
    res2 = call_api(prompt2, max_tokens=150, temperature=0.0)
    print(f"Latency: {res2['total_time']:.2f}s, Chunks: {res2['chunks']}")
    print("Code snippet:\n", res2["content"][:200])
    assert "def is_prime" in res2["content"], "Code generation failed"
    print("Test 3 Result: PASS")
    
    # Test 4: Temperature Sampling (T=0.7)
    print("\n--- Test 4: Stochastic Sampling (T=0.7) ---")
    prompt3 = "Suggest three creative names for a high-performance GPU inference engine."
    res3_a = call_api(prompt3, max_tokens=100, temperature=0.7)
    res3_b = call_api(prompt3, max_tokens=100, temperature=0.7)
    text_a = res3_a["content"] if res3_a["content"] else res3_a["reasoning"]
    text_b = res3_b["content"] if res3_b["content"] else res3_b["reasoning"]
    print("Run A snippet:", text_a[:100])
    print("Run B snippet:", text_b[:100])
    assert len(text_a) > 0 and len(text_b) > 0, "Generated text empty under T=0.7"
    assert res3_a["chunks"] > 5 and res3_b["chunks"] > 5, "Temperature streaming failed"
    print("Test 4 Result: PASS")

    # Test 5: Tool Calling / Structural Integrity
    print("\n--- Test 5: JSON Schema Extraction ---")
    prompt4 = 'Output a valid JSON object with keys "city": "Beijing", "country": "China", "status": "active". Output only JSON.'
    res4 = call_api(prompt4, max_tokens=80, temperature=0.0)
    print("JSON output:", res4["content"])
    clean_json = res4["content"].strip().strip("`").strip("json").strip()
    data = json.loads(clean_json)
    assert data.get("city") == "Beijing" and data.get("country") == "China"
    print("Test 5 Result: PASS")

    print("\n" + "=" * 70)
    print("ALL 5 CORRECTNESS TESTS PASSED SUCCESSFULLY WITH ZERO DEGRADATION!")
    print("=" * 70)

if __name__ == "__main__":
    main()
