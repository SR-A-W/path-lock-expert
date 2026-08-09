#!/usr/bin/env python3
"""临时脚本：用真实 tokenizer 统计数据集 token 长度分布"""
from transformers import AutoTokenizer
import json, random, sys

tokenizer = AutoTokenizer.from_pretrained(
    "./models/source/Qwen/Qwen3-4B-Instruct-2507",
    trust_remote_code=True,
)

think_items = []
nothink_items = []
with open("./train/data/superior_hybrid_65k_65k.jsonl") as f:
    for line in f:
        if not line.strip(): continue
        item = json.loads(line)
        user_msg = next(c["value"] for c in item["conversations"] if c["from"] == "user")
        if "/think" in user_msg and "/no_think" not in user_msg:
            think_items.append(item)
        else:
            nothink_items.append(item)

print(f"Think: {len(think_items)}, No_think: {len(nothink_items)}", flush=True)

def tokenize_len(item):
    u = next(c["value"] for c in item["conversations"] if c["from"] == "user")
    a = next(c["value"] for c in item["conversations"] if c["from"] == "assistant")
    return len(tokenizer.encode(u + a, add_special_tokens=False))

print("Tokenizing think...", flush=True)
think_lengths = [tokenize_len(it) for it in think_items]

print("Tokenizing no_think (sample 5000)...", flush=True)
random.seed(42)
sample = random.sample(nothink_items, min(5000, len(nothink_items)))
nothink_lengths = [tokenize_len(it) for it in sample]

think_lengths.sort()
nothink_lengths.sort()

for cutoff in [16384, 32768]:
    tt = sum(1 for t in think_lengths if t > cutoff)
    nr = sum(1 for t in nothink_lengths if t > cutoff) / len(nothink_lengths)
    ne = int(nr * len(nothink_items))
    total = len(think_items) + len(nothink_items)
    print(f"\n=== cutoff = {cutoff} tokens ===")
    print(f"/think: 超长 {tt}/{len(think_items)} ({tt/len(think_items)*100:.1f}%)")
    print(f"/no_think: 超长 ~{ne}/{len(nothink_items)} ({nr*100:.1f}%)")
    print(f"有毒: ~{tt+ne}/{total} ({(tt+ne)/total*100:.1f}%)")

n = len(think_lengths)
print(f"\n=== /think 分布 ===")
for p in [50,75,80,85,90,95,99]:
    v = think_lengths[min(int(n*p/100),n-1)]
    m = " !!超16k" if v>16384 else ""
    m += " !!超32k" if v>32768 else ""
    print(f"  P{p}: {v:,}{m}")
print(f"  Max: {think_lengths[-1]:,}")

n2 = len(nothink_lengths)
print(f"\n=== /no_think 分布 ===")
for p in [50,75,90,95,99]:
    print(f"  P{p}: {nothink_lengths[min(int(n2*p/100),n2-1)]:,}")
print(f"  Max: {nothink_lengths[-1]:,}")
