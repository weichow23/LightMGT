#!/usr/bin/env python3
"""Check Echo-4o and OpenGPT-4o-Image schemas."""
import json, os
os.environ.setdefault("HF_HOME", "/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache")
os.environ.setdefault("HF_TOKEN", "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")
from huggingface_hub import hf_hub_download

DL = "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/test_schema"

def show(name, row):
    print(f"\n=== {name} ===")
    for k, v in row.items():
        if isinstance(v, str):
            print(f"  {k}: str[{len(v)}] = {v[:150]}")
        elif isinstance(v, list):
            first = str(v[0])[:80] if v else "None"
            print(f"  {k}: list[{len(v)}] first={first}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")

# Echo-4o T2I
p1 = hf_hub_download("Yejy53/Echo-4o-Image", "Instruction-Following-Image/Instruction-Following-Image.jsonl", repo_type="dataset", local_dir=DL)
with open(p1) as f:
    show("Echo-4o T2I", json.loads(f.readline()))
    # count lines
    f.seek(0)
    n = sum(1 for _ in f)
    print(f"  Total rows: {n}")

# Echo-4o Surreal
p2 = hf_hub_download("Yejy53/Echo-4o-Image", "Surrel-Fantasy-Image/conflict.jsonl", repo_type="dataset", local_dir=DL)
with open(p2) as f:
    show("Echo-4o Surreal", json.loads(f.readline()))
    f.seek(0)
    n = sum(1 for _ in f)
    print(f"  Total rows: {n}")

# OpenGPT-4o gen
p3 = hf_hub_download("WINDop/OpenGPT-4o-Image", "gen.json", repo_type="dataset", local_dir=DL)
with open(p3) as f:
    data = json.load(f)
    show(f"OpenGPT-4o gen ({len(data)} total)", data[0])

# OpenGPT-4o editing
p4 = hf_hub_download("WINDop/OpenGPT-4o-Image", "editing.json", repo_type="dataset", local_dir=DL)
with open(p4) as f:
    data2 = json.load(f)
    show(f"OpenGPT-4o editing ({len(data2)} total)", data2[0])
