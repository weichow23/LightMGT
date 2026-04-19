#!/usr/bin/env python3
import os, json
os.environ["HF_HOME"] = "/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache"
os.environ["HF_TOKEN"] = "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"

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

# Try vanloc version (has images embedded in parquet)
print("Downloading vanloc pico-banana parquet...")
path = hf_hub_download(
    "vanloc1808/pico-banana-smolvlm-format-with-rejected-answer",
    "data/train-00000-of-01238.parquet",
    repo_type="dataset",
    local_dir="/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/test_schema/pico_vanloc"
)
print(f"Downloaded: {path}")
pf = pq.ParquetFile(path)
t = pf.read_row_group(0)
print(f"Columns: {t.column_names}")
print(f"Rows: {len(t)}")

for c in t.column_names:
    v = t.column(c)[0].as_py()
    if isinstance(v, dict):
        keys = list(v.keys())
        blen = len(v.get("bytes", b""))
        print(f"  {c}: dict keys={keys} bytes={blen}")
    elif isinstance(v, str):
        print(f"  {c}: str[{len(v)}] = {v[:150]}")
    elif isinstance(v, list):
        print(f"  {c}: list[{len(v)}]")
        for i, item in enumerate(v[:2]):
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, list):
                    for ci, c_item in enumerate(content[:2]):
                        if isinstance(c_item, dict):
                            print(f"    [{i}].content[{ci}] type={c_item.get('type','')} text={str(c_item.get('text',''))[:80]}")
                            if c_item.get("type") == "image":
                                print(f"      image bytes={len(c_item.get('image',''))}")
                else:
                    print(f"    [{i}] content={str(content)[:100]}")
            else:
                print(f"    [{i}] = {str(item)[:100]}")
    else:
        print(f"  {c}: {type(v).__name__} = {v}")
