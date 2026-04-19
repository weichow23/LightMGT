#!/usr/bin/env python3
"""Quick schema detection test for PT datasets."""
import os, sys
os.environ.setdefault("HF_HOME", "/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache")
os.environ.setdefault("HF_TOKEN", "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")

from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

api = HfApi()
TEST_DIR = "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/test_schema"

datasets = {
    "laion_512": "limingcv/LAION_Aesthetics_512",
    "laion_1024": "limingcv/LAION_Aesthetics_1024",
    "journeydb_p1": "limingcv/JourneyDB_part1",
    "journeydb_p2": "limingcv/JourneyDB_part2",
    "cc12m": "pixparse/cc12m-wds",
}

target = sys.argv[1] if len(sys.argv) > 1 else "all"

for key, hf_id in datasets.items():
    if target != "all" and target != key:
        continue
    print(f"\n{'='*60}")
    print(f"Dataset: {key} ({hf_id})")
    print(f"{'='*60}")

    try:
        files = api.list_repo_files(hf_id, repo_type="dataset")
        parquets = [f for f in files if f.endswith(".parquet")]
        tars = [f for f in files if f.endswith(".tar")]
        print(f"  Total files: {len(files)}")
        print(f"  Parquet: {len(parquets)}, Tar: {len(tars)}")

        if parquets:
            path = hf_hub_download(hf_id, parquets[0], repo_type="dataset", local_dir=TEST_DIR)
            pf = pq.ParquetFile(path)
            table = pf.read_row_group(0)
            print(f"  Schema: {table.schema}")
            for col in table.column_names:
                val = table.column(col)[0].as_py()
                if isinstance(val, dict):
                    blen = len(val.get("bytes", b""))
                    print(f"    {col}: dict keys={list(val.keys())} bytes_len={blen}")
                elif isinstance(val, bytes):
                    print(f"    {col}: bytes len={len(val)}")
                elif isinstance(val, str):
                    print(f"    {col}: str[{len(val)}] = {val[:100]!r}")
                else:
                    print(f"    {col}: {type(val).__name__} = {val}")
        elif tars:
            print(f"  Tar files: {tars[:5]}")
    except Exception as e:
        print(f"  ERROR: {e}")
