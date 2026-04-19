#!/usr/bin/env python3
"""Quick test: process one parquet file to verify the pipeline works."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.process_pt_data import _worker_process_parquet

test_pq = "pt_data/test_schema/data/train-00000-of-03041.parquet"
test_tar = "pt_data/test_output.tar"

if not os.path.exists(test_pq):
    print(f"SKIP: {test_pq} not found (run test_schema.py laion_512 first)")
    sys.exit(0)

print(f"Processing {test_pq} ...")
args = (test_pq, test_tar, "laion_aes_512", "image", "prompt", "image_width", "image_height", 0)
records, errors, written, err = _worker_process_parquet(args)

print(f"Records: {len(records)}, Errors: {errors}, Written: {written}")
if err:
    print(f"Error: {err}")
if records:
    print(f"Sample record: {records[0]}")
    tar_size = os.path.getsize(test_tar)
    print(f"Tar size: {tar_size / 1e6:.1f} MB")

# Cleanup
if os.path.exists(test_tar):
    os.remove(test_tar)
    print("Cleaned up test tar")

print("TEST PASSED" if records and not err else "TEST FAILED")
