import pyarrow.parquet as pq
pf = pq.ParquetFile("/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/test_schema/pico2/data/train-00000-of-01238.parquet")
t = pf.read_row_group(0)
print(f"Cols: {t.column_names}")
print(f"Rows: {len(t)}")
for c in t.column_names:
    v = t.column(c)[0].as_py()
    tp = type(v).__name__
    if isinstance(v, dict):
        blen = len(v.get("bytes", b""))
        print(f"  {c}: dict keys={list(v.keys())} bytes_len={blen}")
    elif isinstance(v, str):
        print(f"  {c}: str = {v[:150]}")
    elif isinstance(v, list):
        print(f"  {c}: list[{len(v)}]")
        if v and isinstance(v[0], dict):
            print(f"    [0] keys={list(v[0].keys())}")
            content = v[0].get("content", "")
            if isinstance(content, list):
                for item in content[:2]:
                    print(f"      content item: {str(item)[:100]}")
            else:
                print(f"      content: {str(content)[:100]}")
    else:
        print(f"  {c}: {tp} = {v}")
