import tarfile, json
tgz = "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/downloads/nano150k/Nano-150k.tar.gz"
with tarfile.open(tgz, "r:gz") as tf:
    jsonls = [m for m in tf.getmembers() if m.name.endswith(".jsonl")]
    total = 0
    for jm in jsonls:
        data = tf.extractfile(jm).read().decode()
        lines = data.strip().split("\n")
        row = json.loads(lines[0])
        tt = row.get("task_type", "?")
        has_inp = "input_images" in row
        print(f"  {jm.name}: {len(lines)} rows, task={tt}, has_input={has_inp}")
        total += len(lines)
    print(f"\nTotal: {total} records across {len(jsonls)} jsonls")
