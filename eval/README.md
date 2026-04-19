# LightMGT Evaluation

## Benchmarks

| Benchmark | Type | Samples | Scorer | Metric |
|-----------|------|---------|--------|--------|
| **GenEval** | T2I | 553 | Mask2Former + CLIP | Per-tag accuracy |
| **DPG-Bench** | T2I | 1065 | GPT-4o VQA | DPG Score (0-1) |
| **ImgEdit** | Edit | 785 | GPT-4o | 3 dims × 5-point |
| **GEditBench** | Edit | ~600 | VIEScore (GPT-4o) | SC, PQ, Overall |
| **DreamBench++** | Subject | varies | CLIP-I, CLIP-T, DINO | Cosine similarity |

## Data Paths

```
/mnt/hdfs/weichow/maskedit/eval_data/
├── evaluation_metadata_long.jsonl          # GenEval prompts
├── geneval/evaluation/object_names.txt     # GenEval class names
├── geneval_models/mask2former_*.pth        # GenEval detector
├── ELLA/dpg_bench/prompts/*.txt            # DPG-Bench prompts
├── ELLA/dpg_bench/dpg_bench.csv            # DPG-Bench questions
├── Benchmark/singleturn/                   # ImgEdit
├── Benchmark/hard/                         # ImgEdit (hard)
├── geditbench/                             # GEditBench (HF)
└── dreambench_plus_plus/                   # DreamBench++ (HF)
```

## Usage

Each benchmark: `python eval/{benchmark}/run.py --mode all --ckpt <path>`

```bash
# GenEval (T2I quality)
python eval/geneval/run.py --mode all --ckpt /path/to/checkpoint.pt --device cuda:0

# DPG-Bench (T2I compositional)
python eval/dpg_bench/run.py --mode all --ckpt /path/to/checkpoint.pt

# ImgEdit (editing quality)
python eval/imgedit/run.py --mode all --ckpt /path/to/checkpoint.pt

# GEditBench (editing VIEScore)
python eval/geditbench/run.py --mode all --ckpt /path/to/checkpoint.pt

# DreamBench++ (subject preservation)
python eval/dreambench_pp/run.py --mode all --ckpt /path/to/checkpoint.pt
```

## Target Metrics (vs DreamLite)

| Benchmark | DreamLite | LightMGT Target |
|-----------|-----------|----------------|
| GenEval | 0.72 | ≥0.70 |
| DPG-Bench | 85.8 | ≥80 |
| ImgEdit | 4.11 | ≥4.0 |
| GEditBench | 6.88 | ≥6.0 |
