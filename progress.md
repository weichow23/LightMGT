# LightMGT 项目进展

> 更新: 2026-04-12 (代码修复 v2 完成, 256px 训练中)

---

## 1. 当前状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| 模型代码 | **完成** | 384M params, GPU 验证通过 (H800 bf16) |
| 数据处理 | **完成** | ~75M+ 样本 HDFS, 60 个 tar index pkl |
| Phase 1 128px | **完成** | 249K steps, loss 5.50 plateau, LR 到 floor |
| Phase 1 256px | **进行中** | step 5500+/150K, 8×H800, loss ~5.45 |
| 评测 | **代码对齐** | eval/ 已切换到 OpenMAGVIT2 262K |

---

## 2. 训练状态

**Phase 1 128px (已完成)**:
- Steps: 249K / 300K (提前结束, loss plateau)
- Loss: 5.50 (从 5.68→5.54→5.50, 最后 112K 步仅降 0.04)
- LR: 已到 floor 1.5e-05, 继续训练收益极小
- 配置: BS=128, grad_accum=1, 8×H800, 0.30 s/s
- 最终 checkpoint: checkpoint-249000

**当前 Phase 1 256px (修复后重启, 进行中)**:
- Step: **5500+** / 150K
- Loss: **~5.45** (从 5.95→5.45, 正常收敛)
- 速度: 0.16 s/s, 8×H800
- 配置: BS=32, grad_accum=4, LR=1.5e-4 (cosine w/ warmup 1K, fresh schedule)
- Resume: checkpoint-249000 weights only (optimizer/scheduler/step 重置)
- ETA: ~145K × 6.25s/step / 3600 ≈ **10 天**
- 日志: `fpo_light:/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/train_phase1_256px_v2.log`
- Wandb: `phase1_256px_8gpu_alldata`
- NaN check: 全部 clean
- ⚠️ 数据: echo4o_t2i/echo4o_surreal/opengpt4o_gen tar 文件在 BN 路径缺失, 部分 mico/cc12m 图片 not in tar index

**数据 (Phase 1, 128px/256px 共用)**:
- JSON dirs: `/mnt/hdfs/weichow/maskedit/t2i/` + `/mnt/hdfs/weichow/maskedit/t2i-pt/`
- 26 组 tar patterns (T2I 全量 + T2I-PT 全量 + benchmark T2I)
- 总计 ~60M+ 样本

**Wandb 推理监控**:
- 每 1K step 生成 3 张图推理, 上传 wandb (MaskGIT 20步 CFG=5.0)
- 128px 生成耗时 1.2s, 256px 预计 ~4s

**Checkpoints**: `/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/checkpoints/` (保留最近 5 个)

---

## 3. 模型架构 (384M)

- FLUX.2-style Transformer + MaskGIT + Hybrid GLA
- 4 Double (Softmax) + 14 Single GLA + 6 Single Softmax blocks
- hidden=1024, heads=16, head_dim=64, mlp_ratio=2.6875
- Text: Qwen3.5-0.8B (frozen, 1024D, 无需投影)
- VQ: Open-MAGVIT2-PT 262K (frozen, LFQ 18-bit, 16x↓)
- Gen Head: Factorized 2×512 (18-bit LFQ → 2 groups of 9 bits)
- Mask token ID: 262144
- 详见 `MASTER.md`

---

## 4. 数据清单

权威清单: `DATA_current.md` + 远程 `/mnt/hdfs/weichow/maskedit/DATA.md`

| 类别 | 量级 | HDFS 路径 | 训练使用 |
|------|------|-----------|---------|
| T2I (高质量) | ~16.1M | `t2i/` | Phase 1 ✅ |
| T2I-PT (预训练) | ~43M+ | `t2i-pt/` | Phase 1 ✅ |
| Benchmark T2I | ~145K | `t2i/` (echo4o/opengpt4o) | Phase 1 ✅ |
| Edit | ~2.0M | `edit/` | Phase 2 |
| Multi-Image | ~64.5万 | `multi/` | Phase 2 |
| Control (6种) | ~3.0M | `control/` | Phase 2 |
| VQA | ~1.63M | `vqa/` | Phase 2 |
| Interleaved | ~22.6万 | `interleave/` | Phase 2 |
| T2V | ~3.13M | `t2v/` | 未计划 |
| VEdit | ~1.77M | `vedit/` | 未计划 |

Tar index 缓存:
- BN1: `/mnt/bn/search-auto-eval/zhouwei/maskedit_cache/tar_index/` (18 pkl, T2I+Edit)
- BN2: `/mnt/bn/search-auto-eval-v2/zhouwei/maskedit_cache/tar_index/` (46 pkl, T2I-PT+其他)
- TarImageReader 已支持双缓存目录搜索

**Benchmark T2I tars (⚠️ BN 路径缺失, 训练中跳过)**:

| Dataset | Samples | 状态 |
|---------|---------|------|
| echo4o_t2i | 67,958 | ❌ BN tar 不存在, 训练 log 报 FileNotFoundError |
| echo4o_surreal | 37,548 | ❌ 同上 |
| opengpt4o_gen | 39,607 | ❌ 同上 |

需要重新打包到 BN 或从 HDFS 拉取。当前训练通过 dataset retry 跳过这些样本。

---

## 5. 关键代码文件

```
lightmgt/
  configuration_lightmgt.py   # 模型配置 (LightMGTConfig)
  modeling_lightmgt.py         # 主模型 + FactorizedGenHead
  modeling_gla.py              # Gated Linear Attention (双向 kernel trick + 双向 DWConv)
  modeling_rope.py             # 3D Unified RoPE
  pipeline_lightmgt.py         # 推理 pipeline (MaskGIT + CFG, last-token pooling)
  scheduler_maskgit.py         # MaskGIT arccos/cosine scheduler
  vq_wrapper.py                # VQ-VAE wrapper (OpenMAGVIT2 262K only)

train/
  dataset.py                   # 数据加载 (TarImageReader 双缓存 + 多 JSON dir + BucketBatchSampler + load error tracking)
  train.sh                     # 启动脚本 (torchrun 8×GPU, 26 tar patterns, auto-resume)

scripts/
  train_phase1.py              # Phase 1 训练 (DDP + online VQ encoding + wandb 推理监控 + error dump)
```

---

## 6. 基础设施

| 项目 | 值 |
|------|------|
| GPU 机器 | `fpo_light` (8×H800 80GB) |
| 远程代码 | `/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/` |
| HDFS 数据 | `/mnt/hdfs/weichow/maskedit/` |
| VQ ckpt | `/mnt/bn/search-auto-eval-v2/zhouwei/nextmgt/tokenizer_ckpts/pretrain256_262144.ckpt` |
| Wandb | project=LightMGT, entity=3210103790 |
| 训练日志 | `fpo_light:/mnt/bn/.../LightMGT/train_phase1_256px_v2.log` |

---

## 7. TODO

### P0 v1 (已部署, 256px 已重启):
- [x] **GLA 双向注意力**: 将 causal mask 改为 bidirectional linear attention (kernel trick)
- [x] **text_pooled**: mean pooling → last-token pooling (causal LM 标准做法)
- [x] **删除 dummy loss**: 修复 DoubleBlock 最后一层 unused params
- [x] **BucketBatchSampler**: 加 epoch seed 确保 DDP ranks 一致性
- [x] **TarImageReader**: path normalization 修复 cc12m 等 tar miss
- [x] **删除 accelerate 死代码**: train/train.py + configs/accelerate_8gpu.yaml

### P0 v2 (2026-04-12, 待部署):
- [x] **eval/utils.py**: IBQ16K → OpenMAGVIT2Wrapper + mask_token_id=262144 + 正确 state_dict key
- [x] **pipeline_lightmgt.py**: encode_image/decode_tokens 对齐 OpenMAGVIT2Wrapper API, encode_prompt 改为 last-token pooling
- [x] **GLA DWConv**: causal padding → 双向 symmetric padding (MaskGIT 非自回归)
- [x] **vq_wrapper.py**: 删除 IBQ16KWrapper 遗留代码
- [x] **dataset.py**: data load error tracking (per-error-type warning + JSON dump for offline repair)
- [x] **train_phase1.py**: 训练结束 dump data_load_errors.json, 删除 _nullcontext
- [x] **lightmgt_base.yaml**: 对齐实际训练参数 (LR/warmup/save_steps/log_steps)
- [x] **注释/文档修复**: FactorizedGenHead comment (128→group_vocab), scheduler docstring (128→512)
- [x] **SharedAdaLN**: 清理构造函数中被 __init__ 末尾覆盖的死代码

### P1: 训练中期
- [ ] Phase 1 渐进分辨率: ~~128px~~ → **256px (进行中)** → 512px → 1024px
- [ ] 推理脚本 `generate.py` (独立批量生成, 不依赖训练循环)
- [ ] GenEval / DPG-Bench 评测接入
- [ ] 修复缺失数据源: echo4o_t2i/echo4o_surreal/opengpt4o_gen tar + mico tar index

### P2: Phase 2
- [ ] Edit + Control + Multi-Image 联合训练 (T2I 90% + 其他 10%)
- [ ] 验证 nano150k/pico-banana edit 格式与 edit dataloader 兼容

---

## 8. 心得

### 训练
- **128px 训练**: BS=128 × 8GPU × 1accum = 1024 effective, 0.30 s/s, 1K步≈56分钟
- **256px 训练**: BS=32 × 8GPU × 4accum = 1024 effective, 0.16 s/s, 1K步≈104分钟
- **分辨率切换**: 只加载模型权重, 重置 optimizer/scheduler/step. 新 warmup 1K 步. Loss 从 5.95 快速下降
- **GPU 显存不均**: GPU0 ~80GB vs 其他 ~30GB (frozen text encoder + VQ model 全在 GPU0)
- **Wandb 推理**: MaskGIT 20步 + CFG double forward, 256px ~1.4s, 几乎零开销
- **数据量翻倍效果**: 从 13M (camera+fine_t2i) 扩到 60M+ (全量 T2I+PT), loss 下降更快
- **TarImageReader 双缓存**: BN1+BN2 两个 cache dir 解决了 T2I vs T2I-PT index 分布在不同存储的问题
- **Tar pattern hash 必须精确匹配**: cache key = md5(pattern string), 差一个字符就 cache miss
- **数据 load error tracking**: dataset.py 按 (error_type, path_prefix) 去重 warn, 训练结束 dump data_load_errors.json

### 数据处理
- **流式处理**: download → process → upload HDFS → delete BN → next batch, 否则 BN 30T 配额秒爆
- **HDFS tar index**: 用流式 `tarfile.open(path, "r|")` 不用 `"r"` (HDFS seek 极慢, 差 10x+)
- **多机并行**: 4 台 CPU 机器分 shard, 32 workers 流式读取, 493 tars (6.67M files) 仅 57 分钟

### 架构
- hidden=1024 是 300M-500M 级最优, 所有 SOTA (MaskBit, MAR-L, LlamaGen-L) 均用
- Hybrid GLA (14 GLA + 6 Softmax) 是创新点: MaskGIT 领域首次引入线性注意力
- Factorized Gen Head (2×512) 关键: 避免 262K-way softmax bottleneck
