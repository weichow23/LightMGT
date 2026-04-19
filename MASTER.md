# LightMGT: Lightweight Masked Generative Transformer for On-Device Generation and Editing

> 定位: DreamLite (arXiv:2603.28713) 竞品
> 架构: FLUX.2-style Transformer (4 双流 + 20 单流) + MaskGIT + Hybrid GLA
> Text Encoder: Qwen3.5-0.8B (frozen)
> VQ-VAE: Open-MAGVIT2-PT 262K (frozen, LFQ 18-bit, codebook=262144, 训练于~100M多样化图片)
> 参数: ~0.37B (transformer only)
> 任务: T2I + Edit + Multi-Image + 6 Control (共 9 种)

### 与竞品 DreamLite 技术路线对比


| 项目               | **LightMGT (本项目)**                  | **DreamLite (arXiv:2603.28713)**          |
| ---------------- | ----------------------------------- | ----------------------------------------- |
| **范式**           | Masked Generation (MaskGIT)         | Latent Diffusion (Flow Matching)          |
| **VAE**          | Open-MAGVIT2-PT, 离散LFQ 18-bit, 16x↓ | TinyVAE (TAESD), 连续4通道, 8x↓, ~2.5M params |
| **Backbone**     | FLUX-style Transformer + GLA (368M) | UNet 从 SDXL 压缩 (389M)                     |
| **从头训练?**        | 是 — from scratch                    | 否 — 从 SDXL→SnapGen 蒸馏                     |
| **蒸馏**           | 无 (from scratch)                    | 3级: 架构压缩 + KD训练 + DMD2步数蒸馏                |
| **Text Encoder** | Qwen3.5-0.8B (frozen)               | Qwen3-VL-2B                               |
| **训练数据**         | 14.96M T2I                          | 21.7M (20M T2I + 1.7M edit)               |
| **推理**           | MaskGIT iterative unmasking         | 4步去噪 (DMD2)                               |
| **目标**           | 端侧生成+编辑                             | 端侧生成+编辑, 骁龙8 Gen3 <1s                     |
| **创新点**          | GLA线性注意力 (1024px 4-9x加速)            | 前景加权编辑loss + ReFL后训练                      |


**DreamLite 蒸馏方法**: SDXL 2.5B → SnapGen 372M (架构压缩 + output/feature KD) → DreamLite 389M (加editing) → DMD2 4步推理。详见下方。

---

## 1. 架构总览

### 1.1 整体架构

```
Text → Qwen3.5-0.8B (frozen) → hidden_states [B, T, 1024] + pooled [B, 1024]
                                        ↓ RMSNorm → [B, T, 1024] (无需投影, 维度已对齐)
Image → Open-MAGVIT2-PT encode → discrete tokens [B, N] → LFQ Embedding → [B, N, 1024]
                                        ↓
                            ┌── 4× Double Blocks (Full Softmax) ─┐
                            │  text + image joint attention       │
                            │  Separate MLPs per modality         │
                            │  ← text-image 精确对齐              │
                            └────────────────────────────────────┘
                                        ↓
                            ┌── 14× Single Blocks (GLA, 双向) ───┐
                            │  Bidirectional Linear Attention     │
                            │  + DWConv + AdaLN                   │
                            │  ← 特征提取, 1024px 下 4-9x 加速    │
                            └────────────────────────────────────┘
                                        ↓
                            ┌── 6× Single Blocks (Full Softmax) ─┐
                            │  Full Softmax Self-Attention        │
                            │  ← mask prediction 需要 sharp attn  │
                            └────────────────────────────────────┘
                                        ↓
                            Factorized Gen Head (2×512)
                                        ↓
                            MaskGIT iterative unmasking → Open-MAGVIT2-PT decode → Image
```

### 1.2 参数配置

```python
config = {
    # Core
    "hidden_size": 1024,
    "num_double_blocks": 4,
    "num_single_gla_blocks": 14,
    "num_single_softmax_blocks": 6,
    "num_attention_heads": 16,          # head_dim = 64
    "mlp_type": "SwiGLU",              # effective ratio ≈ 2.67×

    # 3D RoPE (Z-Image 式)
    "pos_embed_type": "3d_unified_rope",
    "rope_axes_dim": [8, 28, 28],       # temporal(8d) + H(28d) + W(28d) = 64d = head_dim

    # VQ (Open-MAGVIT2-PT, LFQ 18-bit)
    "codebook_size": 262144,            # 2^18 LFQ
    "mask_token_id": 262144,
    "num_lfq_bits": 18,

    # Gen Head (MaskBit factorized: 18 bits → 2 groups of 9 bits)
    "gen_head_groups": 2,               # 262144 = 512 × 512
    "gen_head_vocab": 512,

    # Text
    "text_hidden_size": 1024,           # Qwen3.5-0.8B (实测 hidden=1024, 无需投影)

    # Design choices (FLUX.2 + Z-Image)
    "use_bias": False,
    "shared_adaln": True,               # 跨层共享 AdaLN
    "parallel_block": True,             # 单流 attn∥MLP 并行
    "sandwich_norm": True,              # RMSNorm before+after attn/FFN
    "qk_norm": True,

    # Training
    "label_smoothing": 0.1,
    "cfg_dropout": 0.1,
}
```

### 1.3 参数量


| 组件                                          | 参数量               |
| ------------------------------------------- | ----------------- |
| LFQ Embedding (Linear 18→1024 + mask token) | ~19K              |
| Text Projector (RMSNorm, dim 已对齐)           | 0.001M            |
| Shared AdaLN                                | 15.7M             |
| 4× Double Blocks                            | 77.4M             |
| 14× Single GLA Blocks (双向线性注意力)             | 194.6M            |
| 6× Single Softmax Blocks                    | 77.1M             |
| Conditioning (Timestep + Pooled Text)       | 2.4M              |
| Gen Head (Factorized 2×512)                 | 1.05M             |
| **Transformer Total**                       | **~0.37B (368M)** |
| + Qwen3.5-0.8B (frozen)                     | +0.75B            |
| + Open-MAGVIT2-PT 262K (frozen)             | +~80M             |


### 1.4 关键组件


| 组件               | 配置                            | 说明                                                         |
| ---------------- | ----------------------------- | ---------------------------------------------------------- |
| **Text Encoder** | Qwen3.5-0.8B                  | Frozen, hidden=1024 (无需投影)                                 |
| **VQ-VAE**       | Open-MAGVIT2-PT 262K          | Frozen, LFQ 18-bit, 16x↓, 1024px → 64×64 = **4096 tokens** |
| **Gen Head**     | Factorized 2×512              | 避免 262K-way softmax (18 bits → 2 groups of 9 bits)         |
| **Edit**         | OminiControl injection + LoRA | Ref image separate stream                                  |
| **Control**      | Multi-LoRA                    | 7 adapters (edit + 6 control), ~2M each                    |
| **Scheduler**    | MaskGIT arccos                | Confidence-based unmasking                                 |


---

## 2. 核心设计决策

### 2.1 为什么 4+14+6 = 24 blocks, hidden=1024

所有同规模 SOTA 模型均采用此配置:


| 模型         | Params | Hidden | Layers | FID (IN-256) |
| ---------- | ------ | ------ | ------ | ------------ |
| MaskBit    | 305M   | 1024   | 24     | **1.52**     |
| MAR-L      | 479M   | 1024   | 24     | **1.78**     |
| LlamaGen-L | 343M   | 1024   | 24     | 3.07         |
| VAR-d20    | 600M   | 1280   | 20     | 2.95         |


hidden=1536 × 12 blocks (同参数量) 不可取: GLA 收益被削弱 (14→4层), 且无 SOTA 先例。

### 2.2 Hybrid GLA (创新点: MaskGIT + Linear Attention 无人区)

**没有任何已发表工作将 linear attention 用于 masked generative transformer。**


| 分辨率                | Token 数  | GLA 加速   |
| ------------------ | -------- | -------- |
| 256px              | 256      | <10%     |
| 512px              | 1024     | 2-4x     |
| **1024px**         | **4096** | **4-9x** |
| Multi-image 1024px | 8K-12K   | **10x+** |


分层设计理由:

- **4 双流 (softmax)**: text-image cross-modal alignment 需要精确 attention
- **14 单流 (GLA + DWConv)**: DiG (CVPR 2025) 证明 GLA 不掉质量; LiT (ICCV 2025) 证明 9x 加速
- **6 单流 (softmax)**: 靠近输出层, mask prediction 需要 sharp discriminative attention

降级方案: 如 GLA 不 work → (A) 全 softmax (B) runtime token>2048 时启用 (C) Agent Attention 免训插入

### 2.3 双流+单流 — FLUX.2 路线


|       | FLUX.1      | **FLUX.2**     | Z-Image    | **LightMGT**   |
| ----- | ----------- | -------------- | ---------- | -------------- |
| 双流:单流 | 1:2 (19+38) | **1:6** (8+48) | 0:30 (纯单流) | **1:5** (4+20) |
| MLP   | GELU        | **SwiGLU**     | SwiGLU     | **SwiGLU**     |
| AdaLN | 每层独立        | **跨层共享**       | 低秩共享       | **跨层共享**       |
| Bias  | 有           | **无**          | 无          | **无**          |


Ablation 备选: 若双流不优, 用 Z-Image 式 Refiner (2层/模态, ~~8M) 替代 4 双流 (~~77M)。

### 2.4 Z-Image 式 3D Unified RoPE

axes_dim = [8, 28, 28] (temporal + H + W = 64 = head_dim)

```
Text tokens:      [t=0..L-1, 0, 0]     — temporal 递增, spatial=(0,0)
Reference image:  [t=L,   h, w]         — 共享空间坐标
Target image:     [t=L+1, h, w]         — 同空间, temporal+1
Multi-ref img2:   [t=L+2, h, w]         — temporal+2
```

**核心优势**: 编辑时 ref 和 target **共享 spatial RoPE → 天然像素对齐**, 无需额外对齐机制。

vs FLUX (text 全零, 无位置) vs Qwen-Image MSRoPE (对角线, 复杂) → Z-Image 最简洁实用。

```python
def build_position_ids(text_len, img_h, img_w, num_ref_images=0):
    ids = []
    for t in range(text_len):
        ids.append([t, 0, 0])
    t_offset = text_len
    for ref_idx in range(num_ref_images):
        for h in range(img_h):
            for w in range(img_w):
                ids.append([t_offset + ref_idx, h, w])
    t_target = t_offset + num_ref_images
    for h in range(img_h):
        for w in range(img_w):
            ids.append([t_target, h, w])
    return torch.tensor(ids)
```

### 2.5 推理优化: KV Cache + Time Interval CFG

**KV Cache 跨步复用**: MaskGIT 中 unmasked tokens 不变, 其 KV 可跨步缓存:

- Step N 计算所有 token 的 KV
- Step N+1 只重算新 unmask 的 token 的 QKV, 已 unmask 的从 cache 取
- 后期步骤 ~80% token 已 unmask → **省 ~60-80% compute**

**Time Interval CFG** (eMIGM): 只在 step 30%-70% 做 CFG (双 forward), 首尾单 forward → **省 ~40% 推理**

---

## 3. 训练算法

### 3.1 LFQ Embedding + Factorized Gen Head

Open-MAGVIT2-PT 262K 使用 18-bit LFQ: 每个 token = 18 个二值决策 ({-1,+1}^18)。

**LFQ Embedding** (MaskBit 式 embedding-free): 不用 262K embedding table (~~268M params)，
直接将 token index 转为 18-bit binary vector 后投影到 hidden dim (~~19K params)。

```python
# LFQ Embedding: index → bits → projection
bits = ((token_id >> bit_shifts) & 1) * 2 - 1   # [B, N, 18], values {-1, +1}
hidden = bit_proj(bits)                          # Linear(18, 1024), ~19K params
hidden[is_mask] = mask_token                     # learnable 1024-dim vector
```

**Factorized Gen Head**: 262144 = 512 × 512, 分解为两个 512-way 分类 (18 bits → 2组 × 9 bits):

```python
logits_g1 = gen_head_g1(hidden)                # [B, N, 512] ← upper 9 bits
logits_g2 = gen_head_g2(hidden)                # [B, N, 512] ← lower 9 bits
target_g1 = target_token_ids // 512
target_g2 = target_token_ids % 512
loss = 0.5 * CE(logits_g1[mask], target_g1[mask], label_smoothing=0.1) \
     + 0.5 * CE(logits_g2[mask], target_g2[mask], label_smoothing=0.1)

# 推理: confidence = prob_g1.max * prob_g2.max
pred_token = pred_g1 * 512 + pred_g2
```

LFQ 结构化 codebook 保证 factorized head 有效: Hamming 距离近 = 视觉相似。
Random baseline loss: ln(512) = 6.238。

### 3.2 训练流程

```python
def train_step(batch):
    vq_tokens = open_magvit2.encode(image)                # [B, N] frozen, LFQ 18-bit
    r = torch.rand(B)
    mask_ratio = torch.arccos(r) / (math.pi / 2)         # arccos schedule
    num_mask = (mask_ratio * N).long()
    mask = random_mask(vq_tokens, num_mask)
    masked_tokens = vq_tokens.clone()
    masked_tokens[mask] = MASK_TOKEN_ID                   # 262144

    text_hidden = qwen35.encode(text)                     # frozen
    text_pooled = last_token_pool(text_hidden)  # causal LM → last non-pad token
    if random.random() < 0.1:                             # CFG dropout
        text_hidden, text_pooled = zeros, zeros

    logits = transformer(masked_tokens, text_hidden, text_pooled, mask_ratio * 1000)
    loss = factorized_ce(logits[mask], vq_tokens[mask])   # 2×512 factorized CE
    return loss
```

### 3.3 推理流程

```python
def generate(text, num_steps=20):
    tokens = torch.full([1, N], MASK_TOKEN_ID)
    for step in range(num_steps):
        logits = transformer(tokens, text_hidden, text_pooled, timestep)
        pred_tokens, confidence = factorized_sample(logits)
        is_mask = (tokens == MASK_TOKEN_ID)
        tokens = torch.where(is_mask, pred_tokens, tokens)
        if step < num_steps - 1:
            tokens = remask_lowest_confidence(tokens, confidence, schedule(step))
    return open_magvit2.decode_tokens(tokens, img_h, img_w)
```

### 3.4 编辑训练: Partial Mask

```python
def train_edit_step(batch):
    source_tokens = open_magvit2.encode(source_image)
    target_tokens = open_magvit2.encode(target_image)
    edit_mask = detect_edit_region(source_tokens, target_tokens)
    masked_tokens = target_tokens.clone()
    masked_tokens[edit_mask] = MASK_TOKEN_ID
    # 非编辑区域保留 source (exact preservation)
    ref_hidden = transformer.project_reference(source_tokens)  # OminiControl injection
    logits = transformer(masked_tokens, text_hidden, text_pooled,
                         reference_hidden=ref_hidden, lora_enable=True)
    loss = factorized_ce(logits[edit_mask], target_tokens[edit_mask])
    return loss
```

### 3.5 训练超参 (MaskBit-aligned)


| 超参              | 值                        | 来源      |
| --------------- | ------------------------ | ------- |
| Mask schedule   | arccos(r)/(π/2)          | MaskBit |
| β₂              | 0.96                     | MaskBit |
| Weight decay    | 0.045                    | MaskBit |
| LR              | 1e-4, cosine (floor=10%) | MaskBit |
| Label smoothing | 0.1                      | MaskBit |
| CFG dropout     | 0.1                      | -       |
| Grad clip       | 1.0                      | -       |


### 3.6 LFQ Embedding (替代 VQ Embedding)

```python
# MaskBit-style embedding-free: 18-bit → 1024D projection
bit_proj = nn.Linear(18, 1024, bias=False)     # ~19K params (vs 旧 VQ Embedding 16.8M)
mask_token = nn.Parameter(torch.zeros(1024))    # learnable mask embedding
# 无需从 codebook 初始化 — LFQ 没有 learned codebook
```

---

## 4. 训练阶段

### Phase 1: T2I Pretraining (~300K steps)

128 -> 256 -> 512 →1024

- 数据: 14.96M T2I
- Progressive: 256→512→1024
- Batch: 64 (2×H800) / 256 (大集群)

### Phase 2: Joint Training (~40K steps)

- Edit + Control + Multi-Image
- 采样: T2I 90% : (Edit+Control+MultiImg) 10%
- 7 LoRA adapters (edit + 6 control, rank=32, ~2M each)

### Phase 3: RL

- SFT: 高质量 curated data
- RL: GRPO + HPSv3 (T2I) + EditReward (Edit)
- 参考 DreamLite: DPO + GRPO 联合, Qwen-Image 同

### Phase 4: Consistency Unmasking — MaskGIT 步数蒸馏（核心创新）

> 类比 DMD2 (diffusion 50→4步) 对 MaskGIT 的推理步数做蒸馏。
> **目前 MaskGIT 领域无人做过系统的步数蒸馏，这是空白领域。**

#### 4.1 动机


| 对比   | Diffusion (DMD2)        | MaskGIT (ours)            |
| ---- | ----------------------- | ------------------------- |
| 推理步数 | 50步 → 4步                | 16步 → 2-4步                |
| 输入   | 噪声图 → 干净图               | 全 mask → 全 token          |
| 中间状态 | 不同噪声水平                  | 不同 mask ratio             |
| 蒸馏信号 | score function matching | token confidence matching |


#### 4.2 算法: Consistency Unmasking (CU)

核心思想: **多步 MaskGIT teacher 的最终输出 = 少步 student 的直接预测**

```python
# === 训练 Consistency Unmasking ===

# Phase 4a: 用 Phase 1 训好的模型作为 teacher (16-step)
teacher = load_checkpoint("phase1_best.pt")
teacher.eval()

# Phase 4b: Student = 同架构, 初始化自 teacher
student = copy.deepcopy(teacher)

def cu_train_step(batch):
    vq_tokens = vq_model.encode(image)        # [B, N] ground truth
    
    # 1. 对任意 mask_ratio r ~ U(0,1)，构造 masked input
    r = torch.rand(B)
    mask_ratio = arccos_schedule(r)
    masked_tokens = apply_mask(vq_tokens, mask_ratio)
    
    # 2. Teacher 从该 masked_tokens 出发，跑完整 T 步 unmasking
    #    得到最终 token 预测 T* (detach, no grad)
    with torch.no_grad():
        teacher_final = teacher.iterative_unmask(
            masked_tokens, text_hidden, num_steps=16
        )  # [B, N] teacher 的最终输出 tokens
    
    # 3. Student 从同一 masked_tokens 出发，一步直接预测
    student_logits = student(masked_tokens, text_hidden, timestep)
    
    # 4. Loss: student 直接匹配 teacher 的最终输出
    loss_cu = CE(student_logits[mask], teacher_final[mask])
    
    # 5. 可选: 加 GAN/CLIP reward 补偿 teacher 不完美
    loss_reward = -clip_score(decode(student_pred), text)
    
    return loss_cu + lambda_reward * loss_reward
```

#### 4.3 推理对比

```
Teacher (16 步):
  [MASK MASK MASK MASK] → step1 → step2 → ... → step16 → [T1 T2 T3 T4]

Student (2-4 步):
  [MASK MASK MASK MASK] → step1 → (step2) → [T1 T2 T3 T4]
  每步 student 直接预测接近最终的 tokens，不需逐步 refine
```

#### 4.4 进阶: 渐进蒸馏 (Progressive CU)

类比 Progressive Distillation (Salimans & Ho 2022):

```
16 步 teacher → 8 步 student (每 2 步合 1 步)
 8 步 teacher → 4 步 student
 4 步 teacher → 2 步 student
```

每轮蒸馏减半步数，比直接 16→2 更稳定。

#### 4.5 与 DreamLite DMD2 的对应关系


| DMD2 组件                              | Consistency Unmasking 对应                              |
| ------------------------------------ | ----------------------------------------------------- |
| Distribution Matching Loss (score比较) | **Token Consistency Loss** (CE到teacher final tokens)  |
| GAN discriminator                    | **Token-level discriminator** 或 CLIP reward           |
| Backward simulation (用学生输出训练)        | **Student rollout**: 用 student 自己的部分 unmask 结果作为下一步输入 |
| Two Time-Scale Update (TTUR)         | 不需要 (MaskGIT 没有 score model)                          |
| 4步 timestep schedule                 | **4步 mask ratio schedule**: {1.0, 0.75, 0.5, 0.25}    |


#### 4.6 编辑感知蒸馏 (Edit-Aware CU)

编辑任务中，non-edit 区域完全不变 (exact preservation)，只 unmask edit 区域:

- 步数蒸馏只作用在 edit_mask 区域
- 非编辑区域直接 copy source tokens (0 步)
- Edit 任务天然只需 2-4 步 (edit 区域通常 <30% tokens)

#### 4.7 空间选择性 (Spatial-Adaptive Steps)

不同区域需要的步数不同:

- 高 confidence 区域: 1 步即可
- 低 confidence 区域 (复杂纹理、文字): 需要更多步
- 动态分配: 简单区域 1-2 步, 难区域 4+ 步 → 平均 <3 步

这是对 MaskGIT confidence-based unmasking 的自然延伸，也是我们架构的独特优势。

---

## 5. 支持的任务


| 任务              | 说明                    | 数据量      |
| --------------- | --------------------- | -------- |
| **T2I**         | Text-to-Image         | 14.96M   |
| **Edit**        | Instruction-guided 编辑 | 1.75M    |
| **Multi-Image** | 多参考图融合                | 314K     |
| **Canny**       | Edge → Image          | 500K     |
| **Depth**       | Depth → Image         | 500K     |
| **Coloring**    | Grayscale → Color     | 500K     |
| **Deblurring**  | Blur → Sharp          | 500K     |
| **SR**          | Low-res → High-res    | 500K     |
| **Fill**        | Inpainting            | 500K     |
| **总计**          |                       | **~20M** |


---

## 6. 数据

### 6.1 T2I (14.96M)


| 数据集              | 量     | 路径                                                   |
| ---------------- | ----- | ---------------------------------------------------- |
| art_sft_good     | 147K  | `/mnt/hdfs/weichow/maskedit/t2i/art_sft_good_*.json` |
| art_crawler_good | 33K   | 同上                                                   |
| design_good      | 40K   | 同上                                                   |
| movie_good       | 58K   | 同上                                                   |
| photograph_good  | 111K  | 同上                                                   |
| camera_good      | 8.26M | tar                                                  |
| fine_t2i         | 6.31M | tar                                                  |


### 6.2 Edit (1.75M)


| 数据集            | 量    | 路径                                 |
| -------------- | ---- | ---------------------------------- |
| omniedit       | 368K | `/mnt/hdfs/weichow/maskedit/edit/` |
| imgedit        | 708K | 同上                                 |
| gpt_image_edit | 678K | 同上                                 |


### 6.3 Control (3.0M)

canny/depth/coloring/deblurring/sr/fill 各 500K, 路径 `/mnt/hdfs/weichow/maskedit/control/`

### 6.4 Multi-Image (314K)


| 数据集                | 量    | 路径                                |
| ------------------ | ---- | --------------------------------- |
| UNO-1M (score≥3.5) | 250K | `/mnt/hdfs/weichow/maskedit/vqa/` |
| Echo-4o            | 64K  | 同上                                |


### 6.5 RL (13K)

T2I prompts 11.6K + Edit pairs 1.4K

### 6.6 数据扩充方案

#### 新增数据一览


| 数据源                                 | 类型               | 量     | License            | 用途                  |
| ----------------------------------- | ---------------- | ----- | ------------------ | ------------------- |
| **pico-banana-400K** (SFT 部分)       | Edit             | 258K  | CC BY-NC-ND 4.0 ⚠️ | Phase 2 Edit        |
| **MICo-150K** (全部)                  | Multi-Image      | 148K  | Apache 2.0 ✅       | Phase 2 Multi-Image |
| **TextAtlas5M — CoverBook**         | Scene text (T2I) | ~500K | MIT ✅              | Phase 1 T2I 文字渲染    |
| **TextAtlas5M — LongWordsSubset-A** | Scene text (T2I) | ~500K | MIT ✅              | Phase 1 T2I 长文字     |
| **TextAtlas5M — TextScenesHQ**      | Scene text (T2I) | ~500K | MIT ✅              | Phase 1 T2I 场景文字    |


**目录结构总览**:

```
/mnt/hdfs/weichow/maskedit/
├── t2i/
│   ├── art_sft_good_*.json          (现有 147K)
│   ├── camera_good_*.json + tar     (现有 8.26M)
│   ├── fine_t2i_*.json + tar        (现有 6.31M)
│   ├── textatlas_coverbook_*.json + tar      (新增 ~500K)
│   ├── textatlas_longwords_*.json + tar      (新增 ~500K)
│   └── textatlas_textscenes_*.json + tar     (新增 ~500K)
├── edit/
│   ├── gpt_image_edit_*.json        (现有 678K)
│   ├── imgedit_*.json               (现有 708K)
│   ├── omniedit_*.json              (现有 368K)
│   └── pico_banana_sft_*.json + tar (新增 258K)
├── vqa/
│   ├── uno_*.json + tar             (现有 250K)
│   ├── echo4o_*.json + tar          (现有 64K)
│   └── mico_*.json + tar            (新增 148K)
└── control/                          (不变)
```

**数据处理 pipeline**: 下载原始 → 格式转换 (字段映射 + 图片重命名) → 切分 shard → 打包 tar → 上传 HDFS

---

## 7. 竞品对比

### 7.1 架构对比


| 模型           | 参数        | 架构               | Token                      | RoPE           |
| ------------ | --------- | ---------------- | -------------------------- | -------------- |
| DreamLite    | 0.39B     | 简化 U-Net         | 连续 (8x↓)                   | N/A            |
| Z-Image      | 6B        | 纯单流 S3-DiT       | 连续                         | 3D unified     |
| Qwen-Image   | 20B       | MMDiT 60层        | 连续 (16ch)                  | MSRoPE         |
| FLUX.2       | 4-32B     | 8双流+48单流         | 连续                         | 3D             |
| Meissonic    | ~1B       | 19+38 FLUX-style | 离散 8192                    | 3D             |
| **LightMGT** | **0.37B** | **4双流+20单流**     | **离散 262144 (LFQ 18-bit)** | **3D unified** |


### 7.2 性能目标


| Benchmark  | DreamLite | LightMGT 目标 |
| ---------- | --------- | ----------- |
| GenEval    | 0.72      | ≥0.70       |
| DPG-Bench  | 85.8      | ≥80         |
| ImgEdit    | 4.11      | ≥4.0        |
| GEditBench | 6.88      | ≥6.0        |


### 7.4 DreamLite 精确训练配方 (arXiv:2603.28713)


| 维度           | 值                                    |
| ------------ | ------------------------------------ |
| Base model   | SnapGen (0.38B, CVPR 2025)           |
| Blocks       | [0, 2, 4], channels [256, 512, 896]  |
| Attention    | Multi-Query (单 KV head) + QK-RMSNorm |
| VAE          | TinyVAE 2.5M                         |
| Text encoder | Qwen3-VL-2B                          |
| T2I 数据       | 20M                                  |
| Edit 数据      | 1.74M                                |
| 蒸馏           | DMD2 → 4步, 消除 CFG                    |
| 端侧延迟         | **~460ms** (小米14, 1024px, W8A8)      |
| 蒸馏后质量        | GenEval 0.72→0.70, ImgEdit 4.11→~3.8 |


### 7.5 端侧延迟估算 (LightMGT)


|      | DreamLite | LightMGT (4步) |
| ---- | --------- | ------------- |
| 端侧延迟 | ~460ms    | ~380-540ms    |
| 推理步数 | 4         | 4 (蒸馏后)       |
| 量化   | W8A8      | W8A8          |


---

## 8. 蒸馏与加速

### 8.1 现有方法


| 方法               | 论文         | 目标        | Meissonic? |
| ---------------- | ---------- | --------- | ---------- |
| Speed-RL         | 2512.01094 | 6步 (3x)   | **✅**      |
| Di[M]O           | 2503.15457 | 1步        | ❌          |
| Soft-Di[M]O      | 2509.22925 | 1步        | ❌          |
| MIGM-Shortcut    | 2602.23996 | 4-8步      | ❌          |
| EB-Sampler       | 2505.24857 | 自适应 (免训练) | ❌          |
| Halton Scheduler | 2503.17076 | 同步数 (免训练) | ❌          |


### 8.2 创新方向

**方向 A: 编辑感知蒸馏** — 首个在 image-conditioned editing + multi-task control 下做 masked gen distillation
**方向 B: 空间选择性保持** — 少步推理中保留 confidence-based 局部编辑能力

---

## 9. 风险与缓解


| 风险                       | 缓解                                                             |
| ------------------------ | -------------------------------------------------------------- |
| GLA + MaskGIT 不 work     | Plan A: 全 softmax; Plan B: runtime 切换; Plan C: Agent Attention |
| T2I 数据不够 (14.96M vs 20M) | 质量过滤 > 增量; 补 scene text/portrait 专项                            |
| 4096 tokens 推理慢          | GLA 加速 + KV Cache 复用 + Flash Attn                              |
| 蒸馏不及 DMD2                | Speed-RL 保底 8 步                                                |


---

## 10. 参考文献


| 简称              | 论文                          | 关键借鉴                                          |
| --------------- | --------------------------- | --------------------------------------------- |
| DreamLite       | arXiv:2603.28713            | 竞品, SnapGen base                              |
| FLUX.2          | bfl.ai/blog/flux-2          | 双流:单流 1:6, SwiGLU, 无 bias, 共享 AdaLN           |
| Z-Image         | arXiv:2511.22699            | 3D Unified RoPE, Refiner, Sandwich-Norm       |
| Qwen-Image      | arXiv:2508.02324            | MSRoPE, DPO+GRPO, progressive text curriculum |
| MaskBit         | arXiv:2409.16211            | Factorized head, arccos schedule, 训练超参        |
| DiG             | CVPR 2025                   | GLA 在生成中不掉质量                                  |
| LiT             | ICCV 2025                   | Linear attn DiT, 9x 加速                        |
| eMIGM           | arXiv:2503.07197            | Time Interval CFG, exponential mask schedule  |
| OminiControl    | arXiv:2411.15098            | Edit injection                                |
| Meissonic       | github.com/viiika/Meissonic | FLUX-style MaskGIT baseline                   |
| MaskGIT         | arXiv:2202.04200            | Masked generation framework                   |
| Speed-RL        | arXiv:2512.01094            | RL step schedule, 已验证 Meissonic               |
| Di[M]O          | arXiv:2503.15457            | Token-level distribution matching             |
| Agent Attention | ECCV 2024                   | 免训插入 linear attention                         |
| SoLA-Vision     | 2026                        | Hybrid softmax/linear 层分配                     |
| SnapGen         | arXiv:2412.09619            | DreamLite base model, CVPR 2025               |


