"""Data loading for LightMGT training.

Online encoding: images are loaded as PIL, VQ-encoded on-the-fly in the training loop.
NO offline encoding — data passes only ~1 epoch, pre-encoding is wasteful.

Multi-scale training with aspect-ratio bucketing:
- All bucket dimensions are multiples of 16 (VQ 16x downsample factor)
- 3 resolution tiers: 256-center, 512-center, 1024-center
- Progressive training: start 256, then 512, then 1024

Reference: MaskEdit/src/dataset_utils.py
"""

import io
import json
import logging
import math
import os
import random
import struct
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils.data as data
from PIL import Image, ImageOps
from torchvision import transforms

logger = logging.getLogger(__name__)


# ===========================================================================
# Multi-Scale Bucket Definitions
# All dimensions must be multiples of 16 (VQ 16x downsample factor)
# ===========================================================================

ASPECT_RATIO_128 = [
    (128, 128),    # 1:1
    (144, 112),    # ~4:3
    (112, 144),
    (160, 96),     # ~5:3
    (96, 160),
    (176, 80),     # ~2.2:1
    (80, 176),
]

ASPECT_RATIO_256 = [
    (256, 256),    # 1:1
    (272, 240),    # ~1.13:1
    (240, 272),
    (288, 224),    # ~4:3
    (224, 288),
    (304, 208),    # ~3:2
    (208, 304),
    (320, 192),    # ~5:3
    (192, 320),
    (336, 176),    # ~2:1
    (176, 336),
    (368, 160),    # ~2.3:1
    (160, 368),
]

ASPECT_RATIO_512 = [
    (512, 512),    # 1:1
    (544, 480),    # ~1.13:1
    (480, 544),
    (576, 448),    # ~4:3
    (448, 576),
    (608, 416),    # ~3:2
    (416, 608),
    (640, 384),    # ~5:3
    (384, 640),
    (672, 352),    # ~2:1
    (352, 672),
    (736, 320),    # ~2.3:1
    (320, 736),
]

ASPECT_RATIO_1024 = [
    (1024, 1024),  # 1:1
    (1088, 960),   # ~1.13:1
    (960, 1088),
    (1152, 896),   # 4:3
    (896, 1152),
    (1216, 832),   # 3:2
    (832, 1216),
    (1280, 800),   # 16:10
    (800, 1280),
    (1344, 768),   # ~16:9
    (768, 1344),
    (1408, 704),   # 2:1
    (704, 1408),
    (1472, 672),   # ~2.2:1
    (672, 1472),
    (1536, 640),   # ~2.4:1
    (640, 1536),
]

BUCKET_REGISTRY = {
    128: ASPECT_RATIO_128,
    256: ASPECT_RATIO_256,
    512: ASPECT_RATIO_512,
    1024: ASPECT_RATIO_1024,
}


def get_buckets(center_resolution: int = 1024) -> List[Tuple[int, int]]:
    """Get aspect ratio buckets for a given center resolution.

    Args:
        center_resolution: Target center resolution (256, 512, or 1024).

    Returns:
        List of (height, width) bucket tuples, all multiples of 16.
    """
    if center_resolution in BUCKET_REGISTRY:
        return BUCKET_REGISTRY[center_resolution]
    raise ValueError(f"No buckets defined for resolution {center_resolution}. "
                     f"Available: {list(BUCKET_REGISTRY.keys())}")


def find_nearest_bucket(
    orig_height: int,
    orig_width: int,
    buckets: List[Tuple[int, int]],
) -> Tuple[int, int]:
    """Find the nearest bucket using log-ratio distance (symmetric for aspect ratios)."""
    aspect_ratio = orig_width / max(orig_height, 1)
    best_bucket = buckets[0]
    min_metric = float("inf")
    for bh, bw in buckets:
        bucket_ar = bw / bh
        metric = abs(math.log(max(aspect_ratio, 1e-6)) - math.log(bucket_ar))
        if metric < min_metric:
            min_metric = metric
            best_bucket = (bh, bw)
    return best_bucket


def validate_buckets():
    """Verify all bucket dimensions are multiples of 16."""
    for res, buckets in BUCKET_REGISTRY.items():
        for h, w in buckets:
            assert h % 16 == 0, f"Bucket ({h}, {w}) in {res}-tier: height not multiple of 16"
            assert w % 16 == 0, f"Bucket ({h}, {w}) in {res}-tier: width not multiple of 16"
    logger.info("All bucket dimensions validated: multiples of 16 ✓")


# Validate on import
validate_buckets()


# ===========================================================================
# Image Processing
# ===========================================================================


def process_image(
    image: Image.Image,
    size: Union[int, Tuple[int, int]],
    norm: bool = True,
) -> Dict[str, torch.Tensor]:
    """Process PIL image to target size.

    Args:
        image: PIL Image.
        size: int for square, or (height, width) tuple.
        norm: Whether to normalize to [-1, 1] (for VQ encoder).

    Returns:
        Dict with "image" tensor [3, H, W] and "original_size" [2].
    """
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    orig_height, orig_width = image.height, image.width

    if isinstance(size, (list, tuple)):
        target_height, target_width = size
    else:
        target_height = target_width = size

    # Resize: shortest-side matching to cover target area
    scale = max(target_height / orig_height, target_width / orig_width)
    new_height = max(target_height, int(orig_height * scale + 0.5))
    new_width = max(target_width, int(orig_width * scale + 0.5))
    image = transforms.Resize(
        (new_height, new_width),
        interpolation=transforms.InterpolationMode.BILINEAR,
    )(image)

    # Random crop
    c_top, c_left, _, _ = transforms.RandomCrop.get_params(
        image, output_size=(target_height, target_width)
    )
    image = transforms.functional.crop(image, c_top, c_left, target_height, target_width)
    image = transforms.ToTensor()(image)

    if norm:
        image = image * 2.0 - 1.0

    return {
        "image": image,
        "original_size": torch.tensor([orig_height, orig_width]),
    }


# ===========================================================================
# Tar Image Reader (O(1) random access)
# ===========================================================================


class TarImageReader:
    """Read images from tar archives with O(1) random access.

    Builds an in-memory index: filename -> (tar_path, offset, size).
    Thread-safe: each read opens its own file handle.

    Compatible with MaskEdit's cache format: cache key = md5(tar_pattern).
    Reuses MaskEdit's persistent cache at /mnt/bn/search-auto-eval/zhouwei/maskedit_cache/tar_index/.

    Usage:
        # Single pattern (reuses MaskEdit cache if available)
        reader = TarImageReader("/mnt/hdfs/weichow/maskedit/t2i/camera_good_images_*.tar")

        # Multiple patterns (loads each cache separately, merges)
        reader = TarImageReader([
            "/mnt/hdfs/weichow/maskedit/t2i/camera_good_images_*.tar",
            "/mnt/hdfs/weichow/maskedit/t2i/fine_t2i_images_*.tar",
        ])
    """

    # Persistent cache locations (shared, survives reboots)
    # BN1: T2I + Edit indexes, BN2: T2I-PT + others
    DEFAULT_CACHE_DIRS = [
        "/mnt/bn/search-auto-eval/zhouwei/maskedit_cache/tar_index",
        "/mnt/bn/search-auto-eval-v2/zhouwei/maskedit_cache/tar_index",
    ]

    def __init__(self, tar_patterns: Union[str, List[str]], cache_dir: Optional[str] = None):
        if isinstance(tar_patterns, str):
            tar_patterns = [tar_patterns]

        self.cache_dirs = [cache_dir] if cache_dir else self.DEFAULT_CACHE_DIRS
        self.index: Dict[str, Tuple[str, int, int]] = {}

        # Load each pattern separately (compatible with MaskEdit per-pattern caching)
        for pattern in tar_patterns:
            self._load_pattern(pattern)

        logger.info(f"TarImageReader: {len(self.index)} total images from {len(tar_patterns)} pattern(s)")

    def _load_pattern(self, pattern: str):
        """Load index for a single tar pattern, using MaskEdit-compatible cache key.

        Searches multiple cache directories (BN1 for T2I/Edit, BN2 for T2I-PT/others).
        """
        import hashlib
        import pickle

        # MaskEdit cache key: md5 of the pattern string itself
        cache_key = hashlib.md5(pattern.encode()).hexdigest()

        # Search all cache dirs for existing index
        for cache_dir in self.cache_dirs:
            cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        cached = pickle.load(f)
                    self.index.update(cached)
                    logger.info(f"TarImageReader: loaded {len(cached)} entries from cache ({cache_path})")
                    return
                except Exception as e:
                    logger.warning(f"TarImageReader: cache load failed ({cache_path}): {e}")

        # Cache miss — build index from tar files
        import glob as glob_module
        tar_files = sorted(glob_module.glob(pattern))
        if not tar_files:
            logger.warning(f"TarImageReader: no tar files match pattern: {pattern}")
            return

        logger.info(f"TarImageReader: building index from {len(tar_files)} tars ({pattern})...")
        new_entries = {}
        for tar_path in tar_files:
            try:
                with tarfile.open(tar_path) as tf:
                    for member in tf.getmembers():
                        if member.isfile():
                            new_entries[member.name] = (tar_path, member.offset_data, member.size)
            except Exception as e:
                logger.warning(f"TarImageReader: failed to index {tar_path}: {e}")

        # Save cache to first writable cache dir
        save_dir = self.cache_dirs[0]
        save_path = os.path.join(save_dir, f"{cache_key}.pkl")
        os.makedirs(save_dir, exist_ok=True)
        try:
            with open(save_path, "wb") as f:
                pickle.dump(new_entries, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"TarImageReader: cached {len(new_entries)} entries to {save_path}")
        except Exception as e:
            logger.warning(f"TarImageReader: cache write failed: {e}")

        self.index.update(new_entries)

    def read(self, name: str) -> Image.Image:
        """Read an image by name from the indexed tar archives."""
        if name not in self.index:
            raise FileNotFoundError(f"Image '{name}' not found in tar index")
        tar_path, offset, size = self.index[name]
        with open(tar_path, "rb") as f:
            f.seek(offset)
            data = f.read(size)
        return Image.open(io.BytesIO(data))


# ===========================================================================
# JSON Data Loader
# ===========================================================================


def load_json_or_jsonl(filepath: str) -> list:
    """Load data from JSON array or JSONL file."""
    with open(filepath) as f:
        content = f.read().strip()
    if not content:
        return []
    if content[0] == "[":
        return json.loads(content)
    return [json.loads(line) for line in content.split("\n") if line.strip()]


# ===========================================================================
# T2I Dataset (Online Encoding)
# ===========================================================================


class LightMGTT2IDataset(data.Dataset):
    """T2I dataset with online VQ encoding and multi-scale bucketing.

    Reads JSON metadata files: [{uid, caption, image, height, width}, ...]
    Images are loaded as PIL, resized to bucket size, and returned for
    on-the-fly VQ encoding in the training loop.

    Args:
        json_dir: Directory containing JSON shard files.
        tokenizer: Text tokenizer (Qwen3.5-0.8B).
        center_resolution: Center resolution for bucket selection (256/512/1024).
        multi_scale: Whether to use aspect-ratio bucketing.
        base_dir: Base directory for resolving image paths.
        tar_pattern: Glob pattern for tar archives containing images.
        text_max_length: Max text token length.
    """

    def __init__(
        self,
        json_dir: Union[str, List[str]],
        tokenizer,
        center_resolution: int = 256,
        multi_scale: bool = True,
        base_dir: Optional[str] = None,
        tar_pattern: Optional[Union[str, List[str]]] = None,
        text_max_length: int = 256,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.center_resolution = center_resolution
        self.multi_scale = multi_scale
        self.text_max_length = text_max_length
        self.buckets = get_buckets(center_resolution)

        # Support multiple json dirs
        if isinstance(json_dir, str):
            json_dirs = [json_dir]
        else:
            json_dirs = list(json_dir)
        self.base_dir = base_dir or json_dirs[0]

        # Initialize tar reader BEFORE loading shards (needed for validation)
        self.tar_reader = None
        if tar_pattern:
            if isinstance(tar_pattern, str) and "," in tar_pattern:
                tar_pattern = [p.strip() for p in tar_pattern.split(",")]
            self.tar_reader = TarImageReader(tar_pattern)
            logger.info(f"TarImageReader ready: {len(self.tar_reader.index)} images indexed")

        # Load ALL JSON shards from all dirs unconditionally (like MaskEdit).
        # No shard validation — __getitem__ retry handles missing images.
        # Only filter: skip non-dict entries (e.g. .ckpt_*.json that contain plain UIDs).
        self.samples = []
        total_json_files = 0
        non_dict_shards = 0
        for jdir in json_dirs:
            json_files = sorted(Path(jdir).glob("*.json"))
            total_json_files += len(json_files)
            for jf in json_files:
                try:
                    entries = load_json_or_jsonl(str(jf))
                    valid = [e for e in entries if isinstance(e, dict) and ("image" in e or "img" in e or "image_path" in e)]
                    if not valid:
                        non_dict_shards += 1
                        continue
                    self.samples.extend(valid)
                except Exception as e:
                    logger.warning(f"Failed to load {jf}: {e}")

        if non_dict_shards > 0:
            logger.info(f"Skipped {non_dict_shards} non-dict shards (e.g. .ckpt_*.json)")

        logger.info(f"LightMGTT2IDataset: {len(self.samples)} samples from {total_json_files} shards "
                     f"across {len(json_dirs)} dir(s), "
                     f"center={center_resolution}px, multi_scale={multi_scale}"
                     + (f", tar_reader={len(self.tar_reader.index)} images" if self.tar_reader else ""))

    def __len__(self) -> int:
        return len(self.samples)

    def get_image_size(self, idx: int) -> Tuple[int, int]:
        """Get original image dimensions (for bucket pre-assignment)."""
        s = self.samples[idx]
        return s.get("height", self.center_resolution), s.get("width", self.center_resolution)

    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from tar archive or disk (matching MaskEdit logic).

        When tar_reader is present: tar-only, no disk fallback (HDFS is too slow).
        When no tar_reader: read from disk.
        """
        if self.tar_reader:
            if image_path in self.tar_reader.index:
                return self.tar_reader.read(image_path).convert("RGB")
            # Try common path normalizations (JSON may use a different prefix
            # than tar member names, e.g. "images/cc12m/x.jpg" vs "cc12m/x.jpg")
            for alt in (
                image_path.split("/", 1)[-1] if "/" in image_path else None,
                os.path.basename(image_path),
            ):
                if alt and alt in self.tar_reader.index:
                    return self.tar_reader.read(alt).convert("RGB")
            raise FileNotFoundError(f"Not in tar index: {image_path}")

        # No tar reader — read from disk
        if os.path.isabs(image_path):
            return Image.open(image_path).convert("RGB")
        full_path = os.path.join(self.base_dir, image_path)
        return Image.open(full_path).convert("RGB")

    def _get_target_size(self, idx: int) -> Tuple[int, int]:
        """Get target size from metadata (consistent with BucketBatchSampler)."""
        if self.multi_scale:
            h, w = self.get_image_size(idx)
            return find_nearest_bucket(h, w, self.buckets)
        return (self.center_resolution, self.center_resolution)

    def __getitem__(self, idx: int) -> dict:
        # Fix target_size from the ORIGINAL idx — retries change the image but NOT the bucket.
        # This matches BucketBatchSampler which grouped this idx into a specific bucket.
        target_size = self._get_target_size(idx)

        last_err = None
        for retry in range(10):
            try:
                sample = self.samples[idx]
                image_path = sample.get("image", sample.get("img", sample.get("image_path", "")))
                image = self._load_image(image_path)
                rv = process_image(image, target_size, norm=True)

                caption = sample.get("caption", sample.get("text", sample.get("instruction", "")))
                tokens = self.tokenizer(
                    caption,
                    padding="max_length",
                    max_length=self.text_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                rv["prompt_input_ids"] = tokens.input_ids.squeeze(0)
                rv["prompt_attention_mask"] = tokens.attention_mask.squeeze(0)
                rv["target_size"] = torch.tensor([target_size[0], target_size[1]])
                rv["task_type"] = "t2i"
                return rv
            except Exception as e:
                last_err = e
                self._record_load_error(idx, image_path if 'image_path' in dir() else "?", e)
                idx = random.randint(0, len(self) - 1)

        raise RuntimeError(f"Failed to load any image after 10 retries. Last error: {last_err}")

    def _record_load_error(self, idx: int, image_path: str, error: Exception):
        """Record data load errors for offline repair.

        Emits a WARNING log per unique (error_type, path_prefix) pair so
        errors are visible in the training log without flooding. Accumulates
        all failed paths in self._load_errors for post-training dump.
        """
        if not hasattr(self, "_load_errors"):
            self._load_errors = []
            self._load_error_keys = set()

        self._load_errors.append({
            "idx": idx,
            "path": image_path,
            "error": str(error),
        })

        # Deduplicate warning by (error class, path prefix up to second /)
        parts = image_path.split("/")
        prefix = "/".join(parts[:4]) if len(parts) >= 4 else image_path
        key = (type(error).__name__, prefix)
        if key not in self._load_error_keys:
            self._load_error_keys.add(key)
            logger.warning(
                f"Load error idx={idx}: {error} "
                f"(further errors from {prefix}/* suppressed, "
                f"{len(self._load_errors)} total errors so far)"
            )

    def get_load_errors(self) -> list:
        """Return accumulated load errors for offline data repair."""
        return getattr(self, "_load_errors", [])


# ===========================================================================
# BucketBatchSampler
# ===========================================================================


class BucketBatchSampler:
    """Groups samples by aspect ratio bucket for efficient batching.

    Pre-scans image dimensions, assigns each sample to the nearest bucket,
    and yields batches where all samples share the same (H, W).

    Args:
        dataset: Must implement get_image_size(idx) -> (h, w).
        batch_size: Samples per batch.
        buckets: List of (H, W) bucket tuples.
        drop_last: Whether to drop incomplete batches.
        rank: Current process rank (for distributed training).
        world_size: Total number of processes.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        buckets: List[Tuple[int, int]],
        drop_last: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size

        # Pre-assign each sample to a bucket
        import hashlib
        import pickle

        cache_key = hashlib.md5(
            f"{len(dataset)}_{batch_size}_{len(buckets)}".encode()
        ).hexdigest()
        cache_path = os.path.join("/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/cache/bucket_cache", f"{cache_key}.pkl")

        self.bucket_indices = None
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    self.bucket_indices = pickle.load(f)
                logger.info(f"BucketBatchSampler: loaded cached assignments ({len(dataset)} samples)")
            except Exception:
                self.bucket_indices = None

        if self.bucket_indices is None:
            self.bucket_indices = defaultdict(list)
            for idx in range(len(dataset)):
                h, w = dataset.get_image_size(idx)
                bucket = find_nearest_bucket(h, w, buckets)
                self.bucket_indices[bucket].append(idx)

            os.makedirs("/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/cache/bucket_cache", exist_ok=True)
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(dict(self.bucket_indices), f)
            except Exception:
                pass
            logger.info(f"BucketBatchSampler: assigned {len(dataset)} samples to {len(self.bucket_indices)} buckets")

        self.epoch = 0

        # Log distribution
        for bucket, indices in sorted(self.bucket_indices.items()):
            logger.info(f"  Bucket {bucket}: {len(indices)} samples")

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling (consistent across DDP ranks)."""
        self.epoch = epoch

    def __iter__(self):
        # Use a dedicated RNG seeded with epoch so all ranks produce the same
        # global batch ordering, then the round-robin split gives non-overlapping batches.
        rng = random.Random(42 + self.epoch)

        all_batches = []
        for bucket, indices in self.bucket_indices.items():
            indices_copy = list(indices)
            rng.shuffle(indices_copy)
            for i in range(0, len(indices_copy), self.batch_size):
                batch = indices_copy[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        rng.shuffle(all_batches)

        # Distribute across ranks
        if self.world_size > 1:
            all_batches = all_batches[self.rank :: self.world_size]

        yield from all_batches

    def __len__(self) -> int:
        total = 0
        for indices in self.bucket_indices.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        if self.world_size > 1:
            total = total // self.world_size
        return total


# ===========================================================================
# Collate Function
# ===========================================================================


def collate_fn(batch: List[dict]) -> dict:
    """Collate function for multi-scale batches.

    All images in a batch have the same (H, W) thanks to BucketBatchSampler.
    """
    images = torch.stack([b["image"] for b in batch])
    prompt_ids = torch.stack([b["prompt_input_ids"] for b in batch])
    prompt_mask = torch.stack([b["prompt_attention_mask"] for b in batch])
    target_size = batch[0]["target_size"]  # Same for entire batch

    return {
        "image": images,                     # [B, 3, H, W], range [-1, 1]
        "prompt_input_ids": prompt_ids,      # [B, T]
        "prompt_attention_mask": prompt_mask, # [B, T]
        "target_size": target_size,          # [2] (H, W)
    }


# ===========================================================================
# Builder
# ===========================================================================


def build_dataloader(
    json_dir: Union[str, List[str]],
    tokenizer,
    center_resolution: int = 256,
    multi_scale: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
    tar_pattern: Optional[Union[str, List[str]]] = None,
    text_max_length: int = 256,
    rank: int = 0,
    world_size: int = 1,
) -> data.DataLoader:
    """Build a DataLoader with multi-scale bucketing for T2I training.

    Args:
        json_dir: Path(s) to directory(ies) with JSON shards.
        tokenizer: Text tokenizer.
        center_resolution: 256, 512, or 1024.
        multi_scale: Enable aspect-ratio bucketing.
        batch_size: Per-GPU batch size.
        num_workers: DataLoader workers.
        tar_pattern: Glob pattern(s) for image tar archives.
        text_max_length: Max text sequence length.
        rank: DDP rank.
        world_size: DDP world size.
    """
    dataset = LightMGTT2IDataset(
        json_dir=json_dir,
        tokenizer=tokenizer,
        center_resolution=center_resolution,
        multi_scale=multi_scale,
        tar_pattern=tar_pattern,
        text_max_length=text_max_length,
    )

    buckets = get_buckets(center_resolution)

    if multi_scale:
        sampler = BucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            buckets=buckets,
            drop_last=True,
            rank=rank,
            world_size=world_size,
        )
        return data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        return data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
