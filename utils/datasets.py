#!/usr/bin/env python3
"""
Dataset loading utilities for VQA, VQA-spatial, and VSR datasets.
"""

import json
import re
import hashlib
import os
from pathlib import Path
from typing import List, Optional
from datasets import load_dataset


def load_vqa(split: str = "validation"):
    """
    Load VQAv2 dataset split via huggingface datasets.

    Returns HF dataset with keys: "image" (PIL.Image), "question" (str).
    """
    return load_dataset("lmms-lab/VQAv2", split=split)


def _normalize_keywords(keywords: List[str]) -> List[str]:
    return [k.strip().lower() for k in keywords if k and k.strip()]


def _compile_keywords_regex(keywords: List[str]) -> re.Pattern:
    """Compile case-insensitive regex for a list of keywords/phrases.
    Supports multi-word phrases with flexible spacing or hyphen between words.
    """
    escaped_variants: List[str] = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        parts = [re.escape(p) for p in kw.split()]
        if len(parts) == 1:
            pattern = parts[0]
        else:
            joiner = r"(?:\s+|-)"
            pattern = joiner.join(parts)
        escaped_variants.append(rf"\b{pattern}\b")
    combined = "|".join(escaped_variants)
    if not combined:
        combined = r"a^"  # match nothing
    return re.compile(combined, flags=re.IGNORECASE)


def _keywords_hash(split: str, keywords: List[str]) -> str:
    norm = _normalize_keywords(keywords)
    key = f"{split}::" + "||".join(sorted(norm))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def default_spatial_keywords() -> List[str]:
    return [
        # Basic directions and movement
        "left", "right", "front", "back", "ahead", "behind", "forward", "backward",
        "forwards", "backwards", "up", "down", "upward", "downward",
        # Corners, sides, and extremes
        "top", "bottom", "upper", "lower", "leftmost", "rightmost", "topmost", "bottommost",
        "uppermost", "lowermost", "corner", "edge", "border", "side",
        "left side", "right side", "top side", "bottom side",
        # Multi-axis quadrant phrases
        "top left", "top right", "bottom left", "bottom right",
        "upper left", "upper right", "lower left", "lower right",
        "middle left", "middle right", "center left", "center right",
        # Relative spatial relations
        "above", "over", "overhead", "atop", "on top", "on top of",
        "below", "under", "underneath", "beneath",
        "in front", "in front of", "at the front", "at the back",
        "next to", "beside", "alongside", "near", "nearby", "close to",
        "adjacent", "adjacent to", "across from", "opposite", "opposite to", "facing",
        "around", "surrounding", "encircling", "between", "in between", "among", "amid",
        "inside", "inside of", "outside", "outside of", "within",
        "to the left", "to the right", "to the left of", "to the right of",
        # Distance and extent
        "distance", "closer", "closest", "nearest", "nearer",
        "far", "farther", "farthest", "further", "furthest",
        "height", "width",
        # Orientation and axes
        "vertical", "horizontal", "diagonal", "direction", "oriented", "orientation",
        "rotated", "rotation",
        # Compass directions
        "north", "south", "east", "west",
        "north east", "north west", "south east", "south west",
        "northeast", "northwest", "southeast", "southwest",
        # Locative cues
        "position", "positioned", "located", "location", "placement", "placed",
        # Foreground/background
        "foreground", "background", "frontmost", "backmost", "background of",
    ]


def load_vqa_spatial(
    split: str = "validation",
    keywords_file: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    cache_dir: Optional[str] = ".cache/vqa_spatial_filter",
):
    """
    Load VQAv2 and filter by spatial keywords. Returns an HF dataset subset.
    Caches filtered indices for reproducibility/performance when cache_dir is provided.
    """
    base = load_vqa(split)

    if keywords_file is not None and os.path.exists(keywords_file):
        with open(keywords_file, "r") as f:
            file_keywords = [line.strip() for line in f.readlines()]
        keywords_list = file_keywords
    elif keywords is not None and len(keywords) > 0:
        keywords_list = keywords
    else:
        keywords_list = default_spatial_keywords()

    keywords_norm = _normalize_keywords(keywords_list)

    filtered_indices: Optional[List[int]] = None
    cache_used = False
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        fname = f"indices_{split}_{_keywords_hash(split, keywords_norm)}.json"
        cache_file = cache_path / fname
        if cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text())
                if payload.get("split") == split:
                    filtered_indices = list(map(int, payload.get("indices", [])))
                    cache_used = True
            except Exception:
                pass

    if filtered_indices is None:
        pattern = _compile_keywords_regex(keywords_norm)
        tmp_indices: List[int] = []
        for idx in range(len(base)):
            q = str(base[idx]["question"])  # robust to unexpected types
            if pattern.search(q):
                tmp_indices.append(idx)
        filtered_indices = tmp_indices

        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            fname = f"indices_{split}_{_keywords_hash(split, keywords_norm)}.json"
            cache_file = cache_path / fname
            payload = {
                "split": split,
                "keywords": keywords_norm,
                "count": len(filtered_indices),
                "indices": filtered_indices,
            }
            try:
                cache_file.write_text(json.dumps(payload, indent=2))
            except Exception:
                pass

    return base.select(filtered_indices)


def load_vsr(split: str = "train", only_true: bool = True):
    """
    Load the VSR dataset (cambridgeltl/vsr_random). When only_true=True, filter to label==1.
    """
    # The dataset exposes standard splits; data_files mapping is not necessary for default config
    dataset = load_dataset("cambridgeltl/vsr_random", split=split)
    if only_true:
        dataset = dataset.filter(lambda x: x.get("label", 0) == 1)
    return dataset


class VQAPairDataset:
    """Wrapper that exposes (image, prompt) pairs from a VQA HF dataset."""
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        prompt = sample["question"]
        return image, prompt


__all__ = [
    "load_vqa",
    "load_vqa_spatial", 
    "load_vsr",
    "default_spatial_keywords",
    "VQAPairDataset",
]



