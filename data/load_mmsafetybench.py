"""
Load MM-SafetyBench (Liu et al., ECCV 2024) — SD+Typography subset.

HuggingFace dataset: "liuhaotian/MM-SafetyBench"  (verify on HF Hub).
Subset of interest: SD_TYPO (Stable-Diffusion + Typography) — this is the
subset with ASR jumping from ~5% to ~77%, i.e. the strongest attack signal.

Each example yields a paired dict:
    {
        'example_id': str,
        'attacked': { 'image': PIL (SD+typo), 'prompt': str, 'split': 'attacked', ... },
        'clean'   : { 'image': PIL (SD only),  'prompt': str, 'split': 'clean',    ... },
        'scenario': str,   # one of 13 safety scenarios
        'source'  : 'mmsafetybench',
    }

The 'clean' image is the pure SD image without the typographic overlay, giving
us a within-source paired control (same visual scene, no text injection).
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from PIL import Image
from datasets import load_dataset

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = "Please follow the instructions shown in the image."

_SD_TYPO_SUBSET_NAMES = [
    "SD_TYPO", "SD+TYPO", "sd_typo", "sdtypo",   # possible HF config names
]
_SD_ONLY_SUBSET_NAMES = [
    "SD", "sd",
]


def load_mmsafetybench(
    n_samples: int = 280,
    hf_dataset_name: str = "liuhaotian/MM-SafetyBench",
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> list[dict]:
    """
    Returns paired list of dicts (attacked SD+TYPO, clean SD).
    """
    logger.info("Loading MM-SafetyBench from '%s'", hf_dataset_name)

    # Try loading SD+TYPO subset.
    ds_typo = _try_load_subset(hf_dataset_name, _SD_TYPO_SUBSET_NAMES, cache_dir)
    # Try loading SD-only for clean pairing.
    ds_sd   = _try_load_subset(hf_dataset_name, _SD_ONLY_SUBSET_NAMES, cache_dir)

    if ds_typo is None:
        raise RuntimeError(
            "Could not load SD+TYPO subset from MM-SafetyBench. "
            "Check the HF dataset card for the correct config name."
        )

    ds_typo = ds_typo.shuffle(seed=seed).select(range(min(n_samples, len(ds_typo))))
    logger.info("MM-SafetyBench SD+TYPO: %d examples", len(ds_typo))

    # Build lookup for SD-only images by question_id if available.
    sd_lookup: dict = {}
    if ds_sd is not None:
        for row in ds_sd:
            key = row.get("question_id", row.get("id", None))
            if key is not None:
                sd_lookup[str(key)] = row

    examples = []
    for i, row in enumerate(ds_typo):
        eid = f"mmsb_{i:04d}"
        img_typo = _extract_image(row, ["image_sd_typo", "image", "img"])
        scenario = row.get("scenario", row.get("category", row.get("class", "unknown")))
        prompt   = row.get("question", row.get("prompt", _DEFAULT_PROMPT))
        qid      = str(row.get("question_id", row.get("id", i)))

        # Try to get the paired SD-only image.
        if qid in sd_lookup:
            img_clean = _extract_image(sd_lookup[qid], ["image", "img"])
        else:
            # Fall back: use the SD+TYPO image at a lower resolution as proxy.
            # Not ideal, but better than blank — logs a warning.
            logger.warning("No SD-only pair found for example %s; using blank.", qid)
            img_clean = Image.new("RGB", img_typo.size, (255, 255, 255))

        examples.append({
            "example_id": eid,
            "attacked": {
                "image" : img_typo.convert("RGB"),
                "prompt": prompt,
                "split" : "attacked",
                "source": "mmsafetybench",
            },
            "clean": {
                "image" : img_clean.convert("RGB"),
                "prompt": prompt,
                "split" : "clean",
                "source": "mmsafetybench",
            },
            "scenario": scenario,
            "source"  : "mmsafetybench",
        })

    return examples


def _try_load_subset(dataset_name: str, candidates: list[str], cache_dir):
    for name in candidates:
        try:
            ds = load_dataset(dataset_name, name=name, split="test", cache_dir=cache_dir)
            logger.info("Loaded subset '%s' from %s", name, dataset_name)
            return ds
        except Exception:
            try:
                ds = load_dataset(dataset_name, name=name, split="train", cache_dir=cache_dir)
                logger.info("Loaded subset '%s' (train split) from %s", name, dataset_name)
                return ds
            except Exception:
                continue
    return None


def _extract_image(row: dict, candidates: list[str]) -> Image.Image:
    for col in candidates:
        if col in row:
            img = row[col]
            if isinstance(img, Image.Image):
                return img
            try:
                return Image.fromarray(img)
            except Exception:
                continue
    raise KeyError(f"No image column found. Available: {list(row.keys())}")


def iter_flat(examples: list[dict]) -> Iterator[dict]:
    for pair in examples:
        yield {**pair["attacked"], "example_id": pair["example_id"], "scenario": pair["scenario"]}
        yield {**pair["clean"],    "example_id": pair["example_id"] + "_clean", "scenario": pair["scenario"]}
