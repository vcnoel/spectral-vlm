"""
Load JailBreakV-28K (Luo et al.) miniset — Phase 2 (transferability).

HuggingFace: "EdinburghNLP/JailBreakV_28K"  (verify on HF Hub).

Each example yields:
    {
        'example_id': str,
        'image'     : PIL.Image,
        'prompt'    : str,
        'jailbreak_type': str,   # text / image / combined
        'source_model'  : str,   # model the jailbreak was crafted for
        'split'     : 'attacked',
        'source'    : 'jailbreakv',
    }
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from PIL import Image
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_jailbreakv(
    n_samples: int = 280,
    hf_dataset_name: str = "EdinburghNLP/JailBreakV_28K",
    cache_dir: Optional[str] = None,
    seed: int = 42,
    image_only: bool = True,   # restrict to image-based jailbreaks for VLM relevance
) -> list[dict]:
    """
    Returns a list of example dicts from the JailBreakV-28K miniset.
    """
    logger.info("Loading JailBreakV-28K from '%s'", hf_dataset_name)

    try:
        ds = load_dataset(hf_dataset_name, split="test", cache_dir=cache_dir)
    except Exception:
        try:
            ds = load_dataset(hf_dataset_name, split="train", cache_dir=cache_dir)
        except Exception as e:
            raise RuntimeError(
                f"Could not load JailBreakV-28K from '{hf_dataset_name}'. "
                f"Original error: {e}"
            ) from e

    if image_only:
        # Keep only examples that have a visual attack component.
        ds = ds.filter(lambda row: _has_image(row))
        logger.info("After image-only filter: %d examples", len(ds))

    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    logger.info("JailBreakV-28K: %d examples selected", len(ds))

    examples = []
    for i, row in enumerate(ds):
        eid = f"jbv_{i:04d}"
        img  = _extract_image(row)
        prompt = row.get("query", row.get("question", row.get("prompt", "")))
        jb_type = row.get("jailbreak_type", row.get("type", "unknown"))
        src_model = row.get("source_model", row.get("model", "unknown"))

        examples.append({
            "example_id"    : eid,
            "image"         : img.convert("RGB"),
            "prompt"        : prompt,
            "jailbreak_type": jb_type,
            "source_model"  : src_model,
            "split"         : "attacked",
            "source"        : "jailbreakv",
        })

    return examples


def _has_image(row: dict) -> bool:
    for col in ("image", "img", "figure"):
        if col in row and row[col] is not None:
            return True
    return False


def _extract_image(row: dict) -> Image.Image:
    for col in ("image", "img", "figure"):
        if col in row and row[col] is not None:
            img = row[col]
            if isinstance(img, Image.Image):
                return img
            try:
                return Image.fromarray(img)
            except Exception:
                continue
    # No image found — return blank.
    logger.warning("No image found in JailBreakV row; using blank.")
    return Image.new("RGB", (336, 336), (255, 255, 255))


def iter_flat(examples: list[dict]) -> Iterator[dict]:
    for ex in examples:
        yield ex
