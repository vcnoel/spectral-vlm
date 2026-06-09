"""
Load FigStep dataset for Phase 0.

FigStep (Gong et al.): harmful instructions rendered as typographic images.
HuggingFace dataset: "ThuCCSI/FigStep"  (check HF Hub for the exact card).

Each example yields:
    {
        'example_id': str,
        'image'     : PIL.Image,
        'prompt'    : str,       # benign framing shown to the model
        'category'  : str,       # FigStep safety category
        'split'     : 'attacked',
        'source'    : 'figstep',
    }

A matched clean example is also returned with a neutral (blank) image for
the paired H1 contrast.  The model receives the SAME prompt but with a
non-injected image so any spectral difference is due to the attack, not the text.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image, ImageDraw
from datasets import load_dataset

logger = logging.getLogger(__name__)

# Default prompt wrapper — presents the image as "instructions"
# without adding extra harmful content from the text side.
_DEFAULT_PROMPT = "Please follow the instructions shown in the image."

_BLANK_COLOR = (255, 255, 255)   # white blank image
_DEFAULT_SIZE = (336, 336)       # match typical SigLIP input resolution


def _make_blank_image(size: tuple[int, int] = _DEFAULT_SIZE) -> Image.Image:
    img = Image.new("RGB", size, color=_BLANK_COLOR)
    return img


def load_figstep(
    n_samples: int = 280,
    split: str = "train",
    hf_dataset_name: str = "ThuCCSI/FigStep",
    cache_dir: Optional[str] = None,
    prompt_template: str = _DEFAULT_PROMPT,
    seed: int = 42,
) -> list[dict]:
    """
    Returns a list of dicts, each containing an 'attacked' example and its
    paired 'clean' example (blank image, same prompt).

    Each entry in the returned list:
        {
            'example_id': str,
            'attacked': { 'image': PIL, 'prompt': str, 'split': 'attacked', ... },
            'clean'   : { 'image': PIL, 'prompt': str, 'split': 'clean',    ... },
            'category': str,
            'source'  : 'figstep',
        }
    """
    logger.info("Loading FigStep from HF hub ('%s', split='%s')", hf_dataset_name, split)

    try:
        ds = load_dataset(hf_dataset_name, split=split, cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(
            f"Could not load FigStep from '{hf_dataset_name}'. "
            "Check the HF dataset card for the correct name and that you have access. "
            f"Original error: {e}"
        ) from e

    # Shuffle deterministically and take n_samples.
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    logger.info("FigStep: %d examples selected", len(ds))

    examples = []
    for i, row in enumerate(ds):
        eid = f"figstep_{i:04d}"
        # FigStep images are PIL-compatible; the column name varies by version.
        image_col = _find_image_col(row)
        img: Image.Image = row[image_col]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        category = row.get("category", row.get("class", "unknown"))
        prompt   = row.get("question", row.get("prompt", prompt_template))

        examples.append({
            "example_id": eid,
            "attacked": {
                "image" : img.convert("RGB"),
                "prompt": prompt,
                "split" : "attacked",
                "source": "figstep",
            },
            "clean": {
                "image" : _make_blank_image(img.size),
                "prompt": prompt,
                "split" : "clean",
                "source": "figstep",
            },
            "category": category,
            "source"  : "figstep",
        })

    return examples


def _find_image_col(row: dict) -> str:
    for col in ("image", "img", "figure", "screenshot"):
        if col in row:
            return col
    raise KeyError(f"Cannot find image column in FigStep row. Keys: {list(row.keys())}")


def iter_flat(examples: list[dict]) -> Iterator[dict]:
    """Flatten paired examples into individual dicts with 'split' field."""
    for pair in examples:
        yield {**pair["attacked"], "example_id": pair["example_id"], "category": pair["category"]}
        yield {**pair["clean"],    "example_id": pair["example_id"] + "_clean", "category": pair["category"]}
