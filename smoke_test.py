"""
M0 smoke test: load Gemma 4 4B in bf16 with eager attention,
run a single forward pass with one image, verify the 256-token visual span,
and confirm all spectral metrics are finite.

Run with:
    conda activate gemma_spectral
    python smoke_test.py [--model google/gemma-4-4b-it]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

# Add project root to path.
sys.path.insert(0, str(Path(__file__).parent))

from src.extract import ExtractionConfig, VLMExtractor
from src.spectral import compute_spectral_trajectory, METRIC_NAMES, VIEW_NAMES, AGG_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("smoke_test")


def make_test_image(size: tuple[int, int] = (336, 336)) -> Image.Image:
    """Solid red image — just enough to trigger image processing."""
    return Image.new("RGB", size, color=(180, 50, 50))


def main(model_id: str) -> None:
    logger.info("=== spectral-vlm smoke test ===")
    logger.info("Model: %s", model_id)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("VRAM total: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    cfg = ExtractionConfig(
        model_id=model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
        max_new_tokens=32,
        analyze_token_offset=0,
    )

    extractor = VLMExtractor(cfg)
    logger.info("Loading model...")
    extractor.load()
    logger.info("Model loaded. image_token_index=%d  expected_visual_tokens=%s",
                extractor._image_token_index, extractor._expected_visual_tokens)

    # Run forward pass on a single test image.
    image  = make_test_image()
    prompt = "Describe what you see in the image."

    logger.info("Running forward pass...")
    result = extractor.extract_attentions(image, prompt)

    logger.info("Generated text: %r", result.generated_text[:120])
    logger.info("Visual span: [%d, %d)  (%d tokens)",
                result.visual_start, result.visual_end,
                result.visual_end - result.visual_start)
    logger.info("Sequence length: %d tokens", result.seq_len)
    logger.info("Number of layers: %d", len(result.attentions))

    # M0 assertion: visual span must be non-trivial.
    n_visual = result.visual_end - result.visual_start
    assert n_visual >= 64, f"Expected ≥64 visual tokens, got {n_visual}"
    logger.info("M0 PASS: visual span = %d tokens", n_visual)

    # M1: compute all spectral metrics and verify finiteness.
    logger.info("Computing spectral trajectory...")
    trajectory = compute_spectral_trajectory(result)
    n_layers = len(trajectory)
    logger.info("Trajectory computed: %d layers", n_layers)

    # Check a sample of layers for NaN/Inf.
    n_nan = 0
    n_inf = 0
    import numpy as np
    for li, lm in enumerate(trajectory):
        if np.any(np.isnan(lm.data)):
            logger.warning("Layer %d: NaN in metrics", li)
            n_nan += 1
        if np.any(np.isinf(lm.data)):
            logger.warning("Layer %d: Inf in metrics", li)
            n_inf += 1

    # Print a summary table for the first and last layer.
    for li in [0, n_layers // 2, n_layers - 1]:
        lm = trajectory[li]
        logger.info("Layer %2d metrics:", li)
        for mi, m in enumerate(METRIC_NAMES):
            for vi, v in enumerate(VIEW_NAMES):
                vals = [f"{lm.data[mi, vi, ai]:.4f}" for ai in range(len(AGG_NAMES))]
                logger.info("  %-12s %-5s  mean=%s  max=%s", m, v, vals[0], vals[1])

    if n_nan == 0 and n_inf == 0:
        logger.info("M1 PASS: all metrics finite across %d layers", n_layers)
    else:
        logger.error("M1 FAIL: %d NaN layers, %d Inf layers", n_nan, n_inf)
        sys.exit(1)

    logger.info("=== Smoke test PASSED ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-4b-it",
                        help="HuggingFace model ID")
    args = parser.parse_args()
    main(args.model)
