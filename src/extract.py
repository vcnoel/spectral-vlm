"""
VLM attention extractor for spectral-vlm.

Loads a vision-language model with eager attention (so weights are materialized),
finds the visual-token span in each input, runs a forward pass up to the first
generated token, and returns per-layer attention tensors ready for spectral_trust.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    model_id: str = "google/gemma-4-4b-it"
    dtype: torch.dtype = torch.bfloat16
    device_map: str = "cuda"
    max_new_tokens: int = 256
    # Token index (0-based within the generated sequence) at which to capture attentions.
    analyze_token_offset: int = 0
    # Override image token index if auto-detection fails.
    image_token_index: Optional[int] = None
    # Override expected visual token count if auto-detection fails.
    expected_visual_tokens: Optional[int] = None


@dataclass
class ForwardResult:
    """Returned by extract_attentions for a single example."""
    # Per-layer attention: list[n_layers] of [n_heads, seq_len, seq_len] float32 tensors (CPU).
    attentions: List[torch.Tensor]
    # Span [v_start, v_end) of visual tokens inside seq_len.
    visual_start: int
    visual_end: int
    # Full input token count (visual + text).
    seq_len: int
    # The generated text (full decoded string).
    generated_text: str
    # input_ids used for the forward pass that produced attentions.
    input_ids: torch.Tensor


class VLMExtractor:
    """Loads a VLM and extracts per-layer attention at the first generated token."""

    def __init__(self, cfg: ExtractionConfig):
        self.cfg = cfg
        self.model: Optional[AutoModelForImageTextToText] = None
        self.processor: Optional[AutoProcessor] = None
        self._image_token_index: Optional[int] = None
        self._expected_visual_tokens: Optional[int] = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        logger.info("Loading processor for %s", self.cfg.model_id)
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model_id,
            trust_remote_code=True,
        )

        logger.info("Loading model %s (dtype=%s, device_map=%s, attn=eager)",
                    self.cfg.model_id, self.cfg.dtype, self.cfg.device_map)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.model_id,
            torch_dtype=self.cfg.dtype,
            device_map=self.cfg.device_map,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        self.model.eval()

        self._image_token_index = self._detect_image_token_index()
        self._expected_visual_tokens = self._detect_visual_token_count()
        logger.info("image_token_index=%d  expected_visual_tokens=%s",
                    self._image_token_index, self._expected_visual_tokens)

    def _detect_image_token_index(self) -> int:
        if self.cfg.image_token_index is not None:
            return self.cfg.image_token_index

        mcfg = self.model.config
        for attr in ("image_token_index", "image_token_id", "vision_token_id"):
            val = getattr(mcfg, attr, None)
            if val is not None:
                return int(val)

        # Fall back to processor vocabulary search.
        tok = self.processor.tokenizer
        for candidate in ("<image>", "<image_soft>", "[IMG]", "<IMG>", "<|image|>"):
            tid = tok.convert_tokens_to_ids(candidate)
            if tid != tok.unk_token_id:
                return int(tid)

        raise RuntimeError(
            "Cannot detect image_token_index automatically. "
            "Set ExtractionConfig.image_token_index explicitly."
        )

    def _detect_visual_token_count(self) -> Optional[int]:
        if self.cfg.expected_visual_tokens is not None:
            return self.cfg.expected_visual_tokens

        mcfg = self.model.config
        for attr in ("num_image_tokens", "image_seq_length", "vision_tokens", "mm_tokens_per_image"):
            val = getattr(mcfg, attr, None)
            if val is not None:
                return int(val)

        # Will be inferred at runtime from actual input_ids.
        return None

    # ------------------------------------------------------------------
    # Visual span detection
    # ------------------------------------------------------------------

    def find_visual_span(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """
        Returns (v_start, v_end) such that input_ids[v_start:v_end] are all image tokens.
        Raises if the visual span is absent or inconsistent.
        """
        ids = input_ids.squeeze(0).tolist()
        img_tok = self._image_token_index

        positions = [i for i, t in enumerate(ids) if t == img_tok]
        if not positions:
            raise ValueError(
                f"No image tokens (id={img_tok}) found in input_ids. "
                "Check that the processor inserted the image correctly."
            )

        v_start = positions[0]
        v_end = positions[-1] + 1

        # Assert contiguity — image tokens must form a single run.
        if v_end - v_start != len(positions):
            raise ValueError(
                f"Image tokens are not contiguous: {len(positions)} tokens "
                f"but span [{v_start}, {v_end}). Cannot define A_vv cleanly."
            )

        if self._expected_visual_tokens is not None:
            assert v_end - v_start == self._expected_visual_tokens, (
                f"Expected {self._expected_visual_tokens} visual tokens, "
                f"found {v_end - v_start}."
            )

        return v_start, v_end

    # ------------------------------------------------------------------
    # Forward pass + attention extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_attentions(
        self,
        image: Image.Image,
        prompt: str,
        clean_cache: bool = True,
    ) -> ForwardResult:
        """
        1. Runs greedy generation to get the model's response.
        2. Re-runs a teacher-forced forward at the first generated token to
           capture attention weights at that decoding step.
        Returns ForwardResult.
        """
        assert self.model is not None, "Call .load() first."

        # Build the chat message expected by the processor.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Prepare inputs (handles image encoding + chat template).
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}

        # ---- Step 1: greedy generation (no attentions stored — saves memory) ----
        gen_out = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.cfg.max_new_tokens,
            output_attentions=False,
            return_dict_in_generate=True,
        )
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = gen_out.sequences[0, prompt_len:]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        # ---- Step 2: teacher-forced forward at analyze_token_offset ----
        # We feed [prompt + generated_ids[:offset+1]] and capture attentions.
        offset = self.cfg.analyze_token_offset
        teacher_ids = gen_out.sequences[0, : prompt_len + offset + 1].unsqueeze(0)

        # Build pixel_values from original inputs (already computed by processor).
        fwd_inputs: Dict[str, torch.Tensor] = {"input_ids": teacher_ids}
        if "pixel_values" in inputs:
            fwd_inputs["pixel_values"] = inputs["pixel_values"]
        if "attention_mask" in inputs:
            fwd_inputs["attention_mask"] = torch.ones(
                teacher_ids.shape, dtype=torch.long, device=teacher_ids.device
            )
        # Copy any extra image-related keys the processor may have added.
        for key in inputs:
            if key not in fwd_inputs and "image" in key.lower():
                fwd_inputs[key] = inputs[key]

        fwd_out = self.model(
            **fwd_inputs,
            output_attentions=True,
            return_dict=True,
        )

        # fwd_out.attentions: tuple of [1, n_heads, seq, seq] per layer.
        raw_attentions = fwd_out.attentions
        if raw_attentions is None:
            raise RuntimeError(
                "Model returned None for attentions. "
                "Ensure attn_implementation='eager' and output_attentions=True."
            )

        # Squeeze batch dim, cast to float32 on CPU.
        attentions = [
            a.squeeze(0).to(torch.float32).cpu()  # [n_heads, seq, seq]
            for a in raw_attentions
        ]

        # Detect visual span.
        v_start, v_end = self.find_visual_span(teacher_ids)
        seq_len = teacher_ids.shape[1]

        if clean_cache:
            torch.cuda.empty_cache()

        return ForwardResult(
            attentions=attentions,
            visual_start=v_start,
            visual_end=v_end,
            seq_len=seq_len,
            generated_text=generated_text,
            input_ids=teacher_ids.cpu(),
        )

    # ------------------------------------------------------------------
    # Batch helper
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        examples: List[Dict],
    ) -> List[Optional[ForwardResult]]:
        """
        Processes a list of dicts with keys 'image' (PIL) and 'prompt' (str).
        Returns None for any example that raises.
        """
        results = []
        for i, ex in enumerate(examples):
            try:
                res = self.extract_attentions(ex["image"], ex["prompt"])
                results.append(res)
            except Exception as e:
                logger.error("Example %d failed: %s", i, e)
                results.append(None)
        return results
