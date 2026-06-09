"""
Phase 1: PGD adversarial perturbation crafting.

Crafts white-box PGD perturbations δ against an open-weight VLM
to force a target output string, subject to ‖δ‖_∞ ≤ ε.

Memory note: crafting requires a backward pass.  For Gemma 4 4B this is tight
at 16 GB.  Use gradient_checkpointing=True on the language model, or craft on
a lighter model (Qwen2.5-VL-3B / SmolVLM2-2.2B) and run *detection* on Gemma.

Usage:
    crafter = PGDCrafter(model, processor, eps=8/255, n_steps=200, step_size=1/255)
    delta = crafter.craft(image, prompt, target_text)
    attacked_image = crafter.apply(image, delta)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class PGDCrafter:
    def __init__(
        self,
        model,
        processor,
        eps: float = 8 / 255,
        n_steps: int = 200,
        step_size: float = 1 / 255,
        target_text: str = "Sure, here is how to",
        device: str = "cuda",
        gradient_checkpointing: bool = True,
    ):
        self.model = model
        self.processor = processor
        self.eps = eps
        self.n_steps = n_steps
        self.step_size = step_size
        self.target_text = target_text
        self.device = device

        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")

    # ------------------------------------------------------------------

    def craft(
        self,
        image: Image.Image,
        prompt: str,
        target_text: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Returns delta: [C, H, W] float32 tensor (pixel scale 0-255),
        clipped to ‖δ‖_∞ ≤ eps*255.
        """
        target = target_text or self.target_text

        # Encode image to pixel tensor [1, C, H, W] in [0, 1].
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text":  prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}

        # Target token ids.
        target_ids = self.processor.tokenizer(
            target, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        n_target = target_ids.shape[1]

        # Isolate pixel_values and make them require grad.
        pv: torch.Tensor = inputs["pixel_values"].clone().detach().requires_grad_(False)
        pv_orig = pv.clone()
        eps_px = self.eps  # pixel_values are normalised [0,1] by the processor

        delta = torch.zeros_like(pv, requires_grad=True)

        for step in range(self.n_steps):
            perturbed_pv = (pv_orig + delta).clamp(0.0, 1.0)
            inputs_step = {**inputs, "pixel_values": perturbed_pv}

            # Teacher-force: feed prompt + target prefix and maximize log-prob of target.
            full_ids = torch.cat(
                [inputs["input_ids"],
                 target_ids[:, :max(1, (step * n_target) // self.n_steps + 1)]], dim=1
            )
            inputs_step["input_ids"] = full_ids

            self.model.zero_grad()
            out = self.model(**inputs_step, labels=full_ids)
            loss = -out.loss   # maximise likelihood of target → minimise -loss

            loss.backward()

            with torch.no_grad():
                grad_sign = delta.grad.sign()
                delta.data = delta.data - self.step_size * grad_sign
                delta.data = delta.data.clamp(-eps_px, eps_px)
                delta.data = (pv_orig + delta.data).clamp(0.0, 1.0) - pv_orig

            delta.grad = None

            if (step + 1) % 50 == 0:
                logger.debug("PGD step %d/%d, loss=%.4f", step + 1, self.n_steps, loss.item())

        return delta.detach().cpu()

    def apply(self, image: Image.Image, delta: torch.Tensor) -> Image.Image:
        """Apply delta to image and return a new PIL Image."""
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                  {"type": "text",  "text": ""}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        )
        pv = inputs["pixel_values"][0].float()  # [C, H, W]
        pv_adv = (pv + delta.squeeze(0)).clamp(0.0, 1.0)

        # Denormalise using processor image_mean/std if available.
        img_np = (pv_adv.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(img_np)

    def save_delta(self, delta: torch.Tensor, path: Path) -> None:
        torch.save(delta, path)
        logger.info("Saved delta to %s", path)
