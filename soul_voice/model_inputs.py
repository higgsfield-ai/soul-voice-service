"""Turning requests into the tensors the backbone and the projector expect.

Two serialisations, not interchangeable. The projector reads the instruction
span out of the backbone's embedding of the full prompt, so the prompt has to
be laid out the way training laid it out; drift moves the span and the voice
comes out plausible but wrong. Generation instead needs the backbone's inference
serialisation and its negative prompt, which is `prepare_inputs` from the
released `breeze_infer`, used as-is.
"""

from typing import Any

import numpy as np
import torch

from .serialization import AUDIO_CODEBOOKS, prepare_training_example

# The projector only reads the instruction span, so the target audio is a
# placeholder. Four frames is the smallest the serialisation accepts.
PLACEHOLDER_FRAMES = 4


def prompt_batch(tokenizer: Any, model_config: Any, texts: list[str],
                 instructions: list[str], device: str) -> dict[str, torch.Tensor]:
    """Collate prompts into one rectangular batch for the projector.

    Every row carries the same placeholder audio, so only the text needs
    padding.
    """
    if len(texts) != len(instructions):
        raise ValueError("texts and instructions must be the same length")
    codes = np.zeros((PLACEHOLDER_FRAMES, AUDIO_CODEBOOKS), dtype=np.int16)
    rows = [prepare_training_example(tokenizer, model_config, text=text,
                                     instruction=instruction, target_codes=codes)
            for text, instruction in zip(texts, instructions)]

    width = max(int(row["input_ids"].shape[-1]) for row in rows)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    fields: dict[str, list[torch.Tensor]] = {
        "input_ids": [], "attention_mask": [], "text_ids_mask": []}
    lengths: list[int] = []
    for row in rows:
        gap = width - int(row["input_ids"].shape[-1])
        # Right padding with zero attention: the instruction span is found by
        # token id, so where the padding sits only has to be consistent.
        fields["input_ids"].append(
            torch.nn.functional.pad(row["input_ids"].view(1, -1), (0, gap), value=pad_id))
        fields["attention_mask"].append(
            torch.nn.functional.pad(row["attention_mask"].view(1, -1), (0, gap), value=0))
        fields["text_ids_mask"].append(
            torch.nn.functional.pad(row["text_ids_mask"].view(1, -1), (0, gap), value=False))
        lengths.extend(int(v) for v in torch.as_tensor(row["text_ids_len"]).flatten())

    batch = {key: torch.cat(value, dim=0).to(device) for key, value in fields.items()}
    batch["text_ids_len"] = torch.tensor(lengths, dtype=torch.long, device=device)
    batch["input_values"] = torch.zeros(
        len(rows), PLACEHOLDER_FRAMES, AUDIO_CODEBOOKS, dtype=torch.long, device=device)
    return batch


__all__ = ["PLACEHOLDER_FRAMES", "prompt_batch"]
