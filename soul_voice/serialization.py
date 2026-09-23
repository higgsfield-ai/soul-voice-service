"""Soul Voice pinned template, token preparation, and label contracts."""

from pathlib import Path
from typing import Any, Iterable

import numpy as np

BOS = "<bos>"
SPEAKER = "S0"
INS_BOS = "<ins_bos>"
INS_EOS = "<ins_eos>"
AUDIO_CODEBOOKS = 16
TARGET_EOS_CODEBOOK0 = 2051
IGNORE_INDEX = -100


def native_text(text: str, instruction: str) -> str:
    """Return the text segment passed to the pinned tokenizer.

    The pinned ``_prepare_one`` calls the tokenizer with
    ``add_special_tokens=True`` for each text segment, so a literal ``<bos>``
    must never be included here.
    """
    text, instruction = str(text or "").strip(), str(instruction or "").strip()
    if not text or not instruction:
        raise ValueError("Soul Voice requires separate non-empty text and instruction")
    return f"[{SPEAKER}]{INS_BOS}{instruction}{INS_EOS}{text}"


def debug_serialization(text: str, instruction: str) -> str:
    """Human-readable serialization; not an input to the tokenizer."""
    return BOS + native_text(text, instruction)


def request_record(text: str, instruction: str, *, ref_audio: str | None = None,
                   ref_text: str | None = None) -> dict[str, Any]:
    target = native_text(text, instruction)
    if (ref_audio is None) != (ref_text is None):
        raise ValueError("Soul Voice reference editing requires both ref_audio and ref_text")
    if ref_audio is None:
        return {"template_name": "tts_instruction", "text": text, "instruction": instruction,
                "ref_audio": None, "ref_text": None, "target_segment_text": target,
                "serialized_target": BOS + target}
    if not str(ref_text).strip():
        raise ValueError("ref_text must be non-empty")
    return {
        "template_name": "ref_edit_tata", "text": text, "instruction": instruction,
        "ref_audio": str(ref_audio), "ref_text": str(ref_text).strip(),
        "serialized_reference_text": f"{BOS}[{SPEAKER}]{str(ref_text).strip()}",
        "target_segment_text": target, "serialized_target": BOS + target,
    }


def load_split_inference_runtime(
    source_dir: str | Path,
    model_dir: str | Path,
    assets_dir: str | Path,
    *,
    device: str,
    attn_implementation: str = "eager",
) -> tuple[Any, Any, Any]:
    """Load the official inference source with weights and assets split.

    Full-SFT checkpoints saved by the trainer intentionally contain only the
    model/config files. The official inference loader expects tokenizer and
    ``audio_tokenizer`` assets in the same directory, so use the pinned base
    package for those immutable assets without copying anything into or
    modifying the trained checkpoint.
    """
    import sys

    import torch
    from transformers import AutoTokenizer

    source_dir = Path(source_dir)
    model_dir = Path(model_dir)
    assets_dir = Path(assets_dir)
    required_source = [
        source_dir / "models" / "breeze.py",
        source_dir / "breeze_infer" / "templates.py",
        source_dir / "models" / "fast_streaming.py",
    ]
    missing_source = [str(path) for path in required_source if not path.is_file()]
    if missing_source:
        raise FileNotFoundError(f"pinned inference source is incomplete: {missing_source}")
    required_model = [model_dir / "config.json", model_dir / "model.safetensors.index.json"]
    missing_model = [str(path) for path in required_model if not path.is_file()]
    if missing_model:
        raise FileNotFoundError(f"model files are missing: {missing_model}")
    required_assets = [
        assets_dir / "config.json",
        assets_dir / "tokenizer.json",
        assets_dir / "audio_tokenizer",
    ]
    missing_assets = [str(path) for path in required_assets if not path.exists()]
    if missing_assets:
        raise FileNotFoundError(f"inference assets are missing: {missing_assets}")

    inserted = str(source_dir)
    sys.path.insert(0, inserted)
    try:
        from models.breeze import BreezeForConditionalGeneration
    finally:
        if sys.path and sys.path[0] == inserted:
            sys.path.pop(0)

    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        assets_dir,
        local_files_only=True,
        fix_mistral_regex=False,
    )
    model = BreezeForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation=attn_implementation,
    )
    model.config.use_cache = True
    model.to(device).eval()

    from qwen_tts import Qwen3TTSTokenizer

    audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(assets_dir / "audio_tokenizer"),
        device_map=device,
    )
    if int(model.config.num_codebooks) != AUDIO_CODEBOOKS:
        raise ValueError(
            f"trained model has {model.config.num_codebooks} codebooks; expected {AUDIO_CODEBOOKS}"
        )
    return tokenizer, model, audio_tokenizer


def build_target_only_labels(sequence_length: int, target_codes: np.ndarray,
                             target_start: int, *, target_eos_position: int | None = None) -> np.ndarray:
    """Build [L,16] labels with loss only on target codec frames and target EOS.

    Text, reference placeholders/frames/EOS, depth EOS, padding, and nonzero
    EOS codebooks remain -100. This bypasses the stock merge behavior that
    overwrites labels at every audio placeholder.
    """
    codes = np.asarray(target_codes)
    if codes.ndim != 2 or codes.shape[1] != AUDIO_CODEBOOKS:
        raise ValueError(f"target_codes must have shape [T,{AUDIO_CODEBOOKS}]")
    if not np.issubdtype(codes.dtype, np.integer):
        raise TypeError("target_codes must be integer codec IDs")
    target_eos_position = target_start + codes.shape[0] if target_eos_position is None else target_eos_position
    if target_start < 0 or target_start + codes.shape[0] > sequence_length:
        raise ValueError("target codec span is outside sequence")
    if target_eos_position < 0 or target_eos_position >= sequence_length:
        raise ValueError("target EOS position is outside sequence")
    labels = np.full((sequence_length, AUDIO_CODEBOOKS), IGNORE_INDEX, dtype=np.int64)
    labels[target_start:target_start + codes.shape[0], :] = codes.astype(np.int64, copy=False)
    labels[target_eos_position, 0] = TARGET_EOS_CODEBOOK0
    return labels


def collate_target_only(examples: Iterable[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Pad tokenized examples and return three-dimensional [B,L,16] labels."""
    rows = list(examples)
    if not rows:
        raise ValueError("collator received no examples")
    max_len = max(int(row["sequence_length"]) for row in rows)
    labels = np.full((len(rows), max_len, AUDIO_CODEBOOKS), IGNORE_INDEX, dtype=np.int64)
    attention = np.zeros((len(rows), max_len), dtype=np.int64)
    for index, row in enumerate(rows):
        length = int(row["sequence_length"])
        one = build_target_only_labels(length, np.asarray(row["target_codes"]),
                                       int(row["target_start"]),
                                       target_eos_position=row.get("target_eos_position"))
        labels[index, :length] = one
        attention[index, :length] = 1
    return {"labels": labels, "attention_mask": attention}


def _token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    values = encoded["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _render_text_segment(tokenizer: Any, text: str) -> str:
    ids = _token_ids(tokenizer, text, add_special_tokens=True)
    return tokenizer.decode(ids, skip_special_tokens=False)


def prepare_training_example(
    tokenizer: Any,
    model_config: Any,
    *,
    text: str,
    instruction: str,
    target_codes: np.ndarray,
    ref_text: str | None = None,
    ref_codes: np.ndarray | None = None,
) -> dict[str, Any]:
    """Prepare one prompt+target sequence exactly like pinned ``_prepare_one``.

    Text segments are independently tokenized with special tokens, decoded,
    concatenated with audio placeholders, and tokenized once without special
    tokens. Labels are supplied separately as target-only ``[L,16]``.
    """
    import torch

    target = np.asarray(target_codes)
    if target.ndim != 2 or target.shape[1] != AUDIO_CODEBOOKS:
        raise ValueError(f"target_codes must have shape [T,{AUDIO_CODEBOOKS}]")
    if not np.issubdtype(target.dtype, np.integer) or target.shape[0] < 1:
        raise ValueError("target_codes must contain at least one integer frame")
    if (ref_text is None) != (ref_codes is None):
        raise ValueError("reference text and codes must be supplied together")
    reference = None if ref_codes is None else np.asarray(ref_codes)
    if reference is not None and (
        reference.ndim != 2 or reference.shape[1] != AUDIO_CODEBOOKS
        or not np.issubdtype(reference.dtype, np.integer) or reference.shape[0] < 1
    ):
        raise ValueError(f"ref_codes must contain integer [R,{AUDIO_CODEBOOKS}] frames")

    audio_tag = "<|AUDIO|>"
    audio_eos = "<|audio_eos|>"
    rendered: list[tuple[str, str, str]] = []
    if reference is not None:
        reference_text = f"[{SPEAKER}]{str(ref_text).strip()}"
        if not str(ref_text).strip():
            raise ValueError("ref_text must be non-empty")
        rendered.append(("text", "reference_text", _render_text_segment(tokenizer, reference_text)))
        rendered.append(("audio", "reference_audio", audio_tag * reference.shape[0] + audio_eos))
    rendered.append(("text", "target_text", _render_text_segment(tokenizer, native_text(text, instruction))))
    rendered.append(("audio", "target_audio", audio_tag * target.shape[0] + audio_eos))

    final_text = "".join(value for _, _, value in rendered)
    final = tokenizer(final_text, add_special_tokens=False, return_tensors="pt")
    input_ids = final["input_ids"]
    attention_mask = final.get("attention_mask", torch.ones_like(input_ids))
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("tokenizer must return one [1,L] input_ids tensor")

    text_ids_mask: list[bool] = []
    text_ids_len: list[int] = []
    target_start = target_eos_position = None
    reference_positions: list[int] = []
    offset = 0
    audio_token_id = int(model_config.audio_token_id)
    audio_eos_token_id = int(model_config.audio_eos_token_id)
    for kind, name, value in rendered:
        ids = _token_ids(tokenizer, value, add_special_tokens=False)
        length = len(ids)
        if kind == "text":
            text_ids_mask.extend([True] * length)
            text_ids_len.append(length)
        else:
            text_ids_mask.extend([False] * length)
            frame_positions = [offset + index for index, token in enumerate(ids) if token == audio_token_id]
            eos_positions = [offset + index for index, token in enumerate(ids) if token == audio_eos_token_id]
            expected_frames = reference.shape[0] if name == "reference_audio" else target.shape[0]
            if len(frame_positions) != expected_frames or len(eos_positions) != 1:
                raise ValueError(
                    f"{name} tokenization mismatch: expected {expected_frames} placeholders and one EOS"
                )
            if frame_positions != list(range(frame_positions[0], frame_positions[0] + expected_frames)):
                raise ValueError(f"{name} placeholders are not contiguous")
            if eos_positions[0] != frame_positions[-1] + 1:
                raise ValueError(f"{name} EOS does not immediately follow its placeholders")
            if name == "reference_audio":
                reference_positions.extend(frame_positions + eos_positions)
            else:
                target_start, target_eos_position = frame_positions[0], eos_positions[0]
        offset += length

    if offset != input_ids.shape[1] or len(text_ids_mask) != input_ids.shape[1]:
        raise ValueError("segment token lengths do not match the final sequence")
    if target_start is None or target_eos_position is None:
        raise AssertionError("target positions were not constructed")
    labels = build_target_only_labels(
        input_ids.shape[1], target, target_start, target_eos_position=target_eos_position
    )
    if reference_positions and np.any(labels[reference_positions] != IGNORE_INDEX):
        raise AssertionError("reference frames or EOS leaked into target labels")
    input_values = target if reference is None else np.concatenate([reference, target], axis=0)
    if int((input_ids == audio_token_id).sum()) != input_values.shape[0]:
        raise ValueError("audio placeholder count does not equal concatenated codec frame count")
    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "text_ids_mask": torch.tensor([text_ids_mask], dtype=torch.bool),
        "text_ids_len": torch.tensor(text_ids_len, dtype=torch.long),
        "input_values": torch.as_tensor(input_values.astype(np.int64, copy=False)).unsqueeze(0),
        "labels": torch.as_tensor(labels).unsqueeze(0),
        "target_start": target_start,
        "target_eos_position": target_eos_position,
        "reference_positions": reference_positions,
        "debug_serialization": final_text,
    }


def codec_cache_spec(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    return {
        "layout_version": "breeze-codec-cache-v2",
        "records": str(root / "split=<split>" / "shard=<shard>" / "part-*.parquet"),
        "required_columns": [
            "seg_id", "split", "target_codes", "codec_frames", "sample_rate_hz",
            "source_sample_rate_hz", "codec_revision", "source_revision",
            "checkpoint_revision",
        ],
        "codec_codes_contract": (
            "int16 [T,16], codec sample_rate_hz=24000, mono, 12.5 Hz; "
            "source_sample_rate_hz records the decoded input rate"
        ),
        "preprocess_argv": ["python3", "-m", "tts_finetune.trainers.breeze_trainer",
            "--prepare-codec-cache", "--source-dir", "<pinned-breeze-source>",
            "--checkpoint-dir", "<pinned-checkpoint>", "--manifest", "<shared.parquet>",
            "--codec-cache", str(root)],
    }
