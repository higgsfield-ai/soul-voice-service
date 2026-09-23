"""Load a bundle, verify it, synthesise.

Three modes: `design` invents a voice from a description, `clone` copies one
from a reference recording, `direction` keeps a recording's identity but takes
its delivery from the description.

    consumer = VoiceConsumer.load("bundles/raft_round0", device="cuda:0")
    audio = consumer.synthesize([Request(text="Hello.", instruction="A calm voice.")])

Missing pieces here are additive, so they degrade the voice instead of raising.
`verify()` accounts for them, and `load` calls it.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .conditioner import VoiceConditioner
from .instruction import (
    MODE_INDEX,
    InstructionProjector,
    compose_by_mode,
    instruction_span_mask,
)

SAMPLE_RATE = 24_000
MODES = ("design", "clone", "direction")


@dataclass
class Request:
    """One utterance. `mode` decides which of the other fields are read."""

    text: str
    instruction: str
    mode: str = "design"
    reference: str | Path | None = None
    seed: int = 0
    style_mix_alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.mode != "design" and self.reference is None:
            raise ValueError(f"{self.mode} needs a reference recording")


@dataclass
class Sampling:
    """Decoding settings. These defaults are what the released clips used."""

    temperature: float = 0.9
    top_k: int = 50
    depth_temperature: float = 0.9
    depth_top_k: int = 50
    # Guidance contrasts the instructed prompt against the bare text. 1.0
    # switches the trained null branch off, which measurably costs voice
    # fidelity; 2.5 is what every released render used.
    guidance_scale: float = 2.5
    max_new_tokens: int = 1024


@dataclass
class Report:
    """What `verify` found, so a deployment can assert on it."""

    parameters: dict[str, int] = field(default_factory=dict)
    voicebook: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def __str__(self) -> str:
        lines = [f"{name:<22s} {count / 1e6:>8.1f}M parameters"
                 for name, count in self.parameters.items()]
        lines.append(f"{'voicebook':<22s} {self.voicebook:>8d} prototypes"
                     if self.voicebook else
                     f"{'voicebook':<22s} {'absent':>8s} - voices will be averaged, "
                     f"not real speakers")
        lines.extend(f"PROBLEM: {p}" for p in self.problems)
        return "\n".join(lines)


def _beside(bundle: Path, value: str) -> Path:
    """Resolve a path a manifest names, relative to the bundle that names it.

    An absolute one points out of the bundle at something shared on this
    machine, which is how the training exports were written. A relative one
    means the bundle travels with its own copy, so a checkpoint that was
    copied or downloaded somewhere else resolves without anyone editing a
    manifest.
    """
    path = Path(value)
    return path if path.is_absolute() else (bundle / path).resolve()


class VoiceConsumer:
    """A loaded model, ready to synthesise."""

    def __init__(self, model, tokenizer, audio_tokenizer, conditioner, projector,
                 voice_encoder, frontend, manifest: dict, device: str,
                 sampling: Sampling) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.conditioner = conditioner
        self.projector = projector
        self.voice_encoder = voice_encoder
        self.frontend = frontend
        self.manifest = manifest
        self.device = device
        self.sampling = sampling

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, bundle: str | Path, *, source_dir: str | Path,
             device: str = "cuda:0", sampling: Sampling | None = None,
             voicebook: str | Path | None = None,
             depth: str = "fused", compile: bool = False,
             strict: bool = True) -> "VoiceConsumer":
        """Build a consumer from a bundle directory.

        `voicebook` is passed separately because no bundle here ships one, and
        without it the projector returns an averaged voice rather than a real
        speaker. `depth` picks the depth decoder loop; see `decode.py`.

        `compile` captures the depth decoder in CUDA graphs, which is worth
        about 3x. It costs a few minutes of compilation on the first request,
        so it belongs in a process that will serve many - hence off here. It
        falls back to eager if the capture fails; see `accelerate.py`.
        """
        bundle, source_dir = Path(bundle), Path(source_dir)
        path = str(source_dir.resolve())
        if path not in sys.path:
            sys.path.insert(0, path)

        from transformers import AutoTokenizer
        from models.breeze import BreezeForConditionalGeneration
        from qwen_tts import Qwen3TTSTokenizer

        from .voice_encoder import ReferenceFrontend, VoiceEncoder

        manifest = json.loads((bundle / "manifest.json").read_text())
        # The bundle carries only what was retrained. The tokenizer and the
        # codec still come from the released checkpoint it was built on, and a
        # bundle that has drifted from its base is not loadable at all.
        base = _beside(bundle, manifest["base_checkpoint"])
        if not (base / "audio_tokenizer").is_dir():
            raise FileNotFoundError(f"{base} has no audio_tokenizer")

        tokenizer = AutoTokenizer.from_pretrained(str(base), fix_mistral_regex=False)
        model = BreezeForConditionalGeneration.from_pretrained(
            str(_beside(bundle, manifest.get("backbone", "model"))),
            dtype=torch.bfloat16, local_files_only=True,
            attn_implementation="eager").to(device).eval()
        model.config.use_cache = False
        audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            str(base / "audio_tokenizer"), device_map=device)

        conditioner = VoiceConditioner.load(bundle).to(device).eval()
        conditioner.attach(model)
        projector = InstructionProjector.load(bundle).to(device).eval()
        frontend = ReferenceFrontend(str(base / "audio_tokenizer"), device=device)
        encoder = VoiceEncoder.load_stage1(bundle / "voice_encoder").to(device).eval()

        if voicebook is not None:
            from .voicebook import PrototypeVoicebook

            projector.attach_voicebook(
                PrototypeVoicebook.load(Path(voicebook)).to(device))

        from . import decode

        tracker = decode.RowTracker(conditioner).attach(model)
        # A captured graph needs one shape, which the padded frame provides.
        decode.install(model, tracker, mode=depth, pad_frame=compile)
        if compile:
            from . import accelerate

            accelerate.capture(model)

        consumer = cls(model, tokenizer, audio_tokenizer, conditioner, projector,
                       encoder, frontend, manifest, device, sampling or Sampling())
        consumer._apply_sampling()
        report = consumer.verify()
        if strict and not report.ok:
            raise RuntimeError(f"bundle failed verification:\n{report}")
        return consumer

    def _apply_sampling(self) -> None:
        """Push the decoding settings into the backbone's own generation config.

        Both decoders read their settings from there, not from this object.
        """
        from breeze_infer.runtime import update_generation_config_for_breeze

        s = self.sampling
        update_generation_config_for_breeze(self.model, {
            "do_sample": True, "temperature": s.temperature,
            "top_k": s.top_k, "top_p": 1.0,
            "depth_decoder_do_sample": True,
            "depth_decoder_temperature": s.depth_temperature,
            "depth_decoder_top_k": s.depth_top_k, "depth_decoder_top_p": 1.0,
            "max_new_tokens": s.max_new_tokens,
        })

    # ---------------------------------------------------------------- verify

    def verify(self) -> Report:
        """Account for every tensor, and name what is missing."""
        report = Report()
        for name, module in (("model", self.model),
                             ("conditioner", self.conditioner),
                             ("projector", self.projector),
                             ("voice_encoder", self.voice_encoder)):
            report.parameters[name] = sum(p.numel() for p in module.parameters())
            bad = [n for n, p in module.named_parameters()
                   if not torch.isfinite(p).all()]
            if bad:
                report.problems.append(
                    f"{name}: {len(bad)} tensors contain NaN or Inf, first {bad[0]}")

        book = getattr(self.projector, "voicebook", None)
        report.voicebook = 0 if book is None else int(book.id_tokens.shape[0])

        # The conditioner is useless unless it actually reached the backbone.
        from .modules import AdaRMSNorm

        installed = sum(1 for layer in self.model.backbone_model.layers
                        if getattr(layer.self_attn, "voice_prefix_slot", None) is not None)
        modulated = sum(1 for layer in self.model.backbone_model.layers
                        if isinstance(layer.input_layernorm, AdaRMSNorm))
        expected = len(self.conditioner.backbone_identity.layers)
        if installed != expected:
            report.problems.append(
                f"voice prefix reached {installed} backbone layers, expected {expected}")
        if modulated != len(self.conditioner.backbone_style.layers):
            report.problems.append(
                f"style modulation reached {modulated} backbone layers, expected "
                f"{len(self.conditioner.backbone_style.layers)}")
        # Per layer, not per stack: the layers carry their own config objects,
        # so a stack set to "soul_voice_prefix" can still have every layer on
        # stock attention, ignoring the prefix while the norms keep applying.
        # That shipped in an earlier draft of this package and sounded fine.
        for stack, name in ((self.model.backbone_model, "backbone"),
                            (self.model.depth_decoder.model, "depth")):
            stale = [index for index, layer in enumerate(stack.layers)
                     if getattr(layer.self_attn, "config", stack.config)
                     ._attn_implementation != "soul_voice_prefix"]
            if stale:
                report.problems.append(
                    f"{name}: {len(stale)} of {len(stack.layers)} layers are not "
                    f"dispatching through the voice-prefix attention, so their "
                    f"identity prefix is ignored (first is layer {stale[0]})")
        return report

    # ------------------------------------------------------------- condition

    @torch.no_grad()
    def _describe(self, requests: list[Request], *, sample_voices: bool,
                  top_k: int | None, temperature: float) -> dict:
        """Caption to latents, through the instruction projector."""
        from .model_inputs import prompt_batch
        from .instruction import instruction_token_ids

        batch = prompt_batch(self.tokenizer, self.model.config,
                             [r.text for r in requests],
                             [r.instruction for r in requests], self.device)
        start, end = instruction_token_ids(self.tokenizer)
        mask = instruction_span_mask(batch["input_ids"],
                                     begin_token_id=start, end_token_id=end)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            merged = self.model._merge_input_ids_with_input_values(
                input_ids=batch["input_ids"], input_values=batch["input_values"],
                labels=None, text_ids_mask=batch["text_ids_mask"],
                text_ids_len=batch["text_ids_len"],
                attention_mask=batch["attention_mask"])
            drawing = sample_voices and self.projector.voicebook is not None
            projected = (self.projector(merged["inputs_embeds"], mask, sample=True,
                                        top_k=top_k, temperature_scale=temperature)
                         if drawing else
                         self.projector(merged["inputs_embeds"], mask))
        return {key: value.float() for key, value in projected.items()}

    @property
    def reference_highpass_hz(self) -> float:
        """The filter the checkpoint was trained with; references must match."""
        return float(self.manifest.get("reference_highpass_hz", 0.0))

    @torch.no_grad()
    def _listen(self, references: list, seeds: list[int]) -> dict:
        """Reference recordings to latents, through the Stage-1 encoder."""
        from .reference import pad_batch, read_reference

        waveforms = [read_reference(path, seed=seed,
                                    highpass_hz=self.reference_highpass_hz)
                     for path, seed in zip(references, seeds)]
        padded, lengths = pad_batch(waveforms)
        features, frame_lengths = self.frontend(
            torch.from_numpy(padded).to(self.device),
            torch.from_numpy(lengths).to(self.device))
        encoded = self.voice_encoder(features=features, frame_lengths=frame_lengths)
        return {key: value.float() for key, value in encoded.items()}

    # ------------------------------------------------------------ synthesis

    @torch.no_grad()
    def synthesize(self, requests: list[Request], *, batch_size: int = 6,
                   sample_voices: bool = False, voice_top_k: int | None = None,
                   voice_temperature: float = 1.0) -> list[np.ndarray]:
        """Render every request, batching rows that can share one forward pass.

        Rows batch together when they agree on mode, seed and style mix. Mode
        decides which encoder produces a row's latents; the other two are set
        once per batch, so rows that disagree have to be rendered apart or the
        first row's values would silently stand in for the rest.

        `sample_voices` draws a speaker from the voicebook per row and is off:
        a description then maps to one settled voice rather than a different
        person each call. Turn it on to cast distinct speakers.
        """
        from breeze_infer.runtime import set_all_seeds
        from breeze_infer.templates import get_template, prepare_inputs

        results: list[np.ndarray | None] = [None] * len(requests)
        grouped: dict[tuple, list[int]] = {}
        for position, request in enumerate(requests):
            key = (request.mode, request.seed, request.style_mix_alpha)
            grouped.setdefault(key, []).append(position)

        for (mode, seed, style_mix_alpha), positions in grouped.items():
            for start in range(0, len(positions), batch_size):
                span = positions[start:start + batch_size]
                batch = [requests[i] for i in span]

                # Seeded before projecting too: with a voicebook attached the
                # speaker is drawn here, not during generation.
                set_all_seeds(seed)
                text_latents = self._describe(
                    batch, sample_voices=sample_voices, top_k=voice_top_k,
                    temperature=voice_temperature)
                if mode == "design":
                    audio_latents = {key: torch.zeros_like(value)
                                     for key, value in text_latents.items()}
                else:
                    audio_latents = self._listen([r.reference for r in batch],
                                                 [r.seed for r in batch])

                index = torch.full((len(batch),), MODE_INDEX[mode],
                                   dtype=torch.long, device=self.device)
                condition = compose_by_mode(
                    audio_latents, text_latents, index,
                    style_mix_alpha=style_mix_alpha)

                set_all_seeds(seed)
                inputs = prepare_inputs(
                    self.tokenizer, self.audio_tokenizer, self.model,
                    [{"text": r.text, "instruction": r.instruction,
                      "ref_audio": None, "ref_text": None} for r in batch],
                    get_template("tts_instruction"),
                    guidance_scale=self.sampling.guidance_scale,
                    guidance_scale_ref=None, guidance_scale_ins=None)

                self.conditioner.set_condition(condition)
                self.conditioner.enable_guidance_shim(
                    self.sampling.guidance_scale != 1.0)
                try:
                    audio = self.model.generate(
                        **inputs, output_audio=True,
                        audio_tokenizer=self.audio_tokenizer,
                        max_new_tokens=self.sampling.max_new_tokens)
                finally:
                    self.conditioner.reset()

                for position, item in zip(span, audio):
                    results[position] = np.asarray(
                        item.float().cpu().numpy(), dtype=np.float32).reshape(-1)
        return results  # type: ignore[return-value]


__all__ = ["Report", "Request", "Sampling", "VoiceConsumer"]
