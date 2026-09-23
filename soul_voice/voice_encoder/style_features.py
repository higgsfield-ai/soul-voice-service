"""Names of the Stage-1 auxiliary targets, kept only for their counts.

The training package computes these features from audio with several hundred
lines of DSP. None of that runs at inference: the auxiliary heads exist so the
encoder learns a style space during Stage 1, and serving only ever reads the
latents. What still matters is the *number* of targets, because it fixes the
head output widths and therefore the checkpoint shapes, so the names are kept
verbatim and the DSP is not carried over.
"""

STYLE_TARGET_NAMES: tuple[str, ...] = (
    "f0_range",
    "f0_std",
    "f0_delta",
    "voicing_ratio",
    "periodicity",
    "log_rms_range",
    "pause_ratio",
    "syllable_rate",
    "spectral_centroid",
    "hf_ratio",
    "rel_f0_median",
    "rel_log_rms_median",
    "rel_syllable_rate",
    "rel_spectral_tilt",
)

IDENTITY_PROBE_NAMES: tuple[str, ...] = (
    "f0_median",
    "spectral_tilt",
    "log_rms_median",
)

__all__ = ["IDENTITY_PROBE_NAMES", "STYLE_TARGET_NAMES"]
