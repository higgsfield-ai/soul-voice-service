# Breeze source

`breeze-tts/models`, `breeze-tts/breeze_infer`, `breeze-tts/README.md` and
`breeze-tts/LICENSE` are unmodified copies from
https://github.com/breezeblue-ai/breeze-tts at commit
`008f769016b0a24711becd7a4925030bc93f608c`.

The serving package imports `models.breeze` and `breeze_infer` from this
directory. It does not start Breeze's separate HTTP server. Keeping this source
in the repository makes the runtime independent of GitHub and mutable branches.

The upstream source is Apache-2.0 licensed. Model weights have separate terms;
their supplied license is included in the model inventory. See
https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE.

This is the public source revision selected for this service. The original
handoff did not record a Breeze **source** commit; the model's HF revision is
recorded separately in `checkpoints/shared/base_checkpoint/REVISION`.
