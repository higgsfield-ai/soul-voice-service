"""Soul Voice: the inference half of the training repository.

    from soul_voice import Request, VoiceConsumer

    consumer = VoiceConsumer.load("checkpoints/round1",
                                  source_dir="third_party/breeze-tts",
                                  voicebook="checkpoints/voicebook.pt",
                                  device="cuda:0")
    print(consumer.verify())
    audio, = consumer.synthesize([Request(
        text="We go live in thirty seconds.",
        instruction="An urgent young female voice, clipped and low.")])

See README.md for what this contains, what it does not, and the checks that
must pass before it is trusted.
"""

from .conditioner import VoiceConditioner
from .consumer import Report, Request, Sampling, VoiceConsumer

__all__ = ["Report", "Request", "Sampling", "VoiceConditioner", "VoiceConsumer"]
