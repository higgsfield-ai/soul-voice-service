"""Graph capture for the depth decoder, and the toolchain repair it needs.

The depth stack is where the frame budget goes and it is bound by kernel
launches rather than arithmetic - fifteen traversals of twelve small layers,
none of which keeps the GPU busy. Capturing them in a CUDA graph is worth
about three times the rest of the package put together.

`torch.compile(mode="reduce-overhead")` does the capture. Two practical
snags, both handled here:

Triton ships its own `ptxas`, and on a new enough GPU that copy does not know
the architecture - a Blackwell Ultra reports `sm_103a`, which Triton's CUDA
12.8 assembler rejects outright. A newer assembler is usually installed beside
it, so this points Triton at one.

Compilation is lazy, so a failure lands in the middle of the first request
rather than at setup. `capture` therefore falls back to the eager module on
the first failure and stays there, which makes turning it on safe: the worst
case is the speed you already had.
"""

import os
import subprocess
from pathlib import Path

import torch

# Where a working assembler is likely to be if Triton's own is too old.
PTXAS_SEARCH = ("/usr/local/cuda/bin/ptxas", "/usr/local/cuda-13.0/bin/ptxas",
                "/usr/local/cuda-12.9/bin/ptxas")


def _supports(ptxas: Path, architecture: str) -> bool:
    if not ptxas.is_file():
        return False
    try:
        listed = subprocess.run([str(ptxas), "--help"], capture_output=True,
                                text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return architecture in listed


def repair_ptxas(device: int = 0) -> str | None:
    """Point Triton at an assembler that knows this GPU, if it needs one.

    Returns the path adopted, or None if nothing had to change. Respects
    `TRITON_PTXAS_PATH` if it is already set.
    """
    if os.environ.get("TRITON_PTXAS_PATH"):
        return os.environ["TRITON_PTXAS_PATH"]
    major, minor = torch.cuda.get_device_capability(device)
    # Triton appends `a` for the architecture-specific variants it emits.
    architecture = f"sm_{major}{minor}a"
    try:
        from triton.backends.nvidia import compiler as nvidia

        shipped = Path(nvidia.__file__).parent / "bin/ptxas"
    except Exception:
        shipped = Path("")
    if _supports(shipped, architecture):
        return None
    for candidate in PTXAS_SEARCH:
        if _supports(Path(candidate), architecture):
            os.environ["TRITON_PTXAS_PATH"] = candidate
            return candidate
    return None


def capture(model, *, verbose: bool = True) -> bool:
    """Capture the depth decoder in CUDA graphs. True if it took.

    The backbone is deliberately left alone. Its output buffer is held across
    steps by the generation loop, and its cache grows every frame, so capturing
    it means a static cache and a rewrite of the loop that owns it - a much
    larger change than this one, and the depth stack is the larger prize.
    """
    # One graph per batch size, and a handful of shapes around the edges of a
    # render. The default of eight is spent before the first request finishes,
    # and dynamo answers that by silently running eager for the rest of the
    # process - which is what a capture that looks installed but buys nothing
    # turns out to be.
    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit, 64)
    adopted = repair_ptxas()
    if verbose and adopted:
        print(f"graph capture: using {adopted} (Triton's own is too old for "
              f"sm_{''.join(map(str, torch.cuda.get_device_capability()))})")

    # The bound method rather than the module: swapping the module out would
    # unregister the depth decoder's parameters from `model`, and everything
    # that reaches through `model.depth_decoder` for a config or a generation
    # setting would land on a wrapper instead.
    decoder = model.depth_decoder
    eager = decoder.forward
    compiled = torch.compile(eager, mode="reduce-overhead", dynamic=False)
    live = True

    def guarded(*args, **kwargs):
        nonlocal live
        if live:
            try:
                return compiled(*args, **kwargs)
            except Exception as error:
                live = False
                decoder.forward = eager
                print(f"graph capture failed, falling back to eager: "
                      f"{type(error).__name__}: {error}")
        return eager(*args, **kwargs)

    decoder.forward = guarded
    return True


__all__ = ["capture", "repair_ptxas"]
