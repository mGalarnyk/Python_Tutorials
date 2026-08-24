"""Make mamba-ssm importable without compiling selective_scan_cuda.

Nemotron-H / Nemotron-3-Nano uses the Triton Mamba-2 kernels
(`mamba_ssm.ops.triton.*`). The Mamba-1 CUDA extension is unused.
"""

from __future__ import annotations

from pathlib import Path


NEEDLE = "import selective_scan_cuda\n"
PATCH = (
    "try:\n"
    "    import selective_scan_cuda\n"
    "except ImportError:\n"
    "    selective_scan_cuda = None\n"
)


def patch() -> Path | None:
    import mamba_ssm

    path = Path(mamba_ssm.__file__).resolve().parent / "ops" / "selective_scan_interface.py"
    text = path.read_text()
    if "selective_scan_cuda = None" in text:
        return path
    if NEEDLE not in text:
        raise RuntimeError(f"Could not find {NEEDLE!r} in {path}")
    path.write_text(text.replace(NEEDLE, PATCH, 1))
    return path


if __name__ == "__main__":
    print(f"patched {patch()}")
