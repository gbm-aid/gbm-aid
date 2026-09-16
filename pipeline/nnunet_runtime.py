"""nnU-Net komutlarını aktif Python ortamından güvenli biçimde bul."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def find_nnunet_command(name: str) -> str | None:
    """PATH veya ``sys.executable`` yanındaki Scripts/bin dizininde komut bul."""

    from_path = shutil.which(name)
    if from_path is not None:
        return from_path

    executable_dir = Path(sys.executable).resolve().parent
    candidates = (
        executable_dir / f"{name}.exe",
        executable_dir / name,
        executable_dir / "Scripts" / f"{name}.exe",
        executable_dir / "bin" / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None
