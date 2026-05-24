from __future__ import annotations

import os


def latest_checkpoint(base_dir: str) -> str | None:
    if not os.path.isdir(base_dir):
        return None
    candidates = [d for d in os.listdir(base_dir) if d.startswith("epoch_")]
    if not candidates:
        return None
    candidates.sort(key=lambda name: int(name.split("_")[-1]))
    return os.path.join(base_dir, candidates[-1])
