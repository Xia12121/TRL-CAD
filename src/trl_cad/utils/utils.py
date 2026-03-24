from __future__ import annotations

from pathlib import Path


def is_local_peft_checkpoint(model_name: str) -> bool:
    p = Path(model_name)
    return p.exists() and (p / "adapter_config.json").exists()


def is_local_path(model_name: str) -> bool:
    return Path(model_name).exists()
