from __future__ import annotations

from pathlib import Path

from trl_cad.data import load_stage2_dataset, load_stage3_prompts


ROOT = Path(__file__).resolve().parents[1]


def test_stage2_contains_assistant_boundary() -> None:
    ds = load_stage2_dataset(
        str(ROOT / "data" / "samples" / "stage2_instruction.jsonl"),
        streaming=False,
        include_cot=True,
    )
    first = ds[0]["text"]
    assert "<|assistant|>" in first


def test_stage3_prompt_is_templated_and_raw_kept() -> None:
    ds = load_stage3_prompts(
        str(ROOT / "data" / "samples" / "stage3_grpo_prompts.jsonl"),
        streaming=False,
    )
    first = ds[0]
    assert "<|system|>" in first["prompt"]
    assert "<|assistant|>" in first["prompt"]
    assert isinstance(first["raw_prompt"], str) and len(first["raw_prompt"]) > 0
