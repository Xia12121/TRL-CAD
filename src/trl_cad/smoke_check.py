from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .data import load_stage1_dataset, load_stage2_dataset, load_stage3_prompts


ROOT = Path(__file__).resolve().parents[2]


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path)


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_model_chain(cfg: dict, stage_name: str) -> None:
    model_name = str(cfg.get("model_name", "")).strip()
    if not model_name:
        raise ValueError("{}: model_name must not be empty".format(stage_name))

    local = _resolve(model_name)
    if local.exists():
        adapter_cfg = local / "adapter_config.json"
        if adapter_cfg.exists():
            print("[{}] model_name points to local PEFT adapter: {}".format(stage_name, local))
        else:
            print("[{}] model_name points to local directory (base model or ordinary folder): {}".format(stage_name, local))
    else:
        print("[{}] model_name treated as remote model id: {}".format(stage_name, model_name))


def main() -> None:
    parser = argparse.ArgumentParser(description="TRL-CAD minimal smoke/dry-run checks")
    parser.add_argument("--stage1", default="configs/stage1.yaml")
    parser.add_argument("--stage2", default="configs/stage2.yaml")
    parser.add_argument("--stage3", default="configs/stage3.yaml")
    parser.add_argument("--sample_rows", type=int, default=2)
    args = parser.parse_args()

    stage1_cfg = _read_yaml(_resolve(args.stage1))
    stage2_cfg = _read_yaml(_resolve(args.stage2))
    stage3_cfg = _read_yaml(_resolve(args.stage3))

    _check_model_chain(stage2_cfg, "Stage2")
    _check_model_chain(stage3_cfg, "Stage3")

    ds1 = load_stage1_dataset(stage1_cfg["dataset_path"], streaming=True)
    ds2 = load_stage2_dataset(
        stage2_cfg["dataset_path"],
        streaming=True,
        cot_field=stage2_cfg.get("cot_field", "cot"),
        include_cot=stage2_cfg.get("include_cot", True),
        allow_cot_fallback_fields=stage2_cfg.get("allow_cot_fallback_fields", True),
    )
    ds3 = load_stage3_prompts(stage3_cfg["dataset_path"], streaming=True)

    print("\n[Stage1] sample:")
    for i, row in enumerate(ds1):
        print(row)
        if i + 1 >= args.sample_rows:
            break

    print("\n[Stage2] sample(text should contain <|assistant|>):")
    for i, row in enumerate(ds2):
        print(row["text"][:220])
        if i + 1 >= args.sample_rows:
            break

    print("\n[Stage3] sample(prompt should be templated query):")
    for i, row in enumerate(ds3):
        print({"prompt": row["prompt"][:160], "raw_prompt": row["raw_prompt"]})
        if i + 1 >= args.sample_rows:
            break

    print("\nSmoke check passed.")


if __name__ == "__main__":
    main()
