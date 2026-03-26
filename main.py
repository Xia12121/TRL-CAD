from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run_module(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print("[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRL-CAD unified entrypoint: train/infer across Stage1-Stage3"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser("stage1", help="Stage1: SCAD corpus pretraining")
    p1.add_argument("--config", default="configs/stage1.yaml")

    p2 = subparsers.add_parser("stage2", help="Stage2: instruction SFT with CoT")
    p2.add_argument("--config", default="configs/stage2.yaml")

    p3 = subparsers.add_parser("stage3", help="Stage3: GRPO + RLVR training")
    p3.add_argument("--config", default="configs/stage3.yaml")

    psem = subparsers.add_parser("semantic", help="Train semantic CLIP-style scorer")
    psem.add_argument("--config", default="configs/semantic_clip.yaml")

    pg = subparsers.add_parser("generate", help="Generate SCAD code from prompt")
    pg.add_argument("--model", default="outputs/stage3")
    pg.add_argument("--prompt", required=True)
    pg.add_argument("--max_new_tokens", type=int, default=512)

    pp = subparsers.add_parser("pipeline", help="Run full pipeline: stage1 -> stage2 -> semantic -> stage3")
    pp.add_argument("--stage1_config", default="configs/stage1.yaml")
    pp.add_argument("--stage2_config", default="configs/stage2.yaml")
    pp.add_argument("--semantic_config", default="configs/semantic_clip.yaml")
    pp.add_argument("--stage3_config", default="configs/stage3.yaml")

    ps = subparsers.add_parser("smoke", help="Run minimal smoke/dry-run checks")
    ps.add_argument("--stage1_config", default="configs/stage1.yaml")
    ps.add_argument("--stage2_config", default="configs/stage2.yaml")
    ps.add_argument("--stage3_config", default="configs/stage3.yaml")

    args = parser.parse_args()

    if args.command == "stage1":
        _run_module("trl_cad.train_stage1_lm", "--config", args.config)
    elif args.command == "stage2":
        _run_module("trl_cad.train_stage2_sft", "--config", args.config)
    elif args.command == "stage3":
        _run_module("trl_cad.train_stage3_grpo", "--config", args.config)
    elif args.command == "semantic":
        _run_module("trl_cad.train_semantic_clip", "--config", args.config)
    elif args.command == "generate":
        _run_module(
            "trl_cad.generate",
            "--model",
            args.model,
            "--prompt",
            args.prompt,
            "--max_new_tokens",
            str(args.max_new_tokens),
        )
    elif args.command == "pipeline":
        _run_module("trl_cad.train_stage1_lm", "--config", args.stage1_config)
        _run_module("trl_cad.train_stage2_sft", "--config", args.stage2_config)
        _run_module("trl_cad.train_semantic_clip", "--config", args.semantic_config)
        _run_module("trl_cad.train_stage3_grpo", "--config", args.stage3_config)
    elif args.command == "smoke":
        _run_module(
            "trl_cad.smoke_check",
            "--stage1",
            args.stage1_config,
            "--stage2",
            args.stage2_config,
            "--stage3",
            args.stage3_config,
        )

if __name__ == "__main__":
    main()