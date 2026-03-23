from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
import wandb
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from .data import load_stage3_prompts
from .reward import RewardConfig, score_scad_rlvr


def _is_local_peft_checkpoint(model_name: str) -> bool:
    p = Path(model_name)
    return p.exists() and (p / "adapter_config.json").exists()
# 这个函数在 stage2 和 stage3 中都用到了，暂时放在这里。后续如果有更多地方需要用到，可以考虑移到 utils.py 或 data.py 中。

def _is_local_path(model_name: str) -> bool:
    return Path(model_name).exists()


def _normalize_completion_text(item: Any) -> str:
    """Normalize completion payloads across TRL versions."""
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        # chat/completion may look like [{"role": "assistant", "content": "..."}]
        if item and isinstance(item[-1], dict) and "content" in item[-1]:
            return str(item[-1]["content"])
        return "\n".join(str(x) for x in item)
    if isinstance(item, dict) and "content" in item:
        return str(item["content"])
    return str(item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage3: GRPO + RLVR reinforcement learning")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    wb_project = cfg.get("wb_project", "trl-cad")
    wb_group = cfg.get("wb_group", "default-experiment")
    wb_notes = cfg.get("wb_notes", "")
    wandb.init(
        project=wb_project,
        name="stage3",
        group=wb_group,
        job_type="stage3_grpo",
        notes=wb_notes,
        config=cfg,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_stage3_prompts(
        cfg["dataset_path"],
        streaming=cfg.get("streaming", False),
        cache_dir=cfg.get("cache_dir", None),
        keep_in_memory=cfg.get("keep_in_memory", False),
        num_proc=cfg.get("num_proc", 1),
        load_from_cache_file=cfg.get("load_from_cache_file", True),
    )

    require_peft_checkpoint = cfg.get("require_peft_checkpoint", False)
    if require_peft_checkpoint and not _is_local_peft_checkpoint(cfg["model_name"]):
        raise ValueError(
            "Stage3 requires continuing from a PEFT checkpoint, but model_name is not a valid adapter directory: {}".format(
                cfg['model_name']
            )
        )

    if _is_local_peft_checkpoint(cfg["model_name"]):
        print("[Stage3] Continuing from PEFT checkpoint: {}".format(cfg['model_name']))
        model = AutoPeftModelForCausalLM.from_pretrained(
            cfg["model_name"],
            is_trainable=True,
            trust_remote_code=True,
        )
        trainer_peft_config = None
    else:
        source_kind = "local base-model directory" if _is_local_path(cfg["model_name"]) else "remote model id"
        print("[Stage3] Loading from {} and attaching new LoRA: {}".format(source_kind, cfg["model_name"]))
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            trust_remote_code=True,
        )
        trainer_peft_config = lora_cfg

    reward_cfg = RewardConfig(
        openscad_bin=cfg.get("openscad_bin", None),
        verify_with_openscad=cfg.get("verify_with_openscad", False),
        min_len=cfg.get("min_len", 40),
        max_len=cfg.get("max_len", 3500),
        dedup_window_size=cfg.get("dedup_window_size", 1024),
    )
    seen_hashes: set[str] = set()

    def rlvr_reward(prompts, completions, raw_prompt=None, **kwargs):
        prompt_refs = raw_prompt if raw_prompt is not None else prompts
        scores: list[float] = []
        for p, c in zip(prompt_refs, completions):
            prompt_text = str(p)
            completion_text = _normalize_completion_text(c)
            score, _ = score_scad_rlvr(
                prompt_text,
                completion_text,
                cfg=reward_cfg,
                seen_hashes=seen_hashes,
            )
            scores.append(float(score))
        return scores

    grpo_config = GRPOConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg.get("batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=cfg.get("learning_rate", 5.0e-6),
        num_train_epochs=cfg.get("grpo_epochs", 1),
        max_prompt_length=cfg.get("max_prompt_length", 512),
        max_completion_length=cfg.get("max_new_tokens", 384),
        num_generations=cfg.get("num_generations", 4),
        beta=cfg.get("beta", 0.04),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 200),
        bf16=cfg.get("bf16", False),
        fp16=cfg.get("fp16", False),
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=rlvr_reward,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=trainer_peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

if __name__ == "__main__":
    main()
