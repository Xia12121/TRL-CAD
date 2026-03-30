from __future__ import annotations

import argparse
import inspect
import os
import shutil
from typing import Any

import yaml
import wandb
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from .data import load_stage3_prompts
from .reward import RewardConfig, score_scad_rlvr
from .utils import is_local_path, is_local_peft_checkpoint


def _build_grpo_config(cfg: dict[str, Any]) -> GRPOConfig:
    """Build GRPOConfig with best-effort compatibility across TRL versions."""
    params = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    kwargs: dict[str, Any] = {}

    def _set_if_supported(name: str, value: Any) -> None:
        if name in params:
            kwargs[name] = value

    # Core training args
    _set_if_supported("output_dir", cfg["output_dir"])
    _set_if_supported("per_device_train_batch_size", cfg.get("batch_size", 1))
    _set_if_supported("gradient_accumulation_steps", cfg.get("gradient_accumulation_steps", 1))
    _set_if_supported("learning_rate", cfg.get("learning_rate", 5.0e-6))
    _set_if_supported("num_train_epochs", cfg.get("grpo_epochs", 1))
    _set_if_supported("logging_steps", cfg.get("logging_steps", 10))
    _set_if_supported("save_steps", cfg.get("save_steps", 200))
    _set_if_supported("bf16", cfg.get("bf16", False))
    _set_if_supported("fp16", cfg.get("fp16", False))
    _set_if_supported("beta", cfg.get("beta", 0.04))
    _set_if_supported("num_generations", cfg.get("num_generations", 8))

    # Sequence length args vary across TRL versions
    prompt_len = cfg.get("max_prompt_length", 512)
    completion_len = cfg.get("max_new_tokens", 768)

    if "max_prompt_length" in params:
        kwargs["max_prompt_length"] = prompt_len
    elif "max_prompt_tokens" in params:
        kwargs["max_prompt_tokens"] = prompt_len
    else:
        # TRL 0.29.1 may not expose max_prompt_length in some builds.
        # Keep training runnable and rely on tokenizer/model context length.
        print("[Stage3] GRPOConfig has no max_prompt_length field; ignoring config.max_prompt_length")

    if "max_completion_length" in params:
        kwargs["max_completion_length"] = completion_len
    elif "max_new_tokens" in params:
        kwargs["max_new_tokens"] = completion_len
    elif "response_length" in params:
        kwargs["response_length"] = completion_len

    # Fallback for older APIs that only expose max_length
    if "max_length" in params and all(k not in kwargs for k in ("max_completion_length", "max_new_tokens", "response_length")):
        kwargs["max_length"] = prompt_len + completion_len

    # Logging backend control; avoid duplicate wandb runs on non-zero ranks in DDP
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        _set_if_supported("report_to", "wandb")
    else:
        if "report_to" in params:
            kwargs["report_to"] = "none"

    return GRPOConfig(**kwargs)


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
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        wandb.init(
            project=wb_project,
            name="stage3",
            group=wb_group,
            job_type="stage3_grpo",
            notes=wb_notes,
            config=cfg,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    tokenizer.padding_side = "left"
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
    if require_peft_checkpoint and not is_local_peft_checkpoint(cfg["model_name"]):
        raise ValueError(
            "Stage3 requires continuing from a PEFT checkpoint, but model_name is not a valid adapter directory: {}".format(
                cfg['model_name']
            )
        )

    if is_local_peft_checkpoint(cfg["model_name"]):
        print("[Stage3] Continuing from PEFT checkpoint: {}".format(cfg['model_name']))
        model = AutoPeftModelForCausalLM.from_pretrained(
            cfg["model_name"],
            is_trainable=True,
            trust_remote_code=True,
        )
        trainer_peft_config = None
    else:
        source_kind = "local base-model directory" if is_local_path(cfg["model_name"]) else "remote model id"
        print("[Stage3] Loading from {} and attaching new LoRA: {}".format(source_kind, cfg["model_name"]))
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            trust_remote_code=True,
        )
        trainer_peft_config = lora_cfg

    reward_cfg = RewardConfig(
        openscad_bin=cfg.get("openscad_bin", None) or shutil.which("openscad"),
        verify_with_openscad=cfg.get("verify_with_openscad", True),
        openscad_timeout_sec=cfg.get("openscad_timeout_sec", 20),
        compile_non_empty_reward=cfg.get("compile_non_empty_reward", 1.2),
        compile_failure_penalty=cfg.get("compile_failure_penalty", -1.0),
        compile_empty_geometry_penalty=cfg.get("compile_empty_geometry_penalty", -1.2),
        format_ok_reward=cfg.get("format_ok_reward", 0.3),
        format_missing_think_penalty=cfg.get("format_missing_think_penalty", -0.3),
        semantic_model_path=cfg.get("semantic_model_path", None),
        semantic_similarity_weight=cfg.get("semantic_similarity_weight", 1.0),
        semantic_unavailable_reward=cfg.get("semantic_unavailable_reward", 0.0),
        semantic_max_length=cfg.get("semantic_max_length", 256),
    )

    if not reward_cfg.verify_with_openscad:
        raise ValueError(
            "Stage3 reward requires OpenSCAD verification. Set verify_with_openscad=true."
        )
    if not reward_cfg.openscad_bin:
        raise ValueError(
            "OpenSCAD executable not found. Set openscad_bin explicitly or install openscad in PATH."
        )

    require_semantic_model = cfg.get("require_semantic_model", True)
    if require_semantic_model and not reward_cfg.semantic_model_path:
        raise ValueError("Stage3 requires semantic_model_path when require_semantic_model=true")

    def rlvr_reward(prompts, completions, raw_prompt=None, **kwargs):
        prompt_refs = raw_prompt if raw_prompt is not None else prompts
        prompt_texts = [str(p) for p in prompt_refs]
        completion_texts = [_normalize_completion_text(c) for c in completions]

        scores: list[float] = []
        for prompt_text, completion_text in zip(prompt_texts, completion_texts):
            score, _ = score_scad_rlvr(
                prompt_text,
                completion_text,
                cfg=reward_cfg,
                seen_hashes=None,
            )
            scores.append(float(score))
        return scores

    grpo_config = _build_grpo_config(cfg)

    trainer_kwargs = dict(
        model=model,
        reward_funcs=rlvr_reward,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=trainer_peft_config,
    )
    trainer_params = set(inspect.signature(GRPOTrainer.__init__).parameters.keys())
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    if rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()
