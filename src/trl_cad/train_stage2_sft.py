from __future__ import annotations

import argparse

import yaml
import wandb

from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from .data import load_stage2_dataset
from .utils import is_local_path, is_local_peft_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage2: instruction SFT prompt -> CoT + SCAD")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    wb_project = cfg.get("wb_project", "trl-cad")
    wb_group = cfg.get("wb_group", "default-experiment")
    wb_notes = cfg.get("wb_notes", "")
    wandb.init(
        project=wb_project,
        name="stage2",
        group=wb_group,
        job_type="stage2_sft",
        notes=wb_notes,
        config=cfg,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_stage2_dataset(
        cfg["dataset_path"],
        streaming=cfg.get("streaming", False),
        cache_dir=cfg.get("cache_dir", None),
        keep_in_memory=cfg.get("keep_in_memory", False),
        cot_field=cfg.get("cot_field", "cot"),
        include_cot=cfg.get("include_cot", True),
        allow_cot_fallback_fields=cfg.get("allow_cot_fallback_fields", True),
        num_proc=cfg.get("num_proc", 1),
        load_from_cache_file=cfg.get("load_from_cache_file", True),
    )

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    require_peft_checkpoint = cfg.get("require_peft_checkpoint", False)
    if require_peft_checkpoint and not is_local_peft_checkpoint(cfg["model_name"]):
        raise ValueError(
            "Stage2 requires continuing from a PEFT checkpoint, but model_name is not a valid adapter directory: {}".format(
                cfg['model_name']
            )
        )

    if is_local_peft_checkpoint(cfg["model_name"]):
        print("[Stage2] Continuing from PEFT checkpoint: {}".format(cfg['model_name']))
        model = AutoPeftModelForCausalLM.from_pretrained(
            cfg["model_name"],
            is_trainable=True,
            trust_remote_code=True,
        )
        trainer_peft_config = None
    else:
        source_kind = "local base-model directory" if is_local_path(cfg["model_name"]) else "remote model id"
        print("[Stage2] Loading from {} and attaching new LoRA: {}".format(source_kind, cfg["model_name"]))
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            trust_remote_code=True,
        )
        trainer_peft_config = lora_cfg

    train_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        bf16=cfg.get("bf16", False),
        fp16=cfg.get("fp16", False),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        report_to="wandb",
    )

    response_template = cfg.get("response_template", "<|assistant|>\n")
    completion_only_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # completion-only loss is incompatible with packing; sample boundaries must be preserved.
    packing = False

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        data_collator=completion_only_collator,
        peft_config=trainer_peft_config,
        args=train_args,
        max_seq_length=cfg["max_seq_length"],
        packing=packing,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
