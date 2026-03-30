from __future__ import annotations

import argparse
import os
from math import cos, pi

import torch
import yaml
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, random_split
from tqdm import tqdm
from transformers import AutoTokenizer

from .data import load_stage2_pairs
from .semantic_reward import (
    ClipStyleSemanticModel,
    SemanticClipModelConfig,
    clip_contrastive_loss,
    save_semantic_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BERT-like semantic scorer with CLIP-style contrastive loss")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # DDP setup
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        torch.distributed.init_process_group(backend="nccl")
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Only rank 0 logs to wandb
    if rank == 0:
        wb_project = cfg.get("wb_project", "trl-cad")
        wb_group = cfg.get("wb_group", "default-experiment")
        wb_notes = cfg.get("wb_notes", "")
        wandb.init(
            project=wb_project,
            name="semantic_clip",
            group=wb_group,
            job_type="semantic_clip_training",
            notes=wb_notes,
            config=cfg,
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    ds = load_stage2_pairs(
        cfg["dataset_path"],
        streaming=cfg.get("streaming", False),
        cache_dir=cfg.get("cache_dir", None),
        keep_in_memory=cfg.get("keep_in_memory", False),
    )

    if not hasattr(ds, "__len__"):
        raise ValueError("semantic CLIP training requires non-streaming dataset")

    base_model_name = cfg.get("model_name", "bert-base-uncased")
    model_cfg = SemanticClipModelConfig(
        base_model_name=base_model_name,
        projection_dim=cfg.get("projection_dim", 256),
        temperature=cfg.get("temperature", 0.07),
        max_length=cfg.get("max_length", 256),
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = ClipStyleSemanticModel(model_cfg)
    model.to(device)

    batch_size = int(cfg.get("batch_size", 32))
    epochs = int(cfg.get("epochs", 2))
    learning_rate = float(cfg.get("learning_rate", 2.0e-5))
    
    # Partial fine-tuning setup
    freeze_encoder_layers = cfg.get("freeze_encoder_layers", -1)  # -1 means no freezing
    if freeze_encoder_layers > 0:
        # Freeze first N layers of both encoders
        for i, layer in enumerate(model.prompt_encoder.encoder.layer):
            if i < freeze_encoder_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        for i, layer in enumerate(model.code_encoder.encoder.layer):
            if i < freeze_encoder_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        if rank == 0:
            print(f"[Semantic-CLIP] Froze first {freeze_encoder_layers} encoder layers")
    
    # Calculate number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"[Semantic-CLIP] Total params: {total_params:,}, Trainable: {trainable_params:,}")
        print(f"[Semantic-CLIP] Trainable ratio: {100*trainable_params/total_params:.1f}%")

    def _collate(examples: list[dict]) -> dict[str, torch.Tensor]:
        prompts = [str(x["prompt"]) for x in examples]
        codes = [str(x["scad_code"]) for x in examples]
        prompt_inputs = tokenizer(
            prompts,
            truncation=True,
            max_length=model_cfg.max_length,
            padding=True,
            return_tensors="pt",
        )
        code_inputs = tokenizer(
            codes,
            truncation=True,
            max_length=model_cfg.max_length,
            padding=True,
            return_tensors="pt",
        )
        return {
            "prompt_input_ids": prompt_inputs["input_ids"],
            "prompt_attention_mask": prompt_inputs["attention_mask"],
            "code_input_ids": code_inputs["input_ids"],
            "code_attention_mask": code_inputs["attention_mask"],
        }

    # Split dataset for validation
    val_ratio = cfg.get("val_ratio", 0.1)
    if val_ratio > 0 and len(ds) > 10:
        train_size = int(len(ds) * (1 - val_ratio))
        val_size = len(ds) - train_size
        train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    else:
        train_ds = ds
        val_ds = None

    # DDP sampler for data distribution
    sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
    ) if is_ddp else None

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=_collate,
    )
    
    # Validation dataloader (no distributed sampling for simplicity)
    val_loader = None
    if val_ds is not None and rank == 0:
        val_sampler = None
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=(val_sampler is None),
            collate_fn=_collate,
        )

    # Wrap model with DDP
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    
    # Optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    
    # Warmup + Cosine decay scheduler
    num_training_steps = len(loader) * epochs
    num_warmup_steps = int(cfg.get("warmup_ratio", 0.1) * num_training_steps)
    
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + cos(pi * float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps)))))
    
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    if rank == 0:
        print(f"[Semantic-CLIP] Total training steps: {num_training_steps}")
        print(f"[Semantic-CLIP] Warmup steps: {num_warmup_steps}")
        print(f"[Semantic-CLIP] Batch size: {batch_size} (per GPU)")
        print(f"[Semantic-CLIP] Effective batch size (all GPUs): {batch_size * world_size}")

    model.train()
    global_step = 0
    logging_steps = max(1, int(cfg.get("logging_steps", 20)))
    best_val_loss = float("inf")
    patience = cfg.get("early_stopping_patience", 3)
    patience_counter = 0

    for epoch in range(epochs):
        if is_ddp:
            sampler.set_epoch(epoch)
        
        # Training loop
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"semantic_clip epoch {epoch + 1}/{epochs}", disable=(rank != 0))
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                prompt_input_ids=batch["prompt_input_ids"],
                prompt_attention_mask=batch["prompt_attention_mask"],
                code_input_ids=batch["code_input_ids"],
                code_attention_mask=batch["code_attention_mask"],
            )
            loss = clip_contrastive_loss(logits)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()
            
            if global_step % logging_steps == 0:
                loss_value = loss.item()
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{loss_value:.4f}",
                    "lr": f"{current_lr:.2e}",
                })
                if rank == 0:
                    wandb.log({
                        "train/loss": loss_value,
                        "train/lr": current_lr,
                        "train/epoch": epoch,
                        "step": global_step,
                    })
        
        # Validation loop
        if val_loader is not None and rank == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc="Validation", disable=(rank != 0))
                for batch in val_pbar:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    logits = model(
                        prompt_input_ids=batch["prompt_input_ids"],
                        prompt_attention_mask=batch["prompt_attention_mask"],
                        code_input_ids=batch["code_input_ids"],
                        code_attention_mask=batch["code_attention_mask"],
                    )
                    loss = clip_contrastive_loss(logits)
                    val_loss += loss.item()
                    val_steps += 1
                    val_pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
            
            avg_val_loss = val_loss / max(1, val_steps)
            avg_train_loss = epoch_loss / max(1, batch_idx + 1)
            
            wandb.log({
                "val/loss": avg_val_loss,
                "train/epoch_loss": avg_train_loss,
                "epoch": epoch,
            })
            
            print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(f"[Semantic-CLIP] Validation loss improved to {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"[Semantic-CLIP] Validation loss did not improve. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"[Semantic-CLIP] Early stopping triggered after {epoch + 1} epochs")
                    break

    if rank == 0:
        # Extract base model from DDP wrapper if needed
        save_model = model.module if is_ddp else model
        save_semantic_model(save_model, tokenizer, cfg["output_dir"])
        print(f"[Semantic-CLIP] model saved to: {cfg['output_dir']}")
        wandb.finish()

    if is_ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
