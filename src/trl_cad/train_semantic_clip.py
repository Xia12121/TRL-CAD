from __future__ import annotations

import argparse

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
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

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    batch_size = int(cfg.get("batch_size", 16))
    epochs = int(cfg.get("epochs", 2))
    learning_rate = float(cfg.get("learning_rate", 2.0e-5))

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

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.train()
    global_step = 0
    logging_steps = max(1, int(cfg.get("logging_steps", 20)))

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"semantic_clip epoch {epoch + 1}/{epochs}")
        for batch in pbar:
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
            optimizer.zero_grad()

            global_step += 1
            if global_step % logging_steps == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    save_semantic_model(model, tokenizer, cfg["output_dir"])
    print(f"[Semantic-CLIP] model saved to: {cfg['output_dir']}")


if __name__ == "__main__":
    main()
