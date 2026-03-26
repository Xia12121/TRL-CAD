from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class SemanticClipModelConfig:
    base_model_name: str
    projection_dim: int = 256
    temperature: float = 0.07
    max_length: int = 256


class ClipStyleSemanticModel(nn.Module):
    """BERT-like dual-encoder trained with CLIP-style contrastive objective."""

    def __init__(self, cfg: SemanticClipModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.prompt_encoder = AutoModel.from_pretrained(cfg.base_model_name)
        self.code_encoder = AutoModel.from_pretrained(cfg.base_model_name)

        hidden_size = int(self.prompt_encoder.config.hidden_size)
        self.prompt_proj = nn.Linear(hidden_size, cfg.projection_dim)
        self.code_proj = nn.Linear(hidden_size, cfg.projection_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / cfg.temperature).log())

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def encode_prompt(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.prompt_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(out.last_hidden_state, attention_mask)
        emb = self.prompt_proj(pooled)
        return F.normalize(emb, dim=-1)

    def encode_code(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.code_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(out.last_hidden_state, attention_mask)
        emb = self.code_proj(pooled)
        return F.normalize(emb, dim=-1)

    def forward(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        code_input_ids: torch.Tensor,
        code_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        prompt_emb = self.encode_prompt(prompt_input_ids, prompt_attention_mask)
        code_emb = self.encode_code(code_input_ids, code_attention_mask)
        scale = self.logit_scale.exp().clamp(max=100)
        return scale * (prompt_emb @ code_emb.T)


def clip_contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i + loss_t)


def save_semantic_model(
    model: ClipStyleSemanticModel,
    tokenizer,
    output_dir: str,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    torch.save(model.state_dict(), out_dir / "semantic_clip.pt")
    with (out_dir / "semantic_clip_config.json").open("w", encoding="utf-8") as f:
        json.dump(model.cfg.__dict__, f, ensure_ascii=False, indent=2)


class SemanticScorer:
    def __init__(self, model_path: str, device: str | None = None) -> None:
        self.model_path = str(model_path)
        path = Path(model_path)
        cfg_path = path / "semantic_clip_config.json"
        state_path = path / "semantic_clip.pt"
        if not cfg_path.exists() or not state_path.exists():
            raise FileNotFoundError(f"semantic model files not found in {path}")

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = SemanticClipModelConfig(**json.load(f))

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClipStyleSemanticModel(cfg)
        state = torch.load(state_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def similarity(self, prompt: str, scad_code: str, *, max_length: int | None = None) -> float:
        max_len = int(max_length or self.model.cfg.max_length)
        prompt_inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        code_inputs = self.tokenizer(
            scad_code,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_emb = self.model.encode_prompt(
            prompt_inputs["input_ids"].to(self.device),
            prompt_inputs["attention_mask"].to(self.device),
        )
        code_emb = self.model.encode_code(
            code_inputs["input_ids"].to(self.device),
            code_inputs["attention_mask"].to(self.device),
        )
        cosine = F.cosine_similarity(prompt_emb, code_emb).item()
        return (cosine + 1.0) * 0.5


_SCORER_CACHE: dict[str, SemanticScorer] = {}


def compute_semantic_similarity(
    prompt: str,
    scad_code: str,
    *,
    model_path: str,
    max_length: int = 256,
) -> float | None:
    try:
        scorer = _SCORER_CACHE.get(model_path)
        if scorer is None:
            scorer = SemanticScorer(model_path)
            _SCORER_CACHE[model_path] = scorer
        return scorer.similarity(prompt, scad_code, max_length=max_length)
    except Exception:
        return None
