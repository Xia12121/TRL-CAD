from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass

SCAD_KEYWORDS = {
    "cube",
    "sphere",
    "cylinder",
    "translate",
    "rotate",
    "scale",
    "difference",
    "union",
    "intersection",
    "linear_extrude",
    "polygon",
    "module",
}


@dataclass
class RewardConfig:
    openscad_bin: str | None = None
    verify_with_openscad: bool = False
    min_len: int = 40
    max_len: int = 3500
    dedup_window_size: int = 1024


def _balanced_brackets(code: str) -> bool:
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for ch in code:
        if ch in '([{':
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack


def _hash_prefix(text: str, n: int) -> str:
    return hashlib.md5(text[:n].encode("utf-8", errors="ignore")).hexdigest()


def _verify_with_openscad(code: str, openscad_bin: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="trl_cad_rlvr_") as td:
        scad_path = os.path.join(td, "candidate.scad")
        out_path = os.path.join(td, "candidate.stl")
        with open(scad_path, "w", encoding="utf-8") as f:
            f.write(code)

        proc = subprocess.run(
            [openscad_bin, "-o", out_path, scad_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
        ok = proc.returncode == 0 and os.path.exists(out_path)
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return ok, log.strip()


def _extract_prompt_keywords(query: str) -> set[str]:
    words = {w.lower() for w in re.findall(r"[a-zA-Z_]+", query) if len(w) > 2}
    # 若是中文 prompt，保留基础 CAD 关键词提示（弱监督）
    for kw in ["hex", "honeycomb", "vase", "phone", "stand", "thickness", "hole"]:
        if kw in query.lower():
            words.add(kw)
    return words


def score_scad_rlvr(
    query: str,
    generated: str,
    *,
    cfg: RewardConfig | None = None,
    seen_hashes: set[str] | None = None,
) -> tuple[float, dict]:
    """RLVR 奖励：可验证约束 + 语义相关 + 创造性控制。"""
    cfg = cfg or RewardConfig()
    text = generated.strip()
    info: dict[str, float | int | bool | str] = {}

    if not text:
        return -1.5, {"empty": True}

    reward = 0.0

    # A) 可验证语法约束（deterministic）
    balanced = _balanced_brackets(text)
    info["balanced_brackets"] = balanced
    reward += 1.0 if balanced else -1.2

    semicolon_ok = text.count(";") >= 1
    info["has_semicolon"] = semicolon_ok
    reward += 0.35 if semicolon_ok else -0.2

    # B) 关键词覆盖（弱可验证）
    found = sum(1 for k in SCAD_KEYWORDS if re.search(rf"\b{k}\b", text))
    info["keyword_found"] = found
    reward += min(found / 6.0, 1.0)

    # C) 与 prompt 的词重合（弱相关）
    q_words = _extract_prompt_keywords(query)
    g_words = {w.lower() for w in re.findall(r"[a-zA-Z_]+", text) if len(w) > 2}
    overlap = len(q_words & g_words)
    info["prompt_overlap"] = overlap
    reward += min(overlap / 10.0, 0.8)

    # D) 长度约束（可验证）
    n = len(text)
    info["length"] = n
    if n < cfg.min_len:
        reward -= 0.8
    elif n > cfg.max_len:
        reward -= 0.5

    # E) 轻度创造性奖励
    has_module = "module" in text
    transforms = text.count("translate") + text.count("rotate") + text.count("scale")
    info["has_module"] = has_module
    info["transform_count"] = transforms
    if has_module:
        reward += 0.25
    if transforms >= 2:
        reward += 0.25

    # F) 去重惩罚，缓解 reward hacking（重复模板刷分）
    if seen_hashes is not None:
        prefix_hash = _hash_prefix(text, cfg.dedup_window_size)
        duplicated = prefix_hash in seen_hashes
        info["dedup_hit"] = duplicated
        if duplicated:
            reward -= 0.5
        else:
            seen_hashes.add(prefix_hash)

    # G) 外部验证（强 RLVR，可选）
    if cfg.verify_with_openscad and cfg.openscad_bin:
        ok, _ = _verify_with_openscad(text, cfg.openscad_bin)
        info["openscad_compile_ok"] = ok
        reward += 1.5 if ok else -1.5

    return float(reward), info


def score_scad(query: str, generated: str) -> float:
    # 兼容旧接口
    score, _ = score_scad_rlvr(query, generated)
    return score
