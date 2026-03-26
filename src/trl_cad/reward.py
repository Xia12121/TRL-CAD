from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass

from .semantic_reward import compute_semantic_similarity

SCAD_START_PATTERNS = [
    r"module\s+[A-Za-z_]\w*\s*\(",
    r"function\s+[A-Za-z_]\w*\s*\(",
    r"for\s*\(",
    r"if\s*\(",
    r"let\s*\(",
    r"translate\s*\(",
    r"rotate\s*\(",
    r"scale\s*\(",
    r"mirror\s*\(",
    r"multmatrix\s*\(",
    r"color\s*\(",
    r"difference\s*\(",
    r"union\s*\(",
    r"intersection\s*\(",
    r"hull\s*\(",
    r"minkowski\s*\(",
    r"offset\s*\(",
    r"resize\s*\(",
    r"projection\s*\(",
    r"render\s*\(",
    r"linear_extrude\s*\(",
    r"rotate_extrude\s*\(",
    r"surface\s*\(",
    r"import\s*\(",
    r"polygon\s*\(",
    r"polyhedron\s*\(",
    r"text\s*\(",
    r"cube\s*\(",
    r"sphere\s*\(",
    r"cylinder\s*\(",
    r"circle\s*\(",
    r"square\s*\(",
    r"[A-Za-z_]\w*\s*=",
]

SCAD_START_RE = re.compile("|".join(f"(?:{p})" for p in SCAD_START_PATTERNS), re.IGNORECASE)


@dataclass
class RewardConfig:
    """简化版奖励配置：仅保留 3 个主奖励。"""
    openscad_bin: str | None = None
    verify_with_openscad: bool = False
    openscad_timeout_sec: int = 20
    compile_non_empty_reward: float = 1.2
    compile_failure_penalty: float = -1.0
    compile_empty_geometry_penalty: float = -1.2
    format_ok_reward: float = 0.3
    format_missing_think_penalty: float = -0.3
    semantic_model_path: str | None = None
    semantic_similarity_weight: float = 1.0
    semantic_unavailable_reward: float = 0.0
    semantic_max_length: int = 256


@dataclass
class ParsedOutput:
    """模型输出解析结果。

    think: 推理文本（若存在）
    scad: 代码文本（后续评分只基于它）
    format_clean: 是否是标准 <think>...</think> + SCAD 格式
    """
    think: str
    scad: str
    format_clean: bool


def parse_think_and_scad(text: str) -> ParsedOutput:
    """从模型输出中分离 think 与 SCAD。

    解析策略：
    1) 标准格式：<think>...</think> + 代码
    2) think 未闭合：尝试用 SCAD 起始模式切分
    3) 无 think 标签：整段当作 SCAD
    """
    text = text.strip()

    match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if match:
        return ParsedOutput(
            think=match.group(1).strip(),
            scad=match.group(2).strip(),
            format_clean=True,
        )

    match = re.search(r"<think>(.*)", text, re.DOTALL)
    if match:
        remaining = match.group(1)
        code_start = SCAD_START_RE.search(remaining)
        if code_start:
            return ParsedOutput(
                think=remaining[: code_start.start()].strip(),
                scad=remaining[code_start.start() :].strip(),
                format_clean=False,
            )

        has_code_markers = (
            any(k in remaining.lower() for k in ["cube", "sphere", "cylinder", "difference", "union", "translate"])
            or ";" in remaining
            or "{" in remaining
            or "}" in remaining
        )
        if has_code_markers:
            return ParsedOutput(think="", scad=remaining.strip(), format_clean=False)

        return ParsedOutput(think=remaining.strip(), scad="", format_clean=False)

    return ParsedOutput(think="", scad=text, format_clean=False)


def _verify_with_openscad(code: str, openscad_bin: str, timeout_sec: int) -> tuple[bool, str]:
    """调用 OpenSCAD 编译验证。

    返回：
    - ok: 是否成功生成 STL
    - log: stdout+stderr 合并日志
    """
    with tempfile.TemporaryDirectory(prefix="trl_cad_rlvr_") as td:
        scad_path = os.path.join(td, "candidate.scad")
        out_path = os.path.join(td, "candidate.stl")
        with open(scad_path, "w", encoding="utf-8") as f:
            f.write(code)

# CMU 多邻国
        proc = subprocess.run(
            [openscad_bin, "-o", out_path, scad_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(1, int(timeout_sec)),
        )
        ok = proc.returncode == 0 and os.path.exists(out_path)
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return ok, log.strip()


def _parse_compile_signal(ok: bool, log: str) -> dict[str, bool]:
    """将 OpenSCAD 日志解析成结构化信号。"""
    log_lower = log.lower()
    syntax_error = "syntax error" in log_lower or "parser error" in log_lower or "error" in log_lower
    empty_geometry = (
        "top level object is empty" in log_lower
        or "current top level object is empty" in log_lower
    )
    return {
        "compile_ok": ok,
        "syntax_error": syntax_error,
        "empty_geometry": empty_geometry,
    }


def score_scad_rlvr(
    query: str,
    generated: str,
    *,
    cfg: RewardConfig | None = None,
    seen_hashes: set[str] | None = None,
) -> tuple[float, dict]:
    """主奖励函数：仅 3 项。

    1) OpenSCAD 编译 + 非空几何
    2) 输出格式（是否有 `<think>...</think>`）
    3) 语义相似度（BERT-CLIP 风格模型）
    """
    cfg = cfg or RewardConfig()
    info: dict[str, float | int | bool | str] = {}
    raw_text = generated.strip()

    if not raw_text:
        return -1.5, {"empty": True}

    parsed = parse_think_and_scad(raw_text)
    scad_text = parsed.scad
    info["format_clean"] = parsed.format_clean
    info["has_think"] = bool(parsed.think)

    reward = 0.0

    # 1) 输出格式 reward
    if parsed.format_clean and parsed.think and scad_text:
        reward += cfg.format_ok_reward
        info["format_reward"] = cfg.format_ok_reward
    else:
        reward += cfg.format_missing_think_penalty
        info["format_reward"] = cfg.format_missing_think_penalty

    if not scad_text:
        info["scad_empty"] = True
        return float(reward - 1.2), info

    # 2) 编译 + 空几何 reward
    if cfg.verify_with_openscad and cfg.openscad_bin:
        ok, log = _verify_with_openscad(scad_text, cfg.openscad_bin, cfg.openscad_timeout_sec)
        compile_signal = _parse_compile_signal(ok, log)
        info.update({f"openscad_{k}": v for k, v in compile_signal.items()})

        if compile_signal["compile_ok"] and not compile_signal["empty_geometry"]:
            reward += cfg.compile_non_empty_reward
            info["compile_reward"] = cfg.compile_non_empty_reward
        elif compile_signal["empty_geometry"]:
            reward += cfg.compile_empty_geometry_penalty
            info["compile_reward"] = cfg.compile_empty_geometry_penalty
        else:
            reward += cfg.compile_failure_penalty
            info["compile_reward"] = cfg.compile_failure_penalty

        info["openscad_log_excerpt"] = log[-600:]
    else:
        info["compile_skipped"] = True

    # 3) 语义 reward（BERT 双塔 + CLIP 对比学习模型）
    semantic_reward = cfg.semantic_unavailable_reward
    semantic_similarity = None
    if cfg.semantic_model_path:
        semantic_similarity = compute_semantic_similarity(
            query,
            scad_text,
            model_path=cfg.semantic_model_path,
            max_length=cfg.semantic_max_length,
        )
        if semantic_similarity is not None:
            semantic_reward = cfg.semantic_similarity_weight * semantic_similarity
            info["semantic_available"] = True
            info["semantic_similarity"] = float(semantic_similarity)
        else:
            info["semantic_available"] = False
    else:
        info["semantic_available"] = False

    reward += semantic_reward
    info["semantic_reward"] = float(semantic_reward)

    _ = seen_hashes  # kept for backward signature compatibility

    return float(reward), info


def score_scad(query: str, generated: str) -> float:
    # 兼容旧接口：仅返回标量分数
    score, _ = score_scad_rlvr(query, generated)
    return score
