from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from collections import Counter
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

SCAD_BUILTIN_CALLS = {
    "cube",
    "sphere",
    "cylinder",
    "polyhedron",
    "polygon",
    "circle",
    "square",
    "text",
    "import",
    "surface",
    "translate",
    "rotate",
    "scale",
    "resize",
    "mirror",
    "multmatrix",
    "color",
    "offset",
    "hull",
    "minkowski",
    "render",
    "projection",
    "linear_extrude",
    "rotate_extrude",
    "union",
    "difference",
    "intersection",
    "for",
    "if",
    "let",
}

ERROR_TOKEN_RE = re.compile(r"\b(error|errors)\b")
NEGATIVE_ERROR_PHRASE_RE = re.compile(r"\b(?:no|without|zero|0)\s+errors?\b")


def _strip_scad_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    return code


@dataclass
class RewardConfig:
    openscad_bin: str | None = None
    verify_with_openscad: bool = False
    min_len: int = 40
    max_len: int = 3500
    dedup_window_size: int = 1024
    # compile-tier rewards
    compile_success_reward: float = 1.0
    compile_syntax_error_penalty: float = -1.0
    compile_warning_penalty: float = -0.35
    compile_empty_geometry_penalty: float = -0.2
    compile_runtime_error_penalty: float = -0.6
    # static-analysis shaping
    module_definition_bonus: float = 0.2
    module_call_bonus: float = 0.2
    for_loop_bonus: float = 0.2
    transform_diversity_bonus: float = 0.2
    hardcode_repeat_penalty: float = -0.25
    hardcode_repeat_threshold: int = 6
    dedup_repeat_penalty: float = 0.0

PROMPT_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "using",
    "make",
    "create",
    "build",
    "model",
    "design",
    "object",
    "shape",
    "please",
    "need",
    "want",
    "add",
    "then",
    "generate",
}


@dataclass
class ParsedOutput:
    think: str
    scad: str
    format_clean: bool


def parse_think_and_scad(text: str) -> ParsedOutput:
    """Split model output into think content and SCAD code."""
    text = text.strip()

    # case 1: clean <think>...</think> + scad
    match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if match:
        return ParsedOutput(
            think=match.group(1).strip(),
            scad=match.group(2).strip(),
            format_clean=True,
        )

    # case 2: open <think> only, try to find SCAD start
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

        # fallback: if it already looks like code-ish text, avoid swallowing valid SCAD.
        has_code_markers = (
            any(k in remaining.lower() for k in SCAD_KEYWORDS)
            or ";" in remaining
            or "{" in remaining
            or "}" in remaining
        )
        if has_code_markers:
            return ParsedOutput(think="", scad=remaining.strip(), format_clean=False)

        return ParsedOutput(think=remaining.strip(), scad="", format_clean=False)

    # case 3: no think tag, treat all as SCAD
    return ParsedOutput(think="", scad=text, format_clean=False)


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


def _bracket_balance_score(code: str) -> tuple[float, int]:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    mismatch = 0
    for ch in code:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if stack and stack[-1] == pairs[ch]:
                stack.pop()
            else:
                mismatch += 1
    imbalance = mismatch + len(stack)
    if imbalance == 0:
        return 1.0, 0
    return max(-1.2, -0.3 * imbalance), imbalance


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


def _parse_compile_signal(ok: bool, log: str) -> dict[str, bool]:
    log_lower = log.lower()
    syntax_error = "syntax error" in log_lower or "parser error" in log_lower
    has_warning = "warning" in log_lower
    has_error = bool(ERROR_TOKEN_RE.search(log_lower)) and not bool(NEGATIVE_ERROR_PHRASE_RE.search(log_lower))
    empty_geometry = (
        "top level object is empty" in log_lower
        or "current top level object is empty" in log_lower
    )
    return {
        "compile_ok": ok,
        "syntax_error": syntax_error,
        "has_warning": has_warning,
        "has_error": has_error,
        "empty_geometry": empty_geometry,
    }


def _transform_stats(code: str) -> tuple[int, int]:
    transforms = ["translate", "rotate", "scale"]
    counts = {t: len(re.findall(rf"\b{t}\s*\(", code)) for t in transforms}
    total = sum(counts.values())
    diversity = sum(1 for v in counts.values() if v > 0)
    return total, diversity


def _module_stats(code: str) -> tuple[int, int]:
    code_no_comments = _strip_scad_comments(code)
    defs = re.findall(r"\bmodule\s+([A-Za-z_]\w*)\s*\(", code_no_comments)
    user_defs = [name for name in defs if name.lower() not in SCAD_BUILTIN_CALLS]
    unique_defs = set(user_defs)
    if not unique_defs:
        return 0, 0
    calls = 0
    for name in unique_defs:
        refs = len(re.findall(rf"\b{name}\s*\(", code_no_comments))
        calls += max(0, refs - 1)
    return len(user_defs), calls


def _max_repeated_hardcoded_transform_lines(code: str) -> int:
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    transform_lines = [
        ln for ln in lines
        if re.match(r"^(translate|rotate|scale)\s*\([^\n]*\)\s*\{?$", ln)
    ]
    if not transform_lines:
        return 0
    return max(Counter(transform_lines).values())


def _extract_prompt_keywords(query: str) -> set[str]:
    words = {
        w.lower()
        for w in re.findall(r"[a-zA-Z_]+", query)
        if len(w) > 2 and w.lower() not in PROMPT_STOPWORDS and w.lower() not in SCAD_KEYWORDS
    }
    # keep a small CAD hint list for non-English prompts
    for kw in ["hex", "honeycomb", "vase", "phone", "stand", "thickness", "hole"]:
        if kw in query.lower():
            words.add(kw)
    return words


def _extract_scad_identifiers(code: str) -> set[str]:
    words = {
        w.lower()
        for w in re.findall(r"[a-zA-Z_]+", code)
        if len(w) > 2 and w.lower() not in PROMPT_STOPWORDS and w.lower() not in SCAD_KEYWORDS
    }
    return words


def score_scad_rlvr(
    query: str,
    generated: str,
    *,
    cfg: RewardConfig | None = None,
    seen_hashes: set[str] | None = None,
) -> tuple[float, dict]:
    """RLVR reward: deterministic checks + semantic overlap + structural quality."""
    cfg = cfg or RewardConfig()
    info: dict[str, float | int | bool | str] = {}
    raw_text = generated.strip()

    if not raw_text:
        return -1.5, {"empty": True}

    parsed = parse_think_and_scad(raw_text)
    text = parsed.scad
    info["format_clean"] = parsed.format_clean
    info["think_length"] = len(parsed.think.split())
    info["has_think"] = bool(parsed.think)

    reward = 0.0

    # format compliance shaping
    if parsed.format_clean:
        reward += 0.15
    elif parsed.think:
        reward -= 0.2

    # think degeneration guard
    think_len = len(parsed.think.split())
    if parsed.think:
        if think_len > 250:
            reward -= 0.15 * min((think_len - 250) / 250.0, 1.0)
            info["think_too_long"] = True
        elif think_len < 5:
            reward -= 0.1
            info["think_too_short"] = True

    if not text:
        info["scad_empty"] = True
        return float(reward - 1.5), info

    # A) deterministic syntax checks
    balanced = _balanced_brackets(text)
    bracket_score, bracket_imbalance = _bracket_balance_score(text)
    info["balanced_brackets"] = balanced
    info["bracket_imbalance"] = bracket_imbalance
    reward += bracket_score

    semicolon_ok = text.count(";") >= 1
    info["has_semicolon"] = semicolon_ok
    reward += 0.35 if semicolon_ok else -0.2

    # B) keyword coverage (weak supervision)
    found = sum(1 for k in SCAD_KEYWORDS if re.search(rf"\b{k}\b", text))
    info["keyword_found"] = found
    reward += min(found / 6.0, 1.0)

    # C) lexical overlap with prompt
    q_words = _extract_prompt_keywords(query)
    g_words = _extract_scad_identifiers(text)
    overlap = len(q_words & g_words)
    info["prompt_overlap"] = overlap
    overlap_ratio = (overlap / max(4, len(q_words))) if q_words else 0.0
    info["prompt_overlap_ratio"] = overlap_ratio
    reward += min(overlap_ratio, 0.6)

    # D) length constraints
    n = len(text)
    info["length"] = n
    if n < cfg.min_len:
        reward -= 0.8
    elif n > cfg.max_len:
        reward -= 0.5

    # E) structural/static-analysis shaping
    module_defs, module_calls = _module_stats(text)
    info["module_defs"] = module_defs
    info["module_calls"] = module_calls
    if module_defs > 0:
        reward += cfg.module_definition_bonus
    if module_calls > 0:
        reward += cfg.module_call_bonus

    has_for_loop = bool(re.search(r"\bfor\s*\(", text))
    info["has_for_loop"] = has_for_loop
    if has_for_loop:
        reward += cfg.for_loop_bonus

    transform_total, transform_diversity = _transform_stats(text)
    info["transform_count"] = transform_total
    info["transform_diversity"] = transform_diversity
    if transform_diversity >= 2:
        reward += cfg.transform_diversity_bonus

    repeated_transform_max = _max_repeated_hardcoded_transform_lines(text)
    info["max_repeated_transform_lines"] = repeated_transform_max
    if repeated_transform_max >= cfg.hardcode_repeat_threshold:
        reward += cfg.hardcode_repeat_penalty

    # F) dedup penalty against reward hacking
    if seen_hashes is not None:
        prefix_hash = _hash_prefix(text, cfg.dedup_window_size)
        duplicated = prefix_hash in seen_hashes
        info["dedup_hit"] = duplicated
        if duplicated:
            reward += cfg.dedup_repeat_penalty
        else:
            seen_hashes.add(prefix_hash)

    # G) strong verification via OpenSCAD (optional)
    if cfg.verify_with_openscad and cfg.openscad_bin:
        ok, log = _verify_with_openscad(text, cfg.openscad_bin)
        compile_signal = _parse_compile_signal(ok, log)
        info.update({f"openscad_{k}": v for k, v in compile_signal.items()})

        if compile_signal["syntax_error"]:
            reward += cfg.compile_syntax_error_penalty
        elif compile_signal["compile_ok"]:
            reward += cfg.compile_success_reward

        if compile_signal["has_warning"]:
            reward += cfg.compile_warning_penalty
        if compile_signal["empty_geometry"]:
            reward += cfg.compile_empty_geometry_penalty
        if compile_signal["has_error"] and not compile_signal["syntax_error"] and not compile_signal["compile_ok"]:
            reward += cfg.compile_runtime_error_penalty

        info["openscad_log_excerpt"] = log[-600:]

    return float(reward), info


def score_scad(query: str, generated: str) -> float:
    # 兼容旧接口
    score, _ = score_scad_rlvr(query, generated)
    return score
