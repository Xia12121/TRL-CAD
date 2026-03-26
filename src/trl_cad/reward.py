from __future__ import annotations

# reward.py 作用：
# 1) 解析模型输出（可含 <think>...</think>）并提取 SCAD 代码主体
# 2) 对 SCAD 进行多维打分（语法、结构、相关性、去重、可选编译验证）
# 3) 返回 (reward, info) 供 GRPO 训练使用

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

# 这些模式用于在 <think> 未闭合时，尽量识别“代码从哪里开始”。
# 目的：避免把合法 SCAD 误当成 think 文本吞掉。

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

# 解析 OpenSCAD 日志时使用：
# - ERROR_TOKEN_RE 检测 error/errors 词
# - NEGATIVE_ERROR_PHRASE_RE 用于排除 “no errors” 这类否定短语，减少误判

ERROR_TOKEN_RE = re.compile(r"\b(error|errors)\b")
NEGATIVE_ERROR_PHRASE_RE = re.compile(r"\b(?:no|without|zero|0)\s+errors?\b")


def _strip_scad_comments(code: str) -> str:
    """移除 SCAD 注释，避免注释内容干扰规则统计（如 module/call 计数）。"""
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    return code


@dataclass
class RewardConfig:
    """奖励配置。

    说明：
    - compile_*：OpenSCAD 外部验证相关奖励/惩罚（强约束）
    - module/for/transform/hardcode：静态结构奖励（弱到中等约束）
    - dedup_repeat_penalty：重复样本的惩罚（默认 0，表示关闭）
    """
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

    # 情况1：标准格式
    match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if match:
        return ParsedOutput(
            think=match.group(1).strip(),
            scad=match.group(2).strip(),
            format_clean=True,
        )

    # 情况2：只有 <think> 起始，未闭合；尝试识别代码起点
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

        # 兜底：若 remaining 已经“看起来像代码”，整段作为 SCAD，避免误吞有效代码。
        has_code_markers = (
            any(k in remaining.lower() for k in SCAD_KEYWORDS)
            or ";" in remaining
            or "{" in remaining
            or "}" in remaining
        )
        if has_code_markers:
            return ParsedOutput(think="", scad=remaining.strip(), format_clean=False)

        return ParsedOutput(think=remaining.strip(), scad="", format_clean=False)

    # 情况3：没有 think 标签，整体当 SCAD
    return ParsedOutput(think="", scad=text, format_clean=False)


def _balanced_brackets(code: str) -> bool:
    """严格括号匹配：用于布尔判断（True/False）。"""
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
    """柔性括号得分：

    返回 (score, imbalance)
    - imbalance=0 -> +1.0
    - 失配越多，分数越低，最低裁剪到 -1.2
    """
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
    """计算前缀哈希：用于重复样本检测（anti-reward-hacking）。"""
    return hashlib.md5(text[:n].encode("utf-8", errors="ignore")).hexdigest()


def _verify_with_openscad(code: str, openscad_bin: str) -> tuple[bool, str]:
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
    """将 OpenSCAD 日志解析成结构化信号，供分层奖励使用。"""
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
    """统计变换操作：

    total: translate/rotate/scale 总次数
    diversity: 三类中出现了几类（多样性）
    """
    transforms = ["translate", "rotate", "scale"]
    counts = {t: len(re.findall(rf"\b{t}\s*\(", code)) for t in transforms}
    total = sum(counts.values())
    diversity = sum(1 for v in counts.values() if v > 0)
    return total, diversity


def _module_stats(code: str) -> tuple[int, int]:
    """统计用户自定义 module 的定义和调用。

    - 会先去注释，避免注释文本干扰
    - 会排除内建调用名，避免误计
    """
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
    """检测重复硬编码 transform 行的最大重复次数。

    用于惩罚“复制粘贴式”堆代码，鼓励用循环表达结构。
    """
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    transform_lines = [
        ln for ln in lines
        if re.match(r"^(translate|rotate|scale)\s*\([^\n]*\)\s*\{?$", ln)
    ]
    if not transform_lines:
        return 0
    return max(Counter(transform_lines).values())


def _extract_prompt_keywords(query: str) -> set[str]:
    """从 prompt 提取关键词（去停用词、去 SCAD 关键字）。"""
    words = {
        w.lower()
        for w in re.findall(r"[a-zA-Z_]+", query)
        if len(w) > 2 and w.lower() not in PROMPT_STOPWORDS and w.lower() not in SCAD_KEYWORDS
    }
    # 对中英混合 prompt 给一些 CAD 语义提示词兜底
    for kw in ["hex", "honeycomb", "vase", "phone", "stand", "thickness", "hole"]:
        if kw in query.lower():
            words.add(kw)
    return words


def _extract_scad_identifiers(code: str) -> set[str]:
    """从 SCAD 中提取标识词，用于与 prompt 关键词做弱语义重合。"""
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
    """主奖励函数（GRPO 用）。

    核心原则：
    - 先分离 think/scad，再只对 scad 打分
    - 多维奖励相加：语法 + 语义弱相关 + 结构质量 + 可选编译验证
    - 返回 info 便于调参与诊断
    """
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

    # 0) 输出格式奖励：
    #    - 标准 <think>...</think> 小幅加分
    #    - 有 think 但格式坏掉小幅扣分
    if parsed.format_clean:
        reward += 0.15
    elif parsed.think:
        reward -= 0.2

    # 1) think 质量约束：防止过短敷衍/过长灌水
    think_len = len(parsed.think.split())
    if parsed.think:
        if think_len > 250:
            reward -= 0.15 * min((think_len - 250) / 250.0, 1.0)
            info["think_too_long"] = True
        elif think_len < 5:
            reward -= 0.1
            info["think_too_short"] = True

    if not text:
        # 没有可评分代码，直接重罚并返回
        info["scad_empty"] = True
        return float(reward - 1.5), info

    # 2) 语法可验证奖励（deterministic）
    #    2.1 括号平衡（柔性分）
    balanced = _balanced_brackets(text)
    bracket_score, bracket_imbalance = _bracket_balance_score(text)
    info["balanced_brackets"] = balanced
    info["bracket_imbalance"] = bracket_imbalance
    reward += bracket_score

    #    2.2 分号基础检查
    semicolon_ok = text.count(";") >= 1
    info["has_semicolon"] = semicolon_ok
    reward += 0.35 if semicolon_ok else -0.2

    # 3) SCAD 关键词覆盖（弱监督）：鼓励包含基本建模语义
    found = sum(1 for k in SCAD_KEYWORDS if re.search(rf"\b{k}\b", text))
    info["keyword_found"] = found
    reward += min(found / 6.0, 1.0)

    # 4) Prompt-代码弱语义对齐（词面重合，非强语义）
    #    使用 overlap_ratio，避免 prompt 很长时绝对值失真
    q_words = _extract_prompt_keywords(query)
    g_words = _extract_scad_identifiers(text)
    overlap = len(q_words & g_words)
    info["prompt_overlap"] = overlap
    overlap_ratio = (overlap / max(4, len(q_words))) if q_words else 0.0
    info["prompt_overlap_ratio"] = overlap_ratio
    reward += min(overlap_ratio, 0.6)

    # 5) 长度约束：过短通常信息不足，过长可能冗余/跑偏
    n = len(text)
    info["length"] = n
    if n < cfg.min_len:
        reward -= 0.8
    elif n > cfg.max_len:
        reward -= 0.5

    # 6) 结构奖励（静态分析）
    #    6.1 module 定义与调用：鼓励可复用结构
    module_defs, module_calls = _module_stats(text)
    info["module_defs"] = module_defs
    info["module_calls"] = module_calls
    if module_defs > 0:
        reward += cfg.module_definition_bonus
    if module_calls > 0:
        reward += cfg.module_call_bonus

    #    6.2 for 循环：鼓励程序化生成而非重复硬编码
    has_for_loop = bool(re.search(r"\bfor\s*\(", text))
    info["has_for_loop"] = has_for_loop
    if has_for_loop:
        reward += cfg.for_loop_bonus

    #    6.3 变换多样性：至少两种变换时加分
    transform_total, transform_diversity = _transform_stats(text)
    info["transform_count"] = transform_total
    info["transform_diversity"] = transform_diversity
    if transform_diversity >= 2:
        reward += cfg.transform_diversity_bonus

    #    6.4 重复硬编码惩罚：同类 transform 行重复过多扣分
    repeated_transform_max = _max_repeated_hardcoded_transform_lines(text)
    info["max_repeated_transform_lines"] = repeated_transform_max
    if repeated_transform_max >= cfg.hardcode_repeat_threshold:
        reward += cfg.hardcode_repeat_penalty

    # 7) 去重项：抑制模板化重复刷分（当前默认 penalty=0 即关闭）
    if seen_hashes is not None:
        prefix_hash = _hash_prefix(text, cfg.dedup_window_size)
        duplicated = prefix_hash in seen_hashes
        info["dedup_hit"] = duplicated
        if duplicated:
            reward += cfg.dedup_repeat_penalty
        else:
            seen_hashes.add(prefix_hash)

    # 8) OpenSCAD 强验证（可选）：
    #    - 编译成功奖励
    #    - 语法错误/警告/空几何/运行错误分层惩罚
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
    # 兼容旧接口：仅返回标量分数
    score, _ = score_scad_rlvr(query, generated)
    return score
