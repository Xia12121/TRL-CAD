SYSTEM_PROMPT = (
    "你是专业 OpenSCAD 代码助手。你只输出可执行的 OpenSCAD 代码，"
    "不要输出解释，不要输出 Markdown。"
)

INSTRUCTION_TEMPLATE = """<|system|>
{system}
<|user|>
{user_prompt}
<|assistant|>
{scad_code}"""

INSTRUCTION_TEMPLATE_WITH_COT = """<|system|>
{system}
<|user|>
{user_prompt}
<|assistant|>
<think>
{cot}
</think>
{scad_code}"""

RL_QUERY_TEMPLATE = """<|system|>
{system}
<|user|>
{user_prompt}
<|assistant|>
"""


def format_sft_example(user_prompt: str, scad_code: str) -> str:
    return INSTRUCTION_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        user_prompt=user_prompt.strip(),
        scad_code=scad_code.strip(),
    )


def format_sft_example_with_cot(user_prompt: str, cot: str, scad_code: str) -> str:
    return INSTRUCTION_TEMPLATE_WITH_COT.format(
        system=SYSTEM_PROMPT,
        user_prompt=user_prompt.strip(),
        cot=cot.strip(),
        scad_code=scad_code.strip(),
    )


def format_rl_query(user_prompt: str) -> str:
    return RL_QUERY_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        user_prompt=user_prompt.strip(),
    )
