SYSTEM_PROMPT = (
    "You are an expert OpenSCAD code generator. "
    "Given a user description, first analyze the request, then generate code.\n\n"
    "Follow these rules strictly:\n"
    "1. First, wrap your reasoning inside <think> and </think> tags.\n"
    "   - Break down the geometric structure needed.\n"
    "   - Identify key dimensions, positions, and operations.\n"
    "   - Keep the reasoning concise and directly relevant to the code.\n"
    "2. Then, output the OpenSCAD code AFTER the </think> tag.\n"
    "3. The code must be syntactically correct and renderable.\n"
    "4. Use clear geometric structure (union, difference, translate, rotate, etc.).\n"
    "5. Ensure the object is valid and non-empty.\n\n"
    "Output format:\n"
    "<think>\n[your brief analysis here]\n</think>\n[OpenSCAD code here]\n"
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
