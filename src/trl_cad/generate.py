from __future__ import annotations

import argparse

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from .prompts import format_rl_query
'''
这个代码的作用是使用训练好的 LoRA/PEFT 适配器模型，
根据用户输入的提示生成 OpenSCAD 代码。它加载指定的
模型和 tokenizer，格式化用户提示为 RL 查询格式，然
后使用模型生成响应文本，最后提取并打印出生成的 SCAD
 代码部分。
'''

def main() -> None:
    parser = argparse.ArgumentParser(description="用训练好的适配器生成 SCAD")
    parser.add_argument("--model", type=str, required=True, help="LoRA/PEFT checkpoint 路径")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    prompt = format_rl_query(args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(text.split("<|assistant|>")[-1].strip())


if __name__ == "__main__":
    main()
