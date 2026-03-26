# TRL-CAD

该项目使用 TRL + PEFT 训练开源 LLM 生成 OpenSCAD 代码，分三阶段：

1. Stage1：SCAD 语料 LM 式持续预训练
2. Stage2：指令 SFT（支持 CoT；仅 assistant 输出计算 loss）
3. Stage3：GRPO + RLVR 强化学习

另外新增语义评分模型训练阶段：

4. Semantic-CLIP：用 Stage2 的 `prompt` + `scad_code` 训练 BERT 类双塔对比模型（CLIP 风格损失）

## 主要文件

- src/trl_cad/train_stage1_lm.py
- src/trl_cad/train_stage2_sft.py
- src/trl_cad/train_stage3_grpo.py
- src/trl_cad/train_semantic_clip.py
- src/trl_cad/reward.py
- src/trl_cad/semantic_reward.py
- src/trl_cad/generate.py
- src/trl_cad/smoke_check.py
- configs/stage1.yaml
- configs/stage2.yaml
- configs/stage3.yaml
- configs/semantic_clip.yaml

## 数据格式

### Stage1

JSONL 每行：

```json
{"text": "difference() { cube([20,20,20]); sphere(r=8); }"}
```

### Stage2

基础格式：

```json
{"prompt": "生成一个带孔的立方体", "scad_code": "difference(){ cube([20,20,20]); cylinder(h=30,r=4); }"}
```

含 CoT：

```json
{"prompt": "生成一个带孔的立方体", "cot": "先创建基体，再做差集减孔", "scad_code": "difference(){ cube([20,20,20]); cylinder(h=30,r=4); }"}
```

### Stage3

JSONL 每行：

```json
{"prompt": "生成一个参数化手机支架"}
```

## 训练链路（严格接力）

- stage2.yaml 默认 `model_name: outputs/stage1`
- semantic_clip.yaml 默认 `dataset_path: data/samples/stage2_instruction.jsonl`
- stage3.yaml 默认 `model_name: outputs/stage2`
- stage3.yaml 默认 `semantic_model_path: outputs/semantic_clip`
- 两个配置都启用 `require_peft_checkpoint: true`

当 `model_name` 指向本地目录且包含 `adapter_config.json` 时，脚本会按 PEFT checkpoint 继续训练。
如果开启 `require_peft_checkpoint: true` 但目录不是有效适配器，会直接报错并停止。

## 快速开始（PowerShell）

1) 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

2) 先做最小检查

```powershell
python main.py smoke
```

3) 跑完整流程

```powershell
python main.py pipeline
```

4) 分阶段运行

```powershell
python main.py stage1 --config configs/stage1.yaml
python main.py stage2 --config configs/stage2.yaml
python main.py semantic --config configs/semantic_clip.yaml
python main.py stage3 --config configs/stage3.yaml
```

## Stage3 奖励（已简化）

仅保留三项：

1. 编译是否成功，且生成集合体是否为空（OpenSCAD 检查）
2. 输出格式是否符合要求（是否包含 `<think>...</think>`）
3. 语义评分（由 Semantic-CLIP 模型对 prompt 与生成 SCAD 的匹配度打分）

5) 推理

```powershell
python main.py generate --model outputs/stage3 --prompt "生成一个可参数化齿轮外壳"
```

## 测试

```powershell
pytest -q
```

## 备注

- `bitsandbytes` 已从默认硬依赖中移除，避免 Windows 安装脆弱性。
- 若在 Linux + CUDA 需要 4bit/8bit，可自行额外安装。 
