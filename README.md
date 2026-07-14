# Mulun（幕论）— 面向语言引导博弈推理的侧枝决策架构

Mulun 是一种**侧枝决策架构（Sidecar Decision Architecture）**，在不修改底层 Transformer 骨干的前提下，为语言模型添加结构化博弈推理能力。由 **玄幕安全团队** 开发。

> 📄 **论文：**
> - [PAPER_CN.md](./PAPER_CN.md) — 中文版
> - [PAPER.md](./PAPER.md) — English

---

## 架构总览

```
┌────────────────────────────────────────────────────┐
│             语言骨干（MiniMind）                     │
│  64M 参数 · 8 层 Transformer · 768 隐藏维           │
│  词表：7195 token（6400 基础 + 795 安全术语）        │
└────────────────────┬───────────────────────────────┘
                     │ hidden[768] 在 <think> 位置
                     ▼
┌────────────────────────────────────────────────────┐
│           GameNNDecisionHead（侧枝决策头）             │
│                                                      │
│  hidden → 状态编码器 → state[16] → RNN决策步          │
│                                          ↓           │
│                               动作价值头               │
│                                 → 策略（3种）         │
│                                 → 动作（8种）         │
│                                 → 置信度              │
│                                          ↓           │
│                               世界模型                 │
│                                 → 遏制概率            │
│                                 → 预测状态            │
│                                          ↓           │
│                               思考融合器               │
│                                 → 偏置后的 LM logits  │
└────────────────────────────────────────────────────┘
```

## 发布版本

| 版本 | 参数量 | 决策头 | 置信度价差 | 遏制价差 | 核心优势 |
|:-----|:------:|:------:|:---------:|:--------:|:---------|
| **Mulun-State16** | 65.5M | 952K MLP | **0.44** | 0.03 | 策略/动作区分最佳 |
| **Mulun-Fusion16** | 65.7M | 1.2M RSSM | 0.17 | **0.09** | 世界模型/遏制预测最佳 |
| **Mulun-Classic** | 65.5M | 952K MLP | 0.21 | 0.02 | 均衡折衷 |

三个版本共享相同的 MiniMind 骨干（64M），仅在侧枝模块上不同。全部可在单张 RTX 5060 8GB GPU 上于 3 小时内完成训练。

## 权重文件

| 文件 | 对应版本 | 说明 |
|:----|:---------|:------|
| `mulun_state16.pth` | State16 | 策略/动作区分最强的版本 |
| `mulun_fusion16.pth` | Fusion16 | 世界模型/遏制预测最强的版本 |
| `mulun_classic.pth` | Classic | 均衡折衷版本 |

## 项目结构

```
mulun/
├── model_base.py            # MiniMind Transformer 骨干
├── decision_head.py         # GameNNDecisionHead（侧枝）
├── mulun_model.py           # MulunForCausalLM（主模型）
├── tokenizer/               # 扩展词表（7195）
├── trainers/
│   └── train_sft.py         # SFT 训练脚本
├── scripts/                 # 数据生成工具
├── PAPER.md                 # 英文论文
├── PAPER_CN.md              # 中文论文
└── README.md                # 本文件
```

## 快速开始

```bash
pip install -r requirements.txt

python trainers/train_sft.py \
  --data-path ./data/trajectory.jsonl \
  --from-weight ./release/mulun_state16.pth \
  --save-dir ./out \
  --batch-size 16 \
  --epochs 5 \
  --state-dim 16 \
  --dtype float16
```

## 引用

```bibtex
@misc{mulun2026,
  title = {Mulun: A Sidecar Decision Architecture for Language-Guided Game-Theoretic Reasoning},
  author = {XuanMu Security Team},
  year = {2026},
  url = {https://github.com/guaidao2/MuLun-Mind}
}
```
