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

## 项目结构

```
mulun/
├── __init__.py              # 包入口
├── model_base.py            # MiniMind Transformer 骨干
├── decision_head.py         # GameNNDecisionHead（侧枝）
├── mulun_model.py           # MulunForCausalLM（主模型）
├── tokenizer/               # 扩展词表（7195）
├── trainers/
│   └── train_sft.py         # SFT 训练脚本
├── scripts/
│   ├── extend_tokenizer.py  # 词表扩展工具
│   ├── gen_decision_data.py # 决策链数据生成器
│   ├── gen_simulator_data.py# 模拟器轨迹数据生成
│   └── gen_trajectory_data.py# 轨迹数据格式化
├── backup_prev_train/       # Mulun-State16 权重
│   └── mulun_final.pth
├── out/                     # 训练输出
│   ├── mulun_fusion16_final.pth  # Mulun-Fusion16 权重
│   └── mulun_classic_final.pth   # Mulun-Classic 权重
├── PAPER.md                 # 英文论文
├── PAPER_CN.md              # 中文论文
└── requirements.txt         # 依赖
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 从头训练（需要 MiniMind 骨干权重）
python trainers/train_sft.py \
  --data-path ./data/trajectory.jsonl \
  --from-weight ./backup_prev_train/mulun_final.pth \
  --save-dir ./out \
  --batch-size 16 \
  --epochs 5 \
  --lr 3e-4 \
  --state-dim 16 \
  --dtype float16
```

## 核心设计原则

1. **零骨干修改** — 侧枝附加在 Transformer 旁，不改一行代码
2. **独立梯度流** — 决策损失不影响语言理解能力
3. **模块化可升级** — 可独立替换世界模型或添加动作头
4. **决策点监督** — 仅在 `<think>` token 位置计算决策损失

## 引用

```bibtex
@misc{mulun2026,
  title = {Mulun: A Sidecar Decision Architecture for Language-Guided Game-Theoretic Reasoning},
  author = {XuanMu Security Team},
  year = {2026},
  url = {https://github.com/guaidao2/MuLun-Mind}
}
```

## 许可

Apache 2.0
