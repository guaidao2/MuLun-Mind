# Mulun (幕论) — Language-Guided Game-Theoretic Decision Architecture

Mulun is a **sidecar decision architecture** that adds structured game-theoretic reasoning to language models without modifying their internal Transformer backbone. Developed by **玄幕安全团队 (XuanMu Security Team)**.

## Architecture Overview

```
┌────────────────────────────────────────────────────┐
│              Language Backbone (MiniMind)           │
│  64M params · 8 Transformer layers · 768 hidden    │
│  Vocabulary: 7195 tokens (6400 base + 795 sec)     │
└────────────────────┬───────────────────────────────┘
                     │ hidden[768] at <think> positions
                     ▼
┌────────────────────────────────────────────────────┐
│            GameNNDecisionHead (Sidecar)              │
│                                                     │
│  hidden → StateEncoder → state[16] → RNNStep        │
│                                         ↓           │
│                              ActionValueHead         │
│                                → strategy (3)       │
│                                → action (8)         │
│                                → confidence          │
│                                         ↓           │
│                              WorldModel              │
│                                → containment_prob   │
│                                → predicted_state    │
│                                         ↓           │
│                              ThinkFuser              │
│                                → biased LM logits   │
└────────────────────────────────────────────────────┘
```

## Released Versions

| Version | Params | Decision Head | Conf Spread | Contain Spread | Strengths |
|---------|:------:|:-------------:|:-----------:|:--------------:|:----------|
| **Mulun-State16** | 65.5M | 952K MLP | **0.44** | 0.03 | Best strategy/action differentiation |
| **Mulun-Fusion16** | 65.7M | 1.2M RSSM | 0.17 | **0.09** | Best world model / containment |
| **Mulun-Classic** | 65.5M | 952K MLP | 0.21 | 0.02 | Balanced trade-off |

All variants share the same MiniMind backbone (64M) and differ only in the sidecar module. Trainable on a single RTX 5060 8GB GPU in under 3 hours.

## Project Structure

```
mulun/
├── __init__.py              # Package root
├── model_base.py            # MiniMind Transformer backbone
├── decision_head.py         # GameNNDecisionHead (sidecar)
├── mulun_model.py           # MulunForCausalLM (main model)
├── tokenizer/               # Extended tokenizer (7195 vocab)
├── trainers/
│   └── train_sft.py         # SFT training script
├── scripts/
│   ├── extend_tokenizer.py  # Vocabulary extension tool
│   ├── gen_decision_data.py # Synthetic decision chain generator
│   ├── gen_simulator_data.py# Simulator-based trajectory generator
│   └── gen_trajectory_data.py# Trajectory data formatter
├── backup_prev_train/       # Mulun-State16 weights
│   └── mulun_final.pth
├── out/                     # Trained weights output
│   ├── mulun_fusion16_final.pth  # Mulun-Fusion16 weights
│   └── mulun_classic_final.pth   # Mulun-Classic weights
├── dataset/                 # Training data
│   ├── trajectory_data.jsonl     # 6918 trajectory samples
│   └── ...
├── PAPER.md                 # English paper
├── PAPER_CN.md              # Chinese paper
└── requirements.txt         # Dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train from scratch (requires MiniMind backbone weight)
python trainers/train_sft.py \
  --data-path ../dataset/trajectory_data.jsonl \
  --from-weight ./backup_prev_train/mulun_final.pth \
  --save-dir ./out \
  --batch-size 16 \
  --epochs 5 \
  --lr 3e-4 \
  --state-dim 16 \
  --dtype float16
```

## Key Design Principles

1. **Zero backbone modification** — The sidecar attaches without changing Transformer code
2. **Independent gradient flow** — Decision losses don't corrupt language understanding
3. **Modular upgradability** — Replace world model or add action heads independently
4. **Decision-point-only supervision** — Loss computed only at `<think>` tokens

## Citation

```bibtex
@misc{mulun2026,
  title = {Mulun: A Sidecar Decision Architecture for Language-Guided Game-Theoretic Reasoning},
  author = {XuanMu Security Team},
  year = {2026},
  url = {https://github.com/guaidao2/MuLun-Mind}
}
```

## License

Apache 2.0
