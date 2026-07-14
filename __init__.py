"""
mulun — 幕论 AI: Security Decision Language Model.

Architecture:
  - MiniMind backbone (768/8 Transformer)
  - GameNNDecisionHead sidecar (DSpark-style RNN head)
  - World model for one-step imagination
  - Extended security vocabulary (7195 tokens)
"""
from mulun_model import MulunConfig, MulunForCausalLM
from model_base import MiniMindConfig, MiniMindForCausalLM
from decision_head import GameNNDecisionHead, StateEncoder, WorldModelStep

__version__ = "0.1.0"
