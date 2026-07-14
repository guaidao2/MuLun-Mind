"""
GameNNDecisionHead — DSpark-style sidecar decision module for mulun.

Architecture (inspired by DeepSpec dspark RNNHead):
  Hidden[768] from backbone → StateEncoder → state[16]
  RNNStep: (state + prev_decision + hidden) → gate, candidate, bias
  → new_state → strategy head, action head, value head
  → WorldModel: one-step RSSM imagination
  → ThinkFuser: project decision bias back to LM logits

Key design decisions:
  - Sidecar: NO modifications to backbone Transformer
  - RNN state: carries decision history across tokens
  - World model: one-step look-ahead for "what if" reasoning
  - Fuser: biases LM head logits, not hidden states
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class StateEncoder(nn.Module):
    """
    Project backbone hidden[768] into structured state[16].
    """
    def __init__(self, hidden_dim: int = 768, state_dim: int = 16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim * 2),  # mean + logvar
        )

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hidden: [B, D]
        Returns: state [B, state_dim], uncertainty [B, state_dim]
        """
        out = self.proj(hidden)
        mean, logvar = out.chunk(2, dim=-1)
        state = torch.sigmoid(mean)
        uncertainty = torch.sigmoid(logvar)
        return state, uncertainty


class GumbelRouter(nn.Module):
    """
    Lightweight Gumbel-Softmax router for strategy/action selection.
    """
    def __init__(self, in_dim: int, n_choices: int, hidden: int = 64):
        super().__init__()
        self.n_choices = n_choices
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_choices),
        )

    def forward(self, x: torch.Tensor, tau: float = 1.0, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        if self.training and not hard:
            probs = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        else:
            probs = F.softmax(logits / max(tau, 0.01), dim=-1)
        return probs, probs.argmax(dim=-1)


class WorldModelStep(nn.Module):
    """
    Lightweight one-step world model.
    Predicts next state and containment probability.
    """
    def __init__(self, state_dim: int = 16, n_actions: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_actions, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim + 1),
        )
        self.state_dim = state_dim

    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor, **kwargs) -> dict:
        x = torch.cat([state, action_onehot], dim=-1)
        out = self.net(x)
        pred_state = torch.sigmoid(out[:, :-1])
        containment_prob = torch.sigmoid(out[:, -1:])
        return {'predicted_state': pred_state, 'containment_prob': containment_prob}


class RNNDecisionStep(nn.Module):
    """
    Core RNN step — adapted from DSpark's RNNHead._rnn_step.

    Takes (previous state, backbone hidden, action_embedding) and
    produces (new_state, decision_bias) using a GRU-style update.

    The 'state' here is a combination of:
      - Current security posture (from StateEncoder)
      - Accumulated decision context (from RNN recurrence)
    """
    def __init__(self, state_dim: int, hidden_dim: int, markov_rank: int = 16):
        super().__init__()
        self.state_dim = state_dim
        self.markov_rank = markov_rank

        # Joint projection: [state; prev_action_emb; hidden] -> gate + candidate + output
        self.joint_proj = nn.Linear(
            state_dim + markov_rank + hidden_dim,
            3 * state_dim
        )

        # Action embedding: maps one-hot action to markov_rank
        self.action_embed = nn.Linear(8, markov_rank, bias=False)

        # Output bias projection
        self.output_proj = nn.Linear(state_dim, hidden_dim, bias=False)

    def forward(
        self,
        state: torch.Tensor,
        hidden: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: [B, state_dim] — current security posture
            hidden: [B, hidden_dim] — backbone hidden at this position
            prev_action: [B] or None — previous action taken

        Returns:
            new_state: [B, state_dim]
            bias: [B, hidden_dim] — to be added to LM head logits
        """
        B = state.shape[0]
        device = state.device

        if prev_action is None:
            prev_action = torch.zeros(B, dtype=torch.long, device=device)

        action_emb = self.action_embed(
            F.one_hot(prev_action, num_classes=8).float()
        )

        # GRU-style update
        z = torch.cat([state, action_emb, hidden], dim=-1)
        proj = self.joint_proj(z)
        gate_raw, candidate_raw, output_raw = proj.chunk(3, dim=-1)

        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.output_proj(torch.tanh(output_raw))

        return new_state, bias

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        """Initialize zero state for a new decision sequence."""
        return torch.zeros(batch, self.state_dim, device=device)


class ActionValueHead(nn.Module):
    """
    Lightweight heads producing structured decision output from state.
    """
    def __init__(self, state_dim: int = 16, n_strategies: int = 3, n_actions: int = 8):
        super().__init__()
        self.n_strategies = n_strategies
        self.n_actions = n_actions

        self.strategy_net = nn.Linear(state_dim, n_strategies)
        self.action_net = nn.Linear(state_dim, n_actions)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.ReLU(),
            nn.Linear(state_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor) -> dict:
        return {
            'strategy_logits': self.strategy_net(state),
            'action_logits': self.action_net(state),
            'value': self.value_net(state),
        }


class ThinkFuser(nn.Module):
    """
    Projects decision bias back to influence LM head logits.
    Reuses existing LM head weight for vocab projection (zero extra params).
    """
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.bias_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

    def forward(
        self,
        lm_logits: torch.Tensor,
        decision_bias: torch.Tensor,
        state: torch.Tensor,
        decision: dict,
        lm_head_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        bias = self.bias_proj(decision_bias)              # [B, D]
        if lm_head_weight is not None:
            bias = F.linear(bias, lm_head_weight)         # [B, V]
        confidence = torch.sigmoid(decision['value'])
        biased_logits = lm_logits + bias * confidence
        return biased_logits


class GameNNDecisionHead(nn.Module):
    """
    Complete sidecar decision head — the main module.
    
    Flow:
      1. StateEncoder: hidden[768] -> state[16]
      2. RNNDecisionStep: (state, hidden, prev_action) -> new_state, bias
      3. ActionValueHead: new_state -> strategy, action, value
      4. RSSMWorldModel: (new_state, action) -> predicted_state, containment_prob, kl_loss
      5. ThinkFuser: bias -> LM logits adjustment
    """
    def __init__(
        self,
        hidden_dim: int = 768,
        state_dim: int = 16,
        n_strategies: int = 3,
        n_actions: int = 8,
        markov_rank: int = 16,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.state_encoder = StateEncoder(hidden_dim, state_dim)
        self.rnn_step = RNNDecisionStep(state_dim, hidden_dim, markov_rank)
        self.action_value = ActionValueHead(state_dim, n_strategies, n_actions)
        self.world_model = WorldModelStep(state_dim, n_actions)
        self.fuser = ThinkFuser(hidden_dim)  # no extra params, uses lm_head.weight at runtime

        # Strategy names for interpretability
        self._vocab_size = 0  # placeholder, set by set_vocab_size
        self.strategy_names = ['aggressive', 'balanced', 'defensive']
        self.action_names = [
            'BLOCK_IP', 'PATCH_VULN', 'ISOLATE_HOST', 'RESTORE_BACKUP',
            'DEEP_SCAN', 'HUNT_THREATS', 'DEPLOY_HONEYPOT', 'ESCALATE_INCIDENT',
        ]

    def set_vocab_size(self, vocab_size: int):
        """No-op: ThinkFuser reuses lm_head.weight, no vocab_size needed."""
        pass

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        """Initialize empty RNN decision state for a batch."""
        return torch.zeros(batch, self.state_dim, device=device)

    def forward(
        self,
        hidden: torch.Tensor,
        lm_logits: torch.Tensor,
        rnn_state: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        mode: str = 'defender',
        lm_head_weight: torch.Tensor = None,
    ) -> dict:
        """
        Single decision step.

        Args:
            hidden: [B, D] — backbone hidden state at decision position
            lm_logits: [B, V] — LM head logits (to be biased)
            rnn_state: [B, state_dim] — previous RNN decision state
            prev_action: [B] or None — previous action taken

        Returns:
            dict with:
              - new_rnn_state
              - state, strategy, action, value
              - containment_prob, predicted_state
              - biased_logits (for LM generation)
              - action_names, strategy_name (for interpretability)
        """
        # 1. Encode hidden -> structured state
        state, uncertainty = self.state_encoder(hidden)

        # 2. RNN step: (state, hidden, prev_action) -> new_state, bias
        new_state, decision_bias = self.rnn_step(state, hidden, prev_action)

        # 3. Decision heads
        decision = self.action_value(new_state)
        strategy_probs = F.softmax(decision['strategy_logits'], dim=-1)
        action_probs = F.softmax(decision['action_logits'], dim=-1)

        strategy_idx = strategy_probs.argmax(dim=-1)
        action_idx = action_probs.argmax(dim=-1)

        # 4. World model: imagine one step ahead
        action_onehot = F.one_hot(action_idx, num_classes=self.n_actions).float()
        wm_out = self.world_model(new_state, action_onehot)

        # 5. Fuser: bias LM logits
        if mode != 'eval_only':
            biased_logits = self.fuser(lm_logits, decision_bias, new_state, decision,
                                       lm_head_weight=lm_head_weight)
        else:
            biased_logits = lm_logits

        return {
            'new_rnn_state': new_state,
            'state': state,
            'uncertainty': uncertainty,
            'strategy_probs': strategy_probs,
            'strategy_idx': strategy_idx,
            'strategy_name': [self.strategy_names[i] for i in strategy_idx.tolist()],
            'action_probs': action_probs,
            'action_idx': action_idx,
            'action_name': [self.action_names[i] for i in action_idx.tolist()],
            'value': decision['value'],
            'containment_prob': wm_out['containment_prob'],
            'predicted_state': wm_out['predicted_state'],
            'biased_logits': biased_logits,
        }

    def compute_loss(
        self,
        decision_out: dict,
        target_strategy: Optional[torch.Tensor] = None,
        target_action: Optional[torch.Tensor] = None,
        target_outcome: Optional[torch.Tensor] = None,
        target_value: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute all decision-related losses.

        Args:
            decision_out: output from forward()
            valid_mask: [B] bool tensor — which samples have valid targets
        """
        losses = {}
        total = 0.0

        # Strategy cross-entropy
        if target_strategy is not None:
            if valid_mask is not None:
                target_strategy = target_strategy[valid_mask]
                logits = decision_out['strategy_probs'][valid_mask]
            else:
                logits = decision_out['strategy_probs']
            if target_strategy.numel() > 0:
                strat_loss = F.cross_entropy(logits, target_strategy)
                losses['strategy_loss'] = strat_loss
                total += strat_loss

        # Action cross-entropy
        if target_action is not None:
            if valid_mask is not None:
                target_action = target_action[valid_mask]
                logits = decision_out['action_probs'][valid_mask]
            else:
                logits = decision_out['action_probs']
            if target_action.numel() > 0:
                act_loss = F.cross_entropy(logits, target_action)
                losses['action_loss'] = act_loss
                total += act_loss

        # Value prediction
        if target_value is not None:
            if valid_mask is not None:
                target_value = target_value[valid_mask]
                val_pred = torch.sigmoid(decision_out['value'][valid_mask].squeeze(-1))
            else:
                val_pred = torch.sigmoid(decision_out['value'].squeeze(-1))
            if target_value.numel() > 0:
                val_loss = F.mse_loss(val_pred, target_value)
                losses['value_loss'] = val_loss
                total += val_loss

        losses['decision_loss'] = total
        return losses


__all__ = [
    'StateEncoder', 'GumbelRouter', 'WorldModelStep',
    'RNNDecisionStep', 'ActionValueHead', 'ThinkFuser',
    'GameNNDecisionHead',
]
