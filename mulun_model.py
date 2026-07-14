"""
MulunForCausalLM — Main model entry point.

Architecture:
  Base: MiniMindForCausalLM (from model_base.py)
  Sidecar: GameNNDecisionHead (from decision_head.py)

The sidecar attaches to the backbone WITHOUT modifying it.
At selected positions (triggered by <think> tokens or mode flag),
hidden states are routed through the decision head, which:
  1. Encodes state
  2. Runs RNN decision step
  3. Produces strategy/action/value
  4. Runs one-step world model
  5. Biases LM logits with decision info

Usage:
    model = MulunForCausalLM()
    out = model(input_ids, mode='text')  # pure text generation
    out = model(input_ids, mode='decision')  # text + decision sidecar
"""
import os, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from model_base import (
    MiniMindConfig, MiniMindModel, MiniMindForCausalLM,
    precompute_freqs_cis, MiniMindBlock, RMSNorm,
    MOEFeedForward, FeedForward, Attention,
)

from decision_head import (
    GameNNDecisionHead, StateEncoder,
    WorldModelStep, GumbelRouter,
)


class MulunConfig(MiniMindConfig):
    """Extended config for mulun model with decision head."""
    model_type = "mulun"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_dim = kwargs.get("state_dim", 16)
        self.n_strategies = kwargs.get("n_strategies", 3)
        self.n_actions = kwargs.get("n_actions", 8)
        self.markov_rank = kwargs.get("markov_rank", 16)
        self.think_token_id = kwargs.get("think_token_id", 25)  # <think> token id
        self.think_end_ids = kwargs.get("think_end_ids", [26, 234, 234])  # </think>\n\n


class MulunForCausalLM(MiniMindForCausalLM):
    """
    Mulun model: MiniMind backbone + GameNNDecisionHead sidecar.

    The sidecar is entirely optional — if init_decision_head=False,
    this behaves exactly like a normal MiniMindForCausalLM.
    """
    config_class = MulunConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(
        self,
        config: MulunConfig = None,
        init_decision_head: bool = True,
    ):
        config = config or MulunConfig()
        self.config = config
        super().__init__(config)

        # Sidecar: GameNN Decision Head (DSpark-style)
        if init_decision_head:
            self.decision_head = GameNNDecisionHead(
                hidden_dim=config.hidden_size,
                state_dim=config.state_dim,
                n_strategies=config.n_strategies,
                n_actions=config.n_actions,
                markov_rank=config.markov_rank,
            )
        else:
            self.decision_head = None

        # Sidecar output cache (for structured output)
        self._last_decision = None

        # RoPE precompute (extended for YaRN)
        if not hasattr(self.model, 'freqs_cos') or self.model.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=config.head_dim,
                end=config.max_position_embeddings,
                rope_base=config.rope_theta,
                rope_scaling=config.rope_scaling,
            )
            self.model.register_buffer("freqs_cos", freqs_cos, persistent=False)
            self.model.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        labels: Optional[torch.Tensor] = None,
        mode: str = 'text',
        **kwargs,
    ):
        """
        Forward pass with optional decision sidecar.

        Args:
            input_ids: [B, T] token ids
            mode: 'text' — normal LM forward (fastest)
                  'decision' — run decision head at every position
                  'think' — run decision head only at <think> positions
            **kwargs: passed to backbone forward

        Returns:
            MoeCausalLMOutputWithPast with additional fields:
              - decision (if mode != 'text')
              - decision_loss (if labels provided)
        """
        # ── Backbone forward ──
        hidden_states, past_kvs, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )

        slice_idx = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
        h_slice = hidden_states[:, slice_idx, :]
        lm_logits = self.lm_head(h_slice)

        decision_out = None

        # ── Decision sidecar ──
        if self.decision_head is not None and mode != 'text':
            if mode == 'think':
                # Only run at <think> token positions
                think_mask = (input_ids == self.config.think_token_id)
                if think_mask.any():
                    last_think_pos = think_mask.nonzero()[-1, -1].item()
                    h_think = hidden_states[:, last_think_pos:last_think_pos+1, :]
                    lm_think = lm_logits[:, last_think_pos:last_think_pos+1, :]

                    rnn_state = self.decision_head.init_state(
                        input_ids.shape[0], input_ids.device
                    )
                    decision_out = self.decision_head(
                        h_think.squeeze(1), lm_think.squeeze(1),
                        rnn_state, mode='defender',
                        lm_head_weight=self.lm_head.weight,
                    )
            else:
                # Every position (mode='decision')
                rnn_state = self.decision_head.init_state(
                    input_ids.shape[0], input_ids.device
                )
                T = hidden_states.shape[1]
                all_decisions = []

                for t in range(T):
                    h_t = hidden_states[:, t:t+1, :].squeeze(1)
                    lm_t = lm_logits[:, t:t+1, :].squeeze(1)
                    prev_action = None if t == 0 else all_decisions[-1]['action_idx']

                    d = self.decision_head(
                        h_t, lm_t, rnn_state, prev_action, mode='defender',
                        lm_head_weight=self.lm_head.weight,
                    )
                    rnn_state = d['new_rnn_state']
                    all_decisions.append(d)

                decision_out = all_decisions[-1] if all_decisions else None
                # Stack for loss computation
                if all_decisions and labels is not None:
                    decision_out['_all'] = all_decisions

            self._last_decision = decision_out

        # ── Loss computation ──
        loss = None
        if labels is not None:
            # Text CE loss (standard)
            logits_for_loss = lm_logits[:, :-1, :] if labels.shape[-1] == lm_logits.shape[-2] else lm_logits
            labels_for_loss = labels[:, 1:] if labels.shape[-1] == logits_for_loss.shape[-2] + 1 else labels

            x, y = logits_for_loss.contiguous(), labels_for_loss.contiguous()
            text_loss = F.cross_entropy(
                x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100
            )

            # Decision loss with trajectory-aware per-step targets
            decision_loss_val = 0.0
            step_strategies = kwargs.get('step_strategies')
            step_actions = kwargs.get('step_actions')
            step_values = kwargs.get('step_values')

            if decision_out and '_all' in decision_out and step_strategies is not None:
                n_steps = step_strategies.shape[1]
                for k in range(n_steps):
                    d = decision_out['_all'][k] if k < len(decision_out['_all']) else decision_out['_all'][-1]
                    valid_mask = step_strategies[:, k] >= 0
                    if valid_mask.any():
                        dl = self.decision_head.compute_loss(
                            d,
                            target_strategy=step_strategies[:, k],
                            target_action=step_actions[:, k],
                            target_value=step_values[:, k],
                            valid_mask=valid_mask,
                        )
                        decision_loss_val += dl.get('decision_loss', 0.0)

            loss = text_loss + 0.1 * decision_loss_val

        out = MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=lm_logits,
            past_key_values=past_kvs,
            hidden_states=hidden_states,
        )
        out.decision = decision_out
        return out

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1024,
        temperature: float = 0.75,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: int = 2,
        use_cache: bool = True,
        mode: str = 'think',
        return_decision: bool = False,
        **kwargs,
    ):
        """
        Generate with optional decision sidecar influence.

        Args:
            mode: 'text' — standard generation (no decision)
                  'think' — run decision at <think> tokens only
                  'decision' — run decision every step
            return_decision: if True, return decision dict alongside tokens
        """
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        if self.decision_head is None:
            mode = 'text'

        # Initialize RNN state if using decision
        rnn_state = None
        prev_action = None
        if mode != 'text':
            rnn_state = self.decision_head.init_state(
                input_ids.shape[0], input_ids.device
            )

        decisions_log = []

        for step in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0

            # ── Backbone forward ──
            outputs = self.forward(
                input_ids[:, past_len:],
                past_key_values=past_key_values,
                use_cache=use_cache,
                mode='text',  # always text for fast generation
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :] / temperature

            # ── Decision sidecar at <think> positions ──
            if mode in ('think', 'decision') and self.decision_head is not None:
                hidden = outputs.hidden_states[:, -1, :]

                if mode == 'think':
                    # Run at <think> token positions
                    if (input_ids[:, -1:] == self.config.think_token_id).any():
                        d = self.decision_head(
                            hidden, logits, rnn_state, prev_action,
                            lm_head_weight=self.lm_head.weight,
                        )
                        rnn_state = d['new_rnn_state']
                        prev_action = d['action_idx']
                        logits = d['biased_logits']
                        decisions_log.append(d)
                elif mode == 'decision':
                    # Run every step
                    d = self.decision_head(
                        hidden, logits, rnn_state, prev_action,
                        lm_head_weight=self.lm_head.weight,
                    )
                    rnn_state = d['new_rnn_state']
                    prev_action = d['action_idx']
                    logits = d['biased_logits']
                    decisions_log.append(d)

            # ── Token sampling ──
            logits = logits.to(input_ids.device)
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), False
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

            next_token = torch.multinomial(
                torch.softmax(logits, dim=-1), num_samples=1
            )
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token,
                )

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if finished.all():
                break

        if return_decision:
            return input_ids, decisions_log
        return input_ids

    def get_decision(self, text: str, tokenizer) -> dict:
        """
        Analyze a security report and return structured decision output.
        This is the inference-time API matching the old SecurityAgent interface.
        """
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = tokens['input_ids'].to(next(self.parameters()).device)

        with torch.no_grad():
            out = self.forward(input_ids, mode='decision')

        if out.decision is None:
            return {'error': 'decision head not initialized'}

        d = out.decision
        return {
            'state': d['state'].tolist(),
            'uncertainty': d['uncertainty'].tolist(),
            'strategy': d['strategy_name'][0],
            'action': d['action_name'][0],
            'value': d['value'].item(),
            'containment_prob': d['containment_prob'].item(),
            'predicted_state': d['predicted_state'].tolist(),
        }


__all__ = [
    'MulunConfig', 'MulunForCausalLM',
]
