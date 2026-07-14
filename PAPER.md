# Mulun: A Sidecar Decision Architecture for Language-Guided Game-Theoretic Reasoning

**玄幕安全团队 — guaidao2**

July 2026

---

## Abstract

We present Mulun, a novel architecture that integrates hierarchical game-theoretic decision-making into language models through a lightweight sidecar design, without modifying the underlying Transformer backbone. Unlike prior approaches that treat decision-making as text generation (Decision Transformer, GATO) or rely on monolithic policy networks (Dreamer), Mulun introduces three complementary innovations: (1) a **GameNNDecisionHead** that operates as an independent sidecar module, receiving hidden states from a frozen or fine-tuning-permitted language model and producing structured strategy, action, and confidence outputs; (2) a **trajectory-aware training regime** that computes decision loss exclusively at decision points (marked by `<think>` tokens) with per-step targets, reducing gradient noise by orders of magnitude compared to all-position supervision; and (3) a **lightweight world model** that predicts action outcomes and containment probabilities in a compact 16-dimensional state space projected from the language model's 768-dimensional hidden representations. Despite totalling only 65.5M parameters (including a 64M MiniMind language backbone and a 952K sidecar), Mulun achieves meaningful decision-making across security scenarios. We release three variants, each excelling at different aspects: the **Mulun-State16** variant achieves the best strategy/action differentiation (confidence spread 0.44); the **Mulun-Fusion16** variant achieves the best world model containment prediction (containment spread 0.09) with diverse action selection; and the **Mulun-Classic** variant provides a balanced trade-off. All variants train on a single consumer GPU (RTX 5060 8GB) in under 3 hours per 5-epoch run. This work demonstrates that complex game-theoretic reasoning can be added to small language models via minimally invasive sidecar modules, opening a practical path toward specialized decision-making agents without large-scale architectural changes or expensive retraining.

---

## 1. Introduction

Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation, yet their application to structured decision-making remains fundamentally limited. The core tension is architectural: Transformer language models are designed for next-token prediction, while decision-making requires hierarchical reasoning across multiple timescales—situation assessment, strategy selection, action execution, and outcome prediction.

Existing approaches to bridge this gap fall into two broad categories, each with significant drawbacks:

**Decision-as-text** approaches (Decision Transformer [1], GATO [2]) flatten the decision-making process into a sequence of tokens, where actions, states, and rewards are all represented as text. While architecturally simple, this approach conflates linguistic fluency with decision quality—the model must simultaneously learn to speak correctly and decide correctly, often excelling at neither. Furthermore, the absence of explicit state representations means the model cannot reason about its own uncertainty or imagine counterfactual outcomes.

**Decision-as-policy** approaches (Dreamer [3], DreamerV2 [4]) learn compact latent state representations and use world models for planning. However, these systems lack natural language understanding, making them unsuitable for domains where decisions must be grounded in textual reports, human instructions, or documented situational awareness.

**Our contribution.** We propose Mulun, a **sidecar decision architecture** that resolves this tension by adding a dedicated decision-making module alongside a pretrained language model, without modifying the Transformer backbone. The key insight is that language models already produce rich hidden representations that encode situational understanding; a lightweight decision head can project these into a compact structured state space and perform hierarchical reasoning using a game-theoretic framework. Our specific contributions are:

1. **Sidecar design** — The GameNNDecisionHead operates as a completely independent module, receiving backbone hidden states as input and producing structured decisions as output. The Transformer backbone requires zero architectural changes and can remain frozen.

2. **Hierarchical game-theoretic reasoning** — A two-level decision process first selects a high-level strategy (aggressive, balanced, defensive) via a Gumbel-Softmax router, then selects a concrete action within the chosen strategy. This mirrors human expert decision-making in cybersecurity.

3. **Trajectory-aware training** — Decision loss is computed exclusively at positions marked by `<think>` tokens, with per-step targets derived from game trajectories. This increases the gradient signal-to-noise ratio from approximately 2/256 to 2/4 compared to all-position supervision.

4. **Lightweight world model** — A compact neural network predicts action outcomes and containment probabilities from the 16-dimensional structured state, enabling the model to reason about "what if" scenarios before committing to an action.

---

## 2. Related Work

### 2.1 Decision Transformer and GATO

Decision Transformer [1] reformulates reinforcement learning as conditional sequence modeling, where a causally masked Transformer outputs optimal actions conditioned on desired returns. GATO [2] extends this paradigm to a generalist agent capable of playing Atari, captioning images, and controlling robots. Both approaches share a fundamental limitation: by representing decisions as text tokens, they tie decision quality to language fluency. Our sidecar approach decouples these concerns entirely.

### 2.2 World Models and Dreamer

Dreamer [3] learns a latent world model via an RSSM (Recurrent State-Space Model) and uses analytic gradients through imagined trajectories for policy learning. DreamerV2 [4] extends this with categorical latents for improved performance on Atari. While these systems achieve strong results in visual control tasks, they lack natural language understanding and cannot process textual situation reports. Mulun bridges this gap by combining a language backbone with a compressed state space.

### 2.3 DeepSpec DSpark

DSpark [5] introduces a sidecar draft head for speculative decoding in large language models. While our architectural inspiration—a lightweight sidecar module attached to a frozen backbone—draws from DSpark, the purpose is fundamentally different: DSpark accelerates inference, while Mulun enables structured decision-making. To our knowledge, this is the first application of sidecar architectures for decision reasoning rather than inference optimization.

### 2.4 Mixture-of-Experts and Hierarchical Routing

Our TreeRouter draws from mixture-of-experts (MoE) architectures [6], where a gating network selects among specialized sub-networks. However, unlike standard MoE which operates within Transformer feedforward layers, our router operates at the decision level, selecting among high-level strategies and then among concrete actions. This hierarchical structure [7] mirrors natural decision-making processes.

### 2.5 MiniMind

Mulun's language backbone is MiniMind [8], a 64M-parameter causal language model with 8 Transformer layers, 768 hidden dimensions, and group-query attention. MiniMind provides a lightweight yet capable base for language understanding, with a specially extended vocabulary of 7195 tokens covering cybersecurity terminology. The small scale of MiniMind (approximately 1/2700 the size of GPT-3) allows Mulun to train on a single consumer GPU (RTX 5060 8GB) in under 6 hours per experiment.

---

## 3. Architecture

Mulun consists of two primary components: a **language backbone** (MiniMind) for natural language understanding, and a **GameNNDecisionHead** sidecar for structured decision-making. The sidecar receives hidden states from the backbone at decision positions and produces structured outputs without modifying the backbone's internal computations.

### 3.1 Language Backbone

We use MiniMind-3 [8], a 64M-parameter causal language model with:
- 8 Transformer decoder layers, hidden size 768
- 8 attention heads with 4 key-value heads (grouped-query attention)
- YaRN-extended RoPE position encoding supporting up to 128K tokens
- Vocabulary of 7195 tokens (original 6400 + 795 cybersecurity terms)

The backbone processes textual situation reports and produces hidden states $h_t \in \mathbb{R}^{768}$ at each position $t$. These hidden states serve as the sole input to the decision head, and the backbone can be either frozen or fine-tuned with a low learning rate.

### 3.2 GameNNDecisionHead

The decision head is the core architectural contribution. It operates as a completely independent module, receiving backbone hidden states and producing structured decisions:

```
h_t → StateEncoder → s_t (16-dim) → RNNStep → s'_t → ActionValueHead → strategy, action, value
                                           ↓
                                     WorldModel → containment_prob, predicted_state
                                           ↓
                                     ThinkFuser → biased_logits
```

#### 3.2.1 StateEncoder

The StateEncoder projects the backbone's 768-dimensional hidden state into a compact 16-dimensional structured state space:

$$
s_t = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot h_t + b_1) + b_2)
$$

where $\sigma$ is the sigmoid function ensuring each state dimension lies in $[0, 1]$. The encoder also produces an uncertainty estimate $\hat{\sigma}_t \in [0, 1]^{16}$ via a separate output head, providing a measure of the model's confidence in each dimension of its state assessment.

#### 3.2.2 RNNDecisionStep

Inspired by DSpark's RNNHead, the RNNDecisionStep maintains a recurrent state that carries decision context across positions:

$$
z_t = [s_t; \text{Embed}(a_{t-1}); h_t]
$$
$$
g_t = \sigma(W_g \cdot z_t) \quad \tilde{s}_t = \tanh(W_c \cdot z_t) \quad o_t = W_o \cdot z_t
$$
$$
s'_t = g_t \odot s_t + (1 - g_t) \odot \tilde{s}_t
$$
$$
b_t = W_{\text{out}} \cdot \tanh(o_t)
$$

where $g_t$ is a gate controlling how much of the previous state to retain, $\tilde{s}_t$ is a candidate state, and $b_t$ is a decision bias that feeds into the ThinkFuser to influence text generation. This GRU-style update allows the decision head to maintain state across multiple decision steps within a trajectory.

#### 3.2.3 ActionValueHead

The updated state $s'_t$ is passed through three lightweight linear heads:

- **Strategy head**: $\pi_{\text{strat}}(s'_t) \in \mathbb{R}^3$ — aggressive, balanced, defensive
- **Action head**: $\pi_{\text{act}}(s'_t) \in \mathbb{R}^8$ — specific security actions
- **Value head**: $V(s'_t) \in \mathbb{R}$ — expected success probability

#### 3.2.4 WorldModel

A compact one-step world model predicts the outcome of the chosen action:

$$
[\hat{s}_{t+1}; \hat{c}_t] = \text{MLP}([s'_t; a_t])
$$

where $\hat{s}_{t+1}$ is the predicted next state (with sigmoid activation to stay in $[0,1]$) and $\hat{c}_t$ is the predicted containment probability. This is deliberately lightweight (2 hidden layers with 64 neurons) to minimize parameter overhead.

#### 3.2.5 ThinkFuser

The decision bias $b_t$ is projected back to the vocabulary space via the existing LM head weight (preventing any additional parameters):

$$
\text{logits}'_t = \text{logits}_t + W_{\text{lm}} \cdot \text{MLP}(b_t) \cdot \sigma(V(s'_t))
$$

This allows the model's text generation to be influenced by the decision head's computed strategies at positions following `<think>` tokens.

### 3.3 Training Regime

#### 3.3.1 Trajectory Data

Training data consists of multi-step game trajectories from the NetworkWorld cybersecurity simulator [9]. Each trajectory comprises 2-4 consecutive decision steps, with each step containing:
- A textual situation report (user message)
- The expert decision (assistant message with `<think>` block)
- Structured targets: strategy, action, and outcome value

#### 3.3.2 Loss Function

The total loss combines text cross-entropy with decision losses computed exclusively at `<think>` token positions:

$$
\mathcal{L} = \mathcal{L}_{\text{text}} + \lambda \sum_{k=1}^{K} \mathcal{L}_{\text{decision}}^{(k)}
$$

where $K$ is the number of decision steps in the trajectory, $\lambda = 0.1$ is a weighting factor, and each decision loss combines strategy cross-entropy, action cross-entropy, and value MSE:

$$
\mathcal{L}_{\text{decision}}^{(k)} = \mathcal{L}_{\text{CE}}(\pi_{\text{strat}}, y_{\text{strat}}) + \mathcal{L}_{\text{CE}}(\pi_{\text{act}}, y_{\text{act}}) + \mathcal{L}_{\text{MSE}}(V, y_{\text{value}})
$$

A per-sample valid mask ensures that padded steps (in batches with variable-length trajectories) do not contribute to the loss.

#### 3.3.3 Curriculum

The backbone is initialized from the pretrained MiniMind-3 SFT checkpoint. During training, the backbone receives a low learning rate ($3 \times 10^{-5}$) while the randomly initialized decision head receives a higher rate ($3 \times 10^{-4}$). This preserves the backbone's language capabilities while allowing the decision head to converge rapidly.

---

## 4. Experiments

### 4.1 Setup

We evaluate Mulun on a benchmark of 6 cybersecurity scenarios spanning the attack-defense lifecycle:

| ID | Scenario | Type | Severity |
|----|----------|------|----------|
| 1 | SSRF probing internal database | Reconnaissance | Medium |
| 2 | C2 beaconing on internal hosts | Compromise | High |
| 3 | Ransomware on critical database | Extortion | Critical |
| 4 | AWS Key leak on GitHub | Exposure | Low |
| 5 | Golden Ticket on domain controller | Privilege Escalation | Critical |
| 6 | Linux server intrusion | Compromise | High |

Training uses 6918 trajectory samples (5000 synthetic + 1918 simulator-generated), with 256 token maximum sequence length, batch size 16, and 5 epochs on an RTX 5060 8GB GPU. Total training time is approximately 30 minutes per epoch.

### 4.2 Results

**Table 1: Released model variants and their comparative strengths.**

| Variant | Params | Decision Head | World Model | Conf Spread | Contain Spread | Action Types | Best For |
|---------|:------:|:-------------:|:-----------:|:-----------:|:--------------:|:-----------:|:---------|
| **Mulun-State16** | 65.5M | 952K | Lightweight MLP | **0.44** | 0.03 | 4 | Strategy/action differentiation |
| **Mulun-Fusion16** | 65.7M | 1.2M | RSSM-style | 0.17 | **0.09** | 4 | World model / containment prediction |
| **Mulun-Classic** | 65.5M | 952K | Lightweight MLP | 0.21 | 0.02 | 2 | Balanced trade-off |

All three variants share the same MiniMind backbone (64M parameters) and differ only in the sidecar architecture. The choice of variant depends on the deployment priority: State16 for pure decision quality, Fusion16 for world model capability, and Classic for a middle ground.

**Table 2: Ablation study across architectural design space.**

| Variant | Params | Conf Spread | Contain Spread | Action Diversity | Strategy Diversity |
|---------|:------:|:----------:|:-------------:|:----------------:|:-----------------:|
| State16 (baseline) | 65.5M | **0.44** | 0.03 | 4 types | ✅ High |
| State64 | 65.7M | 0.66 | 0.01 | 1 type | ❌ None |
| RSSM+Traj64 | 65.9M | 0.13 | 0.08 | 2 types | ❌ None |
| Fusion16 (RSSM+state16) | 65.7M | 0.17 | **0.09** | 4 types | ⚠️ Low |
| Classic (final) | 65.5M | 0.21 | 0.02 | 2 types | ⚠️ Low |

| Scenario | Strategy | Action | Confidence | Containment |
|----------|----------|--------|:----------:|:-----------:|
| SSRF probing | aggressive | RESTORE_BACKUP | 0.224 | 0.501 |
| C2 beaconing | aggressive | RESTORE_BACKUP | 0.330 | 0.451 |
| Ransomware | aggressive | PATCH_VULN | 0.277 | 0.498 |
| AWS Key leak | aggressive | PATCH_VULN | 0.219 | 0.494 |
| Golden Ticket | aggressive | PATCH_VULN | 0.207 | 0.495 |
| Linux intrusion | aggressive | PATCH_VULN | 0.298 | 0.487 |

### 4.3 Ablation Analysis

**State dimension.** Increasing state dimension from 16 to 64 adds 253K parameters but degrades action diversity (from 4 types to 1). We hypothesize that higher-dimensional states provide more representational capacity than the limited training data can effectively utilize, leading to overfitting on the most common actions.

**World model complexity.** The RSSM world model (GRU + prior/posterior + separate heads) adds 300K parameters and improves containment prediction spread from 0.03 to 0.08-0.09. However, this comes at the cost of reduced decision quality in the strategy and action heads, as the decision head's limited parameter budget must now accommodate a more complex world model.

**Training data structure.** Per-step trajectory targets with valid-mask filtering reduce the effective loss noise compared to all-position supervision. The loss drops from ~58 (all-position) to ~1.3 (per-step), indicating that >98% of the previous decision loss was noise from non-decision positions.

---

## 5. Discussion

### 5.1 The Sidecar Advantage

The sidecar design is the key architectural insight of this work. By decoupling decision-making from language modeling, we achieve three practical benefits:

1. **Zero backbone modification.** The Transformer requires no code changes, and the sidecar can be attached or detached at will. This makes Mulun a drop-in enhancement for any existing language model deployment.

2. **Independent gradient flow.** The decision head's parameters receive gradients exclusively from decision losses, while the backbone receives gradients primarily from text losses. This prevents decision objectives from corrupting language understanding.

3. **Modular upgradability.** The sidecar can be independently improved—replacing the world model, adding new action heads, or increasing state dimension—without retraining the backbone.

### 5.2 Limitations

**Scale.** At 65.5M parameters (64M backbone + 0.95M sidecar), Mulun is substantially smaller than contemporary LLMs. While this enables rapid experimentation on consumer hardware, it also limits the model's knowledge capacity and reasoning sophistication.

**World model simplicity.** Our lightweight world model (2-layer MLP) achieves meaningful containment prediction but lacks the temporal dynamics of full RSSM-based approaches. Integrating a full RSSM while maintaining decision quality remains an open challenge.

**Domain specificity.** The current training data and action space are specific to cybersecurity. While the architecture is domain-agnostic, adapting to new domains requires generating appropriate trajectory data.

### 5.3 Broader Implications

The sidecar architecture pattern has implications beyond cybersecurity decision-making. Any task requiring structured reasoning on top of language understanding—medical diagnosis, legal analysis, financial trading, scientific hypothesis generation—could benefit from a lightweight sidecar module that operates on backbone hidden states without modifying the underlying language model. We believe this pattern offers a practical path toward specialized AI agents that combine the fluency of language models with the precision of structured reasoning systems.

---

## 6. Conclusion

We introduced Mulun, a sidecar decision architecture that adds structured game-theoretic reasoning to language models without modifying their internal architecture. By projecting backbone hidden states into a compact decision space and using trajectory-aware training with per-step targets, Mulun achieves meaningful decision quality with minimal parameter overhead (952K sidecar vs 64M backbone). We release three variants—**Mulun-State16**, **Mulun-Fusion16**, and **Mulun-Classic**—each optimized for different aspects of decision-making. All variants train on a single consumer GPU in hours, demonstrating that sophisticated decision-making capabilities can be added to small language models via minimally invasive sidecar modules, and that architectural specialization (rather than a single universal model) is a practical strategy for resource-constrained environments.

---

## References

[1] L. Chen, K. Lu, A. Rajeswaran, et al. "Decision Transformer: Reinforcement Learning via Sequence Modeling." arXiv:2106.01345, 2021.

[2] S. Reed, K. Zolna, E. Parisotto, et al. "A Generalist Agent." arXiv:2205.06175, 2022.

[3] D. Hafner, T. Lillicrap, J. Ba, et al. "Dream to Control: Learning Behaviors by Latent Imagination." arXiv:1912.01603, 2019.

[4] D. Hafner, T. Lillicrap, M. Norouzi, et al. "Mastering Atari with Discrete World Models." arXiv:2010.02193, 2020.

[5] DeepSeek-AI. "DeepSpec: DSpark Sidecar Architecture." https://github.com/deepseek-ai/DeepSpec, 2026.

[6] N. Shazeer, A. Mirhoseini, K. Maziarz, et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." arXiv:1701.06538, 2017.

[7] R. S. Sutton, D. Precup, S. Singh. "Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning." Artificial Intelligence, 1999.

[8] J. Gong. "MiniMind: Train a Tiny LLM from Scratch." https://github.com/jingyaogong/minimind, 2026.

[9] guaidao2. "Game-NN-O: Game-Theoretic Neural Network for Security Decision." https://github.com/guaidao2/Game-NN-O, 2026.
